"""A sparse observation reader view over immutable native-rate navigation data.

Observation sampling and action execution have separate clocks. A view row
keeps the native future-eight target, then executes the native controls up to
the following sampled observation. The terminal observation appears once.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np


NATIVE_DT_S = .2
VIEW_SCHEMA = 'lunar_training_observation_stride/14.0'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def jsonl_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def safe_base_path(base, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Dataset references must be relative paths')
    path = (base / relative).resolve(strict=True)
    if not path.is_relative_to(base):
        raise ValueError('Dataset reference escapes its dense base')
    return path


def selected_native_indices(action_count, stride=5):
    if action_count < 1 or stride < 1:
        raise ValueError('Positive native action count and observation stride are required')
    selected = list(range(0, action_count, stride))
    if selected[-1] != action_count - 1:
        selected.append(action_count - 1)
    return selected


def _episode_data(base, record, stride):
    trajectory = safe_base_path(base, record['trajectory'])
    metadata_path = trajectory / 'metadata.json'
    frames_path = trajectory / 'frame_records.jsonl'
    controls_path = trajectory / 'supervision.npz'
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('terminal_success') is not True or metadata.get('capture_stride') != 1:
        raise ValueError('Training views require complete, densely captured episodes')
    with np.load(controls_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in
                  ('vw_si', 'stop', 'dt_s', 'future_vw_si', 'horizon_valid')}
    n = len(arrays['vw_si'])
    if (arrays['vw_si'].shape != (n, 2) or arrays['future_vw_si'].shape != (n, 8, 2)
            or arrays['horizon_valid'].shape != (n, 8) or arrays['stop'].shape != (n,)
            or arrays['dt_s'].shape != (n,) or not np.allclose(arrays['dt_s'], NATIVE_DT_S, atol=1e-8)):
        raise ValueError('Dense episode does not satisfy the native 0.2s / future-eight contract')
    if n < 1 or np.count_nonzero(arrays['stop']) != 1 or arrays['stop'][-1] != 1:
        raise ValueError('Dense episode must have one final native STOP control')
    if not np.isfinite(arrays['vw_si']).all() or not np.isfinite(arrays['future_vw_si']).all():
        raise ValueError('Native controls contain nonfinite values')
    for offset in range(8):
        count = max(0, n - offset)
        if (not np.array_equal(arrays['horizon_valid'][:, offset], np.arange(n) + offset < n)
                or not np.array_equal(arrays['future_vw_si'][:count, offset], arrays['vw_si'][offset:])):
            raise ValueError('Dense future-eight labels are not consecutive native controls')
    if not np.array_equal(arrays['vw_si'][-1], np.zeros(2)):
        raise ValueError('Native terminal STOP must have zero linear and angular controls')
    frames = list(jsonl_rows(frames_path))
    if len(frames) != n or [frame['step_index'] for frame in frames] != list(range(n)):
        raise ValueError('Dense frame identities are incomplete or unordered')
    prefix = Path(record['trajectory'])
    observations = []
    for frame in frames:
        original = frame['inputs']['observation']
        observations.append(dict(observation_id=original['observation_id'],
            rgb=(prefix / original['rgb']).as_posix(),
            depth=(prefix / original['depth']).as_posix(), timestamp_s=original['timestamp_s']))
    selected = selected_native_indices(n, stride)
    following = {index: selected[position + 1] if position + 1 < len(selected) else n
                 for position, index in enumerate(selected)}
    source_paths = {path.relative_to(base).as_posix() for path in (metadata_path, frames_path, controls_path)}
    source_paths.update(record.get('sha256', {}))
    source_files = {relative: sha256(safe_base_path(base, relative)) for relative in sorted(source_paths)}
    if any(source_files[name] != digest for name, digest in record.get('sha256', {}).items()):
        raise ValueError('Dense episode no longer matches its standard manifest hashes')
    return dict(record=record, count=n, arrays=arrays, observations=observations,
                selected=selected, following=following, source_files=source_files)


class _Episodes(Mapping):
    """Keep at most two native episodes in memory, independent of dataset size."""
    def __init__(self, base, records, stride):
        self.base, self.records, self.stride = base, records, stride
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, identity):
        if identity not in self.cache:
            self.cache[identity] = _episode_data(self.base, self.records[identity], self.stride)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(identity)
        return self.cache[identity]


def _base_data(base, stride):
    manifest_path = base / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('complete') is not True or not manifest.get('episodes'):
        raise ValueError('Dense base manifest must contain completed physical episodes')
    episodes = {}
    for record in manifest['episodes']:
        identity = record['episode_id']
        if identity in episodes:
            raise ValueError('Duplicate physical episode in dense base manifest')
        episodes[identity] = record
    return manifest, _Episodes(base, episodes, stride)


def _view_row(dense_row, episode, stride, history_frames):
    row = copy.deepcopy(dense_row)
    index = int(row['step_index'])
    position = episode['selected'].index(index)
    past = episode['selected'][max(0, position - history_frames):position] if history_frames else []
    following = episode['following'][index]
    arrays = episode['arrays']
    row.update(schema_version=VIEW_SCHEMA, path_base='dense_base',
               native_step_index=index, observation_stride=stride,
               base_reader_sample_id=dense_row['sample_id'])
    row['inputs']['history'] = [copy.deepcopy(episode['observations'][previous]) for previous in past]
    row['sampling'] = dict(current_native_index=index,
        next_observation_native_index=following if following < episode['count'] else None,
        actual_next_observation_interval_s=(following - index) * NATIVE_DT_S if following < episode['count'] else None,
        terminal_observation=index == episode['count'] - 1)
    row['supervision'].update(native_action_dt_s=NATIVE_DT_S,
        future_horizon_native_steps=8,
        future_vw_si=arrays['future_vw_si'][index].tolist(),
        horizon_valid=arrays['horizon_valid'][index].tolist(),
        native_stop=int(arrays['stop'][index]),
        execution_chunk=dict(start_native_index=index, end_native_index_exclusive=following,
            control_dt_s=NATIVE_DT_S, native_control_indices=list(range(index, following)),
            vw_si=arrays['vw_si'][index:following].tolist(),
            stop=arrays['stop'][index:following].tolist()))
    return row


def _track_dense_row(row, episode, aliases):
    alias = row['episode_id']
    identity = (row['physical_trajectory_id'], row['inputs']['instruction'], row['split'])
    if alias not in aliases:
        aliases[alias] = dict(identity=identity, count=0)
    state = aliases[alias]
    index = row['step_index']
    if state['identity'] != identity or type(index) is not int or index != state['count']:
        raise ValueError('Dense language episode identity changed, or native frames are incomplete/unordered')
    if not 0 <= index < episode['count']:
        raise ValueError('Dense native observation index is outside its trajectory')
    if (row['inputs']['observation'] != episode['observations'][index]
            or row['split'] != episode['record']['split']
            or row['layout_seed'] != episode['record']['layout_seed']
            or row['supervision']['step_index'] != index):
        raise ValueError('Dense reader row does not match its physical frame/split/label identity')
    state['count'] += 1


def export_training_view(base, output, *, stride=5, history_frames=8):
    """Export a new, atomic reader view; never write inside the dense base.

    ``base`` and ``output`` can be sibling directories in one transportable
    package. The view needs that hash-bound base at its recorded relative path.
    Whole-episode arrays use a two-episode LRU, and the reader index is streamed.
    """
    base = Path(base).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if type(stride) is not int or stride < 1 or type(history_frames) is not int or not 0 <= history_frames <= 32:
        raise ValueError('Stride must be positive and sampled history must be within 0..32')
    if output == base or output.is_relative_to(base):
        raise ValueError('Write the observation view outside the immutable dense base')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing view: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.partial-', dir=output.parent))
    try:
        manifest, report = _export_staged_view(base, staging, stride, history_frames)
        if output.exists():
            raise FileExistsError('View destination appeared while exporting')
        staging.rename(output)
        return manifest, report
    except Exception as exc:
        write_json(staging / 'status.json', dict(phase='failed', passed=False, error=str(exc)))
        raise ValueError('Export was not published; diagnostics remain at ' + str(staging)) from exc


def _export_staged_view(base, output, stride, history_frames):
    manifest, episodes = _base_data(base, stride)
    base_manifest_hash = sha256(base / 'dataset_manifest.json')
    base_reader_hash = sha256(base / 'reader_index.jsonl')
    alias_identity = {}
    seen = set()
    selected_files = {}
    statistics = dict(dense_reader_rows=0, view_reader_rows=0,
                      retained_cot_reader_rows=0, dropped_off_grid_cot_reader_rows=0)
    index_path = output / 'reader_index.jsonl'
    with index_path.open('w') as stream:
        for dense_row in jsonl_rows(base / 'reader_index.jsonl'):
            statistics['dense_reader_rows'] += 1
            episode = episodes[dense_row['physical_trajectory_id']]
            _track_dense_row(dense_row, episode, alias_identity)
            index = int(dense_row['step_index'])
            if index not in episode['following']:
                statistics['dropped_off_grid_cot_reader_rows'] += int(dense_row['supervision'].get('cot_valid', False))
                continue
            if dense_row['sample_id'] in seen:
                raise ValueError('Duplicate selected reader sample')
            seen.add(dense_row['sample_id'])
            row = _view_row(dense_row, episode, stride, history_frames)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            statistics['view_reader_rows'] += 1
            statistics['retained_cot_reader_rows'] += int(row['supervision'].get('cot_valid', False))
            for key in ('rgb', 'depth'):
                relative = row['inputs']['observation'][key]
                if relative not in selected_files:
                    selected_files[relative] = sha256(safe_base_path(base, relative))
    for state in alias_identity.values():
        if state['count'] != episodes[state['identity'][0]]['count']:
            raise ValueError('Dense language episode is missing native observations')
    if {state['identity'][0] for state in alias_identity.values()} != set(episodes):
        raise ValueError('A physical episode is absent from the dense reader index')
    write_json(output / 'base_observation_hashes.json', selected_files)
    source_files, physical_episodes = {}, []
    for identity, episode in episodes.items():
        source_files.update(episode['source_files'])
        physical_episodes.append(dict(episode_id=identity, native_action_count=episode['count'],
            selected_native_indices=episode['selected'], split=episode['record']['split']))
    view_manifest = dict(schema_version=VIEW_SCHEMA, complete=True, training_ready=False,
        base_dataset=dict(relative_path=os.path.relpath(base, output),
            manifest_sha256=base_manifest_hash, reader_index_sha256=base_reader_hash,
            source_files_sha256=source_files, base_files_required=True),
        observation_stride=stride, nominal_observation_interval_s=stride * NATIVE_DT_S,
        native_action_dt_s=NATIVE_DT_S, native_future_horizon=8,
        sampled_history_frames=history_frames,
        terminal_rule='native final observation appended once if it is outside the regular stride grid',
        action_rule='preserve native future-eight targets; execute every actual native control until the next sampled observation, including the final STOP',
        cot_rule='retain only exact selected native-frame annotations; off-grid text is excluded',
        native_dense_base_unchanged=True,
        native_physical_frame_count=sum(episode['native_action_count'] for episode in physical_episodes),
        sampled_physical_frame_count=sum(len(episode['selected_native_indices']) for episode in physical_episodes),
        physical_episodes=physical_episodes,
        language_episode_count=len(alias_identity), statistics=statistics,
        reader_index_sha256=sha256(index_path),
        base_observation_hashes_sha256=sha256(output / 'base_observation_hashes.json'))
    write_json(output / 'view_manifest.json', view_manifest)
    report = validate_training_view(output)
    write_json(output / 'validation.json', report)
    write_json(output / 'status.json', dict(phase='complete' if report['passed'] else 'failed',
        passed=report['passed'], observation_stride=stride, **statistics))
    if not report['passed']:
        raise ValueError('Observation view failed alignment validation; inspect validation.json')
    return view_manifest, report


def _view_base(view):
    view = Path(view).expanduser().resolve(strict=True)
    manifest = json.loads((view / 'view_manifest.json').read_text())
    relative = manifest['base_dataset']['relative_path']
    if Path(relative).is_absolute():
        raise ValueError('View dense-base reference must be relative')
    return view, manifest, (view / relative).resolve(strict=True)


def load_policy_inputs(view, row):
    """Read only permitted RGB-D modalities, excluding NPZ privileged arrays."""
    _, _, base = _view_base(view)
    inputs = row['inputs']
    if set(inputs) != {'instruction', 'observation', 'history'}:
        raise ValueError('Policy inputs contain unsupported fields')
    from PIL import Image

    def observation(record):
        with Image.open(safe_base_path(base, record['rgb'])) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        with np.load(safe_base_path(base, record['depth']), allow_pickle=False) as archive:
            depth = archive['depth_m'].copy()
        if rgb.shape != (224, 224, 3) or depth.shape != (256, 256) or depth.dtype != np.float32:
            raise ValueError('Unexpected RGB-D input contract')
        return dict(rgb=rgb, depth_m=depth, timestamp_s=record['timestamp_s'])

    current = inputs['observation']
    if any(past['timestamp_s'] >= current['timestamp_s'] for past in inputs['history']):
        raise ValueError('History is not strictly earlier than the current observation')
    now = observation(current)
    return dict(instruction=inputs['instruction'], rgb=now['rgb'], depth_m=now['depth_m'],
                history=[observation(past) for past in inputs['history']])


def validate_training_view(view):
    view, manifest, base = _view_base(view)
    checks = 0
    errors = []

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition and len(errors) < 100:
            errors.append(message)

    try:
        stride = manifest['observation_stride']
        if type(stride) is not int or stride < 1 or not 0 <= manifest['sampled_history_frames'] <= 32:
            raise ValueError('Invalid observation/history stride configuration')
        _, episodes = _base_data(base, stride)
        check(manifest['schema_version'] == VIEW_SCHEMA, 'Wrong observation-view schema')
        check(manifest['complete'] is True and manifest['training_ready'] is False, 'Wrong view completion/readiness state')
        check(manifest['native_action_dt_s'] == NATIVE_DT_S, 'Native action dt was changed')
        check(manifest['nominal_observation_interval_s'] == stride * NATIVE_DT_S, 'Wrong observation interval')
        check(sha256(base / 'dataset_manifest.json') == manifest['base_dataset']['manifest_sha256'], 'Dense manifest changed')
        check(sha256(base / 'reader_index.jsonl') == manifest['base_dataset']['reader_index_sha256'], 'Dense reader index changed')
        check(sha256(view / 'reader_index.jsonl') == manifest['reader_index_sha256'], 'View index hash changed')
        check(sha256(view / 'base_observation_hashes.json') == manifest['base_observation_hashes_sha256'], 'Observation hash list changed')
        for name, digest in manifest['base_dataset']['source_files_sha256'].items():
            check(sha256(safe_base_path(base, name)) == digest, 'Dense native label/identity file changed: ' + name)
        observation_hashes = json.loads((view / 'base_observation_hashes.json').read_text())
        for name, digest in observation_hashes.items():
            check(sha256(safe_base_path(base, name)) == digest, 'Selected source observation changed: ' + name)
        aliases, selected_ids, selected_paths = {}, set(), set()
        rows = iter(jsonl_rows(view / 'reader_index.jsonl'))
        dense_count = view_count = retained_cot = dropped_cot = 0
        for original in jsonl_rows(base / 'reader_index.jsonl'):
            episode = episodes[original['physical_trajectory_id']]
            _track_dense_row(original, episode, aliases)
            dense_count += 1
            if original['step_index'] not in episode['following']:
                dropped_cot += int(original['supervision'].get('cot_valid', False))
                continue
            check(original['sample_id'] not in selected_ids, 'Duplicate selected dense reader sample ID')
            selected_ids.add(original['sample_id'])
            row = next(rows, None)
            if row is None:
                check(False, 'Selected dense reader samples are missing from the view')
                break
            view_count += 1
            check(row['sample_id'] == row['base_reader_sample_id'] == original['sample_id'], 'View order/sample differs from the selected dense frame')
            index = row['native_step_index']
            arrays = episode['arrays']
            check(row['step_index'] == index == original['step_index'] and index in episode['following'], 'Native observation index changed or is off-grid')
            check(row['schema_version'] == VIEW_SCHEMA and row['path_base'] == 'dense_base'
                  and row['observation_stride'] == stride, 'Wrong row schema/path base/stride')
            check(row['episode_id'] == original['episode_id'] and row['physical_trajectory_id'] == original['physical_trajectory_id'], 'View crossed language or physical episodes')
            check(row['split'] == original['split'] and row['layout_seed'] == original['layout_seed'], 'Split or layout changed')
            inputs = row['inputs']
            check(set(inputs) == {'instruction', 'observation', 'history'}, 'Privileged fields entered inputs')
            check(inputs['instruction'] == original['inputs']['instruction'], 'Query/instruction changed')
            check(inputs['observation'] == original['inputs']['observation'] == episode['observations'][index], 'Current RGB-D identity differs from the native frame')
            selected_paths.update(inputs['observation'][key] for key in ('rgb', 'depth'))
            for key, value in original.items():
                if key not in {'schema_version', 'inputs', 'supervision'}:
                    check(row.get(key) == value, 'Dense row metadata was not preserved: ' + key)
            position = episode['selected'].index(index)
            history = episode['selected'][max(0, position - manifest['sampled_history_frames']):position]
            if manifest['sampled_history_frames'] == 0:
                history = []
            check(inputs['history'] == [episode['observations'][i] for i in history], 'History is not exact sampled past observations')
            check(all(record['timestamp_s'] < inputs['observation']['timestamp_s'] for record in inputs['history']), 'History contains the current or a future observation')
            check(abs(inputs['observation']['timestamp_s'] - index * NATIVE_DT_S) < 1e-8, 'Native frame time changed')
            labels = row['supervision']
            for key, value in original['supervision'].items():
                check(labels.get(key) == value, 'Original exact-frame supervision changed: ' + key)
            check(labels['step_index'] == index and labels['native_action_dt_s'] == NATIVE_DT_S, 'Labels were retimed to the sparse observation interval')
            check(labels['future_horizon_native_steps'] == 8, 'Native future horizon changed')
            check(np.array_equal(np.asarray(labels['future_vw_si'], np.float32), arrays['future_vw_si'][index]), 'Native future-eight controls were skipped or averaged')
            check(np.array_equal(np.asarray(labels['horizon_valid'], bool), arrays['horizon_valid'][index]), 'Native future validity mask changed')
            following = episode['following'][index]
            expected_sampling = dict(current_native_index=index,
                next_observation_native_index=following if following < episode['count'] else None,
                actual_next_observation_interval_s=(following - index) * NATIVE_DT_S if following < episode['count'] else None,
                terminal_observation=index == episode['count'] - 1)
            check(row['sampling'] == expected_sampling, 'Sparse observation timing/terminal metadata changed')
            chunk = labels['execution_chunk']
            check(chunk['start_native_index'] == index and chunk['end_native_index_exclusive'] == following and chunk['control_dt_s'] == NATIVE_DT_S, 'Execution chunk boundaries or dt changed')
            check(chunk['native_control_indices'] == list(range(index, following)), 'Intermediate native controls are missing')
            check(np.array_equal(np.asarray(chunk['vw_si'], np.float32), arrays['vw_si'][index:following]), 'Executed native controls were averaged or modified')
            check(np.array_equal(np.asarray(chunk['stop'], np.uint8), arrays['stop'][index:following]), 'Execution chunk STOP changed')
            check(labels['native_stop'] == int(arrays['stop'][index]), 'Native STOP target changed')
            retained_cot += int(labels.get('cot_valid', False))
        check(next(rows, None) is None, 'View contains extra, duplicated or off-grid samples')
        for state in aliases.values():
            check(state['count'] == episodes[state['identity'][0]]['count'], 'Dense language episode is incomplete')
        check({state['identity'][0] for state in aliases.values()} == set(episodes), 'Dense reader lost a physical episode')
        check(len(aliases) == manifest['language_episode_count'], 'Language alias count differs from manifest')
        check(dense_count == manifest['statistics']['dense_reader_rows'], 'Dense reader count differs from manifest')
        check(view_count == manifest['statistics']['view_reader_rows'], 'View row count differs from manifest')
        check(retained_cot == manifest['statistics']['retained_cot_reader_rows'], 'Retained CoT count differs from manifest')
        check(dropped_cot == manifest['statistics']['dropped_off_grid_cot_reader_rows'], 'Dropped off-grid CoT count differs from manifest')
        check(set(observation_hashes) == selected_paths, 'Selected RGB-D hash coverage is incomplete or foreign')
        physical = []
        source_files = {}
        for identity, episode in episodes.items():
            source_files.update(episode['source_files'])
            physical.append(dict(episode_id=identity, native_action_count=episode['count'],
                selected_native_indices=episode['selected'], split=episode['record']['split']))
        check(physical == manifest['physical_episodes'], 'Physical native/selected frame identities differ from manifest')
        check(source_files == manifest['base_dataset']['source_files_sha256'], 'Dense native label/identity hash coverage changed')
        check(sum(item['native_action_count'] for item in physical) == manifest['native_physical_frame_count'], 'Native physical frame count changed')
        check(sum(len(item['selected_native_indices']) for item in physical) == manifest['sampled_physical_frame_count'], 'Sampled physical frame count changed')
    except (KeyError, ValueError, TypeError, OSError, IndexError) as exc:
        check(False, 'Malformed or incomplete training view: ' + type(exc).__name__ + ': ' + str(exc))
    return dict(schema_version='lunar_training_stride_validation/14.0', passed=not errors,
        checks=checks, errors=errors, observation_stride=manifest.get('observation_stride'),
        native_action_dt_s=NATIVE_DT_S, training_ready=False,
        scope='native RGB-D/action/history/text alignment of an observation-stride view')
