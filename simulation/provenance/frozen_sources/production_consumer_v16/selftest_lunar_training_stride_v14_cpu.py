"""CPU alignment regressions for sparse observations and dense native actions."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from cave.lunar_training_stride_v14 import (export_training_view, jsonl_rows,
    load_policy_inputs, sha256, validate_training_view, write_json)


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def fixture(base):
    base.mkdir(parents=True)
    episodes, reader = [], []
    for episode_number, count in enumerate((13, 11, 1)):
        identity = f'episode_{episode_number}'
        trajectory = Path('episodes') / identity / 'trajectory'
        directory = base / trajectory
        (directory / 'frames').mkdir(parents=True)
        write_json(directory / 'metadata.json', dict(terminal_success=True, capture_stride=1))
        controls = np.column_stack((np.arange(1, count + 1) / 100,
                                    np.arange(count) / 1000)).astype(np.float32)
        controls[-1] = 0
        stop = np.zeros(count, np.uint8)
        stop[-1] = 1
        future = np.zeros((count, 8, 2), np.float32)
        valid = np.zeros((count, 8), bool)
        for index in range(count):
            length = min(8, count - index)
            future[index, :length] = controls[index:index + length]
            valid[index, :length] = True
        np.savez_compressed(directory / 'supervision.npz', vw_si=controls, stop=stop,
            dt_s=np.full(count, .2, np.float32), future_vw_si=future, horizon_valid=valid)
        frames = []
        for index in range(count):
            rgb = f'frames/{index:06d}_rgb224.png'
            depth = f'frames/{index:06d}_sensors.npz'
            Image.new('RGB', (224, 224), (index, episode_number, 0)).save(directory / rgb)
            np.savez_compressed(directory / depth, depth_m=np.full((256, 256), index + .5, np.float32),
                                privileged_goal_xyz=np.asarray([999, 888, 777]))
            observation = dict(observation_id=f'{identity}:{index:06d}', rgb=rgb,
                               depth=depth, timestamp_s=index * .2)
            frames.append(dict(step_index=index, inputs=dict(observation=observation)))
        write_jsonl(directory / 'frame_records.jsonl', frames)
        split = 'validation' if episode_number == 1 else 'train'
        episodes.append(dict(episode_id=identity, trajectory=trajectory.as_posix(),
                             layout_seed=episode_number, split=split))
        for variant in ('canonical', 'query_01'):
            language_id = f'{identity}:{variant}'
            for index, frame in enumerate(frames):
                observation = copy.deepcopy(frame['inputs']['observation'])
                for key in ('rgb', 'depth'):
                    observation[key] = (trajectory / observation[key]).as_posix()
                cot = index in {3, 5, count - 1}
                reader.append(dict(schema_version='lunar_reader_row/11.0', domain='isaac_continuous',
                    sample_id=f'{language_id}:{index:06d}', episode_id=language_id,
                    physical_trajectory_id=identity, layout_seed=episode_number, split=split,
                    step_index=index, training_ready=False, query_alias=variant,
                    inputs=dict(instruction=f'{identity} instruction {variant}', observation=observation, history=[]),
                    supervision=dict(continuous_controls=(trajectory / 'supervision.npz').as_posix(),
                        step_index=index, cot_valid=cot,
                        text=dict(observation=f'exact frame {index}') if cot else None,
                        cot_text=f'exact frame {index}' if cot else None),
                    privileged=dict(teacher_only='not_a_policy_input.npz')))
    write_json(base / 'dataset_manifest.json', dict(complete=True, episodes=episodes))
    write_jsonl(base / 'reader_index.jsonl', reader)


def main():
    with tempfile.TemporaryDirectory(prefix='lunar_stride_v14_selftest_') as temporary:
        package = Path(temporary) / 'package'
        base, view = package / 'dense', package / 'stride5'
        fixture(base)
        before = {p.relative_to(base).as_posix(): sha256(p) for p in base.rglob('*') if p.is_file()}
        manifest, report = export_training_view(base, view)
        assert report['passed'], report
        assert manifest['native_physical_frame_count'] == 25
        assert manifest['sampled_physical_frame_count'] == 8
        assert manifest['statistics']['view_reader_rows'] == 16
        assert manifest['statistics']['dropped_off_grid_cot_reader_rows'] == 4
        rows = list(jsonl_rows(view / 'reader_index.jsonl'))
        for episode, indices in (('episode_0', [0, 5, 10, 12]), ('episode_1', [0, 5, 10]), ('episode_2', [0])):
            for variant in ('canonical', 'query_01'):
                selected = [row for row in rows if row['episode_id'] == f'{episode}:{variant}']
                assert [row['step_index'] for row in selected] == indices
                with np.load(base / f'episodes/{episode}/trajectory/supervision.npz') as source:
                    concatenated = np.concatenate([row['supervision']['execution_chunk']['vw_si'] for row in selected])
                    np.testing.assert_array_equal(concatenated.astype(np.float32), source['vw_si'])
                    assert sum(sum(row['supervision']['execution_chunk']['stop']) for row in selected) == 1
                for position, row in enumerate(selected):
                    chunk = row['supervision']['execution_chunk']
                    assert chunk['control_dt_s'] == .2 and len(chunk['vw_si']) <= 5
                    assert [past['timestamp_s'] for past in row['inputs']['history']] == [i * .2 for i in indices[:position]]
                    assert row['query_alias'] == variant
                    if row['supervision']['cot_valid']:
                        assert row['supervision']['cot_text'] == f'exact frame {row["step_index"]}'
        first = [row for row in rows if row['episode_id'] == 'episode_0:canonical']
        assert [row['supervision']['execution_chunk']['native_control_indices'] for row in first] == [list(range(5)), list(range(5, 10)), [10, 11], [12]]
        assert first[-2]['sampling']['actual_next_observation_interval_s'] == .4
        assert first[-1]['sampling']['next_observation_native_index'] is None
        # Eight targets at native indices 5..12, not sparse indices 5,10,... .
        np.testing.assert_array_equal(np.asarray(first[1]['supervision']['future_vw_si'], np.float32)[:, 0],
                                      np.asarray([.06, .07, .08, .09, .10, .11, .12, 0], np.float32))
        policy = load_policy_inputs(view, first[1])
        assert set(policy) == {'instruction', 'rgb', 'depth_m', 'history'}
        assert set(policy['history'][0]) == {'rgb', 'depth_m', 'timestamp_s'}
        assert policy['rgb'][0, 0, 0] == 5 and policy['history'][0]['rgb'][0, 0, 0] == 0
        # Keep index hashes valid, so semantic checks must detect each corruption.
        original_index = (view / 'reader_index.jsonl').read_bytes()
        original_manifest = (view / 'view_manifest.json').read_bytes()
        mutations = {
            'retimed_action_dt': lambda row: row['supervision']['execution_chunk'].update(control_dt_s=1.),
            'averaged_or_modified_control': lambda row: row['supervision']['execution_chunk']['vw_si'][0].__setitem__(0, .399),
            'current_frame_history_leak': lambda row: row['inputs']['history'].append(row['inputs']['observation']),
            'snapped_offgrid_cot': lambda row: row['supervision'].update(cot_text='exact frame 3'),
            'wrong_future_horizon': lambda row: row['supervision'].update(future_horizon_native_steps=40),
        }
        for name, mutate in mutations.items():
            corrupted = copy.deepcopy(rows)
            mutate(corrupted[1])
            write_jsonl(view / 'reader_index.jsonl', corrupted)
            changed = copy.deepcopy(manifest)
            changed['reader_index_sha256'] = sha256(view / 'reader_index.jsonl')
            write_json(view / 'view_manifest.json', changed)
            assert not validate_training_view(view)['passed'], name
        (view / 'reader_index.jsonl').write_bytes(original_index)
        (view / 'view_manifest.json').write_bytes(original_manifest)
        # Entire package remains usable after moving; no workspace path dependency.
        relocated = Path(temporary) / 'relocated'
        shutil.copytree(package, relocated)
        assert validate_training_view(relocated / 'stride5')['passed']
        assert load_policy_inputs(relocated / 'stride5', first[1])['rgb'][0, 0, 0] == 5
        bad_base = Path(temporary) / 'bad_dense'
        shutil.copytree(base, bad_base)
        incomplete = list(jsonl_rows(bad_base / 'reader_index.jsonl'))
        incomplete.pop(2)  # A skipped observation still must exist in the dense base.
        write_jsonl(bad_base / 'reader_index.jsonl', incomplete)
        unpublished = Path(temporary) / 'must_not_publish'
        try:
            export_training_view(bad_base, unpublished)
            raise AssertionError('Incomplete dense base was accepted')
        except ValueError:
            assert not unpublished.exists()
        assert before == {p.relative_to(base).as_posix(): sha256(p) for p in base.rglob('*') if p.is_file()}
        print(json.dumps(dict(passed=True, synthetic_native_frames=25, sampled_physical_frames=8,
            reader_rows=16, negative_regressions=list(mutations), relocation_passed=True,
            terminal_cases=[13, 11, 1], atomic_failure_not_published=True, dense_base_unchanged=True)))


if __name__ == '__main__':
    main()
