"""Ground route instructions and their current-frame language in real events."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

from .lunar_action_intent_v12 import action_intent

OBJECT_NAMES = {'astronaut': '宇航员', 'lunar_rover': '月球车',
                'lunar_base': '月球基地', 'chinese_flag': '红旗'}
OBJECT_ALIASES = {'astronaut': ('宇航员', '航天员'),
                  'lunar_rover': ('月球车', '探月车'),
                  'lunar_base': ('月球基地', '基地'),
                  'chinese_flag': ('中国国旗', '红旗', '国旗')}
ROUTE_QUERY_SCHEMA = dict(type='object', additionalProperties=False, required=['queries'], properties={
    'queries': dict(type='array', minItems=3, maxItems=3,
                    items=dict(type='string', minLength=4, maxLength=240))})


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def object_digest(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_frames(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def raw_manifest_files(source, manifest_path=None, validation_binding_path=None, validation_path=None):
    paths = [path for path in sorted(source.rglob('*')) if path.is_file()
             and path.name != 'REQUEST_STOP' and not path.name.endswith('.tmp')]
    if any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError('Raw route source cannot reference external symlinks')
    if manifest_path is None and validation_binding_path is None:
        return [[path.relative_to(source).as_posix(), path.stat().st_size, digest(path)] for path in paths], None
    if manifest_path is None or validation_binding_path is None:
        raise ValueError('A reused raw manifest requires its completed integrity binding')
    manifest_path = Path(manifest_path).resolve(strict=True)
    validation_binding_path = Path(validation_binding_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    bound = json.loads(validation_binding_path.read_text())
    files = manifest['files']
    validation = Path(validation_path or bound['validation_file']).expanduser()
    relocated = Path(bound['raw_source']).resolve() != source
    if (bound['passed'] is not True or object_digest(files) != manifest['manifest_sha256']
            or manifest['manifest_sha256'] != bound['raw_manifest_sha256']
            or bound['raw_file_count'] != len(files) or bound['raw_bytes'] != sum(row[1] for row in files)):
        raise ValueError('Raw integrity report does not bind this capture manifest')
    # Absolute paths in historical audit JSON are provenance, not portable inputs.
    # A relocated source or absent old report uses fresh full hashing instead of
    # trusting the original machine's size-only fast path.
    if relocated or not validation.is_file():
        fresh = [[path.relative_to(source).as_posix(), path.stat().st_size, digest(path)] for path in paths]
        if fresh != files:
            raise ValueError('Relocated raw capture differs from its original manifest')
        return fresh, None
    validation = validation.resolve(strict=True)
    if (digest(validation) != bound['validation_sha256']
            or json.loads(validation.read_text())['passed'] is not True):
        raise ValueError('Raw validation report bytes differ from their integrity binding')
    expected = {name: (size, sha) for name, size, sha in files}
    if len(expected) != len(files) or {path.relative_to(source).as_posix() for path in paths} != set(expected):
        raise ValueError('Raw file list changed after its integrity audit')
    for path in paths:
        name = path.relative_to(source).as_posix()
        if path.stat().st_size != expected[name][0]:
            raise ValueError('Raw file size changed after its integrity audit: ' + name)
    for name in ('metadata.json', 'frame_records.jsonl', 'teacher_only.npz', 'planning_teacher_only.json', 'status.json'):
        if digest(source / name) != expected[name][1]:
            raise ValueError('Raw control/evidence metadata changed after its integrity audit: ' + name)
    return files, dict(binding_path=str(validation_binding_path), binding_sha256=digest(validation_binding_path),
                       validation_path=str(validation), validation_sha256=digest(validation))


def recompute_fork_evidence(source, frames, raw_files, layout=None):
    """Recompute only fork fields from each unchanged calibrated depth frame."""
    import io
    import numpy as np
    from build_lunar_cave_v4_cpu import evaluate_scene_field
    from .lunar_fork_evidence_v13 import ForkEvidence

    metadata = json.loads((source / 'metadata.json').read_text())
    scene = Path(layout or metadata['source_scene']).expanduser().resolve(strict=True)
    scene_hashes = {name: digest(scene / name) for name in metadata['source_sha256']}
    if scene_hashes != metadata['source_sha256']:
        raise ValueError('Fork geometry differs from the scene used in this capture')
    config = json.loads((scene / 'scene_config.json').read_text())
    targets = json.loads((scene / 'targets.json').read_text())['targets']
    world = SimpleNamespace(scene=scene, config=config, targets={target['id']: target for target in targets},
                            source_hashes=scene_hashes,
                            field=lambda points: np.asarray(evaluate_scene_field(config, points), float))
    hashes = {name: sha for name, _, sha in raw_files}
    with np.load(source / 'teacher_only.npz', allow_pickle=False) as teacher:
        guide = teacher['source_path_xy_m'].copy()
    detector = ForkEvidence(world, metadata['target_id'], guide)
    enriched, observations, changed = [], [], 0
    keys = ('fork_visible_verified', 'fork_choice_side', 'fork_evidence_criterion')
    for frame in frames:
        path = source / frame['sensors_file']
        payload = path.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if sha != hashes[frame['sensors_file']]:
            raise ValueError('Fork depth input differs from the completed raw integrity audit')
        with np.load(io.BytesIO(payload), allow_pickle=False) as sensor:
            state = detector.observe(sensor['K256'], sensor['world_from_camera'], sensor['depth_m'], sensor['depth_valid'])
        if (set(state) != set(keys) or not isinstance(state['fork_visible_verified'], bool)
                or state['fork_choice_side'] not in ('left', 'right', 'unknown')
                or state['fork_visible_verified'] and state['fork_choice_side'] == 'unknown'):
            raise ValueError('Fork evidence module returned an invalid current-frame result')
        row = deepcopy(frame)
        before = {key: frame['machine_evidence'].get(key) for key in keys}
        row['machine_evidence'].update(state)
        changed += int(before != state)
        enriched.append(row)
        observations.append(dict(step_index=frame['step_index'], source_observation_id=frame['sample_id'],
                                 sensors_file=frame['sensors_file'], sensors_sha256=sha,
                                 before=before, after=state))
    module = Path(__file__).with_name('lunar_fork_evidence_v13.py')
    audit = dict(schema_version='lunar_current_depth_fork_enrichment/13.0', revision='v13',
                 physical_trajectory_id=metadata['episode_id'], module='cave/lunar_fork_evidence_v13.py',
                 module_sha256=digest(module), source_frame_records_sha256=hashes['frame_records.jsonl'],
                 source_teacher_only_sha256=hashes['teacher_only.npz'], scene_sha256=scene_hashes,
                 raw_capture_manifest_sha256=object_digest(raw_files), fields=list(keys), observations=observations,
                 changed_frames=changed, positive_frames=sum(row['after']['fork_visible_verified'] for row in observations),
                 criterion='new module evaluates current K/T/depth with validated arm geometry and actual depth occlusion',
                 original_rgb_depth_controls_poses_unchanged=True, GPU_rendered=False)
    return enriched, audit


def frame_intent(frame):
    action = frame['executed_action']
    return action_intent(action['v_mps'], action['omega_radps'], action['stop'],
                         action.get('motion_controller', 'segment'))


def visible_ids(frame):
    evidence = frame['machine_evidence']
    threshold = evidence.get('landmark_visibility_threshold_pixels')
    if not isinstance(threshold, int) or threshold < 12 or not evidence.get('landmark_evidence_method'):
        raise ValueError('Route language requires verified current-object mask evidence')
    result = set()
    for item in evidence['visible_objects']:
        identifier = item['id']
        if (identifier not in OBJECT_NAMES or identifier in result
                or item.get('is_goal') is not (identifier == frame['target_id'])
                or not isinstance(item['pixels'], int) or item['pixels'] < threshold):
            raise ValueError('Malformed current visible-object evidence')
        box = item['bbox_xyxy']
        if (len(box) != 4 or not 0 <= box[0] < box[2] <= 256
                or not 0 <= box[1] < box[3] <= 256):
            raise ValueError('Visible-object bbox is outside the current camera')
        result.add(identifier)
    return result


def build_route_contract(frames):
    if frames and frames[0]['machine_evidence'].get('semantic_visibility_revision') == '14.0':
        from .lunar_route_materializer_v14 import build_route_contract as build_v14
        return build_v14(frames)
    if not frames:
        raise ValueError('An actual completed episode is required')
    first = frames[0]
    name, target = first['episode_id'], first['target_id']
    if target not in OBJECT_NAMES:
        raise ValueError('Unknown route goal')
    seen, first_seen, events = set(), [], []
    last_time = -math.inf
    entry = None
    fork = None
    for index, frame in enumerate(frames):
        timestamp = frame['inputs']['observation']['timestamp_s']
        if ((frame['episode_id'], frame['target_id'], frame['step_index']) != (name, target, index)
                or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp <= last_time):
            raise ValueError('Route frame identity/order is inconsistent')
        last_time = timestamp
        evidence = frame['machine_evidence']
        visible = visible_ids(frame)
        now = visible - seen
        if (set(evidence['landmark_seen_before']) != seen
                or set(evidence['landmark_first_seen']) != now
                or set(evidence['landmark_seen_so_far']) != seen | visible):
            raise ValueError('Landmark history is not strictly causal')
        declared = evidence['landmark_events']
        if (len(declared) != len(now) or {event['id'] for event in declared} != now
                or any(event.get('type') != 'first_seen' or event['timestamp_s'] != timestamp
                       or event.get('is_goal') is not (event['id'] == target) for event in declared)):
            raise ValueError('Landmark first-seen events do not match the current observation')
        support = dict(step_index=index, timestamp_s=timestamp,
                       source_observation_id=frame['sample_id'])
        for identifier in sorted(now):
            first_seen.append(dict(type='landmark_continue', id=identifier,
                                   name=OBJECT_NAMES[identifier], **support))
        if entry is None and evidence.get('entry_event') == 'entered_cave':
            entry = dict(type='enter_cave', **support)
        side = evidence.get('fork_choice_side')
        if (fork is None and evidence.get('fork_visible_verified') is True
                and side in ('left', 'right')):
            fork = dict(type='first_fork', ordinal=1, side=side,
                        criterion=evidence['fork_evidence_criterion'], **support)
        seen |= visible

    terminal = frames[-1]
    evidence = terminal['machine_evidence']
    if (frame_intent(terminal) != 'stop' or evidence.get('terminal_accepted') is not True
            or evidence.get('target_visible') is not True or target not in visible_ids(terminal)):
        raise ValueError('Route contract requires an actually visible accepted terminal goal STOP')
    stop = dict(type='goal_stop', id=target, name=OBJECT_NAMES[target],
                step_index=terminal['step_index'], timestamp_s=terminal['inputs']['observation']['timestamp_s'],
                source_observation_id=terminal['sample_id'])
    if entry is not None:
        events.append(entry)
    if fork is not None:
        events.append(fork)
    landmarks = []
    for event in first_seen:
        index = event['step_index']
        if (event['id'] == target or entry is None or index <= entry['step_index']
                or index >= stop['step_index'] or frame_intent(frames[index]) != 'move_forward'
                or fork is not None and index == fork['step_index']):
            continue
        landmarks.append(event)
        events.append(event)
    events.sort(key=lambda event: event['step_index'])
    events.append(stop)
    return dict(schema_version='lunar_route_language_contract/13.0', physical_trajectory_id=name,
                goal=dict(id=target, name=OBJECT_NAMES[target]), events=events,
                intermediate_landmarks=landmarks,
                deep_target_hint=bool(entry and fork and entry['step_index'] < fork['step_index'] < stop['step_index']),
                semantic_scope='instruction hints derived from verified completed route events; no world coordinates',
                omitted_unobserved_landmarks=True)


def canonical_instruction(contract):
    target = contract['goal']['name']
    start = f'找到{target}'
    if contract['deep_target_hint']:
        start += '，它在洞穴深处'
    clauses = []
    for event in contract['events']:
        if event['type'] == 'enter_cave':
            clauses.append('进入洞穴')
        elif event['type'] == 'first_fork':
            clauses.append('在第一个分叉向' + ('左' if event['side'] == 'left' else '右') + '转')
        elif event['type'] == 'landmark_continue':
            clauses.append(f'看到{event["name"]}之后继续前进')
        elif event['type'] == 'goal_stop':
            clauses.append(f'最后靠近{target}并停下来')
    return start + '。' + '，'.join(clauses) + '。'


def route_query_prompt(contract):
    return ('将这条已核验的月洞导航任务改写为三个不同的中文 query，JSON 只有 queries 列表。'
            '每条最多240字。必须保留目标、进入洞穴、每个地标后继续前进、'
            '第一个分叉的左右方向、这些提示的先后顺序，以及最后靠近目标停下来。'
            '只保留原任务具有的提示，不添加地标、方向或精确距离。'
            '用中文“第一个”，不要写数字或坐标。看到目标并不等于已经靠近目标，'
            '末尾明确靠近后停止。\n已核验任务：' + canonical_instruction(contract))


def check_route_queries(value, contract, *, expected_count=3):
    if (not isinstance(value, dict) or set(value) != {'queries'}
            or not isinstance(value['queries'], list) or len(value['queries']) != expected_count
            or len(set(value['queries'])) != expected_count):
        raise ValueError('Route queries must contain the prescribed number of distinct variants')
    goal = contract['goal']['id']
    allowed_objects = {goal} | {event['id'] for event in contract['intermediate_landmarks']}
    for original in value['queries']:
        if not isinstance(original, str) or not 4 <= len(original) <= 240 or '\n' in original:
            raise ValueError('Invalid route query length/type')
        query = re.sub(r'\s+', '', original).replace(',', '，')
        if re.search(r'\d|坐标|米|公里|度|秒|出洞|离开洞穴|返回入口|掉头|第二|第三|不要|不必|无需', query):
            raise ValueError('Route query adds a metric, new route event or negative instruction')
        aliases = '|'.join(map(re.escape, OBJECT_ALIASES[goal]))
        goal_verb = r'(?:寻找|找到|去找|前往|找|目标(?:是|为))'
        if not re.search(goal_verb + r'[^，。；]{0,12}(?:' + aliases + ')', query):
            raise ValueError('Route query changed or omitted its goal')
        for identifier, names in OBJECT_ALIASES.items():
            if identifier not in allowed_objects and any(name in query for name in names):
                raise ValueError('Route query adds an unsupported landmark')
            if identifier != goal:
                alternative = '|'.join(map(re.escape, names))
                if re.search(goal_verb + r'[^，。；]{0,12}(?:' + alternative + ')', query):
                    raise ValueError('Route query changes an intermediate landmark into another goal')
        if bool(re.search(r'(?:洞穴|洞内|熔岩洞)?(?:更)?深处', query)) != contract['deep_target_hint']:
            raise ValueError('Route query changed the deep-cave hint')
        positions = []
        fork_side = None
        for event in contract['events']:
            kind = event['type']
            if kind == 'enter_cave':
                match = re.search(r'进入洞穴|进洞|走进洞穴|穿过洞口|进入熔岩洞', query)
            elif kind == 'first_fork':
                fork_side = '左' if event['side'] == 'left' else '右'
                match = re.search(r'(?:第一个|首个|第一处)(?:分叉|岔口|岔路口|岔路)[^，。；]{0,10}' + fork_side, query)
            elif kind == 'landmark_continue':
                names = '|'.join(map(re.escape, OBJECT_ALIASES[event['id']]))
                match = re.search(r'(?:看到|看见|见到|经过)[^，。；]{0,6}(?:' + names +
                                  r')[^，。；]{0,4}(?:之后|以后|后)[，、]?(?:请|再|就|仍|沿路|沿通道){0,2}'
                                  r'(?:继续(?:向前|往前)?(?:前进|走|行驶)?|接着(?:前进|走)|向前(?:前进|走))', query)
            else:
                continue
            if not match:
                raise ValueError('Route query omitted or changed a verified route hint')
            positions.append(match.start())
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise ValueError('Route query changed the order of verified route hints')
        opposite = '右' if fork_side == '左' else '左'
        if ((fork_side is None and re.search(r'左|右|分叉|岔口|岔路', query))
                or (fork_side is not None and opposite in query)):
            raise ValueError('Route query adds or reverses a branch direction')
        stop = list(re.finditer(r'靠近[^。；]{0,15}(?:停下来|停下|停止|停车)|(?:附近|旁边)[^。；]{0,8}(?:停下来|停下|停止|停车)', query))
        if not stop or positions and stop[-1].start() < positions[-1]:
            raise ValueError('Route query must stop near the goal after its route hints')
    return True


def route_frame_pair(frame, contract):
    if contract.get('semantic_visibility_revision') == '14.0':
        from .lunar_route_materializer_v14 import route_frame_pair as pair_v14
        return pair_v14(frame, contract)
    """Return only a currently verified waypoint pair from the fixed instruction."""
    index = frame['step_index']
    evidence = frame['machine_evidence']
    intent = frame_intent(frame)
    for event in contract['events']:
        if event['step_index'] != index:
            continue
        if event['type'] == 'first_fork':
            if (evidence.get('fork_visible_verified') is True
                    and evidence.get('fork_choice_side') == event['side']
                    and intent in ('move_forward', 'turn_left', 'turn_right')):
                side = '左' if event['side'] == 'left' else '右'
                return '洞内搜索', ('我看到了第一个分叉', f'提示让我走{side}边')
        elif event['type'] == 'landmark_continue':
            if event['id'] in visible_ids(frame) and intent == 'move_forward':
                return '洞内搜索', (f'我看到了{event["name"]}', '按照提示我应该继续前进')
        elif event['type'] == 'enter_cave' and intent == 'move_forward':
            return '进入洞穴', ('我看到了洞口', '按照提示我应该进去')
    return None


def materialize_route_episode(source, output, *, fork_evidence_revision=None,
                              raw_manifest_path=None, raw_validation_binding_path=None,
                              layout=None, raw_validation_path=None):
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if output.exists() or output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Choose a fresh, disjoint route-materialization output')
    metadata = json.loads((source / 'metadata.json').read_text())
    status = json.loads((source / 'status.json').read_text())
    if metadata.get('instruction_style') == 'route_grounded':
        raise ValueError('This source is already materialized; validate or annotate that route view')
    if status['phase'] != 'complete' or metadata['terminal_success'] is not True:
        raise ValueError('Route language materialization requires a completed raw capture')
    frames = load_frames(source / 'frame_records.jsonl')
    original_instruction = frames[0]['instruction']
    if any(frame['instruction'] != original_instruction or frame['inputs']['instruction'] != original_instruction for frame in frames):
        raise ValueError('Raw capture instruction changes within its episode')
    if layout is not None:
        layout = Path(layout).expanduser().resolve(strict=True)
        if {name: digest(layout / name) for name in metadata['source_sha256']} != metadata['source_sha256']:
            raise ValueError('Explicit layout override differs from the captured scene hashes')
    files, raw_proof = raw_manifest_files(source, raw_manifest_path, raw_validation_binding_path, raw_validation_path)
    enrichment = None
    if fork_evidence_revision is not None:
        if fork_evidence_revision != 'v13':
            raise ValueError('Unknown fork evidence enrichment revision')
        frames, enrichment = recompute_fork_evidence(source, frames, files, layout)
    contract = build_route_contract(frames)
    canonical = canonical_instruction(contract)
    output.mkdir(parents=True)
    for name, _, _ in files:
        if name in ('metadata.json', 'frame_records.jsonl'):
            continue
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source / name, destination)
        except OSError:
            shutil.copy2(source / name, destination)
    audit = output / 'raw_capture'
    audit.mkdir()
    shutil.copy2(source / 'metadata.json', audit / 'metadata.json')
    shutil.copy2(source / 'frame_records.jsonl', audit / 'frame_records.jsonl')
    raw_manifest = dict(files=files, source=str(source), manifest_sha256=object_digest(files))
    if raw_proof is not None:
        raw_manifest['reused_integrity_binding'] = dict(binding_sha256=raw_proof['binding_sha256'],
                                                       validation_sha256=raw_proof['validation_sha256'])
        shutil.copy2(raw_proof['binding_path'], audit / 'integrity_binding.json')
        shutil.copy2(raw_proof['validation_path'], audit / 'integrity_validation.json')
    write_json(audit / 'manifest.json', raw_manifest)
    if enrichment is not None:
        write_json(output / 'fork_evidence_enrichment.json', enrichment)
    write_json(output / 'route_contract.json', contract)
    binding = dict(schema_version='lunar_route_instruction_binding/13.0', physical_trajectory_id=contract['physical_trajectory_id'],
                   original_instruction=original_instruction, canonical_instruction=canonical,
                   raw_capture_manifest_sha256=raw_manifest['manifest_sha256'],
                   route_contract_sha256=digest(output / 'route_contract.json'),
                   materializer_code_sha256=digest(Path(__file__)),
                   changed_fields=['frame_records[].instruction', 'frame_records[].inputs.instruction', 'metadata route-language provenance'],
                   observation_history_controls_and_poses_unchanged=True)
    if enrichment is not None:
        binding['changed_fields'].append('frame_records[].machine_evidence current-depth fork fields')
        binding['fork_evidence_enrichment_sha256'] = digest(output / 'fork_evidence_enrichment.json')
    write_json(output / 'instruction_binding.json', binding)
    rebound = []
    for frame in frames:
        row = deepcopy(frame)
        row['instruction'] = canonical
        row['inputs']['instruction'] = canonical
        rebound.append(row)
    (output / 'frame_records.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rebound))
    metadata.update(instruction_style='route_grounded', canonical_instruction=canonical,
                    route_language_binding=dict(file='instruction_binding.json', sha256=digest(output / 'instruction_binding.json'),
                                                original_raw_capture='raw_capture/manifest.json'))
    if enrichment is not None:
        metadata['route_language_binding']['fork_evidence_enrichment_sha256'] = digest(output / 'fork_evidence_enrichment.json')
    write_json(output / 'metadata.json', metadata)
    validate_materialized_episode(output, metadata, rebound)
    return dict(passed=True, source=str(source), output=str(output), frames=len(rebound),
                canonical_instruction=canonical, route_contract=contract)


def validate_materialized_episode(source, metadata=None, frames=None):
    source = Path(source).resolve(strict=True)
    metadata = metadata or json.loads((source / 'metadata.json').read_text())
    if metadata.get('semantic_visibility_revision') == '14.0':
        from .lunar_route_materializer_v14 import validate_materialized
        return validate_materialized(source, metadata, frames)
    frames = frames or load_frames(source / 'frame_records.jsonl')
    contract = json.loads((source / 'route_contract.json').read_text())
    binding = json.loads((source / 'instruction_binding.json').read_text())
    manifest = json.loads((source / 'raw_capture/manifest.json').read_text())
    original_frames = load_frames(source / 'raw_capture/frame_records.jsonl')
    original_metadata = json.loads((source / 'raw_capture/metadata.json').read_text())
    enrichment = None
    enriched_path = source / 'fork_evidence_enrichment.json'
    if enriched_path.is_file():
        enrichment = json.loads(enriched_path.read_text())
    canonical = canonical_instruction(contract)
    if (metadata.get('instruction_style') != 'route_grounded' or metadata['canonical_instruction'] != canonical
            or contract != build_route_contract(frames) or len(frames) != len(original_frames)
            or object_digest(manifest['files']) != manifest['manifest_sha256']
            or binding['raw_capture_manifest_sha256'] != manifest['manifest_sha256']
            or binding['canonical_instruction'] != canonical
            or binding['physical_trajectory_id'] != contract['physical_trajectory_id']
            or binding['original_instruction'] != original_frames[0]['instruction']
            or binding['route_contract_sha256'] != digest(source / 'route_contract.json')
            or metadata['route_language_binding']['sha256'] != digest(source / 'instruction_binding.json')):
        raise ValueError('Derived route instruction/contract/provenance mismatch')
    expected_metadata = deepcopy(original_metadata)
    expected_metadata.update(instruction_style='route_grounded', canonical_instruction=canonical,
                             route_language_binding=dict(file='instruction_binding.json',
                                                         sha256=digest(source / 'instruction_binding.json'),
                                                         original_raw_capture='raw_capture/manifest.json'))
    if enrichment is not None:
        expected_metadata['route_language_binding']['fork_evidence_enrichment_sha256'] = digest(enriched_path)
        hashes = {name: sha for name, _, sha in manifest['files']}
        module = Path(__file__).with_name('lunar_fork_evidence_v13.py')
        if (binding.get('fork_evidence_enrichment_sha256') != digest(enriched_path)
                or enrichment['module_sha256'] != digest(module)
                or enrichment['physical_trajectory_id'] != original_metadata['episode_id']
                or enrichment['scene_sha256'] != original_metadata['source_sha256']
                or enrichment['source_frame_records_sha256'] != hashes['frame_records.jsonl']
                or enrichment['source_teacher_only_sha256'] != hashes['teacher_only.npz']
                or enrichment['raw_capture_manifest_sha256'] != manifest['manifest_sha256']
                or len(enrichment['observations']) != len(frames)):
            raise ValueError('Derived fork observations lack their raw depth/module provenance')
    if metadata != expected_metadata:
        raise ValueError('Route materialization changed unrelated raw metadata')
    if 'reused_integrity_binding' in manifest:
        proof = manifest['reused_integrity_binding']
        if (digest(source / 'raw_capture/integrity_binding.json') != proof['binding_sha256']
                or digest(source / 'raw_capture/integrity_validation.json') != proof['validation_sha256']):
            raise ValueError('Reused raw integrity evidence changed during materialization')
    paths = [row[0] for row in manifest['files']]
    if len(paths) != len(set(paths)):
        raise ValueError('Raw route manifest has duplicate files')
    for name, size, sha in manifest['files']:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Raw route manifest contains an external file path')
        path = source / ('raw_capture/' + name if name in ('metadata.json', 'frame_records.jsonl') else name)
        if not path.is_file() or path.stat().st_size != size or digest(path) != sha:
            raise ValueError('Route materialization changed an immutable raw capture file: ' + name)
    for index, (original, frame) in enumerate(zip(original_frames, frames)):
        expected = deepcopy(original)
        expected['instruction'] = canonical
        expected['inputs']['instruction'] = canonical
        if enrichment is not None:
            observation = enrichment['observations'][index]
            keys = ('fork_visible_verified', 'fork_choice_side', 'fork_evidence_criterion')
            if (observation['step_index'] != index or observation['source_observation_id'] != original['sample_id']
                    or observation['sensors_file'] != original['sensors_file']
                    or observation['sensors_sha256'] != hashes[original['sensors_file']]
                    or observation['before'] != {key: original['machine_evidence'].get(key) for key in keys}
                    or set(observation['after']) != set(keys)):
                raise ValueError('Derived fork observation is not bound to its current calibrated raw depth frame')
            state = observation['after']
            visible, side = state['fork_visible_verified'], state['fork_choice_side']
            if (not isinstance(visible, bool) or not isinstance(state['fork_evidence_criterion'], str)
                    or (visible and side not in ('left', 'right')) or (not visible and side != 'unknown')):
                raise ValueError('Derived fork visibility/side evidence is inconsistent')
            expected['machine_evidence'].update(observation['after'])
        if expected != frame:
            raise ValueError('Route materialization changed observations, actions or history')
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fork-evidence-revision', choices=['v13'])
    parser.add_argument('--raw-manifest', type=Path)
    parser.add_argument('--raw-validation-binding', type=Path)
    parser.add_argument('--raw-validation', type=Path, help='Relocated completed integrity report (optional)')
    parser.add_argument('--layout', type=Path, help='Portable scene directory; checked against raw source_sha256')
    args = parser.parse_args()
    result = materialize_route_episode(args.source, args.output, fork_evidence_revision=args.fork_evidence_revision,
                                      raw_manifest_path=args.raw_manifest,
                                      raw_validation_binding_path=args.raw_validation_binding,
                                      raw_validation_path=args.raw_validation, layout=args.layout)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
