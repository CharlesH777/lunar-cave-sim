"""Immutable raw -> salient route-language view, with exact stride5 cue frames."""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from .lunar_route_language_v13 import (
    OBJECT_NAMES, canonical_instruction, digest, frame_intent, load_frames,
    object_digest, raw_manifest_files, recompute_fork_evidence, visible_ids, write_json,
)
from .lunar_semantic_visibility_v14 import SemanticVisibility, checked_config, config_hash


def semantic_ids(frame):
    return {row['id'] for row in frame['machine_evidence'].get('semantic_visible_objects', [])}


def eligible(frame):
    return frame['step_index'] % 5 == 0 or bool(frame['executed_action']['stop'])


def moving(frame):
    return frame['executed_action']['v_mps'] > .02 and not frame['executed_action']['stop']


def build_route_contract(frames):
    if not frames:
        raise ValueError('A completed salient physical episode is required')
    first, terminal = frames[0], frames[-1]
    name, target = first['episode_id'], first['target_id']
    seen, entry, fork, first_goal = set(), None, None, None
    previous_time = -math.inf
    landmarks, events = [], []
    for index, frame in enumerate(frames):
        evidence = frame['machine_evidence']
        timestamp = frame['inputs']['observation']['timestamp_s']
        if ((frame['episode_id'], frame['target_id'], frame['step_index']) != (name, target, index)
                or evidence.get('semantic_visibility_revision') != '14.0'
                or evidence.get('semantic_observation_eligible') is not eligible(frame)
                or not math.isfinite(timestamp) or timestamp <= previous_time):
            raise ValueError('Salient route frame identity/revision/grid mismatch')
        previous_time = timestamp
        now = sorted(semantic_ids(frame) - seen) if eligible(frame) else []
        if (evidence['semantic_cue_seen_before'] != sorted(seen) or evidence['semantic_cue_first_seen'] != now
                or evidence['semantic_cue_seen_so_far'] != sorted(seen | set(now))):
            raise ValueError('Semantic cue history is not strictly past eligible observations')
        declared = evidence['semantic_cue_events']
        if (len(declared) != len(now) or {row['id'] for row in declared} != set(now)
                or any(row.get('type') != 'first_salient_eligible_observation' or row.get('step_index') != index
                       or row.get('timestamp_s') != timestamp or row.get('is_goal') is not (row['id'] == target)
                       for row in declared)):
            raise ValueError('Semantic first-salient event does not match current eligible observation')
        seen.update(now)
        if not eligible(frame):
            continue
        support = dict(step_index=index, timestamp_s=frame['inputs']['observation']['timestamp_s'],
                       source_observation_id=frame['sample_id'])
        inside = evidence.get('semantic_entry_support')
        if entry is None and moving(frame) and inside is not None:
            entry = dict(type='enter_cave', **support, entry_support=inside)
            events.append(entry)
        side = evidence.get('fork_choice_side')
        if fork is None and evidence.get('fork_visible_verified') is True and side in ('left', 'right'):
            fork = dict(type='first_fork', ordinal=1, side=side, **support,
                        criterion=evidence['fork_evidence_criterion'])
            events.append(fork)
        if first_goal is None and target in semantic_ids(frame):
            first_goal = support
        # "Continue" describes ongoing translational motion along the channel;
        # the separate action_intent still records an actual moving curve/turn.
        if entry is not None and index > entry['step_index'] and moving(frame):
            for identifier in evidence['semantic_cue_first_seen']:
                if (identifier == target or any(row['id'] == identifier for row in landmarks)
                        or fork is not None and index == fork['step_index']):
                    continue
                event = dict(type='landmark_continue', id=identifier, name=OBJECT_NAMES[identifier], **support,
                             criterion='current salient eligible-grid observation with ongoing forward translation')
                landmarks.append(event)
                events.append(event)
    evidence = terminal['machine_evidence']
    if (frame_intent(terminal) != 'stop' or evidence.get('terminal_accepted') is not True
            or evidence.get('target_visible') is not True or target not in visible_ids(terminal)):
        raise ValueError('Salient route requires the actual accepted visible terminal STOP')
    stop = dict(type='goal_stop', id=target, name=OBJECT_NAMES[target], step_index=terminal['step_index'],
                timestamp_s=terminal['inputs']['observation']['timestamp_s'], source_observation_id=terminal['sample_id'],
                goal_salient_in_current_rgb224=target in semantic_ids(terminal))
    events.sort(key=lambda row: row['step_index'])
    events.append(stop)
    return dict(schema_version='lunar_route_language_contract/14.0', semantic_visibility_revision='14.0',
        physical_trajectory_id=name, goal=dict(id=target, name=OBJECT_NAMES[target]), events=events,
        intermediate_landmarks=landmarks, first_salient_goal=first_goal,
        deep_target_hint=bool(entry and fork and entry['step_index'] < fork['step_index'] < stop['step_index']),
        observation_stride=5, semantic_visibility_config_sha256=evidence['semantic_visibility_config_sha256'],
        semantic_scope='fixed route hints from actual salient current RGB-D at eligible frames; no coordinates',
        raw_detection_preserved=True, omitted_unrecognizable_landmarks=True)


def route_frame_pair(frame, contract):
    if not eligible(frame):
        return None
    intent, target = frame_intent(frame), frame['target_id']
    if intent == 'stop':
        if frame['machine_evidence'].get('terminal_accepted') is True and target in semantic_ids(frame):
            return '准备停止', (f'我看到了{OBJECT_NAMES[target]}', '我应该停下来')
        return None
    for event in contract['events']:
        if event['step_index'] != frame['step_index']:
            continue
        if event['type'] == 'first_fork' and frame['machine_evidence'].get('fork_visible_verified') is True:
            side = '左' if event['side'] == 'left' else '右'
            return '洞内搜索', ('我看到了第一个分叉', f'提示让我走{side}边')
        if event['type'] == 'landmark_continue' and event['id'] in semantic_ids(frame) and moving(frame):
            return '洞内搜索', (f'我看到了{event["name"]}', '按照提示我应该继续前进')
        if event['type'] == 'enter_cave' and moving(frame):
            if frame['machine_evidence'].get('entry_event') == 'entered_cave':
                return '进入洞穴', ('我看到了洞口', '按照提示我应该进去')
            return '洞内搜索', ('我已经进入洞穴', '按照提示我应该继续前进')
    if moving(frame) and frame['machine_evidence'].get('entry_event') == 'at_cave_entrance':
        return '到达洞口', ('我看到了洞口', '我应该进去')
    if moving(frame) and target in semantic_ids(frame):
        return '接近目标', (f'我看到了{OBJECT_NAMES[target]}', '我应该继续接近目标')
    return None


def select_frames(frames, contract, maximum=8):
    if not 1 <= maximum <= 8:
        raise ValueError('V14 task-relevant text selection is capped at eight frames')
    selected, entrance = {}, False
    for frame in frames:
        if not eligible(frame) or route_frame_pair(frame, contract) is None:
            continue
        events = []
        if frame['executed_action']['stop']:
            events.append('terminal_stop')
        for event in contract['events']:
            if event['step_index'] == frame['step_index']:
                tag = {'enter_cave':'entered_cave', 'first_fork':'verified_fork', 'landmark_continue':'route_landmark'}.get(event['type'])
                if tag:
                    events.append(tag)
        first_goal = contract['first_salient_goal']
        if first_goal is not None and first_goal['step_index'] == frame['step_index']:
            events.append('target_first_salient')
        if not entrance and frame['machine_evidence'].get('entry_event') == 'at_cave_entrance':
            events.append('entrance')
            entrance = True
        if events:
            selected[frame['sample_id']] = events
    priorities = {'terminal_stop':0,'verified_fork':1,'route_landmark':2,'entered_cave':3,
                  'entrance':4,'target_first_salient':5}
    if len(selected) > maximum:
        chosen = sorted(selected, key=lambda key:(min(priorities[e] for e in selected[key]), key))[:maximum]
        selected = {key:events for key,events in selected.items() if key in chosen}
    return selected


def derive_semantics(source, frames, files, layout, config):
    targets = json.loads((layout/'targets.json').read_text())['targets']
    observer = SemanticVisibility(targets, frames[0]['target_id'], config)
    hashes = {name:sha for name,_,sha in files}
    crossing, enriched, observations = None, [], []
    for frame in frames:
        payload = (source/frame['sensors_file']).read_bytes()
        if hashlib_digest(payload) != hashes[frame['sensors_file']]:
            raise ValueError('Semantic depth/mask source differs from its immutable raw audit')
        mapping = json.loads((source/frame['instances_file']).read_text())
        with np.load(io.BytesIO(payload), allow_pickle=False) as sensor:
            state = observer.observe(sensor['instance_id'], mapping, sensor['K256'], sensor['depth_m'], sensor['depth_valid'],
                frame['inputs']['observation']['timestamp_s'], frame['step_index'], terminal_stop=bool(frame['executed_action']['stop']))
        evidence = frame['machine_evidence']
        if evidence.get('entry_event') == 'entered_cave':
            crossing = dict(step_index=frame['step_index'], source_observation_id=frame['sample_id'])
        inside = (evidence.get('entry_event') == 'entered_cave'
                  or evidence.get('progress',{}).get('inside_cave') is True
                  or evidence.get('fork_visible_verified') is True)
        state['semantic_entry_support'] = dict(crossing, current_inside_verified=True) if crossing is not None and inside else None
        row = deepcopy(frame)
        row['machine_evidence'].update(state)
        enriched.append(row)
        observations.append(dict(step_index=frame['step_index'], source_observation_id=frame['sample_id'],
            sensors_file=frame['sensors_file'], sensors_sha256=hashes[frame['sensors_file']],
            instances_file=frame['instances_file'], instances_sha256=hashes[frame['instances_file']], fields=state))
    return enriched, dict(schema_version='lunar_semantic_visibility_enrichment/14.0',
        module_sha256=digest(Path(__file__).with_name('lunar_semantic_visibility_v14.py')),
        semantic_config_sha256=config_hash(config), raw_capture_manifest_sha256=object_digest(files),
        physical_trajectory_id=frames[0]['episode_id'], observations=observations,
        current_rgb_depth_controls_poses_unchanged=True, raw_detection_fields_unchanged=True)


def hashlib_digest(payload):
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def materialize(source, output, *, layout, semantic_config=None, fork_evidence_revision=None):
    source,output,layout=Path(source).resolve(strict=True),Path(output).absolute(),Path(layout).resolve(strict=True)
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Choose a fresh disjoint V14 derived output')
    metadata=json.loads((source/'metadata.json').read_text())
    if (metadata.get('instruction_style') == 'route_grounded' or not metadata['terminal_success']
            or json.loads((source/'status.json').read_text())['phase'] != 'complete'):
        raise ValueError('V14 materialization requires immutable completed original raw')
    if {name:digest(layout/name) for name in metadata['source_sha256']} != metadata['source_sha256']:
        raise ValueError('Explicit V14 scene differs from raw source hashes')
    config=checked_config(semantic_config)
    files,_=raw_manifest_files(source)
    original_frames=load_frames(source/'frame_records.jsonl')
    if any(frame['instruction'] != original_frames[0]['instruction'] or frame['inputs']['instruction'] != original_frames[0]['instruction']
           for frame in original_frames):
        raise ValueError('Original raw instruction is not fixed within the episode')
    frames=original_frames
    fork_audit=None
    if fork_evidence_revision:
        if fork_evidence_revision != 'v13':
            raise ValueError('Unknown optional raw fork enrichment revision')
        frames,fork_audit=recompute_fork_evidence(source,frames,files,layout)
    frames,semantic_audit=derive_semantics(source,frames,files,layout,config)
    contract=build_route_contract(frames)
    canonical=canonical_instruction(contract)
    output.mkdir(parents=True)
    for name,_,_ in files:
        if name in ('metadata.json','frame_records.jsonl'):
            continue
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(source/name,dest)
        except OSError:shutil.copy2(source/name,dest)
    audit=output/'raw_capture';audit.mkdir()
    for name in ('metadata.json','frame_records.jsonl'):shutil.copy2(source/name,audit/name)
    write_json(audit/'manifest.json',dict(files=files,source=str(source),manifest_sha256=object_digest(files)))
    write_json(output/'semantic_visibility_config.json',config)
    write_json(output/'semantic_visibility_enrichment.json',semantic_audit)
    if fork_audit is not None:write_json(output/'fork_evidence_enrichment.json',fork_audit)
    write_json(output/'route_contract.json',contract)
    binding=dict(schema_version='lunar_route_instruction_binding/14.0',physical_trajectory_id=metadata['episode_id'],
        original_instruction=original_frames[0]['instruction'],canonical_instruction=canonical,
        raw_capture_manifest_sha256=object_digest(files),route_contract_sha256=digest(output/'route_contract.json'),
        materializer_code_sha256=digest(Path(__file__)),semantic_visibility_config_sha256=config_hash(config),
        semantic_enrichment_sha256=digest(output/'semantic_visibility_enrichment.json'),
        semantic_module_sha256=digest(Path(__file__).with_name('lunar_semantic_visibility_v14.py')),
        observation_history_controls_and_poses_unchanged=True,raw_detection_fields_unchanged=True,
        changed_fields=['instruction','inputs.instruction','new semantic_* evidence','metadata derived provenance'])
    if fork_audit is not None:binding['fork_evidence_enrichment_sha256']=digest(output/'fork_evidence_enrichment.json')
    write_json(output/'instruction_binding.json',binding)
    for frame in frames:
        frame['instruction']=canonical;frame['inputs']['instruction']=canonical
    (output/'frame_records.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in frames))
    metadata.update(instruction_style='route_grounded',canonical_instruction=canonical,semantic_visibility_revision='14.0',
        semantic_visibility_config_sha256=config_hash(config),route_language_binding=dict(file='instruction_binding.json',
        sha256=digest(output/'instruction_binding.json'),original_raw_capture='raw_capture/manifest.json'))
    write_json(output/'metadata.json',metadata)
    validate_materialized(output,metadata,frames)
    return dict(passed=True,output=str(output),frames=len(frames),canonical_instruction=canonical,
                selected_frames=select_frames(frames,contract,8),route_contract=contract)


def validate_materialized(source, metadata=None, frames=None):
    source=Path(source).resolve(strict=True)
    metadata=metadata or json.loads((source/'metadata.json').read_text())
    frames=frames or load_frames(source/'frame_records.jsonl')
    original_meta=json.loads((source/'raw_capture/metadata.json').read_text())
    original=load_frames(source/'raw_capture/frame_records.jsonl')
    manifest=json.loads((source/'raw_capture/manifest.json').read_text())
    binding=json.loads((source/'instruction_binding.json').read_text())
    config=checked_config(json.loads((source/'semantic_visibility_config.json').read_text()))
    audit=json.loads((source/'semantic_visibility_enrichment.json').read_text())
    contract=json.loads((source/'route_contract.json').read_text())
    canonical=canonical_instruction(contract)
    from . import lunar_semantic_visibility_v14 as semantic_module
    compatible = (semantic_module.legacy_replay_module_hashes(config)
                  if hasattr(semantic_module,'legacy_replay_module_hashes') else ())
    accepted_semantic_modules={digest(Path(__file__).with_name('lunar_semantic_visibility_v14.py')), *compatible}
    if (metadata.get('semantic_visibility_revision') != '14.0' or len(frames) != len(original)
            or contract != build_route_contract(frames) or binding['route_contract_sha256'] != digest(source/'route_contract.json')
            or binding['canonical_instruction'] != canonical or binding['physical_trajectory_id'] != original_meta['episode_id']
            or binding['raw_capture_manifest_sha256'] != object_digest(manifest['files'])
            or manifest['manifest_sha256'] != object_digest(manifest['files'])
            or binding['semantic_visibility_config_sha256'] != config_hash(config)
            or audit['semantic_config_sha256'] != config_hash(config)
            or binding['semantic_enrichment_sha256'] != digest(source/'semantic_visibility_enrichment.json')
            or audit['raw_capture_manifest_sha256'] != manifest['manifest_sha256']
            or audit['module_sha256'] != binding['semantic_module_sha256']
            or audit['module_sha256'] not in accepted_semantic_modules
            or len(audit['observations']) != len(frames)):
        raise ValueError('V14 semantic route/config/raw provenance mismatch')
    expected_meta=deepcopy(original_meta)
    expected_meta.update(instruction_style='route_grounded',canonical_instruction=canonical,semantic_visibility_revision='14.0',
        semantic_visibility_config_sha256=config_hash(config),route_language_binding=dict(file='instruction_binding.json',
        sha256=digest(source/'instruction_binding.json'),original_raw_capture='raw_capture/manifest.json'))
    if metadata != expected_meta:
        raise ValueError('V14 materialization changed unrelated raw metadata')
    hashes={name:sha for name,_,sha in manifest['files']}
    if len(hashes) != len(manifest['files']):raise ValueError('Duplicate raw source files')
    for name,size,sha in manifest['files']:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('External raw source path')
        path=source/('raw_capture/'+name if name in ('metadata.json','frame_records.jsonl') else name)
        if not path.is_file() or path.stat().st_size != size or digest(path) != sha:
            raise ValueError('V14 immutable source changed: '+name)
    fork=None
    if (source/'fork_evidence_enrichment.json').exists():
        fork=json.loads((source/'fork_evidence_enrichment.json').read_text())
        if (binding.get('fork_evidence_enrichment_sha256') != digest(source/'fork_evidence_enrichment.json')
                or fork['module_sha256'] != digest(Path(__file__).with_name('lunar_fork_evidence_v13.py'))
                or fork['raw_capture_manifest_sha256'] != manifest['manifest_sha256']
                or len(fork['observations']) != len(frames)):
            raise ValueError('V14 optional fork evidence is not bound to raw')
    for index,(raw,frame,observation) in enumerate(zip(original,frames,audit['observations'])):
        if (observation['step_index'] != index or observation['source_observation_id'] != raw['sample_id']
                or observation['sensors_file'] != raw['sensors_file'] or observation['sensors_sha256'] != hashes[raw['sensors_file']]
                or observation['instances_file'] != raw['instances_file'] or observation['instances_sha256'] != hashes[raw['instances_file']]
                or any(not key.startswith('semantic_') for key in observation['fields'])):
            raise ValueError('V14 semantic evidence is not bound to actual current mask/depth')
        expected=deepcopy(raw);expected['instruction']=canonical;expected['inputs']['instruction']=canonical
        if fork is not None:
            old=fork['observations'][index]
            if old['sensors_sha256'] != hashes[raw['sensors_file']] or old['source_observation_id'] != raw['sample_id']:
                raise ValueError('V14 fork input frame mismatch')
            expected['machine_evidence'].update(old['after'])
        expected['machine_evidence'].update(observation['fields'])
        if frame != expected:raise ValueError('V14 modified raw detection, observations, history or controls')
    return contract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layout',type=Path,required=True)
    parser.add_argument('--semantic-config',type=Path)
    parser.add_argument('--fork-evidence-revision',choices=['v13'])
    args=parser.parse_args()
    config=json.loads(args.semantic_config.read_text()) if args.semantic_config else None
    print(json.dumps(materialize(args.source,args.output,layout=args.layout,semantic_config=config,
        fork_evidence_revision=args.fork_evidence_revision),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
