"""Frozen raw-only integrity checks, without importing annotation/provider code.

The real-observation/control checks are extracted unchanged from the reviewed
V11 validator. Language/derived-trajectory branches are excluded by design.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from cave.lunar_raw_physical_v16 import sha256,write_json
SOURCE_VALIDATOR_SHA256='3a23a76dbdf0c1a873f0587bbd82613528d7933461c13f4588029a2f5cd7722c'

def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def safe_path(source,value):
    path=(Path(source)/value).resolve(strict=True)
    if not path.is_relative_to(Path(source).resolve(strict=True)):raise ValueError('Source path escapes trajectory')
    return path
def teacher_intent(action):
    from cave.lunar_action_intent_v12 import action_intent
    return action_intent(action['v_mps'],action['omega_radps'],action['stop'],motion_controller=action.get('motion_controller','segment'))

def wrapped(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def target_label_paths(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [p for v in value.values() for p in target_label_paths(v)]
    if isinstance(value, list):
        return [p for v in value for p in target_label_paths(v)]
    return []

def validate_raw(episode, layout=None):
    episode = Path(episode).resolve(strict=True)
    errors = []
    warnings = []
    checks = 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition and len(errors) < 100:
            errors.append(message)
    try:
        metadata = json.loads((episode / 'metadata.json').read_text())
        status = json.loads((episode / 'status.json').read_text())
        with np.load(episode / 'supervision.npz', allow_pickle=False) as z:
            supervision = {k: z[k] for k in z.files}
        with np.load(episode / 'teacher_only.npz', allow_pickle=False) as z:
            teacher = {k: z[k] for k in z.files}
        frames = read_jsonl(episode / 'frame_records.jsonl')
        transitions = read_jsonl(episode / 'transitions.jsonl')
        instruction_style = metadata.get('instruction_style', 'legacy')
        route_contract = None
        if instruction_style == 'route_grounded':
            raise ValueError('Derived language trajectories are not raw inputs')
        check(metadata['domain'] == 'isaac_continuous', 'Wrong annotation domain')
        check(metadata['terminal_success'] is True and status['phase'] == 'complete', 'Episode is partial or terminal acceptance failed')
        check(metadata['training_ready'] is False and metadata['native_action_labels_provided'] is False, 'Incorrect Native/training-ready claim')
        check(metadata['capture_stride'] == 1 and metadata['all_action_states_have_current_rgbd'] is True, 'Dense observations required; capture_stride must be one')
        check(metadata['source_assets_unchanged'] is True, 'Source scene changed during capture')
        vw = supervision['vw_si']
        n = len(vw)
        dt = supervision['dt_s']
        stop = supervision['stop']
        check(n > 0 and vw.shape == (n, 2) and (vw.dtype == np.float32), 'vw_si must be nonempty float32[N,2]')
        check(np.isfinite(vw).all() and (vw[:, 0] >= -1e-07).all() and (vw[:, 0] <= 0.400001).all() and (np.abs(vw[:, 1]) <= 0.800001).all(), 'Action finite/bounds failure')
        check(dt.shape == (n,) and dt.dtype == np.float32 and np.allclose(dt, 0.2, atol=1e-08), 'dt_s must be float32[N] equal .2')
        check(stop.shape == (n,) and stop.dtype == np.uint8 and (np.count_nonzero(stop) == 1) and (stop[-1] == 1) and (set(stop.tolist()) <= {0, 1}), 'Exactly one terminal STOP is required')
        check(np.allclose(vw[-1], 0.0, atol=1e-08), 'STOP control must be zero')
        times = np.arange(n) * 0.2
        check(supervision['observation_time_s'].shape == (n,) and np.allclose(supervision['observation_time_s'], times, atol=1e-09), 'Observation times lost control alignment')
        check(supervision['phase'].shape == (n,) and supervision['phase'].dtype == np.uint8, 'Phase array has wrong contract')
        future = supervision['future_vw_si']
        valid = supervision['horizon_valid']
        check(future.shape == (n, 8, 2) and future.dtype == np.float32, 'future_vw_si must be float32[N,8,2]')
        check(valid.shape == (n, 8) and valid.dtype == np.bool_, 'horizon_valid must be bool[N,8]')
        for horizon in range(8):
            wanted = np.arange(n) + horizon < n
            check(np.array_equal(valid[:, horizon], wanted), f'Horizon validity mismatch at {horizon}')
            check(np.array_equal(future[wanted, horizon], vw[horizon:]), f'Future action mismatch at {horizon}')
            check(np.all(future[~wanted, horizon] == 0), f'Invalid horizon slots are not zero padded at {horizon}')
        positions = teacher['executed_positions']
        yaws = teacher['executed_yaw_rad']
        check(positions.shape == (n + 1, 3) and positions.dtype == np.float64 and np.isfinite(positions).all(), 'Executed position must be finite float64[N+1,3]')
        check(yaws.shape == (n + 1,) and yaws.dtype == np.float64 and np.isfinite(yaws).all(), 'Executed yaw must be finite float64[N+1]')
        check(np.allclose(teacher['timestamp_s'], np.arange(n + 1) * 0.2, atol=1e-09), 'N+1 pose timestamps mismatch')
        v, w = (vw[:, 0].astype(float), vw[:, 1].astype(float))
        angle = yaws[:-1]
        delta = w * 0.2
        moving = np.abs(w) > 1e-10
        dx = v * 0.2 * np.cos(angle)
        dy = v * 0.2 * np.sin(angle)
        dx[moving] = v[moving] / w[moving] * (np.sin(angle[moving] + delta[moving]) - np.sin(angle[moving]))
        dy[moving] = v[moving] / w[moving] * (-np.cos(angle[moving] + delta[moving]) + np.cos(angle[moving]))
        check(np.allclose(np.diff(positions[:, :2], axis=0), np.c_[dx, dy], atol=2e-06, rtol=0), 'Positions are not exact dt=.2 unicycle integration of controls')
        check(np.all(np.abs(wrapped(np.diff(yaws) - delta)) <= 2e-06), 'Yaw is not integrated from executed omega')
        if metadata.get('motion_controller') == 'smooth_v13':
            check(metadata.get('runtime_collision_mode') == 'actual_geometry', 'V13 runtime collision mode is not declared')
            wall = metadata.get('body_wall_clearance') or {}
            check(wall.get('passed') is True and wall.get('required_gap_m') == 1.0, 'V13 lacks its one-metre body-wall preflight')
            check(wall.get('expanded_air_envelope_gap_m', 0.0) >= 1.0, 'V13 wall audit did not cover the required body gap')
            check(metadata.get('canonical_stop_radius_m') == 1.0, 'V13 canonical stop radius must be 1 metre')
            check(np.linalg.norm(positions[-1, :2] - teacher['original_approach_xy_m']) <= 1.000001, 'Terminal pose is outside canonical one-metre approach region')
            check(np.linalg.norm(teacher['planned_stop_xy_m'] - teacher['original_approach_xy_m']) <= 1.000001, 'Planned arrival is outside canonical approach region')
            check(np.linalg.norm(positions[-1, :2] - teacher['planned_stop_xy_m']) <= 0.050001, 'Executed arrival differs from its planned stopping pose')
            check(not np.any((vw[:, 0] <= 1e-08) & (np.abs(vw[:, 1]) > 1e-06)), 'V13 contains a stationary turn')
            check(np.max(np.abs(vw[:, 1])) <= 0.320001, 'V13 angular speed limit violated')
            acceleration = np.diff(np.vstack([[0.0, 0.0], vw]), axis=0) / 0.2
            check(np.max(np.abs(acceleration[:, 0])) <= 0.25001, 'V13 linear acceleration limit violated')
            check(np.max(np.abs(acceleration[:, 1])) <= 0.24001, 'V13 angular acceleration limit violated')
        else:
            check(np.linalg.norm(positions[-1, :2] - teacher['original_approach_xy_m']) <= 0.050001, 'Terminal pose is not within 5cm of original approach')
        check(metadata['action_count'] == n and metadata['pose_count'] == n + 1 and (metadata['captured_frame_count'] == n), 'Metadata counts disagree with dense episode')
        check(len(frames) == n and len(transitions) == n, 'Frame/control lengths differ')
        check([f['step_index'] for f in frames] == list(range(n)), 'Dense frame steps missing or unordered')
        check(len({f['sample_id'] for f in frames}) == n, 'Duplicate sample IDs')
        check(len({f['instruction'] for f in frames}) == 1, 'Instruction changes within an episode')
        by_id = {f['sample_id']: f for f in frames}
        target = metadata['target_id']
        target_prim = '/World/SearchTargets/' + target
        landmark_replay = None
        semantic_replay = None
        semantic_crossing = None
        if metadata.get('semantic_visibility_revision') == '14.0' and layout is None:
            raise ValueError('V14 semantic replay requires its hash-bound layout')
        if layout:
            layout = Path(layout).resolve(strict=True)
            target_rows = json.loads((layout / 'targets.json').read_text())['targets']
            target_row = next((t for t in target_rows if t['id'] == target))
            target_prim = target_row['prim_path']
            check(np.allclose(teacher['original_approach_xy_m'], target_row['approach_xy_m'], atol=1e-09), 'Teacher goal differs from target approach point')
            check(not np.allclose(target_row['xyz_m'][:2], teacher['original_approach_xy_m']), 'Collision object center incorrectly used as navigation goal')
            for name, digest in metadata['source_sha256'].items():
                check(sha256(safe_path(layout, name)) == digest, 'Layout source hash mismatch: ' + name)
            if 'visible_objects' in frames[0]['machine_evidence']:
                from cave.lunar_landmark_evidence_v13 import LandmarkEvidence
                landmark_replay = LandmarkEvidence(target_rows, target, frames[0]['machine_evidence']['landmark_visibility_threshold_pixels'])
            if metadata.get('semantic_visibility_revision') == '14.0':
                from cave.lunar_semantic_visibility_v14 import SemanticVisibility
                semantic_replay = SemanticVisibility(target_rows, target, json.loads((episode / 'semantic_visibility_config.json').read_text()))
        for i, frame in enumerate(frames):
            try:
                check(frame['episode_id'] == metadata['episode_id'] and frame['target_id'] == target and (frame['domain'] == 'isaac_continuous'), f'{i}: Frame identity/domain mismatch')
                check(frame['source_layout_seed'] == metadata['source_layout_seed'], f'{i}: Layout seed mismatch')
                action = frame['executed_action']
                evidence = frame['machine_evidence']
                inputs = frame['inputs']
                check(np.allclose([action['v_mps'], action['omega_radps']], vw[i], atol=1e-07) and action['stop'] == int(stop[i]), f'{i}: Text-frame action differs from supervision')
                check(evidence['expected_action_intent'] == teacher_intent(action), f'{i}: Evidence intent differs from actual command')
                check(set(inputs) == {'instruction', 'observation', 'history'} and inputs['instruction'] == frame['instruction'], f'{i}: Inputs contain privileged fields or changed instruction')
                observation = inputs['observation']
                check(set(observation) == {'observation_id', 'rgb', 'depth', 'timestamp_s'}, f'{i}: Observation contains privileged fields')
                check(observation['observation_id'] == frame['sample_id'] and observation['rgb'] == frame['rgb_224'] and (observation['depth'] == frame['sensors_file']) and (abs(observation['timestamp_s'] - i * 0.2) < 1e-09), f'{i}: Observation identity/time binding mismatch')
                rgb = np.asarray(Image.open(safe_path(episode, frame['rgb_224'])))
                rgb256 = np.asarray(Image.open(safe_path(episode, frame['rgb_256'])))
                check(rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8, f'{i}: RGB224 contract mismatch')
                check(rgb256.shape == (256, 256, 3) and rgb256.dtype == np.uint8, f'{i}: RGB256 source contract mismatch')
                with np.load(safe_path(episode, frame['sensors_file']), allow_pickle=False) as sensor:
                    depth = sensor['depth_m']
                    depth_valid = sensor['depth_valid']
                    instance = sensor['instance_id']
                    check(depth.shape == (256, 256) and depth.dtype == np.float32 and (depth_valid.dtype == np.bool_) and (depth_valid.shape == depth.shape), f'{i}: Depth256 metres/valid contract mismatch')
                    check(np.array_equal(depth_valid, np.isfinite(depth) & (depth > 0)) and depth_valid.any(), f'{i}: Depth validity does not describe finite positive axial metres')
                    K = np.array([[128.0, 0, 128.0], [0, 128.0, 128.0], [0, 0, 1.0]])
                    check(np.allclose(sensor['K256'], K, atol=1e-06) and np.allclose(sensor['K'], K, atol=1e-06), f'{i}: Camera K256 does not match HFOV90')
                    K224 = K.copy()
                    K224[:2] *= 224 / 256
                    check(np.allclose(sensor['K224'], K224, atol=1e-06), f'{i}: K224 resize mismatch')
                    T = sensor['world_from_camera']
                    R = T[:3, :3]
                    check(T.shape == (4, 4) and np.isfinite(T).all() and np.allclose(T[3], [0, 0, 0, 1]) and np.allclose(R.T @ R, np.eye(3), atol=1e-05) and (abs(np.linalg.det(R) - 1) < 1e-05), f'{i}: Camera transform is not SE3')
                    check(np.allclose(T[:3, 3], positions[i] + [0, 0, 1.5], atol=1e-05) and np.allclose(R[:, 2], [np.cos(yaws[i]), np.sin(yaws[i]), 0], atol=1e-05), f'{i}: Camera pose differs from executed vehicle state')
                    check(np.allclose(sensor['vehicle_xy_yaw'], [*positions[i, :2], yaws[i]], atol=1e-07) and abs(float(sensor['observation_time_s']) - times[i]) < 1e-09, f'{i}: Sensor state/time mismatch')
                    check(instance.shape == (256, 256) and instance.dtype.kind in 'ui', f'{i}: Instance image mismatch')
                    mapping = json.loads(safe_path(episode, frame['instances_file']).read_text())['idToLabels']
                    if landmark_replay is not None:
                        observed = landmark_replay.observe(instance, mapping, sensor['K256'], times[i])
                        check(all((evidence.get(key) == value for key, value in observed.items())), f'{i}: Landmark evidence differs from actual current masks or strict past history')
                    if semantic_replay is not None:
                        current = semantic_replay.observe(instance, mapping, sensor['K256'], depth, depth_valid, float(times[i]), i, terminal_stop=bool(stop[i]))
                        if evidence.get('entry_event') == 'entered_cave':
                            semantic_crossing = dict(step_index=i, source_observation_id=frame['sample_id'])
                        inside = evidence.get('entry_event') == 'entered_cave' or evidence.get('progress', {}).get('inside_cave') is True or evidence.get('fork_visible_verified') is True
                        current['semantic_entry_support'] = dict(semantic_crossing, current_inside_verified=True) if semantic_crossing is not None and inside else None
                        check({key: value for key, value in evidence.items() if key.startswith('semantic_')} == current, f'{i}: Semantic recognition/eligible cue history differs from actual current mask/depth')
                    matched = [int(k) for k, value in mapping.items() if any((p == target_prim or p.startswith(target_prim + '/') for p in target_label_paths(value)))]
                    if not matched:
                        matched = [int(k) for k, value in mapping.items() if value == target]
                    mask = np.isin(instance, matched)
                    pixels = int(mask.sum())
                    visible = pixels >= 12
                    check(evidence['target_visible_pixels'] == pixels and evidence['target_visible'] == visible and (evidence['visibility_threshold_pixels'] == 12), f'{i}: Target visibility not established by current instance mask')
                    if pixels:
                        vv, uu = np.nonzero(mask)
                        box = [int(uu.min()), int(vv.min()), int(uu.max() + 1), int(vv.max() + 1)]
                        bearing = float(np.degrees(np.arctan2(128 - (box[0] + box[2]) * 0.5, 128)))
                        check(evidence['target_bbox_xyxy_256'] == box, f'{i}: Target mask bbox mismatch')
                        if visible:
                            expected_bearing = 'left' if bearing > 10 else 'right' if bearing < -10 else 'ahead'
                            check(abs(evidence['bearing_deg'] - bearing) < 1e-06 and evidence['target_bearing'] == expected_bearing, f'{i}: Target bearing is not camera-relative mask direction')
                    if not visible:
                        check(evidence['target_bearing'] == 'unknown' and evidence['bearing_deg'] is None, f'{i}: Invisible target has leaked direction')
                check('axial Z in metres' in frame['camera']['depth_convention'] and 'world Z-up' in frame['camera']['coordinate_convention'], f'{i}: Missing depth/coordinate convention')
                history = inputs['history']
                seen_history = []
                for past in history:
                    past_frame = by_id.get(past['observation_id'])
                    seen_history.append(past['observation_id'])
                    check(past_frame is not None and past_frame['step_index'] < i and (past['timestamp_s'] < observation['timestamp_s']), f'{i}: Noncausal/cross-episode history')
                    if past_frame:
                        check(past == past_frame['inputs']['observation'], f'{i}: History reference differs from actual previous observation')
                check(len(history) <= 32 and len(set(seen_history)) == len(history) and ([h['timestamp_s'] for h in history] == sorted((h['timestamp_s'] for h in history))), f'{i}: History duplicated/unordered/oversized')
                check(set(evidence['source_observation_ids']) <= {frame['sample_id'], *seen_history}, f'{i}: Evidence references a future/foreign observation')
                entered = i > 0 and positions[i - 1, 0] < 0 <= positions[i, 0] and (abs(positions[i, 1]) <= 14)
                at_entry = abs(positions[i, 0]) <= 1 and abs(positions[i, 1]) <= 14
                check(evidence['entry_event'] == ('entered_cave' if entered else 'at_cave_entrance' if at_entry else 'none'), f'{i}: Entrance event does not match actual consecutive crossing')
                transition = transitions[i]
                check(transition['step_index'] == i and transition['current_observation_id'] == frame['sample_id'] and (not transition['current_frame_missing']), f'{i}: Transition observation alignment mismatch')
                check(transition['collision_check_passed'] is True and np.allclose(transition['action_vw_si'], vw[i], atol=1e-07), f'{i}: Transition collision/action invalid')
                check(np.allclose(transition['teacher_pose_xy_yaw'], [*positions[i, :2], yaws[i]], atol=1e-07) and np.allclose(transition['teacher_next_pose_xy_yaw'], [*positions[i + 1, :2], yaws[i + 1]], atol=1e-07), f'{i}: Transition N+1 state mismatch')
                if stop[i]:
                    check(evidence['terminal_accepted'] is True and evidence['target_visible'] is True, f'{i}: STOP has no terminal visible evidence')
            except (KeyError, ValueError, TypeError, OSError, IndexError) as exc:
                check(False, f'{i}: Malformed frame: {type(exc).__name__}: {exc}')
        warnings.append('Raw-only integrity validation; no language stage')
    except (KeyError, ValueError, TypeError, OSError, IndexError, StopIteration) as exc:
        check(False, f'Incomplete/malformed episode: {type(exc).__name__}: {exc}')
    return dict(schema_version='lunar_raw_validation/16.0', passed=not errors, checks=checks, errors=errors, warnings=warnings, annotation_domain='isaac_continuous', training_ready=False, current_129_adapter_implemented=False, current_129_text_loss_supported=False, scope='offline real-observation/teacher integrity; no learned-policy or tyre-dynamics evaluation')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--episode',type=Path,required=True);parser.add_argument('--layout',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=validate_raw(args.episode,args.layout);write_json(args.output,report)
    print(json.dumps(dict(passed=report['passed'],checks=report['checks'],errors=report['errors'])),flush=True)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
