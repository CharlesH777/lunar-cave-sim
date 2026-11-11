"""Capture an actually integrated kinematic teacher with synchronized Isaac RGB-D.

The controller is an oracle using the preserved map. It is not a learned-policy
rollout or a wheel/soil simulation. Every accepted command advances pose once
by an exact dt=.2s unicycle integration; only the camera follows that pose.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
TARGETS = ['astronaut', 'lunar_rover', 'lunar_base', 'chinese_flag']
PHASES = {'align_path_segment': 0, 'follow_path_segment': 1,
          'align_target_at_goal': 2, 'request_terminal_stop': 3}


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'): return jsonable(value.tolist())
    if hasattr(value, 'item'): return jsonable(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def append(stream, value):
    stream.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()


def experience_path(argument):
    """Use an explicit portable deployment configuration, never a local path."""
    if argument: return str(Path(argument).expanduser().resolve(strict=True))
    configured = os.environ.get('ISAAC_EXPERIENCE')
    if configured: return str(Path(configured).expanduser().resolve(strict=True))
    lab = os.environ.get('ISAACLAB_PATH')
    if lab:
        candidate = Path(lab)/'apps/isaaclab.python.headless.rendering.kit'
        if candidate.is_file(): return str(candidate.resolve())
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/lunar_branch_v10/scene_01_equal')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--target', choices=TARGETS, default='chinese_flag')
    parser.add_argument('--episode-id', help='Stable unique physical trajectory ID; default output directory name')
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--capture-stride', type=int, default=1)
    parser.add_argument('--motion-controller', choices=['segment', 'smooth', 'smooth_v13'], default='segment')
    parser.add_argument('--history-frames', type=int, default=8)
    parser.add_argument('--experience', type=Path)
    parser.add_argument('--ext-folder', type=Path, action='append', default=[])
    parser.add_argument('--cpu-validate', action='store_true', help='Geometry/integration only, no rendering and no dataset claim')
    parser.add_argument('--self-test', action='store_true', help='Pure CPU units/integration/horizon regression tests')
    args = parser.parse_args(argv); sys.argv = [sys.argv[0]]
    if args.episode_id is not None and (not args.episode_id.strip() or '\n' in args.episode_id or '\r' in args.episode_id):
        parser.error('--episode-id must be a nonempty single-line identity')
    if args.max_steps < 1 or args.capture_stride < 1 or args.history_frames < 0: parser.error('Invalid step/stride/history count')
    if args.self_test:
        from cave.lunar_trajectory_v11 import self_test
        print(json.dumps(self_test()), flush=True); return 0
    if args.output is None: parser.error('--output is required for capture or CPU validation')
    source = args.source.expanduser().resolve(strict=True); output = args.output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); app = None; sensors = None; world = None
    controls = []; stop_flags = []; phase_ids = []; poses = []; frame_count = 0
    completed = False; reason = 'not_started'; error = None; planning = None; episode_id = args.episode_id or output.name
    collision_aggregate = {'checks': 0, 'maximum_abs_pitch_deg': 0., 'maximum_abs_roll_deg': 0.,
                           'minimum_belly_clearance_m': None, 'minimum_circle_separation_m': None}
    render_verification = {'requested_antialiasing': 'FXAA (2)', 'captures_checked': 0,
        'capture_readbacks_match_request': True, 'internal_rtx_resolution_measured': False,
        'scope': 'Carb setting readbacks and RGB product size; array size alone does not measure internal RTX resolution'}

    def status(phase, **extra):
        record = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.monotonic()-started, 3), domain='isaac_continuous',
            training_ready=False, executed_controls=len(controls), captured_frames=frame_count, **extra)
        write(output/'status.json', record); print(json.dumps(jsonable(record), ensure_ascii=False), flush=True)

    try:
        if args.cpu_validate:
            from cave.lunar_trajectory_v11 import TerrainWorld, simulate_plan, self_test
            status('cpu_geometry_validation')
            world = TerrainWorld(source, runtime_collision_mode='actual_geometry' if args.motion_controller == 'smooth_v13' else 'planning_grid_and_geometry')
            if args.motion_controller != 'segment':
                if args.motion_controller == 'smooth_v13':
                    from cave.lunar_smooth_teacher_v13 import precompute_smooth_teacher
                else:
                    from cave.lunar_smooth_teacher_v12 import precompute_smooth_teacher
                teacher, plan = precompute_smooth_teacher(world, args.target, max_steps=args.max_steps)
                result = dict(passed=True, reason='swept_smooth_kinematics_checked', planning=plan,
                              executed_poses=teacher.poses, controls=teacher.controls)
            else:
                result = simulate_plan(world, args.target, args.max_steps)
            import numpy as np
            arrays = {k: result.pop(k) for k in ['executed_poses', 'controls'] if k in result}
            if arrays: np.savez_compressed(output/'cpu_kinematics_only.npz', **arrays)
            result.update(unit_regressions=self_test(), source_assets_unchanged=world.verify_sources(),
                          domain='isaac_continuous', actual_rgbd_captured=False, training_ready=False)
            write(output/'cpu_validation.json', result)
            status('cpu_validation_complete', passed=result['passed'], reason=result['reason'])
            return 0 if result['passed'] and result['source_assets_unchanged'] else 1

        status('starting_isaac_continuous_capture')
        from isaacsim import SimulationApp
        extra = ['--/app/extensions/registryEnabled=false', '--/renderer/multiGpu/enabled=false',
                 '--/rtx/indirectDiffuse/enabled=true', '--/rtx/ambientOcclusion/enabled=true',
                 '--/rtx-transient/post/aa/limitedOps=false', '--/rtx/post/aa/op=2',
                 '--/rtx/post/scaling/staticRatio=1.0', '--/rtx/post/dlss/enabled=false',
                 '--/exts/omni.replicator.core/Orchestrator/enabled=true']
        ext_folders = list(args.ext_folder)
        ext_folders += [Path(p) for p in os.environ.get('ISAAC_EXT_FOLDER', '').split(os.pathsep) if p]
        for folder in ext_folders: extra += ['--ext-folder', str(folder.expanduser().resolve(strict=True))]
        app_config = dict(headless=True, renderer='RayTracedLighting', width=256, height=256,
            anti_aliasing=2, multi_gpu=False, active_gpu=0, physics_gpu=0, sync_loads=True, extra_args=extra)
        experience = experience_path(args.experience)
        app = SimulationApp(app_config, **({'experience': experience} if experience else {}))
        import numpy as np
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom, UsdLux
        from PIL import Image
        from cave.scene import camera, set_camera, sphere_light
        from cave.sensors import CaveSensors
        from cave.lunar_trajectory_v11 import TerrainWorld, SegmentTeacher, make_supervision
        from cave.lunar_action_intent_v12 import action_intent

        np.random.seed(args.seed)
        world = TerrainWorld(source, runtime_collision_mode='actual_geometry' if args.motion_controller == 'smooth_v13' else 'planning_grid_and_geometry'); target = world.targets[args.target]
        if args.motion_controller != 'segment':
            status('precomputing_collision_checked_smooth_motion')
            if args.motion_controller == 'smooth_v13':
                from cave.lunar_smooth_teacher_v13 import precompute_smooth_teacher
            else:
                from cave.lunar_smooth_teacher_v12 import precompute_smooth_teacher
            teacher, planning = precompute_smooth_teacher(world, args.target, max_steps=max(5000, args.max_steps))
            path = teacher.path
        else:
            path, planning = world.plan(args.target)
            teacher = SegmentTeacher(path, np.asarray(target['xyz_m'][:2], float))
        fork_evidence = None
        if args.motion_controller != 'segment':
            if args.motion_controller == 'smooth_v13':
                from cave.lunar_fork_evidence_v13 import ForkEvidence
            else:
                from cave.lunar_fork_evidence_v12 import ForkEvidence
            fork_evidence = ForkEvidence(world, args.target, path)
        landmarks = None
        if args.motion_controller == 'smooth_v13':
            from cave.lunar_landmark_evidence_v13 import LandmarkEvidence
            landmarks = LandmarkEvidence(world.targets, args.target)
        write(output/'planning_teacher_only.json', planning)
        pose = np.r_[world.start_xy, world.start_yaw]; poses.append(pose.copy())
        safe, initial_checks = world.pose_checks([pose])
        if not safe.all(): raise ValueError('Original task_spawn is not body-safe: '+str(initial_checks))
        context = omni.usd.get_context()
        if not context.open_stage(str(source/'cave.usdc')): raise RuntimeError('Could not open source stage')
        stage = context.get_stage(); stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath) and value.resolvedPath: attr.Set(Sdf.AssetPath(value.resolvedPath))
        settings = carb.settings.get_settings()
        native_values = {'/rtx-transient/post/aa/limitedOps': False, '/rtx/post/aa/op': 2,
            '/rtx/post/dlss/enabled': False, '/rtx/post/scaling/staticRatio': 1.,
            '/rtx-transient/dlssg/enabled': False, '/rtx/post/motionblur/enabled': False}

        def force_native_settings():
            # NVIDIA Replicator's set_render_rtx_realtime(FXAA) first disables
            # limitedOps; without that gate RTX may force AA back to DLSS/DLAA.
            for key, value in native_values.items(): settings.set(key, value)

        def render_readback():
            actual = {key: settings.get(key) for key in native_values}
            actual['render_mode'] = settings.get('/rtx/rendermode')
            actual['matches_requested_values'] = all(actual[key] == value for key, value in native_values.items())
            return actual

        force_native_settings(); render_verification['requested_settings'] = native_values
        settings.set('/rtx/sceneDb/ambientLightIntensity', 0.)
        prefix = '/World/ContinuousTeacherV11'; UsdGeom.Xform.Define(stage, prefix)
        sun = UsdLux.DistantLight.Define(stage, prefix+'/Sun'); sun.CreateIntensityAttr(3500.); sun.CreateAngleAttr(.53)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(24., -24., -25.))
        dome = UsdLux.DomeLight.Define(stage, prefix+'/GentleFill')
        dome.CreateIntensityAttr(45.); dome.CreateColorAttr(Gf.Vec3f(.70, .75, .80))
        cam = camera(stage, prefix+'/Camera', (0, 0, 1.5), (1, 0, 1.5), focal=18.)
        cam.CreateHorizontalApertureAttr(36.); cam.CreateVerticalApertureAttr(36.)
        head = sphere_light(stage, prefix+'/Headlamp', (0, 0, 1.5), (.94, .97, 1.), 24000., .28)
        head.CreateNormalizeAttr(True); head.CreateExposureAttr(6.)
        lights = []
        for side in [-1, 1]:
            light = sphere_light(stage, prefix+'/WorkLight'+('Left' if side > 0 else 'Right'),
                                 (0, 0, 1.2), (.94, .97, 1.), 8000., .45)
            light.CreateNormalizeAttr(True); light.CreateExposureAttr(6.); lights.append((side, light))
        rep.orchestrator.set_capture_on_play(False)
        sensors = CaveSensors(stage, str(cam.GetPath()), (256, 256))
        for _ in range(25): app.update()
        force_native_settings(); app.update(); force_native_settings()
        render_verification['after_warmup'] = render_readback()
        write(output/'renderer_settings.json', render_verification)
        (output/'frames').mkdir(); history = []; last_status = time.monotonic(); reason = 'max_steps'
        has_entered_cave = False
        instruction = target.get('instruction_zh') or '找到'+target['label_zh']+'并靠近。'
        with (output/'transitions.jsonl').open('w') as transitions, (output/'frame_records.jsonl').open('w') as records:
            for step in range(args.max_steps):
                if (output/'REQUEST_STOP').exists(): reason = 'REQUEST_STOP'; break
                control, request_stop, phase = teacher.command(pose)
                accepted, next_pose, checks = world.safe_motion(pose, control)
                if not accepted:
                    write(output/'rejected_motion.json', dict(step=step, pose=pose, proposed_vw=control, checks=checks))
                    reason = 'unsafe_integrated_motion'; break
                for key, src in [('maximum_abs_pitch_deg', 'max_abs_pitch_deg'), ('maximum_abs_roll_deg', 'max_abs_roll_deg')]:
                    collision_aggregate[key] = max(collision_aggregate[key], checks[src])
                for key in ['minimum_belly_clearance_m', 'minimum_circle_separation_m']:
                    old = collision_aggregate[key]; collision_aggregate[key] = checks[key] if old is None else min(old, checks[key])
                collision_aggregate['checks'] += 1
                capture_now = step % args.capture_stride == 0 or request_stop or step == args.max_steps-1
                ground = float(world.floor(pose[:2])[0]); eye = np.r_[pose[:2], ground+1.5]
                goal_error = float(np.linalg.norm(pose[:2]-teacher.goal))
                canonical_error = float(np.linalg.norm(pose[:2]-np.asarray(target['approach_xy_m'])))
                entered = bool(len(poses) > 1 and poses[-2][0] < 0. <= pose[0] and abs(pose[1]) <= 14.)
                has_entered_cave = has_entered_cave or entered
                at_entry = bool(abs(pose[0]) <= 1. and abs(pose[1]) <= 14.)
                observation = None; visible = None; evidence = None; stop = 0; record = None
                if capture_now:
                    forward = np.array([np.cos(pose[2]), np.sin(pose[2]), 0.]); left = np.array([-forward[1], forward[0], 0.])
                    set_camera(cam, eye, eye+forward*10.)
                    head.GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float, eye+forward*.12)))
                    for side, light in lights:
                        light.GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float, eye+forward*.22+left*(side*.45)+[0., 0., -.25])))
                    for _ in range(2): app.update()
                    force_native_settings(); before_render = render_readback()
                    frame = sensors.capture(target_prim_path=target['prim_path'])
                    after_render = render_readback(); render_verification['captures_checked'] += 1
                    render_verification['last_capture_before'] = before_render
                    render_verification['last_capture_after'] = after_render
                    render_verification['capture_readbacks_match_request'] &= bool(before_render['matches_requested_values'] and after_render['matches_requested_values'])
                    write(output/'renderer_settings.json', render_verification)
                    if not render_verification['capture_readbacks_match_request']:
                        raise RuntimeError('Renderer overrode requested non-upscaling settings; rejecting this capture')
                    stem = f'frames/{step:06d}'; frame_id = f'{episode_id}:{step:06d}'
                    rgb256 = stem+'_rgb256.png'; rgb224 = stem+'_rgb224.png'; sensor_file = stem+'_sensors.npz'
                    image = Image.fromarray(frame['rgb']); image.save(output/rgb256)
                    image.resize((224, 224), Image.Resampling.BILINEAR).save(output/rgb224)
                    K256 = np.asarray(frame['camera']['K'], float); K224 = K256.copy(); K224[:2] *= 224./256.
                    np.savez_compressed(output/sensor_file, depth_m=frame['depth_m'], depth_valid=frame['depth_valid'],
                        instance_id=frame['instance_id'], K=K256, K256=K256, K224=K224,
                        world_from_camera=frame['camera']['world_from_camera'], vehicle_xy_yaw=pose,
                        vehicle_ground_z_m=ground, observation_time_s=step*.2)
                    write(output/(stem+'_instances.json'), frame['instance_mapping'])
                    matched = frame['evaluation']['matched_instance_ids']
                    mask = np.isin(frame['instance_id'], matched) if matched else np.zeros((256, 256), bool)
                    pixels = int(mask.sum()); visible = pixels >= 12; bbox = None; bearing = None
                    if pixels:
                        vv, uu = np.nonzero(mask); bbox = [int(uu.min()), int(vv.min()), int(uu.max()+1), int(vv.max()+1)]
                        bearing = float(np.degrees(np.arctan2(K256[0, 2]-(bbox[0]+bbox[2])*.5, K256[0, 0])))
                    terminal_position_ok = goal_error <= .05
                    if args.motion_controller == 'smooth_v13':
                        terminal_position_ok = terminal_position_ok and canonical_error <= 1.
                    stop = int(request_stop and visible and terminal_position_ok)
                    intent = action_intent(control[0], control[1], stop, args.motion_controller)
                    observation = dict(observation_id=frame_id, rgb=rgb224, depth=sensor_file, timestamp_s=step*.2)
                    evidence = dict(status='verified', method='current Isaac instance mask + calibrated bbox bearing + integrated pose entry gate',
                        source_observation_ids=[frame_id], target_id=args.target, target_class=target['class_name'],
                        target_visible=visible, target_visible_pixels=pixels, visibility_threshold_pixels=12,
                        bearing_deg=bearing if visible else None, target_bearing='unknown' if not visible else 'left' if bearing > 10. else 'right' if bearing < -10. else 'ahead',
                        bearing_convention='camera-relative, positive left; bbox centre and K256',
                        bearing_source='rendered_instance_mask_only', target_bbox_xyxy_256=bbox,
                        target_bbox_xyxy_224=None if bbox is None else (np.asarray(bbox)*224./256.).tolist(),
                        at_cave_entrance=at_entry, entry_event='entered_cave' if entered else 'at_cave_entrance' if at_entry else 'none',
                        progress=dict(temporal_entered_event=entered, entrance_geometry_case=at_entry,
                            source='actual consecutive integrated poses crossing authored x=0 join plane inside |y|<=14m; current entry slab |x|<=1m',
                            **({'inside_cave': has_entered_cave} if args.motion_controller != 'segment' else {})),
                        action_intent=intent, expected_action_intent=intent, action_intent_source='actually executed kinematic control',
                        terminal_accepted=bool(stop))
                    if fork_evidence is not None:
                        evidence.update(fork_evidence.observe(K256, frame['camera']['world_from_camera'],
                                                              frame['depth_m'], frame['depth_valid']))
                    if landmarks is not None:
                        evidence.update(landmarks.observe(frame['instance_id'], frame['instance_mapping'], K256, step*.2))
                    record = dict(schema_version='lunar_continuous_frame/11.0', sample_id=frame_id,
                        episode_id=episode_id, step=step, step_index=step, target_id=args.target,
                        annotation_mode='trajectory_step', annotation_status='annotation_proto', domain='isaac_continuous',
                        training_ready=False, instruction=instruction, rgb_224=rgb224, rgb_256=rgb256,
                        sensors_file=sensor_file, instances_file=stem+'_instances.json',
                        inputs=dict(instruction=instruction, observation=observation, history=list(history[-args.history_frames:]) if args.history_frames else []),
                        machine_evidence=evidence,
                        pose=dict(xyz_m=eye, ground_z_m=ground, yaw_rad=pose[2], origin='camera optical centre; privileged metadata, excluded from policy inputs'),
                        camera=dict(K256=K256, K224=K224, source_resolution=[256, 256], derived_rgb_resolution=[224, 224],
                            depth_resolution=[256, 256], hfov_deg=90., height_above_terrain_m=1.5, pitch_deg=0.,
                            depth_convention=frame['camera']['depth_convention'], coordinate_convention=frame['camera']['coordinate_convention'],
                            world_from_camera=frame['camera']['world_from_camera'], renderer_timeline_seconds=frame['camera'].get('timeline_seconds'),
                            renderer_before_capture=before_render, renderer_after_capture=after_render),
                        executed_action=dict(v_mps=control[0], omega_radps=control[1], stop=stop, dt_s=.2,
                            **({'motion_controller': args.motion_controller} if args.motion_controller != 'segment' else {})),
                        source_layout_seed=world.config.get('search_layout_seed'), independent_probe=False)
                next_ground = float(world.floor(next_pose[:2])[0])
                transition = dict(episode_id=episode_id, domain='isaac_continuous', step_index=step,
                    timestamp_s=step*.2, next_timestamp_s=(step+1)*.2, dt_s=.2,
                    current_observation_id=None if observation is None else observation['observation_id'],
                    current_observation=observation, current_frame_missing=observation is None,
                    action_vw_si=control, stop=stop, phase=phase['phase'],
                    teacher_pose_xy_yaw=pose, teacher_next_pose_xy_yaw=next_pose,
                    teacher_ground_z_m=ground, teacher_next_ground_z_m=next_ground,
                    terminal_goal_error_m=goal_error if request_stop else None,
                    collision_check_passed=True, training_ready=False)
                controls.append(control.copy()); stop_flags.append(stop); phase_ids.append(PHASES[phase['phase']])
                pose = next_pose; poses.append(pose.copy())
                append(transitions, transition)
                if record is not None:
                    append(records, record); history.append(observation); frame_count += 1
                if request_stop:
                    completed = bool(stop); reason = 'terminal_approach_visible_STOP' if stop else 'terminal_target_not_visible'; break
                if step % 20 == 0 or time.monotonic()-last_status >= 5.:
                    status('capturing_integrated_trajectory', target=args.target, simulated_time_s=(step+1)*.2)
                    last_status = time.monotonic()
    except BaseException as exc:
        reason = 'exception'; error = dict(error=repr(exc), traceback=traceback.format_exc())
        write(output/'error.json', error)
    finally:
        if not args.cpu_validate:
            try:
                import numpy as np
                from cave.lunar_trajectory_v11 import make_supervision
                executed = np.asarray(controls, np.float32).reshape(-1, 2)
                supervision = make_supervision(executed, np.asarray(stop_flags, np.uint8), phase=phase_ids)
                np.savez_compressed(output/'supervision.npz', **supervision)
                if poses and world is not None:
                    actual = np.asarray(poses, float); z = world.floor(actual[:, :2])
                    if len(actual) != len(executed)+1: raise RuntimeError('Controls/actual poses lost N+1 alignment')
                    np.savez_compressed(output/'teacher_only.npz', executed_positions=np.column_stack([actual[:, :2], z]),
                        executed_yaw_rad=actual[:, 2], original_approach_xy_m=world.targets[args.target]['approach_xy_m'],
                        source_path_xy_m=path, timestamp_s=np.arange(len(actual), dtype=float)*.2,
                        **({'planned_stop_xy_m': teacher.goal} if args.motion_controller == 'smooth_v13' else {}))
                unchanged = world.verify_sources() if world is not None else None
                if unchanged is False: completed = False; reason = 'source_asset_changed'
                write(output/'metadata.json', dict(schema_version='lunar_continuous_trajectory/11.0',
                    episode_id=episode_id, domain='isaac_continuous', source_scene=str(source),
                    source_sha256=None if world is None else world.source_hashes, source_assets_unchanged=unchanged,
                    source_layout_seed=None if world is None else world.config.get('search_layout_seed'), seed=args.seed,
                    target_id=args.target, terminal_success=completed, end_reason=reason,
                    action_count=len(controls), pose_count=len(poses), captured_frame_count=frame_count,
                    capture_stride=args.capture_stride, all_action_states_have_current_rgbd=args.capture_stride == 1,
                    dt_s=.2, max_v_mps=.4, max_abs_omega_radps=.8 if args.motion_controller == 'segment' else .32, future_horizon=8, phase_enum=PHASES,
                    motion_controller=args.motion_controller,
                    runtime_collision_mode=None if world is None else world.runtime_collision_mode,
                    motion_model=('exact integrated unicycle kinematics with collision-checked smooth path and slew-limited controls'
                                  if args.motion_controller != 'segment' else 'exact integrated unicycle kinematics, finite-rate pivot/straight map oracle'),
                    physics_scope='kinematic simulation; no tyre/soil/rigid-body dynamics',
                    terminal_rule=('canonical approach region <=1m, planned arrival <=0.05m, current instance visibility >=12px; zero control STOP without stationary turns'
                                   if args.motion_controller == 'smooth_v13' else 'original approach <=0.05m, terminal yaw faces target, current instance visibility >=12px; final zero control STOP'),
                    **({'canonical_stop_radius_m': 1., 'stationary_turns_allowed': False,
                        'body_wall_clearance': None if planning is None else planning.get('body_wall_clearance')}
                       if args.motion_controller == 'smooth_v13' else {}),
                    stop_scope='terminal teacher signal; separate near-goal STOP expert labels are not synthesized',
                    collision_checks=collision_aggregate,
                    obstacle_layer_provenance=None if world is None else world.obstacle_metadata,
                    target_collision_circles=0 if world is None else len(world.targets),
                    small_gravel_scope='below serialized 25cm collision threshold: visual detail, not wheel/soil collision simulation',
                    camera=dict(resolution=[256, 256], rgb_resized=[224, 224],
                        hfov_deg=90., height_above_ground_m=1.5, pitch_deg=0., rgb_resize='Pillow bilinear; K224 scaled explicitly',
                        antialiasing='FXAA requested; actual setting readbacks in renderer_settings.json',
                        renderer_verification=render_verification),
                    light=dict(headlamp=24000., work_each=8000., exposure=6., dome=45., sun=3500., normalized_vehicle_lights=True),
                    time_basis='integer executed control index * 0.2s; renderer timeline is not control time',
                    inputs_allowlist=['instruction', 'current RGB224', 'current depth256 metres', 'strictly past same-episode observations'],
                    privileged_files=['teacher_only.npz', 'planning_teacher_only.json', 'transitions.jsonl'],
                    native_action_labels_provided=False, training_ready=False, annotation_status='annotation_proto',
                    readiness_blockers=['independent 129 loader/adapter contract validation', 'new step-text output and loss integration', 'layout/trajectory split and annotation guards']))
                status('complete' if completed else 'partial', passed=completed, reason=reason, error=error)
            except BaseException as exc:
                completed = False; error = dict(finalization_error=repr(exc), traceback=traceback.format_exc())
                write(output/'finalization_error.json', error); status('failed_finalization', error=error)
        if sensors is not None:
            try: sensors.release()
            except Exception as exc: print('Sensor release:', repr(exc), flush=True)
        if app is not None: app.close()
    return 0 if completed else 1


if __name__ == '__main__': raise SystemExit(main())
