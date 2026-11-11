"""Capture nine independent, real RGB-D probes for grounded annotation review.

These are sensor probes, not a temporal rollout or a training-ready action
dataset. Direction suggestions describe target bearing; no control is executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def _jsonable(value):
    if isinstance(value, dict):return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):return [_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):return _jsonable(value.tolist())
    if hasattr(value, 'item'):return _jsonable(value.item())
    if isinstance(value, Path):return str(value)
    return value


def write(path, value):
    path = Path(path);temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):digest.update(part)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/lunar_search_v8/demo_seed_20261118')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subframes', type=int, default=3, choices=[3],
                        help='CaveSensors owns the synchronized three-subframe capture')
    args = parser.parse_args(argv);sys.argv = [sys.argv[0]]
    source = args.source.resolve(strict=True);output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic();app = None;sensors = None
    protected = {name: sha(source/name) for name in [
        'cave.usdc', 'scene_config.json', 'geometry_report.json', 'mesh.npz',
        'boulders.json', 'obstacle_report.json', 'targets.json', 'task_obstacles.json']}

    def status(phase, **extra):
        record = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=round(time.monotonic()-started, 3),
                      training_ready=False, temporal_rollout=False, **extra)
        write(output/'status.json', record);print(json.dumps(record), flush=True)

    try:
        status('starting_isaac_sensor_probes')
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer='RayTracedLighting', width=256, height=256,
            multi_gpu=False, active_gpu=0, physics_gpu=0, sync_loads=True,
            extra_args=['--ext-folder', '/home/charles/.local/share/ov/data/exts/v2',
                '--/app/extensions/registryEnabled=false', '--/renderer/multiGpu/enabled=false',
                '--/rtx/indirectDiffuse/enabled=true', '--/rtx/ambientOcclusion/enabled=true',
                '--/exts/omni.replicator.core/Orchestrator/enabled=true']),
            experience='/home/charles/ISAAC_SIM/IsaacLab/apps/isaaclab.python.headless.rendering.kit')
        import numpy as np
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom, UsdLux
        from scipy.ndimage import label
        from PIL import Image
        from cave.scene import camera, set_camera, sphere_light
        from cave.sensors import CaveSensors
        from cave.lunar_gravel_v5 import _FloorProjector
        from build_lunar_cave_v4_cpu import evaluate_scene_field

        config = json.loads((source/'scene_config.json').read_text())
        targets = json.loads((source/'targets.json').read_text())['targets']
        by_id = {item['id']: item for item in targets}
        with np.load(source/'mesh.npz') as data:projector = _FloorProjector(data['vertices'], data['faces'])
        with np.load(source/'validation/grid_data.npz') as data:
            grid_x, grid_y, free = [np.array(data[k], copy=True) for k in ['x', 'y', 'free']]
        path = np.asarray(json.loads((source/'validation/obstacle_map_report.json').read_text())['path_xyz_m'], float)
        entry_index = int(np.argmin(np.abs(path[:, 0])))
        entry_xy = path[entry_index, :2]
        tangent = path[min(entry_index+3, len(path)-1), :2] - path[max(0, entry_index-3), :2]
        entry_yaw = float(np.arctan2(tangent[1], tangent[0]))
        components, _ = label(free)
        xx, yy = np.meshgrid(grid_x, grid_y, indexing='ij')
        spacing = float(grid_x[1]-grid_x[0])

        def floor_at(xy):
            height, _, faces, _ = projector.project(np.asarray(xy, float).reshape(-1, 2))
            if not np.isfinite(height).all() or np.any(faces < 0):raise ValueError('No actual terrain support')
            return height

        probes = [dict(sample_id='probe_000', target=by_id['chinese_flag'], entry=True,
                       candidates=[(entry_xy.copy(), entry_yaw)], desired_bearing_deg=None)]
        for target in targets:
            center = np.asarray(target['xyz_m'][:2], float);approach = np.asarray(target['approach_xy_m'], float)
            cell = np.rint((approach - [grid_x[0], grid_y[0]])/spacing).astype(int)
            component = int(components[tuple(cell)])
            if not component:raise ValueError('Target approach is not in a free component')
            distance = np.hypot(xx-center[0], yy-center[1])
            near_approach = np.hypot(xx-approach[0], yy-approach[1]) <= 3.0
            eligible = free & (components == component) & near_approach & (distance >= 2.8) & (distance <= 5.0)
            ids = np.argwhere(eligible)
            if not len(ids):raise ValueError('No nearby free sensor position for '+target['id'])
            candidates = np.column_stack([grid_x[ids[:, 0]], grid_y[ids[:, 1]]])
            radial = approach-center;radial /= np.linalg.norm(radial)
            for side in (1., -1.):
                # Choose two nearby, independent camera positions. The optical
                # axis stays horizontal; only world yaw is selected for a probe.
                angle = side*np.radians(16.)
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                desired = center + max(3.5, float(target['footprint_radius_m'])+1.15)*(rotation@radial)
                ordering = np.argsort(np.linalg.norm(candidates-desired, axis=1), kind='stable')
                selected = [];last_xy = None
                for index in ordering:
                    xy = candidates[index]
                    if last_xy is not None and np.linalg.norm(xy-last_xy)<.30:continue
                    ground = float(floor_at(xy)[0]);eye = np.r_[xy, ground+1.5]
                    if float(evaluate_scene_field(config, eye)) <= .15:continue
                    bearing = float(np.arctan2(center[1]-xy[1], center[0]-xy[0]))
                    selected.append((xy.copy(), bearing-side*np.radians(25.)))
                    last_xy = xy
                    if len(selected) == 3:break
                if not selected:raise ValueError('No valid camera for '+target['id'])
                probes.append(dict(sample_id=f'probe_{len(probes):03d}', target=target, entry=False,
                                   candidates=selected, desired_bearing_deg=side*25.))

        context = omni.usd.get_context()
        if not context.open_stage(str(source/'cave.usdc')):raise RuntimeError('Could not open source cave')
        stage = context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            for attribute in prim.GetAttributes():
                value = attribute.Get()
                if isinstance(value, Sdf.AssetPath) and value.resolvedPath:attribute.Set(Sdf.AssetPath(value.resolvedPath))
        settings = carb.settings.get_settings();settings.set('/rtx/post/dlss/execMode', 2)
        settings.set('/rtx/sceneDb/ambientLightIntensity', 0.)
        UsdGeom.Xform.Define(stage, '/World/StepCoTProbe')
        sun = UsdLux.DistantLight.Define(stage, '/World/StepCoTProbe/Sun')
        sun.CreateIntensityAttr(3500.);sun.CreateAngleAttr(.53)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(24., -24., -25.))
        dome = UsdLux.DomeLight.Define(stage, '/World/StepCoTProbe/GentleFill')
        dome.CreateIntensityAttr(45.);dome.CreateColorAttr(Gf.Vec3f(.70, .75, .80))
        cam = camera(stage, '/World/StepCoTProbe/Camera', (0, 0, 1.5), (1, 0, 1.5), focal=18.)
        cam.CreateHorizontalApertureAttr(36.);cam.CreateVerticalApertureAttr(36.)
        headlamp = sphere_light(stage, '/World/StepCoTProbe/Headlamp', (0, 0, 1.5), (.94, .97, 1.), 24000., .28)
        headlamp.CreateNormalizeAttr(True);headlamp.CreateExposureAttr(6.)
        work_lights=[]
        for side in [-1,1]:
            light=sphere_light(stage,'/World/StepCoTProbe/WorkLight'+('Left' if side<0 else 'Right'),(0,0,1.2),
                (.94,.97,1.),8000.,.45)
            light.CreateNormalizeAttr(True);light.CreateExposureAttr(6.);work_lights.append((side,light))
        rep.orchestrator.set_capture_on_play(False)
        sensors = CaveSensors(stage, str(cam.GetPath()), (256, 256))
        for _ in range(25):app.update()
        captures = 0;records = [];attempts = []
        for probe_index, probe in enumerate(probes):
            if (output/'REQUEST_STOP').exists():raise RuntimeError('Stop requested before all independent probes finished')
            selected_record = None
            for candidate_index, (xy, yaw) in enumerate(probe['candidates']):
                # Reserve one capture for each remaining probe; at most three
                # additional retries exist across the complete nine-probe set.
                if captures >= 12:break
                if candidate_index and captures >= 12-(len(probes)-probe_index-1):break
                ground = float(floor_at(xy)[0]);eye = np.r_[xy, ground+1.5]
                forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
                set_camera(cam, eye, eye+forward*10.)
                headlamp.GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float, eye+forward*.12)))
                left=np.array([-forward[1],forward[0],0.])
                for side,light in work_lights:
                    position=eye+forward*.22+left*(side*.45)+[0,0,-.25]
                    light.GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float,position)))
                for _ in range(2):app.update()
                frame = sensors.capture(target_prim_path=probe['target']['prim_path']);captures += 1
                stem = f"{probe['sample_id']}_attempt_{candidate_index}"
                rgb256 = stem+'_rgb256.png';rgb224 = stem+'_rgb224.png'
                sensor_file = stem+'_sensors.npz';instances_file = stem+'_instances.json'
                image = Image.fromarray(frame['rgb']);image.save(output/rgb256)
                image.resize((224, 224), Image.Resampling.BILINEAR).save(output/rgb224)
                K256 = np.asarray(frame['camera']['K'], float);K224 = K256.copy();K224[:2] *= 224./256.
                np.savez_compressed(output/sensor_file, depth_m=frame['depth_m'], depth_valid=frame['depth_valid'],
                    instance_id=frame['instance_id'], K256=K256, K224=K224, K=K256,
                    world_from_camera=frame['camera']['world_from_camera'])
                write(output/instances_file, frame['instance_mapping'])
                matched = frame['evaluation']['matched_instance_ids']
                target_mask = np.isin(frame['instance_id'], matched) if matched else np.zeros((256, 256), bool)
                pixel_count = int(target_mask.sum());visible = pixel_count >= 12
                target_xy = np.asarray(probe['target']['xyz_m'][:2], float)
                bearing = (np.arctan2(target_xy[1]-xy[1], target_xy[0]-xy[0])-yaw+np.pi)%(2*np.pi)-np.pi
                center_bearing_deg = float(np.degrees(bearing))
                bbox = None
                if pixel_count:
                    v, u = np.nonzero(target_mask);bbox = [int(u.min()), int(v.min()), int(u.max()+1), int(v.max()+1)]
                bbox_center = None if bbox is None else [(bbox[0]+bbox[2])*.5, (bbox[1]+bbox[3])*.5]
                mask_bearing_deg = None if bbox_center is None else float(np.degrees(
                    np.arctan2(K256[0,2]-bbox_center[0], K256[0,0])))
                bearing_deg = mask_bearing_deg if visible else center_bearing_deg
                intent = 'explore' if not visible else ('turn_left' if bearing_deg > 10. else 'turn_right' if bearing_deg < -10. else 'move_forward')
                selected_record = dict(sample_id=probe['sample_id'], step=0, independent_probe=True,
                    instruction='找到'+probe['target']['label_zh']+'并靠近。',
                    rgb_224=rgb224, rgb_256=rgb256, sensors_file=sensor_file, instances_file=instances_file,
                    pose=dict(xyz_m=eye.tolist(), yaw_rad=float(yaw), yaw_deg=float(np.degrees(yaw)),
                              ground_z_m=ground, origin='camera optical origin in world Z-up metres'),
                    camera=dict(K256=K256, K224=K224, source_resolution=[256, 256], derived_rgb_resolution=[224, 224],
                        hfov_deg=90., pitch_deg=0., height_above_terrain_m=1.5,
                        depth_resolution=[256, 256], depth_convention=frame['camera']['depth_convention'],
                        coordinate_convention=frame['camera']['coordinate_convention'],
                        world_from_camera=frame['camera']['world_from_camera'], resize='RGB only: Pillow bilinear 256->224'),
                    machine_evidence=dict(target_id=probe['target']['id'], target_class=probe['target']['class_name'],
                        target_visible_pixels=pixel_count, target_visible=visible, visibility_threshold_pixels=12,
                        bearing_deg=bearing_deg, bearing_convention='world Z-up CCW; positive left, negative right',
                        bearing_source='rendered_mask_bbox_center' if visible else 'privileged_center_reference_only',
                        visible_mask_bearing_deg=mask_bearing_deg, target_center_bearing_deg_gt=center_bearing_deg,
                        bbox_center_uv_256=bbox_center,
                        at_cave_entrance=bool(probe['entry']),
                        progress=dict(entry_event=False, entrance_geometry_case=bool(probe['entry']),
                            temporal_entered_event=False, source='explicit_current_geometry_case_not_temporal_event'),
                        action_intent=intent, action_intent_source='geometric_bearing_probe_not_executed_control',
                        target_bbox_xyxy_256=bbox,
                        target_bbox_xyxy_224=None if bbox is None else (np.asarray(bbox)*224./256.).tolist(),
                        bbox_convention='xyxy, exclusive maximum; 224 coordinates are the explicit resize transform'),
                    capture_attempt=candidate_index, source_layout_seed=int(config['search_layout_seed']),
                    training_ready=False, executed_action=None, future_motion_ground_truth=None)
                attempts.append(selected_record)
                if visible or probe['entry']:break
            if selected_record is None:raise RuntimeError('Capture budget exhausted before '+probe['sample_id'])
            records.append(selected_record)
            status('capturing_independent_probes', completed=len(records), capture_count=captures,
                   target=probe['target']['id'], target_visible=selected_record['machine_evidence']['target_visible'])
        sensors.release();sensors = None
        with (output/'frame_records.jsonl').open('w') as stream:
            for record in records:stream.write(json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False)+'\n')
        with (output/'attempt_records.jsonl').open('w') as stream:
            for record in attempts:stream.write(json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False)+'\n')
        unchanged = all(sha(source/name) == digest for name, digest in protected.items())
        if not unchanged:raise RuntimeError('Protected source asset changed')
        write(output/'metadata.json', dict(schema_version='lunar_independent_sensor_probes/9.0',
            purpose='frame-grounded annotation prototype', source_scene=str(source), source_sha256=protected,
            source_assets_unchanged=True, selected_frame_count=len(records), actual_capture_count=captures,
            maximum_capture_count=12, synchronized_render_subframes=3,
            camera=dict(source_resolution=[256,256], rgb_training_preview_resolution=[224,224], hfov_deg=90.,
                        pitch_deg=0., height_above_ground_m=1.5, no_target_lookat_pitch=True),
            sensor_coordinates='world Z-up metres; camera OpenCV +X right,+Y down,+Z forward; column-vector transforms',
            progress_entry_event_scope='always false: independent probes cannot establish a past entered event',
            independent_probes=True, temporal_rollout=False, training_ready=False,
            labels_are_executed_controls=False, future_eight_step_ground_truth_available=False,
            action_intent_scope='bearing-based suggestion for review, not an executed teacher action',
            lighting='constant vehicle headlamp and two broad work lights plus gentle fill and fixed Sun',
            headlamp_exposure=6.,headlamp_intensity_usd=24000.,work_light_intensity_usd_each=8000.,
            gentle_fill_intensity_usd=45.,lighting_target_independent=True,
            annotation_progress='annotation_proto', navigation_policy_evaluated=False, physics_evaluated=False))
        status('annotation_proto', passed=True, selected_frame_count=len(records), actual_capture_count=captures,
               frame_records='frame_records.jsonl', source_assets_unchanged=True)
        return 0
    except BaseException as exc:
        status('failed', error=repr(exc));write(output/'error.json', dict(error=repr(exc), traceback=traceback.format_exc()));raise
    finally:
        if sensors is not None:sensors.release()
        if app is not None:app.close()


if __name__ == '__main__':raise SystemExit(main())
