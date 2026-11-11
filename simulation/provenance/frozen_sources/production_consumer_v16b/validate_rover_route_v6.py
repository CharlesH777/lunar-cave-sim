"""CPU geometry screening for the pit-floor rover route, using saved triangles.

This checks one declared route and a simple support-plane vehicle envelope.
It does not run PhysX, wheel dynamics, a controller, or a navigation policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def _axis_samples(low, high, step):
    result = np.arange(float(low), float(high), float(step))
    return np.r_[result, float(high)]


def _finite_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(minimum=None, maximum=None, mean=None)
    return dict(minimum=float(finite.min()), maximum=float(finite.max()), mean=float(finite.mean()))


def _world_xy(xy, yaw, offsets):
    forward = np.column_stack([np.cos(yaw), np.sin(yaw)])
    left = np.column_stack([-np.sin(yaw), np.cos(yaw)])
    return xy[:, None, :] + offsets[None, :, 0, None]*forward[:, None, :] + offsets[None, :, 1, None]*left[:, None, :]


def _measure_geometry(projector, x, y, field_function, limits):
    """Measure saved-floor contact at every pose; injectable for controlled toys."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.shape != y.shape or np.any(np.diff(x) <= 0):
        raise ValueError('Route samples require matching increasing x and finite y')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Route coordinates must be finite')
    xy = np.column_stack([x, y])
    yaw = np.arctan2(np.gradient(y, x, edge_order=2), 1.)
    wheel_offsets = np.array([[.26, .33], [.26, -.33], [-.26, .33], [-.26, -.33]])
    wheel_xy = _world_xy(xy, yaw, wheel_offsets)
    wheel_z, _, wheel_face, _ = projector.project(wheel_xy.reshape(-1, 2))
    wheel_z = wheel_z.reshape(-1, 4); wheel_face = wheel_face.reshape(-1, 4)
    wheel_support = np.isfinite(wheel_z).all(1) & (wheel_face >= 0).all(1)
    # Constant local XY layout makes the least-squares plane unambiguous.
    design = np.column_stack([np.ones(4), wheel_offsets])
    plane = np.full((len(x), 3), np.nan)
    plane[wheel_support] = wheel_z[wheel_support] @ np.linalg.pinv(design).T
    predicted = plane @ design.T
    residual = np.abs(wheel_z - predicted)
    pitch = np.degrees(np.arctan(plane[:, 1]))
    roll = np.degrees(np.arctan(plane[:, 2]))
    wheel_steps = np.abs(np.diff(wheel_z, axis=0))
    xy_distances = np.linalg.norm(np.diff(wheel_xy, axis=0), axis=-1)

    gx, gy = np.meshgrid(_axis_samples(-.4, .4, .1), _axis_samples(-.26, .26, .1), indexing='ij')
    belly_offsets = np.column_stack([gx.ravel(), gy.ravel()])
    belly_xy = _world_xy(xy, yaw, belly_offsets)
    belly_z, _, belly_face, _ = projector.project(belly_xy.reshape(-1, 2))
    belly_z = belly_z.reshape(len(x), -1); belly_face = belly_face.reshape(len(x), -1)
    belly_plane = (plane[:, 0, None] + plane[:, 1, None]*belly_offsets[None, :, 0]
                   + plane[:, 2, None]*belly_offsets[None, :, 1])
    clearance = belly_plane + limits['belly_ground_allowance_m'] - belly_z
    belly_support = np.isfinite(belly_z).all(1) & (belly_face >= 0).all(1)

    # Full visual footprint plus 20cm above the approximately 1m mast.
    ex, ey = np.meshgrid(np.linspace(-.475, .475, 5), np.linspace(-.425, .425, 5), indexing='ij')
    envelope_offsets = np.column_stack([ex.ravel(), ey.ravel()])
    envelope_xy = _world_xy(xy, yaw, envelope_offsets)
    envelope_floor = (plane[:, 0, None] + plane[:, 1, None]*envelope_offsets[None, :, 0]
                      + plane[:, 2, None]*envelope_offsets[None, :, 1])
    heights = np.array([.3, .8, 1.2])
    all_air = np.zeros(len(x), dtype=bool)
    minimum_margin = np.full(len(x), np.nan)
    if np.any(wheel_support):
        selected = np.flatnonzero(wheel_support)
        pxy = np.broadcast_to(envelope_xy[selected, :, None, :], (len(selected), len(envelope_offsets), len(heights), 2))
        pz = envelope_floor[selected, :, None] + heights[None, None, :]
        points = np.concatenate([pxy, pz[..., None]], axis=-1).reshape(-1, 3)
        values = np.concatenate([np.asarray(field_function(points[i:i+65536]), float).reshape(-1)
                                 for i in range(0, len(points), 65536)])
        values = values.reshape(len(selected), len(envelope_offsets)*len(heights))
        all_air[selected] = np.isfinite(values).all(1) & (values > 0).all(1)
        minimum_margin[selected] = np.min(values, axis=1)

    max_pitch = _finite_summary(np.abs(pitch))['maximum']
    max_roll = _finite_summary(np.abs(roll))['maximum']
    max_step = _finite_summary(wheel_steps)['maximum']
    max_residual = _finite_summary(residual)['maximum']
    min_clearance = _finite_summary(clearance)['minimum']
    checks = dict(
        all_wheels_have_mesh_support=bool(wheel_support.all()),
        belly_grid_has_mesh_support=bool(belly_support.all()),
        longitudinal_grade_within_limit=max_pitch is not None and max_pitch <= limits['max_pitch_deg']+1e-7,
        cross_grade_within_limit=max_roll is not None and max_roll <= limits['max_roll_deg']+1e-7,
        wheel_short_step_within_limit=max_step is not None and max_step <= limits['max_step_m']+1e-7,
        wheel_contact_nonplanarity_within_limit=max_residual is not None and max_residual <= limits['max_wheel_plane_residual_m']+1e-7,
        belly_clearance_within_limit=min_clearance is not None and min_clearance >= limits['min_belly_clearance_m']-1e-7,
        sampled_air_envelope_clear=bool(all_air.all()),
    )
    violations = []
    per_pose_bad = (~wheel_support | ~belly_support | ~all_air
                    | (np.abs(pitch) > limits['max_pitch_deg']+1e-7)
                    | (np.abs(roll) > limits['max_roll_deg']+1e-7)
                    | np.any(residual > limits['max_wheel_plane_residual_m']+1e-7, axis=1)
                    | np.any(clearance < limits['min_belly_clearance_m']-1e-7, axis=1))
    for index in np.flatnonzero(per_pose_bad)[:30]:
        violations.append(dict(index=int(index), x_m=float(x[index]), y_m=float(y[index]),
            pitch_deg=None if not np.isfinite(pitch[index]) else float(pitch[index]),
            roll_deg=None if not np.isfinite(roll[index]) else float(roll[index]),
            wheel_support=bool(wheel_support[index]), belly_support=bool(belly_support[index]),
            air_envelope_clear=bool(all_air[index])))
    step_bad = np.argwhere(wheel_steps > limits['max_step_m']+1e-7)
    return dict(passed=bool(all(checks.values())), checks=checks, pose_count=len(x),
        sampled_x_range_m=[float(x[0]), float(x[-1])],
        sampled_xy_length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        vehicle_geometry=dict(wheel_radius_m=.18, wheelbase_m=.52, track_width_m=.66,
            full_visual_footprint_m=[.95, .85], body_underfloor_allowance_m=limits['belly_ground_allowance_m']),
        wheel_contact=dict(count=int(wheel_z.size), missing_count=int(np.count_nonzero(wheel_face < 0)),
            maximum_abs_pitch_deg=max_pitch, maximum_abs_roll_deg=max_roll,
            maximum_adjacent_sample_height_change_m=max_step,
            adjacent_wheel_xy_spacing_m=_finite_summary(xy_distances),
            maximum_four_wheel_plane_residual_m=max_residual,
            maximum_wheel_step_violations=int(len(step_bad)),
            first_step_violations=[dict(pose_index=int(i), wheel_index=int(j), delta_z_m=float(wheel_steps[i, j]))
                                   for i,j in step_bad[:20]],
            method='four saved-mesh height contacts fitted to a local least-squares support plane'),
        belly=dict(samples_per_pose=len(belly_offsets), minimum_vertical_clearance_m=min_clearance,
                   method='saved-mesh heights on a 0.1m local underbody grid compared with support plane plus 0.15m'),
        air_envelope=dict(samples_per_pose=int(len(envelope_offsets)*len(heights)),
            heights_above_support_plane_m=heights.tolist(), all_positive_air=bool(all_air.all()),
            minimum_implicit_margin_not_euclidean_clearance_m=_finite_summary(minimum_margin)['minimum'],
            source='declared composed air field; sign only, not a Euclidean clearance distance'),
        first_pose_violations=violations, violating_pose_count=int(np.count_nonzero(per_pose_bad)))


def _boulder_check(scene, route, distance_function):
    path = scene/'boulders.json'
    if not path.exists():
        return dict(passed=True, available=False, count=0, note='No separate macro-boulder file in this scene')
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError('boulders.json must contain a list')
    half_width = float(route.get('drivable_half_width_m', 1.2))
    bad, checked, gaps = [], 0, []
    for index, item in enumerate(data):
        center, scale = np.asarray(item['center_xyz_m'], float), np.asarray(item['scale_xyz_m'], float)
        if center.shape != (3,) or scale.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError('Malformed macro-boulder dimensions')
        if center[2] >= 22.:
            continue  # Surface-rim blocks are not in the floor-level mission.
        distance = float(np.asarray(distance_function(route, center[None, :2])).reshape(-1)[0])
        radius = 1.35*float(np.linalg.norm(scale[:2]))
        gap = distance-radius-half_width
        checked += 1; gaps.append(gap)
        if gap < -1e-7:
            bad.append(dict(index=index, center_xyz_m=center.tolist(), distance_to_route_m=distance,
                            conservative_xy_radius_m=radius, band_clearance_m=gap))
    return dict(passed=not bad, available=True, count=len(data), floor_level_count=checked,
        minimum_conservative_band_clearance_m=min(gaps) if gaps else None,
        drivable_half_width_m=half_width, xy_radius_bound='1.35 * hypot(scale_x, scale_y)',
        source='saved boulder specs; conservative footprint screening, not exact triangle collision',
        overlap_count=len(bad), first_overlaps=bad[:20])


def _gravel_check(scene, route):
    path = scene/'surface_detail_report.json'
    if not path.exists():
        return dict(passed=True, available=False, maximum_actual_support_plane_protrusion_m=0.,
                    note='No micro-gravel surface-detail layer is declared by this base scene')
    detail = json.loads(path.read_text())
    gravel = detail.get('gravel', {})
    if int(gravel.get('instance_count', 0)) == 0:
        return dict(passed=True, available=True, instance_count=0,
                    maximum_actual_support_plane_protrusion_m=0.)
    r = gravel.get('route_gravel')
    if not isinstance(r, dict):
        return dict(passed=False, available=True, error='Micro-gravel exists but lacks route-specific size/protrusion evidence')
    maximum = r.get('maximum_actual_support_plane_protrusion_m')
    diameter = r.get('maximum_actual_full_diameter_m')
    allowed = float(route.get('max_gravel_protrusion_m', .02))
    diameter_limit = float(route.get('max_gravel_diameter_m', .06))
    passed = (r.get('passed') is True and maximum is not None and diameter is not None
              and float(maximum) <= allowed+1e-7 and float(diameter) <= diameter_limit+1e-7
              and int(r.get('diameter_cap_violation_count', 0)) == 0
              and int(r.get('protrusion_cap_violation_count', 0)) == 0
              and int(r.get('radius_guard_missed_count', 0)) == 0)
    return dict(passed=bool(passed), available=True, instance_count=int(gravel.get('instance_count', 0)),
        maximum_actual_support_plane_protrusion_m=maximum, allowed_support_plane_protrusion_m=allowed,
        maximum_actual_full_diameter_m=diameter, allowed_full_diameter_m=diameter_limit,
        source='serialized-instance route-gravel checks in surface_detail_report.json', evidence=r,
        micro_gravel_collision=bool(detail.get('micro_gravel_collision', False)))


def validate_route(scene_dir, output=None):
    """Return route geometry checks; optional output writes only the requested JSON."""
    scene = Path(scene_dir).expanduser().resolve(strict=True)
    config_path, mesh_path = scene/'scene_config.json', scene/'mesh.npz'
    config = json.loads(config_path.read_text())
    route = config.get('rover_route')
    if not isinstance(route, dict):
        raise ValueError('scene_config.json requires an explicit top-level rover_route')
    from cave.rover_route_v6 import route_y, route_distance_xy
    from cave.lunar_gravel_v5 import _FloorProjector
    from build_lunar_cave_v4_cpu import evaluate_scene_field

    protected = [config_path, mesh_path]
    for name in ['cave.usdc', 'geometry_report.json', 'boulders.json', 'surface_detail_report.json']:
        if (scene/name).is_file():protected.append(scene/name)
    initial = {str(path):_sha(path) for path in protected}
    low, high = map(float, route.get('drive_x_range_m', [-15., 74.]))
    sample_step = float(route.get('validation_sample_step_m', .1))
    if high <= low or not 0.02 <= sample_step <= .1:
        raise ValueError('Invalid drive range or route validation sampling interval')
    limits = dict(max_pitch_deg=float(route.get('max_pitch_deg', 5.)),
        max_roll_deg=float(route.get('max_roll_deg', 3.)), max_step_m=float(route.get('max_step_m', .03)),
        max_wheel_plane_residual_m=float(route.get('max_wheel_plane_residual_m', .015)),
        min_belly_clearance_m=float(route.get('min_belly_clearance_m', .10)), belly_ground_allowance_m=.15)
    if any(not np.isfinite(value) or value <= 0 for value in limits.values()):
        raise ValueError('Validation limits must be positive finite engineering values')
    with np.load(mesh_path) as data:
        projector = _FloorProjector(data['vertices'], data['faces'])
    x = _axis_samples(low, high, sample_step); y = route_y(route, x)
    measure = _measure_geometry(projector, x, y, lambda p:evaluate_scene_field(config, p), limits)
    macro = _boulder_check(scene, route, route_distance_xy)
    gravel = _gravel_check(scene, route)
    hook_match = all(config.get(part, {}).get('rover_route_floor') == route for part in ['cavern', 'pit'])
    geometry_report = json.loads((scene/'geometry_report.json').read_text()) if (scene/'geometry_report.json').is_file() else {}
    base_passed = geometry_report.get('passed') is True
    micro = float(gravel.get('maximum_actual_support_plane_protrusion_m') or 0.)
    mesh_step = measure['wheel_contact']['maximum_adjacent_sample_height_change_m']
    mesh_clear = measure['belly']['minimum_vertical_clearance_m']
    combined_step = None if mesh_step is None else mesh_step+micro
    combined_clear = None if mesh_clear is None else mesh_clear-micro
    combined_checks = dict(
        short_step_plus_micro_bound=combined_step is not None and combined_step <= limits['max_step_m']+1e-7,
        belly_clearance_minus_micro_bound=combined_clear is not None and combined_clear >= limits['min_belly_clearance_m']-1e-7)
    result = dict(passed=bool(measure['passed'] and macro['passed'] and gravel['passed'] and hook_match
                            and base_passed and all(combined_checks.values())),
        schema_version='rover_route_geometry_check/6.0', updated_utc=datetime.now(timezone.utc).isoformat(),
        scene=str(scene), source_hashes=initial, task_start='pit floor; vertical skylight excluded from driving route',
        route_floor_hooks_match=hook_match, base_geometry_report_passed=base_passed,
        declared_route=route, engineering_limits=limits, geometry=measure,
        macro_boulders=macro, micro_gravel=gravel,
        conservative_compound_bounds=dict(checks=combined_checks,
            mesh_short_step_plus_micro_protrusion_m=combined_step,
            belly_clearance_minus_micro_protrusion_m=combined_clear,
            note='Worst-case scalar geometric bounds; not a wheel/contact dynamics simulation'),
        CPU_only=True, GPU_work=False, physics_evaluated=False, navigation_evaluated=False,
        limitations=['One sampled declared route, not all free space in the cave',
            'Support-plane fitting approximates a rigid rover; no suspension, traction, friction or wheel dynamics',
            'Air checks sample the declared implicit field and omit separately authored rock meshes; macro rocks and micro caps are screened separately',
            'No PhysX cooking or continuous swept triangle-intersection proof',
            'Configured grades and clearances are engineering assumptions, not validated hardware performance limits'])
    unchanged = all(_sha(path) == value for path,value in initial.items())
    result['source_files_unchanged'] = unchanged
    result['passed'] = bool(result['passed'] and unchanged)
    if output is not None:
        destination = Path(output).expanduser().absolute()
        if destination in protected:
            raise ValueError('Validation output may not replace a source asset')
        _write(destination,result)
    return result


def _self_test():
    """Small real-triangle controls: flat, graded, banked, stepped, blocked air."""
    from cave.lunar_gravel_v5 import _FloorProjector
    xx, yy = np.meshgrid(np.linspace(-4, 4, 81), np.linspace(-3, 3, 31), indexing='ij')
    ii,jj=np.meshgrid(np.arange(len(xx)-1),np.arange(xx.shape[1]-1),indexing='ij')
    a=(ii*xx.shape[1]+jj).ravel();b=a+xx.shape[1]
    faces=np.r_[np.column_stack([a,b,b+1]),np.column_stack([a,b+1,a+1])]
    limits=dict(max_pitch_deg=5.,max_roll_deg=3.,max_step_m=.03,
        max_wheel_plane_residual_m=.015,min_belly_clearance_m=.10,belly_ground_allowance_m=.15)
    x=_axis_samples(-2,2,.1);y=np.zeros_like(x);checks={}
    for name,fn in [('flat',lambda x,y:0*x),('long_grade',lambda x,y:np.tan(np.deg2rad(8))*x),
                    ('cross_grade',lambda x,y:np.tan(np.deg2rad(5))*y),
                    ('step',lambda x,y:np.where(x<0,0.,.12)),('blocked_air',lambda x,y:0*x)]:
        vertices=np.column_stack([xx.ravel(),yy.ravel(),fn(xx,yy).ravel()])
        projector=_FloorProjector(vertices,faces)
        field=(lambda p:np.where((p[:,0]>-.5)&(p[:,0]<.5),-1.,1.)) if name=='blocked_air' else (lambda p: p[:,2]-fn(p[:,0],p[:,1]))
        report=_measure_geometry(projector,x,y,field,limits)
        expected=name=='flat'
        okay=report['passed']==expected
        if name=='long_grade':okay &= not report['checks']['longitudinal_grade_within_limit']
        if name=='cross_grade':okay &= not report['checks']['cross_grade_within_limit']
        if name=='step':okay &= not report['checks']['wheel_short_step_within_limit']
        if name=='blocked_air':okay &= not report['checks']['sampled_air_envelope_clear']
        checks[name]=dict(passed=bool(okay),actual_acceptance=report['passed'],checks=report['checks'])
    return dict(passed=all(row['passed'] for row in checks.values()),tests=checks,GPU_work=False,physics_evaluated=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test:
        report=_self_test()
        if args.output:_write(args.output,report)
    else:
        if args.scene is None:parser.error('--scene is required unless --self-test is used')
        report=validate_route(args.scene,args.output)
    print(json.dumps(dict(passed=report['passed'],output=str(args.output) if args.output else None,
                         GPU_work=False,physics_evaluated=False)))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
