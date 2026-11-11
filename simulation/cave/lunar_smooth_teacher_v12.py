"""Preflighted continuous teacher for the accepted lunar cave geometry.

The centre follows a collision-checked spline using exact unicycle arcs at
the same 0.2 s control interval as the collector. The camera can therefore
use the executed body yaw directly, including during gentle end alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import CubicHermiteSpline, splprep, splev
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from cave.hybrid_planner import smooth_hybrid_path
from cave.lunar_trajectory_v11 import compress_collinear_path, wrap_angle
from validate_lunar_obstacle_map_v7 import _astar, _cell


DT = .2
MAX_SPEED = .4
MAX_OMEGA = .32
MAX_ACCEL = .25
MAX_ANGULAR_ACCEL = .24


class _Corridor:
    """Cheap spline checks using the conservative raster and exact circles."""

    def __init__(self, world):
        self.world = world
        self.origin = np.array([world.x[0], world.y[0]])
        self.shape = np.array(world.free.shape)

    def _grid_free(self, points):
        cells = np.rint((points - self.origin) / self.world.resolution).astype(int)
        if np.any(cells < 0) or np.any(cells >= self.shape):
            return False
        return bool(np.all(self.world.free[cells[:, 0], cells[:, 1]]))

    def walkable(self, x, y, planning=True):
        return self._grid_free(np.array([[x, y]]))

    def _clear_segment(self, a, b, planning=True, *, margin=0.):
        delta = np.asarray(b) - a
        length_sq = float(delta @ delta)
        count = max(1, int(math.ceil(math.sqrt(length_sq) / .035)))
        points = np.asarray(a) + np.linspace(0., 1., count + 1)[:, None] * delta
        if not self._grid_free(points):
            return False
        centres = self.world.centres - a
        frac = np.clip(centres @ delta / length_sq, 0., 1.) if length_sq > 1e-16 else 0.
        nearest = np.asarray(frac).reshape(-1, 1) * delta
        distances = np.linalg.norm(centres - nearest, axis=1)
        return bool(np.all(distances > self.world.radii + self.world.radius + margin))


def _resample(points, spacing):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    uniform = np.linspace(0., distance[-1], max(2, int(math.ceil(distance[-1] / spacing)) + 1))
    return uniform, np.column_stack([np.interp(uniform, distance, points[:, axis]) for axis in (0, 1)])


def _shortcut(path, corridor, window_m, margin):
    # Include every original corner, so a failed shortcut can keep its next
    # known-valid segment instead of accidentally jumping across a corner.
    pieces = [_resample(np.asarray([a, b]), .12)[1][:-1] for a, b in zip(path[:-1], path[1:])]
    dense = np.vstack([*pieces, path[-1]])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    indices = [0]
    while indices[-1] < len(dense) - 1:
        i = indices[-1]
        end = np.searchsorted(distance, distance[i] + window_m, side='right') - 1
        for j in range(end, i + 1, -1):
            if corridor._clear_segment(dense[i], dense[j], margin=margin):
                break
        else:
            j = i + 1
        indices.append(j)
    return dense[indices]


def _fit_guide(corridor, path, window_m, margin, deviation_m, curvature_limit):
    nodes = _shortcut(path, corridor, window_m, margin)
    _, points = _resample(nodes, .12)
    derivative = np.gradient(points, axis=0)
    yaw = np.arctan2(derivative[:, 1], derivative[:, 0])
    guide, report = smooth_hybrid_path(corridor, np.column_stack([points, yaw]),
        maximum_curvature_per_m=curvature_limit, sample_spacing_m=.035, max_deviation_m=deviation_m)
    report.update(shortcut_window_m=window_m, shortcut_obstacle_margin_m=margin,
                  shortcut_waypoints=len(nodes))
    return guide, report


def _fit_free_tangents(corridor, path, window_m=11., margin=.08):
    """Fit position endpoints while allowing a natural smooth arrival tangent."""
    nodes = _shortcut(path, corridor, window_m, margin)
    distance, points = _resample(nodes, .12)
    u = distance / distance[-1]
    weights = np.ones(len(points))
    weights[[0, -1]] = 1000.
    tree = cKDTree(points)
    report = dict(method='C2 B-spline with fixed endpoint positions and free endpoint tangents',
                  accepted=False, shortcut_waypoints=len(nodes), candidates=[])
    probes = np.linspace(0., 1., int(math.ceil(distance[-1] / .008)) + 1)
    for residual in (2., 1., .5, .25, .1, .05, .02, .005):
        attempt = dict(weighted_squared_residual=residual)
        report['candidates'].append(attempt)
        try:
            spline, _ = splprep(points.T, u=u, w=weights, s=residual, k=3)
            ends = np.asarray(splev([0., 1.], spline)).T
            correction = points[[0, -1]] - ends

            def evaluate(parameters, derivative=0):
                value = np.asarray(splev(parameters, spline, der=derivative)).T
                if derivative == 0:
                    value += (1. - parameters[:, None]) * correction[0] + parameters[:, None] * correction[1]
                elif derivative == 1:
                    value += correction[1] - correction[0]
                return value

            xy = evaluate(probes)
            first, second = evaluate(probes, 1), evaluate(probes, 2)
            speed = np.linalg.norm(first, axis=1)
            if speed.min() < 1e-5 or not np.isfinite(xy).all():
                attempt['rejected'] = 'nonregular_spline'
                continue
            curvature = abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / speed**3
            maximum_curvature = float(curvature.max())
            deviation = float(max(tree.query(xy)[0].max(), cKDTree(xy).query(points)[0].max()))
            attempt.update(maximum_curvature_per_m=maximum_curvature, sampled_hausdorff_distance_m=deviation)
            if maximum_curvature > 8. or deviation > .65:
                attempt['rejected'] = 'curvature_or_deviation_limit'
                continue
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            selected = np.interp(np.linspace(0., arc[-1], int(math.ceil(arc[-1] / .035)) + 1), arc, probes)
            xy, derivative = evaluate(selected), evaluate(selected, 1)
            if not all(corridor.walkable(*point) for point in xy):
                attempt['rejected'] = 'planning_footprint_collision'
                continue
            if not all(corridor._clear_segment(a, b) for a, b in zip(xy[:-1], xy[1:])):
                attempt['rejected'] = 'continuous_segment_collision'
                continue
            yaw = np.arctan2(derivative[:, 1], derivative[:, 0])
            attempt['accepted'] = True
            report.update(accepted=True, selected_weighted_squared_residual=residual,
                          maximum_curvature_per_m=maximum_curvature, path_length_m=float(arc[-1]),
                          sampled_hausdorff_distance_m=deviation, endpoint_tangent_constraints=False)
            return np.column_stack([xy, yaw]), report
        except (ValueError, TypeError) as exc:
            attempt['rejected'] = str(exc)
    return np.empty((0, 3)), report


def _clearance_routes(world, goal):
    clearance = distance_transform_edt(np.pad(world.free, 1))[1:-1, 1:-1] * world.resolution
    x, y = world.x[:, None], world.y[None, :]
    near_start = (x - world.start_xy[0])**2 + (y - world.start_xy[1])**2 < 1.2**2
    near_goal = (x - goal[0])**2 + (y - goal[1])**2 < 1.2**2
    start_cell, goal_cell = _cell(world.start_xy, world.x, world.y), _cell(goal, world.x, world.y)
    for margin in (.45, .32, .24, .65):
        routing_grid = world.free & ((clearance >= margin) | near_start | near_goal)
        cells, _ = _astar(routing_grid, world.height, world.slope, start_cell, goal_cell,
                          world.resolution, world.options)
        if cells is not None:
            route = np.vstack([world.start_xy, np.column_stack([world.x[cells[:, 0]], world.y[cells[:, 1]]]), goal])
            yield compress_collinear_path(route), margin


def _guide_candidates(world, corridor, original, goal):
    for window, margin, deviation in [(11., .065, .24), (6., .035, .20), (3.2, .015, .15)]:
        yield _fit_guide(corridor, original, window, margin, deviation, 2.)
    # A final 10–20cm grid step can point sideways at an otherwise open goal.
    # Arrival yaw is free because the stopped rover aligns to the object next.
    yield _fit_free_tangents(corridor, original)
    # If an efficient grid route hugs rocks too closely to round the corners,
    # try routes with more room rather than relaxing any execution checks.
    for route, route_margin in _clearance_routes(world, goal):
        for window, margin, deviation, curvature in [(11., .16, .32, 2.), (6., .12, .30, 3.5)]:
            guide, report = _fit_guide(corridor, route, window, margin, deviation, curvature)
            report['routing_grid_clearance_m'] = route_margin
            yield guide, report
        guide, report = _fit_free_tangents(corridor, route)
        report['routing_grid_clearance_m'] = route_margin
        yield guide, report


def _arc_controls(curve, positions):
    xy = curve(positions)
    tangent = curve(positions, 1)
    yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    turn = np.diff(yaw)
    chord = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = chord / (DT * np.sinc(turn / (2. * np.pi)))
    # Advance by the midpoint tangent of the exact circular arc. Do not force
    # each XY sample to be hit: enforcing both chord endpoints recursively
    # induces alternating angular commands. Preflight the integrated path.
    midpoint = yaw[:-1] + .5 * turn
    integrated_end = xy[0] + chord @ np.column_stack([np.cos(midpoint), np.sin(midpoint)])
    return np.column_stack([speed, turn / DT]), yaw, integrated_end


def _alignment(delta):
    if abs(delta) < 1e-7:
        return np.empty((0, 2))
    # Quintic orientation has zero rate and acceleration at both boundaries.
    duration = max(.8, 1.875 * abs(delta) / MAX_OMEGA,
                   math.sqrt(5.774 * abs(delta) / MAX_ANGULAR_ACCEL))
    count = int(math.ceil(duration / DT))
    u = np.linspace(0., 1., count + 1)
    angle = delta * u**3 * (10. + u * (-15. + 6. * u))
    return np.column_stack([np.zeros(count), np.diff(angle) / DT])


def _time_parameterize(guide, start_yaw, target_xy, max_steps):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide[:, :2], axis=0), axis=1))]
    tangent = np.column_stack([np.cos(guide[:, 2]), np.sin(guide[:, 2])])
    curve = CubicHermiteSpline(arc, guide[:, :2], tangent, axis=0)
    curvature = abs(np.gradient(np.unwrap(guide[:, 2]), arc))
    speed = np.minimum(MAX_SPEED, .28 / np.maximum(curvature, 1e-5))
    speed[0] = speed[-1] = 0.
    # Spatial acceleration bound, applied in both directions for braking.
    for i, distance in enumerate(np.diff(arc), 1):
        speed[i] = min(speed[i], math.sqrt(speed[i - 1]**2 + 2. * .18 * distance))
    for i in range(len(arc) - 2, -1, -1):
        speed[i] = min(speed[i], math.sqrt(speed[i + 1]**2 + 2. * .18 * (arc[i + 1] - arc[i])))
    time = np.r_[0., np.cumsum(2. * np.diff(arc) / (speed[:-1] + speed[1:]))]
    duration = float(time[-1])
    for _ in range(6):
        count = max(3, int(math.ceil(duration / DT)))
        if count >= max_steps:
            raise ValueError('Safe slew retiming exceeds the allowed step budget')
        positions = np.interp(np.linspace(0., time[-1], count + 1), time, arc)
        movement, yaw, integrated_end = _arc_controls(curve, positions)
        initial_turn = _alignment(wrap_angle(yaw[0] - start_yaw))
        target_delta = target_xy - integrated_end
        final_yaw = math.atan2(target_delta[1], target_delta[0])
        final_turn = _alignment(wrap_angle(final_yaw - yaw[-1]))
        controls = np.vstack([initial_turn, movement, final_turn, [[0., 0.]]])
        acceleration = np.diff(np.vstack([[0., 0.], controls]), axis=0) / DT
        ratio = max(np.max(controls[:, 0]) / MAX_SPEED,
                    np.max(abs(controls[:, 1])) / MAX_OMEGA,
                    math.sqrt(np.max(abs(acceleration[:, 0])) / MAX_ACCEL),
                    math.sqrt(np.max(abs(acceleration[:, 1])) / MAX_ANGULAR_ACCEL))
        if ratio <= 1. + 1e-7 and len(controls) <= max_steps:
            phases = (['align_path_segment'] * len(initial_turn)
                      + ['follow_path_segment'] * len(movement)
                      + ['align_target_at_goal'] * len(final_turn)
                      + ['request_terminal_stop'])
            return controls, phases
        duration *= max(1.015, ratio * 1.025)
    raise ValueError('Could not fit exact arc controls within slew bounds')


def motion_metrics(controls):
    """Compare camera motion through actual integrated command derivatives."""
    controls = np.asarray(controls, float)
    acceleration = np.diff(np.vstack([[0., 0.], controls]), axis=0) / DT
    jerk = np.diff(acceleration[:, 1], prepend=0.) / DT
    pivots = (controls[:, 0] < 1e-8) & (abs(controls[:, 1]) > 1e-6)
    return dict(steps=len(controls), duration_s=len(controls) * DT,
        total_yaw_variation_deg=float(np.degrees(np.sum(abs(controls[:, 1])) * DT)),
        maximum_yaw_step_deg=float(np.degrees(np.max(abs(controls[:, 1])) * DT)),
        maximum_angular_speed_radps=float(np.max(abs(controls[:, 1]))),
        maximum_angular_acceleration_radps2=float(np.max(abs(acceleration[:, 1]))),
        rms_angular_acceleration_radps2=float(np.sqrt(np.mean(acceleration[:, 1]**2))),
        rms_angular_jerk_radps3=float(np.sqrt(np.mean(jerk**2))),
        maximum_linear_acceleration_mps2=float(np.max(abs(acceleration[:, 0]))),
        stationary_turn_steps=int(pivots.sum()),
        stationary_turn_groups=int(np.count_nonzero(pivots & ~np.r_[False, pivots[:-1]])),
        executed_distance_m=float(np.sum(controls[:, 0]) * DT))


@dataclass
class SmoothTeacher:
    path: np.ndarray
    goal: np.ndarray
    controls: np.ndarray
    poses: np.ndarray
    phases: list[str]
    step: int = 0

    def command(self, pose):
        if self.step >= len(self.controls):
            raise RuntimeError('Smooth sequence ended before collector STOP')
        if not np.allclose(pose, self.poses[self.step], atol=1e-7, rtol=0.):
            raise ValueError('Executed body pose diverged from preflighted smooth sequence')
        i = self.step
        self.step += 1
        return self.controls[i].copy(), i == len(self.controls) - 1, dict(phase=self.phases[i])


def _validate_swept(world, controls):
    pose = np.r_[world.start_xy, world.start_yaw]
    poses = [pose.copy()]
    min_circle = min_belly = math.inf
    for step, control in enumerate(controls):
        safe, next_pose, check = world.safe_motion(pose, control)
        v, omega = control
        # Chord-circle clearance minus sagitta conservatively covers the arc
        # between the collision sampling points, as well as the endpoints.
        sagitta = 2. * abs(v / omega) * math.sin(omega * DT / 4.)**2 if abs(omega) > 1e-10 else 0.
        separation = check['minimum_circle_separation_m'] - sagitta
        if not safe or separation <= 0.:
            raise ValueError(f'Unsafe smooth motion at step {step}, xy={pose[:2].tolist()}, circle margin={separation:.5f}m')
        min_circle = min(min_circle, separation)
        min_belly = min(min_belly, check['minimum_belly_clearance_m'])
        pose = next_pose
        poses.append(pose.copy())
    return np.asarray(poses), dict(all_commands_safe=True,
        minimum_conservative_arc_circle_separation_m=min_circle, minimum_belly_clearance_m=min_belly)


def precompute_smooth_teacher(world, target_id, max_steps=5000):
    """Build, retime and verify all controls before committing any capture."""
    original, planning = world.plan(target_id)
    target_xy = np.asarray(world.targets[target_id]['xyz_m'][:2], float)
    goal = original[-1].copy()
    corridor = _Corridor(world)
    attempts = []
    for guide, fit in _guide_candidates(world, corridor, original, goal):
        attempts.append(fit)
        if not fit['accepted']:
            continue
        try:
            controls, phases = _time_parameterize(guide, world.start_yaw, target_xy, max_steps)
            poses, checks = _validate_swept(world, controls)
            endpoint_error = float(np.linalg.norm(poses[-1, :2] - goal))
            target_delta = target_xy - poses[-1, :2]
            heading_error = abs(wrap_angle(math.atan2(target_delta[1], target_delta[0]) - poses[-1, 2]))
            if endpoint_error > .05 or heading_error > .001:
                raise ValueError('Final integrated pose failed position or target-facing gate')
            fit['swept_result'] = checks
            planning.update(controller='safe_spline_exact_unicycle_slew_v12',
                smooth_path_length_m=float(fit['path_length_m']), smoothing_attempts=attempts,
                executed_motion=motion_metrics(controls), original_goal_error_m=endpoint_error,
                terminal_heading_error_rad=heading_error, max_linear_acceleration_mps2=MAX_ACCEL,
                max_angular_acceleration_radps2=MAX_ANGULAR_ACCEL, max_angular_speed_radps=MAX_OMEGA,
                motion_preflight_only=True, visibility_requires_rendered_mask=True)
            return SmoothTeacher(guide[:, :2].copy(), goal, controls, poses, phases), planning
        except ValueError as exc:
            fit['swept_rejected'] = str(exc)
    raise ValueError(f'No safe smooth trajectory for {target_id}: {attempts}')
