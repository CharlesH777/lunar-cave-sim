"""Natural moving arrivals through globally simplified cave routes.

The teacher plans an arrival region around the task's canonical approach. It
keeps the object in the forward camera cone and brakes while already facing
that region. Straight segments join local quintic corner curves with zero
curvature at each join, avoiding global spline overshoot and terminal pivots.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import distance_transform_edt
from scipy.special import comb

from cave.lunar_smooth_teacher_v12 import (
    DT, MAX_ACCEL, MAX_ANGULAR_ACCEL, MAX_OMEGA, MAX_SPEED,
    SmoothTeacher, _Corridor, _arc_controls, _validate_swept, motion_metrics)
from cave.lunar_trajectory_v11 import compress_collinear_path, wrap_angle
from cave.lunar_trajectory_v11 import integrate_unicycle
from validate_lunar_obstacle_map_v7 import _astar, _cell


TERMINAL_RADIUS_M = .95
TARGET_CONE_RAD = math.radians(30.)
MIN_BODY_WALL_GAP_M = 1.0
WALL_ROUTING_PROXY_GAP_M = .45
EXPANDED_BODY_GUARD_M = 1.05
WALL_AUDIT_HEIGHTS_M = (.6, 1., 1.4)


@dataclass
class MovingArrivalTeacher(SmoothTeacher):
    canonical_goal: np.ndarray | None = None


def _wall_solid(world, points, values=None):
    """Identify lateral rock surfaces without classifying floor as a wall."""
    if values is None:
        values = world.field(points)
    solid = np.asarray(values) <= 0.
    wall = np.zeros(len(points), bool)
    if not solid.any():
        return wall
    selected = np.asarray(points)[solid]
    gradient = []
    epsilon = .03
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = epsilon
        gradient.append((world.field(selected + offset) - world.field(selected - offset)) / (2. * epsilon))
    gradient = np.column_stack(gradient)
    wall[solid] = np.linalg.norm(gradient[:, :2], axis=1) > abs(gradient[:, 2])
    return wall


class WallCorridor(_Corridor):
    """Wall clearance is independent of gravel and support/grade constraints."""

    def __init__(self, world):
        super().__init__(world)
        cached = getattr(world, '_wall_gap_v13', None)
        if cached is None:
            # Extend nearby drivable floor heights horizontally across rock.
            # Following the highest wall triangle would climb to its surface
            # and incorrectly label exterior air as the tunnel interior.
            _, nearest = distance_transform_edt(~world.terrain, return_indices=True)
            reference_height = world.height[nearest[0], nearest[1]]
            x, y = np.meshgrid(world.x, world.y, indexing='ij')
            xy = np.column_stack([x.ravel(), y.ravel()])
            heights = reference_height.ravel()
            lateral_wall = np.zeros(len(xy), bool)
            for begin in range(0, len(xy), 12000):
                end = min(begin + 12000, len(xy))
                points = np.repeat(np.column_stack([xy[begin:end], heights[begin:end]]), 3, axis=0)
                points[:, 2] += np.tile([.6, 1., 1.4], end - begin)
                lateral_wall[begin:end] = _wall_solid(world, points).reshape(-1, 3).any(1)
            wall_grid = lateral_wall.reshape(world.terrain.shape)
            distance = distance_transform_edt(np.pad(~wall_grid, 1))[1:-1, 1:-1] * world.resolution
            # One half-cell diagonal covers wall rasterization and another
            # covers a query's offset from the selected grid centre.
            gap = distance - world.radius - math.sqrt(2.) * world.resolution
            cached = gap, wall_grid
            world._wall_gap_v13 = cached
        self.wall_gap, self.wall_grid = cached
        # This raster estimate includes a deliberately large discretization
        # deduction. Use it to reject clearly tight routing candidates; the
        # physical acceptance gate below tests the actual expanded body shell.
        self.allowed = world.free & (self.wall_gap >= WALL_ROUTING_PROXY_GAP_M)

    def gaps(self, points):
        points = np.asarray(points).reshape(-1, 2)
        cells = np.rint((points - self.origin) / self.world.resolution).astype(int)
        result = np.full(len(points), -np.inf)
        inside = np.all((cells >= 0) & (cells < self.shape), axis=1)
        result[inside] = self.wall_gap[cells[inside, 0], cells[inside, 1]]
        return result

    def _grid_free(self, points):
        return super()._grid_free(points) and bool(np.all(self.gaps(points) >= WALL_ROUTING_PROXY_GAP_M))


def _global_pull(points, corridor):
    selected = [0]
    while selected[-1] < len(points) - 1:
        i = selected[-1]
        for j in range(len(points) - 1, i, -1):
            if corridor._clear_segment(points[i], points[j], margin=.025):
                selected.append(j)
                break
        else:
            raise ValueError('No safe visibility edge along the grid route')
    return points[selected]


def _grid_path(world, corridor, start, goal, clearance_m):
    free = corridor.allowed
    if clearance_m:
        clearance = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * world.resolution
        x, y = world.x[:, None], world.y[None, :]
        endpoint_region = ((x - start[0])**2 + (y - start[1])**2 < .85**2)
        endpoint_region |= ((x - goal[0])**2 + (y - goal[1])**2 < .85**2)
        free = free & ((clearance >= clearance_m) | endpoint_region)
    path, report = _astar(free, world.height, world.slope,
        _cell(start, world.x, world.y), _cell(goal, world.x, world.y),
        world.resolution, world.options)
    if path is None:
        return None
    xy = np.column_stack([world.x[path[:, 0]], world.y[path[:, 1]]])
    points = compress_collinear_path(np.vstack([start, xy, goal]))
    pulled = _global_pull(points, corridor)
    return pulled


def _turns(points, start_yaw):
    delta = np.diff(points, axis=0)
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    angles = np.asarray([wrap_angle(a - b) for a, b in zip(heading[1:], heading[:-1])])
    first_error = abs(wrap_angle(heading[0] - start_yaw))
    significant = np.sign(angles[abs(angles) > math.radians(6.)])
    reversals = int(np.count_nonzero(np.diff(significant))) if len(significant) > 1 else 0
    return first_error + float(np.sum(abs(angles))), reversals


def _candidate_goals(world, target):
    canonical = np.asarray(target['approach_xy_m'], float)
    candidates = [canonical]
    for radius in (.85, .45):
        for angle in np.linspace(0., 2. * np.pi, 12, endpoint=False):
            point = canonical + radius * np.array([math.cos(angle), math.sin(angle)])
            if np.linalg.norm(point - np.asarray(target['xyz_m'][:2])) > target['footprint_radius_m'] + world.radius + .04:
                candidates.append(point)
    candidates.sort(key=lambda point: float(np.linalg.norm(point - world.start_xy)) + .10 * np.linalg.norm(point - canonical))
    return candidates


def _target_line_clear(world, target, xy):
    z = float(world.floor(xy)[0]) + 1.5
    aim = np.asarray(target['xyz_m'], float).copy()
    aim[2] += float(target.get('camera_target_height_m', 1.1))
    eye = np.r_[xy, z]
    ray = eye + np.linspace(0., 1., max(12, int(np.linalg.norm(aim - eye) / .04)))[:, None] * (aim - eye)
    return bool(np.all(world.field(ray) > .025))


def _route_candidates(world, target):
    corridor = WallCorridor(world)
    forward = np.array([math.cos(world.start_yaw), math.sin(world.start_yaw)])
    start = world.start_xy
    initial_anchor = start + forward
    if not corridor._clear_segment(start, initial_anchor):
        initial_anchor = start + .5 * forward
    if not corridor._clear_segment(start, initial_anchor):
        raise ValueError('Initial body heading lacks a safe moving departure')
    candidates = []
    seen = set()
    for clearance in (.24, .0):
        for goal in _candidate_goals(world, target):
            if not corridor.walkable(*goal) or not _target_line_clear(world, target, goal):
                continue
            route = _grid_path(world, corridor, initial_anchor, goal, clearance)
            if route is None:
                continue
            direction = goal - route[-2]
            incoming = math.atan2(direction[1], direction[0])
            bearing = math.atan2(target['xyz_m'][1] - goal[1], target['xyz_m'][0] - goal[0])
            error = wrap_angle(bearing - incoming)
            routes = []
            if abs(error) <= TARGET_CONE_RAD:
                routes.append(route)
            else:
                # Choose the closest acceptable heading, leaving useful space
                # inside the 90-degree camera view for the object's width.
                heading = bearing - math.copysign(TARGET_CONE_RAD * .80, error)
                for distance in (2., 1.2):
                    anchor = goal - distance * np.array([math.cos(heading), math.sin(heading)])
                    if not corridor._clear_segment(anchor, goal, margin=.025):
                        continue
                    middle = _grid_path(world, corridor, initial_anchor, anchor, clearance)
                    if middle is not None:
                        routes.append(np.vstack([middle, goal]))
            for route in routes:
                nodes = compress_collinear_path(np.vstack([start, route]))
                identity = tuple(np.round(nodes, 4).ravel())
                if identity in seen:
                    continue
                seen.add(identity)
                length = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum())
                turn, reversals = _turns(nodes, world.start_yaw)
                score = length + 4.0 * turn + 3.0 * reversals
                candidates.append((score, nodes, dict(
                    routing_clearance_m=clearance, visibility_waypoints=len(nodes),
                    polyline_length_m=length, polyline_turn_variation_deg=math.degrees(turn),
                    polyline_substantial_turn_sign_reversals=reversals)))
    return sorted(candidates, key=lambda row: row[0]), corridor


def _bezier(control, u, derivative=0):
    control = np.asarray(control)
    degree = len(control) - 1
    factor = 1.
    for _ in range(derivative):
        factor *= degree
        control = np.diff(control, axis=0)
        degree -= 1
    basis = np.column_stack([comb(degree, i) * u**i * (1. - u)**(degree - i)
                             for i in range(degree + 1)])
    return factor * basis @ control


def _corner(control, spacing=.012):
    polygon_length = np.linalg.norm(np.diff(control, axis=0), axis=1).sum()
    u = np.linspace(0., 1., max(20, int(math.ceil(polygon_length / spacing))) + 1)
    xy = _bezier(control, u)
    tangent = _bezier(control, u, 1)
    return xy, np.arctan2(tangent[:, 1], tangent[:, 0])


def _fillet_path(nodes, corridor):
    corners = []
    for previous, point, following in zip(nodes[:-2], nodes[1:-1], nodes[2:]):
        incoming = point - previous
        outgoing = following - point
        before, after = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        incoming, outgoing = incoming / before, outgoing / after
        turn = math.atan2(np.cross(incoming, outgoing), incoming @ outgoing)
        if abs(turn) < 1e-5:
            corners.append((point[None], np.array([math.atan2(incoming[1], incoming[0])])))
            continue
        maximum = min(2.5, .44 * before, .44 * after)
        for scale in (1., .8, .6, .4, .25, .15, .08, .04):
            distance = maximum * scale
            entry, exit = point - distance * incoming, point + distance * outgoing
            control = np.asarray([entry, entry + .4 * distance * incoming,
                entry + .8 * distance * incoming, exit - .8 * distance * outgoing,
                exit - .4 * distance * outgoing, exit])
            xy, yaw = _corner(control)
            if all(corridor.walkable(*sample) for sample in xy) and all(
                    corridor._clear_segment(a, b) for a, b in zip(xy[:-1], xy[1:])):
                corners.append((xy, yaw))
                break
        else:
            raise ValueError('No collision-free continuous-curvature corner fit')
    pieces = []
    point = nodes[0]
    for xy, yaw in corners:
        delta = xy[0] - point
        length = float(np.linalg.norm(delta))
        if length > 1e-8:
            straight = point + np.linspace(0., 1., max(2, int(math.ceil(length / .025))) + 1)[:, None] * delta
            angle = math.atan2(delta[1], delta[0])
            pieces.append(np.column_stack([straight[:-1], np.full(len(straight) - 1, angle)]))
        pieces.append(np.column_stack([xy[:-1], yaw[:-1]]))
        point = xy[-1]
    delta = nodes[-1] - point
    length = float(np.linalg.norm(delta))
    straight = point + np.linspace(0., 1., max(2, int(math.ceil(length / .025))) + 1)[:, None] * delta
    angle = math.atan2(delta[1], delta[0])
    pieces.append(np.column_stack([straight, np.full(len(straight), angle)]))
    result = np.vstack(pieces)
    keep = np.r_[True, np.linalg.norm(np.diff(result[:, :2], axis=0), axis=1) > 1e-8]
    return result[keep]


def _moving_controls(guide, start_yaw, max_steps):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide[:, :2], axis=0), axis=1))]
    yaw = np.unwrap(guide[:, 2])
    if abs(wrap_angle(yaw[0] - start_yaw)) > 1e-5:
        raise ValueError('Guide departure does not match initial body heading')
    tangent = np.column_stack([np.cos(yaw), np.sin(yaw)])
    curve = CubicHermiteSpline(arc, guide[:, :2], tangent, axis=0)
    curvature = np.gradient(yaw, arc)
    curvature_rate = np.gradient(curvature, arc)
    speed = np.minimum(MAX_SPEED, .27 / np.maximum(abs(curvature), 1e-5))
    speed = np.minimum(speed, np.sqrt(.12 / np.maximum(abs(curvature_rate), 1e-5)))
    local_accel = np.minimum(.16, .075 / np.maximum(abs(curvature), 1e-5))
    speed[0] = speed[-1] = 0.
    for i, distance in enumerate(np.diff(arc), 1):
        accel = min(local_accel[i - 1], local_accel[i])
        speed[i] = min(speed[i], math.sqrt(speed[i - 1]**2 + 2. * accel * distance))
    for i in range(len(arc) - 2, -1, -1):
        accel = min(local_accel[i], local_accel[i + 1])
        speed[i] = min(speed[i], math.sqrt(speed[i + 1]**2 + 2. * accel * (arc[i + 1] - arc[i])))
    time = np.r_[0., np.cumsum(2. * np.diff(arc) / (speed[:-1] + speed[1:]))]
    duration = time[-1]
    for _ in range(7):
        count = int(math.ceil(duration / DT))
        if count + 1 > max_steps:
            raise ValueError('Smooth moving plan exceeds step cap')
        positions = np.interp(np.linspace(0., time[-1], count + 1), time, arc)
        controls, _, _ = _arc_controls(curve, positions)
        controls = np.vstack([controls, [0., 0.]])
        acceleration = np.diff(np.vstack([[0., 0.], controls]), axis=0) / DT
        ratio = max(np.max(controls[:, 0]) / MAX_SPEED,
            np.max(abs(controls[:, 1])) / MAX_OMEGA,
            math.sqrt(np.max(abs(acceleration[:, 0])) / MAX_ACCEL),
            math.sqrt(np.max(abs(acceleration[:, 1])) / MAX_ANGULAR_ACCEL))
        if ratio <= 1. + 1e-7:
            return controls
        duration *= max(1.02, ratio * 1.03)
    raise ValueError('Moving arrival could not satisfy slew limits')


def substantial_turn_reversals(controls, threshold_deg=6.):
    angles = np.asarray(controls)[:, 1] * DT
    significant = abs(np.asarray(controls)[:, 1]) > .02
    groups = []
    current_sign = 0
    total = 0.
    for angle, keep in zip(angles, significant):
        sign = int(np.sign(angle)) if keep else 0
        if sign != current_sign and sign:
            if current_sign and abs(total) >= math.radians(threshold_deg):
                groups.append(current_sign)
            current_sign, total = sign, 0.
        if sign:
            total += angle
    if current_sign and abs(total) >= math.radians(threshold_deg):
        groups.append(current_sign)
    return int(np.count_nonzero(np.diff(groups))) if len(groups) > 1 else 0


def geometric_turn_reversals(guide, threshold_deg=6.):
    change = np.diff(np.unwrap(guide[:, 2]))
    groups = []
    current_sign = 0
    total = 0.
    for angle in change:
        if abs(angle) < 1e-9:
            continue
        sign = int(np.sign(angle))
        if sign != current_sign:
            if current_sign and abs(total) >= math.radians(threshold_deg):
                groups.append(current_sign)
            current_sign, total = sign, 0.
        total += angle
    if current_sign and abs(total) >= math.radians(threshold_deg):
        groups.append(current_sign)
    return int(np.count_nonzero(np.diff(groups))) if len(groups) > 1 else 0


def _local_wall_repulsion(world, guide, corridor):
    gaps = corridor.gaps(guide[:, :2])
    index = int(gaps.argmin())
    if gaps[index] >= EXPANDED_BODY_GUARD_M:
        return guide, dict(applied=False)
    xy = guide[index, :2]
    ground = float(world.floor(xy)[0])
    angle = np.linspace(0., 2. * np.pi, 180, endpoint=False)
    ring = xy + (world.radius + EXPANDED_BODY_GUARD_M) * np.column_stack([np.cos(angle), np.sin(angle)])
    probes = np.repeat(np.column_stack([ring, np.full(len(ring), ground)]), len(WALL_AUDIT_HEIGHTS_M), axis=0)
    probes[:, 2] += np.tile(WALL_AUDIT_HEIGHTS_M, len(ring))
    values = world.field(probes)
    wall = _wall_solid(world, probes, values)
    if not wall.any():
        return guide, dict(applied=False, reason='conservative raster estimate; actual expanded shell is clear')
    worst = np.flatnonzero(wall)[int(np.argmin(values[wall]))]
    point = probes[worst]
    normal = []
    for axis in (0, 1):
        offset = np.zeros(3)
        offset[axis] = .03
        normal.append(float((world.field((point + offset)[None])[0] - world.field((point - offset)[None])[0]) / .06))
    normal = np.asarray(normal)
    normal /= np.linalg.norm(normal)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide[:, :2], axis=0), axis=1))]
    low, high = max(0., arc[index] - 5.), min(arc[-1], arc[index] + 5.)
    parameter = np.clip((arc - low) / (high - low), 0., 1.)
    weight = np.sin(np.pi * parameter)**4
    derivative = 4. * np.sin(np.pi * parameter)**3 * np.cos(np.pi * parameter) * np.pi / (high - low)
    derivative[(arc <= low) | (arc >= high)] = 0.
    displacement = .15
    shifted = guide[:, :2] + displacement * weight[:, None] * normal
    tangent = np.column_stack([np.cos(guide[:, 2]), np.sin(guide[:, 2])])
    tangent += displacement * derivative[:, None] * normal
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    return np.column_stack([shifted, yaw]), dict(applied=True,
        maximum_offset_m=displacement, centre_xy_m=xy.tolist(), away_from_wall_unit_xy=normal.tolist(),
        support_arclength_m=[low, high], method='compact sin^4 offset with zero endpoint derivatives; original terminal and departure retained')


def _wall_clearance_audit(world, poses, controls, corridor):
    samples = []
    for pose, control in zip(poses[:-1], controls):
        count = max(2, int(math.ceil(abs(control[0]) * DT / .025)) + 1)
        samples.extend([pose if t == 0. else integrate_unicycle(pose, control, t)
                        for t in np.linspace(0., DT, count)])
    samples = np.asarray(samples)
    gaps = corridor.gaps(samples[:, :2])
    minimum = float(gaps.min())
    angle = np.linspace(0., 2. * np.pi, 180, endpoint=False)
    ring = (world.radius + EXPANDED_BODY_GUARD_M) * np.column_stack([np.cos(angle), np.sin(angle)])
    for begin in range(0, len(samples), 64):
        xy = samples[begin:begin + 64, :2]
        ground = world.floor(xy)
        boundary = xy[:, None, :] + ring
        points = np.repeat(np.column_stack([boundary.reshape(-1, 2), np.repeat(ground, len(ring))]), len(WALL_AUDIT_HEIGHTS_M), axis=0)
        points[:, 2] += np.tile(WALL_AUDIT_HEIGHTS_M, len(xy) * len(ring))
        if _wall_solid(world, points).any():
            raise ValueError('Expanded 1.05m body envelope intersects a lateral cave wall')
    index = int(gaps.argmin())
    return dict(minimum_body_wall_clearance_raster_lower_bound_m=minimum,
        required_gap_m=MIN_BODY_WALL_GAP_M, passed=True,
        minimum_required_body_wall_gap_m=MIN_BODY_WALL_GAP_M,
        minimum_at_xy_m=samples[index, :2].tolist(),
        definition='distance to lateral cave-air boundary minus 0.65m circumscribed body radius and one full raster-cell diagonal',
        acceptance='actual expanded body shell at 1.05m sampled every <=2.5cm and 2 degrees, with lateral-wall gradient classification',
        raster_scope='conservative routing diagnostic, not the physical acceptance threshold',
        raster_resolution_m=world.resolution, raster_uncertainty_deducted_m=math.sqrt(2.) * world.resolution,
        wall_classifier='implicit cave field solid with horizontal gradient norm greater than absolute vertical derivative; floor-like surfaces excluded',
        rubble_in_wall_distance=False, original_rubble_collision_checks_retained=True,
        expanded_air_envelope_gap_m=EXPANDED_BODY_GUARD_M, expanded_air_ring_angle_step_deg=2.,
        expanded_air_heights_above_body_floor_m=list(WALL_AUDIT_HEIGHTS_M),
        swept_sample_count=len(samples), expanded_air_envelope_passed=True)


def precompute_smooth_teacher(world, target_id, max_steps=5000):
    world.runtime_collision_mode = 'actual_geometry'
    target = world.targets[target_id]
    canonical = np.asarray(target['approach_xy_m'], float)
    candidates, corridor = _route_candidates(world, target)
    attempts = []
    for score, nodes, route_report in candidates:
        try:
            guide = _fillet_path(nodes, corridor)
            guide, wall_repulsion = _local_wall_repulsion(world, guide, corridor)
            controls = _moving_controls(guide, world.start_yaw, max_steps)
            poses, safety = _validate_swept(world, controls)
            wall_report = _wall_clearance_audit(world, poses, controls, corridor)
            terminal = poses[-1]
            error = float(np.linalg.norm(terminal[:2] - canonical))
            bearing = math.atan2(target['xyz_m'][1] - terminal[1], target['xyz_m'][0] - terminal[0])
            cone_error = abs(wrap_angle(bearing - terminal[2]))
            if error > 1. or cone_error > TARGET_CONE_RAD + .01:
                raise ValueError('Integrated arrival left its canonical region or forward target cone')
            metrics = motion_metrics(controls)
            metrics['substantial_turn_sign_reversals'] = substantial_turn_reversals(controls)
            metrics['geometric_curvature_sign_reversals_above_6deg'] = geometric_turn_reversals(guide)
            if metrics['stationary_turn_steps']:
                raise ValueError('Moving arrival contains an unexpected stationary turn')
            phases = ['follow_path_segment'] * (len(controls) - 1) + ['request_terminal_stop']
            planning = dict(target_id=target_id, controller='global_visibility_quintic_moving_arrival_v13',
                runtime_collision_mode='actual_geometry',
                canonical_approach_xy_m=canonical.tolist(), original_goal_xy_m=canonical.tolist(),
                actual_stopping_xy_m=terminal[:2].tolist(), terminal_region_radius_m=1.,
                planned_terminal_xy_m=nodes[-1].tolist(), original_goal_error_m=error,
                terminal_target_bearing_rad=cone_error, terminal_camera_cone_half_angle_deg=30.,
                terminal_heading_rule='target inside the forward cone during moving deceleration',
                goal_is_target_approach_region=True, visibility_requires_rendered_mask=True,
                original_geometry_obstacles_retained=True, route_score=score,
                route=route_report, candidate_rejections=attempts, executed_motion=metrics,
                safety=safety, path_length_m=metrics['executed_distance_m'],
                body_wall_clearance=wall_report,
                local_wall_repulsion=wall_repulsion,
                geometric_detour_ratio=float(metrics['executed_distance_m'] / np.linalg.norm(terminal[:2] - world.start_xy)),
                stationary_terminal_alignment=False, motion_preflight_only=True)
            teacher = MovingArrivalTeacher(guide[:, :2].copy(), terminal[:2].copy(),
                controls, poses, phases, canonical_goal=canonical.copy())
            return teacher, planning
        except ValueError as exc:
            attempts.append(dict(route_score=score, reason=str(exc)))
    raise ValueError('No smooth moving-arrival candidate passed: ' + str(attempts))
