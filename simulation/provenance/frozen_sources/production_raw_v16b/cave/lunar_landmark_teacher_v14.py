"""CPU geometric planning for a moving, closer landmark pass on the way to a goal.

This predicts a viewing opportunity, not a rendered detection. Actual RGB-D
salience remains a separate acceptance gate. The scene and the V13 controller
are read only; one global control sequence stops only at the final task goal.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
import numpy as np

from cave import lunar_smooth_teacher_v13 as base
from cave.lunar_trajectory_v11 import compress_collinear_path, wrap_angle


@dataclass(frozen=True)
class LandmarkPassConfig:
    view_range_m: tuple = (4.,6.)
    candidate_view_radii_m: tuple = (5.,5.5,4.5)
    maximum_bearing_deg: float = 25.
    candidate_heading_offsets_deg: tuple = (20.,-20.,0.)
    view_azimuth_step_deg: float = 10.
    entry_distance_m: float = 2.
    exit_distance_m: float = 1.5
    terminal_anchor_distance_m: float = 2.
    routing_clearance_m: float = .24
    maximum_view_proposals: int = 40
    maximum_control_attempts: int = 16
    minimum_contiguous_view_frames: int = 15
    diagnostic_stride: int = 5
    minimum_view_frames_per_stride_phase: int = 3
    camera_hfov_deg: float = 90.
    camera_height_m: float = 1.5
    minimum_goal_progress_cosine: float = 0.
    maximum_added_yaw_deg: float = 180.
    maximum_length_ratio_to_direct_route: float = 1.4


class LandmarkPlanningError(ValueError):
    def __init__(self,message,report):
        super().__init__(message)
        self.report=report


def _routes(world, target, landmark, config):
    originals, corridor = base._route_candidates(world, target)
    if not originals:
        raise ValueError('No original natural final arrival is available')
    baseline_turn=originals[0][2]['polyline_turn_variation_deg']
    baseline_length=originals[0][2]['polyline_length_m']
    terminal_options = []
    for _, nodes, _ in originals:
        goal = nodes[-1]
        delta = goal-nodes[-2]
        heading = delta/np.linalg.norm(delta)
        anchor = goal-config.terminal_anchor_distance_m*heading
        if corridor._clear_segment(anchor,goal,margin=.025):
            identity = tuple(np.round(np.r_[goal,heading],3))
            if not any(row[0]==identity for row in terminal_options):
                terminal_options.append((identity,anchor,goal))
        if len(terminal_options)>=2:
            break
    forward = np.array([math.cos(world.start_yaw),math.sin(world.start_yaw)])
    initial = world.start_xy+forward
    if not corridor._clear_segment(world.start_xy,initial):
        raise ValueError('No safe moving departure')
    centre = np.asarray(landmark['xyz_m'][:2])
    proposals = []
    incoming = math.degrees(math.atan2(*(world.start_xy-centre)[::-1]))
    azimuths = sorted(set(float(x%360) for x in [*np.arange(0.,360.,config.view_azimuth_step_deg),incoming,incoming-15.,incoming+15.]))
    for azimuth in azimuths:
        direction = np.array([math.cos(math.radians(azimuth)),math.sin(math.radians(azimuth))])
        for radius in config.candidate_view_radii_m:
            view = centre+radius*direction
            for offset in config.candidate_heading_offsets_deg:
                angle = math.radians(azimuth+180.+offset)
                tangent = np.array([math.cos(angle),math.sin(angle)])
                goal_delta=np.asarray(target['approach_xy_m'])-view
                if float(goal_delta@tangent)<config.minimum_goal_progress_cosine*np.linalg.norm(goal_delta):
                    continue
                entry, exit = view-config.entry_distance_m*tangent,view+config.exit_distance_m*tangent
                if not corridor._clear_segment(entry,exit,margin=.025):
                    continue
                estimate = np.linalg.norm(entry-initial)+np.linalg.norm(np.asarray(target['approach_xy_m'])-exit)
                proposals.append((float(estimate),azimuth,radius,offset,view,entry,exit))
    candidates = []
    for _,azimuth,radius,offset,view,entry,exit in sorted(proposals,key=lambda row:row[:4])[:config.maximum_view_proposals]:
        prefix = base._grid_path(world,corridor,initial,entry,config.routing_clearance_m)
        if prefix is None:
            continue
        for _,anchor,goal in terminal_options:
            suffix = base._grid_path(world,corridor,exit,anchor,config.routing_clearance_m)
            if suffix is None:
                continue
            nodes = compress_collinear_path(np.vstack([world.start_xy,prefix,exit,suffix[1:],goal]))
            length = float(np.linalg.norm(np.diff(nodes,axis=0),axis=1).sum())
            turn,reversals = base._turns(nodes,world.start_yaw)
            if math.degrees(turn)>baseline_turn+config.maximum_added_yaw_deg or length>baseline_length*config.maximum_length_ratio_to_direct_route:
                continue
            report = dict(view_xy_m=view.tolist(),view_range_m=radius,
                view_heading_offset_deg=offset,view_segment_xy_m=[entry.tolist(),exit.tolist()],
                polyline_length_m=length,polyline_turn_variation_deg=math.degrees(turn),
                polyline_substantial_turn_sign_reversals=reversals)
            candidates.append((length+4.*turn+3.*reversals,nodes,report))
    return sorted(candidates,key=lambda row:row[0]),corridor


def _cave_view_clear(world,landmark,xy,config):
    eye=np.r_[xy,float(world.floor(xy)[0])+config.camera_height_m]
    aim=np.asarray(landmark['xyz_m'],float).copy()
    aim[2]+=float(landmark.get('camera_target_height_m',1.1))
    ray=eye+np.linspace(0.,1.,max(12,int(np.linalg.norm(aim-eye)/.04)))[:,None]*(aim-eye)
    return bool(np.all(world.field(ray)>.025))


def _view_opportunity(world, landmark, poses, controls, config):
    centre = np.asarray(landmark['xyz_m'][:2])
    delta = centre-poses[:, :2]
    distance = np.linalg.norm(delta,axis=1)
    bearing = (np.arctan2(delta[:,1],delta[:,0])-poses[:,2]+np.pi)%(2.*np.pi)-np.pi
    maximum_bearing = min(config.maximum_bearing_deg,config.camera_hfov_deg/2.-10.)
    eligible = (distance>=config.view_range_m[0])&(distance<=config.view_range_m[1])&(abs(bearing)<=math.radians(maximum_bearing))
    eligible[-1] = False
    eligible[:-1] &= controls[:,0]>1e-4
    for index in np.flatnonzero(eligible):
        if not _cave_view_clear(world,landmark,poses[index,:2],config):
            eligible[index] = False
    indices = np.flatnonzero(eligible)
    groups = np.split(indices,np.flatnonzero(np.diff(indices)>1)+1) if len(indices) else []
    longest = max((len(group) for group in groups),default=0)
    phase_counts = [int(np.count_nonzero(eligible[offset::config.diagnostic_stride])) for offset in range(config.diagnostic_stride)]
    if longest<config.minimum_contiguous_view_frames or min(phase_counts)<config.minimum_view_frames_per_stride_phase:
        raise ValueError('No sustained moving 4–6m, central-camera landmark view opportunity')
    best = int(indices[np.argmin(abs(bearing[indices]))])
    return dict(prediction_only=True,rendered_instance_mask_confirmed=False,
        requires_rendered_mask_salience_confirmation=True,landmark_id=landmark['id'],
        eligible_native_frames=len(indices),longest_contiguous_native_frames=longest,
        minimum_stride5_phase_eligible_frames=min(phase_counts),
        first_predicted_view_step=int(indices[0]),best_predicted_view_step=best,
        best_predicted_view_xy_yaw=poses[best].tolist(),best_predicted_view_range_m=float(distance[best]),
        best_predicted_view_bearing_deg=math.degrees(float(bearing[best])),
        closest_route_range_m=float(distance.min()),
        criteria=dict(range_m=config.view_range_m,maximum_camera_relative_bearing_deg=maximum_bearing,
            camera_hfov_deg=config.camera_hfov_deg,camera_height_m=config.camera_height_m,
            cave_field_ray_clear=True,actual_prop_occlusion_mask_and_bbox_unverified=True))


def precompute_landmark_teacher(world,target_id,landmark_id='lunar_rover',max_steps=5000,*,config=None):
    """Return the V13-compatible moving teacher plus explicit view predictions."""
    config = config or LandmarkPassConfig(camera_hfov_deg=getattr(world,'camera_hfov_deg',90.),
        camera_height_m=getattr(world,'camera_height_m',1.5))
    if (not 0<config.view_range_m[0]<config.view_range_m[1] or config.maximum_view_proposals<1
            or config.maximum_view_proposals>128 or not 1<=config.maximum_control_attempts<=64
            or config.diagnostic_stride<1 or not 5<=config.view_azimuth_step_deg<=90
            or not 0<config.maximum_bearing_deg<config.camera_hfov_deg/2.
            or config.camera_height_m<=0 or not 1<=len(config.candidate_view_radii_m)<=6
            or not 1<=len(config.candidate_heading_offsets_deg)<=5
            or any(not config.view_range_m[0]<=r<=config.view_range_m[1] for r in config.candidate_view_radii_m)):
        raise ValueError('Invalid bounded landmark-pass configuration')
    if landmark_id is None:
        rejected = []
        targets = sorted((key for key in world.targets if key!=target_id),key=lambda key:(
            float(np.linalg.norm(np.asarray(world.targets[key]['xyz_m'][:2])-world.start_xy)
                +np.linalg.norm(np.asarray(world.targets[key]['xyz_m'][:2])-world.targets[target_id]['approach_xy_m'])),key))
        for candidate in targets:
            try:
                teacher,planning = precompute_landmark_teacher(world,target_id,candidate,max_steps,config=config)
                planning['landmark_selection_rejections']=rejected
                return teacher,planning
            except ValueError as exc:
                rejected.append(dict(landmark_id=candidate,reason=str(exc)))
        raise LandmarkPlanningError('No feasible non-goal landmark pass',dict(passed=False,stage='landmark_selection',rejections=rejected))
    if target_id==landmark_id:
        raise ValueError('A pass-by landmark must differ from the final goal')
    world.runtime_collision_mode='actual_geometry'
    target,landmark = world.targets[target_id],world.targets[landmark_id]
    canonical = np.asarray(target['approach_xy_m'],float)
    candidates,corridor = _routes(world,target,landmark,config)
    attempts = []
    for score,nodes,route in candidates[:config.maximum_control_attempts]:
        try:
            guide = base._fillet_path(nodes,corridor)
            guide,repulsion = base._local_wall_repulsion(world,guide,corridor)
            controls = base._moving_controls(guide,world.start_yaw,max_steps)
            poses,safety = base._validate_swept(world,controls)
            opportunity = _view_opportunity(world,landmark,poses,controls,config)
            wall = base._wall_clearance_audit(world,poses,controls,corridor)
            error = float(np.linalg.norm(poses[-1,:2]-canonical))
            delta = np.asarray(target['xyz_m'][:2])-poses[-1,:2]
            bearing = abs(wrap_angle(math.atan2(delta[1],delta[0])-poses[-1,2]))
            metrics = base.motion_metrics(controls)
            metrics['substantial_turn_sign_reversals']=base.substantial_turn_reversals(controls)
            metrics['geometric_curvature_sign_reversals_above_6deg']=base.geometric_turn_reversals(guide)
            intermediate_stops = int(np.count_nonzero(np.linalg.norm(controls[:-1],axis=1)<1e-8))
            if error>1. or bearing>base.TARGET_CONE_RAD+.01 or metrics['stationary_turn_steps'] or intermediate_stops:
                raise ValueError('Final cone/region or continuous-moving arrival failed')
            planning = dict(target_id=target_id,controller='landmark_pass_v14',teacher_variant='landmark_pass_v14',
                runtime_collision_mode='actual_geometry',canonical_approach_xy_m=canonical.tolist(),
                original_goal_xy_m=canonical.tolist(),actual_stopping_xy_m=poses[-1,:2].tolist(),
                terminal_region_radius_m=1.,planned_terminal_xy_m=nodes[-1].tolist(),original_goal_error_m=error,
                terminal_target_bearing_rad=bearing,terminal_camera_cone_half_angle_deg=30.,
                stationary_terminal_alignment=False,intermediate_stop_count=intermediate_stops,
                original_geometry_obstacles_retained=True,route_score=score,route=route,
                candidate_rejections=attempts,executed_motion=metrics,safety=safety,path_length_m=metrics['executed_distance_m'],
                body_wall_clearance=wall,local_wall_repulsion=repulsion,landmark_view_prediction=opportunity,
                landmark_pass_config=asdict(config),
                motion_preflight_only=True,visibility_requires_rendered_mask=True)
            phases=['follow_path_segment']*(len(controls)-1)+['request_terminal_stop']
            teacher=base.MovingArrivalTeacher(guide[:,:2].copy(),poses[-1,:2].copy(),controls,poses,phases,canonical_goal=canonical.copy())
            return teacher,planning
        except ValueError as exc:
            attempts.append(dict(route_score=score,reason=str(exc),view_xy_m=route['view_xy_m']))
    raise LandmarkPlanningError('No safe sustained landmark-pass candidate',dict(passed=False,stage='bounded_route_search',
        route_candidates=len(candidates),control_attempts=len(attempts),rejections=attempts,config=asdict(config)))
