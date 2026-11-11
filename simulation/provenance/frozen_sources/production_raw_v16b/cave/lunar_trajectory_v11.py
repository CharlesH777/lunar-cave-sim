"""Collision-checked continuous kinematic teachers for authored lunar scenes.

No renderer or learned model is imported here.  Commands are integrated once
at dt=.2s; an ordered A* path is a teacher plan, never a sequence of teleports.
The circumscribed vehicle radius makes finite-rate in-place turns conservative.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from validate_lunar_obstacle_map_v7 import (
    DEFAULTS, _astar, _cell, _obstacle_masks, _path_geometry, _point, _read_obstacles)


def wrap_angle(angle):
    return (float(angle)+math.pi) % (2.*math.pi)-math.pi


def integrate_unicycle(pose, control, dt=.2):
    """Exact SE(2) constant-command integration; yaw is kept continuous."""
    pose=np.asarray(pose,float);control=np.asarray(control,float)
    if pose.shape!=(3,) or control.shape!=(2,) or not np.isfinite(np.r_[pose,control,dt]).all() or dt<=0:
        raise ValueError('Finite pose[3], control[v,w] and positive dt required')
    x,y,yaw=pose;v,w=control;turn=w*dt
    if abs(w)<1e-10:
        return np.array([x+v*dt*math.cos(yaw),y+v*dt*math.sin(yaw),yaw])
    return np.array([x+v/w*(math.sin(yaw+turn)-math.sin(yaw)),
                     y+v/w*(math.cos(yaw)-math.cos(yaw+turn)),yaw+turn])


def compress_collinear_path(points):
    """Remove only exactly collinear A* runs; no corner is cut or smoothed."""
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all() or len(points)<2:
        raise ValueError('Finite Nx2 path with at least two points required')
    points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-9]]
    if len(points)<2:raise ValueError('Path has no length')
    keep=[0]
    for i in range(1,len(points)-1):
        a=points[i]-points[i-1];b=points[i+1]-points[i]
        cross=a[0]*b[1]-a[1]*b[0]
        if abs(cross)>1e-9 or float(a@b)<=0:keep.append(i)
    return points[keep+[len(points)-1]].copy()


@dataclass
class SegmentTeacher:
    """Finite-rate pivot/straight controller with exact fractional last commands.

    This deliberate conservative controller is not a suspension or tyre model.
    It follows each safe grid segment and then aligns at the actual approach
    point. It never snaps pose to a waypoint or to the target centre.
    """
    path: np.ndarray
    target_xy: np.ndarray
    dt: float=.2
    max_v: float=.4
    max_w: float=.8
    yaw_tolerance: float=1e-6
    waypoint_tolerance: float=1e-7

    def __post_init__(self):
        self.path=compress_collinear_path(self.path)
        self.target_xy=np.asarray(self.target_xy,float)
        if self.target_xy.shape!=(2,) or not np.isfinite(self.target_xy).all():raise ValueError('Invalid target centre')
        if self.dt!=.2 or not 0<self.max_v<=.4 or not 0<self.max_w<=.8:raise ValueError('Teacher units/bounds do not match protocol')
        self.waypoint=1

    @property
    def goal(self):return self.path[-1].copy()

    def command(self,pose):
        pose=np.asarray(pose,float)
        while self.waypoint<len(self.path) and np.linalg.norm(self.path[self.waypoint]-pose[:2])<=self.waypoint_tolerance:
            self.waypoint+=1
        if self.waypoint<len(self.path):
            delta=self.path[self.waypoint]-pose[:2];distance=float(np.linalg.norm(delta))
            desired=math.atan2(delta[1],delta[0]);error=wrap_angle(desired-pose[2])
            if abs(error)>self.yaw_tolerance:
                return np.array([0.,np.clip(error/self.dt,-self.max_w,self.max_w)]),False,dict(phase='align_path_segment',waypoint=self.waypoint)
            return np.array([min(self.max_v,distance/self.dt),0.]),False,dict(phase='follow_path_segment',waypoint=self.waypoint)
        delta=self.target_xy-pose[:2]
        desired=pose[2] if np.linalg.norm(delta)<1e-8 else math.atan2(delta[1],delta[0])
        error=wrap_angle(desired-pose[2])
        if abs(error)>self.yaw_tolerance:
            return np.array([0.,np.clip(error/self.dt,-self.max_w,self.max_w)]),False,dict(phase='align_target_at_goal',waypoint=self.waypoint)
        return np.array([0.,0.]),True,dict(phase='request_terminal_stop',waypoint=self.waypoint)


def make_supervision(controls,stop,dt=.2,horizon=8,phase=None):
    controls=np.asarray(controls,np.float32);raw_stop=np.asarray(stop)
    if controls.ndim!=2 or controls.shape[1]!=2 or raw_stop.shape!=(len(controls),):raise ValueError('Bad control/STOP arrays')
    if dt!=.2 or horizon!=8:raise ValueError('Protocol requires dt=.2 seconds and horizon=8')
    if not np.isfinite(controls).all() or np.any(controls[:,0]<0) or np.any(controls[:,0]>.400001) or np.any(abs(controls[:,1])>.800001):raise ValueError('Physical control bounds violated')
    if np.any((raw_stop!=0)&(raw_stop!=1)):raise ValueError('STOP must be binary before uint8 conversion')
    stop=raw_stop.astype(np.uint8)
    if stop.sum() and not (stop.sum()==1 and stop[-1]==1 and np.allclose(controls[-1],0.)):
        raise ValueError('Only the final stationary control can carry STOP')
    n=len(controls);future=np.zeros((n,horizon,2),np.float32);valid=np.zeros((n,horizon),bool)
    for i in range(n):
        available=min(horizon,n-i);future[i,:available]=controls[i:i+available];valid[i,:available]=True
    result=dict(vw_si=controls,future_vw_si=future,horizon_valid=valid,stop=stop,
        dt_s=np.full(n,dt,np.float32),observation_time_s=np.arange(n,dtype=np.float64)*dt)
    if phase is not None:
        if np.asarray(phase).shape!=(n,):raise ValueError('Bad phase shape')
        result['phase']=np.asarray(phase,np.uint8)
    return result


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class TerrainWorld:
    """Read-only actual terrain, static obstacle footprints and teacher planner."""
    def __init__(self,scene,*,runtime_collision_mode='planning_grid_and_geometry'):
        if runtime_collision_mode not in ('planning_grid_and_geometry', 'actual_geometry'):
            raise ValueError('Unknown runtime collision mode')
        self.runtime_collision_mode = runtime_collision_mode
        self.scene=Path(scene).expanduser().resolve(strict=True)
        self.config=json.loads((self.scene/'scene_config.json').read_text())
        self.targets={t['id']:t for t in json.loads((self.scene/'targets.json').read_text())['targets']}
        names=['cave.usdc','scene_config.json','mesh.npz','boulders.json','obstacle_report.json','targets.json','task_obstacles.json']
        self.source_hashes={name:sha256(self.scene/name) for name in names}
        from cave.lunar_gravel_v5 import _FloorProjector
        from build_lunar_cave_v4_cpu import evaluate_scene_field
        with np.load(self.scene/'mesh.npz') as d:self.projector=_FloorProjector(d['vertices'],d['faces'])
        self.field=lambda p:np.asarray(evaluate_scene_field(self.config,p),float)
        self.obstacles,self.obstacle_metadata=_read_obstacles(self.scene)
        extras=json.loads((self.scene/'task_obstacles.json').read_text())
        if len(extras)!=len(self.targets):raise ValueError('Target collision metadata is incomplete')
        self.obstacles+=extras
        self.centres=np.asarray([r['center_xy_m'] for r in self.obstacles],float)
        self.radii=np.asarray([r['xy_radius_m'] for r in self.obstacles],float)
        self.options=dict(DEFAULTS);self.options.update(self.config.get('obstacle_map_validation',{}))
        self.resolution=float(self.options['grid_resolution_m']);self.radius=float(self.options['vehicle_radius_m'])
        self.cache_source=None
        for path in [self.scene/'validation/obstacle_map_report.json',self.scene/'validation_branch/obstacle_map_report.json']:
            if not path.is_file():continue
            report=json.loads(path.read_text());archive=path.parent/'grid_data.npz'
            if not report.get('passed') or not archive.is_file():continue
            if not all(report.get('source_hashes',{}).get(str(self.scene/name))==digest for name,digest in self.source_hashes.items()):continue
            with np.load(archive) as d:
                self.x,self.y,self.height,self.slope,self.terrain=[np.array(d[k],copy=True) for k in ['x','y','height','slope_deg','terrain_free']]
            self.cache_source=str(archive);break
        if self.cache_source is None:self._fresh_grid()
        inflation=self.radius+(self.resolution/math.sqrt(2.) if self.options['grid_cell_margin'] else 0.)
        _,mask,_=_obstacle_masks(self.obstacles,self.x,self.y,inflation)
        self.free=self.terrain&~mask
        self.start_xy=_point(self.config,'task_spawn',[-15.,-3.])
        self.start_yaw=float(self.config.get('task_spawn',{}).get('yaw_rad',0.))
        self._wheel_offsets=np.array([[.26,.33],[.26,-.33],[-.26,.33],[-.26,-.33]])
        self._wheel_design=np.column_stack([np.ones(4),self._wheel_offsets])
        self._wheel_pinv=np.linalg.pinv(self._wheel_design)
        gx,gy=np.meshgrid(np.linspace(-.4,.4,9),np.linspace(-.26,.26,7),indexing='ij')
        self._belly=np.column_stack([gx.ravel(),gy.ravel()])
        gx,gy=np.meshgrid(np.linspace(-.475,.475,5),np.linspace(-.425,.425,5),indexing='ij')
        self._envelope=np.column_stack([gx.ravel(),gy.ravel()])

    def _fresh_grid(self):
        low,high=np.asarray(self.options['grid_bounds_xy_m'],float);r=self.resolution
        self.x=low[0]+np.arange(int(np.floor((high[0]-low[0])/r))+1)*r
        self.y=low[1]+np.arange(int(np.floor((high[1]-low[1])/r))+1)*r
        xx,yy=np.meshgrid(self.x,self.y,indexing='ij');xy=np.column_stack([xx.ravel(),yy.ravel()])
        z,n,face,_=self.projector.project(xy);z=z.reshape(xx.shape);n=n.reshape(*xx.shape,3)
        supported=np.isfinite(z)&(face.reshape(xx.shape)>=0)
        slope=np.full(xx.shape,np.nan);slope[supported]=np.degrees(np.arccos(np.clip(n[supported,2],-1,1)))
        indices=np.flatnonzero(supported.ravel());xyz=np.column_stack([xy[indices],z.ravel()[indices]])
        probes=np.repeat(xyz,3,axis=0);probes[:,2]+=np.tile([.3,.8,1.2],len(xyz))
        values=np.concatenate([self.field(probes[i:i+65536]).reshape(-1) for i in range(0,len(probes),65536)]).reshape(-1,3)
        air=np.zeros(xx.shape,bool);air.ravel()[indices]=(np.isfinite(values)&(values>0)).all(1)
        local=supported&air&(slope<=self.options['max_local_grade_deg']+1e-7)
        inflation=self.radius+(r/math.sqrt(2.) if self.options['grid_cell_margin'] else 0.)
        distance=distance_transform_edt(np.pad(local,1,constant_values=False))[1:-1,1:-1]*r
        self.height,self.slope,self.terrain=z,slope,local&(distance>=inflation-1e-8)

    def floor(self,xy):
        points=np.asarray(xy,float).reshape(-1,2);z,_,face,_=self.projector.project(points)
        if not np.isfinite(z).all() or np.any(face<0):raise ValueError('No actual floor support')
        return z

    def plan(self,target_id):
        if target_id not in self.targets:raise ValueError('Unknown target id')
        target=self.targets[target_id];goal=np.asarray(target['approach_xy_m'],float)
        if np.linalg.norm(goal-np.asarray(target['xyz_m'][:2]))<=float(target['footprint_radius_m'])+self.radius:
            raise ValueError('Target approach point enters its physical footprint')
        search_free=self.free.copy();attempts=[]
        for attempt in range(int(self.options['max_search_attempts'])):
            grid_path,search=_astar(search_free,self.height,self.slope,_cell(self.start_xy,self.x,self.y),
                _cell(goal,self.x,self.y),self.resolution,self.options)
            if grid_path is None:raise ValueError('Target approach has no path through the preserved obstacles: '+str(search))
            points=np.vstack([self.start_xy,np.column_stack([self.x[grid_path[:,0]],self.y[grid_path[:,1]]]),goal])
            post,dense,bad,_=_path_geometry(points,self.projector,self.field,self.options)
            attempts.append(dict(attempt=attempt+1,search=search,geometry=post))
            if post['passed']:
                path=compress_collinear_path(points)
                return path,dict(target_id=target_id,original_goal_xy_m=goal.tolist(),goal_is_target_approach=True,
                    goal_projected=False,path_length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                    waypoints=len(path),attempts=attempts,grid_cache=self.cache_source)
            for point in dense[bad]:
                cell=_cell(point,self.x,self.y)
                if cell is not None:search_free[cell]=False
        raise ValueError('No body-safe route after the bounded geometric refinements')

    def pose_checks(self,poses):
        poses=np.asarray(poses,float).reshape(-1,3);xy=poses[:,:2];yaw=poses[:,2]
        forward=np.column_stack([np.cos(yaw),np.sin(yaw)]);left=np.column_stack([-np.sin(yaw),np.cos(yaw)])
        def world(offset):return xy[:,None,:]+offset[None,:,0,None]*forward[:,None,:]+offset[None,:,1,None]*left[:,None,:]
        indices=np.rint((xy-np.array([self.x[0],self.y[0]]))/self.resolution).astype(int)
        in_grid=(indices[:,0]>=0)&(indices[:,0]<len(self.x))&(indices[:,1]>=0)&(indices[:,1]<len(self.y))
        free=np.zeros(len(poses),bool);free[in_grid]=self.free[indices[in_grid,0],indices[in_grid,1]]
        obstacle_clear=(np.linalg.norm(xy[:,None,:]-self.centres[None,:,:],axis=2)>self.radii[None,:]+self.radius-1e-8).all(1)
        wheels=world(self._wheel_offsets);wz,_,wf,_=self.projector.project(wheels.reshape(-1,2));wz=wz.reshape(-1,4)
        supported=np.isfinite(wz).all(1)&(wf.reshape(-1,4)>=0).all(1)
        plane=wz@self._wheel_pinv.T;residual=np.abs(wz-plane@self._wheel_design.T).max(1)
        pitch=np.degrees(np.arctan(plane[:,1]));roll=np.degrees(np.arctan(plane[:,2]))
        belly=world(self._belly);bz,_,bf,_=self.projector.project(belly.reshape(-1,2));bz=bz.reshape(len(poses),-1)
        bp=plane[:,0,None]+plane[:,1,None]*self._belly[None,:,0]+plane[:,2,None]*self._belly[None,:,1]
        clearance=(bp+.15-bz).min(1);belly_support=np.isfinite(bz).all(1)&(bf.reshape(len(poses),-1)>=0).all(1)
        env=world(self._envelope);ep=plane[:,0,None]+plane[:,1,None]*self._envelope[None,:,0]+plane[:,2,None]*self._envelope[None,:,1]
        pxy=np.repeat(env,3,axis=1);pz=np.repeat(ep,3,axis=1)+np.tile([.3,.8,1.2],len(self._envelope))[None,:]
        probes=np.concatenate([pxy,pz[:,:,None]],axis=2).reshape(-1,3);values=self.field(probes).reshape(len(poses),-1)
        air=np.isfinite(values).all(1)&(values>0).all(1)
        geometry_good=obstacle_clear&supported&belly_support&air&(abs(pitch)<=8.+1e-7)&(abs(roll)<=8.+1e-7)&(residual<=.025+1e-7)&(clearance>=.10-1e-7)
        # Grid cells include an extra inflation for planning. Exact circle,
        # wheel, belly and air checks decide actual continuous motion in V13.
        routing_gate=free if self.runtime_collision_mode == 'planning_grid_and_geometry' else in_grid
        good=routing_gate&geometry_good
        return good,dict(wheel_z=wz,max_abs_pitch_deg=float(np.nanmax(abs(pitch))),
            max_abs_roll_deg=float(np.nanmax(abs(roll))),minimum_belly_clearance_m=float(np.nanmin(clearance)),
            max_contact_plane_residual_m=float(np.nanmax(residual)),sample_count=len(poses),
            invalid_pose_count=int((~good).sum()),runtime_collision_mode=self.runtime_collision_mode,
            planning_grid_rejected_pose_count=int((~free).sum()),
            actual_geometry_invalid_pose_count=int((~geometry_good).sum()))

    def safe_motion(self,pose,control,dt=.2):
        pose=np.asarray(pose,float);control=np.asarray(control,float)
        distance=abs(float(control[0]))*dt;angle=abs(float(control[1]))*dt
        count=max(2,int(math.ceil(distance/.02))+1,int(math.ceil(angle/.04))+1)
        times=np.linspace(0.,dt,count)
        samples=np.asarray([pose if t==0 else integrate_unicycle(pose,control,t) for t in times])
        good,report=self.pose_checks(samples)
        wheel_step=float(np.max(np.abs(report['wheel_z'][-1]-report['wheel_z'][0])))
        delta=samples[-1,:2]-pose[:2];norm=float(delta@delta)
        projection=np.zeros(len(self.centres)) if norm<1e-20 else np.clip((self.centres-pose[:2])@delta/norm,0.,1.)
        separation=np.linalg.norm(self.centres-(pose[:2]+projection[:,None]*delta),axis=1)-self.radii-self.radius
        exact_circle_segment_clear=bool(np.all(separation>0.))
        accepted=bool(good.all() and wheel_step<=.03+1e-7 and exact_circle_segment_clear)
        report={k:v for k,v in report.items() if k!='wheel_z'}
        report.update(passed=accepted,maximum_control_wheel_height_change_m=wheel_step,
            circle_segment_clear=exact_circle_segment_clear,minimum_circle_separation_m=float(separation.min()),
            method='exact straight-segment circle test; sampled actual floor/air/vehicle poses',
            mixed_arc_note='collector controller uses v=0 or omega=0; generic mixed arcs are only discretely checked')
        return accepted,samples[-1],report

    def verify_sources(self):return all(sha256(self.scene/name)==digest for name,digest in self.source_hashes.items())


def simulate_plan(world,target_id,max_steps=5000):
    path,planning=world.plan(target_id);target=world.targets[target_id]
    teacher=SegmentTeacher(path,np.asarray(target['xyz_m'][:2],float))
    pose=np.r_[world.start_xy,world.start_yaw];poses=[pose.copy()];controls=[];phases=[];stats=[]
    for step in range(max_steps):
        control,request,phase=teacher.command(pose)
        safe,next_pose,check=world.safe_motion(pose,control)
        if not safe:return dict(passed=False,reason='unsafe_integrated_motion',step=step,pose=pose.tolist(),check=check,planning=planning)
        controls.append(control.copy());phases.append(phase['phase']);poses.append(next_pose.copy());stats.append(check);pose=next_pose
        if request:
            error=float(np.linalg.norm(pose[:2]-teacher.goal))
            return dict(passed=bool(error<=.05),reason='geometry_goal_reached_no_sensor_gate',steps=len(controls),
                original_goal_error_m=error,planning=planning,controller='finite_rate_pivot_and_straight_exact_unicycle',
                max_v_mps=float(np.max(np.asarray(controls)[:,0])),max_abs_omega_radps=float(np.max(abs(np.asarray(controls)[:,1]))),
                maximum_abs_pitch_deg=max(r['max_abs_pitch_deg'] for r in stats),
                maximum_abs_roll_deg=max(r['max_abs_roll_deg'] for r in stats),
                minimum_belly_clearance_m=min(r['minimum_belly_clearance_m'] for r in stats),
                actual_rgbd_captured=False,visibility_verified=False,training_ready=False,
                executed_poses=np.asarray(poses),controls=np.asarray(controls),phases=phases)
    return dict(passed=False,reason='max_steps',steps=max_steps,planning=planning,training_ready=False)


def self_test():
    checks=[]
    p=integrate_unicycle([0,0,0],[.4,0],.2);assert np.allclose(p,[.08,0,0]);checks.append('physical_velocity_dt_integrated')
    p=integrate_unicycle([0,0,0],[0,.8],.2);assert np.allclose(p,[0,0,.16]);checks.append('bounded_turn_integrated')
    p=integrate_unicycle([0,0,0],[.4,.8],.2);assert np.allclose(p,[.5*np.sin(.16),.5*(1-np.cos(.16)),.16]);checks.append('exact_arc_integrated')
    path=np.array([[0,0],[.2,0],[.4,0],[.4,.2],[.4,.4],[.2,.4]])
    compressed=compress_collinear_path(path);assert np.array_equal(compressed,[[0,0],[.4,0],[.4,.4],[.2,.4]]);checks.append('corners_and_backtracking_preserved')
    teacher=SegmentTeacher(path,np.array([.2,1.]));pose=np.array([0.,0.,0.]);trace=[pose.copy()];actions=[]
    for _ in range(200):
        control,stop,_=teacher.command(pose);actions.append(control.copy());pose=integrate_unicycle(pose,control);trace.append(pose.copy())
        if stop:break
    assert stop and np.linalg.norm(pose[:2]-teacher.goal)<1e-8 and abs(wrap_angle(math.pi/2-pose[2]))<1e-6
    assert all(np.allclose(integrate_unicycle(a,u),b) for a,u,b in zip(trace[:-1],actions,trace[1:]));checks.append('no_pose_snap_and_original_goal_reached')
    a=np.asarray(actions);assert np.max(a[:,0])<=.4 and np.max(abs(a[:,1]))<=.8;checks.append('control_limits')
    flags=np.zeros(len(a),np.uint8);flags[-1]=1;s=make_supervision(a,flags)
    assert s['future_vw_si'].shape==(len(a),8,2) and s['horizon_valid'][-1].tolist()==[True]+[False]*7
    assert np.array_equal(s['future_vw_si'][0],a[:8].astype(np.float32));checks.append('future8_real_commands_and_tail_mask')
    try:make_supervision([[0,0],[0,0]],[1,1])
    except ValueError:checks.append('nonterminal_stop_rejected')
    else:raise AssertionError('Multiple STOP labels accepted')
    # Exercise the actual swept checker without Kit, PXR, a GPU or source data.
    # A grazing circle intersects the segment although both endpoint centres
    # have sufficient clearance; endpoint-only checks must not accept it.
    class FlatProjector:
        grade=0.
        def project(self,xy):
            xy=np.asarray(xy,float);n=len(xy)
            return xy[:,0]*self.grade,np.tile([0.,0.,1.],(n,1)),np.zeros(n,int),np.zeros((n,3))
    world=TerrainWorld.__new__(TerrainWorld);world.projector=FlatProjector()
    world.runtime_collision_mode='planning_grid_and_geometry'
    world.radius=.65;world.resolution=.2;world.x=np.arange(-2,2.01,.2);world.y=world.x.copy()
    world.free=np.ones((len(world.x),len(world.y)),bool)
    world.centres=np.array([[5.,5.]]);world.radii=np.array([.2]);world.field=lambda p:np.ones(len(np.asarray(p).reshape(-1,3)))
    world._wheel_offsets=np.array([[.26,.33],[.26,-.33],[-.26,.33],[-.26,-.33]])
    world._wheel_design=np.column_stack([np.ones(4),world._wheel_offsets]);world._wheel_pinv=np.linalg.pinv(world._wheel_design)
    gx,gy=np.meshgrid(np.linspace(-.4,.4,9),np.linspace(-.26,.26,7),indexing='ij')
    world._belly=np.column_stack([gx.ravel(),gy.ravel()])
    gx,gy=np.meshgrid(np.linspace(-.475,.475,5),np.linspace(-.425,.425,5),indexing='ij')
    world._envelope=np.column_stack([gx.ravel(),gy.ravel()])
    assert world.safe_motion([0,0,0],[.4,0])[0];checks.append('flat_floor_control_accepted')
    world.free[:]=False
    assert not world.safe_motion([0,0,0],[.4,0])[0]
    world.runtime_collision_mode='actual_geometry'
    assert world.safe_motion([0,0,0],[.4,0])[0]
    checks.append('actual_geometry_mode_does_not_repeat_inflated_planning_cell_rejection')
    world.centres=np.array([[.04,.8498]])
    assert not world.safe_motion([0,0,0],[.4,0])[0]
    checks.append('actual_geometry_mode_still_rejects_real_swept_obstacle')
    world.runtime_collision_mode='planning_grid_and_geometry';world.free[:]=True
    world.centres=np.array([[.04,.8498]])
    assert world.pose_checks([[0,0,0],[.08,0,0]])[0].all()
    assert not world.safe_motion([0,0,0],[.4,0])[0];checks.append('swept_grazing_circle_rejected_despite_safe_endpoints')
    world.centres=np.array([[5.,5.]]);world.projector.grade=.2
    assert not world.safe_motion([0,0,0],[.4,0])[0];checks.append('excessive_actual_floor_pitch_rejected')
    world.projector.grade=0.;world.field=lambda p:np.zeros(len(np.asarray(p).reshape(-1,3)))
    assert not world.safe_motion([0,0,0],[0,.8])[0];checks.append('vehicle_air_envelope_collision_rejected')
    return dict(passed=True,checks=checks,GPU_work=False)
