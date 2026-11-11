"""CPU path-existence screening of a preserved lunar-cave obstacle map.

The saved terrain triangles and all declared obstacles are read-only. This is
an oracle geometric feasibility check, not a VLA score or wheel-dynamics test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt


DEFAULTS = dict(grid_resolution_m=.20, grid_bounds_xy_m=[[-35.,-23.],[79.,23.]],
    max_local_grade_deg=8., max_neighbor_height_change_m=.04,
    vehicle_radius_m=.65, grid_cell_margin=True, slope_cost_weight=.5,
    max_path_pitch_deg=8., max_path_roll_deg=8., max_wheel_sample_step_m=.03,
    max_wheel_plane_residual_m=.025, min_belly_clearance_m=.10,
    path_sample_spacing_m=.10, max_search_attempts=3, require_detour=True)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe(value):
    if isinstance(value, np.ndarray):return _safe(value.tolist())
    if isinstance(value, np.generic):return _safe(value.item())
    if isinstance(value, dict):return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):return None
    return value


def _write(path, report):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(_safe(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def _summary(values):
    values=np.asarray(values,float);finite=values[np.isfinite(values)]
    if not len(finite):return dict(minimum=None,maximum=None,mean=None)
    return dict(minimum=float(finite.min()),maximum=float(finite.max()),mean=float(finite.mean()))


def _point(config, name, default):
    value=config.get(name,config.get(name+'_xy_m',default))
    if isinstance(value,dict):
        value=value.get('xy_m',value.get('position_xy_m',value.get('position',value.get('xyz_m'))))
    result=np.asarray(value,float)
    if result.ndim!=1 or len(result) not in (2,3) or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite XY/XYZ point or a dictionary containing xy_m')
    return result[:2]


def _cell(point,x,y):
    i=int(np.rint((point[0]-x[0])/(x[1]-x[0])))
    j=int(np.rint((point[1]-y[0])/(y[1]-y[0])))
    return (i,j) if 0<=i<len(x) and 0<=j<len(y) else None


def _nearest_free(point,free,x,y):
    where=np.argwhere(free)
    if not len(where):return None
    points=np.column_stack([x[where[:,0]],y[where[:,1]]])
    index=int(np.argmin(np.sum((points-point)**2,axis=1)))
    return dict(xy_m=points[index].tolist(),distance_m=float(np.linalg.norm(points[index]-point)),
                grid_index=where[index].tolist(),applied=False)


def _read_obstacles(scene):
    """Retain every declared floor-level macro rock and >=25cm gravel rock."""
    obstacles=[];counts=dict(macro_total=0,macro_floor_level=0,macro_surface_excluded=0,
                            gravel_total=0,gravel_floor_level=0,gravel_surface_excluded=0)
    macro=scene/'boulders.json'
    if not macro.is_file():
        raise ValueError('v7 requires boulders.json; do not silently omit the macro-boulder collision layer')
    if macro.is_file():
        data=json.loads(macro.read_text())
        if not isinstance(data,list):raise ValueError('boulders.json must contain a list')
        counts['macro_total']=len(data)
        for index,item in enumerate(data):
            center=np.asarray(item['center_xyz_m'],float);scale=np.asarray(item['scale_xyz_m'],float)
            if center.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale<=0):
                raise ValueError('Malformed macro-boulder specification')
            if center[2]>=22.:
                counts['macro_surface_excluded']+=1;continue
            radius=1.35*float(np.linalg.norm(scale[:2]))
            obstacles.append(dict(id='macro_'+str(index),kind='macro',center_xy_m=center[:2].tolist(),
                xy_radius_m=radius,z_min_m=float(center[2]-1.35*scale[2]),z_max_m=float(center[2]+1.35*scale[2]),
                source_radius='conservative 1.35*hypot(authored XY half-scales)',source_index=index))
            counts['macro_floor_level']+=1
    gravel=scene/'obstacle_report.json'
    if not gravel.is_file():
        raise ValueError('v7 requires obstacle_report.json; do not silently omit the large-gravel collision layer')
    data=json.loads(gravel.read_text())
    if data.get('passed') is not True or data.get('macro_boulders_included') is not False:
        raise ValueError('Large-gravel obstacle report is rejected or ambiguously includes macro boulders')
    entries=data.get('obstacles')
    if not isinstance(entries,list):raise ValueError('obstacle_report.json requires an obstacles list')
    counts['gravel_total']=len(entries)
    for index,item in enumerate(entries):
        center=np.asarray(item['center_xyz_m'],float);radius=float(item['xy_radius_m'])
        low,high=float(item['z_min_m']),float(item['z_max_m'])
        if center.shape!=(3,) or not np.isfinite(center).all() or not all(map(math.isfinite,[radius,low,high])) or radius<0 or high<low:
            raise ValueError('Malformed serialized large-gravel obstacle')
        if center[2]>=22.:
            counts['gravel_surface_excluded']+=1;continue
        obstacles.append(dict(id='gravel_'+str(item.get('instance_id',index)),kind='gravel',
            center_xy_m=center[:2].tolist(),xy_radius_m=radius,z_min_m=low,z_max_m=high,
            source_radius='radius of all serialized transformed world vertices',
            full_diameter_m=item.get('full_diameter_m'),source_index=item.get('instance_index',index)))
        counts['gravel_floor_level']+=1
    return obstacles,dict(counts=counts,threshold_full_diameter_m=data.get('threshold_full_diameter_m',.25),
        reported_gravel_obstacle_count=data.get('obstacle_count'),collider_path=data.get('collider_path'),
        all_floor_obstacles_retained=True,exclusion_scope='Only surface-rim objects with center Z>=22m are outside this underground-floor task',
        micro_under_threshold_scope='Small gravel remains in the scene but is not an obstacle in this coarse geometric map')


def _obstacle_masks(obstacles,x,y,inflation):
    inflated=np.zeros((len(x),len(y)),bool);raw=np.zeros_like(inflated)
    outside=0
    for rock in obstacles:
        cx,cy=rock['center_xy_m'];radius=rock['xy_radius_m'];total=radius+inflation
        ia=max(0,int(np.searchsorted(x,cx-total,'left')));ib=min(len(x),int(np.searchsorted(x,cx+total,'right')))
        ja=max(0,int(np.searchsorted(y,cy-total,'left')));jb=min(len(y),int(np.searchsorted(y,cy+total,'right')))
        if ia>=ib or ja>=jb:outside+=1;continue
        d=(x[ia:ib,None]-cx)**2+(y[None,ja:jb]-cy)**2
        inflated[ia:ib,ja:jb]|=d<=total*total
        raw[ia:ib,ja:jb]|=d<=radius*radius
    return raw,inflated,outside


def _astar(free,height,slope,start,goal,resolution,options):
    """Eight neighbours; forbid diagonal corner-cutting and unsafe height edges."""
    if start is None or goal is None or not free[start] or not free[goal]:
        return None,dict(expanded_nodes=0,error='start_or_goal_not_free')
    score=np.full(free.shape,np.inf);score[start]=0.
    parent=np.full((*free.shape,2),-1,np.int32);closed=np.zeros(free.shape,bool)
    def heuristic(a):return math.hypot(a[0]-goal[0],a[1]-goal[1])*resolution
    queue=[(heuristic(start),0.,start[0],start[1])];expanded=0
    neighbours=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while queue:
        _,cost,i,j=heapq.heappop(queue)
        if closed[i,j] or cost>score[i,j]+1e-10:continue
        closed[i,j]=True;expanded+=1
        if (i,j)==goal:
            path=[goal];cur=goal
            while cur!=start:
                cur=tuple(map(int,parent[cur]));path.append(cur)
            path.reverse()
            return np.asarray(path,np.int32),dict(expanded_nodes=expanded,weighted_cost_m=float(cost),
                                                  diagonal_corner_cutting=False)
        for di,dj in neighbours:
            ni,nj=i+di,j+dj
            if not (0<=ni<free.shape[0] and 0<=nj<free.shape[1]) or not free[ni,nj] or closed[ni,nj]:continue
            diagonal=bool(di and dj)
            if diagonal and (not free[i+di,j] or not free[i,j+dj]):continue
            factor=math.sqrt(2.) if diagonal else 1.
            delta=abs(float(height[ni,nj]-height[i,j]))
            if delta>options['max_neighbor_height_change_m']*factor+1e-8:continue
            local=(float(slope[i,j])+float(slope[ni,nj]))*.5/options['max_local_grade_deg']
            step=resolution*factor*(1.+options['slope_cost_weight']*local*local)
            candidate=cost+step
            if candidate<score[ni,nj]-1e-10:
                score[ni,nj]=candidate;parent[ni,nj]=[i,j]
                heapq.heappush(queue,(candidate+heuristic((ni,nj)),candidate,ni,nj))
    return None,dict(expanded_nodes=expanded,error='no_path_through_preserved_obstacles',diagonal_corner_cutting=False)


def _resample(points,spacing):
    points=np.asarray(points,float)
    keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8];points=points[keep]
    if len(points)<2:raise ValueError('Path has no length')
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    s=np.r_[np.arange(0.,lengths[-1],spacing),lengths[-1]]
    return np.column_stack([np.interp(s,lengths,points[:,i]) for i in range(2)]),s


def _world_xy(xy,yaw,offsets):
    forward=np.column_stack([np.cos(yaw),np.sin(yaw)]);left=np.column_stack([-np.sin(yaw),np.cos(yaw)])
    return xy[:,None,:]+offsets[None,:,0,None]*forward[:,None,:]+offsets[None,:,1,None]*left[:,None,:]


def _path_geometry(points,projector,field_function,options):
    xy,s=_resample(points,options['path_sample_spacing_m'])
    tangent=np.column_stack([np.gradient(xy[:,i],s,edge_order=1) for i in range(2)])
    yaw=np.arctan2(tangent[:,1],tangent[:,0])
    wheel_offsets=np.array([[.26,.33],[.26,-.33],[-.26,.33],[-.26,-.33]])
    wheel_xy=_world_xy(xy,yaw,wheel_offsets)
    wheel_z,_,face,_=projector.project(wheel_xy.reshape(-1,2));wheel_z=wheel_z.reshape(-1,4);face=face.reshape(-1,4)
    supported=np.isfinite(wheel_z).all(1)&(face>=0).all(1)
    design=np.column_stack([np.ones(4),wheel_offsets]);plane=np.full((len(xy),3),np.nan)
    plane[supported]=wheel_z[supported]@np.linalg.pinv(design).T
    residual=np.abs(wheel_z-plane@design.T)
    pitch=np.degrees(np.arctan(plane[:,1]));roll=np.degrees(np.arctan(plane[:,2]))
    steps=np.abs(np.diff(wheel_z,axis=0))
    gx,gy=np.meshgrid(np.linspace(-.4,.4,9),np.linspace(-.26,.26,7),indexing='ij')
    offsets=np.column_stack([gx.ravel(),gy.ravel()]);bxy=_world_xy(xy,yaw,offsets)
    bz,_,bf,_=projector.project(bxy.reshape(-1,2));bz=bz.reshape(len(xy),-1);bf=bf.reshape(len(xy),-1)
    body_plane=plane[:,0,None]+plane[:,1,None]*offsets[None,:,0]+plane[:,2,None]*offsets[None,:,1]
    clear=body_plane+.15-bz;belly_support=np.isfinite(bz).all(1)&(bf>=0).all(1)
    ex,ey=np.meshgrid(np.linspace(-.475,.475,5),np.linspace(-.425,.425,5),indexing='ij')
    eo=np.column_stack([ex.ravel(),ey.ravel()]);exy=_world_xy(xy,yaw,eo)
    ep=plane[:,0,None]+plane[:,1,None]*eo[None,:,0]+plane[:,2,None]*eo[None,:,1]
    air_ok=np.zeros(len(xy),bool);air_min=np.full(len(xy),np.nan)
    selected=np.flatnonzero(supported)
    if len(selected):
        h=np.array([.3,.8,1.2]);pxy=np.broadcast_to(exy[selected,:,None,:],(len(selected),len(eo),len(h),2))
        pz=ep[selected,:,None]+h[None,None,:]
        xyz=np.concatenate([pxy,pz[...,None]],axis=-1).reshape(-1,3)
        values=np.concatenate([np.asarray(field_function(xyz[i:i+65536]),float).reshape(-1) for i in range(0,len(xyz),65536)])
        values=values.reshape(len(selected),-1);air_ok[selected]=np.isfinite(values).all(1)&(values>0).all(1)
        air_min[selected]=values.min(1)
    bad=(~supported|~belly_support|~air_ok|(np.abs(pitch)>options['max_path_pitch_deg']+1e-7)
         |(np.abs(roll)>options['max_path_roll_deg']+1e-7)
         |np.any(residual>options['max_wheel_plane_residual_m']+1e-7,axis=1)
         |np.any(clear<options['min_belly_clearance_m']-1e-7,axis=1))
    step_bad=np.any(steps>options['max_wheel_sample_step_m']+1e-7,axis=1)
    bad[:-1]|=step_bad;bad[1:]|=step_bad
    checks=dict(all_wheels_supported=bool(supported.all()),belly_mesh_support=bool(belly_support.all()),
        pitch_within_limit=bool(np.isfinite(pitch).all() and np.max(abs(pitch))<=options['max_path_pitch_deg']+1e-7),
        roll_within_limit=bool(np.isfinite(roll).all() and np.max(abs(roll))<=options['max_path_roll_deg']+1e-7),
        wheel_short_step_within_limit=bool(np.isfinite(steps).all() and np.max(steps)<=options['max_wheel_sample_step_m']+1e-7),
        contact_plane_residual_within_limit=bool(np.isfinite(residual).all() and residual.max()<=options['max_wheel_plane_residual_m']+1e-7),
        belly_clearance_within_limit=bool(np.isfinite(clear).all() and clear.min()>=options['min_belly_clearance_m']-1e-7),
        sampled_air_envelope_clear=bool(air_ok.all()))
    report=dict(passed=bool(all(checks.values())),checks=checks,pose_count=len(xy),sample_spacing_m=options['path_sample_spacing_m'],
        maximum_abs_pitch_deg=_summary(abs(pitch))['maximum'],maximum_abs_roll_deg=_summary(abs(roll))['maximum'],
        maximum_wheel_height_change_m=_summary(steps)['maximum'],maximum_wheel_plane_residual_m=_summary(residual)['maximum'],
        minimum_belly_clearance_m=_summary(clear)['minimum'],air_envelope_samples_per_pose=75,
        minimum_air_margin_not_euclidean_clearance=_summary(air_min)['minimum'],violating_pose_count=int(bad.sum()),
        first_violations=[dict(index=int(i),xy_m=xy[i].tolist(),pitch_deg=pitch[i],roll_deg=roll[i],air_clear=bool(air_ok[i]))
                          for i in np.flatnonzero(bad)[:25]],
        support='actual saved floor triangles and four-wheel least-squares plane',
        geometry=dict(wheel_radius_m=.18,wheelbase_m=.52,track_width_m=.66,footprint_m=[.95,.85],belly_allowance_m=.15),
        motion_scope='untimed polyline poses; rotations and continuous tyre/contact dynamics are not simulated')
    return report,xy,bad,plane[:,0]


def _line_stats(start,goal,x,y,terrain_free,obstacle_mask,free,obstacles,inflation):
    length=float(np.linalg.norm(goal-start));count=max(2,int(math.ceil(length/((x[1]-x[0])*.5)))+1)
    points=start+np.linspace(0,1,count)[:,None]*(goal-start)
    cells=[_cell(p,x,y) for p in points]
    good=np.array([cell is not None and free[cell] for cell in cells])
    terrain=np.array([cell is not None and terrain_free[cell] for cell in cells])
    obstacle=np.array([cell is not None and obstacle_mask[cell] for cell in cells])
    vector=goal-start;norm=float(vector@vector);hits=[]
    for rock in obstacles:
        c=np.asarray(rock['center_xy_m']);t=float(np.clip((c-start)@vector/max(norm,1e-12),0,1))
        distance=float(np.linalg.norm(c-start-t*vector))
        if distance<=rock['xy_radius_m']+inflation:hits.append(rock['id'])
    return dict(length_m=length,samples=count,blocked=bool(not good.all()),blocked_samples=int((~good).sum()),
        blocked_fraction=float((~good).mean()),terrain_inflation_blocked_samples=int((~terrain).sum()),
        inflated_obstacle_blocked_samples=int(obstacle.sum()),exact_inflated_circle_intersections=len(hits),
        first_intersected_obstacle_ids=hits[:30],sample_xy_m=points)


def _plot(path,x,y,height,terrain_free,obstacle_mask,free,start,goal,path_xy,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    figure,ax=plt.subplots(figsize=(17,7.5),facecolor='#f4f6f8')
    valid=np.isfinite(height);values=height[valid]
    low,high=(np.percentile(values,[1,99]) if len(values) else [-2,2])
    if high-low<.01:low-=.01;high+=.01
    half=(x[1]-x[0])*.5;extent=[x[0]-half,x[-1]+half,y[0]-half,y[-1]+half]
    image=ax.imshow(np.ma.masked_invalid(height.T),origin='lower',extent=extent,
                    cmap='terrain',vmin=low,vmax=high,interpolation='nearest',aspect='equal')
    ax.imshow(np.where((~terrain_free).T,1.,np.nan),origin='lower',extent=extent,
              cmap='Greys',vmin=0,vmax=1,alpha=.38,interpolation='nearest')
    ax.imshow(np.where(obstacle_mask.T,1.,np.nan),origin='lower',extent=extent,
              cmap='Reds',vmin=0,vmax=1,alpha=.56,interpolation='nearest')
    if path_xy is not None:ax.plot(path_xy[:,0],path_xy[:,1],color='#19dfed',lw=2.1,zorder=6)
    ax.scatter(*start,marker='*',s=160,c='#04ca68',edgecolors='black',zorder=8)
    ax.scatter(*goal,marker='X',s=90,c='#ffe45d',edgecolors='black',zorder=8)
    handles=[Patch(facecolor='#777',alpha=.5,label='unsupported / steep / air boundary, vehicle-inflated'),
             Patch(facecolor='#cb3535',alpha=.7,label='preserved rocks, vehicle-inflated'),
             Line2D([0],[0],color='#19dfed',lw=2,label='geometric A* path'),
             Line2D([0],[0],marker='*',color='none',markerfacecolor='#04ca68',markeredgecolor='black',markersize=12,label='declared pit-floor start'),
             Line2D([0],[0],marker='X',color='none',markerfacecolor='#ffe45d',markeredgecolor='black',markersize=9,label='declared goal')]
    ax.legend(handles=handles,loc='upper left',fontsize=8,framealpha=.95)
    ax.set(xlabel='X (m)',ylabel='Y (m)',title='Preserved lunar-cave obstacle map / geometric feasibility only')
    ax.grid(alpha=.15);figure.colorbar(image,ax=ax,pad=.02,label='Actual mesh floor height (m), display clipped to 1-99 percentile')
    found=report['path_found'];length=report.get('path_length_m');clear=(report.get('post_path_geometry') or {}).get('passed')
    figure.text(.5,.025,f"Path found: {found} | length: {length if length is not None else 'n/a'} m | post-check: {clear} | no obstacle removal | not a VLA or dynamics result",ha='center',fontsize=10)
    figure.tight_layout(rect=[0,.05,1,1]);figure.savefig(path,dpi=150);plt.close(figure)


def _save_outputs(output,result,x,y,height,slope,terrain_free,obstacle_mask,free,search_free,start,goal,path_xy,started):
    """Persist geometric evidence before attempting optional image rendering."""
    output=Path(output).expanduser().absolute();output.mkdir(parents=True,exist_ok=True)
    accepted_path=path_xy if result['path_found'] else np.empty((0,2))
    np.savez_compressed(output/'grid_data.npz',x=x,y=y,height=height,
        slope_deg=slope,terrain_free=terrain_free,obstacle_mask=obstacle_mask,
        free=free,search_free=search_free,path_xy=accepted_path)
    result['artifacts']=dict(map=None,grid='grid_data.npz',report='obstacle_map_report.json')
    result['visualization']=dict(status='pending',requested_file='obstacle_map.png')
    result['elapsed_seconds']=time.monotonic()-started
    _write(output/'obstacle_map_report.json',result)
    try:
        _plot(output/'obstacle_map.png',x,y,height,terrain_free,obstacle_mask,free,start,goal,
              accepted_path if result['path_found'] else None,result)
    except Exception as error:
        result['visualization']=dict(status='failed',requested_file='obstacle_map.png',
                                     error_type=type(error).__name__,error=str(error))
    else:
        result['artifacts']['map']='obstacle_map.png'
        result['visualization']=dict(status='saved',requested_file='obstacle_map.png')
    result['elapsed_seconds']=time.monotonic()-started
    _write(output/'obstacle_map_report.json',result)


def validate_obstacle_map(scene_dir,output_dir=None,*,extra_obstacles=None,goal_override_xy_m=None):
    scene=Path(scene_dir).expanduser().resolve(strict=True);started=time.monotonic()
    config=json.loads((scene/'scene_config.json').read_text())
    if 'rover_route' in config or any('rover_route_floor' in config.get(k,{}) for k in ['cavern','pit']):
        raise ValueError('v7 rejects fixed-route floor configuration; obstacles and terrain must remain unscripted')
    if not config.get('ground_profile_v7'):raise ValueError('v7 requires an explicit ground_profile_v7')
    options=dict(DEFAULTS);options.update(config.get('obstacle_map_validation',{}))
    resolution=float(options['grid_resolution_m'])
    if not .10<=resolution<=.25:raise ValueError('Grid resolution must remain within [.10,.25]m')
    low,high=np.asarray(options['grid_bounds_xy_m'],float)
    if low.shape!=(2,) or high.shape!=(2,) or np.any(high<=low):raise ValueError('Invalid XY bounds')
    start=_point(config,'task_spawn',[-15.,-3.]);goal=_point(config,'task_goal',[74.,-2.6])
    if goal_override_xy_m is not None:
        goal=np.asarray(goal_override_xy_m,float)
        if goal.shape!=(2,) or not np.isfinite(goal).all():raise ValueError('Invalid explicit branch goal')
    if np.linalg.norm(start-goal)<resolution:raise ValueError('Start and goal must be distinct')
    protected=[scene/name for name in ['scene_config.json','mesh.npz','cave.usdc','geometry_report.json','boulders.json','obstacle_report.json','targets.json','task_obstacles.json'] if (scene/name).is_file()]
    hashes={str(path):_sha(path) for path in protected}
    from cave.lunar_gravel_v5 import _FloorProjector
    from build_lunar_cave_v4_cpu import evaluate_scene_field
    with np.load(scene/'mesh.npz') as data:projector=_FloorProjector(data['vertices'],data['faces'])
    x=low[0]+np.arange(int(np.floor((high[0]-low[0])/resolution))+1)*resolution
    y=low[1]+np.arange(int(np.floor((high[1]-low[1])/resolution))+1)*resolution
    xx,yy=np.meshgrid(x,y,indexing='ij');points=np.column_stack([xx.ravel(),yy.ravel()])
    height,normals,faces,_=projector.project(points);height=height.reshape(xx.shape)
    normals=normals.reshape(*xx.shape,3);faces=faces.reshape(xx.shape)
    supported=np.isfinite(height)&(faces>=0)
    slope=np.full(xx.shape,np.nan);slope[supported]=np.degrees(np.arccos(np.clip(normals[supported,2],-1,1)))
    air=np.zeros(xx.shape,bool)
    rows=np.flatnonzero(supported.ravel());xyz=np.column_stack([points[rows],height.ravel()[rows]])
    probes=np.repeat(xyz,3,axis=0);probes[:,2]+=np.tile([.3,.8,1.2],len(xyz))
    values=np.concatenate([np.asarray(evaluate_scene_field(config,probes[i:i+65536]),float).reshape(-1) for i in range(0,len(probes),65536)]) if len(probes) else np.empty(0)
    if len(rows):
        field_values=values.reshape(-1,3)
        air.ravel()[rows]=(np.isfinite(field_values)&(field_values>0)).all(1)
    local_ok=supported&air&(slope<=options['max_local_grade_deg']+1e-7)
    cell_margin=resolution/math.sqrt(2.) if options['grid_cell_margin'] else 0.
    inflation=float(options['vehicle_radius_m'])+cell_margin
    distance=distance_transform_edt(np.pad(local_ok,1,constant_values=False))[1:-1,1:-1]*resolution
    terrain_free=local_ok&(distance>=inflation-1e-8)
    obstacles,obstacle_meta=_read_obstacles(scene)
    if extra_obstacles is not None:
        for item in extra_obstacles:
            xy=np.asarray(item['center_xy_m'],float);radius=float(item['xy_radius_m'])
            if xy.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite(radius) or radius<=0:
                raise ValueError('Invalid extra target obstacle footprint')
            obstacles.append(dict(item,center_xy_m=xy.tolist(),xy_radius_m=radius))
        obstacle_meta['extra_target_obstacle_count']=len(extra_obstacles)
    raw_obstacles,inflated_obstacles,outside=_obstacle_masks(obstacles,x,y,inflation)
    free=terrain_free&~inflated_obstacles
    start_cell,goal_cell=_cell(start,x,y),_cell(goal,x,y)
    start_free=bool(start_cell is not None and free[start_cell]);goal_free=bool(goal_cell is not None and free[goal_cell])
    line=_line_stats(start,goal,x,y,terrain_free,inflated_obstacles,free,obstacles,inflation)
    attempts=[];path_xy=None;post=None;grid_path=None;search_free=free.copy()
    for attempt in range(int(options['max_search_attempts'])):
        grid_path,search=_astar(search_free,height,slope,start_cell,goal_cell,resolution,options)
        if grid_path is None:
            attempts.append(dict(attempt=attempt+1,path_found=False,search=search));break
        center_path=np.column_stack([x[grid_path[:,0]],y[grid_path[:,1]]])
        candidate=np.vstack([start,center_path,goal])
        post,dense,bad,path_z=_path_geometry(candidate,projector,lambda p:evaluate_scene_field(config,p),options)
        attempts.append(dict(attempt=attempt+1,path_found=True,search=search,post_check_passed=post['passed'],
                             invalid_pose_count=post['violating_pose_count']))
        path_xy=dense
        if post['passed']:break
        # Reject additional geometry-unsafe cells; no source obstacle is removed or changed.
        for point in dense[bad]:
            cell=_cell(point,x,y)
            if cell is not None:search_free[cell]=False
    path_found=bool(grid_path is not None and post is not None and post['passed'])
    path_length=None;deviation=None;extra=None;detour=False;path_xyz=None
    if path_found:
        path_length=float(np.linalg.norm(np.diff(path_xy,axis=0),axis=1).sum())
        vector=goal-start;t=np.clip((path_xy-start)@vector/(vector@vector),0,1)
        deviation=float(np.linalg.norm(path_xy-(start+t[:,None]*vector),axis=1).max())
        extra=path_length-line['length_m'];detour=bool(line['blocked'] and deviation>resolution*.5 and extra>1e-6)
        pz,_,_,_=projector.project(path_xy);path_xyz=np.column_stack([path_xy,pz])
    base=json.loads((scene/'geometry_report.json').read_text())
    ratios={name:float(mask.mean()) for name,mask in [('mesh_supported',supported),('local_grade_and_air_ok',local_ok),
        ('terrain_vehicle_inflated_free',terrain_free),('obstacle_inflated_blocked',inflated_obstacles),('combined_free',free)]}
    result=dict(schema_version='preserved_lunar_obstacle_map/7.0',updated_utc=datetime.now(timezone.utc).isoformat(),
        passed=bool(base.get('passed') is True and start_free and goal_free and path_found and (detour or not options['require_detour'])),
        scene=str(scene),source_hashes=hashes,base_geometry_report_passed=base.get('passed') is True,
        explicit_goal_override_xy_m=None if goal_override_xy_m is None else goal.tolist(),
        settings=options,vehicle_inflation_radius_m=inflation,grid_half_diagonal_margin_m=cell_margin,
        start=dict(declared_xy_m=start.tolist(),grid_index=start_cell,free=start_free,
                   grid_xy_m=None if start_cell is None else [float(x[start_cell[0]]),float(y[start_cell[1]])],
                   nearest_free_suggestion=None if start_free else _nearest_free(start,free,x,y),silently_relocated=False),
        goal=dict(declared_xy_m=goal.tolist(),grid_index=goal_cell,free=goal_free,
                  grid_xy_m=None if goal_cell is None else [float(x[goal_cell[0]]),float(y[goal_cell[1]])],
                  nearest_free_suggestion=None if goal_free else _nearest_free(goal,free,x,y),silently_relocated=False),
        grid_shape=list(xx.shape),grid_ratios=ratios,supported_floor_height_m=_summary(height),
        supported_local_grade_deg=_summary(slope),obstacle_summary=dict(obstacle_meta,
            retained_map_obstacle_count=len(obstacles),circles_outside_grid=outside,source_geometry_modified=False,
            raw_circle_blocked_cells=int(raw_obstacles.sum()),inflated_circle_blocked_cells=int(inflated_obstacles.sum())),
        straightline={k:v for k,v in line.items() if k!='sample_xy_m'},
        path_found=path_found,path_length_m=path_length,path_extra_length_over_straight_m=extra,
        path_maximum_straightline_deviation_m=deviation,detour_required=bool(options['require_detour']),detour_observed=detour,
        search_attempts=attempts,post_path_geometry=post,path_xyz_m=path_xyz,
        obstacle_removal_count=0,terrain_changed_by_validator=False,fixed_lane_or_cleared_route=False,
        GPU_work=False,physics_evaluated=False,navigation_policy_evaluated=False,
        limitations=['A geometric oracle search, not VLA navigation performance or success rate',
            'The vehicle radius, grades and support-plane clearances are engineering assumptions, not validated hardware capabilities',
            'No PhysX cooking, suspension, wheel slip, friction or timed controller simulation',
            'Circular projections conservatively enclose saved large rocks; small gravel below the collision threshold is not modelled as individual obstacles',
            'Finite grid and sampled footprint checks do not prove continuous swept-mesh collision freedom'])
    result['source_files_unchanged']=all(_sha(path)==sha for path,sha in hashes.items())
    result['passed']=bool(result['passed'] and result['source_files_unchanged'])
    if output_dir is not None:
        _save_outputs(output_dir,result,x,y,height,slope,terrain_free,inflated_obstacles,free,search_free,
                      start,goal,path_xy,started)
    return _safe(result)


def _self_test():
    options=dict(DEFAULTS);options['slope_cost_weight']=0.
    axis=np.linspace(-5,5,101);height=np.zeros((101,101));slope=np.zeros_like(height);free=np.ones_like(height,bool)
    xx,yy=np.meshgrid(axis,axis,indexing='ij');free[(xx*xx+yy*yy)<1.7**2]=False
    start=(10,50);goal=(90,50)
    path,search=_astar(free,height,slope,start,goal,.1,options)
    detour=path is not None and np.max(abs(axis[path[:,1]]))>1.5
    wall=free.copy();wall[50,:]=False
    blocked,_=_astar(wall,height,slope,start,goal,.1,options)
    corner=np.array([[True,False],[False,True]])
    diagonal,_=_astar(corner,np.zeros((2,2)),np.zeros((2,2)),(0,0),(1,1),.1,options)
    samples,arc=_resample(np.array([[0,0],[1,0],[0,1],[-1,1]]),.1)
    class ToyProjector:
        def __init__(self,floor):self.floor=floor
        def project(self,points):
            points=np.asarray(points)
            return self.floor(points),np.tile([0.,0.,1.],(len(points),1)),np.zeros(len(points),int),np.zeros((len(points),3))
    flat=ToyProjector(lambda p:np.zeros(len(p)))
    positive=lambda p:np.ones(len(p))
    bend=np.array([[0.,0.],[2.,0.],[1.,2.],[-1.,2.]])
    flat_geometry,_,_,_=_path_geometry(bend,flat,positive,options)
    steep=ToyProjector(lambda p:p[:,0]*np.tan(np.deg2rad(10.)))
    steep_geometry,_,_,_=_path_geometry(np.array([[0.,0.],[3.,0.]]),steep,positive,options)
    step_projector=ToyProjector(lambda p:np.where(p[:,0]>1.5,.12,0.))
    step_geometry,_,_,_=_path_geometry(np.array([[0.,0.],[3.,0.]]),step_projector,positive,options)
    air_geometry,_,_,_=_path_geometry(bend,flat,lambda p:np.full(len(p),-1.),options)
    tests=dict(obstacle_requires_preserved_detour=bool(detour),full_wall_rejects_path=blocked is None,
               diagonal_corner_cut_is_rejected=diagonal is None,
               path_resampling_allows_returning_x=bool(np.any(np.diff(samples[:,0])<0) and np.all(np.diff(arc)>0)),
               flat_bent_path_contact_geometry_passes=flat_geometry['passed'],
               steep_path_contact_geometry_rejected=not steep_geometry['checks']['pitch_within_limit'],
               abrupt_wheel_height_step_rejected=not step_geometry['checks']['wheel_short_step_within_limit'],
               blocked_air_envelope_rejected=not air_geometry['checks']['sampled_air_envelope_clear'])
    return dict(passed=all(tests.values()),tests=tests,GPU_work=False,physics_evaluated=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test:
        report=_self_test()
        if args.output_dir:_write(args.output_dir/'self_test.json',report)
    else:
        if args.scene is None or args.output_dir is None:parser.error('--scene and --output-dir are required')
        report=validate_obstacle_map(args.scene,args.output_dir)
    print(json.dumps({k:report.get(k) for k in ['passed','path_found','path_length_m','detour_observed','tests'] if k in report}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
