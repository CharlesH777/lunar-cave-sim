"""Seeded search-target placement on the preserved, uneven v7 lunar cave.

Targets are sampled without moving terrain or rocks and treated as conservative
obstacle circles. Category-to-longitudinal-segment assignment is randomized;
neither the Chinese flag nor any target class has a fixed world position.
This is geometric placement and reachability screening, not rover dynamics.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from validate_lunar_obstacle_map_v7 import (
    DEFAULTS, _astar, _cell, _obstacle_masks, _path_geometry, _point, _read_obstacles)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def _specifications(specs):
    if not isinstance(specs,(list,tuple)) or not specs:raise ValueError('specs must be a nonempty sequence')
    result=[]
    for spec in specs:
        if not isinstance(spec,dict):raise ValueError('each target spec must be a dictionary')
        item={key:spec[key] for key in ['id','class_name','footprint_radius_m','height_m']}
        item['id']=str(item['id']);item['class_name']=str(item['class_name'])
        item['footprint_radius_m']=float(item['footprint_radius_m']);item['height_m']=float(item['height_m'])
        if not item['id'] or not item['class_name'] or not np.isfinite([item['footprint_radius_m'],item['height_m']]).all():
            raise ValueError('Target identity and dimensions must be valid and finite')
        if not .05<=item['footprint_radius_m']<=3.5 or not .1<=item['height_m']<=5.:
            raise ValueError('Target dimensions exceed the small-cave-demo placement bounds')
        result.append(item)
    if len({s['id'] for s in result})!=len(result):raise ValueError('Target ids must be unique')
    return result


def _grid(scene,config,hashes,projector,field):
    """Reuse only a grid whose complete source-file digest matches this scene."""
    for path in sorted(scene.parent.glob('validation_*/obstacle_map_report.json'),reverse=True):
        report=json.loads(path.read_text())
        if report.get('passed') is not True:continue
        source=report.get('source_hashes',{})
        if any(source.get(str(scene/name))!=digest for name,digest in hashes.items()):continue
        archive=path.parent/'grid_data.npz'
        if not archive.is_file():continue
        with np.load(archive) as data:
            arrays={name:np.array(data[name],copy=True) for name in ['x','y','height','slope_deg','terrain_free']}
        shape=(len(arrays['x']),len(arrays['y']))
        if any(arrays[name].shape!=shape for name in ['height','slope_deg','terrain_free']):
            raise ValueError('Validated source grid shape is inconsistent')
        return arrays,dict(report['settings']),dict(source='validated_grid_cache',report=str(path),
            report_sha256=_sha(path),grid_sha256=_sha(archive))
    options=dict(DEFAULTS);options.update(config.get('obstacle_map_validation',{}))
    resolution=float(options['grid_resolution_m']);low,high=np.asarray(options['grid_bounds_xy_m'],float)
    x=low[0]+np.arange(int(np.floor((high[0]-low[0])/resolution))+1)*resolution
    y=low[1]+np.arange(int(np.floor((high[1]-low[1])/resolution))+1)*resolution
    xx,yy=np.meshgrid(x,y,indexing='ij');xy=np.column_stack([xx.ravel(),yy.ravel()])
    height,normals,faces,_=projector.project(xy);height=height.reshape(xx.shape)
    normals=normals.reshape(*xx.shape,3);faces=faces.reshape(xx.shape)
    supported=np.isfinite(height)&(faces>=0);slope=np.full(xx.shape,np.nan)
    slope[supported]=np.degrees(np.arccos(np.clip(normals[supported,2],-1,1)))
    selected=np.flatnonzero(supported.ravel());xyz=np.column_stack([xy[selected],height.ravel()[selected]])
    probes=np.repeat(xyz,3,axis=0);probes[:,2]+=np.tile([.3,.8,1.2],len(xyz))
    values=np.asarray(field(probes),float).reshape(-1,3)
    air=np.zeros(xx.shape,bool);air.ravel()[selected]=(np.isfinite(values)&(values>0)).all(1)
    local=supported&air&(slope<=options['max_local_grade_deg']+1e-7)
    inflation=float(options['vehicle_radius_m'])+(resolution/math.sqrt(2.) if options['grid_cell_margin'] else 0.)
    distance=distance_transform_edt(np.pad(local,1,constant_values=False))[1:-1,1:-1]*resolution
    terrain=local&(distance>=inflation-1e-8)
    return dict(x=x,y=y,height=height,slope_deg=slope,terrain_free=terrain),options,dict(source='fresh_actual_mesh_grid')


def _footprint(projector,field,xy,radius,height):
    axis=np.arange(-radius,radius+.125,.25)
    a,b=np.meshgrid(axis,axis,indexing='ij')
    offsets=np.column_stack([a.ravel(),b.ravel()]);offsets=offsets[np.linalg.norm(offsets,axis=1)<=radius+1e-7]
    angles=np.linspace(0.,2.*np.pi,32,endpoint=False)
    offsets=np.vstack([[[0.,0.]],offsets,radius*np.column_stack([np.cos(angles),np.sin(angles)])])
    positions=xy+offsets
    z,normal,face,_=projector.project(positions)
    if not (np.isfinite(z).all() and (face>=0).all()):return None,'unsupported_footprint'
    if np.max(z)>.5:return None,'elevated_ledge_not_cave_floor'
    design=np.column_stack([np.ones(len(offsets)),offsets])
    plane=np.linalg.lstsq(design,z,rcond=None)[0]
    tilt=float(np.degrees(np.arctan(np.linalg.norm(plane[1:]))))
    spread=float(np.ptp(z));allowance=.15 if radius>1.3 else .08
    residual=float(np.max(np.abs(z-design@plane)))
    if tilt>6.+1e-7:return None,'support_plane_grade'
    if spread>allowance+1e-7 or residual>.04:return None,'support_height_spread'
    root_z=float(z.max()+.005)
    levels=np.unique(np.r_[.04,.3,np.linspace(.5,height+.10,5)])
    pxy=np.repeat(positions,len(levels),axis=0)
    pz=np.tile(root_z+levels,len(positions))
    values=np.asarray(field(np.column_stack([pxy,pz])),float)
    if not np.isfinite(values).all() or np.any(values<=.01):return None,'solid_or_roof_overlap'
    return dict(xyz_m=[float(xy[0]),float(xy[1]),root_z],
        support_height_range_m=[float(z.min()),float(z.max())],support_height_spread_m=spread,
        support_plane_grade_deg=tilt,support_plane_residual_m=residual,
        ground_plane_gradient_xy=plane[1:].tolist(),
        root_height_policy='highest sampled terrain support + 0.005m',
        support_padding_allowance_m=allowance,footprint_sample_count=len(z),
        minimum_air_margin_not_euclidean_clearance_m=float(values.min())),None


def sample_target_layout(scene_dir,specs,seed):
    """Return JSON-safe ``targets`` and ``report`` for a reproducible layout.

    Each spec has ``id``, ``class_name``, ``footprint_radius_m`` (the entire
    object's conservative XY projection) and ``height_m``. All returned poses
    are local-Z-up rigid placements with independent yaw; a support-plane
    variation allowance is reported, not simulated suspension or levelling.
    At most 1000 candidate poses and 12 complete layouts are considered.
    """
    started=time.monotonic();scene=Path(scene_dir).expanduser().resolve(strict=True)
    specs=_specifications(specs)
    if not isinstance(seed,(int,np.integer)) or isinstance(seed,bool) or int(seed)<0:
        raise ValueError('seed must be a nonnegative integer')
    rng=np.random.default_rng(int(seed));config=json.loads((scene/'scene_config.json').read_text())
    names=['scene_config.json','mesh.npz','cave.usdc','geometry_report.json','boulders.json','obstacle_report.json']
    hashes={name:_sha(scene/name) for name in names}
    from cave.lunar_gravel_v5 import _FloorProjector
    from build_lunar_cave_v4_cpu import evaluate_scene_field
    with np.load(scene/'mesh.npz') as data:projector=_FloorProjector(data['vertices'],data['faces'])
    field=lambda points:evaluate_scene_field(config,points)
    grid,options,grid_provenance=_grid(scene,config,hashes,projector,field)
    x,y=grid['x'],grid['y'];height,slope=grid['height'],grid['slope_deg'];terrain=grid['terrain_free']
    resolution=float(options['grid_resolution_m'])
    inflation=float(options['vehicle_radius_m'])+(resolution/math.sqrt(2.) if options['grid_cell_margin'] else 0.)
    rocks,rock_metadata=_read_obstacles(scene)
    rock_xy=np.asarray([r['center_xy_m'] for r in rocks],float);rock_radius=np.asarray([r['xy_radius_m'] for r in rocks])
    _,old_mask,_=_obstacle_masks(rocks,x,y,inflation)
    baseline_free=terrain&~old_mask
    start=_point(config,'task_spawn',[-15.,-3.]);goal=_point(config,'task_goal',[74.,-2.6])
    start_cell,goal_cell=_cell(start,x,y),_cell(goal,x,y)
    base_path,base_search=_astar(baseline_free,height,slope,start_cell,goal_cell,resolution,options)
    if base_path is None:raise ValueError('Source scene is not connected under the preserved-obstacle checks')
    xx,yy=np.meshgrid(x,y,indexing='ij')
    branching=bool(config.get('branch_v10'))
    allowed=terrain&np.isfinite(height)&(height<.5)&(slope<=6.)&(xx>=8.)&(xx<=72.)&((np.abs(yy)<10.)|((yy>14.)&(yy<58.)) if branching else (np.abs(yy)<10.))
    candidates=np.column_stack([xx[allowed],yy[allowed]])
    if not len(candidates):raise ValueError('No supported underground target candidates')
    cuts=np.linspace(8.,72.,len(specs)+1);attempts=0;rejects={};layout_failures=[]
    def reject(reason):rejects[reason]=rejects.get(reason,0)+1
    for layout_attempt in range(12):
        assignments=rng.permutation(len(specs));accepted=[]
        # Place largest footprints first while category-to-segment assignment
        # remains independent and randomized across layout seeds.
        order=sorted(range(len(specs)),key=lambda i:-specs[i]['footprint_radius_m'])
        for index in order:
            spec=specs[index];segment=int(assignments[index]);radius=spec['footprint_radius_m']
            if branching and len(specs)==4:
                if segment>=2:
                    lower,upper=(40.,54.) if segment==2 else (54.,72.)
                    pool=candidates[(candidates[:,1]>16.)&(candidates[:,0]>=lower)&(candidates[:,0]<upper)]
                else:
                    lower,upper=(8.,40.) if segment==0 else (40.,72.)
                    pool=candidates[(candidates[:,0]>=lower)&(candidates[:,0]<upper)&(np.abs(candidates[:,1])<10.)]
            else:pool=candidates[(candidates[:,0]>=cuts[segment])&(candidates[:,0]<cuts[segment+1])]
            found=None
            for _ in range(min(100,1000-attempts)):
                attempts+=1
                if not len(pool):reject('empty_segment');break
                xy=pool[int(rng.integers(len(pool)))]+rng.uniform(-.08,.08,2)
                if np.any(np.linalg.norm(rock_xy-xy,axis=1)<=rock_radius+radius+.12):reject('existing_rock_overlap');continue
                if any(np.linalg.norm(np.asarray(t['xyz_m'][:2])-xy)<=t['footprint_radius_m']+radius+.30 for t in accepted):
                    reject('target_overlap');continue
                if min(np.linalg.norm(start-xy),np.linalg.norm(goal-xy))<=radius+inflation+.20:
                    reject('task_endpoint_overlap');continue
                support,reason=_footprint(projector,field,xy,radius,spec['height_m'])
                if support is None:reject(reason);continue
                found=dict(spec,**support,yaw_deg=float(rng.uniform(-180.,180.)),
                    longitudinal_segment_index=segment,segment_x_range_m=[float(cuts[segment]),float(cuts[segment+1])])
                if branching:
                    found['placement_region']='passage_B' if segment>=2 else 'passage_A'
                    found['segment_x_range_m']=[lower,upper]
                accepted.append(found);break
            if found is None:break
        if len(accepted)!=len(specs):
            layout_failures.append(dict(attempt=layout_attempt+1,reason='target_pose_candidates_exhausted',placed=len(accepted)))
            if attempts>=1000:break
            continue
        props=[dict(id=t['id'],kind='search_target',center_xy_m=t['xyz_m'][:2],xy_radius_m=t['footprint_radius_m']) for t in accepted]
        _,prop_mask,_=_obstacle_masks(props,x,y,inflation)
        free=baseline_free&~prop_mask
        path,search=_astar(free,height,slope,start_cell,goal_cell,resolution,options)
        if path is None:
            reject('main_path_blocked');layout_failures.append(dict(attempt=layout_attempt+1,reason='main_path_blocked'));continue
        path_xy=np.column_stack([x[path[:,0]],y[path[:,1]]])
        post,dense,bad,_=_path_geometry(np.vstack([start,path_xy,goal]),projector,field,options)
        if not post['passed']:
            reject('main_path_pose_geometry');layout_failures.append(dict(attempt=layout_attempt+1,reason='main_path_pose_geometry'));continue
        all_approaches=True
        for target in accepted:
            center=np.asarray(target['xyz_m'][:2]);approach_radius=target['footprint_radius_m']+inflation+.30
            approach_angles=rng.permutation(np.linspace(-np.pi,np.pi,24,endpoint=False))
            approach=None
            for angle in approach_angles:
                query=center+approach_radius*np.array([np.cos(angle),np.sin(angle)])
                cell=_cell(query,x,y)
                if cell is None or not free[cell]:continue
                trial,reach=_astar(free,height,slope,start_cell,cell,resolution,options)
                if trial is None:continue
                approach=np.array([x[cell[0]],y[cell[1]]],float)
                target['approach_xy_m']=approach.tolist()
                target['approach_distance_to_origin_m']=float(np.linalg.norm(approach-center))
                target['approach_route_grid_length_m']=float(np.linalg.norm(np.diff(np.column_stack([x[trial[:,0]],y[trial[:,1]]]),axis=0),axis=1).sum())
                break
            if approach is None:all_approaches=False;break
        if not all_approaches:
            reject('unreachable_target_approach');layout_failures.append(dict(attempt=layout_attempt+1,reason='unreachable_target_approach'));continue
        if any(_sha(scene/name)!=digest for name,digest in hashes.items()):raise RuntimeError('Placement changed a protected source asset')
        by_id={t['id']:t for t in accepted};targets=[by_id[s['id']] for s in specs]
        return dict(targets=targets,report=dict(passed=True,schema_version='lunar_search_placement/8.0',seed=int(seed),
            source_scene=str(scene),source_sha256=hashes,source_files_unchanged=True,grid_provenance=grid_provenance,
            candidate_pose_attempts=attempts,layout_attempts=layout_attempt+1,rejection_counts=rejects,layout_failures=layout_failures,
            category_segment_assignment_randomized=True,positions_and_yaws_randomized=True,flag_position_fixed=False,
            sampling_x_range_m=[8.,72.],all_targets_on_underground_floor=True,all_target_approaches_reachable=True,
            original_obstacle_count=len(rocks),original_obstacles=rock_metadata,new_target_count=len(targets),
            preserved_start_xy_m=start.tolist(),preserved_goal_xy_m=goal.tolist(),vehicle_inflation_radius_m=inflation,
            full_start_to_goal_path_preserved=True,path_geometry=post,
            main_path_length_m=float(np.linalg.norm(np.diff(dense,axis=0),axis=1).sum()),
            main_path_xy_m=dense.tolist(),main_path_search=search,original_obstacle_removal_count=0,
            terrain_modified=False,fixed_cleared_lane=False,GPU_work=False,physics_evaluated=False,
            navigation_policy_evaluated=False,elapsed_seconds=time.monotonic()-started,
            limitations=['Conservative sampled placement and geometric planning, not continuous swept-object collision proof',
                'Ground-height allowance is an authored static support arrangement, not simulated self-levelling',
                'Gravel below the existing 25cm collision threshold is not individually checked']))
    raise RuntimeError('Could not place all targets while preserving rocks, supported footprints and reachable paths: '+
                       json.dumps(dict(seed=int(seed),candidate_attempts=attempts,rejections=rejects,layouts=layout_failures)))
