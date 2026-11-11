"""Forward Hybrid A* for the collision-checked cave rover.

Search nodes retain continuous x,y,yaw; a discretized SE(2) key only removes
duplicates. Every successor is an exact constant-curvature arc, so returned
paths are tangent-continuous and respect a minimum turning radius. A final
collision-checked C2 cubic B-spline removes arc-switch steering chatter. A
separate holonomic grid search supplies a guide, never the output path.
This implementation is weighted and resolution-limited, not optimality-certified.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time

import numpy as np


def smooth_hybrid_path(task, path, *, maximum_curvature_per_m=2.,
                       sample_spacing_m=.03, max_deviation_m=.10):
    """Fit and collision-check a C2 cubic B-spline, preserving endpoint poses.

    Weighted least squares removes primitive-to-primitive steering chatter.
    A small cubic Hermite correction enforces both endpoint positions/tangents
    exactly without breaking the spline's C2 interior continuity. Stronger fits
    are attempted first; a failed fit never replaces a feasible Hybrid path.
    """
    from scipy.interpolate import splprep,splev
    from scipy.spatial import cKDTree
    original=np.asarray(path,dtype=float)
    diagnostics={'method':'collision-checked C2 cubic B-spline with endpoint Hermite constraints',
                 'accepted':False,'maximum_deviation_m':max_deviation_m,
                 'curvature_limit_per_m':maximum_curvature_per_m,'sample_spacing_m':sample_spacing_m,
                 'candidates':[]}
    if len(original)<8:
        diagnostics['reason']='too_few_original_samples';return original.copy(),diagnostics
    distances=np.linalg.norm(np.diff(original[:,:2],axis=0),axis=1)
    if np.any(distances<1e-8):
        diagnostics['reason']='repeated_original_samples';return original.copy(),diagnostics
    arclength=np.r_[0.,np.cumsum(distances)];length=float(arclength[-1])
    if length<.2:
        diagnostics['reason']='short_path';return original.copy(),diagnostics
    parameters=arclength/length
    weights=np.ones(len(original));weights[:3]=weights[-3:]=1000.
    original_tree=cKDTree(original[:,:2])
    # Five-millimetre probes bound sampled curvature more tightly than the
    # returned three-centimetre trajectory and stabilize arclength resampling.
    probe_u=np.linspace(0.,1.,max(2001,int(math.ceil(length/.005))+1))
    for residual in [2.,1.,.5,.25,.1,.05,.02]:
        attempt={'weighted_squared_residual':residual}
        diagnostics['candidates'].append(attempt)
        try:
            spline,_=splprep(original[:,:2].T,u=parameters,w=weights,s=residual,k=3)
            endpoints=np.array(splev([0.,1.],spline)).T
            endpoint_derivatives=np.array(splev([0.,1.],spline,der=1)).T
            desired_derivatives=np.column_stack([np.cos(original[[0,-1],2]),np.sin(original[[0,-1],2])])
            desired_derivatives*=np.linalg.norm(endpoint_derivatives,axis=1)[:,None]
            corrections=np.vstack([original[0,:2]-endpoints[0],
                                   desired_derivatives[0]-endpoint_derivatives[0],
                                   original[-1,:2]-endpoints[1],
                                   desired_derivatives[1]-endpoint_derivatives[1]])
            def evaluate(u,derivative=0):
                u=np.atleast_1d(np.asarray(u,dtype=float))
                if derivative==0:
                    basis=np.column_stack([2*u**3-3*u**2+1,u**3-2*u**2+u,
                                           -2*u**3+3*u**2,u**3-u**2])
                elif derivative==1:
                    basis=np.column_stack([6*u**2-6*u,3*u**2-4*u+1,
                                           -6*u**2+6*u,3*u**2-2*u])
                else:
                    basis=np.column_stack([12*u-6,6*u-4,-12*u+6,6*u-2])
                return np.array(splev(u,spline,der=derivative)).T+basis@corrections
            probe_xy=evaluate(probe_u);d1=evaluate(probe_u,1);d2=evaluate(probe_u,2)
            speed=np.linalg.norm(d1,axis=1)
            if not np.isfinite(probe_xy).all() or np.min(speed)<1e-5:
                attempt['rejected']='nonregular_spline';continue
            curvature=(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/(speed**3)
            maximum_curvature=float(np.max(np.abs(curvature)))
            attempt['maximum_curvature_per_m']=maximum_curvature
            if maximum_curvature>maximum_curvature_per_m*(1.+1e-5):
                attempt['rejected']='curvature_limit';continue
            # Distance to the original dense samples is a conservative upper
            # bound on distance to its piecewise-arc path; check both directions.
            deviation=float(max(original_tree.query(probe_xy)[0].max(),
                                cKDTree(probe_xy).query(original[:,:2])[0].max()))
            attempt['sampled_hausdorff_distance_m']=deviation
            if deviation>max_deviation_m:
                attempt['rejected']='deviation_limit';continue
            probe_s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(probe_xy,axis=0),axis=1))]
            count=max(2,int(math.ceil(probe_s[-1]/sample_spacing_m)))
            u=np.interp(np.linspace(0.,probe_s[-1],count+1),probe_s,probe_u)
            xy=evaluate(u);tangents=evaluate(u,1)
            yaw=np.arctan2(tangents[:,1],tangents[:,0])
            candidate=np.column_stack([xy,yaw])
            endpoint_error=float(np.max(np.linalg.norm(xy[[0,-1]]-original[[0,-1],:2],axis=1)))
            yaw_error=float(np.max(np.abs((yaw[[0,-1]]-original[[0,-1],2]+math.pi)%(2*math.pi)-math.pi)))
            attempt.update(endpoint_position_error_m=endpoint_error,endpoint_yaw_error_rad=yaw_error)
            if endpoint_error>.03 or yaw_error>.03:
                attempt['rejected']='endpoint_pose';continue
            # Check every returned sample with the full planning footprint and
            # continuous short chords at the task's stricter integration spacing.
            if not all(task.walkable(float(p[0]),float(p[1]),planning=True) for p in candidate):
                attempt['rejected']='planning_footprint_collision';continue
            if not all(task._clear_segment(a,b,planning=True) for a,b in zip(xy[:-1],xy[1:])):
                attempt['rejected']='continuous_segment_collision';continue
            attempt['accepted']=True
            diagnostics.update(accepted=True,selected_weighted_squared_residual=residual,
                               maximum_curvature_per_m=maximum_curvature,
                               sampled_hausdorff_distance_m=deviation,
                               endpoint_position_error_m=endpoint_error,endpoint_yaw_error_rad=yaw_error,
                               path_length_m=float(probe_s[-1]),returned_samples=len(candidate),
                               continuity='C2 regular cubic B-spline; endpoint position and tangent constrained')
            return candidate,diagnostics
        except (ValueError,TypeError,RuntimeError) as exc:
            attempt['rejected']='fit_error';attempt['error']=str(exc)
    diagnostics['reason']='no_candidate_passed_all_geometry_curvature_and_endpoint_checks'
    return original.copy(),diagnostics


def _wrap(a):
    return (float(a) + math.pi) % (2 * math.pi) - math.pi


def sample_arc(pose, curvature, length, max_step=.04):
    """Exact forward unicycle arc, excluding its initial pose."""
    x,y,yaw=map(float,pose)
    count=max(1,int(math.ceil(float(length)/float(max_step))))
    distances=np.linspace(length/count,length,count)
    if abs(curvature)<1e-10:
        return np.column_stack([x+distances*math.cos(yaw),y+distances*math.sin(yaw),
                                np.full(count,yaw)])
    angles=yaw+float(curvature)*distances
    return np.column_stack([x+(np.sin(angles)-math.sin(yaw))/curvature,
                            y-(np.cos(angles)-math.cos(yaw))/curvature,
                            (angles+math.pi)%(2*math.pi)-math.pi])


def _grid_guide(task, deadline=None):
    """Fast holonomic A* used only as an obstacle-aware search heuristic."""
    resolution=float(task.resolution)
    xmin,xmax,ymin,ymax=map(float,task.bounds)
    def cell(p):return (int(round((p[0]-xmin)/resolution)),int(round((p[1]-ymin)/resolution)))
    def xy(c):return np.array([xmin+c[0]*resolution,ymin+c[1]*resolution])
    cache={}
    def free(c):
        if c not in cache:
            px,py=xy(c)
            cache[c]=bool(task.walkable(float(px),float(py),planning=True))
        return cache[c]
    start=cell(task.pose[:2]);goal=np.asarray(task.target['position'][:2],float)
    if not free(start):
        # A continuous initial state may be free while its nearest lattice point
        # falls just over a cliff/obstacle boundary. Search nearby snap points.
        candidates=sorted(((start[0]+dx,start[1]+dy) for dx in (-1,0,1) for dy in (-1,0,1)),
                          key=lambda c:float(np.linalg.norm(xy(c)-task.pose[:2])))
        start=next((c for c in candidates if free(c) and task._clear_segment(task.pose[:2],xy(c),planning=True)),None)
        if start is None:raise RuntimeError('Hybrid A*: no safe holonomic start cell')
    def h(c):return max(0.,float(np.linalg.norm(xy(c)-goal))-1.4)
    queue=[(h(start),0.,start)];cost={start:0.};parent={};final=None;expansions=0
    directions=[(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if dx or dy]
    while queue:
        if deadline is not None and expansions%32==0 and time.monotonic()>deadline:
            raise RuntimeError('Hybrid A*: search budget exceeded during holonomic guide construction')
        _,g,c=heapq.heappop(queue)
        if g>cost[c]+1e-9:continue
        expansions+=1;point=xy(c);distance=float(np.linalg.norm(point-goal))
        if 1.0<=distance<=1.45 and task._target_line_of_sight(point):final=c;break
        for dx,dy in directions:
            nxt=(c[0]+dx,c[1]+dy)
            if not free(nxt):continue
            if dx and dy and not (free((c[0]+dx,c[1])) and free((c[0],c[1]+dy))):continue
            ng=g+resolution*math.hypot(dx,dy)
            if ng<cost.get(nxt,math.inf):
                cost[nxt]=ng;parent[nxt]=c
                heapq.heappush(queue,(ng+h(nxt),ng,nxt))
    if final is None:raise RuntimeError(f'Hybrid A*: target {task.target["id"]} is holonomically unreachable')
    cells=[final]
    while cells[-1]!=start:cells.append(parent[cells[-1]])
    points=np.asarray([task.pose[:2].copy()]+[xy(c) for c in reversed(cells)])
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    return points,lengths,expansions,len(cache)


def plan_hybrid(task, *, minimum_turning_radius_m=.5, primitive_length_m=.30,
                sample_spacing_m=.04, xy_resolution_m=.14, heading_bins=48,
                heuristic_weight=1.25, max_expansions=50000, max_seconds=60., smooth_path=True):
    """Return (dense N×3 poses, diagnostics) or explicitly raise on failure.

    The task's planning footprint already includes its physical rover radius and
    tracking margin. Independent final alignment may be performed by a controller;
    goal search itself attempts to finish facing the object within .35 radians.
    No old A* polyline is silently substituted if the curvature search fails.
    """
    started=time.monotonic()
    if minimum_turning_radius_m<=0 or primitive_length_m<=0 or not 0<sample_spacing_m<=.05:
        raise ValueError('Positive radius/primitive and sampling spacing <=.05m required')
    if heading_bins<12 or xy_resolution_m<=0 or max_expansions<1 or max_seconds<=0:
        raise ValueError('Invalid Hybrid A* search discretization or budget')
    initial=np.asarray(task.pose,dtype=float).copy()
    if initial.shape!=(3,) or not np.isfinite(initial).all():raise ValueError('Initial pose must be finite x,y,yaw')
    if not task.walkable(*initial[:2],planning=True):raise RuntimeError('Hybrid A*: initial footprint lacks planning clearance')
    guide,lengths,grid_expansions,grid_cells=_grid_guide(task,started+max_seconds)
    guide_seconds=time.monotonic()-started
    goal=np.asarray(task.target['position'][:2],float)
    xmin,xmax,ymin,ymax=map(float,task.bounds)
    def key(p):return (int(round((p[0]-xmin)/xy_resolution_m)),int(round((p[1]-ymin)/xy_resolution_m)),
                       int(round((_wrap(p[2])+math.pi)*heading_bins/(2*math.pi)))%heading_bins)
    def heuristic(p):
        d2=np.sum((guide-np.asarray(p)[:2])**2,axis=1)
        nearest=int(np.argmin(d2))
        # Reference remaining arclength encodes which side of rocks/loop walls
        # leads to the chosen viewing region. Off-guide penalty permits arcs.
        remaining=float(lengths[-1]-lengths[nearest])+math.sqrt(float(d2[nearest]))
        euclidean=max(0.,float(np.linalg.norm(np.asarray(p)[:2]-goal))-1.4)
        return max(euclidean,remaining)
    maximum_curvature=1./minimum_turning_radius_m
    curvatures=maximum_curvature*np.array([0.,-.25,.25,-.625,.625,-1.,1.])
    sequence=itertools.count()
    # Each immutable record keeps its own exact pose and parent edge. Duplicate
    # keys may be reopened without changing previously reconstructed ancestry.
    records=[(initial,-1,0.,0.)]
    best={key(initial):0.};queue=[(heuristic_weight*heuristic(initial),next(sequence),0)]
    expansions=0;collision_checks=0;goal_record=None
    while queue:
        if expansions>=max_expansions or time.monotonic()-started>max_seconds:
            raise RuntimeError(f'Hybrid A*: search budget exceeded ({expansions} expansions, '
                               f'{time.monotonic()-started:.2f}s); no fallback path returned')
        _,_,index=heapq.heappop(queue)
        pose,parent,g,previous_curvature=records[index]
        if g>best.get(key(pose),math.inf)+1e-9:continue
        expansions+=1
        delta=goal-pose[:2];distance=float(np.linalg.norm(delta))
        if 1.0<=distance<=1.5:
            error=abs(_wrap(math.atan2(delta[1],delta[0])-pose[2]))
            if error<=.35 and task._target_line_of_sight(pose[:2]):goal_record=index;break
        for curvature in curvatures:
            edge=sample_arc(pose,float(curvature),primitive_length_m,sample_spacing_m)
            successor=edge[-1]
            successor_key=key(successor)
            ng=g+primitive_length_m*(1.+.035*abs(curvature))+.020*abs(curvature-previous_curvature)
            if ng>=best.get(successor_key,math.inf)-1e-9:continue
            # End/mid checks reject blocked primitives before the full dense pass.
            order=[len(edge)-1,len(edge)//2]+[i for i in range(len(edge)) if i not in [len(edge)-1,len(edge)//2]]
            safe=True
            for j in order:
                collision_checks+=1
                if not task.walkable(float(edge[j,0]),float(edge[j,1]),planning=True):safe=False;break
            if not safe:continue
            best[successor_key]=ng;records.append((successor.copy(),index,ng,float(curvature)))
            heapq.heappush(queue,(ng+heuristic_weight*heuristic(successor),next(sequence),len(records)-1))
    if goal_record is None:raise RuntimeError('Hybrid A*: no curvature-feasible route to a visible viewing region')
    chain=[];index=goal_record
    while records[index][1]>=0:
        pose,parent,g,curvature=records[index]
        chain.append(sample_arc(records[parent][0],curvature,primitive_length_m,sample_spacing_m))
        index=parent
    path=np.concatenate([initial[None,:]]+list(reversed(chain)),axis=0)
    # Validate the exact returned samples with the same planning footprint, not
    # rounded heuristic cells. This catches reconstruction mistakes separately.
    if not all(task.walkable(float(p[0]),float(p[1]),planning=True) for p in path):
        raise RuntimeError('Hybrid A*: reconstructed path failed its full-footprint check')
    raw_samples=len(path)
    if smooth_path:
        path,smoothing=smooth_hybrid_path(task,path,maximum_curvature_per_m=maximum_curvature)
    else:
        smoothing={'accepted':False,'reason':'disabled_by_caller'}
    diagnostics={'planner':'Hybrid A*','implementation':'continuous SE(2) constant-curvature arc search',
                 'motion_model':'forward unicycle','minimum_turning_radius_m':minimum_turning_radius_m,
                 'maximum_curvature_per_m':maximum_curvature,'primitive_length_m':primitive_length_m,
                 'sample_spacing_m':sample_spacing_m,'heading_bins':heading_bins,'xy_resolution_m':xy_resolution_m,
                 'heuristic':'obstacle-aware holonomic guide; weighted, no optimality guarantee',
                 'heuristic_weight':heuristic_weight,'grid_guide_expansions':grid_expansions,'grid_guide_cells':grid_cells,
                 'hybrid_expansions':expansions,'collision_samples_checked':collision_checks,
                 'guide_seconds':guide_seconds,'planning_seconds':time.monotonic()-started,
                 'path_length_m':float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum()),
                 'raw_path_length_m':primitive_length_m*len(chain),'raw_samples':raw_samples,
                 'primitives':len(chain),'samples':len(path),'smoothing':smoothing,
                 'fallback_used':False,'goal_distance_m':float(np.linalg.norm(path[-1,:2]-goal)),
                 'goal_facing_error_rad':abs(_wrap(math.atan2(*(goal-path[-1,:2])[::-1])-path[-1,2]))}
    return path,diagnostics
