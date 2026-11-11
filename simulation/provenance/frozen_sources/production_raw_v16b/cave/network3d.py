"""Pure-CPU candidate cave networks with independent three-dimensional floors.

This is an authored implicit-volume hypothesis, not a lunar reconstruction or a
vehicle-dynamics model. Coordinates and profile dimensions are metres/Z-up.
An edge centerline lies on its local floor. Left/right widths are the maximum
lateral extents from that floor line, not half of a supplied total width.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.optimize import brentq
from skimage.measure import marching_cubes, euler_number


def _number(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def validate_network(network: dict) -> dict:
    """Return a JSON-friendly schema/graph report without changing the input."""
    errors, warnings = [], []
    if not isinstance(network, dict):
        return dict(valid=False, passed=False, errors=['network must be a dictionary'], warnings=[])
    nodes, edges = network.get('nodes', []), network.get('edges', [])
    if not isinstance(nodes, list) or not nodes: errors.append('nodes must be a nonempty list')
    if not isinstance(edges, list) or not edges: errors.append('edges must be a nonempty list')
    if errors: return dict(valid=False, passed=False, errors=errors, warnings=warnings)
    xyz, ids = {}, set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get('id'), str) or not node['id']:
            errors.append(f'nodes[{i}] requires a nonempty string id'); continue
        name = node['id']
        if name in xyz: errors.append('duplicate node id: ' + name)
        try:
            point = np.asarray(node['xyz_m'], dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all(): raise ValueError()
        except (KeyError, TypeError, ValueError):
            errors.append('invalid finite xyz_m for node ' + name); continue
        xyz[name] = point
    adjacency = {name: set() for name in xyz}
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict) or not isinstance(edge.get('id'), str) or not edge['id']:
            errors.append(f'edges[{i}] requires a nonempty string id'); continue
        name = edge['id']
        if name in ids: errors.append('duplicate edge id: ' + name)
        ids.add(name)
        a, b = edge.get('from'), edge.get('to')
        if not isinstance(a, str) or not isinstance(b, str) or a not in xyz or b not in xyz:
            errors.append('unknown edge endpoint: ' + name); continue
        adjacency[a].add(b); adjacency[b].add(a)
        try:
            points = np.asarray(edge['floor_centerline_xyz_m'], dtype=float)
            if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or not np.isfinite(points).all(): raise ValueError()
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            if np.any(lengths < 1e-6): raise ValueError()
        except (KeyError, TypeError, ValueError):
            errors.append('invalid or repeated floor centerline points: ' + name); continue
        if np.linalg.norm(points[0]-xyz[a]) > 1e-5 or np.linalg.norm(points[-1]-xyz[b]) > 1e-5:
            errors.append('edge centerline endpoints differ from node xyz: ' + name)
        profile = edge.get('cross_section', {})
        if not isinstance(profile, dict): errors.append('cross_section must be a dictionary: ' + name); continue
        for key in ['left_width_m', 'right_width_m', 'height_m']:
            if not _number(profile.get(key)) or float(profile[key]) <= 0:
                errors.append('positive finite ' + key + ' required: ' + name)
        offset, sag = profile.get('roof_offset_m', 0.), profile.get('roof_sag_m', 0.)
        if not _number(offset) or not _number(sag):
            errors.append('roof offset/sag must be finite: ' + name); continue
        if all(_number(profile.get(k)) and float(profile[k]) > 0 for k in ['left_width_m', 'right_width_m', 'height_m']):
            if not -float(profile['right_width_m']) < float(offset) < float(profile['left_width_m']):
                errors.append('roof_offset_m must lie strictly between the lateral walls: ' + name)
            if not 0 <= float(sag) < float(profile['height_m']):
                errors.append('roof_sag_m must be nonnegative and smaller than height_m: ' + name)
        if np.any(np.hypot(np.diff(points, axis=0)[:,0], np.diff(points, axis=0)[:,1]) < 1e-6):
            warnings.append('near-vertical segment has no rover support-plane interpretation: ' + name)
    for crossing in network.get('nonconnecting_crossings', []):
        if not isinstance(crossing, dict): errors.append('crossing must be a dictionary'); continue
        if crossing.get('edge_a') not in ids or crossing.get('edge_b') not in ids or crossing.get('edge_a') == crossing.get('edge_b'):
            errors.append('crossing must identify two distinct existing edges')
        try:
            point = np.asarray(crossing['xy_m'], dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all(): raise ValueError()
        except (KeyError, TypeError, ValueError): errors.append('crossing xy_m must be finite xy')
        gap = crossing.get('min_rock_thickness_m')
        if not _number(gap) or gap < 0: errors.append('crossing min_rock_thickness_m must be finite/nonnegative')
    components, unseen = [], set(xyz)
    while unseen:
        pending = [min(unseen)]; group = set()
        while pending:
            name = pending.pop()
            if name in group: continue
            group.add(name); pending.extend(adjacency[name] - group)
        unseen -= group; components.append(sorted(group))
    if len(components) != 1: warnings.append('declared graph is not connected')
    if network.get('units', 'm') != 'm' or network.get('up_axis', 'Z') != 'Z':
        errors.append('this implementation requires metres and Z-up')
    return dict(valid=not errors, passed=not errors, errors=errors, warnings=warnings,
                node_count=len(nodes), edge_count=len(edges), graph_component_count=len(components),
                graph_components=components, graph_cycle_rank=len(edges)-len(nodes)+len(components),
                expected_air_euler=len(nodes)-len(edges),
                expected_boundary_mesh_euler=2*(len(nodes)-len(edges)),
                width_semantics='left/right maximum lateral extents from the floor centerline',
                artificial_caps_not_geological_endpoints=True, no_observed_entrance=True)


def _segments(network):
    check = validate_network(network)
    if not check['valid']: raise ValueError('; '.join(check['errors']))
    segments = []
    for edge in network['edges']:
        p = np.asarray(edge['floor_centerline_xyz_m'], dtype=float)
        vectors = np.diff(p, axis=0); lengths = np.linalg.norm(vectors, axis=1)
        cumulative = np.r_[0., np.cumsum(lengths)]
        profile = edge['cross_section']
        left, right, height = (float(profile[k]) for k in ['left_width_m','right_width_m','height_m'])
        offset, sag = float(profile.get('roof_offset_m', 0.)), float(profile.get('roof_sag_m', 0.))
        cap_radius = min(left, right, height) * .65
        for i, (a, b, direction, length) in enumerate(zip(p[:-1], p[1:], vectors, lengths)):
            tangent = direction / length
            up = np.array([0., 0., 1.]) - tangent[2]*tangent
            if np.linalg.norm(up) < 1e-8:
                ref = np.eye(3)[int(np.argmin(abs(tangent)))]
                up = ref - np.dot(ref, tangent)*tangent
            up /= np.linalg.norm(up)
            lateral = np.cross(up, tangent); lateral /= np.linalg.norm(lateral)
            segments.append(dict(edge_id=edge['id'], a=a, b=b, tangent=tangent, up=up,
                lateral=lateral, length=float(length), edge_length=float(cumulative[-1]),
                edge_start_s=float(cumulative[i]), left=left, right=right, height=height,
                roof_offset=offset, roof_sag=sag, cap_radius=cap_radius))
    return segments


def _segment_field(x, y, z, segment):
    """Asymmetric upper arch with floor plane and rounded longitudinal caps."""
    a = segment['a']; t = segment['tangent']; lateral = segment['lateral']; up = segment['up']
    dx, dy, dz = x-a[0], y-a[1], z-a[2]
    along = dx*t[0]+dy*t[1]+dz*t[2]
    s = np.clip(along, 0., segment['length'])
    u = dx*lateral[0]+dy*lateral[1]+dz*lateral[2]
    v = dx*up[0]+dy*up[1]+dz*up[2]
    offset = segment['roof_offset']
    radius = np.where(u >= offset, segment['left']-offset, segment['right']+offset)
    radial2 = ((u-offset)/radius)**2 + ((along-s)/segment['cap_radius'])**2
    edge_s = segment['edge_start_s'] + s
    sag = segment['roof_sag'] * np.exp(-((u-offset)/(.22*(segment['left']+segment['right'])))**2)
    sag = sag * np.exp(-((edge_s-.5*segment['edge_length'])/max(.18*segment['edge_length'], 1e-6))**2)
    roof = segment['height']*np.sqrt(np.maximum(0., 1.-radial2))-sag
    rim = (1.-np.sqrt(radial2))*min(segment['left'],segment['right'])
    return np.minimum(np.minimum(v, roof-v), rim)


def _evaluate_segments(segments, points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1:] != (3,) or not np.isfinite(p).all():
        raise ValueError('points must be a finite Nx3 array')
    result = np.full(len(p), -np.inf, dtype=float)
    for segment in segments:
        np.maximum(result, _segment_field(p[:,0],p[:,1],p[:,2],segment), out=result)
    return result


def evaluate_field(network: dict, pointsNx3) -> np.ndarray:
    """Positive in candidate air; values are margins, not a Euclidean SDF."""
    return _evaluate_segments(_segments(network), pointsNx3)


def _edge_samples(segments, step):
    rows = {}
    for segment in segments:
        count = max(2, int(math.ceil(segment['length']/step))+1)
        points = segment['a'] + np.linspace(0.,1.,count)[:,None]*(segment['b']-segment['a'])
        rows.setdefault(segment['edge_id'], []).append(points)
    return {name: np.concatenate(parts) for name, parts in rows.items()}


def _air_intervals(segments, xy, zmin, zmax, sample_step):
    z = np.linspace(zmin,zmax,max(2,int(math.ceil((zmax-zmin)/sample_step))+1))
    points = np.column_stack([np.full(len(z),xy[0]),np.full(len(z),xy[1]),z])
    values = _evaluate_segments(segments,points); positive = values > 0
    cuts = []
    for i in np.flatnonzero(positive[:-1] != positive[1:]):
        root = brentq(lambda zz: float(_evaluate_segments(segments,np.array([[xy[0],xy[1],zz]]))[0]),float(z[i]),float(z[i+1]),xtol=1e-8)
        cuts.append(float(root))
    if positive[0]: cuts.insert(0,float(z[0]))
    if positive[-1]: cuts.append(float(z[-1]))
    if len(cuts) % 2: raise RuntimeError('Unpaired vertical air interval boundary')
    return [[cuts[i],cuts[i+1]] for i in range(0,len(cuts),2)]


def _crossing_report(network, segments, low, high, spacing):
    results = []
    for crossing in network.get('nonconnecting_crossings',[]):
        xy = np.asarray(crossing['xy_m'],dtype=float); sample_step = min(spacing/4.,.25)
        groups = {key:[s for s in segments if s['edge_id']==crossing[key]] for key in ['edge_a','edge_b']}
        intervals = {key:_air_intervals(group,xy,low[2],high[2],sample_step) for key,group in groups.items()}
        all_intervals = _air_intervals(segments,xy,low[2],high[2],sample_step)
        gaps=[]
        for a in intervals['edge_a']:
            for b in intervals['edge_b']:
                if max(a[0],b[0]) <= min(a[1],b[1]): gaps.append(0.); continue
                lower,upper=(a,b) if a[0]<b[0] else (b,a)
                # Any other candidate void between these two layers must also
                # retain the minimum solid separator; a bridging void gives0.
                lower_id=next((i for i,v in enumerate(all_intervals) if v[0]-1e-6<=.5*sum(lower)<=v[1]+1e-6),None)
                upper_id=next((i for i,v in enumerate(all_intervals) if v[0]-1e-6<=.5*sum(upper)<=v[1]+1e-6),None)
                if lower_id is None or upper_id is None: continue
                if lower_id==upper_id:gaps.append(0.)
                else:gaps.append(min(all_intervals[i+1][0]-all_intervals[i][1] for i in range(lower_id,upper_id)))
        gap=min(gaps) if gaps else None
        results.append(dict(edge_a=crossing['edge_a'],edge_b=crossing['edge_b'],xy_m=xy.tolist(),
            edge_a_air_intervals_z_m=intervals['edge_a'],edge_b_air_intervals_z_m=intervals['edge_b'],
            union_air_intervals_z_m=all_intervals,minimum_actual_rock_thickness_m=gap,
            required_rock_thickness_m=float(crossing['min_rock_thickness_m']),
            passed=gap is not None and gap+1e-5>=float(crossing['min_rock_thickness_m']),
            method='vertical implicit-air intervals with bracketed boundary roots',sample_step_m=sample_step))
    return results


def _triangle_areas(vertices, faces):
    # Check the exact final float32 point positions using float64 arithmetic;
    # float32 cross products can themselves round near-collinear triangles.
    triangles = vertices[faces].astype(np.float64)
    return np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                   triangles[:,2]-triangles[:,0]),axis=1)*.5


def _collapse_tiny_mesh_edges(vertices, faces, spacing):
    """Collapse only sub-voxel edges in tiny faces, preserving a 2-manifold.

    Float32 world-coordinate conversion can turn marching-cubes intersections
    near a grid corner into micrometre slivers. Merely deleting such a triangle
    opens a hole. Instead collapse an actual mesh edge only when its two vertex
    links intersect in exactly the two opposite vertices of its incident faces.
    This is the closed-triangle-manifold edge-collapse link condition.
    """
    vertices = np.asarray(vertices,dtype=np.float32)
    faces = np.asarray(faces,dtype=np.int32).copy()
    original = vertices.copy(); representative = np.arange(len(vertices))
    tolerance = float(spacing)*1e-4
    minimum_area = 1e-10*float(spacing)**2  # The existing acceptance threshold.
    areas = _triangle_areas(vertices,faces)
    initial_minimum_area = float(areas.min())
    before = int(np.count_nonzero(areas<=minimum_area)); steps=[]
    # Deterministic finite repair, with no growing tolerance or acceptance change.
    for _ in range(min(1000,max(1,before*4))):
        bad = np.flatnonzero(areas<=minimum_area)
        if not len(bad): break
        changed = False
        for face_index in bad:
            tri = faces[face_index]
            candidates=[]
            for u,v in [(int(tri[0]),int(tri[1])),(int(tri[1]),int(tri[2])),(int(tri[2]),int(tri[0]))]:
                length=float(np.linalg.norm(vertices[u].astype(float)-vertices[v]))
                if u!=v and length<=tolerance:candidates.append((length,min(u,v),max(u,v)))
            for length,u,v in sorted(candidates):
                has_u=np.any(faces==u,axis=1);has_v=np.any(faces==v,axis=1)
                incident=np.flatnonzero(has_u & has_v)
                if len(incident)!=2:continue
                opposite=set(map(int,faces[incident].ravel()))-{u,v}
                link_u=set(map(int,faces[has_u].ravel()))-{u,v}
                link_v=set(map(int,faces[has_v].ravel()))-{u,v}
                if len(opposite)!=2 or link_u.intersection(link_v)!=opposite:continue
                # Also compare the one-dimensional links. On a tetrahedral
                # shell both links contain the edge joining the opposite
                # vertices; collapsing would create duplicate opposite faces.
                opposite_ids=list(opposite)
                common_link_edge=(np.any(np.isin(faces[has_u],opposite_ids).sum(axis=1)==2)
                                  and np.any(np.isin(faces[has_v],opposite_ids).sum(axis=1)==2))
                if common_link_edge:continue
                for keep,drop in [(u,v),(v,u)]:
                    members=(representative==keep)|(representative==drop)
                    displacement=np.linalg.norm(original[members].astype(float)-vertices[keep],axis=1)
                    if bool(np.any(displacement>tolerance)):continue
                    rewritten=faces.copy();rewritten[rewritten==drop]=keep
                    repeated=(rewritten[:,0]==rewritten[:,1])|(rewritten[:,1]==rewritten[:,2])|(rewritten[:,2]==rewritten[:,0])
                    # Exactly the two faces incident on this edge disappear as
                    # part of the collapse; the rest of their star is rewired.
                    if int(np.count_nonzero(repeated))!=2:continue
                    steps.append(dict(face_index_at_collapse=int(face_index),
                        kept_vertex_id=keep,removed_vertex_id=drop,edge_length_m=length,
                        removed_incident_faces=2,link_condition_passed=True))
                    representative[representative==drop]=keep
                    faces=rewritten[~repeated];areas=_triangle_areas(vertices,faces)
                    changed=True;break
                if changed:break
            if changed:break
        if not changed:break
    displacement=np.linalg.norm(original.astype(float)-vertices[representative],axis=1)
    used,remap=np.unique(faces.ravel(),return_inverse=True)
    cleaned_vertices=vertices[used];cleaned_faces=remap.reshape(-1,3).astype(np.int32)
    after_areas=_triangle_areas(cleaned_vertices,cleaned_faces)
    report=dict(method='short mesh-edge collapse with closed-manifold link condition',
        tolerance_m=tolerance,unchanged_minimum_area_m2=minimum_area,
        tiny_faces_before=before,tiny_faces_after=int(np.count_nonzero(after_areas<=minimum_area)),
        collapsed_edges=len(steps),removed_incident_faces=2*len(steps),
        maximum_vertex_displacement_m=float(displacement.max()) if len(displacement) else 0.,
        minimum_face_area_before_m2=initial_minimum_area,
        minimum_face_area_after_m2=float(after_areas.min()),
        steps=steps,holes_are_not_patched_by_unrelated_faces=True)
    return cleaned_vertices,cleaned_faces,report


def build_network_geometry(network: dict, spacing_m: float = 1.0) -> dict:
    """Build a closed candidate air boundary and CPU-only geometric checks.

    The report separates synthetic air connectivity from a sampled rover
    envelope. No collision cooking, rendering, dynamics or policy is executed.
    """
    validation=validate_network(network)
    if not validation['valid']:raise ValueError('; '.join(validation['errors']))
    if not _number(spacing_m) or float(spacing_m)<=0:raise ValueError('spacing_m must be positive/finite')
    spacing=float(spacing_m); segments=_segments(network)
    # Conservative padded AABB, independent of the old flat-floor scene bounds.
    points=np.concatenate([np.stack([s['a'],s['b']]) for s in segments])
    padding=max(max(s['left'],s['right'],s['height']) for s in segments)+3*spacing
    low=np.floor((points.min(0)-padding)/spacing)*spacing
    high=np.ceil((points.max(0)+padding)/spacing)*spacing
    shape=np.rint((high-low)/spacing).astype(np.int64)+1
    cells=int(np.prod(shape))
    if cells>80_000_000:raise ValueError(f'Requested grid has {cells} voxels; increase spacing or use smaller bounds')
    axes=[low[i]+np.arange(int(shape[i]),dtype=np.float32)*spacing for i in range(3)]
    volume=np.full(tuple(shape),-1e6,dtype=np.float32)
    for segment in segments:
        reach=max(segment['left'],segment['right'],segment['height'])+2*spacing
        a=np.minimum(segment['a'],segment['b'])-reach;b=np.maximum(segment['a'],segment['b'])+reach
        first=np.maximum(0,np.floor((a-low)/spacing).astype(int));last=np.minimum(shape,np.ceil((b-low)/spacing).astype(int)+1)
        for start in range(int(first[0]),int(last[0]),16):
            stop=min(start+16,int(last[0]));sl=(slice(start,stop),slice(first[1],last[1]),slice(first[2],last[2]))
            x=axes[0][start:stop,None,None];y=axes[1][None,first[1]:last[1],None];z=axes[2][None,None,first[2]:last[2]]
            value=_segment_field(x,y,z,segment)
            np.maximum(volume[sl],value,out=volume[sl])
    air=volume>0
    if not bool(air.any()):raise RuntimeError('No cavity resolved at requested voxel spacing')
    air_labels,components=ndimage.label(air,structure=ndimage.generate_binary_structure(3,1))
    air_euler=int(euler_number(air,connectivity=3))
    boundary_positive=bool(air[[0,-1],:,:].any() or air[:,[0,-1],:].any() or air[:,:,[0,-1]].any())
    vertices,faces,_,_=marching_cubes(volume,0.,spacing=(spacing,)*3,gradient_direction='descent',allow_degenerate=False)
    vertices=(vertices+low).astype(np.float32);faces=faces.astype(np.int32)
    used,remap=np.unique(faces.ravel(),return_inverse=True);vertices=vertices[used];faces=remap.reshape(-1,3).astype(np.int32)
    vertices,faces,cleanup=_collapse_tiny_mesh_edges(vertices,faces,spacing)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    mesh_euler=int(len(vertices)-len(counts)+len(faces))
    areas=_triangle_areas(vertices,faces)
    finite=bool(np.isfinite(vertices).all() and np.isfinite(areas).all());degenerate=int(np.count_nonzero(areas<=1e-10*spacing**2))
    samples=_edge_samples(segments,min(spacing/2.,.5));probe_rows=[];all_points=[];slope_rows=[]
    envelope=network.get('robot_envelope',{});radius=float(envelope.get('radius_m',.4));max_slope=float(envelope.get('max_slope_deg',12.))
    if radius<=0 or not math.isfinite(radius) or max_slope<=0 or not math.isfinite(max_slope):raise ValueError('Invalid robot_envelope radius/slope')
    for edge in network['edges']:
        p=samples[edge['id']];probe=p+np.array([0.,0.,1.]);values=_evaluate_segments(segments,probe)
        locations=np.rint((probe-low)/spacing).astype(int);locations=np.clip(locations,0,shape-1)
        labels=air_labels[tuple(locations.T)];all_points.extend(labels.tolist())
        offsets=np.array([[0.,0.,0.]]+[[radius*math.cos(a),radius*math.sin(a),0.] for a in np.linspace(0,2*np.pi,12,endpoint=False)])
        robot=np.concatenate([(p[:,None,:]+offsets[None,:,:]+[0.,0.,height]).reshape(-1,3) for height in [.5,1.]])
        robot_values=_evaluate_segments(segments,robot)
        control=np.asarray(edge['floor_centerline_xyz_m'],dtype=float);d=np.diff(control,axis=0)
        slopes=np.degrees(np.arctan2(abs(d[:,2]),np.hypot(d[:,0],d[:,1])))
        slope_ok=bool(np.all(slopes<=max_slope+1e-8))
        probe_rows.append(dict(edge_id=edge['id'],samples=len(p),probe='floor centerline + world Z1m',
            all_analytically_inside_air=bool(np.all(values>0)),minimum_implicit_margin_not_euclidean_distance_m=float(values.min()),
            sampled_air_components=sorted(set(map(int,labels))),
            rover_probe_heights_above_floor_m=[.5,1.],rover_probe_radius_m=radius,
            sampled_rover_envelope_inside_air=bool(np.all(robot_values>0)),
            minimum_rover_implicit_margin_not_euclidean_distance_m=float(robot_values.min()),
            slope_within_assumed_limit=slope_ok))
        slope_rows.append(dict(edge_id=edge['id'],segment_slopes_deg=slopes.tolist(),maximum_slope_deg=float(slopes.max()),
            floor_z_range_m=[float(control[:,2].min()),float(control[:,2].max())],
            centerline_length_m=float(np.linalg.norm(d,axis=1).sum()),assumed_rover_max_slope_deg=max_slope,passed=slope_ok))
    actual_labels=sorted(set(map(int,all_points)));analytic_ok=all(r['all_analytically_inside_air'] for r in probe_rows)
    # A finite grid can place a witness exactly on a boundary voxel. Report this
    # independently from the continuous-field test; it is not silently ignored.
    centerline_connected=analytic_ok and len(actual_labels)==1 and actual_labels[0]!=0
    crossings=_crossing_report(network,segments,low,high,spacing)
    topology_ok=(components==validation['graph_component_count'] and air_euler==validation['expected_air_euler']
                 and mesh_euler==validation['expected_boundary_mesh_euler'])
    mesh_ok=finite and not degenerate and not int(np.count_nonzero(counts!=2)) and not boundary_positive
    report=dict(passed=bool(mesh_ok and topology_ok and centerline_connected and all(x['passed'] for x in crossings)),
        validation=validation,method='3D asymmetric local arches, rounded caps, exact implicit union, marching cubes',
        implicit_field_is_euclidean_signed_distance=False,units='m',up_axis='Z',voxel_spacing_m=spacing,
        grid_shape=shape.tolist(),grid_voxels=cells,grid_bounds_m=[low.tolist(),high.tolist()],
        mesh=dict(vertices=len(vertices),faces=len(faces),unique_edges=len(counts),finite=finite,
            degenerate_faces=degenerate,boundary_edges=int(np.count_nonzero(counts==1)),
            nonmanifold_edges=int(np.count_nonzero(counts>2)),all_edges_incident_to_two_faces=bool(np.all(counts==2)),
            euler_characteristic=mesh_euler,expected_euler_characteristic=validation['expected_boundary_mesh_euler'],
            subvoxel_cleanup=cleanup,
            bounds_m=[vertices.min(0).tolist(),vertices.max(0).tolist()],air_touches_grid_boundary=boundary_positive),
        candidate_air_connectivity=dict(connected_components_6_neighbor=int(components),expected_graph_components=validation['graph_component_count'],
            voxel_euler_26_neighbor=air_euler,expected_voxel_euler=validation['expected_air_euler'],
            air_voxel_count=int(np.count_nonzero(air)),topology_matches_declared_graph=bool(topology_ok),
            centerline_air_connected=bool(centerline_connected),centerline_sample_component_ids=actual_labels),
        edge_probes=probe_rows,edge_slopes=slope_rows,nonconnecting_crossings=crossings,
        sampled_rover_passability=dict(passed=all(r['sampled_rover_envelope_inside_air'] and r['slope_within_assumed_limit'] for r in probe_rows),
            scope='sampled candidate geometry and assumed slope limit; junction floor ledges and continuous surface traversal are not certified; no wheel dynamics or PhysX test'),
        no_observed_entrance=True,artificial_caps_not_geological_endpoints=True,
        scientific_status='authored candidate geometry; no lunar reconstruction, stability, or probability claim',
        GPU_work=False,simulation_started=False,old_maps_modified=False)
    return dict(vertices=vertices,faces=faces,report=report)
