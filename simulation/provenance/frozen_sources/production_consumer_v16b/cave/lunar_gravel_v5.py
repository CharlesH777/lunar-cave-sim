"""Mesh-supported, instanced gravel for a derived lunar-cave display stage.

Only the supplied stage is authored. The base mesh/config remain read-only;
gravel is render detail without collision shapes or a particle simulation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


INSTANCER_PATH = '/World/SurfaceDetail/Gravel'
MATERIAL_PATH = '/World/Materials/GravelFragmentsV5'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _FloorProjector:
    """Highest upward-triangle hit using one-metre XY buckets, not a height map."""
    def __init__(self, vertices, faces):
        self.vertices=np.asarray(vertices,dtype=float)
        self.faces=np.asarray(faces,dtype=np.int32)
        tri=self.vertices[self.faces]
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        length=np.linalg.norm(normals,axis=1)
        normals/=np.maximum(length[:,None],1e-30)
        selected=(normals[:,2]>.5)&(tri[:,:,2].mean(1)<22.)&(length>1e-12)
        self.original_face_ids=np.flatnonzero(selected)
        self.triangles=tri[selected]
        self.normals=normals[selected]
        if not len(self.triangles):raise ValueError('No upward base triangles below Z22m')
        p=self.triangles
        self.denominator=(p[:,1,1]-p[:,2,1])*(p[:,0,0]-p[:,2,0])+(p[:,2,0]-p[:,1,0])*(p[:,0,1]-p[:,2,1])
        self.buckets={};self.neighborhood_cache={}
        low=np.floor(p[:,:,:2].min(1)).astype(int);high=np.floor(p[:,:,:2].max(1)).astype(int)
        for i,(a,b) in enumerate(zip(low,high)):
            for x in range(a[0],b[0]+1):
                for y in range(a[1],b[1]+1):self.buckets.setdefault((x,y),[]).append(i)

    def _candidates(self,key):
        key=tuple(map(int,key))
        if key not in self.neighborhood_cache:
            found=[]
            for x in range(key[0]-1,key[0]+2):
                for y in range(key[1]-1,key[1]+2):found.extend(self.buckets.get((x,y),()))
            self.neighborhood_cache[key]=np.unique(found).astype(np.int32)
        return self.neighborhood_cache[key]

    def project(self,xy):
        xy=np.asarray(xy,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Require finite Nx2 positions')
        heights=np.full(len(xy),np.nan);normal=np.zeros((len(xy),3));face_ids=np.full(len(xy),-1,dtype=np.int32)
        barycentric=np.full((len(xy),3),np.nan)
        keys,inverse=np.unique(np.floor(xy).astype(int),axis=0,return_inverse=True)
        for group,key in enumerate(keys):
            rows=np.flatnonzero(inverse==group);candidates=self._candidates(key)
            if not len(candidates):continue
            p=self.triangles[candidates];q=xy[rows]
            x=q[:,0,None];y=q[:,1,None];den=self.denominator[candidates][None,:]
            u=((p[None,:,1,1]-p[None,:,2,1])*(x-p[None,:,2,0])+(p[None,:,2,0]-p[None,:,1,0])*(y-p[None,:,2,1]))/den
            v=((p[None,:,2,1]-p[None,:,0,1])*(x-p[None,:,2,0])+(p[None,:,0,0]-p[None,:,2,0])*(y-p[None,:,2,1]))/den
            w=1.-u-v
            valid=(u>=-1e-7)&(v>=-1e-7)&(w>=-1e-7)&(u<=1.+1e-7)&(v<=1.+1e-7)&(w<=1.+1e-7)
            z=u*p[None,:,0,2]+v*p[None,:,1,2]+w*p[None,:,2,2]
            valid&=z<22.
            z=np.where(valid,z,-np.inf)
            best=np.argmax(z,axis=1);hit=z[np.arange(len(rows)),best]
            okay=np.isfinite(hit)
            if not np.any(okay):continue
            output_rows=rows[okay];columns=best[okay];chosen=candidates[columns]
            heights[output_rows]=hit[okay];normal[output_rows]=self.normals[chosen]
            face_ids[output_rows]=self.original_face_ids[chosen]
            barycentric[output_rows]=np.column_stack([u[okay,columns],v[okay,columns],w[okay,columns]])
        return heights,normal,face_ids,barycentric


def _prototype(seed):
    """Low-face fractured convex fragment; deliberately not an icosphere bean."""
    rng=np.random.default_rng(seed)
    corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    corners*=rng.uniform(.55,1.,corners.shape)
    extra=rng.uniform(-.8,.8,(6,3))
    for i in range(6):extra[i,i//2]=(-1. if i%2==0 else 1.)*rng.uniform(.8,1.1)
    points=np.vstack([corners,extra])
    shear=np.eye(3);shear[0,1:]=rng.uniform(-.25,.25,2);shear[1,2]=rng.uniform(-.2,.2)
    points=points@shear.T
    points-=(points.min(0)+points.max(0))*.5
    hull=ConvexHull(points)
    faces=hull.simplices.copy()
    tri=points[faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    flip=np.einsum('ij,ij->i',n,hull.equations[:,:3])<0
    faces[flip]=faces[flip][:,[0,2,1]]
    used,remap=np.unique(faces.ravel(),return_inverse=True)
    points=points[used];faces=remap.reshape(-1,3).astype(np.int32)
    pairs=points[:,None,:]-points[None,:,:]
    diameter=float(np.linalg.norm(pairs,axis=2).max())
    points*=2./diameter
    tri=points[faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=1,keepdims=True)
    a,b=np.triu_indices(len(points),1)
    points=points.astype(np.float32)
    return dict(vertices=points,faces=faces,normals=n.astype(np.float32),
                pairs=points[a].astype(float)-points[b].astype(float))


def _candidate_xy(rng,scene_config,count,group):
    if group=='branch':
        points=np.asarray(scene_config['branch_v10']['centerline_xy_m'],float)
        distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        samples=rng.uniform(distances[0],distances[-1],count)
        xx=np.interp(samples,distances,points[:,0]);yy=np.interp(samples,distances,points[:,1])
        index=np.clip(np.searchsorted(distances,samples,side='right')-1,0,len(points)-2)
        direction=points[index+1]-points[index];direction/=np.linalg.norm(direction,axis=1)[:,None]
        side=np.column_stack([-direction[:,1],direction[:,0]])
        widths=np.asarray(scene_config['branch_v10'].get('half_widths_m',[4.5]*len(points)),float)
        half_width=np.interp(samples,distances,widths)
        offset=rng.uniform(-.82,.82,count)*half_width
        return np.column_stack([xx,yy])+side*offset[:,None]
    if group=='focus':
        return np.column_stack([rng.uniform(11.,23.,count),rng.uniform(-3.5,3.5,count)])
    if group=='pit':
        cx,cy=scene_config['pit']['center_xy_m']
        theta=rng.uniform(-np.pi,np.pi,count);u=rng.random(count)
        rho=np.where(u<.68,rng.uniform(.56,.98,count),np.sqrt(rng.random(count))*.82)
        xy=np.column_stack([cx+26.*rho*np.cos(theta),cy+23.*rho*np.sin(theta)])
        clustered=u>.82
        centers=np.array([[cx-15.,cy+6.],[cx+12.,cy+14.],[cx+2.,cy-17.]])
        if np.any(clustered):xy[clustered]=centers[rng.integers(0,3,int(clustered.sum()))]+rng.normal(0,[4.5,3.5],(int(clustered.sum()),2))
        return xy
    x=rng.uniform(3.,78.,count)
    from .lunar_morphology_v4 import _axis_y
    axis=np.asarray(_axis_y(x),float);u=rng.random(count)
    offset=np.where(u<.70,rng.choice([-1.,1.],count)*rng.uniform(6.,12.,count),rng.normal(0,2.5,count))
    xy=np.column_stack([x,axis+offset])
    clustered=(u>=.70)&(u<.90)
    deposits=scene_config['cavern'].get('floor_deposits',[])
    if np.any(clustered) and deposits:
        chosen=[deposits[i] for i in rng.integers(0,len(deposits),int(clustered.sum()))]
        xc=np.array([d['x'] for d in chosen])+rng.normal(0,3.8,int(clustered.sum()))
        yc=np.asarray(_axis_y(xc))+np.array([d['u'] for d in chosen])+rng.normal(0,1.4,int(clustered.sum()))
        xy[clustered]=np.column_stack([xc,yc])
    return xy


def _place(projector,scene_config,settings,rng):
    from build_lunar_cave_v4_cpu import evaluate_scene_field
    count=settings['count'];focus=int(round(count*.022))
    targets=[('focus',focus),('pit',(count-focus)//2),('cavern',count-focus-(count-focus)//2)]
    if scene_config.get('branch_v10'):
        branch_count=int(round((count-focus)*.18));remaining=count-focus-branch_count
        targets=[('focus',focus),('pit',remaining//2),('cavern',remaining-remaining//2),('branch',branch_count)]
    support=[];normals=[];faces=[];bary=[];groups=[];field_values=[]
    attempted=0;no_support=0;field_rejected=0
    for group,target in targets:
        made=0
        for _ in range(40):
            if made>=target:break
            batch=min(8192,max(128,(target-made)*2))
            xy=_candidate_xy(rng,scene_config,batch,group);attempted+=batch
            z,n,f,b=projector.project(xy)
            valid=np.isfinite(z);no_support+=int(np.count_nonzero(~valid))
            if not np.any(valid):continue
            p=np.column_stack([xy[valid],z[valid]])
            nn=n[valid];ff=f[valid];bb=b[valid]
            # Actual MC triangles determine contact. The analytic field is only
            # checked above the triangle, beyond sub-voxel interpolation error.
            phi=evaluate_scene_field(scene_config,p+nn*.4)
            keep=phi>.02;field_rejected+=int(np.count_nonzero(~keep))
            take=np.flatnonzero(keep)[:target-made]
            if not len(take):continue
            support.append(p[take]);normals.append(nn[take]);faces.append(ff[take]);bary.append(bb[take]);field_values.append(phi[take])
            groups.extend([group]*len(take));made+=len(take)
        if made!=target:raise RuntimeError(f'Could only place {made}/{target} supported gravel in {group}')
    return dict(support=np.concatenate(support),normals=np.concatenate(normals),face_ids=np.concatenate(faces),
                barycentric=np.concatenate(bary),groups=np.asarray(groups),field_values=np.concatenate(field_values),
                attempted=attempted,no_support=no_support,field_rejected=field_rejected)


def _quaternions(normal,yaw):
    # q_align * q_yaw: local Z becomes the triangle normal; yaw changes planform.
    w=np.sqrt((1.+normal[:,2])*.5)
    x=-normal[:,1]/(2.*w);y=normal[:,0]/(2.*w)
    c=np.cos(yaw*.5);s=np.sin(yaw*.5)
    q=np.column_stack([w*c,x*c+y*s,y*c-x*s,w*s])
    q/=np.linalg.norm(q,axis=1,keepdims=True)
    return q


def _fragment_radii(prototypes, proto_index, scales):
    """Conservative centre-to-vertex radius, invariant to authored rotation."""
    radius=np.empty(len(proto_index),dtype=float)
    for i,proto in enumerate(prototypes):
        rows=np.flatnonzero(proto_index==i)
        if len(rows):
            radius[rows]=np.linalg.norm(proto['vertices'][None,:,:].astype(float)*scales[rows,None,:],axis=2).max(1)
    return radius


def _limit_route_fragments(scene_config,prototypes,proto_index,scales,bottom_radius,top_radius,
                           real_diameter,embedding,placement):
    """Shrink only potentially route-overlapping stones; consume no RNG values.

    A 3D vertex radius conservatively bounds the projected footprint after any
    rotation. The original sphere also bounds its shrunken/repositioned version:
    the normal-offset change is no greater than the removed sphere radius.
    """
    route=scene_config.get('rover_route')
    if route is None:return None
    if not isinstance(route,dict):raise ValueError('rover_route must be a route configuration object')
    from .rover_route_v6 import route_distance_xy
    half=float(route.get('drivable_half_width_m',1.2))
    max_diameter=min(.06,float(route.get('max_gravel_diameter_m',.06)))
    max_protrusion=min(.02,float(route.get('max_gravel_protrusion_m',.02)))
    if not np.isfinite([half,max_diameter,max_protrusion]).all() or half<=0 or min(max_diameter,max_protrusion)<=0:
        raise ValueError('Route width and gravel diameter/protrusion caps must be positive and finite')
    offset=bottom_radius*embedding
    initial_positions=placement['support']+placement['normals']*offset[:,None]
    radius=_fragment_radii(prototypes,proto_index,scales)
    distances=np.asarray(route_distance_xy(route,initial_positions[:,:2]),dtype=float)
    if distances.shape!=(len(scales),) or not np.isfinite(distances).all() or np.any(distances<0):
        raise ValueError('Route distance query must return one finite nonnegative distance per fragment')
    # Leave 2mm of horizontal guard against float32 position and quaternion
    # serialization, and 20um vertically against position rounding near a cap.
    guard=.002
    affected=distances<=half+radius+guard
    target_diameter=max_diameter*(1.-2e-5)
    target_protrusion=max_protrusion-max(2e-5,max_protrusion*1e-4)
    if target_protrusion<=0:raise ValueError('Route protrusion cap is below serialization safety margin')
    factor=np.ones(len(scales),dtype=float)
    factor[affected]=np.minimum(1.,np.minimum(target_diameter/real_diameter[affected],
        target_protrusion/(offset[affected]+top_radius[affected])))
    scales*=factor[:,None]
    bottom_radius*=factor;top_radius*=factor;real_diameter*=factor
    return dict(config=route,half_width=half,max_diameter=max_diameter,max_protrusion=max_protrusion,
                target_diameter=target_diameter,target_protrusion=target_protrusion,
                horizontal_guard=guard,affected=affected,scale_factor=factor,
                initial_overlap_distance=distances,initial_radius=radius)


def _route_gravel_report(state,prototypes,proto_index,authored_positions,authored_scales,
                         real_diameter,highest,lowest,actual_radius):
    """Verify serialized geometry, including centres outside the drive band."""
    from .rover_route_v6 import route_distance_xy
    radius=actual_radius
    distance=np.asarray(route_distance_xy(state['config'],authored_positions[:,:2]),dtype=float)
    actual=distance<=state['half_width']+radius+.001
    missed=actual&~state['affected']
    over_diameter=actual&(real_diameter>state['max_diameter']+1e-8)
    over_height=actual&(highest>state['max_protrusion']+1e-8)
    if np.any(missed|over_diameter|over_height):
        raise ValueError('Serialized route gravel violates the radius-expanded drive-band caps')
    selected=state['affected']
    return dict(passed=True,route_config_source='base scene_config.rover_route',
        corridor_full_width_m=state['half_width']*2,
        distance_method='nearest finite route polyline in XY',
        footprint_method='centre distance expanded by maximum actual scaled 3D vertex radius',
        guard_before_serialization_m=state['horizontal_guard'],guard_after_serialization_m=.001,
        pre_cap_potential_overlap_count=int(selected.sum()),
        actually_intersecting_or_guarded_count=int(actual.sum()),
        scaled_fragment_count=int(np.count_nonzero(state['scale_factor']<1.)),
        centres_outside_band_with_potential_footprint_overlap=int(np.count_nonzero(
            actual&(distance>state['half_width']))),
        maximum_allowed_full_diameter_m=state['max_diameter'],
        maximum_allowed_support_plane_protrusion_m=state['max_protrusion'],
        maximum_actual_full_diameter_m=float(real_diameter[actual].max()) if np.any(actual) else None,
        maximum_actual_support_plane_protrusion_m=float(highest[actual].max()) if np.any(actual) else None,
        minimum_actual_support_plane_offset_m=float(lowest[actual].min()) if np.any(actual) else None,
        diameter_cap_violation_count=int(over_diameter.sum()),protrusion_cap_violation_count=int(over_height.sum()),
        radius_guard_missed_count=int(missed.sum()),
        outside_potential_overlap_scaled_count=int(np.count_nonzero((~selected)&(state['scale_factor']!=1.))),
        minimum_scale_factor=float(state['scale_factor'].min()),
        post_serialization_correction_passes=int(state.get('correction_passes',0)),
        post_serialization_corrected_fragment_count=int(np.count_nonzero(state.get('corrected',np.zeros(len(actual),bool)))),
        policy='Only potential route overlaps shrink; original XY support positions, orientation RNG and outside-route scales are preserved',
        validation_scope='actual USD instance matrices including half-precision quaternion serialization, original vertices and sampled support planes; no tire dynamics')


def _authored_fragment_geometry(instancer,prototypes,proto_index,placement):
    """Measure the actual composed USD transforms, including half quaternions."""
    matrices=np.asarray(instancer.ComputeInstanceTransformsAtTime(
        Usd.TimeCode.Default(),Usd.TimeCode.Default()),dtype=float)
    count=len(proto_index)
    if matrices.shape!=(count,4,4) or not np.isfinite(matrices).all():
        raise ValueError('USD did not produce one finite instance matrix per fragment')
    lowest=np.empty(count);highest=np.empty(count);diameter=np.empty(count);radius=np.empty(count)
    for i,proto in enumerate(prototypes):
        rows=np.flatnonzero(proto_index==i)
        if not len(rows):continue
        homogeneous=np.column_stack([proto['vertices'].astype(float),np.ones(len(proto['vertices']))])
        world=np.einsum('vj,ijk->ivk',homogeneous,matrices[rows])[:,:,:3]
        offsets=np.einsum('ivj,ij->iv',world-placement['support'][rows,None,:],placement['normals'][rows])
        lowest[rows]=offsets.min(1);highest[rows]=offsets.max(1)
        radius[rows]=np.linalg.norm(world-matrices[rows,3,None,:3],axis=2).max(1)
        world_pairs=np.einsum('vj,ijk->ivk',proto['pairs'],matrices[rows,:3,:3])
        diameter[rows]=np.linalg.norm(world_pairs,axis=2).max(1)
    return lowest,highest,diameter,radius


def _recheck_route_serialization(instancer,state,prototypes,proto_index,embedding,placement,ideal_positions):
    """Deterministically correct tiny quaternion/float serialization expansion.

    Half-quaternion matrices can change a nominal 6cm diameter by tens of
    micrometres. Corrections shrink only previously route-guarded fragments;
    positions/scales of every other fragment remain exactly as first authored.
    """
    ideal_positions=ideal_positions.copy()
    corrected=np.zeros(len(proto_index),dtype=bool);state['correction_passes']=0
    for attempt in range(3):
        position=np.asarray(instancer.GetPositionsAttr().Get(),float)
        scales=np.asarray(instancer.GetScalesAttr().Get(),float)
        lowest,highest,diameter,radius=_authored_fragment_geometry(instancer,prototypes,proto_index,placement)
        needs=state['affected']&((diameter>state['max_diameter'])|(highest>state['max_protrusion']))
        if not np.any(needs):
            state['corrected']=corrected
            return ideal_positions,position,scales,lowest,highest,diameter,radius
        if attempt==2:raise ValueError('Route gravel still exceeds caps after bounded serialization correction')
        factor=np.ones(len(proto_index))
        factor[needs]=np.minimum(state['target_diameter']/diameter[needs],state['target_protrusion']/highest[needs])
        scales[needs]*=factor[needs,None]
        for i,proto in enumerate(prototypes):
            rows=np.flatnonzero(needs&(proto_index==i))
            if len(rows):
                bottom=-float(proto['vertices'][:,2].min())*scales[rows,2]
                ideal_positions[rows]=placement['support'][rows]+placement['normals'][rows]*(bottom*embedding[rows])[:,None]
        position[needs]=ideal_positions[needs]
        instancer.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales.astype(np.float32)))
        instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(position.astype(np.float32)))
        state['scale_factor'][needs]*=factor[needs];corrected|=needs
        state['correction_passes']+=1


def _material(stage):
    if stage.GetPrimAtPath(MATERIAL_PATH):raise ValueError('Gravel material path already exists')
    mat=UsdShade.Material.Define(stage,MATERIAL_PATH)
    shader=UsdShade.Shader.Define(stage,MATERIAL_PATH+'/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    reader=UsdShade.Shader.Define(stage,MATERIAL_PATH+'/Color');reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor')
    reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(),'result')
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.94)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    return mat


def _size_mixture(count,fractions):
    """Validate global size weights without changing the legacy rounding."""
    try:fractions=np.asarray(fractions,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError('gravel size_fractions must contain three finite nonnegative values summing to1') from exc
    if (fractions.shape!=(3,) or not np.isfinite(fractions).all() or np.any(fractions<0)
            or not np.isclose(float(fractions.sum()),1.,rtol=0.,atol=1e-12)):
        raise ValueError('gravel size_fractions must contain three finite nonnegative values summing to1')
    # Preserve the original first-two floors and final remainder exactly. This
    # also consumes no RNG, so omitted/default weights reproduce old instances.
    counts=[int(count*fractions[0]),int(count*fractions[1])]
    counts.append(count-sum(counts))
    if min(counts)<0:raise ValueError('gravel size_fractions produced negative class counts')
    return fractions.tolist(),counts


def add_gravel(stage,base_scene_dir,config=None):
    """Add about15k partially buried mesh-supported instances to a derived stage.

    Options may be under config['gravel'] or passed directly: count (15000),
    prototype_count (12), seed (20261110), size_fractions ([.75,.22,.03]) for
    3--12cm/12--30cm/30--55cm full diameters. Weights must be three finite
    nonnegative values summing to1. Source scene_config.json controls the
    geometry/field checks and is never replaced by the appearance configuration.
    """
    source=Path(base_scene_dir).resolve(strict=True)
    mesh_file=source/'mesh.npz';cfg_file=source/'scene_config.json'
    before={str(p):_sha(p) for p in [mesh_file,cfg_file]}
    scene_config=json.loads(cfg_file.read_text())
    supplied={} if config is None else config.get('gravel',config)
    settings=dict(count=int(supplied.get('count',15000)),prototype_count=int(supplied.get('prototype_count',12)),seed=int(supplied.get('seed',20261110)))
    if not 1<=settings['count']<=100000:raise ValueError('gravel count must be between1 and100000')
    if not 8<=settings['prototype_count']<=16:raise ValueError('Use8 to16 fracture prototypes')
    size_fractions,class_counts=_size_mixture(settings['count'],supplied.get('size_fractions',[.75,.22,.03]))
    if stage.GetPrimAtPath(INSTANCER_PATH):raise ValueError('Refusing to replace an existing gravel instancer')
    data=np.load(mesh_file);projector=_FloorProjector(data['vertices'],data['faces']);data.close()
    rng=np.random.default_rng(settings['seed']);placement=_place(projector,scene_config,settings,rng)
    count=settings['count'];prototype_count=settings['prototype_count']
    prototypes=[_prototype(settings['seed']+7919*(i+1)) for i in range(prototype_count)]
    # Full diameter means actual longest scaled vertex-to-vertex distance.
    focus=int(np.count_nonzero(placement['groups']=='focus'))
    # Closeup samples retain their historical small-fragment preference, but a
    # globally sparse/zero small class must never request a negative array size.
    focus_small=min(focus,class_counts[0])
    size_class=np.r_[np.zeros(focus_small,int),rng.permutation(np.r_[np.zeros(class_counts[0]-focus_small,int),np.ones(class_counts[1],int),np.full(class_counts[2],2,int)])]
    diameter=np.empty(count)
    for cls,(lo,hi) in enumerate([(.03,.12),(.12,.30),(.30,.55)]):
        chosen=size_class==cls;diameter[chosen]=rng.uniform(lo,hi,int(chosen.sum()))
    proto_index=rng.integers(0,prototype_count,count,dtype=np.int32)
    aspect=np.column_stack([np.ones(count),rng.uniform(.55,1.,count),rng.uniform(.30,.65,count)])
    scales=np.empty((count,3));bottom_radius=np.empty(count);top_radius=np.empty(count);real_diameter=np.empty(count)
    for i,proto in enumerate(prototypes):
        rows=np.flatnonzero(proto_index==i)
        if not len(rows):continue
        lengths=np.sqrt(np.sum((proto['pairs'][None,:,:]*aspect[rows,None,:])**2,axis=2)).max(1)
        scales[rows]=aspect[rows]*(diameter[rows]/lengths)[:,None]
        bottom_radius[rows]=-float(proto['vertices'][:,2].min())*scales[rows,2]
        top_radius[rows]=float(proto['vertices'][:,2].max())*scales[rows,2]
        real_diameter[rows]=lengths*diameter[rows]/lengths
    embedding=rng.uniform(.60,.80,count)
    route_state=_limit_route_fragments(scene_config,prototypes,proto_index,scales,bottom_radius,top_radius,
                                       real_diameter,embedding,placement)
    offset=bottom_radius*embedding
    positions=placement['support']+placement['normals']*offset[:,None]
    quats=_quaternions(placement['normals'],rng.uniform(-np.pi,np.pi,count))
    UsdGeom.Xform.Define(stage,'/World/SurfaceDetail')
    instancer=UsdGeom.PointInstancer.Define(stage,INSTANCER_PATH)
    mat=_material(stage);paths=[];prototype_faces=[]
    for i,proto in enumerate(prototypes):
        path=f'{INSTANCER_PATH}/Prototypes/Fragment_{i:02d}';paths.append(Sdf.Path(path))
        mesh=UsdGeom.Mesh.Define(stage,path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(proto['vertices']))
        mesh.CreateFaceVertexCountsAttr([3]*len(proto['faces']))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(proto['faces'].ravel()))
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(False)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(proto['normals'],3,axis=0)))
        mesh.SetNormalsInterpolation('faceVarying')
        gray=float(rng.uniform(.095,.16));mesh.CreateDisplayColorAttr([Gf.Vec3f(gray*.98,gray,gray*1.02)])
        mesh.GetDisplayColorPrimvar().SetInterpolation('constant')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        mesh.GetPrim().CreateAttribute('cave:scope',Sdf.ValueTypeNames.String,custom=True).Set('render_detail_no_collision')
        prototype_faces.append(len(proto['faces']))
    instancer.CreatePrototypesRel().SetTargets(paths)
    instancer.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(proto_index))
    instancer.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
    instancer.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scales.astype(np.float32)))
    instancer.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(float(q[0]),Gf.Vec3h(*map(float,q[1:]))) for q in quats]))
    instancer.CreateIdsAttr(Vt.Int64Array.FromNumpy(np.arange(count,dtype=np.int64)))
    pad=float(diameter.max());lo=positions.min(0)-pad;hi=positions.max(0)+pad
    instancer.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float,lo)),Gf.Vec3f(*map(float,hi))]))
    instancer.GetPrim().CreateAttribute('cave:scope',Sdf.ValueTypeNames.String,custom=True).Set('render_detail_no_collision')
    instancer.GetPrim().CreateAttribute('cave:seed',Sdf.ValueTypeNames.Int,custom=True).Set(settings['seed'])
    authored_positions=np.asarray(instancer.GetPositionsAttr().Get(),float)
    authored_scales=np.asarray(instancer.GetScalesAttr().Get(),float)
    for i,proto in enumerate(prototypes):
        rows=np.flatnonzero(proto_index==i)
        if len(rows):
            real_diameter[rows]=np.sqrt(np.sum((proto['pairs'][None,:,:]*authored_scales[rows,None,:])**2,axis=2)).max(1)
            bottom_radius[rows]=-float(proto['vertices'][:,2].min())*authored_scales[rows,2]
            top_radius[rows]=float(proto['vertices'][:,2].max())*authored_scales[rows,2]
    if route_state is not None:
        positions,authored_positions,authored_scales,lowest,highest,real_diameter,actual_radius=_recheck_route_serialization(
            instancer,route_state,prototypes,proto_index,embedding,placement,positions)
    actual_offset=np.einsum('ij,ij->i',authored_positions-placement['support'],placement['normals'])
    if route_state is None:lowest=actual_offset-bottom_radius;highest=actual_offset+top_radius
    plane_floating=int(np.count_nonzero(lowest>1e-4));buried=int(np.count_nonzero(highest<=0))
    if plane_floating or buried:raise ValueError('Gravel contact QA rejected floating or fully buried fragments')
    route_report=(_route_gravel_report(route_state,prototypes,proto_index,authored_positions,authored_scales,
                                      real_diameter,highest,lowest,actual_radius) if route_state is not None else None)
    if any(_sha(path)!=expected for path,expected in before.items()):raise RuntimeError('Source mesh/config changed')
    focus_region=(authored_positions[:,0]>=11)&(authored_positions[:,0]<=23)&(np.abs(authored_positions[:,1])<=3.5)
    size_report=dict(definition='actual longest scaled prototype vertex-pair distance; orientation invariant',
        ranges=[[.03,.12],[.12,.30],[.30,.55]],target_fractions=size_fractions,actual_class_counts=class_counts,
        quantiles={str(q):float(np.quantile(real_diameter,q)) for q in [0.,.1,.25,.5,.75,.9,.99,1.]})
    if route_report is not None:
        size_report.update(definition='actual longest vertex-pair distance after the composed USD instance matrix, including half-quaternion serialization',
            ranges=[[0.,.12],[.12,.30],[.30,None]],target_fractions=None,
            pre_route_sampling_ranges=[[.03,.12],[.12,.30],[.30,.55]],
            pre_route_sampling_target_fractions=size_fractions,
            actual_class_counts=[int(np.count_nonzero(real_diameter<.12)),
                int(np.count_nonzero((real_diameter>=.12)&(real_diameter<.30))),
                int(np.count_nonzero(real_diameter>=.30))],
            sub_3cm_route_fragments=int(np.count_nonzero(real_diameter<.03)),
            size_policy='route overlap diameter/protrusion caps supersede the pre-route size mixture')
    report=dict(passed=True,instance_count=count,prototype_count=prototype_count,instancer_path=INSTANCER_PATH,
        material_path=MATERIAL_PATH,seed=settings['seed'],prototype_triangle_counts=prototype_faces,
        estimated_instanced_triangles=int(np.asarray(prototype_faces)[proto_index].sum()),
        size_diameter_m=size_report,
        support=dict(source='highest barycentric vertical hit on actual base mesh; 1m buckets and3x3 neighborhood',
            eligible_triangles=len(projector.triangles),face_normal_z_min=float(placement['normals'][:,2].min()),
            highest_hit_z_m=[float(placement['support'][:,2].min()),float(placement['support'][:,2].max())],
            center_offset_along_normal_m=[float(actual_offset.min()),float(actual_offset.max())],
            embedding_fraction_range=[float(embedding.min()),float(embedding.max())],
            lowest_vertex_support_plane_offset_m=[float(lowest.min()),float(lowest.max())],
            highest_vertex_support_plane_offset_m=[float(highest.min()),float(highest.max())],
            floating_by_support_plane=plane_floating,fully_buried_by_support_plane=buried,
            maximum_authored_position_rounding_m=float(np.linalg.norm(authored_positions-positions,axis=1).max()),
            minimum_barycentric_weight=float(placement['barycentric'].min())),
        placement=dict(group_counts={g:int(np.count_nonzero(placement['groups']==g)) for g in (['pit','cavern','focus','branch'] if scene_config.get('branch_v10') else ['pit','cavern','focus'])},
            attempted_candidates=placement['attempted'],no_mesh_support_rejected=placement['no_support'],
            field_rejected=placement['field_rejected'],field_probe_offset_m=.4,
            field_probe_minimum_air_margin=float(placement['field_values'].min()),
            closeup_region_xy_m=[[11.,-3.5],[23.,3.5]],closeup_instances=int(focus_region.sum()),
            closeup_density_per_m2=float(focus_region.sum()/84.),regular_grid=False),
        source_mesh_sha256=before[str(mesh_file)],source_config_sha256=before[str(cfg_file)],source_files_unchanged=True,
        no_collision=True,collision_scope='render_detail_no_collision; original coarse terrain collision is owned by caller',
        GPU_work=False,simulation_started=False,
        limitations=['No particle dynamics, global overlap/packing or friction stability solution',
            'One supporting triangle plane per fragment; neighbor-face curvature and full footprint penetration are approximate',
            'Analytic air field checked above MC support; centimetre placement follows the actual mesh, not analytic floor',
            'Locations, size mixture and shapes are authored appearance assumptions, not measured lunar gravel statistics'])
    if route_report is not None:report['route_gravel']=route_report
    return report
