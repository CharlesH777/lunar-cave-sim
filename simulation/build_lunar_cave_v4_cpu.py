"""Pure-CPU composite lunar entrance/cavern authoring; never starts Kit or a GPU.

The outer terrain is intentionally an open rectangular patch. Its one boundary
loop must lie on the four XY sampling limits; underground cracks are rejected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes, euler_number
import trimesh

from cave.lunar_morphology_v4 import candidate_config, evaluate_field, floor_height_at
from cave.lunar_pit_v4 import pit_config, evaluate_pit_field, evaluate_pit_grid, surface_height, pit_floor_height
from cave.network3d import _collapse_tiny_mesh_edges, _triangle_areas


ROOT = Path(__file__).resolve().parent


def scene_config() -> dict:
    return dict(
        schema_version='lunar_cave_v4', candidate_id='full_scene_01', units='m', up_axis='Z',
        cavern=candidate_config(), pit=pit_config(),
        appearance_revision=2,floor_texture_tile_m=.65,floor_normal_strength=.12,
        scientific_scope='generic Marius-scale entry + Zhou-inspired cavity, not the same observed site',
        bounds_m=[[-80.,-70.,-10.],[105.,70.,78.]],
        cavern_evaluation_bounds_m=[[-10.,-25.,-10.],[95.,25.,25.]],
        random_seed=20261110,
        authored_assumptions=['entrance-to-cavern connection and all relative coordinates',
            'longitudinal extent, wall and floor relief, ledge locations and boulder positions',
            'regional surface datum and all material/illumination appearance proxies'],
        runtime_scope='static geometry and display; no traversability, stability or policy certification',
    )


def evaluate_scene_field(config: dict, pointsNx3):
    """Positive-air union; array and single-point queries use the same code."""
    points=np.asarray(pointsNx3,dtype=float)
    single=points.shape==(3,)
    if single:points=points.reshape(1,3)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('points must be a finite 3-vector or Nx3 array')
    result=np.asarray(evaluate_pit_field(config['pit'],points),dtype=float)
    low,high=np.asarray(config.get('cavern_evaluation_bounds_m',[[-10.,-25.,-10.],[95.,25.,25.]]),float)
    mask=np.all((points>=low)&(points<=high),axis=1)
    if np.any(mask):
        result[mask]=np.maximum(result[mask],evaluate_field(config['cavern'],points[mask]))
    if config.get('branch_v10'):
        from cave.lunar_branch_v10 import evaluate_branch_field
        branch_air=evaluate_branch_field(config,points)
        if config.get('junction_v11'):
            from cave.lunar_junction_v11 import compose_junction_field
            result=compose_junction_field(config,points,result,branch_air)
        else:
            result=np.maximum(result,branch_air)
    return float(result[0]) if single else result


def _boundary_report(vertices,edges,counts,low,high):
    boundary=edges[counts==1]
    if not len(boundary):
        return dict(edge_count=0,vertex_count=0,loop_count=0,all_degrees_two=False,
                    all_on_outer_xy_limits=False,passed=False)
    ids,remap=np.unique(boundary.ravel(),return_inverse=True)
    compact=remap.reshape(-1,2)
    degree=np.bincount(compact.ravel(),minlength=len(ids))
    graph=coo_matrix((np.ones(len(compact)*2,dtype=np.uint8),
        (np.r_[compact[:,0],compact[:,1]],np.r_[compact[:,1],compact[:,0]])),shape=(len(ids),len(ids)))
    loops=int(connected_components(graph.tocsr(),directed=False,return_labels=False))
    p=vertices[ids]
    on_plane=np.any(np.column_stack([np.abs(p[:,0]-low[0]),np.abs(p[:,0]-high[0]),
                                    np.abs(p[:,1]-low[1]),np.abs(p[:,1]-high[1])])<=1e-3,axis=1)
    return dict(edge_count=int(len(boundary)),vertex_count=int(len(ids)),loop_count=loops,
        all_degrees_two=bool(np.all(degree==2)),all_on_outer_xy_limits=bool(np.all(on_plane)),
        off_outer_xy_vertex_count=int(np.count_nonzero(~on_plane)),
        scope='intentional terrain perimeter only; no underground boundary openings allowed',
        passed=bool(loops==1 and np.all(degree==2) and np.all(on_plane)))


def _orientation_report(config,vertices,faces,spacing):
    areas=_triangle_areas(vertices,faces)
    eligible=np.flatnonzero(areas>max(1e-7*spacing*spacing,1e-10))
    if not len(eligible):raise ValueError('No usable faces for normal-orientation checks')
    rng=np.random.default_rng(8817)
    select=rng.choice(eligible,min(2200,len(eligible)),replace=False)
    tri=vertices[faces[select]].astype(float)
    center=tri.mean(1);normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    epsilon=max(.12,.7*spacing)
    plus=evaluate_scene_field(config,center+epsilon*normal)
    minus=evaluate_scene_field(config,center-epsilon*normal)
    fraction=float(np.mean(plus>minus))
    # A consistent full reversal can be corrected; mixed bad orientations fail.
    flipped=False
    if fraction<.02:
        faces=faces[:,[0,2,1]].copy();plus,minus=minus,plus;normal=-normal
        fraction=float(np.mean(plus>minus));flipped=True
    far=(np.hypot(center[:,0]-config['pit']['center_xy_m'][0],center[:,1]-config['pit']['center_xy_m'][1])>45)&(center[:,2]>40)
    top_up=float(np.mean(normal[far,2]>0)) if np.any(far) else 0.
    return faces,dict(samples=len(select),probe_offset_m=epsilon,
        normal_points_toward_larger_air_field_fraction=fraction,
        plus_air_minus_rock_fraction=float(np.mean((plus>0)&(minus<0))),
        outside_terrain_upward_normal_fraction=top_up,faces_globally_flipped=flipped,
        planar_positive_z_air_toy='marching_cubes gradient_direction=descent gives +Z faces',
        passed=bool(fraction>=.98 and top_up>=.99))


def build_scene_geometry(config: dict,spacing_m=.45,status_callback=None):
    spacing=float(spacing_m)
    if not np.isfinite(spacing) or not .15<=spacing<=1.:
        raise ValueError('spacing must be finite and within [.15,1]m')
    requested_low,requested_high=np.asarray(config['bounds_m'],float)
    if requested_low.shape!=(3,) or requested_high.shape!=(3,) or np.any(requested_high<=requested_low):
        raise ValueError('bounds_m must be finite increasing xyz bounds')
    shape=np.ceil((requested_high-requested_low)/spacing).astype(int)+1
    if int(np.prod(shape))>55_000_000:raise ValueError('Requested volume exceeds 55 million voxels')
    axes=[requested_low[i]+np.arange(shape[i],dtype=float)*spacing for i in range(3)]
    low=requested_low;high=np.array([axis[-1] for axis in axes])
    volume=np.empty(tuple(shape),np.float32)
    slab_size=6
    for begin in range(0,int(shape[0]),slab_size):
        end=min(begin+slab_size,int(shape[0]))
        xx,yy,zz=np.meshgrid(axes[0][begin:end],axes[1],axes[2],indexing='ij')
        p=np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])
        if config['pit'].get('wall_roughness_model')=='aperiodic_fracture_v2':
            # XY surface/floor noise is independent of Z: compute it once per
            # column, while preserving the same point-query field exactly.
            values=evaluate_pit_grid(config['pit'],axes[0][begin:end,None,None],
                                    axes[1][None,:,None],axes[2][None,None,:]).ravel()
            body_low,body_high=np.asarray(config['cavern_evaluation_bounds_m'],float)
            mask=np.all((p>=body_low)&(p<=body_high),axis=1)
            if np.any(mask):values[mask]=np.maximum(values[mask],evaluate_field(config['cavern'],p[mask]))
            if config.get('branch_v10'):
                from cave.lunar_branch_v10 import evaluate_branch_field
                branch_air=evaluate_branch_field(config,p)
                if config.get('junction_v11'):
                    from cave.lunar_junction_v11 import compose_junction_field
                    values=compose_junction_field(config,p,values,branch_air)
                else:
                    values=np.maximum(values,branch_air)
            volume[begin:end]=values.reshape(end-begin,shape[1],shape[2])
        else:
            volume[begin:end]=evaluate_scene_field(config,p).reshape(end-begin,shape[1],shape[2])
        if status_callback and (begin==0 or begin%60==0 or end==shape[0]):
            status_callback('sampling_cpu_field',x_planes_done=int(end),x_planes_total=int(shape[0]))
    air=volume>0
    labels,n_components=ndimage.label(air,structure=ndimage.generate_binary_structure(3,1))
    air_euler=int(euler_number(air,connectivity=3))
    if status_callback:status_callback('extracting_cpu_mesh',grid_voxels=int(np.prod(shape)))
    vertices,faces,_,_=marching_cubes(volume,0.,spacing=(spacing,)*3,
                                     gradient_direction='descent',allow_degenerate=False)
    vertices=(vertices+low).astype(np.float32);faces=faces.astype(np.int32)
    vertices,faces,cleanup=_collapse_tiny_mesh_edges(vertices,faces,spacing)
    faces,orientation=_orientation_report(config,vertices,faces,spacing)
    edges,counts=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    boundary=_boundary_report(vertices,edges,counts,low,high)
    areas=_triangle_areas(vertices,faces)
    tm=trimesh.Trimesh(vertices,faces,process=False)
    normals=np.asarray(tm.vertex_normals,dtype=np.float32)
    finite=bool(np.isfinite(vertices).all() and np.isfinite(normals).all() and np.isfinite(areas).all())
    degenerates=int(np.count_nonzero(areas<=1e-10*spacing*spacing))
    mesh_euler=int(len(vertices)-len(edges)+len(faces))
    witness=np.array([[-22.,0.,70.],[-22.,0.,8.],[1.,0.,5.],
                      [12.,-3.5,3.],[26.,-3.,3.6],[35.,-4.,3.8],[70.,0.,5.]])
    witness_values=evaluate_scene_field(config,witness)
    indices=np.rint((witness-low)/spacing).astype(int)
    indices=np.clip(indices,0,shape-1)
    witness_labels=labels[tuple(indices.T)]
    connected=bool(np.all(witness_values>0) and len(set(witness_labels.tolist()))==1 and witness_labels[0]!=0)
    cover=np.array([[45.,0.,30.],[60.,0.,40.],[30.,5.,40.]])
    cover_values=evaluate_scene_field(config,cover)
    cover_ok=bool(np.all(cover_values<0))
    all_inside=bool(np.all(vertices>=low-1e-3)&np.all(vertices<=high+1e-3))
    passed=bool(finite and degenerates==0 and not np.any(counts>2) and boundary['passed']
        and tm.is_winding_consistent and tm.body_count==1 and mesh_euler==1
        and n_components==1 and air_euler==1 and connected and cover_ok and orientation['passed'] and all_inside)
    report=dict(passed=passed,scientific_scope=config['scientific_scope'],voxel_spacing_m=spacing,
        grid_shape=shape.tolist(),grid_voxels=int(np.prod(shape)),grid_bounds_m=[low.tolist(),high.tolist()],
        mesh=dict(vertices=len(vertices),faces=len(faces),finite=finite,degenerate_faces=degenerates,
            nonmanifold_edges=int(np.count_nonzero(counts>2)),boundary=boundary,
            euler_characteristic=mesh_euler,expected_euler_characteristic=1,
            body_count=int(tm.body_count),winding_consistent=bool(tm.is_winding_consistent),
            watertight=bool(tm.is_watertight),intentional_open_terrain_patch=True,
            bounds_m=[vertices.min(0).tolist(),vertices.max(0).tolist()],subvoxel_cleanup=cleanup),
        normal_orientation=orientation,
        air=dict(connected_components_6_neighbor=int(n_components),euler_26_neighbor=air_euler,
            exterior_and_cavern_connected=connected,witness_xyz_m=witness.tolist(),
            witness_field_values=witness_values.tolist(),witness_component_ids=witness_labels.tolist()),
        underground_cover=dict(passed=cover_ok,rock_probe_xyz_m=cover.tolist(),field_values=cover_values.tolist()),
        method='union of independent natural-cavity and lunar-pit hypotheses; streamed 3D field and marching cubes',
        limitations=['Air field is an implicit margin, not an exact Euclidean SDF',
            'Boulder display meshes are not subtracted from the field-connectivity volume',
            'No global mesh self-intersection proof, PhysX cooking, stability or traversability certification',
            'An artificial outer terrain perimeter is explicitly allowed and validated'],
        GPU_work=False,simulation_started=False,training_started=False,navigation_evaluated=False)
    return dict(vertices=vertices,faces=faces,normals=normals,report=report)


def boulder_specs(config: dict):
    """Authored deterministic off-axis clusters, with half-size/radius scales."""
    rng=np.random.default_rng(config.get('random_seed',20261110));result=[]
    def add(x,y,zbase,size,kind):
        scale=np.array([size*rng.uniform(.7,1.3),size*rng.uniform(.6,1.15),size*rng.uniform(.4,.8)])
        center=[float(x),float(y),float(zbase+scale[2]*rng.uniform(.4,.6))]
        result.append(dict(center_xyz_m=center,scale_xyz_m=scale.tolist(),seed=int(rng.integers(1,2**30)),
                           group=kind,provenance='authored deterministic position and shape; not measured lunar rock'))
    cx,cy=config['pit']['center_xy_m'];pit_centers=[(cx-15,cy+6),(cx+12,cy+14),(cx+2,cy-17)]
    for _ in range(32):
        for _attempt in range(20):
            ax,ay=pit_centers[int(rng.integers(len(pit_centers)))];x,y=rng.normal([ax,ay],[3.4,2.8])
            z=float(pit_floor_height(config['pit'],x,y));size=float(rng.uniform(.3,1.5))
            if evaluate_pit_field(config['pit'],[x,y,z+size])>.2:
                add(x,y,z,size,'pit_floor_talus');break
    a,b=config['pit']['opening_radii_m']
    for _ in range(30):
        theta=float(rng.choice([-2.4,-.6,.6,2.0])+rng.normal(0,.15));radius=float(rng.uniform(1.22,1.60))
        x,y=cx+a*radius*np.cos(theta),cy+b*radius*np.sin(theta)
        add(x,y,float(surface_height(config['pit'],x,y)),float(rng.uniform(.2,.8)),'surface_rim')
    for _ in range(18):
        for _attempt in range(30):
            ax,ay=[(22.,-6.7),(51.,6.5),(66.,-5.5)][int(rng.integers(3))]
            x,y=rng.normal([ax,ay],[3.0,.8]);z=float(floor_height_at(config['cavern'],x,y));size=float(rng.uniform(.2,.8))
            if evaluate_scene_field(config,[x,y,z+size])>.15:
                add(x,y,z,size,'cavern_side_talus');break
    if config.get('additional_boulders_v10'):
        from cave.lunar_branch_v10 import branch_floor_height
        for item in config['additional_boulders_v10']:
            x,y=map(float,item['xy_m']);scale=np.asarray(item['scale_xyz_m'],float)
            zbase=float(branch_floor_height(config,x,y) if item.get('region')=='branch' else floor_height_at(config['cavern'],x,y))
            if evaluate_scene_field(config,[x,y,zbase+float(scale[2])])<=.05:
                raise ValueError('Authored large obstacle is outside the cave air: '+str(item))
            result.append(dict(center_xyz_m=[x,y,zbase+float(scale[2])*.5],scale_xyz_m=scale.tolist(),
                seed=int(item['seed']),group=item['group'],provenance='v10 authored collapse boulder obstacle; no path-dependent removal'))
    if config.get('rover_route'):
        from cave.rover_route_v6 import route_distance_xy
        route=config['rover_route'];kept=[]
        for item in result:
            point=np.asarray(item['center_xyz_m'],float)
            radius=1.6*float(np.linalg.norm(np.asarray(item['scale_xyz_m'],float)[:2]))
            distance=float(route_distance_xy(route,point[:2]))
            # Rocks on the surface 60m above the rover route are unrelated to its footprint.
            if point[2]<22. and distance<float(route['drivable_half_width_m'])+radius+.2:
                continue
            item['route_xy_radius_bound_m']=radius
            item['route_distance_m']=distance
            item['route_placement_scope']='large blocks kept outside the drive band; high surface-rim blocks exempt'
            kept.append(item)
        result=kept
    return result


def _json_default(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,Path):return str(value)
    raise TypeError('Not JSON serializable: '+type(value).__name__)


def _write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=_json_default)+'\n')
    temporary.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/lunar_morphology_v4/full_scene_01')
    parser.add_argument('--spacing',type=float,default=.45,help='Metres; .35 is supported at about53.8M voxels')
    parser.add_argument('--config',type=Path)
    args=parser.parse_args(argv);output=args.output.absolute();output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def status(phase,**kw):
        record=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.monotonic()-start,3),GPU_work=False,simulation_started=False,training_started=False,**kw)
        _write(output/'status.json',record);print(json.dumps(record),flush=True)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1'):
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for CPU authoring")
        config=json.loads(args.config.read_text()) if args.config else scene_config()
        _write(output/'scene_config.json',config)
        geometry=build_scene_geometry(config,args.spacing,status)
        np.savez_compressed(output/'mesh.npz',vertices=geometry['vertices'],faces=geometry['faces'],normals=geometry['normals'])
        report=geometry['report'];_write(output/'geometry_report.json',report)
        if not report['passed']:
            status('rejected',usd_authored=False,report='geometry_report.json');return 2
        status('authoring_cpu_usd',vertices=len(geometry['vertices']),faces=len(geometry['faces']))
        from cave.lunar_visual_v4 import write_scene
        boulders=boulder_specs(config);_write(output/'boulders.json',boulders)
        usd_report=write_scene(output/'cave.usdc',geometry['vertices'],geometry['faces'],config,
                               normals=geometry['normals'],boulders=boulders)
        report['usd_export']=usd_report;report['boulder_count']=len(boulders)
        report['boulder_provenance']='deterministic authored display geometry, not measured rock positions'
        _write(output/'geometry_report.json',report)
        if not isinstance(usd_report,dict) or usd_report.get('passed') is not True:
            status('rejected',usd_authored=True,usd_checks_passed=False,report='geometry_report.json');return 2
        status('complete',passed=True,usd='cave.usdc',report='geometry_report.json');return 0
    except BaseException as exc:
        _write(output/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc));raise


if __name__=='__main__':raise SystemExit(main())
