"""A deliberately irregular lunar-cave *hypothesis*, in local metre coordinates.

The volume is a rounded unequal-shoulder cavity, not an extruded half ellipse.
Zhou et al. 2025 Figures 4(f)/A6 motivate its asymmetric roof/floor shape only.
Longitudinal dimensions, noise, protrusions and deposits are authored assumptions.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes, euler_number


def candidate_config() -> dict:
    return {
        'schema_version': 'lunar_morphology_hypothesis/4.0',
        'candidate_id': 'representative_01', 'units': 'm', 'up_axis': 'Z',
        'length_m': 80., 'rounded_end_extension_m': 5.,
        'nominal_left_width_m': 10., 'nominal_right_width_m': 12.6,
        'nominal_height_m': 14.5, 'cross_section_exponent': 3.05,
        'roof_depression_m': 4.7, 'wall_geometry_roughness_m': .68,
        'geometric_noise_model': 'aperiodic_fracture_v2',
        'geometric_noise_scales_m': [6.,2.,.7],
        'geometric_noise_amplitudes_m': [.8,.35,.12],
        'legacy_wall_roughness_gain_applies_only_to': 'configs without geometric_noise_model',
        'random_seed': 20261110,
        'recommended_scale_rover_xy_m': [18., -3.],
        'coordinates': 'local floor near z=0; no surface or burial datum is assigned',
        'shape_reference': {
            'doi': '10.3847/1538-4357/adcf18',
            'locator': 'Figure 4(f), final formation hypothesis; Figure A6(b), EM forward-model geometry',
            'use': 'Rounded asymmetric void, localized roof depression, one-sided residual basalt, bottom accumulation',
            'reported_estimates_m_approx': {'maximum_width': 26, 'maximum_height': 16},
            'epistemic_status': 'shape inspired by an interpretation, not digitized or measured 3D geometry',
        },
        'authored_assumptions': [
            '80m longitudinal extent, direction and curved axis',
            'all longitudinal width/height and floor changes',
            'all roughness wavelengths, amplitudes and locations',
            'the size and 3D position of the wall-attached residual shoulder',
            'floor relief, angular debris mounds and their distribution',
            'finite-domain end caps, not observed geological ends',
        ],
        'floor_deposits': [
            {'x': 22., 'u': -6.2, 'rx': 4.4, 'ry': 2.5, 'height': 1.7, 'angle': .3},
            {'x': 31., 'u': 4.2, 'rx': 5.3, 'ry': 3.0, 'height': 2.1, 'angle': -.45},
            {'x': 42., 'u': -.8, 'rx': 7.2, 'ry': 2.7, 'height': 1.9, 'angle': .15},
            {'x': 53., 'u': -5.5, 'rx': 4.8, 'ry': 2.8, 'height': 1.8, 'angle': -.2},
            {'x': 65., 'u': 5.0, 'rx': 4.0, 'ry': 2.7, 'height': 2.0, 'angle': .65},
        ],
        'scientific_status': 'synthetic representative underground cavity, not a lunar reconstruction',
        'no_observed_entrance': True,
        'runtime_scope': 'geometry only; no navigation, structural or vehicle dynamics certification',
    }


def _axis_y(x):
    return 1.30*np.sin(.071*x) + .65*np.sin(.173*x+.55)


def _datum(x):
    return .62*np.sin(.077*x) - .28*np.sin(.21*x+.2)


def _uses_aperiodic(config):
    mode=config.get('geometric_noise_model','legacy_plane_wave_v1')
    if mode not in ('legacy_plane_wave_v1','aperiodic_fracture_v2'):
        raise ValueError('Unknown geometric_noise_model: '+str(mode))
    return mode=='aperiodic_fracture_v2'


def _datum_for(config,x):
    if _uses_aperiodic(config):
        from .geologic_noise_v4 import rock_noise
        return .9*rock_noise(.32*x+51.,0.*x-17.,0.*x+8.,seed=int(config['random_seed'])+1009)
    return _datum(x)


def _noise(x, y, z):
    # Legacy only: saved configs without the v2 flag replay this exact formula.
    broad = (np.sin(.39*x+.53*y+.31*z)
             + .72*np.sin(-.27*x+.44*y-.59*z+1.7)
             + .51*np.sin(.71*x-.23*y+.46*z+3.2))/2.23
    ridges = (np.abs(np.sin(1.17*x+.69*y-.41*z+.8))-.637)
    detail = np.sin(1.93*x-1.21*y+.83*z+2.1)*np.sin(.47*x+.95*y+1.39*z)
    return .78*broad + .34*ridges + .15*detail


def _floor(config, x, y):
    if config.get('ground_profile_v7') is not None:
        if config.get('rover_route_floor') is not None:
            raise ValueError('global ground_profile_v7 cannot be combined with a prescribed rover route floor')
        from .lunar_ground_v7 import cavern_floor
        return cavern_floor(config['ground_profile_v7'],config,x,y)
    u = y-_axis_y(x)
    if _uses_aperiodic(config):
        from .geologic_noise_v4 import rock_noise
        floor=(_datum_for(config,x)
               +.36*rock_noise(x+137.,y-29.,0.*(x+y)+5.7,seed=int(config['random_seed'])+70313)
               +.006*u*u)
    else:
        floor = (_datum(x) + .26*np.sin(.29*x+.39*u)
                 + .20*np.sin(.63*u-.11*x+1.1)
                 + .13*np.sin(1.39*x+.88*u)*np.sin(.43*x-.67*u)
                 + .006*u*u)
    for pile in config['floor_deposits']:
        dx,dy=x-pile['x'],u-pile['u']
        c,s=math.cos(pile['angle']),math.sin(pile['angle'])
        a=(c*dx+s*dy)/pile['rx'];b=(-s*dx+c*dy)/pile['ry']
        # Hexagonal/pyramidal talus ridges, rather than smooth decorative bumps.
        r=np.maximum(np.abs(a), np.maximum(np.abs(.5*a+.8660254*b), np.abs(-.5*a+.8660254*b)))
        floor += pile['height']*np.maximum(1.-r,0.)**.88
    if config.get('rover_route_floor') is not None:
        from .rover_route_v6 import apply_floor
        floor=apply_floor(config['rover_route_floor'],x,y,floor)
    return floor


def floor_height_at(config: dict, x, y):
    """Authored continuous deposit surface for this one-level representative body."""
    x,y=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    return _floor(config,x,y)


def _field(config, x, y, z):
    u=y-_axis_y(x);datum=_datum_for(config,x)
    left=config['nominal_left_width_m']*(1.+.14*np.sin(.087*x+.7)+.055*np.sin(.231*x+1.4))
    right=config['nominal_right_width_m']*(1.+.13*np.cos(.068*x+.3)+.045*np.sin(.207*x+2.1))
    height=config['nominal_height_m']*(1.+.12*np.sin(.069*x+.6)+.055*np.cos(.163*x+.4))
    vertical=z-datum
    top_weight=.5+.5*np.tanh((vertical-.48*height)/1.6)
    depression=config['roof_depression_m']*np.exp(-((x-42.)/16.)**2-((u+1.7)/4.1)**2)
    # Warp the upper closed profile, leaving a rounded irregular bottom contour.
    lower_shift=0.
    floor_v7=None
    if config.get('ground_profile_v7') is not None:
        if config.get('rover_route_floor') is not None:
            raise ValueError('global ground_profile_v7 cannot be combined with a prescribed rover route floor')
        from .lunar_ground_v7 import lower_envelope_shift
        floor_v7=_floor(config,x,y)
        lower_shift=lower_envelope_shift(config['ground_profile_v7'],x,y,z,datum,height,floor_v7)
    warped=vertical+lower_shift+top_weight*depression
    width=np.where(u<0,left,right)
    p=config['cross_section_exponent']
    longitudinal=np.maximum(np.maximum(-x,x-config['length_m']),0.)/config['rounded_end_extension_m']
    radius=(np.abs(u/width)**p+np.abs((warped-.5*height)/(.5*height))**p+longitudinal**2)**(1./p)
    body=(1.-radius)*np.minimum(left,height*.5)
    if _uses_aperiodic(config):
        from .geologic_noise_v4 import rock_noise
        body += rock_noise(x,y,z,seed=int(config['random_seed']))
    else:
        body += config['wall_geometry_roughness_m']*_noise(x,y,z)
    if config.get('rover_route_floor') is not None:
        from .rover_route_v6 import corridor_air_field
        body=np.maximum(body,corridor_air_field(config['rover_route_floor'],x,y,z))
    ground=z-(_floor(config,x,y) if floor_v7 is None else floor_v7)
    # A wall-attached residual mass. A continuous lateral bound avoids an
    # ellipsoid/wall tangency creating artificial pinched air pockets below it.
    intrusion=6.3*np.exp(-((x-46.)/14.)**2-((vertical-6.5)/7.)**2)
    if _uses_aperiodic(config):
        shoulder=right-intrusion-u+.22*rock_noise(x+.7,y-2.,z+1.,seed=int(config['random_seed'])+31)
    else:
        shoulder=right-intrusion-u + .11*_noise(x+.7,y-2.,z+1.)
    return np.minimum(np.minimum(body,ground),shoulder)


def evaluate_field(config: dict, pointsNx3) -> np.ndarray:
    """Return positive in air. This implicit margin is not an exact Euclidean SDF."""
    points=np.asarray(pointsNx3,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('points must be a finite Nx3 array')
    return _field(config,points[:,0],points[:,1],points[:,2])


def recommended_views(config):
    specs=[('wide_body',[12.,-3.5,3.0],[36.,.2,7.4]),
           ('depressed_roof',[26.,-3.,3.6],[43.,-1.0,8.2]),
           ('residual_shoulder',[35.,-4.0,3.8],[52.,3.0,6.0])]
    result=[]
    for name,eye,target in specs:
        eye,target=np.asarray(eye),np.asarray(target)
        values=evaluate_field(config,eye+np.linspace(0.,1.,201)[:,None]*(target-eye))
        result.append(dict(name=name,eye=eye.tolist(),target=target.tolist(),
                           full_sampled_view_ray_inside_air=bool(np.all(values>0)),
                           minimum_implicit_margin_m_not_clearance=float(values.min())))
    return result


def build_geometry(config: dict, spacing_m: float=.28) -> dict:
    """Pure CPU closed-air meshing; source geometry and other projects are untouched."""
    from .network3d import _collapse_tiny_mesh_edges, _triangle_areas
    import trimesh
    spacing=float(spacing_m)
    if not np.isfinite(spacing) or not .10<=spacing<=1.:
        raise ValueError('spacing must be finite and between .10 and 1m')
    low=np.array([-7.,-22.,-4.]);high=np.array([87.,22.,22.])
    axes=[np.arange(low[i],high[i]+spacing*.5,spacing,dtype=np.float64) for i in range(3)]
    shape=tuple(map(len,axes));volume=np.empty(shape,np.float32)
    if np.prod(shape)>40_000_000:raise ValueError('Grid exceeds 40 million voxels')
    for start in range(0,shape[0],12):
        sl=slice(start,min(start+12,shape[0]))
        volume[sl]=_field(config,axes[0][sl,None,None],axes[1][None,:,None],axes[2][None,None,:])
    air=volume>0
    labels,components=ndimage.label(air,structure=ndimage.generate_binary_structure(3,1))
    air_euler=int(euler_number(air,connectivity=3))
    grid_boundary=bool(air[[0,-1],:,:].any() or air[:,[0,-1],:].any() or air[:,:,[0,-1]].any())
    vertices,faces,_,_=marching_cubes(volume,0.,spacing=(spacing,)*3,gradient_direction='descent',allow_degenerate=False)
    vertices=(vertices+low).astype(np.float32);faces=faces.astype(np.int32)
    vertices,faces,cleanup=_collapse_tiny_mesh_edges(vertices,faces,spacing)
    tm=trimesh.Trimesh(vertices,faces,process=False)
    edge_counts=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0,return_counts=True)[1]
    areas=_triangle_areas(vertices,faces)
    finite=bool(np.isfinite(vertices).all() and np.isfinite(areas).all())
    degenerates=int(np.count_nonzero(areas<=1e-10*spacing*spacing))
    mesh_euler=int(len(vertices)-len(edge_counts)+len(faces))
    views=recommended_views(config)
    path_x=np.linspace(1.,79.,313);path_y=_axis_y(path_x)-2.
    path_z=floor_height_at(config,path_x,path_y)+1.1
    path_values=evaluate_field(config,np.column_stack([path_x,path_y,path_z]))
    floor_grid_x,floor_grid_u=np.meshgrid(np.linspace(5,75,141),np.linspace(-7,7,57),indexing='ij')
    floor_grid_y=floor_grid_u+_axis_y(floor_grid_x)
    floor_sample=floor_height_at(config,floor_grid_x,floor_grid_y)
    passed=bool(finite and degenerates==0 and not grid_boundary and components==1 and air_euler==1
                and mesh_euler==2 and np.all(edge_counts==2) and tm.is_watertight
                and tm.is_winding_consistent and tm.body_count==1 and len(vertices)<=400_000)
    report=dict(passed=passed, scientific_status=config['scientific_status'],
        geometric_noise_model=config.get('geometric_noise_model','legacy_plane_wave_v1'),
        method='deformed asymmetric closed superellipse + 3D wall roughness + floor deposits + wall-attached residual shoulder',
        half_ellipse_with_flat_floor=False, units='m',up_axis='Z',voxel_spacing_m=spacing,
        grid_shape=list(shape),grid_voxels=int(np.prod(shape)),
        mesh=dict(vertices=len(vertices),faces=len(faces),finite=finite,degenerate_faces=degenerates,
                  boundary_edges=int(np.count_nonzero(edge_counts==1)),nonmanifold_edges=int(np.count_nonzero(edge_counts>2)),
                  watertight=bool(tm.is_watertight),winding_consistent=bool(tm.is_winding_consistent),body_count=int(tm.body_count),
                  euler_characteristic=mesh_euler,bounds_m=[vertices.min(0).tolist(),vertices.max(0).tolist()],
                  subvoxel_cleanup=cleanup),
        air=dict(connected_components_6_neighbor=int(components),euler_26_neighbor=air_euler,
                 touches_grid_boundary=grid_boundary,voxel_count=int(np.count_nonzero(air))),
        floor_geometry=dict(sampled_z_range_m=[float(floor_sample.min()),float(floor_sample.max())],
                            peak_to_peak_m=float(np.ptp(floor_sample)),std_m=float(np.std(floor_sample))),
        diagnostic_floor_path=dict(all_inside_air=bool(np.all(path_values>0)),minimum_implicit_margin_m_not_clearance=float(path_values.min()),
                                    note='sampled reference only; no robot envelope or continuous traversal certification'),
        recommended_views=views, artificial_end_caps=True,no_observed_entrance=True,
        restrictions=['No structural stability solution','No vehicle or wheel dynamics','No measured lunar 3D reconstruction',
                      'No full global triangle self-intersection proof','No assigned lunar burial/surface datum'],
        GPU_work=False,simulation_started=False,navigation_evaluated=False)
    return dict(vertices=vertices,faces=faces,report=report)
