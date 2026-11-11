"""An authored gentle floor corridor starting on the lunar-pit bottom.

This is an engineering accommodation for the visual rover, not measured lunar
terrain or a claim about its tyre dynamics. World axes remain metres/Z-up. A
saved cave or pit configuration enables it only with ``rover_route_floor``.
"""
from __future__ import annotations

import numpy as np


def route_config() -> dict:
    """Return independent JSON parameters for a finite pit-bottom route."""
    return {
        'schema_version': 'rover_floor_corridor/6.0',
        'route_id': 'pit_bottom_to_cavern',
        'units': 'm', 'up_axis': 'Z',
        'start_scope': 'pit_bottom; the natural shaft is not a driving route',
        'drive_x_range_m': [-15., 74.],
        'core_x_range_m': [-16., 77.],
        'longitudinal_blend_m': 4.,
        'drivable_half_width_m': 1.2,
        'blend_half_width_m': 3.,
        'centerline_y_offset_m': -2.,
        'ramp_x_range_m': [-12., 28.],
        'profile_z_range_m': [-1.5, .25],
        'micro_relief_m': .002,
        'micro_wavelength_m': 4.,
        'micro_seed': 20261110 + 6029,
        'carve_height_m': 2.,
        'distance_polyline_step_m': .25,
        'max_pitch_deg': 5.,
        'max_roll_deg': 3.,
        'max_step_m': .03,
        'max_gravel_protrusion_m': .02,
        'provenance': 'engineering-authored geometry for a gentle rover corridor; not a lunar observation',
        'scope': 'geometric floor/clearance constraints only; no wheel, suspension or friction certification',
    }


def _smooth(value):
    value=np.clip(value,0.,1.)
    return value*value*(3.-2.*value)


def _parameters(cfg):
    """Reject malformed corridor widths/ranges before modifying a field."""
    if not isinstance(cfg,dict):raise ValueError('route cfg must be a dictionary')
    inner=float(cfg.get('drivable_half_width_m',1.2));outer=float(cfg.get('blend_half_width_m',3.))
    core=np.asarray(cfg.get('core_x_range_m',[-16.,77.]),dtype=float)
    ramp=np.asarray(cfg.get('ramp_x_range_m',[-12.,28.]),dtype=float)
    drive=np.asarray(cfg.get('drive_x_range_m',[-15.,74.]),dtype=float)
    heights=np.asarray(cfg.get('profile_z_range_m',[-1.5,.25]),dtype=float)
    fade=float(cfg.get('longitudinal_blend_m',4.));headroom=float(cfg.get('carve_height_m',2.))
    if not all(np.isfinite(v).all() for v in [core,ramp,drive,heights]):raise ValueError('route values must be finite')
    if any(a.shape!=(2,) or a[0]>=a[1] for a in [core,ramp,drive]):raise ValueError('route ranges must be increasing pairs')
    if heights.shape!=(2,):raise ValueError('profile_z_range_m must be a pair')
    if not (np.isfinite(inner) and np.isfinite(outer) and 0<inner<outer):raise ValueError('require positive drivable width below blend width')
    if not (core[0]<=drive[0]<drive[1]<=core[1]):raise ValueError('drive range must lie in the full-strength core')
    if not (np.isfinite(fade) and fade>0 and np.isfinite(headroom) and headroom>1.2):raise ValueError('invalid route fade/headroom')
    return inner,outer,core,ramp,drive,heights,fade,headroom


def route_y(cfg: dict, x):
    """Route center follows the existing cave's mild planar meander."""
    x=np.asarray(x,dtype=float)
    return 1.30*np.sin(.071*x)+.65*np.sin(.173*x+.55)+float(cfg.get('centerline_y_offset_m',-2.))


def route_z(cfg: dict, x):
    """Center floor elevation; a clamped C1 cubic ramp, not a staircase."""
    _,_,_,ramp,_,heights,_,_=_parameters(cfg)
    x=np.asarray(x,dtype=float)
    t=_smooth((x-ramp[0])/(ramp[1]-ramp[0]))
    return heights[0]+(heights[1]-heights[0])*t


def _weight(cfg,x,y):
    inner,outer,core,_,_,_,fade,_=_parameters(cfg)
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    dy=np.abs(y-route_y(cfg,x))
    lateral=1.-_smooth((dy-inner)/(outer-inner))
    along=_smooth((x-(core[0]-fade))/fade)*(1.-_smooth((x-core[1])/fade))
    return lateral*along


def _target_floor(cfg,x,y):
    """Millimetre cross-relief vanishes on the exact route centerline."""
    from .geologic_noise_v4 import value_noise
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    amp=float(cfg.get('micro_relief_m',.002));length=float(cfg.get('micro_wavelength_m',4.))
    if not (np.isfinite(amp) and 0<=amp<=.004 and np.isfinite(length) and length>=2.):
        raise ValueError('route micro-relief must remain <=4mm with wavelength >=2m')
    dy=y-route_y(cfg,x);seed=int(cfg.get('micro_seed',20261110+6029))
    baseline=value_noise(x/length+11.7,0.*dy-6.2,2.4,seed=seed)
    relief=value_noise(x/length+11.7,dy/length-6.2,2.4,seed=seed)-baseline
    return route_z(cfg,x)+amp*relief


def apply_floor(cfg: dict,x,y,original_floor):
    """Blend only the corridor into a smooth profile, retaining side deposits."""
    x,y,original=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float),np.asarray(original_floor,float))
    weight=_weight(cfg,x,y)
    if not np.any(weight>0):return original
    # Exact arithmetic at w=0/1 keeps untouched side material and matches the
    # independently authored pit/body floors throughout the full driving band.
    result=original.copy();selected=weight>0
    target=_target_floor(cfg,x[selected],y[selected])
    w=weight[selected]
    result[selected]=np.where(w>=1.,target,(1.-w)*original[selected]+w*target)
    return result


def corridor_air_field(cfg: dict,x,y,z):
    """A finite floor-near air carve removes an old rounded body-bottom lip.

    This field only adds room below the existing cavern. It ends 2m above the
    target floor; the normal cave field retains all of the roof and wall form.
    The final body still intersects its blended ground and residual shoulder.
    """
    _,outer,core,_,_,_,fade,headroom=_parameters(cfg)
    x,y,z=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float),np.asarray(z,float))
    floor=_target_floor(cfg,x,y)
    lateral=outer-np.abs(y-route_y(cfg,x))
    along=np.minimum(x-(core[0]-fade),(core[1]+fade)-x)
    return np.minimum(np.minimum(lateral,along),np.minimum(z-floor,floor+headroom-z))


def route_distance_xy(cfg: dict,pointsXY):
    """Exact nearest Euclidean distance to the finite sampled driving polyline.

    The default 0.25m polyline spans only drive_x_range_m. Distances past either
    endpoint include endpoint separation; they are not an infinite lateral
    offset. A single XY input returns a float, an Nx2 array returns an N array.
    """
    points=np.asarray(pointsXY,dtype=float);single=points.shape==(2,)
    if single:points=points.reshape(1,2)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('pointsXY must be finite 2-vector or Nx2')
    _,_,_,_,drive,_,_,_=_parameters(cfg)
    step=float(cfg.get('distance_polyline_step_m',.25))
    if not np.isfinite(step) or not 0<step<=.5:raise ValueError('distance_polyline_step_m must be in (0,.5]')
    xx=np.linspace(drive[0],drive[1],int(np.ceil((drive[1]-drive[0])/step))+1)
    path=np.column_stack([xx,route_y(cfg,xx)]);a=path[:-1];d=np.diff(path,axis=0)
    length2=np.einsum('ij,ij->i',d,d);answer=np.empty(len(points),float)
    for start in range(0,len(points),1024):
        p=points[start:start+1024,None,:]-a[None,:,:]
        t=np.clip(np.einsum('nsi,si->ns',p,d)/length2,0.,1.)
        p-=t[:,:,None]*d[None,:,:]
        answer[start:start+len(p)]=np.sqrt(np.einsum('nsi,nsi->ns',p,p).min(1))
    return float(answer[0]) if single else answer
