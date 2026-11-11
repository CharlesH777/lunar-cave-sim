"""Moderately uneven two-dimensional lunar-pit/cavern floor hypothesis.

The entire floor has shared low-frequency relief plus local deposits. There
is no driving centreline, flattened corridor, preselected path, or removal of
obstacles. These are authored simulation dimensions, not a measured Moon map.
"""
from __future__ import annotations

import math
import numpy as np


def ground_config() -> dict:
    """Parameters for a shared entrance floor with local, avoidable relief."""
    return {
        'schema_version': 'shared_lunar_ground/7.0',
        'model': 'world_xy_aperiodic_relief_with_local_deposits',
        'units': 'm', 'up_axis': 'Z',
        'datum_z_m': -1.4,
        'seed': 20261110 + 701,
        'background_scales_m': [14., 6., 2.2, .7],
        'background_amplitudes_m': [.34, .08, .015, .003],
        'cavern_deposit_height_scale': .28,
        'cavern_deposit_height_range_m': [.20, .65],
        'pit_talus_heights_m': [.65, .45, .30],
        'lower_envelope_reference_lift_m': 1.8,
        'lower_envelope_merge_height_fraction': .52,
        'engineering_scope': 'shared pit/cavern datum, moderate background relief and local deposits; local obstacles remain',
        'no_prescribed_route': True,
        'provenance': 'authored global terrain hypothesis; not measured lunar elevation or rover mobility certification',
    }


def _validate(config):
    if not isinstance(config,dict):raise ValueError('ground config must be a dictionary')
    datum=float(config.get('datum_z_m',-1.4));scales=np.asarray(config.get('background_scales_m',[14.,6.,2.2,.7]),float)
    amps=np.asarray(config.get('background_amplitudes_m',[.34,.08,.015,.003]),float)
    if not np.isfinite(datum) or scales.shape!=(4,) or amps.shape!=(4,):
        raise ValueError('ground datum/scales/amplitudes must be finite and have four octaves')
    if not np.isfinite(scales).all() or not np.isfinite(amps).all() or np.any(scales<=0) or np.any(amps<0):
        raise ValueError('ground scales must be positive and amplitudes nonnegative')
    if amps.sum()>.6 or np.any(scales<.5):
        raise ValueError('background ground relief exceeds the authored moderate range')
    return datum,scales,amps


def base_floor(config: dict,x,y):
    """One continuous world-XY background shared by the pit and the cavern."""
    from .geologic_noise_v4 import value_noise
    datum,scales,amps=_validate(config)
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    seed=int(config.get('seed',20261110+701))
    # A fixed rotation avoids privilege for either X/Y. It is not a path or a
    # direction of travel: every queried point uses the same 2D terrain field.
    a=.8*x+.6*y;b=-.6*x+.8*y
    floor=np.full(x.shape,datum,dtype=float)
    for i,(scale,amp) in enumerate(zip(scales,amps)):
        floor+=float(amp)*value_noise(a/scale+19.7+i*2.31,b/scale-8.3+i*.79,
                                     4.1+i*3.7,seed=seed+i*104729)
    return floor


def _axis_y(x):
    # Existing cave-axis coordinates locate geological deposits only. They do
    # not suppress relief or reserve an obstacle-free driving strip.
    return 1.30*np.sin(.071*x)+.65*np.sin(.173*x+.55)


def _smooth_compact_mound(r):
    t=np.clip(1.-r,0.,1.)
    return t*t*(3.-2.*t)


def cavern_floor(config: dict,cavern_cfg: dict,x,y):
    """Shared relief plus reduced, local cave-deposit mounds in all XY."""
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    floor=base_floor(config,x,y);u=y-_axis_y(x)
    scale=float(config.get('cavern_deposit_height_scale',.28))
    limits=np.asarray(config.get('cavern_deposit_height_range_m',[.20,.65]),float)
    if not np.isfinite(scale) or scale<0 or limits.shape!=(2,) or not np.isfinite(limits).all() or not 0<=limits[0]<=limits[1]<=.8:
        raise ValueError('invalid local cavern deposit heights')
    for pile in cavern_cfg.get('floor_deposits',[]):
        old_height=float(pile['height'])
        if old_height<=0:continue
        height=float(np.clip(old_height*scale,limits[0],limits[1]))
        dx,dy=x-float(pile['x']),u-float(pile['u'])
        c,s=math.cos(float(pile['angle'])),math.sin(float(pile['angle']))
        a=(c*dx+s*dy)/float(pile['rx']);b=(-s*dx+c*dy)/float(pile['ry'])
        r=np.maximum(np.abs(a),np.maximum(np.abs(.5*a+.8660254*b),np.abs(-.5*a+.8660254*b)))
        floor+=height*_smooth_compact_mound(r)
    return floor


def pit_floor(config: dict,pit_cfg: dict,x,y):
    """The same background, with three local talus mounds at the pit walls."""
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    floor=base_floor(config,x,y)
    cx,cy=map(float,pit_cfg['center_xy_m']);dx=x-cx;dy=y-cy
    heights=np.asarray(config.get('pit_talus_heights_m',[.65,.45,.30]),float)
    if heights.shape!=(3,) or not np.isfinite(heights).all() or np.any(heights<0) or np.any(heights>.8):
        raise ValueError('pit talus heights must be three finite values in [0,.8]m')
    floor+=heights[0]*np.exp(-((dx+15.)/8.7)**2-((dy-6.)/10.)**2)
    floor+=heights[1]*np.exp(-((dx-12.)/8.)**2-((dy-14.)/7.)**2)
    floor+=heights[2]*np.exp(-((dx-2.)/10.)**2-((dy+17.)/6.)**2)
    return floor


def lower_envelope_shift(config: dict,x,y,z,datum,height,floor):
    """Warp the full lower cave envelope down to a shared, uneven floor.

    At the new floor, the closed superellipse is sampled at a safely interior
    reference height, avoiding its old rounded bottom blocking the entrance.
    The shift smoothly vanishes near half-height, preserving the roof and
    avoiding a fixed-width corridor or an artificial driving centreline.
    """
    x,y,z,datum,height,floor=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float),np.asarray(z,float),
                                               np.asarray(datum,float),np.asarray(height,float),np.asarray(floor,float))
    lift=float(config.get('lower_envelope_reference_lift_m',1.8))
    merge_fraction=float(config.get('lower_envelope_merge_height_fraction',.52))
    if not (np.isfinite(lift) and .5<=lift<=2.5 and np.isfinite(merge_fraction) and .45<=merge_fraction<=.65):
        raise ValueError('invalid lower-envelope transition')
    delta=datum+np.minimum(lift,.20*height)-floor
    start=floor+.10
    end=datum+height*merge_fraction
    span=end-start
    if np.any(span<=1.):raise ValueError('lower ground would leave insufficient cave height')
    t=np.clip((z-start)/span,0.,1.);weight=1.-t*t*(3.-2.*t)
    # A positive derivative keeps this a lower-envelope deformation rather than
    # a folded coordinate field. The default margin is comfortably positive.
    if np.any(1.-1.5*np.maximum(delta,0.)/span<=.05):
        raise ValueError('lower-envelope warp would be too steep or folded')
    return delta*weight
