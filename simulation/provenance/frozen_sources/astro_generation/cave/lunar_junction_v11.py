"""Locally rounded CSG fork junction with an unchanged shared ground surface.

Hard maximum of two positive-air fields leaves a sharp rock wedge at their
intersection. A compact C1 polynomial smooth maximum rounds that intersection,
then an explicit shared-floor intersection prevents it excavating the ground.
The blend dimensions are authored engineering parameters, not paper-measured
lunar dimensions. Legacy scenes without ``junction_v11`` keep their exact max.
"""
from __future__ import annotations

import numpy as np


def junction_config() -> dict:
    """A fresh, local blend centred on the measured V10 fork intersection."""
    return dict(
        schema_version='rounded_lunar_junction/11.0',
        model='compact_polynomial_smooth_max_shared_floor', enabled=True,
        support_center_xy_m=[50., 9.], support_radii_xy_m=[15., 12.],
        full_strength_inner_radius_fraction=.55,
        blend_window_m=6.0,
        blend_aperiodic_fraction=.08, blend_noise_scale_m=4.5,
        blend_noise_seed=20261110+1103,
        z_support_above_floor_m=[-1., 20.], z_full_strength_above_floor_m=[0., 14.],
        floor_clip='minimum_of_blended_air_and_z_minus_shared_cavern_floor',
        gradient_normalization=False,
        reference_scope='blunt irregular residual rock between connected lava passages; no lunar fork survey is asserted',
        parameter_provenance='junction centre from old implicit-field sections; blend size and irregularity are authored engineering choices',
        paper_dimensions_used_as_junction_parameters=False,
    )


def _smoothstep(value):
    value = np.clip(value, 0., 1.)
    return value*value*value*(value*(value*6.-15.)+10.)


def _configuration(scene):
    value = scene.get('junction_v11')
    if value is None or value is False:
        return None
    if not isinstance(value, dict):
        raise TypeError('junction_v11 must be an explicit parameter dictionary')
    if value.get('enabled', True) is False:
        return None
    config = junction_config();config.update(value)
    if config['model'] != 'compact_polynomial_smooth_max_shared_floor':
        raise ValueError('Unknown junction composition model')
    center = np.asarray(config['support_center_xy_m'], float)
    radii = np.asarray(config['support_radii_xy_m'], float)
    extent = np.asarray(config['z_support_above_floor_m'], float)
    full = np.asarray(config['z_full_strength_above_floor_m'], float)
    parameters = [config['full_strength_inner_radius_fraction'], config['blend_window_m'],
                  config['blend_aperiodic_fraction'], config['blend_noise_scale_m']]
    if center.shape != (2,) or radii.shape != (2,) or extent.shape != (2,) or full.shape != (2,):
        raise ValueError('Invalid junction support dimensions')
    if not np.isfinite(np.r_[center, radii, extent, full, parameters]).all():
        raise ValueError('Junction parameters must be finite')
    if np.any(radii < 2.) or np.any(radii > 30.) or not extent[0] < full[0] < full[1] < extent[1]:
        raise ValueError('Junction support and taper ranges are invalid')
    if not .1 <= float(parameters[0]) <= .85 or not .1 <= float(parameters[1]) <= 8.:
        raise ValueError('Invalid junction blend window')
    if not 0. <= float(parameters[2]) <= .20 or not 1. <= float(parameters[3]) <= 12.:
        raise ValueError('Invalid junction geological variation')
    if config.get('gradient_normalization') is not False:
        raise ValueError('This model operates on the supplied metre-valued implicit margins')
    return config, center, radii, extent, full


def compose_junction_field(scene: dict, pointsNx3, air_a, air_b):
    """Compose pit/main air A and branch air B, with scalar/N×3 support.

    Outside the compact junction support, the result is bitwise the original
    hard max. Within it, a true polynomial smooth max removes the CSG crease;
    existing aperiodic fractured surfaces remain. The floor clamp is mandatory.
    Input margins are metre-valued implicit fields, not exact Euclidean SDFs.
    """
    if not isinstance(scene, dict):
        raise TypeError('scene must be a dictionary')
    points = np.asarray(pointsNx3, float)
    single = points.shape == (3,)
    if single:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Junction points must be a finite XYZ vector or N×3 array')
    a, b = np.asarray(air_a), np.asarray(air_b)
    try:
        a, b = np.broadcast_to(a, (len(points),)), np.broadcast_to(b, (len(points),))
    except ValueError as exc:
        raise ValueError('Air arrays must be scalar or match the point count') from exc
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Junction air fields must be finite')
    result = np.maximum(a, b)
    configuration = _configuration(scene)
    if configuration is None:
        return float(result[0]) if single else result
    config, center, radii, z_extent, z_full = configuration
    rho = np.linalg.norm((points[:, :2]-center)/radii, axis=1)
    candidates = np.flatnonzero(rho < 1.)
    if not len(candidates):
        return float(result[0]) if single else result
    ground = scene.get('ground_profile_v7', scene.get('cavern', {}).get('ground_profile_v7'))
    cavern = scene.get('cavern')
    if not isinstance(ground, dict) or not isinstance(cavern, dict):
        raise ValueError('Rounded junction requires the actual shared v7 cavern floor')
    from .lunar_ground_v7 import cavern_floor
    from .geologic_noise_v4 import value_noise
    xyz = points[candidates]
    floor = np.asarray(cavern_floor(ground, cavern, xyz[:, 0], xyz[:, 1]), float)
    relative_z = xyz[:, 2]-floor
    z_weight = _smoothstep((relative_z-z_extent[0])/(z_full[0]-z_extent[0]))
    z_weight *= 1.-_smoothstep((relative_z-z_full[1])/(z_extent[1]-z_full[1]))
    inner = float(config['full_strength_inner_radius_fraction'])
    xy_weight = 1.-_smoothstep((rho[candidates]-inner)/(1.-inner))
    weight = np.clip(xy_weight*z_weight, 0., 1.)
    active = weight > 0.
    if not active.any():
        return float(result[0]) if single else result
    rows = candidates[active];xyz = xyz[active];floor = floor[active]
    scale = float(config['blend_noise_scale_m'])
    irregularity = value_noise(xyz[:, 0]/scale+3.7, xyz[:, 1]/scale-6.2,
                              xyz[:, 2]/scale+9.4, seed=int(config['blend_noise_seed']))
    k = float(config['blend_window_m'])*weight[active]*(1.+float(config['blend_aperiodic_fraction'])*irregularity)
    av, bv = np.asarray(a[rows], float), np.asarray(b[rows], float)
    h = np.maximum(1.-np.abs(av-bv)/k, 0.)
    blended = np.maximum(av, bv)+.25*k*h*h
    # Without this cap, smooth-max of equal floor margins adds k/4 and makes
    # the floor lower than either original cave. Keep the real uneven ground.
    blended = np.minimum(blended, xyz[:, 2]-floor)
    result = np.asarray(result, float).copy();result[rows] = blended
    return float(result[0]) if single else result
