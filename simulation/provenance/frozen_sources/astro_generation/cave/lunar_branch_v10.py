"""One closed, drivable-scale side branch for the authored lunar cave.

The 80 m main cave is not extended. This positive-air field can be unioned
with the existing scene using ``maximum``. The branch bends to +Y, shares the
world-XY floor and junction deposits, and ends below the existing overburden.
It is a synthetic topology hypothesis, not a surveyed lunar connection.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import math

import numpy as np


def branch_config() -> dict:
    """Return a fresh, self-contained branch config; full scenes override floor data."""
    from .lunar_ground_v7 import ground_config
    from .lunar_morphology_v4 import candidate_config
    return {
        'schema_version': 'synthetic_lunar_equal_caliber_fork/10.1',
        'morphology_model': 'equal_caliber_aperiodic_v10b',
        'units': 'm', 'up_axis': 'Z',
        'centerline_xy_m': [[24., 1.], [31., 4.], [40., 13.],
                            [48., 25.], [56., 37.], [64., 43.]],
        'half_widths_m': [10.6, 11.1, 10.1, 11.2, 10.2, 10.7],
        'heights_m': [15.8, 16.4, 14.6, 16.7, 15.2, 16.0],
        'cross_section_exponent': 3.0,
        'lower_envelope_center_height_fraction': .42,
        'lower_envelope_half_height_fraction': .58,
        'roughness_gain': .72,
        'roughness_seed': 20261110 + 1007,
        'centerline_samples_per_span': 10,
        'evaluation_bounds_m': [[8., -15., -7.], [82., 61., 22.]],
        'axis_lateral_relief_m': 1.10,
        'longitudinal_width_relief': .055,
        'wall_side_asymmetry': .045,
        'roof_sag_m': 1.7,
        'wall_buttress_m': 2.8,
        'ground_profile_v7': ground_config(),
        # A stand-alone branch config reproduces the default junction floor.
        # A complete scene always uses its actual cavern deposits instead.
        'junction_floor_deposits': copy.deepcopy(candidate_config()['floor_deposits']),
        'junction_reference_xy_m': [36., 8.],
        'branch_goal_xy_m': [63., 42.],
        'capped_end_center_xy_m': [64., 43.],
        'main_cave_longitudinal_length_m_preserved': 80.,
        'topology_scope': 'two comparable-width natural cave branches; new arm has a closed underground cap, not a narrow added pipe',
        'floor_policy': 'same v7 world-XY relief and existing junction deposits; no route-conditioned flattening',
        'provenance': 'authored simulation topology, width, height, bends and roughness; not measured lunar geometry',
    }


def _parts(config):
    if not isinstance(config, dict):
        raise TypeError('branch or scene config must be a dict')
    if 'branch_v10' in config:
        branch = config['branch_v10']
        if not isinstance(branch, dict):
            raise TypeError('scene branch_v10 must be a dict')
        ground = config.get('ground_profile_v7', branch.get('ground_profile_v7'))
        cavern = config.get('cavern', {'floor_deposits': branch.get('junction_floor_deposits', [])})
    else:
        branch = config
        ground = branch.get('ground_profile_v7')
        cavern = {'floor_deposits': branch.get('junction_floor_deposits', [])}
    if not isinstance(ground, dict):
        raise ValueError('branch requires the shared v7 ground profile')
    return branch, ground, cavern


def branch_floor_height(config: dict, x, y):
    """Shared actual cavern floor, including smooth local junction deposits.

    Accepts either a full scene containing ``branch_v10`` or ``branch_config``.
    This is the intended floor reference. Close to rounded side walls the
    actual support envelope may be higher; validate vehicle poses on the mesh.
    """
    from .lunar_ground_v7 import cavern_floor
    _, ground, cavern = _parts(config)
    values = cavern_floor(ground, cavern, x, y)
    return float(values) if np.ndim(values) == 0 else values


def _parameters(branch):
    points = np.asarray(branch.get('centerline_xy_m'), float)
    widths = np.asarray(branch.get('half_widths_m'), float)
    heights = np.asarray(branch.get('heights_m'), float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('branch centerline must contain at least three XY points')
    if widths.shape != (len(points),) or heights.shape != (len(points),):
        raise ValueError('branch widths and heights must match centerline vertices')
    if not np.isfinite(np.r_[points.ravel(), widths, heights]).all():
        raise ValueError('branch geometry must be finite')
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1.):
        raise ValueError('branch centerline anchors are too close')
    wide = branch.get('morphology_model') == 'equal_caliber_aperiodic_v10b'
    if np.any(widths < (8. if wide else 2.5)) or np.any(widths > (16. if wide else 8.)) or np.any(heights < (12. if wide else 4.)) or np.any(heights > (20. if wide else 12.)):
        raise ValueError('branch width/height is outside the authored small-branch range')
    exponent = float(branch.get('cross_section_exponent', 2.5))
    center = float(branch.get('lower_envelope_center_height_fraction', .25))
    half_height = float(branch.get('lower_envelope_half_height_fraction', .75))
    gain = float(branch.get('roughness_gain', .24))
    samples = int(branch.get('centerline_samples_per_span', 12))
    if not (2. <= exponent <= 4. and .15 <= center <= (.48 if wide else .35) and
            (.52 if wide else .6) <= half_height <= .85):
        raise ValueError('invalid branch cross-section parameters')
    if not math.isclose(center+half_height, 1., abs_tol=1e-8):
        raise ValueError('upper envelope fractions must sum to one')
    if not (0. <= gain <= (1. if wide else .4) and 4 <= samples <= 24):
        raise ValueError('invalid branch roughness or interpolation density')
    bounds = np.asarray(branch.get('evaluation_bounds_m'), float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError('branch evaluation bounds must be two increasing XYZ vectors')
    return points, widths, heights, exponent, center, half_height, gain, samples, bounds


@lru_cache(maxsize=16)
def _sample_curve(points_tuple, widths_tuple, heights_tuple, samples):
    """Cubic Catmull-Rom centreline with smoothly interpolated cross sections."""
    anchors = np.asarray(points_tuple, float)
    widths, heights = np.asarray(widths_tuple), np.asarray(heights_tuple)
    curve, curve_widths, curve_heights = [], [], []
    for i in range(len(anchors)-1):
        p0 = anchors[max(0, i-1)]
        p1, p2 = anchors[i], anchors[i+1]
        p3 = anchors[min(len(anchors)-1, i+2)]
        values = np.linspace(0., 1., samples, endpoint=False)
        for t in values:
            xy = .5*((2.*p1)+(-p0+p2)*t+(2.*p0-5.*p1+4.*p2-p3)*t*t
                      +(-p0+3.*p1-3.*p2+p3)*t*t*t)
            smooth = t*t*(3.-2.*t)
            curve.append(xy)
            curve_widths.append((1.-smooth)*widths[i]+smooth*widths[i+1])
            curve_heights.append((1.-smooth)*heights[i]+smooth*heights[i+1])
    curve.append(anchors[-1]);curve_widths.append(widths[-1]);curve_heights.append(heights[-1])
    curve = np.asarray(curve, float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    return curve, np.asarray(curve_widths), np.asarray(curve_heights), arc


def sampled_centerline(config: dict):
    """Return independent XY/width/height/arc arrays for geometric diagnostics."""
    branch, _, _ = _parts(config)
    points, widths, heights, _, _, _, _, samples, _ = _parameters(branch)
    return tuple(a.copy() for a in _sample_curve(tuple(map(tuple, points)), tuple(widths), tuple(heights), samples))


def _nearest_profile(x, y, curve, widths, heights, details=False):
    distance2 = np.full(x.shape, np.inf)
    chosen_width = np.empty(x.shape)
    chosen_height = np.empty(x.shape)
    if details:
        chosen_arc = np.empty(x.shape)
        chosen_side = np.empty(x.shape)
        chosen_cap2 = np.empty(x.shape)
    arc = 0.
    for i, (a, b) in enumerate(zip(curve[:-1], curve[1:])):
        vector = b-a;length2 = float(vector@vector)
        t = np.clip(((x-a[0])*vector[0]+(y-a[1])*vector[1])/length2, 0., 1.)
        dx, dy = x-(a[0]+t*vector[0]), y-(a[1]+t*vector[1])
        candidate = dx*dx+dy*dy
        selected = candidate < distance2
        distance2[selected] = candidate[selected]
        chosen_width[selected] = (1.-t[selected])*widths[i]+t[selected]*widths[i+1]
        chosen_height[selected] = (1.-t[selected])*heights[i]+t[selected]*heights[i+1]
        if details:
            length = math.sqrt(length2)
            side = (-dx*vector[1]+dy*vector[0])/length
            chosen_arc[selected] = arc+t[selected]*length
            chosen_side[selected] = side[selected]
            chosen_cap2[selected] = np.maximum(candidate[selected]-side[selected]*side[selected], 0.)
            arc += length
    if details:
        return np.sqrt(distance2), chosen_width, chosen_height, dict(
            arc_m=chosen_arc, signed_lateral_m=chosen_side, cap_distance2_m=chosen_cap2, total_arc_m=arc)
    return np.sqrt(distance2), chosen_width, chosen_height


def _natural_equal_caliber_body(branch, x, y, z, floor, width, height, profile, exponent, center, half_height):
    """Unequal shoulders and broad, local roof sag rather than an extruded pipe."""
    from .geologic_noise_v4 import value_noise
    seed = int(branch.get('roughness_seed', 20262117))
    s = profile['arc_m'];length = profile['total_arc_m']
    axis_relief = float(branch.get('axis_lateral_relief_m', 1.10))
    width_relief = float(branch.get('longitudinal_width_relief', .055))
    asymmetry = float(branch.get('wall_side_asymmetry', .045))
    sag_gain = float(branch.get('roof_sag_m', 1.7))
    buttress_gain = float(branch.get('wall_buttress_m', 2.8))
    values = [axis_relief, width_relief, asymmetry, sag_gain, buttress_gain]
    if not np.isfinite(values).all() or not (0 <= axis_relief <= 1.5 and 0 <= width_relief <= .08 and
            0 <= asymmetry <= .07 and 0 <= sag_gain <= 3. and 0 <= buttress_gain <= 4.):
        raise ValueError('invalid equal-caliber natural morphology relief')
    # Aperiodic along-arc variations are authored in physical scales and do not
    # clear a driving strip. Local deposits and world-XY ground remain intact.
    offset = axis_relief*value_noise(s/10.5+6.1, 1.7, 4.2, seed=seed+11)
    lateral = profile['signed_lateral_m']-offset
    width = width*(1.+width_relief*value_noise(s/13.5-4.2, 8.7, 1.1, seed=seed+19)
                    +.022*value_noise(s/4.8+2.5, 3.4, 7.2, seed=seed+29))
    left = width*(1.+asymmetry*value_noise(s/11.0+9.3, 5.5, 3.1, seed=seed+41))
    right = width*(1.-asymmetry*value_noise(s/12.3-3.7, 7.8, 2.4, seed=seed+53))
    side_width = np.where(lateral < 0., left, right)
    distance = np.sqrt(lateral*lateral+profile['cap_distance2_m'])
    height = height*(1.+.035*value_noise(s/12.0+4.3, 8.9, 1.2, seed=seed+67))
    vertical = z-floor
    top = np.clip((vertical/height-.35)/.45, 0., 1.)
    top = top*top*(3.-2.*top)
    sag = sag_gain*np.exp(-((s-.49*length)/8.5)**2-((lateral+4.3)/5.2)**2)
    sag += .65*sag_gain*np.exp(-((s-.78*length)/6.8)**2-((lateral-5.5)/4.5)**2)
    warped = vertical+top*(sag+.35*lateral/np.maximum(width, 1.))
    radius = (np.abs(distance/side_width)**exponent+
              np.abs((warped-center*height)/(half_height*height))**exponent)**(1./exponent)
    body = (1.-radius)*np.minimum(side_width, half_height*height)
    # Solid wall-attached shoulders, not decorative side caves or periodic
    # rings. Their opposite-side positions differ along the fork.
    left_mass = buttress_gain*np.exp(-((s-.34*length)/7.5)**2-((vertical-6.0)/5.5)**2)
    right_mass = .80*buttress_gain*np.exp(-((s-.67*length)/8.5)**2-((vertical-7.5)/6.)**2)
    shoulder = np.where(lateral < 0., left-left_mass+lateral, right-right_mass-lateral)
    return body, shoulder


def evaluate_branch_field(config: dict, pointsNx3):
    """Return positive air; accepts finite XYZ or N×3, never opens the surface.

    The field is an implicit margin, not an exact Euclidean SDF. Outside the
    declared evaluation box it returns a negative value, so a caller may query
    the full scene or mask the branch box before evaluating it. Union with the
    existing field using ``np.maximum``; do not replace the main-cave field.
    """
    branch, _, _ = _parts(config)
    points = np.asarray(pointsNx3, float)
    single = points.shape == (3,)
    if single:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('branch points must be a finite XYZ vector or N×3 array')
    anchors, widths, heights, exponent, center, half_height, gain, samples, bounds = _parameters(branch)
    curve, sampled_widths, sampled_heights, _ = _sample_curve(
        tuple(map(tuple, anchors)), tuple(widths), tuple(heights), samples)
    result = np.full(len(points), -1000., dtype=float)
    mask = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
    if np.any(mask):
        x, y, z = points[mask].T
        floor = np.asarray(branch_floor_height(config, x, y), float)
        # The closed roof envelope extends below ground. Clipping it with the
        # *shared* floor preserves a broad uneven driving surface instead of a
        # rounded-bottom 1m entrance step. Its end in XY is a closed round cap.
        wide = branch.get('morphology_model') == 'equal_caliber_aperiodic_v10b'
        if wide:
            _, width, height, profile = _nearest_profile(x, y, curve, sampled_widths, sampled_heights, details=True)
            body, shoulder = _natural_equal_caliber_body(branch, x, y, z, floor, width, height,
                profile, exponent, center, half_height)
        else:
            distance, width, height = _nearest_profile(x, y, curve, sampled_widths, sampled_heights)
            vertical = (z-floor-center*height)/(half_height*height)
            radius = (np.abs(distance/width)**exponent + np.abs(vertical)**exponent)**(1./exponent)
            body = (1.-radius)*np.minimum(width, half_height*height)
        if gain:
            from .geologic_noise_v4 import rock_noise
            body += gain*rock_noise(x, y, z, seed=int(branch.get('roughness_seed', 20262117)))
        if wide:
            body = np.minimum(body, shoulder)
        result[mask] = np.minimum(body, z-floor)
    return float(result[0]) if single else result
