"""Deterministic, synthetic lunar-pit entrance field (metres, Z-up).

Positive field values represent air, including the space above the local lunar
surface. This is not an exact Euclidean SDF or a reconstruction of a named pit.
It can be unioned with ``lunar_morphology_v4.evaluate_field`` using ``maximum``.

The default entrance uses Marius-scale nominal dimensions reported by Wagner
and Robinson (2022), DOI 10.1029/2022JE007328, Table 1 and Section 5.2. The
outline, overhang locations, surface relief, layer ledges and talus are authored
hypotheses. Connecting this entrance to a Zhou-inspired underground section is
a synthetic composition of different sites, not an observed lunar connection.
"""
from __future__ import annotations

import numpy as np


def pit_config() -> dict:
    """Return fresh parameters, with evidence separate from authored detail."""
    return {
        "schema_version": "synthetic_lunar_pit/4.0",
        "scientific_status": "Marius-scale synthetic entrance, not a surveyed reconstruction",
        "units": "m",
        "up_axis": "Z",
        "center_xy_m": [-22.0, 0.0],
        "opening_radii_m": [27.5, 24.5],
        "regional_surface_z_m": 58.0,
        "floor_base_z_m": -2.0,
        "nominal_depth_m": 60.0,
        "funnel_depth_m": 7.0,
        "funnel_outer_scale": 0.42,
        "wall_roughness_model": "aperiodic_fracture_v2",
        "roughness_seed": 20261110,
        "wall_fracture_scale": 1.0,
        "surface_fracture_scale": 0.42,
        "floor_fracture_scale": 0.38,
        "outline_relief_m": 2.0,
        "wall_bed_relief_m": 0.0,
        "wall_bed_spacing_m": 4.8,
        "local_ledge_patches": [
            {"theta_rad": -2.65, "z_m": 12.5, "arc_width_m": 7.0, "height_extent_m": 2.0, "amplitude_m": .12},
            {"theta_rad": -1.21, "z_m": 38.0, "arc_width_m": 9.0, "height_extent_m": 2.5, "amplitude_m": -.14},
            {"theta_rad": -.08, "z_m": 26.5, "arc_width_m": 6.5, "height_extent_m": 1.7, "amplitude_m": .16},
            {"theta_rad": 1.44, "z_m": 46.0, "arc_width_m": 8.0, "height_extent_m": 2.2, "amplitude_m": -.11},
            {"theta_rad": 2.35, "z_m": 30.5, "arc_width_m": 6.0, "height_extent_m": 1.8, "amplitude_m": .13},
        ],
        "upper_lip_undercut_m": 0.85,
        "lower_undercut_m": 5.4,
        "lower_undercut_start_depth_m": 27.0,
        "lower_undercut_transition_m": 10.0,
        "terrain_relief_m": 0.65,
        "floor_relief_m": 0.48,
        "recommended_bounds_m": [[-80.0, -70.0, -10.0], [105.0, 70.0, 78.0]],
        "sources": [{
            "doi": "10.1029/2022JE007328",
            "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JE007328",
            "locator": "Table 1; Sections 5.2 and 6",
            "reported_reference": "Marius pit approximately 55 by 49 m and 60 m deep; funnel, steep walls, overhangs and floor debris",
            "scope": "Nominal entrance scale and morphology only; no point cloud was ingested.",
        }],
        "parameter_provenance": {
            "opening_radii_m": "Nominal half dimensions from the reported 55 by 49 m reference; irregular outline is authored.",
            "nominal_depth_m": "Reference scale; exact deepest point varies with the authored funnel and floor.",
            "center_and_z_datum": "Arbitrary local coordinates chosen to connect a synthetic underground segment near x=0.",
            "funnel_and_walls": "Observed morphology motivates a shallow funnel and locally overhung walls; positions and amplitudes are not fitted observations.",
            "talus_and_relief": "Deterministic authored geometric roughness and mounds, not measured individual rocks.",
            "wall_roughness_model": "Aperiodic hash-grid fracture geometry and sparse local ledge patches; no periodic all-around bed rings. Legacy configs without this flag replay the previous field.",
            "connectivity": "The connection to the Zhou-inspired cave is a generation assumption, not one measured lunar site.",
        },
        "field_scope": "Air sign/implicit margin; the exterior-air region reaches finite sampling-domain boundaries.",
    }


def _xy(config: dict, x, y):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    cx, cy = config["center_xy_m"]
    dx, dy = x - float(cx), y - float(cy)
    a, b = map(float, config["opening_radii_m"])
    theta = np.arctan2(dy, dx)
    rho = np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
    return dx, dy, theta, rho


def surface_height(config: dict, x, y):
    """Regional surface and shallow funnel; scalar and arrays share this path.

    The funnel lowers the surface just outside the shaft. Inside the opening,
    this nominal surface is removed by the shaft-air union rather than forming
    a floor or a lid. Z=58 is the default regional datum, not a fitted rim mesh.
    """
    if config.get("wall_roughness_model") == "aperiodic_fracture_v2":
        from .geologic_noise_v4 import rock_noise
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        dx, dy, _theta, rho = _xy(config, x, y)
        terrain = float(config.get("surface_fracture_scale", .42)) * rock_noise(
            x, y, float(config["regional_surface_z_m"]), seed=int(config.get("roughness_seed", 20261110))+17)
        outside = np.maximum(rho - 1.0, 0.0)
        funnel = float(config.get("funnel_depth_m", 7.0)) * np.exp(
            -(outside / float(config.get("funnel_outer_scale", .42))) ** 2)
        shoulder = (.8*np.exp(-((dx-35.)/10.)**2-((dy-13.)/8.)**2)
                    + .6*np.exp(-((dx+24.)/9.)**2-((dy+28.)/11.)**2)
                    + .9*np.exp(-((dx+10.)/12.)**2-((dy-31.)/8.)**2))
        return float(config["regional_surface_z_m"]) + terrain - funnel + shoulder
    dx, dy, theta, rho = _xy(config, x, y)
    relief = float(config.get("terrain_relief_m", 0.65))
    terrain = relief * (
        0.56 * np.sin(0.074 * dx + 0.9) * np.cos(0.061 * dy - 0.2)
        + 0.31 * np.sin(0.183 * dx - 0.113 * dy + 1.6)
        + 0.16 * np.cos(0.49 * dx + 0.38 * dy)
    )
    # A down-sloping funnel outside the opening, with no artificial ring trough.
    outside = np.maximum(rho - 1.0, 0.0)
    funnel = float(config.get("funnel_depth_m", 7.0)) * np.exp(
        -(outside / float(config.get("funnel_outer_scale", 0.42))) ** 2
    )
    # Uneven low rubble shoulders around parts of the outer funnel.
    shoulder = 0.70 * np.exp(-((rho - 1.47) / 0.21) ** 2) * (
        0.65 + 0.35 * np.sin(3.0 * theta + 0.4)
    )
    return float(config["regional_surface_z_m"]) + terrain - funnel + shoulder


def pit_floor_height(config: dict, x, y):
    """Uneven floor and contiguous talus mounds; no separate floating blocks."""
    if config.get('ground_profile_v7') is not None:
        if config.get('rover_route_floor') is not None:
            raise ValueError('global ground_profile_v7 cannot be combined with a prescribed rover route floor')
        from .lunar_ground_v7 import pit_floor
        return pit_floor(config['ground_profile_v7'],config,x,y)
    if config.get("wall_roughness_model") == "aperiodic_fracture_v2":
        from .geologic_noise_v4 import rock_noise
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        dx, dy, _theta, _rho = _xy(config, x, y)
        rough = float(config.get("floor_fracture_scale", .38)) * rock_noise(
            x, y, float(config["floor_base_z_m"]), seed=int(config.get("roughness_seed", 20261110))+41)
        west_talus = 3.9*np.exp(-((dx+15.)/8.7)**2-((dy-6.)/10.)**2)
        north_talus = 2.6*np.exp(-((dx-12.)/8.)**2-((dy-14.)/7.)**2)
        south_talus = 1.7*np.exp(-((dx-2.)/10.)**2-((dy+17.)/6.)**2)
        floor=float(config["floor_base_z_m"]) + rough + west_talus + north_talus + south_talus
        if config.get('rover_route_floor') is not None:
            from .rover_route_v6 import apply_floor
            floor=apply_floor(config['rover_route_floor'],x,y,floor)
        return floor
    dx, dy, _theta, _rho = _xy(config, x, y)
    relief = float(config.get("floor_relief_m", 0.48))
    rough = relief * (
        0.68 * np.sin(0.22 * dx + 0.4) * np.cos(0.18 * dy - 0.6)
        + 0.38 * np.sin(0.53 * dx + 0.31 * dy)
        + 0.19 * np.cos(1.11 * dx - 0.73 * dy + 1.1)
    )
    # Large mounds cluster off the east-side portal instead of leveling the pit.
    west_talus = 3.9 * np.exp(-((dx + 15.0) / 8.7) ** 2 - ((dy - 6.0) / 10.0) ** 2)
    north_talus = 2.6 * np.exp(-((dx - 12.0) / 8.0) ** 2 - ((dy - 14.0) / 7.0) ** 2)
    south_talus = 1.7 * np.exp(-((dx - 2.0) / 10.0) ** 2 - ((dy + 17.0) / 6.0) ** 2)
    floor=float(config["floor_base_z_m"]) + rough + west_talus + north_talus + south_talus
    if config.get('rover_route_floor') is not None:
        from .rover_route_v6 import apply_floor
        floor=apply_floor(config['rover_route_floor'],x,y,floor)
    return floor


def _angular_delta(theta, center):
    return (theta - float(center) + np.pi) % (2.*np.pi) - np.pi


def _aperiodic_wall_base(config, theta, z):
    """Gross wall, angular fracture blocks and sparse ledges, without noise."""
    theta, z = np.broadcast_arrays(np.asarray(theta, float), np.asarray(z, float))
    a, b = map(float, config["opening_radii_m"])
    ellipse = a*b/np.sqrt((b*np.cos(theta))**2+(a*np.sin(theta))**2)
    depth = np.maximum(float(config["regional_surface_z_m"])-z, 0.)
    # Unequally spaced angular blocks make real corners without a repeated wave.
    outline = np.zeros_like(theta)
    for center, width, amplitude in [(-2.85,.22,-.75),(-1.68,.31,1.),(-.37,.19,-.65),(.81,.38,.85),(2.09,.24,-.55)]:
        outline += amplitude*np.maximum(1.-np.abs(_angular_delta(theta,center))/width,0.)
    outline *= float(config.get("outline_relief_m",2.))
    lip_direction = np.exp(-.5*(_angular_delta(theta,-1.0)/1.1)**2)
    upper_lip = float(config.get("upper_lip_undercut_m",.85))*(1.-np.exp(-depth/4.))*(.7+.3*lip_direction)
    q = np.clip((depth-float(config.get("lower_undercut_start_depth_m",27.)))/float(config.get("lower_undercut_transition_m",10.)),0.,1.)
    smooth = q*q*(3.-2.*q)
    deep_direction = np.exp(-.5*(_angular_delta(theta,2.1)/.9)**2)
    deep = float(config.get("lower_undercut_m",5.4))*smooth*(.48+.52*deep_direction)
    ledges = np.zeros_like(theta)
    for patch in config.get("local_ledge_patches",[]):
        arc = ellipse*_angular_delta(theta,patch['theta_rad'])/patch['arc_width_m']
        dz = (z-patch['z_m'])/patch['height_extent_m']
        ledges += patch['amplitude_m']*np.exp(-arc*arc-dz*dz)*np.tanh(dz*2.)
    base_depth = float(config["regional_surface_z_m"])-float(config["floor_base_z_m"])
    toe = .8*np.clip((depth-(base_depth-6.))/6.,0.,1.)**2
    return ellipse+outline+upper_lip+deep+ledges-toe


def _wall_fracture(config, theta, z, base_radius):
    from .geologic_noise_v4 import rock_noise
    cx,cy=map(float,config['center_xy_m'])
    return float(config.get('wall_fracture_scale',1.))*rock_noise(
        cx+base_radius*np.cos(theta),cy+base_radius*np.sin(theta),z,
        seed=int(config.get('roughness_seed',20261110)))


def pit_wall_radius(config: dict, theta, z):
    """Synthetic wall radius along a ray, including local overhang and ledges.

    The upper wall remains near vertical. The deeper expansion is directional,
    not a complete, uniform undercut ring. This models the morphology class,
    without claiming the reported local wall angles were fitted here.
    """
    theta, z = np.broadcast_arrays(np.asarray(theta, float), np.asarray(z, float))
    if config.get("wall_roughness_model") == "aperiodic_fracture_v2":
        base = _aperiodic_wall_base(config,theta,z)
        return base+_wall_fracture(config,theta,z,base)
    a, b = map(float, config["opening_radii_m"])
    ellipse = a * b / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
    depth = np.maximum(float(config["regional_surface_z_m"]) - z, 0.0)
    outline = float(config.get("outline_relief_m", 1.05)) * (
        0.52 * np.sin(3.0 * theta + 0.5)
        + 0.31 * np.cos(5.0 * theta - 0.8)
        + 0.17 * np.sin(9.0 * theta + 0.4)
    )
    upper_lip = float(config.get("upper_lip_undercut_m", 0.85)) * (
        1.0 - np.exp(-depth / 4.0)
    ) * (0.72 + 0.28 * np.cos(2.0 * theta - 0.2))
    deep_start = float(config.get("lower_undercut_start_depth_m", 27.0))
    transition = float(config.get("lower_undercut_transition_m", 10.0))
    q = np.clip((depth - deep_start) / transition, 0.0, 1.0)
    smooth = q * q * (3.0 - 2.0 * q)
    deep = float(config.get("lower_undercut_m", 5.4)) * smooth * (
        0.48 + 0.52 * (0.5 + 0.5 * np.cos(theta - 2.1)) ** 2
    )
    beds = float(config.get("wall_bed_relief_m", 0.24)) * np.sin(
        2.0 * np.pi * depth / float(config.get("wall_bed_spacing_m", 4.8))
        + 0.25 * np.sin(3.0 * theta)
    )
    macro = 0.36 * np.sin(0.16 * depth + 2.0 * theta) * np.cos(5.0 * theta + 0.08 * depth)
    # A gentle inward taper at the very base represents talus against the wall.
    base_depth = float(config["regional_surface_z_m"]) - float(config["floor_base_z_m"])
    toe = 0.8 * np.clip((depth - (base_depth - 6.0)) / 6.0, 0.0, 1.0) ** 2
    return ellipse + outline + upper_lip + deep + beds + macro - toe


def evaluate_pit_field(config: dict, pointsNx3):
    """Return positive air margins for one 3-vector or a finite Nx3 array.

    Above-surface air and shaft air are united. Below-floor points remain solid.
    No lateral cave is fabricated here: the caller adds its separately sourced
    cave field using ``np.maximum``. This function allocates only O(N) arrays.
    """
    points = np.asarray(pointsNx3, dtype=float)
    single = points.shape == (3,)
    if single:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite 3-vector or Nx3 array")
    x, y, z = points.T
    if config.get("wall_roughness_model") == "aperiodic_fracture_v2":
        field=_aperiodic_field_arrays(config,x,y,z)
        return float(field[0]) if single else field
    dx, dy, theta, _rho = _xy(config, x, y)
    radial = pit_wall_radius(config, theta, z) - np.hypot(dx, dy)
    floor = z - pit_floor_height(config, x, y)
    shaft = np.minimum(radial, floor)
    outside = z - surface_height(config, x, y)
    field = np.maximum(outside, shaft)
    return float(field[0]) if single else field


def _aperiodic_field_arrays(config,x,y,z):
    """Broadcast path: reuse XY surface/floor values through streamed Z slabs."""
    dx,dy,theta,_rho=_xy(config,x,y)
    base=_aperiodic_wall_base(config,theta,z)
    radial=np.asarray(base-np.hypot(dx,dy),dtype=float).copy()
    # Noise is bounded by the component amplitudes. Six metres exceeds that
    # bound by several voxels, so skipping it here cannot clip the zero set.
    # Its far-field margin is not promised to be an exact Euclidean distance.
    near=np.abs(radial)<6.*max(1.,abs(float(config.get('wall_fracture_scale',1.))))
    if np.any(near):
        theta_all=np.broadcast_to(theta,radial.shape)
        z_all=np.broadcast_to(np.asarray(z,float),radial.shape)
        radial[near] += _wall_fracture(config,theta_all[near],z_all[near],base[near])
    floor=np.asarray(z,float)-pit_floor_height(config,x,y)
    outside=np.asarray(z,float)-surface_height(config,x,y)
    return np.maximum(outside,np.minimum(radial,floor))


def evaluate_pit_grid(config: dict, x, y, z):
    """Same positive-air model on broadcast coordinate grids, with less work."""
    if config.get("wall_roughness_model") == "aperiodic_fracture_v2":
        return _aperiodic_field_arrays(config,x,y,z)
    x,y,z=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float),np.asarray(z,float))
    points=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    return evaluate_pit_field(config,points).reshape(x.shape)


# Explicit alias for callers that name floor queries consistently with the body.
floor_height = pit_floor_height
