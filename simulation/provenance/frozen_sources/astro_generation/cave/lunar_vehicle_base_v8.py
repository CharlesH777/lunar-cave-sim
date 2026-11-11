"""Original static search targets for a lunar cave, in metres and Z-up.

The caller owns the target root pose. These models add no lights, roads,
signboards, wheel controllers, downloaded assets, or rigid bodies.
"""
from __future__ import annotations

import math
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .props import box, cylinder, material, ring, tube, xform


def _palette(stage):
    return {
        'gold': material(stage, 'V8Target_GoldInsulation', (.62, .38, .052), .46, .65),
        'gold_dark': material(stage, 'V8Target_GoldSeam', (.26, .125, .018), .60, .58),
        'silver': material(stage, 'V8Target_Aluminium', (.57, .61, .65), .29, .79),
        'white': material(stage, 'V8Target_CeramicWhite', (.78, .81, .83), .54, .08),
        'offwhite': material(stage, 'V8Target_BaseSeam', (.44, .49, .54), .56, .16),
        'rubber': material(stage, 'V8Target_DarkWheel', (.027, .031, .036), .79, .05),
        'solar': material(stage, 'V8Target_SolarCell', (.014, .039, .085), .26, .25),
        'grid': material(stage, 'V8Target_CellGrid', (.11, .19, .25), .38, .60),
        'glass': material(stage, 'V8Target_Window', (.009, .035, .060), .11, .55),
        'blue': material(stage, 'V8Target_AirlockBlue', (.046, .11, .17), .45, .30),
    }


def _sphere(stage, path, position, scales, mat, collision=False):
    root = xform(stage, path, position)
    root.AddScaleOp().Set(Gf.Vec3f(*map(float, scales)))
    shape = UsdGeom.Sphere.Define(stage, path+'/Shape')
    shape.CreateRadiusAttr(1.)
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
    return root


def _panel(stage, path, position, size, palette, rotation=None):
    """A framed photovoltaic panel with individually readable cell divisions."""
    frame = xform(stage, path, position, rotation or (0., 0., 0.))
    sx, sy = map(float, size)
    box(stage, path+'/Frame', (0, 0, 0), (sx, sy, .038), palette['silver'],
        .012, collision=True)
    box(stage, path+'/CellSurface', (0, 0, .023), (sx-.045, sy-.045, .012),
        palette['solar'], .004)
    # Thin electrical busbars remain visible at ordinary inspection distances.
    nx, ny = 8, 6
    for i in range(1, nx):
        x = -(sx-.045)/2+i*(sx-.045)/nx
        box(stage, path+f'/CellColumn{i}', (x, 0, .0305),
            (.003, sy-.045, .0015), palette['grid'], 0.)
    for i in range(1, ny):
        y = -(sy-.045)/2+i*(sy-.045)/ny
        box(stage, path+f'/CellRow{i}', (0, y, .0305),
            (sx-.045, .003, .0015), palette['grid'], 0.)
    return frame


def _metadata(stage, path, label, class_name, contacts, camera_target_height):
    root = stage.GetPrimAtPath(path)
    root.CreateAttribute('cave:searchClass', Sdf.ValueTypeNames.String, custom=True).Set(class_name)
    root.CreateAttribute('cave:labelZh', Sdf.ValueTypeNames.String, custom=True).Set(label)
    root.CreateAttribute('cave:scope', Sdf.ValueTypeNames.String, custom=True).Set(
        'original static search target with simplified collision; no vehicle or habitat dynamics')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    radius = 0.
    collision_paths = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Static target must not contain a rigid body')
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Boundable):
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not bounds.IsEmpty():
                low = np.asarray(bounds.GetMin(), float)
                high = np.asarray(bounds.GetMax(), float)
                extent = np.maximum(np.abs(low[:2]), np.abs(high[:2]))
                radius = max(radius, float(np.linalg.norm(extent)))
    bounds = cache.ComputeWorldBound(root).ComputeAlignedRange()
    low = np.asarray(bounds.GetMin(), float)
    high = np.asarray(bounds.GetMax(), float)
    # Root is authored at identity by this function. This is a per-boundable
    # conservative circle, rather than combining unrelated far X/Y extrema.
    view_direction = np.asarray([1., -.65], float)
    view_direction /= np.linalg.norm(view_direction)
    return dict(prim_path=path, label_zh=label, class_name=class_name,
        local_ground_footprint_radius_m=round(radius+.02, 6), height_m=float(high[2]),
        local_bounds_m=[low.tolist(), high.tolist()], ground_contact_points_local_m=contacts,
        preferred_view_direction_local_xy=view_direction.tolist(),
        camera_target_height_m=float(camera_target_height),
        ground_origin_z_m=0., forward_axis='+X', collision_prim_count=len(collision_paths),
        collision_prim_paths=collision_paths,
        collision_scope='static body/wheels/panels/feet; small trim is visual geometry',
        rigid_body=False, authored_asset=True, downloaded_asset=False)


def _fresh_root(stage, path):
    if stage.GetPrimAtPath(path):
        raise ValueError('Refusing to replace an existing search target: '+path)
    return xform(stage, path)


def build_target_rover(stage, path):
    """Build a gold six-wheel rover, distinct from the dark observer vehicle."""
    _fresh_root(stage, path)
    m = _palette(stage)
    box(stage, path+'/Chassis', (0, 0, .43), (1.08, .70, .28), m['gold'],
        .042, collision=True)
    box(stage, path+'/LowerChassis', (0, 0, .31), (.94, .61, .10), m['silver'],
        .022, collision=True)
    # Raised strips and stitched edges read as thermal insulation rather than
    # a second version of the olive-green navigation rover.
    for i, y in enumerate([-.352, .352]):
        box(stage, path+f'/FoilSeam{i}', (0, y, .43), (.97, .008, .018),
            m['gold_dark'], .003)
        for j, x in enumerate(np.linspace(-.44, .44, 7)):
            box(stage, path+f'/FoilPleat{i}_{j}', (float(x), y, .43),
                (.012, .012, .21), m['gold_dark'], .002)
    contacts = []
    for i, xx in enumerate([-.43, 0., .43]):
        cylinder(stage, path+f'/Axle{i}', (xx, 0, .205), .028, .99,
            m['silver'], 'Y', collision=True)
        for side, sign in enumerate([-1, 1]):
            yy = sign*.48
            cylinder(stage, path+f'/Wheel{i}_{side}', (xx, yy, .205),
                .19, .145, m['rubber'], 'Y', collision=True)
            cylinder(stage, path+f'/Hub{i}_{side}', (xx, yy+sign*.078, .205),
                .106, .017, m['silver'], 'Y')
            cylinder(stage, path+f'/HubCap{i}_{side}', (xx, yy+sign*.091, .205),
                .047, .011, m['gold'], 'Y')
            for k in range(16):
                angle = 2*math.pi*k/16
                box(stage, path+f'/Grouser{i}_{side}_{k}',
                    (xx+.185*math.sin(angle), yy, .205+.185*math.cos(angle)),
                    (.038, .152, .021), m['silver'], .003,
                    rot=(0, math.degrees(angle), 0))
            contacts.append([xx, yy, 0.])
    for side, sign in enumerate([-1, 1]):
        tube(stage, path+f'/Suspension{side}', [(-.43, sign*.36, .21),
            (-.18, sign*.39, .34), (.17, sign*.39, .34), (.43, sign*.36, .21)],
            .034, m['silver'])
    _panel(stage, path+'/SolarDeck', (-.06, 0, .615), (1.10, .84), m)
    cylinder(stage, path+'/CameraMast', (.40, 0, .91), .022, .55,
        m['silver'], collision=True)
    box(stage, path+'/CameraHead', (.43, 0, 1.195), (.18, .30, .105),
        m['white'], .024, collision=True)
    for i, yy in enumerate([-.097, .097]):
        cylinder(stage, path+f'/OpticsHousing{i}', (.53, yy, 1.195), .035,
            .026, m['rubber'], 'X')
        cylinder(stage, path+f'/OpticsGlass{i}', (.546, yy, 1.195), .027,
            .007, m['glass'], 'X')
    cylinder(stage, path+'/AntennaStem', (-.40, .26, .96), .009, .63,
        m['silver'])
    cylinder(stage, path+'/AntennaDish', (-.40, .26, 1.285), .12, .025,
        m['white'], rot=(0, 28, 0))
    ring(stage, path+'/DishEdge', (-.40, .26, 1.285), .117, .006,
        m['silver'])
    box(stage, path+'/FrontEquipment', (.565, 0, .425), (.10, .39, .18),
        m['gold'], .018, collision=True)
    for i, yy in enumerate([-.22, .22]):
        cylinder(stage, path+f'/NavigationEye{i}', (.548, yy, .49),
            .030, .025, m['glass'], 'X')
    return _metadata(stage, path, '小型探月车', 'lunar_rover', contacts, .66)


def build_lunar_base(stage, path):
    """Build one compact habitat with an airlock, support feet and solar wings."""
    _fresh_root(stage, path)
    m = _palette(stage)
    cylinder(stage, path+'/HabitatShell', (0, 0, 1.24), .82, 2.26,
        m['white'], 'X', collision=True)
    for i, xx in enumerate([-1.13, 1.13]):
        _sphere(stage, path+f'/RoundedEnd{i}', (xx, 0, 1.24),
            (.25, .817, .817), m['white'], collision=True)
    # Metal hoop frames break the white module into convincing pressure-shell
    # segments while leaving the silhouette legible from a distant viewpoint.
    for i, xx in enumerate([-.91, -.46, 0., .46, .91]):
        ring(stage, path+f'/ShellBand{i}', (xx, 0, 1.24), .825, .018,
            m['offwhite'], 'X')
    contacts = []
    for i, xx in enumerate([-.82, .82]):
        for side, sign in enumerate([-1, 1]):
            yy = sign*.63
            cylinder(stage, path+f'/LandingFoot{i}_{side}', (xx, yy, .055),
                .19, .11, m['silver'], collision=True)
            cylinder(stage, path+f'/LandingLeg{i}_{side}', (xx, yy, .325),
                .055, .47, m['offwhite'], collision=True)
            box(stage, path+f'/LegCollar{i}_{side}', (xx, yy, .47),
                (.21, .18, .11), m['silver'], .015)
            contacts.append([xx, yy, 0.])
    cylinder(stage, path+'/AirlockDrum', (1.37, 0, 1.04), .55, .48,
        m['white'], 'X', collision=True)
    cylinder(stage, path+'/DoorInset', (1.619, 0, 1.04), .46, .025,
        m['blue'], 'X', collision=True)
    ring(stage, path+'/AirlockSeal', (1.637, 0, 1.04), .474, .027,
        m['silver'], 'X')
    cylinder(stage, path+'/DoorWindow', (1.641, 0, 1.20), .145, .014,
        m['glass'], 'X')
    ring(stage, path+'/DoorWindowFrame', (1.652, 0, 1.20), .153, .018,
        m['silver'], 'X')
    box(stage, path+'/DoorHandle', (1.658, -.26, 1.0), (.045, .032, .18),
        m['silver'], .012)
    for i, (xx, zz, length) in enumerate([(1.82, .085, .46), (1.70, .245, .36), (1.57, .405, .26)]):
        box(stage, path+f'/Step{i}', (xx, 0, zz), (length, .78, .15),
            m['silver'], .012, collision=True)
        for j, yy in enumerate(np.linspace(-.32, .32, 8)):
            box(stage, path+f'/StepGrating{i}_{j}', (xx, float(yy), zz+.077),
                (length-.025, .009, .005), m['offwhite'], .001)
    for side, sign in enumerate([-1, 1]):
        # Side windows face +/-Y, with shallow external pressure frames.
        for i, xx in enumerate([-.56, .38]):
            cylinder(stage, path+f'/WindowFrame{side}_{i}', (xx, sign*.788, 1.46),
                .205, .07, m['silver'], 'Y')
            cylinder(stage, path+f'/WindowGlass{side}_{i}', (xx, sign*.829, 1.46),
                .171, .016, m['glass'], 'Y')
        cylinder(stage, path+f'/SolarSupport{side}', (-.28, sign*1.03, .80),
            .038, .77, m['silver'], 'Y', collision=True)
        _panel(stage, path+f'/SolarWing{side}', (-.28, sign*1.28, .89),
            (1.62, .68), m, rotation=(sign*8., 0., 0.))
    box(stage, path+'/RoofServiceBox', (-.58, 0, 2.025), (.59, .52, .15),
        m['white'], .025, collision=True)
    for i in range(8):
        box(stage, path+f'/RoofRadiator{i}', (-.82+i*.065, 0, 2.105),
            (.025, .44, .012), m['silver'], .003)
    cylinder(stage, path+'/CommunicationStem', (.23, .24, 2.235), .015,
        .35, m['silver'])
    cylinder(stage, path+'/CommunicationDish', (.23, .24, 2.425), .17,
        .025, m['white'], rot=(0, -26, 0))
    box(stage, path+'/ExternalBattery', (-.61, -.02, .29), (.58, .47, .27),
        m['gold'], .022, collision=True)
    return _metadata(stage, path, '小型月球基地', 'lunar_base', contacts, 1.12)
