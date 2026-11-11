"""Original, relocatable search props: an astronaut and a supported Chinese flag.

Each builder authors below a caller-owned placement root at local ground Z=0.
These are static scene props, not a human articulation or a cloth simulation.
No external models, raster images, text boards, arrows or signs are used.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .props import box, cylinder, material, ring, tube, xform


def _root(stage, path):
    if stage.GetPrimAtPath(path):
        raise ValueError('Refusing to overwrite an existing prop: '+str(path))
    root = xform(stage, path)
    root.GetPrim().CreateAttribute('cave:asset_revision', Sdf.ValueTypeNames.Int).Set(8)
    return root


def _sphere(stage, path, center, radii, mat, collision=False):
    transform = xform(stage, path, center)
    transform.AddScaleOp().Set(Gf.Vec3d(*map(float, radii)))
    sphere = UsdGeom.Sphere.Define(stage, path+'/Shape')
    sphere.CreateRadiusAttr(1.)
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr(True)
    return transform


def _metadata(stage, path, category, label, collision_scope):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    bounds = cache.ComputeLocalBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    low, high = np.asarray(bounds.GetMin(), float), np.asarray(bounds.GetMax(), float)
    if not np.isfinite(np.r_[low, high]).all() or np.any(high <= low):
        raise ValueError('Prop bounds are empty or invalid: '+path)
    xy_extent = np.maximum(np.abs(low[:2]), np.abs(high[:2]))
    colliders = [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(path))
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders:
        raise ValueError('Prop must contain static collision shapes: '+path)
    return dict(prim_path=path, class_name=category, label_zh=label,
        local_ground_footprint_radius_m=float(np.linalg.norm(xy_extent)),
        local_bounds_m=[low.tolist(), high.tolist()], height_m=float(high[2]),
        collision_scope=collision_scope, static_collision=True, collider_count=len(colliders),
        collider_paths=colliders, articulation=False, external_assets_used=False,
        provenance='locally authored procedural geometry for the cave search demo')


def build_astronaut(stage, path='/World/SearchObjects/Astronaut'):
    """Build an approximately 1.8m white-suited astronaut facing local +X."""
    root = _root(stage, path)
    mats = dict(
        suit=material(stage, 'v8_astronaut_white_fabric', (.78, .80, .78), .83),
        seam=material(stage, 'v8_astronaut_seams', (.40, .43, .43), .82),
        dark=material(stage, 'v8_astronaut_joint_rubber', (.035, .043, .048), .85),
        red=material(stage, 'v8_astronaut_red_cuff', (.52, .018, .013), .68),
        silver=material(stage, 'v8_astronaut_brushed_metal', (.46, .49, .51), .34, .72),
        gold=material(stage, 'v8_astronaut_gold_visor', (.67, .36, .065), .16, .88),
        glass=material(stage, 'v8_astronaut_visor_rim', (.012, .022, .027), .18, .43),
        control=material(stage, 'v8_astronaut_control_panel', (.16, .19, .20), .59),
        blue=material(stage, 'v8_astronaut_cable_connector', (.035, .11, .30), .45))

    # Collision geometry follows the visible torso, backpack, boots and limbs.
    # It is intentionally static, rather than an unvalidated human articulation.
    for index, side in enumerate((-1, 1)):
        yy = side*.14
        box(stage, path+f'/Boot{index}', (.055, yy, .105), (.34, .215, .21),
            mats['suit'], .055, collision=True)
        box(stage, path+f'/Sole{index}', (.065, yy, .028), (.35, .224, .056),
            mats['dark'], .016, collision=True)
        for ridge in range(4):
            box(stage, path+f'/BootRidge{index}_{ridge}', (.18, yy, .065+ridge*.025),
                (.026, .19, .011), mats['seam'], .003)
        box(stage, path+f'/LowerLeg{index}', (0., yy, .395), (.20, .205, .40),
            mats['suit'], .065, collision=True)
        cylinder(stage, path+f'/KneeJoint{index}', (0., yy, .60), .112, .10,
            mats['seam'])
        box(stage, path+f'/KneePad{index}', (.116, yy, .60), (.045, .152, .14),
            mats['suit'], .022)
        box(stage, path+f'/UpperLeg{index}', (-.015, yy, .795), (.23, .228, .34),
            mats['suit'], .074, collision=True)
        cylinder(stage, path+f'/LegBand{index}', (0., yy, .335), .105, .036,
            mats['red'])
        for seam in range(3):
            cylinder(stage, path+f'/AnkleFold{index}_{seam}', (0., yy, .235+seam*.035),
                .104, .010, mats['seam'])

    box(stage, path+'/Hip', (-.015, 0., .97), (.34, .48, .20), mats['suit'], .066, collision=True)
    box(stage, path+'/Torso', (0., 0., 1.18), (.39, .52, .47), mats['suit'], .095, collision=True)
    box(stage, path+'/Belt', (.0, 0., 1.004), (.405, .525, .045), mats['seam'], .014)
    box(stage, path+'/ChestControl', (.221, 0., 1.23), (.075, .29, .20), mats['control'], .025)
    box(stage, path+'/ChestFace', (.263, 0., 1.25), (.012, .248, .12), mats['suit'], .008)
    for index, yy in enumerate((-.08, 0., .08)):
        cylinder(stage, path+f'/ChestDial{index}', (.275, yy, 1.245), .018, .014,
            mats['dark'], 'X')
    box(stage, path+'/ChestRedBand', (.274, 0., 1.292), (.014, .214, .014), mats['red'], .003)
    for index, side in enumerate((-1, 1)):
        yy = side*.323
        _sphere(stage, path+f'/Shoulder{index}', (0., yy, 1.33), (.16, .146, .15), mats['suit'])
        box(stage, path+f'/UpperArm{index}', (.005, side*.365, 1.15), (.175, .174, .30),
            mats['suit'], .060, collision=True, rot=(side*9., 0., 0.))
        cylinder(stage, path+f'/Elbow{index}', (.023, side*.388, 1.005), .095, .072,
            mats['seam'])
        box(stage, path+f'/Forearm{index}', (.055, side*.408, .925), (.17, .17, .19),
            mats['suit'], .055, collision=True)
        cylinder(stage, path+f'/WristBand{index}', (.055, side*.412, .837), .088, .045,
            mats['red'])
        _sphere(stage, path+f'/Glove{index}', (.065, side*.412, .758), (.11, .087, .11), mats['suit'])
        box(stage, path+f'/GlovePalm{index}', (.14, side*.412, .75), (.055, .118, .13), mats['dark'], .025)
        cylinder(stage, path+f'/ChestConnector{index}', (.237, side*.174, 1.08), .032, .04,
            mats['blue'], 'X')
        tube(stage, path+f'/ChestHose{index}', [(.262, side*.174, 1.08), (.31, side*.22, 1.02),
            (.27, side*.27, .94), (.13, side*.25, .94)], .028, mats['seam'])

    box(stage, path+'/Backpack', (-.285, 0., 1.19), (.23, .43, .51), mats['suit'], .04, collision=True)
    box(stage, path+'/BackpackRear', (-.41, 0., 1.20), (.023, .38, .40), mats['seam'], .012)
    box(stage, path+'/BackpackTop', (-.285, 0., 1.46), (.21, .39, .04), mats['silver'], .009)
    for index, side in enumerate((-1, 1)):
        cylinder(stage, path+f'/OxygenCylinder{index}', (-.426, side*.13, 1.20),
            .059, .34, mats['silver'])
        cylinder(stage, path+f'/OxygenCap{index}', (-.426, side*.13, 1.38),
            .038, .038, mats['dark'])
    cylinder(stage, path+'/NeckSeal', (0., 0., 1.395), .176, .085, mats['dark'])
    cylinder(stage, path+'/NeckRing', (0., 0., 1.421), .192, .034, mats['silver'])
    _sphere(stage, path+'/HelmetShell', (.005, 0., 1.56), (.248, .255, .246), mats['suit'])
    _sphere(stage, path+'/VisorGasket', (.154, 0., 1.575), (.126, .224, .191), mats['glass'])
    _sphere(stage, path+'/GoldVisor', (.183, 0., 1.576), (.110, .204, .165), mats['gold'])
    box(stage, path+'/HelmetTopPlate', (.015, 0., 1.796), (.11, .13, .018), mats['suit'], .008)
    for index, side in enumerate((-1, 1)):
        cylinder(stage, path+f'/HelmetSideMount{index}', (.02, side*.252, 1.57),
            .046, .026, mats['silver'], 'Y')
    root.GetPrim().CreateAttribute('cave:simulation_scope', Sdf.ValueTypeNames.String).Set(
        'static astronaut search object, simplified limb and torso collision; no articulation')
    result = _metadata(stage, path, 'astronaut', '宇航员',
        'static torso, backpack, hip, boots and limb collision shapes; helmet and soft accessories are visual')
    result.update(preferred_view_direction_local_xy=[1., 0.], camera_target_height_m=1.15)
    return result


def _fabric_y(u):
    # A shallow, static material crease: the rigid crossbar supports the flag.
    return .012*math.sin(4.*math.pi*float(u))*(.2+.8*float(u))


def _flag_panel(stage, path, mat):
    nx, nz = 48, 28
    vertices = []
    for i in range(nx+1):
        uu = i/nx
        for j in range(nz+1):
            vv = j/nz
            vertices.append([.045+.9*uu, _fabric_y(uu), 2.0-.6*vv])
    faces = []
    for i in range(nx):
        for j in range(nz):
            aa = i*(nz+1)+j
            bb = (i+1)*(nz+1)+j
            faces.extend([[aa, bb, bb+1], [aa, bb+1, aa+1]])
    panel = UsdGeom.Mesh.Define(stage, path)
    panel.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, np.float32)))
    panel.CreateFaceVertexCountsAttr([3]*len(faces))
    panel.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).ravel()))
    panel.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    panel.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(panel.GetPrim()).Bind(mat)
    return panel


def _star(stage, path, center_uv, radius_m, direction, mat):
    # Five-pointed concave outline, triangulated around its centre, on both
    # sides of the same flag. The reverse naturally mirrors the physical flag.
    cx = .045+.9*float(center_uv[0])
    cz = 2.0-.6*float(center_uv[1])
    inner_ratio = math.sin(math.pi/10.)/math.sin(3.*math.pi/10.)
    coords = [[cx, cz]]
    for index in range(10):
        angle = direction+index*math.pi/5.
        radius = radius_m*(1. if index % 2 == 0 else inner_ratio)
        coords.append([cx+radius*math.cos(angle), cz+radius*math.sin(angle)])
    vertices, faces = [], []
    subdivisions = 6
    for side in (-1, 1):
        for index in range(10):
            # Subdivide each wedge so the yellow material follows the same
            # shallow crease as the red panel, avoiding buried star centres.
            aa, bb, cc = np.asarray([coords[0], coords[1+index], coords[1+(index+1) % 10]], float)
            ids = {}
            for ii in range(subdivisions+1):
                for jj in range(subdivisions+1-ii):
                    xx, zz = aa+(bb-aa)*(ii/subdivisions)+(cc-aa)*(jj/subdivisions)
                    uu = (xx-.045)/.9
                    ids[ii, jj] = len(vertices)
                    vertices.append([xx, _fabric_y(uu)+side*.0015, zz])
            for ii in range(subdivisions):
                for jj in range(subdivisions-ii):
                    triangles = [[ids[ii, jj], ids[ii+1, jj], ids[ii, jj+1]]]
                    if ii+jj+2 <= subdivisions:
                        triangles.append([ids[ii+1, jj], ids[ii+1, jj+1], ids[ii, jj+1]])
                    # XZ coordinates wind with normal -Y; flip for front (+Y).
                    faces.extend([tri[::-1] if side > 0 else tri for tri in triangles])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def build_chinese_flag(stage, path='/World/SearchObjects/ChineseFlag'):
    """Build a red, five-star Chinese flag with a pole and rigid top support."""
    root = _root(stage, path)
    red = material(stage, 'v8_flag_red_fabric', (.66, .012, .016), .91)
    yellow = material(stage, 'v8_flag_yellow_stars', (.95, .65, .018), .78)
    aluminium = material(stage, 'v8_flag_brushed_aluminium', (.49, .52, .54), .36, .78)
    base_mat = material(stage, 'v8_flag_dark_base', (.085, .10, .11), .86)
    cylinder(stage, path+'/WeightedBase', (0., 0., .04), .21, .08, base_mat, collision=True)
    cylinder(stage, path+'/PoleSocket', (0., 0., .12), .067, .16, aluminium, collision=True)
    cylinder(stage, path+'/Pole', (0., 0., 1.075), .021, 2.05, aluminium, collision=True)
    _sphere(stage, path+'/Finial', (0., 0., 2.113), (.033, .033, .033), aluminium)
    cylinder(stage, path+'/RigidTopSupport', (.48, 0., 2.008), .010, .93, aluminium, 'X', collision=True)
    _flag_panel(stage, path+'/RedFabric', red)
    # 30-by-20 construction grid: one large star and four small stars. Small
    # stars point one outer vertex towards the centre of the large star.
    large = (5./30., 5./20.)
    _star(stage, path+'/LargeStar', large, .09, math.pi/2., yellow)
    big_x, big_z = .9*large[0], -.6*large[1]
    for index, (grid_x, grid_y) in enumerate(((10, 2), (12, 4), (12, 7), (10, 9))):
        point = (grid_x/30., grid_y/20.)
        angle = math.atan2(big_z-(-.6*point[1]), big_x-.9*point[0])
        _star(stage, path+f'/SmallStar{index}', point, .03, angle, yellow)
    for index, zz in enumerate((1.43, 1.97)):
        ring(stage, path+f'/AttachmentRing{index}', (0., 0., zz), .025, .004, aluminium)
    root.GetPrim().CreateAttribute('cave:simulation_scope', Sdf.ValueTypeNames.String).Set(
        'static supported flag landmark and pole/base obstacle; no wind or cloth dynamics')
    result = _metadata(stage, path, 'chinese_flag', '中国国旗',
        'static pole, weighted base, socket and rigid top support; fabric and stars are visual')
    result.update(flag_dimensions_m=[.9, .6], star_count=5, rigid_top_support=True,
        cloth_simulation=False, wind_simulation=False,
        preferred_view_direction_local_xy=[0., -1.], camera_target_height_m=1.62)
    return result
