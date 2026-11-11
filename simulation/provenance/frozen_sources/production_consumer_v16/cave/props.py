"""Original detailed search objects and a visual exploration rover (metres, Z up).

Import after Isaac Sim's SimulationApp. All geometry is self-authored; no remote
assets are referenced. The rover is a visual kinematic shell, not an articulated
or validated vehicle dynamics model. Target meshes have static collision shapes.
"""
from __future__ import annotations

import math
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, UsdPhysics, Vt


def material(stage, name, color, roughness=.5, metallic=0., emission=None):
    path = '/World/Looks/Prop_' + name
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + '/Surface')
    s.CreateIdAttr('UsdPreviewSurface')
    s.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    s.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    s.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    if emission is not None:
        s.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), 'surface')
    return m


def xform(stage, path, pos=(0.,0.,0.), rot=(0.,0.,0.)):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float,pos)))
    if any(rot):
        x.AddRotateXYZOp().Set(Gf.Vec3f(*map(float,rot)))
    return x


def _mesh(stage, path, vertices, faces, mat, normals=None, collision=False):
    m = UsdGeom.Mesh.Define(stage, path)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    m.CreateFaceVertexCountsAttr([faces.shape[1]] * len(faces))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    m.CreateSubdivisionSchemeAttr('none')
    if normals is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals,dtype=np.float32)))
        m.SetNormalsInterpolation('vertex')
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
    return m


def box(stage,path,pos,size,mat,bevel=.018,steps=7,collision=False,rot=None):
    """Rounded cuboid with explicit surface normals, not a scaled unit cube."""
    root = xform(stage,path,pos,rot or (0.,0.,0.))
    h = np.asarray(size,dtype=float) / 2
    r = min(float(bevel),float(h.min())*.8)
    verts=[]; normals=[]; faces=[]
    # Nonuniform grid resolves the bevel arcs without tessellating flat interiors.
    grids=[]
    for hh in h:
        if r > 1e-6:
            edge = np.linspace(0,math.pi/2,max(3,steps//2+1))
            vals = np.r_[-hh+r-r*np.cos(edge), hh-r+r*np.sin(edge)]
            grids.append(np.unique(vals))
        else:
            grids.append(np.array([-hh,hh]))
    for axis in range(3):
        a,b = [k for k in range(3) if k != axis]
        for sign in [-1,1]:
            start=len(verts); na=len(grids[a]); nb=len(grids[b])
            for u in grids[a]:
                for v in grids[b]:
                    q=np.zeros(3); q[axis]=sign*h[axis]; q[a]=u; q[b]=v
                    inner=np.clip(q,-h+r,h-r)
                    d=q-inner; length=np.linalg.norm(d)
                    n=d/length if length > 1e-9 else np.eye(3)[axis]*sign
                    p=inner+n*r if r > 1e-6 else q
                    verts.append(p); normals.append(n)
            for i in range(na-1):
                for j in range(nb-1):
                    quad=[start+i*nb+j,start+(i+1)*nb+j,start+(i+1)*nb+j+1,start+i*nb+j+1]
                    p=np.asarray([verts[k] for k in quad])
                    if np.cross(p[1]-p[0],p[2]-p[0])[axis]*sign < 0:
                        quad=quad[::-1]
                    faces.extend([[quad[0],quad[1],quad[2]],[quad[0],quad[2],quad[3]]])
    _mesh(stage,path+'/Mesh',verts,faces,mat,normals,collision)
    return root


def cylinder(stage,path,pos,radius,height,mat,axis='Z',rot=None,collision=False):
    root=xform(stage,path,pos,rot or (0.,0.,0.))
    c=UsdGeom.Cylinder.Define(stage,path+'/Shape')
    c.CreateRadiusAttr(float(radius)); c.CreateHeightAttr(float(height)); c.CreateAxisAttr(axis)
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return root


def ring(stage,path,pos,radius,tube,mat,axis='Z',segments=48,sides=10):
    verts=[]; faces=[]
    for i in range(segments):
        u=2*math.pi*i/segments
        for j in range(sides):
            v=2*math.pi*j/sides
            p=np.array([(radius+tube*math.cos(v))*math.cos(u),
                        (radius+tube*math.cos(v))*math.sin(u),tube*math.sin(v)])
            if axis=='Y': p=p[[0,2,1]]
            if axis=='X': p=p[[2,0,1]]
            verts.append(p+np.asarray(pos))
    for i in range(segments):
        for j in range(sides):
            a=i*sides+j; b=((i+1)%segments)*sides+j
            c=((i+1)%segments)*sides+(j+1)%sides; d=i*sides+(j+1)%sides
            faces.extend([[a,b,c],[a,c,d]])
    _mesh(stage,path,verts,faces,mat)


def tube(stage,path,points,width,mat):
    c=UsdGeom.BasisCurves.Define(stage,path)
    c.CreateTypeAttr('linear'); c.CreateCurveVertexCountsAttr([len(points)])
    c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points,dtype=np.float32)))
    c.CreateWidthsAttr([float(width)]); c.SetWidthsInterpolation('constant')
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    return c


def _palette(stage):
    return {
        'red':material(stage,'rescue_red',(.48,.018,.012),.38),
        'red_dark':material(stage,'red_seam',(.15,.008,.006),.54),
        'orange':material(stage,'woven_orange',(.68,.17,.028),.82),
        'orange_dark':material(stage,'woven_orange_shadow',(.31,.065,.011),.86),
        'yellow':material(stage,'safety_yellow',(.77,.51,.025),.4),
        'blue':material(stage,'drill_blue',(.018,.105,.34),.36),
        'rubber':material(stage,'rubber',(.012,.016,.019),.8),
        'cloth':material(stage,'webbing',(.031,.043,.039),.93),
        'white':material(stage,'warm_white',(.83,.85,.77),.48),
        'silver':material(stage,'brushed_aluminium',(.58,.62,.64),.28,.88),
        'steel':material(stage,'steel',(.22,.25,.27),.29,.92),
        'chrome':material(stage,'chrome',(.65,.7,.72),.18,1.),
        'glass':material(stage,'sensor_glass',(.012,.036,.048),.08,.48),
        'rover':material(stage,'rover_sand',(.28,.32,.27),.42,.23),
        'light':material(stage,'lamp_lens',(.8,.82,.73),.1,0.,(3.,3.,2.5)),
        'led':material(stage,'status_led',(.02,.32,.2),.2,0.,(.05,2.,.7)),
    }


def _aid(stage,p,m):
    box(stage,p+'/Body',(0,0,.255),(.23,.58,.42),m['red'],.035,collision=True)
    box(stage,p+'/Gasket',(-.078,0,.255),(.026,.584,.414),m['red_dark'],.014)
    box(stage,p+'/Lid',(-.112,0,.255),(.055,.553,.389),m['red'],.026)
    # Raised universal rescue cross on the front of the case.
    box(stage,p+'/CrossVertical',(-.144,0,.264),(.007,.066,.23),m['white'],.003)
    box(stage,p+'/CrossHorizontal',(-.145,0,.264),(.007,.23,.066),m['white'],.003)
    for i,y in enumerate([-.215,.215]):
        box(stage,p+f'/LatchBase{i}',(-.14,y,.414),(.022,.06,.075),m['rubber'],.007)
        box(stage,p+f'/LatchSteel{i}',(-.156,y,.417),(.009,.043,.047),m['silver'],.004)
        cylinder(stage,p+f'/LatchPin{i}',(-.166,y,.411),.006,.043,m['chrome'],'Y')
        box(stage,p+f'/RearHinge{i}',(.12,y,.28),(.026,.064,.06),m['steel'],.007)
    tube(stage,p+'/Handle',[(-.005,-.11,.466),(-.005,-.11,.52),(-.005,-.085,.543),
                           (-.005,.085,.543),(-.005,.11,.52),(-.005,.11,.466)],.026,m['rubber'])
    for i,y in enumerate([-.245,.245]):
        box(stage,p+f'/Foot{i}',(.01,y,.035),(.18,.06,.045),m['rubber'],.012)
    for i,z in enumerate([.105,.15,.36]):
        for j,s in enumerate([-1,1]):
            box(stage,p+f'/MoldedRib{i}_{j}',(.0,s*.29,z),(.16,.012,.013),m['red_dark'],.005)


def _backpack(stage,p,m):
    box(stage,p+'/Body',(.015,0,.32),(.29,.42,.58),m['orange'],.095,steps=10,collision=True)
    box(stage,p+'/FrontPocket',(-.16,0,.26),(.10,.35,.32),m['orange_dark'],.04)
    box(stage,p+'/PocketFace',(-.215,0,.26),(.016,.31,.25),m['orange'],.012)
    # Front zipper, a broad curved lid seam, and two horizontal webbing rows.
    tube(stage,p+'/LidSeam',[(-.112,-.16,.52),(-.139,-.1,.565),(-.145,0,.58),
                             (-.139,.1,.565),(-.112,.16,.52)],.012,m['rubber'])
    tube(stage,p+'/PocketZip',[(-.224,-.13,.365),(-.228,0,.38),(-.224,.13,.365)],.009,m['rubber'])
    for i,z in enumerate([.2,.29]):
        box(stage,p+f'/Webbing{i}',(-.23,0,z),(.012,.33,.022),m['cloth'],.003)
        for j,y in enumerate([-.10,0,.10]):
            box(stage,p+f'/Stitch{i}_{j}',(-.237,y,z),(.004,.004,.028),m['orange_dark'],.001)
    for i,y in enumerate([-.12,.12]):
        tube(stage,p+f'/ShoulderStrap{i}',[(.09,y,.57),(.22,y,.48),(.26,y,.27),
                                         (.20,y,.12),(.10,y,.10)],.052,m['cloth'])
        box(stage,p+f'/CompressionStrap{i}',(-.153,y,.45),(.022,.039,.12),m['cloth'],.006)
        box(stage,p+f'/Buckle{i}',(-.168,y,.435),(.022,.052,.048),m['rubber'],.008)
        box(stage,p+f'/BuckleInset{i}',(-.181,y,.435),(.005,.025,.024),m['orange_dark'],.002)
    tube(stage,p+'/TopHandle',[(.02,-.065,.59),(.02,-.065,.66),(.02,.065,.66),(.02,.065,.59)],.022,m['cloth'])
    for i,s in enumerate([-1,1]):
        box(stage,p+f'/SidePocket{i}',(.015,s*.22,.2),(.18,.075,.23),m['cloth'],.028)
        for j in range(4):
            tube(stage,p+f'/PocketMesh{i}_{j}',[(-.06,s*.258,.11+j*.04),(.09,s*.258,.13+j*.04)],.004,m['orange_dark'])
    cylinder(stage,p+'/ZipperPull',(-.228,.13,.355),.012,.018,m['silver'],'X')


def _canteen(stage,p,m):
    cylinder(stage,p+'/Bottle',(0,0,.19),.097,.28,m['silver'],collision=True)
    cylinder(stage,p+'/Base',(0,0,.052),.099,.035,m['rubber'])
    cylinder(stage,p+'/Shoulder',(0,0,.337),.086,.035,m['silver'])
    cylinder(stage,p+'/Neck',(0,0,.363),.052,.045,m['silver'])
    cylinder(stage,p+'/Cap',(0,0,.392),.063,.045,m['rubber'])
    for j in range(5):
        ring(stage,p+f'/CapGrip{j}',(0,0,.376+j*.008),.063,.0025,m['cloth'])
    ring(stage,p+'/ShoulderRing',(0,0,.315),.097,.004,m['chrome'])
    ring(stage,p+'/BottomRing',(0,0,.072),.097,.004,m['chrome'])
    tube(stage,p+'/LidKeeper',[(0,.052,.4),(.0,.12,.43),(.0,.15,.39),(.0,.12,.34),(.0,.075,.32)],.012,m['orange'])
    box(stage,p+'/Label',(-.097,0,.215),(.005,.09,.11),m['blue'],.004)
    box(stage,p+'/LabelStripe',(-.1005,0,.218),(.003,.07,.011),m['white'],.002)


def _toolbox(stage,p,m):
    box(stage,p+'/Base',(0,0,.09),(.35,.61,.16),m['rubber'],.028,collision=True)
    box(stage,p+'/Shell',(0,0,.2),(.35,.60,.16),m['yellow'],.024,collision=True)
    box(stage,p+'/Seam',(0,0,.193),(.354,.607,.016),m['rubber'],.006)
    box(stage,p+'/Lid',(0,0,.287),(.35,.60,.055),m['yellow'],.021)
    for i,y in enumerate([-.22,-.11,0,.11,.22]):
        box(stage,p+f'/LidRib{i}',(0,y,.319),(.27,.023,.012),m['yellow'],.006)
    for i,y in enumerate([-.22,.22]):
        box(stage,p+f'/LatchMount{i}',(-.178,y,.203),(.022,.072,.09),m['rubber'],.008)
        box(stage,p+f'/Latch{i}',(-.194,y,.203),(.014,.048,.063),m['silver'],.006)
        box(stage,p+f'/Hinge{i}',(.181,y,.21),(.022,.085,.045),m['steel'],.005)
    tube(stage,p+'/Handle',[(-.02,-.10,.33),(-.02,-.10,.40),(-.02,.10,.40),(-.02,.10,.33)],.025,m['rubber'])
    for i,x in enumerate([-.15,.15]):
        for j,y in enumerate([-.278,.278]):
            box(stage,p+f'/Corner{i}_{j}',(x,y,.1),(.056,.067,.14),m['rubber'],.014)
    box(stage,p+'/Badge',(-.184,0,.24),(.009,.12,.038),m['rubber'],.005)
    box(stage,p+'/BadgeLine',(-.190,0,.24),(.003,.078,.008),m['yellow'],.001)


def _drill(stage,p,m):
    # Drill stands on its battery; its motor/chuck points toward upstream -X.
    box(stage,p+'/Battery',(0.025,0,.067),(.22,.16,.10),m['rubber'],.018,collision=True)
    box(stage,p+'/BatterySide',(-.01,-.082,.068),(.16,.012,.058),m['blue'],.008)
    box(stage,p+'/Handle',(.055,0,.192),(.083,.085,.20),m['rubber'],.023,rot=(0,-12,0),collision=True)
    box(stage,p+'/HandleGrip',(.08,-.047,.187),(.055,.009,.135),m['blue'],.008,rot=(0,-12,0))
    box(stage,p+'/Motor',(-.01,0,.33),(.235,.13,.14),m['blue'],.038,collision=True)
    cylinder(stage,p+'/MotorRear',(.115,0,.33),.055,.023,m['rubber'],'X')
    cylinder(stage,p+'/Clutch',(-.14,0,.33),.058,.046,m['rubber'],'X')
    cylinder(stage,p+'/Chuck',(-.185,0,.33),.039,.065,m['steel'],'X')
    cylinder(stage,p+'/ChuckRim',(-.217,0,.33),.034,.012,m['chrome'],'X')
    cylinder(stage,p+'/Bit',(-.27,0,.33),.005,.10,m['steel'],'X')
    for i in range(6):
        angle=2*math.pi*i/6
        y,z=.061*math.sin(angle),.33+.061*math.cos(angle)
        cylinder(stage,p+f'/ClutchRib{i}',(-.139,y,z),.004,.037,m['blue'],'X')
    for i in range(4):
        box(stage,p+f'/Vent{i}',(.078,-.066,.30+i*.017),(.036,.006,.006),m['rubber'],.002)
    box(stage,p+'/Trigger',(-.003,0,.245),(.039,.07,.039),m['rubber'],.008)
    box(stage,p+'/TorqueBadge',(-.04,-.069,.337),(.065,.007,.035),m['white'],.005)
    cylinder(stage,p+'/Worklight',(-.09,0,.248),.013,.009,m['light'],'X')
    box(stage,p+'/BatteryRelease',(-.07,-.087,.09),(.036,.018,.024),m['yellow'],.004)


def _label(prim,category,identifier):
    prim.CreateAttribute('cave:category',Sdf.ValueTypeNames.String).Set(category)
    prim.CreateAttribute('cave:object_id',Sdf.ValueTypeNames.String).Set(identifier)
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim,category,'class')
    except ImportError:
        # The custom metadata remains inspectable in a plain USD runtime.
        pass


def build_props(stage,parent='/World/Targets'):
    """Build five targets and return language labels, locations and footprints."""
    UsdGeom.Xform.Define(stage,parent)
    m=_palette(stage)
    definitions=[
        ('first_aid_kit','急救箱',(21.,1.2,.05),.34,_aid,(-0.,0.,-12.)),
        ('backpack','橙色背包',(11.,-2.3,.05),.29,_backpack,(0.,0.,18.)),
        ('canteen','银色水壶',(16.,2.,.05),.13,_canteen,(0.,0.,0.)),
        ('toolbox','黄色工具箱',(26.,-1.8,.05),.36,_toolbox,(0.,0.,-14.)),
        ('drill','蓝色电钻',(7.,1.7,.05),.30,_drill,(0.,0.,-25.)),
    ]
    result=[]
    for category,label,pos,radius,fn,rotation in definitions:
        path=parent+'/'+category
        root=xform(stage,path,pos,rotation)
        _label(root.GetPrim(),category,category)
        fn(stage,path,m)
        result.append({'id':category,'prim_path':path,'category':category,
                       'label_zh':label,'position':list(pos),'radius':radius})
    return result


def build_rover(stage,path='/World/Rover'):
    """A detailed visual shell; root XY translation/yaw is owned by the caller.

    Root ground origin is Z=0, forward +X. Camera mounting height is .90m.
    There are no rigid bodies, joints or motor dynamics in this decorative shell.
    """
    root=xform(stage,path)
    m=_palette(stage)
    box(stage,path+'/Chassis',(0,0,.29),(.78,.53,.18),m['rover'],.05)
    box(stage,path+'/Belly',(0,0,.19),(.67,.45,.08),m['rubber'],.025)
    box(stage,path+'/TopDeck',(-.03,0,.395),(.66,.49,.05),m['steel'],.025)
    box(stage,path+'/Electronics',(-.08,0,.47),(.38,.35,.12),m['rover'],.025)
    for i,x in enumerate([-.26,.26]):
        cylinder(stage,path+f'/Axle{i}',(x,0,.18),.029,.69,m['steel'],'Y')
        for j,sign in enumerate([-1,1]):
            y=sign*.33
            cylinder(stage,path+f'/Wheel{i}_{j}',(x,y,.18),.18,.13,m['rubber'],'Y')
            cylinder(stage,path+f'/WheelHub{i}_{j}',(x,y+sign*.071,.18),.095,.015,m['silver'],'Y')
            cylinder(stage,path+f'/HubCap{i}_{j}',(x,y+sign*.081,.18),.037,.016,m['rover'],'Y')
            for k in range(18):
                angle=2*math.pi*k/18
                xx=x+.177*math.sin(angle); zz=.18+.177*math.cos(angle)
                box(stage,path+f'/Tread{i}_{j}_{k}',(xx,y,zz),(.048,.139,.025),m['rubber'],.003,
                    rot=(0,math.degrees(angle),0))
            for k in range(6):
                angle=2*math.pi*k/6
                cylinder(stage,path+f'/WheelBolt{i}_{j}_{k}',
                         (x+.064*math.sin(angle),y+sign*.085,.18+.064*math.cos(angle)),
                         .008,.009,m['steel'],'Y')
    for i,sign in enumerate([-1,1]):
        # Protective fender on each side, longitudinal bumper and stripe.
        box(stage,path+f'/Fender{i}',(0,sign*.285,.38),(.74,.082,.038),m['rover'],.017)
        box(stage,path+f'/Stripe{i}',(.0,sign*.268,.30),(.52,.009,.026),m['yellow'],.003)
    box(stage,path+'/FrontBumper',(.425,0,.28),(.075,.58,.065),m['rubber'],.022)
    box(stage,path+'/RearBumper',(-.43,0,.27),(.06,.55,.06),m['rubber'],.018)
    for i,y in enumerate([-.19,.19]):
        cylinder(stage,path+f'/HeadlampHousing{i}',(.397,y,.36),.056,.052,m['rubber'],'X')
        cylinder(stage,path+f'/HeadlampLens{i}',(.428,y,.36),.043,.008,m['light'],'X')
        ring(stage,path+f'/HeadlampRing{i}',(.431,y,.36),.044,.004,m['chrome'],'X')
    cylinder(stage,path+'/Mast',(0,0,.665),.028,.43,m['steel'])
    cylinder(stage,path+'/MastFoot',(0,0,.445),.08,.035,m['rubber'])
    box(stage,path+'/SensorHead',(.025,0,.89),(.16,.29,.095),m['rubber'],.022)
    for i,y in enumerate([-.095,.095]):
        cylinder(stage,path+f'/CameraHousing{i}',(.107,y,.89),.031,.035,m['steel'],'X')
        cylinder(stage,path+f'/CameraGlass{i}',(.126,y,.89),.024,.006,m['glass'],'X')
    cylinder(stage,path+'/LidarBase',(.02,0,.967),.069,.055,m['steel'])
    cylinder(stage,path+'/LidarBand',(.02,0,.973),.071,.024,m['glass'])
    cylinder(stage,path+'/LidarLid',(.02,0,1.005),.068,.013,m['rover'])
    cylinder(stage,path+'/Antenna',(-.30,.13,.71),.006,.50,m['rubber'])
    cylinder(stage,path+'/AntennaFoot',(-.30,.13,.46),.024,.04,m['steel'])
    box(stage,path+'/StatusLED',(.12,-.18,.48),(.034,.012,.014),m['led'],.004)
    # Exposed fasteners and radiator details make close shots legible.
    for i,x in enumerate([-.25,.23]):
        for j,y in enumerate([-.185,.185]):
            cylinder(stage,path+f'/DeckBolt{i}_{j}',(x,y,.425),.009,.008,m['silver'])
    for i in range(6):
        box(stage,path+f'/HeatSink{i}',(-.22+i*.045,0,.538),(.014,.21,.013),m['steel'],.003)
    tube(stage,path+'/SensorCable',[(0,.04,.46),(-.05,.05,.60),(-.05,.03,.78),(.0,.03,.86)],.013,m['rubber'])
    root.GetPrim().CreateAttribute('cave:simulation_scope',Sdf.ValueTypeNames.String).Set('visual kinematic rover shell; no wheel or motor dynamics')
    return {'prim_path':path,'camera_mount':[.15,0.,.90], 'forward_axis':'+X',
            'footprint_radius_m':.5,'simulation_scope':'kinematic visual shell'}
