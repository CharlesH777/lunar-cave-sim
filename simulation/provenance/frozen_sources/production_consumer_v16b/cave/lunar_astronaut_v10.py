"""Original high-detail static EVA search target, metres and Z-up.

The pressure garment uses continuously lofted, softly folded surfaces rather
than cuboid limbs. This is a self-authored visual asset, not a certified suit,
human articulation, or downloaded reconstruction of a specific mission.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .props import box, cylinder, material, ring, tube, xform


def _mesh(stage, path, points, faces, mat, collision=False, fabric=False):
    points = np.asarray(points, np.float64)
    faces = np.asarray(faces, np.int32)
    triangles = points[faces]
    weighted = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    if not np.isfinite(points).all() or np.any(np.linalg.norm(weighted, axis=1)<1e-13):
        raise ValueError('Invalid astronaut triangles: '+path)
    normals = np.zeros_like(points)
    for corner in range(3):np.add.at(normals, faces[:, corner], weighted)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    mesh.SetNormalsInterpolation('vertex')
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    if fabric:
        # Small, continuous dye variation remains subtle under bright work lamps.
        p=points;variation=.965+.022*np.sin(89*p[:,0]+17*p[:,2])*np.sin(73*p[:,1]-11*p[:,2])
        colors=variation[:,None]*np.array([.445,.455,.435])[None]
        primvar=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('suitColor',Sdf.ValueTypeNames.Color3fArray,UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    return mesh


def _fabric_material(stage):
    mat=material(stage,'V10_EVA_WovenBetaCloth',(.445,.455,.435),.89)
    shader=UsdShade.Shader.Get(stage,mat.GetPath().AppendChild('Surface'))
    reader=UsdShade.Shader.Define(stage,mat.GetPath().AppendChild('SuitColor'))
    reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('suitColor')
    reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
    shader.GetInput('diffuseColor').ConnectToSource(reader.ConnectableAPI(),'result')
    return mat


def _loft(stage,path,centres,radii,mat,*,rings=36,sides=48,fold=.003,seed=0,
          collision=False,fabric=False,seam_mat=None):
    """Closed cloth surface, framed around a gently bent centreline."""
    centres=np.asarray(centres,float);radii=np.asarray(radii,float)
    knots=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(centres,axis=0),axis=1))]
    knots/=knots[-1]
    s=np.linspace(0.,1.,rings)
    center=PchipInterpolator(knots,centres,axis=0)(s)
    widths=PchipInterpolator(knots,radii,axis=0)(s)
    tangent=np.gradient(center,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    forward=np.tile([1.,0.,0.],(rings,1))
    forward-=np.sum(forward*tangent,axis=1,keepdims=True)*tangent
    forward/=np.maximum(np.linalg.norm(forward,axis=1,keepdims=True),1e-20)
    lateral=np.cross(tangent,forward)
    theta=np.arange(sides)*2.*np.pi/sides
    rng=np.random.default_rng(seed)
    # Nonuniform fabric creases have finite locations, not regular body ribs.
    folds=[(float(rng.uniform(.1,.9)),float(rng.uniform(.025,.075)),
            float(rng.uniform(0,2*np.pi)),float(rng.uniform(.55,1.))) for _ in range(7)]
    perturb=np.zeros((rings,sides))
    for location,width,phase,gain in folds:
        perturb+=gain*np.exp(-((s[:,None]-location)/width)**2)*np.cos(2.*theta[None]+phase+.9*s[:,None])
    perturb+=.18*np.sin(7.*theta[None]+5.*s[:,None]+seed*.1)
    perturb*=fold*np.sin(np.pi*s[:,None])**.55
    a=widths[:,0,None]+perturb;b=widths[:,1,None]+perturb*.8
    points=center[:,None,:]+a[:,:,None]*np.cos(theta)[None,:,None]*forward[:,None,:]
    points+=b[:,:,None]*np.sin(theta)[None,:,None]*lateral[:,None,:]
    rows=points.copy();points=points.reshape(-1,3).tolist();faces=[]
    for i in range(rings-1):
        for j in range(sides):
            a0=i*sides+j;b0=i*sides+(j+1)%sides;c0=(i+1)*sides+(j+1)%sides;d0=(i+1)*sides+j
            faces.extend([[a0,b0,c0],[a0,c0,d0]])
    first=len(points);points.append(center[0].tolist())
    last=len(points);points.append(center[-1].tolist())
    for j in range(sides):
        faces.append([first,(j+1)%sides,j])
        faces.append([last,(rings-1)*sides+j,(rings-1)*sides+(j+1)%sides])
    mesh=_mesh(stage,path,points,faces,mat,collision,fabric)
    if seam_mat:
        for index,j in enumerate([sides//8,3*sides//8,5*sides//8,7*sides//8]):
            seam=rows[:,j]+.0012*(rows[:,j]-center)/np.linalg.norm(rows[:,j]-center,axis=1,keepdims=True)
            tube(stage,path+f'_BoundSeam{index}',seam.tolist(),.0025,seam_mat)
    return mesh,rows


def _ellipsoid(stage,path,center,radii,mat,*,rings=30,sides=64,collision=False):
    """Smooth closed ellipsoid with unique pole vertices."""
    center=np.asarray(center,float);radii=np.asarray(radii,float)
    points=[(center+[0.,0.,radii[2]]).tolist()];faces=[]
    for i in range(1,rings):
        phi=np.pi*i/rings
        for j in range(sides):
            theta=2*np.pi*j/sides
            points.append((center+radii*np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])).tolist())
    bottom=len(points);points.append((center-[0.,0.,radii[2]]).tolist())
    for j in range(sides):faces.append([0,1+j,1+(j+1)%sides])
    for i in range(rings-2):
        for j in range(sides):
            a=1+i*sides+j;b=1+i*sides+(j+1)%sides;c=1+(i+1)*sides+(j+1)%sides;d=1+(i+1)*sides+j
            faces.extend([[a,d,c],[a,c,b]])
    offset=1+(rings-2)*sides
    for j in range(sides):faces.append([bottom,offset+(j+1)%sides,offset+j])
    return _mesh(stage,path,points,faces,mat,collision)


def _visor_patch(stage,path,center,helmet_radii,half_y,half_z,offset,mat):
    """A curved, squared oval inset following the helmet, not a face globe."""
    center=np.asarray(center,float);rx,ry,rz=helmet_radii
    count=96;radial=18;theta=np.arange(count)*2*np.pi/count
    y=half_y*np.sign(np.cos(theta))*np.abs(np.cos(theta))**.68
    z=half_z*np.sign(np.sin(theta))*np.abs(np.sin(theta))**.78
    points=[(center+[rx+offset,0.,0.]).tolist()];faces=[]
    for i in range(1,radial+1):
        r=i/radial
        for yy,zz in zip(y*r,z*r):
            xx=rx*np.sqrt(max(1e-9,1.-(yy/ry)**2-(zz/rz)**2))+offset
            points.append((center+[xx,yy,zz]).tolist())
    for j in range(count):faces.append([0,1+j,1+(j+1)%count])
    for i in range(radial-1):
        for j in range(count):
            a=1+i*count+j;b=1+i*count+(j+1)%count;c=1+(i+1)*count+(j+1)%count;d=1+(i+1)*count+j
            faces.extend([[a,d,c],[a,c,b]])
    # All visible patch triangles must point forward (+X).
    p=np.asarray(points);f=np.asarray(faces)
    if np.mean(np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]])[:,0])<0:f=f[:,[0,2,1]]
    return _mesh(stage,path,p,f,mat),np.asarray(points[-count:])


def _closed_curve(stage,path,points,width,mat):
    points=np.asarray(points,float)
    return tube(stage,path,np.vstack([points,points[0]]).tolist(),width,mat)


def build_astronaut(stage,path='/World/SearchTargets/Astronaut'):
    """Construct a ~1.86m serious EVA-suit figure facing local +X."""
    if stage.GetPrimAtPath(path):raise ValueError('Refusing to replace an existing prop: '+path)
    root=xform(stage,path);root.GetPrim().CreateAttribute('cave:asset_revision',Sdf.ValueTypeNames.Int).Set(10)
    fabric=_fabric_material(stage)
    m=dict(fabric=fabric,
        ivory=material(stage,'V10_EVA_ReinforcedIvory',(.38,.39,.365),.82),
        stitch=material(stage,'V10_EVA_Stitching',(.285,.300,.283),.91),
        seal=material(stage,'V10_EVA_PressureJoint',(.055,.064,.063),.84),
        red=material(stage,'V10_EVA_RedIdentification',(.31,.018,.013),.79),
        hardware=material(stage,'V10_EVA_TitaniumHardware',(.31,.34,.35),.37,.78),
        rubber=material(stage,'V10_EVA_BootSole',(.025,.029,.028),.90),
        gasket=material(stage,'V10_EVA_VisorGasket',(.009,.015,.016),.35,.1),
        gold=material(stage,'V10_EVA_GoldSunVisor',(.38,.225,.052),.20,.92),
        lens=material(stage,'V10_EVA_Lens',(.008,.035,.045),.14,.35),
        panel=material(stage,'V10_EVA_ControlPanel',(.12,.14,.14),.60),
        connector=material(stage,'V10_EVA_BlueConnector',(.025,.075,.14),.56,.30))
    # Continuous garment, realistic adult proportions, restrained asymmetry.
    _loft(stage,path+'/PelvicGarment',[[0.,0.,.87],[.002,0.,.97],[-.005,0.,1.075]],
        [[.15,.225],[.18,.25],[.165,.22]],fabric,rings=26,sides=56,fold=.004,seed=10,
        fabric=True,collision=True,seam_mat=m['stitch'])
    _loft(stage,path+'/TorsoGarment',[[-.008,0.,1.02],[0.,0.,1.16],[0.,0.,1.32],[0.,0.,1.40],[0.,0.,1.465]],
        [[.16,.215],[.185,.245],[.194,.275],[.174,.26],[.135,.17]],fabric,
        rings=42,sides=64,fold=.0045,seed=11,fabric=True,collision=True,seam_mat=m['stitch'])
    for limb,sign in enumerate([-1,1]):
        y=sign*.137;x=.025 if sign>0 else -.015
        _loft(stage,path+f'/UpperLeg{limb}',[[x,y,.94],[x+.01,sign*.145,.78],[x+.025,sign*.151,.625]],
            [[.125,.125],[.123,.128],[.101,.104]],fabric,rings=34,sides=48,
            fold=.0047,seed=20+limb,collision=True,fabric=True,seam_mat=m['stitch'])
        _loft(stage,path+f'/KneeBellows{limb}',[[x+.025,sign*.151,.64],[x+.035,sign*.153,.575],[x+.035,sign*.152,.535]],
            [[.104,.106],[.11,.111],[.098,.10]],m['seal'],rings=18,sides=48,fold=.003,seed=22+limb)
        _loft(stage,path+f'/LowerLeg{limb}',[[x+.035,sign*.152,.555],[x+.015,sign*.15,.37],[x+.012,sign*.147,.195]],
            [[.104,.106],[.102,.109],[.088,.10]],fabric,rings=34,sides=48,
            fold=.0055,seed=24+limb,collision=True,fabric=True,seam_mat=m['stitch'])
        # Shaped boot vamp and sole; pressure shell sits inside the sewn overboot.
        box(stage,path+f'/BootSole{limb}',(x+.06,sign*.147,.027),(.335,.224,.054),m['rubber'],.023,collision=True)
        _loft(stage,path+f'/Overboot{limb}',[[x+.06,sign*.147,.063],[x+.07,sign*.147,.12],[x+.02,sign*.147,.205],[x+.012,sign*.147,.245]],
            [[.155,.108],[.151,.108],[.092,.102],[.089,.10]],fabric,rings=26,sides=48,fold=.002,
            seed=26+limb,fabric=True,collision=True,seam_mat=m['stitch'])
        for j in range(9):
            xx=x-.075+j*.034
            box(stage,path+f'/SoleTread{limb}_{j}',(xx,sign*.147,.011),(.017,.22,.021),m['rubber'],.002)
        for j in range(3):
            box(stage,path+f'/BootToeGuard{limb}_{j}',(x+.20,sign*.147,.075+j*.023),(.013,.14,.009),m['ivory'],.002)
        box(stage,path+f'/KneeReinforcement{limb}',(x+.13,sign*.151,.59),(.027,.146,.135),m['ivory'],.026)
        # Shoulder-to-elbow-to-wrist sleeves bend naturally forward.
        _loft(stage,path+f'/UpperSleeve{limb}',[[0.,sign*.258,1.395],[.012,sign*.335,1.28],[.055,sign*.377,1.155]],
            [[.115,.121],[.108,.112],[.087,.095]],fabric,rings=30,sides=48,
            fold=.0045,seed=30+limb,fabric=True,collision=True,seam_mat=m['stitch'])
        _loft(stage,path+f'/ElbowBellows{limb}',[[.055,sign*.377,1.17],[.085,sign*.387,1.105],[.102,sign*.386,1.065]],
            [[.091,.098],[.097,.102],[.084,.087]],m['seal'],rings=18,sides=48,fold=.003,seed=32+limb)
        _loft(stage,path+f'/ForeSleeve{limb}',[[.098,sign*.386,1.092],[.142,sign*.397,1.015],[.163,sign*.405,.959]],
            [[.088,.09],[.085,.087],[.074,.077]],fabric,rings=28,sides=48,
            fold=.004,seed=34+limb,fabric=True,collision=True,seam_mat=m['stitch'])
        cylinder(stage,path+f'/WristPressureRing{limb}',(.164,sign*.405,.947),.080,.033,m['hardware'])
        cylinder(stage,path+f'/WristRedBand{limb}',(.164,sign*.405,.974),.083,.024,m['red'])
        _loft(stage,path+f'/GlovePalm{limb}',[[.166,sign*.405,.947],[.177,sign*.408,.887],[.192,sign*.407,.843]],
            [[.065,.068],[.072,.066],[.061,.058]],fabric,rings=19,sides=36,fold=.0018,seed=40+limb,fabric=True)
        for finger in range(4):
            yy=sign*.407+(finger-1.5)*.027
            length=[.078,.094,.089,.071][finger]
            _loft(stage,path+f'/GloveFinger{limb}_{finger}',[[.207,yy,.852],[.221,yy,.821],[.226,yy,.852-length]],
                [[.018,.013],[.017,.013],[.010,.009]],fabric,rings=12,sides=20,fold=.0008,seed=50+limb*4+finger,fabric=True)
            for joint in range(2):
                box(stage,path+f'/FingerGrip{limb}_{finger}_{joint}',(.239,yy,.827-joint*.025),(.007,.017,.013),m['rubber'],.002)
        _loft(stage,path+f'/GloveThumb{limb}',[[.168,sign*.355,.901],[.195,sign*.336,.865],[.215,sign*.344,.834]],
            [[.024,.021],[.022,.019],[.014,.013]],fabric,rings=16,sides=24,fold=.001,seed=62+limb,fabric=True)
        # Mechanical locking rings and separate restrained identification stripes.
        for j,zz in enumerate([.613,.585,.557]):
            ring(stage,path+f'/KneePressureFold{limb}_{j}',(x+.035,sign*.153,zz),.106,.003,m['ivory'],segments=48,sides=8)
        if limb==1:
            cylinder(stage,path+'/ArmIdentificationStripe',(.030,sign*.35,1.237),.112,.032,m['red'])
            cylinder(stage,path+'/LegIdentificationStripe',(x+.015,sign*.149,.738),.131,.035,m['red'])
    # Collar, helmet and a flush curved visor aperture.
    cylinder(stage,path+'/NeckPressureSeal',(0.,0.,1.468),.146,.068,m['seal'])
    cylinder(stage,path+'/HelmetLockRing',(0.,0.,1.495),.164,.032,m['hardware'])
    ring(stage,path+'/HelmetLockBead',(0.,0.,1.509),.165,.005,m['ivory'],segments=80,sides=10)
    helmet_center=np.array([.003,0.,1.657]);helmet_radii=[.219,.228,.215]
    _ellipsoid(stage,path+'/HelmetOuterShell',helmet_center,helmet_radii,m['ivory'],rings=32,sides=80)
    _,gasket_edge=_visor_patch(stage,path+'/VisorDarkSeal',helmet_center,helmet_radii,.185,.145,.004,m['gasket'])
    _visor_patch(stage,path+'/GoldSolarVisor',helmet_center,helmet_radii,.174,.132,.008,m['gold'])
    _closed_curve(stage,path+'/VisorPressureRim',gasket_edge+[.002,0.,0.],.005,m['hardware'])
    brow=[]
    for yy in np.linspace(-.17,.17,80):
        zz=.142-.019*(yy/.17)**2
        xx=helmet_radii[0]*np.sqrt(1-(yy/helmet_radii[1])**2-(zz/helmet_radii[2])**2)+.01
        brow.append((helmet_center+[xx,yy,zz]).tolist())
    tube(stage,path+'/ProtectiveVisorBrow',brow,.018,m['ivory'])
    for side,sign in enumerate([-1,1]):
        cylinder(stage,path+f'/HelmetPivot{side}',(.005,sign*.227,1.657),.047,.025,m['hardware'],'Y')
        cylinder(stage,path+f'/HelmetPivotCap{side}',(.005,sign*.243,1.657),.034,.012,m['ivory'],'Y')
        for j in range(6):
            a=j*np.pi/3
            cylinder(stage,path+f'/HelmetPivotBolt{side}_{j}',(.005+.039*np.cos(a),sign*.244,1.657+.039*np.sin(a)),.0032,.005,m['gasket'],'Y')
        box(stage,path+f'/HelmetWorklightHousing{side}',(.098,sign*.173,1.766),(.060,.041,.035),m['hardware'],.008)
        cylinder(stage,path+f'/HelmetWorklightLens{side}',(.132,sign*.173,1.766),.012,.008,m['lens'],'X')
    # Rectangular life support equipment, deliberately distinct from soft suit.
    box(stage,path+'/PLSSMainEnclosure',(-.277,0.,1.22),(.232,.406,.492),m['ivory'],.035,collision=True)
    box(stage,path+'/PLSSRearHeatExchanger',(-.402,0.,1.235),(.025,.349,.382),m['hardware'],.012)
    for j in range(15):
        box(stage,path+f'/RadiatorChannel{j}',(-.417,-.149+j*.0213,1.235),(.006,.006,.331),m['seal'],.001)
    for sign in [-1,1]:
        box(stage,path+('/PLSSLeftRail' if sign<0 else '/PLSSRightRail'),(-.414,sign*.185,1.23),(.025,.021,.43),m['hardware'],.006)
    box(stage,path+'/SecondaryOxygenHousing',(-.273,0.,1.492),(.213,.356,.077),m['ivory'],.018)
    box(stage,path+'/PLSSLowerBattery',(-.275,0.,.982),(.229,.354,.087),m['panel'],.017)
    # Chest controls are small flush equipment, not a cartoon chest screen.
    box(stage,path+'/ChestController',(.199,0.,1.315),(.045,.212,.113),m['panel'],.014)
    box(stage,path+'/ChestControllerFace',(.226,0.,1.315),(.012,.185,.087),m['ivory'],.007)
    for j,yy in enumerate([-.063,-.021,.021,.063]):
        cylinder(stage,path+f'/ChestControlKnob{j}',(.237,yy,1.325),.010,.013,m['hardware'],'X')
        cylinder(stage,path+f'/ChestKnobCenter{j}',(.245,yy,1.325),.004,.005,m['gasket'],'X')
    box(stage,path+'/ChestReadout',(.238,0.,1.288),(.006,.098,.018),m['lens'],.003)
    for side,sign in enumerate([-1,1]):
        cylinder(stage,path+f'/VentilationConnector{side}',(.19,sign*.151,1.16),.026,.041,m['hardware'],'X')
        ring(stage,path+f'/ConnectorLock{side}',(.215,sign*.151,1.16),.022,.004,m['connector'],'X',segments=40,sides=8)
        hose=np.array([[.222,sign*.151,1.16],[.254,sign*.195,1.09],[.224,sign*.243,1.01],
            [.07,sign*.262,.987],[-.18,sign*.238,1.02],[-.33,sign*.215,1.11]])
        knots=np.linspace(0.,1.,len(hose));samples=PchipInterpolator(knots,hose,axis=0)(np.linspace(0,1,64))
        tube(stage,path+f'/LifeSupportUmbilical{side}',samples.tolist(),.023,m['seal'])
        for j in range(0,64,4):
            _ellipsoid(stage,path+f'/HoseReinforcement{side}_{j}',samples[j],[.014,.014,.014],m['hardware'],rings=6,sides=12)
        box(stage,path+f'/HarnessBuckle{side}',(.17,sign*.207,1.05),(.033,.046,.043),m['hardware'],.006)
        box(stage,path+f'/HarnessStrap{side}',(.158,sign*.221,1.086),(.015,.032,.13),m['seal'],.004)
    # Small garment pockets, tether eye and fastener details survive close view.
    box(stage,path+'/ThighUtilityPocket',(.155,-.14,.828),(.035,.154,.148),m['ivory'],.019)
    tube(stage,path+'/PocketBoundEdge',[[.175,-.213,.88],[.178,-.213,.76],[.178,-.067,.76],[.175,-.067,.88]],.003,m['stitch'])
    ring(stage,path+'/TetherAnchor',(.208,.20,1.035),.026,.006,m['hardware'],'X',segments=48,sides=10)
    for j,yy in enumerate([-.12,.12]):
        cylinder(stage,path+f'/WaistLatch{j}',(.194,yy,1.02),.013,.014,m['hardware'],'X')
    prim=root.GetPrim()
    prim.CreateAttribute('cave:simulation_scope',Sdf.ValueTypeNames.String).Set('static visual EVA search object with simplified collision; no human articulation or suit certification')
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
    bound=cache.ComputeWorldBound(prim).ComputeAlignedRange();low=np.asarray(bound.GetMin(),float);high=np.asarray(bound.GetMax(),float)
    radius=float(np.linalg.norm(np.maximum(np.abs(low[:2]),np.abs(high[:2]))))+.01
    colliders=[str(p.GetPath()) for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
    return dict(prim_path=path,class_name='astronaut',label_zh='宇航员',asset_revision=10,
        local_ground_footprint_radius_m=radius,height_m=float(high[2]),local_bounds_m=[low.tolist(),high.tolist()],
        collision_scope='static garment torso/pelvis, limbs, soles, boots and PLSS convex shapes; fine equipment visual',
        static_collision=True,collider_count=len(colliders),collider_paths=colliders,articulation=False,
        external_assets_used=False,preferred_view_direction_local_xy=[1.,-.28],camera_target_height_m=1.08,
        provenance='original procedural EVA suit with tailored folded loft meshes; not an official or mission-certified suit',
        visual_features=['tailored folded pressure garment','woven vertex albedo','continuous adult proportions',
            'curved inset gold visor','pressure seals and mechanical helmet fittings','PLSS radiator and umbilicals',
            'individual glove fingers','boot soles and tread','reinforced seams and utility pocket'])
