"""Original compact, engineering-inspired static lunar habitat search prop.

This authored visual asset has panelized insulation, pressure-shell framing,
mechanical airlock detail and service hardware. It is not a certified pressure
vessel, a functioning habitat or a reconstruction of a particular spacecraft.
The caller places the identity root on local ground Z=0; dimensions are metres.
"""
from __future__ import annotations

import math
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .props import box, cylinder, material, ring, tube, xform


def _palette(stage):
    return {
        'shell': material(stage, 'V10Hab_PaintedPressureShell', (.43, .45, .44), .67, .12),
        'blanket': material(stage, 'V10Hab_WarmMLIBlanket', (.43, .425, .39), .84, .09),
        'blanket_cool': material(stage, 'V10Hab_CoolMLIBlanket', (.37, .405, .42), .81, .13),
        'aluminium': material(stage, 'V10Hab_MachinedAluminium', (.37, .41, .44), .35, .77),
        'steel': material(stage, 'V10Hab_Fasteners', (.20, .24, .27), .29, .83),
        'gasket': material(stage, 'V10Hab_Seals', (.023, .029, .033), .84),
        'dark': material(stage, 'V10Hab_EquipmentDark', (.064, .079, .086), .66, .22),
        'glass': material(stage, 'V10Hab_PressureWindowGlass', (.013, .046, .061), .13, .47),
        'solar': material(stage, 'V10Hab_SolarCells', (.013, .029, .060), .29, .32),
        'solar_edge': material(stage, 'V10Hab_PVInterconnects', (.13, .17, .20), .36, .68),
        'orange': material(stage, 'V10Hab_LockIndicators', (.47, .17, .035), .68),
        'gold': material(stage, 'V10Hab_EquipmentMLI', (.34, .235, .075), .70, .37),
        'radiator': material(stage, 'V10Hab_RadiatorFaces', (.41, .44, .445), .82, .07),
        'status': material(stage, 'V10Hab_StatusLamp', (.035, .14, .075), .39, .12, (.035, .16, .07)),
    }


def _mesh(stage, path, points, faces, mat, collision=False, smooth=False):
    points=np.asarray(points,np.float32);faces=np.asarray(faces,np.int32)
    shape=UsdGeom.Mesh.Define(stage,path)
    shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    shape.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(faces)))
    shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    shape.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    shape.CreateDoubleSidedAttr(False)
    normal=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    if smooth:
        normals=np.zeros_like(points)
        for j in range(3):np.add.at(normals,faces[:,j],normal)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
        shape.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals));shape.SetNormalsInterpolation('vertex')
    else:
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-20)
        shape.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(normal,3,axis=0)))
        shape.SetNormalsInterpolation('faceVarying')
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
    return shape


def _beam(stage,path,a,b,radius,mat,collision=False):
    a,b=np.asarray(a,float),np.asarray(b,float);direction=b-a;length=float(np.linalg.norm(direction))
    if length<=1e-7:raise ValueError('Degenerate structural beam')
    root=cylinder(stage,path,(a+b)*.5,radius,length,mat,collision=collision)
    q=Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*map(float,direction/length))).GetQuat()
    root.AddOrientOp().Set(Gf.Quatf(float(q.GetReal()),Gf.Vec3f(*map(float,q.GetImaginary()))))
    return root


def _section(theta,scale=1.):
    return np.array([.82*math.cos(theta)*scale,1.43+.98*math.sin(theta)*scale])


def _pressure_shell(stage,path,mat):
    n=12;angles=(np.arange(n)+.5)*2*math.pi/n
    stations=[(-1.36,.84),(-1.13,1.),(1.02,1.),(1.26,.88)]
    points=[]
    for xx,scale in stations:
        for theta in angles:
            yy,zz=_section(theta,scale);points.append([xx,yy,zz])
    faces=[]
    for k in range(len(stations)-1):
        for j in range(n):
            a=k*n+j;b=k*n+(j+1)%n;c=(k+1)*n+(j+1)%n;d=(k+1)*n+j
            faces.extend([[a,b,c],[a,c,d]])
    # Cap fans point outward along -X/+X. Surface normals point out of the prop.
    rear=len(points);points.append([stations[0][0],0,1.43])
    front=len(points);points.append([stations[-1][0],0,1.43])
    for j in range(n):
        faces.append([rear,(j+1)%n,j])
        faces.append([front,(len(stations)-1)*n+j,(len(stations)-1)*n+(j+1)%n])
    return _mesh(stage,path,points,faces,mat,collision=True)


def _insulation_patch(stage,path,x0,x1,theta0,theta1,mat,phase):
    nu,nv=18,5;points=[];faces=[]
    a=_section(theta0);b=_section(theta1)
    outward=np.array([(a[0]+b[0])/.82,(a[1]+b[1]-2*1.43)/.98],float)
    outward/=np.linalg.norm(outward)
    for i in range(nu+1):
        u=i/nu
        for j in range(nv+1):
            v=j/nv;yz=(1-v)*a+v*b
            # Low millimetre-scale folds on each separate blanket, tapering
            # to a sewn perimeter rather than warping the pressure hull.
            taper=math.sin(math.pi*u)*math.sin(math.pi*v)
            wrinkle=(.0035*math.sin(31*u+9*v+phase)+.002*math.sin(19*u-13*v+phase*1.7))*taper
            yz=yz+outward*(.014+wrinkle)
            points.append([x0+(x1-x0)*u,yz[0],yz[1]])
    for i in range(nu):
        for j in range(nv):
            aa=i*(nv+1)+j;bb=(i+1)*(nv+1)+j
            faces.extend([[aa,bb,bb+1],[aa,bb+1,aa+1]])
    # Surface grid's orientation depends on the facet: explicitly orient it.
    tri=np.asarray(points)[np.asarray(faces)[0]]
    normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
    if normal[1]*outward[0]+normal[2]*outward[1]<0:faces=[f[::-1] for f in faces]
    return _mesh(stage,path,points,faces,mat,smooth=True)


def _solar_panel(stage,path,pos,size,m,tilt):
    root=xform(stage,path,pos,(tilt,0,0));sx,sy=size
    box(stage,path+'/StructuralFrame',(0,0,0),(sx,sy,.045),m['aluminium'],.009,collision=True)
    box(stage,path+'/DarkBackplane',(0,0,.025),(sx-.04,sy-.04,.012),m['gasket'],.003)
    nx,ny=7,4;cell_x=(sx-.075)/nx;cell_y=(sy-.075)/ny
    for i in range(nx):
        for j in range(ny):
            xx=-(sx-.075)*.5+(i+.5)*cell_x;yy=-(sy-.075)*.5+(j+.5)*cell_y
            box(stage,path+f'/PVCell{i}_{j}',(xx,yy,.035),(cell_x-.012,cell_y-.012,.006),m['solar'],.004,steps=3)
            for bus in [-1,1]:
                box(stage,path+f'/Bus{i}_{j}_{"A" if bus<0 else "B"}',(xx+bus*cell_x*.21,yy,.039),
                    (.0026,cell_y-.02,.0015),m['solar_edge'],0.)
    for i,xx in enumerate([-.5*sx+.035,.5*sx-.035]):
        for j,yy in enumerate([-.5*sy+.035,.5*sy-.035]):
            cylinder(stage,path+f'/MountBolt{i}_{j}',(xx,yy,.030),.008,.01,m['steel'])
    return root


def _dish(stage,path,position,m):
    root=xform(stage,path,position,(0,-23,0));n=48;nr=7;points=[[0,0,0]];faces=[]
    for k in range(1,nr+1):
        r=.155*k/nr
        for j in range(n):
            theta=2*math.pi*j/n;points.append([r*math.cos(theta),r*math.sin(theta),.11*(r/.155)**2])
    for j in range(n):faces.append([0,1+j,1+(j+1)%n])
    for k in range(nr-1):
        for j in range(n):
            a=1+k*n+j;b=1+k*n+(j+1)%n;c=1+(k+1)*n+(j+1)%n;d=1+(k+1)*n+j
            faces.extend([[a,b,c],[a,c,d]])
    mesh=_mesh(stage,path+'/Reflector',points,faces,m['radiator'],smooth=True)
    mesh.CreateDoubleSidedAttr(True)
    ring(stage,path+'/Rim',(0,0,.11),.155,.005,m['aluminium'])
    for j in range(3):
        theta=2*math.pi*j/3
        _beam(stage,path+f'/FeedStrut{j}',(.14*math.cos(theta),.14*math.sin(theta),.095),(0,0,.19),.0035,m['steel'])
    cylinder(stage,path+'/Feed',(0,0,.19),.014,.035,m['dark'])
    return root


def build_lunar_base(stage,path='/World/SearchTargets/lunar_base'):
    """Create an original compact static habitat, front airlock at local +X."""
    if stage.GetPrimAtPath(path):raise ValueError('Refusing to overwrite a target: '+path)
    root=xform(stage,path);m=_palette(stage)
    root.GetPrim().CreateAttribute('cave:asset_revision',Sdf.ValueTypeNames.Int,custom=True).Set(10)
    root.GetPrim().CreateAttribute('cave:simulation_scope',Sdf.ValueTypeNames.String,custom=True).Set(
        'engineering-inspired static habitat search prop; no pressure, life-support or mechanism simulation')
    _pressure_shell(stage,path+'/PressureHull',m['shell'])
    n=12;angles=(np.arange(n)+.5)*2*math.pi/n
    cuts=[-1.09,-.40,.30,.98]
    for segment,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
        for facet in range(n):
            lo=angles[facet]+.023;hi=angles[facet]+2*math.pi/n-.023
            _insulation_patch(stage,path+f'/Insulation/Segment{segment}_Facet{facet}',a+.023,b-.023,lo,hi,
                m['blanket'] if (facet+segment)%3 else m['blanket_cool'],facet*1.37+segment*.49)
    for k,xx in enumerate([-1.12,-.40,.30,1.01]):
        for j,theta in enumerate(angles):
            p=_section(theta);q=_section(theta+2*math.pi/n)
            _beam(stage,path+f'/PressureFrame/Ring{k}_Beam{j}',[xx,p[0],p[1]],[xx,q[0],q[1]],.018,m['aluminium'])
            # Shallow radial bolt seats make the ring structure readable.
            cylinder(stage,path+f'/PressureFrame/Ring{k}_Bolt{j}',(xx,p[0],p[1]),.014,.035,m['steel'],'X')
    for j,theta in enumerate(angles):
        yy,zz=_section(theta)
        _beam(stage,path+f'/PressureFrame/Longeron{j}',[-1.12,yy,zz],[1.01,yy,zz],.011,m['aluminium'])

    # Rear service bulkhead: connection manifold, removable equipment and ducts.
    cylinder(stage,path+'/RearBulkhead/InnerPanel',(-1.365,0,1.43),.64,.023,m['aluminium'],'X')
    for i,yy in enumerate([-.31,.31]):
        box(stage,path+f'/RearBulkhead/ServiceBay{i}',(-1.405,yy,1.42),(.09,.39,.49),m['dark'],.018)
        box(stage,path+f'/RearBulkhead/MLICover{i}',(-1.458,yy,1.42),(.015,.35,.43),m['gold'],.009)
        for j,zz in enumerate([1.28,1.42,1.56]):
            cylinder(stage,path+f'/RearBulkhead/Connector{i}_{j}',(-1.48,yy,zz),.025,.025,m['steel'],'X')
    for i,yy in enumerate([-.55,.55]):
        tube(stage,path+f'/RearBulkhead/ProtectedHarness{i}',[(-1.42,yy,1.50),(-1.43,yy,1.02),
            (-1.20,yy,.78),(-.92,yy,.63)],.025,m['gasket'])

    # Compact circular pressure hatch, segmented dogs and mechanical handles.
    cylinder(stage,path+'/Airlock/Drum',(1.38,0,1.29),.615,.45,m['shell'],'X',collision=True)
    ring(stage,path+'/Airlock/OuterFlange',(1.618,0,1.29),.605,.028,m['aluminium'],'X',segments=64)
    ring(stage,path+'/Airlock/Seal',(1.646,0,1.29),.550,.018,m['gasket'],'X',segments=64)
    cylinder(stage,path+'/Airlock/Hatch',(1.655,0,1.29),.531,.047,m['shell'],'X',collision=True)
    ring(stage,path+'/Airlock/HatchFrame',(1.683,0,1.29),.510,.012,m['aluminium'],'X')
    for i in range(12):
        theta=2*math.pi*i/12;yy=.586*math.cos(theta);zz=1.29+.586*math.sin(theta)
        cylinder(stage,path+f'/Airlock/FlangeFastener{i}',(1.646,yy,zz),.013,.021,m['steel'],'X')
    for i,theta in enumerate([math.pi*.25,math.pi*.75,math.pi*1.25,math.pi*1.75]):
        yy=.448*math.cos(theta);zz=1.29+.448*math.sin(theta)
        box(stage,path+f'/Airlock/LockDog{i}',(1.701,yy,zz),(.038,.09,.035),m['steel'],.006,rot=(math.degrees(theta),0,0))
        cylinder(stage,path+f'/Airlock/LockPin{i}',(1.728,yy,zz),.013,.024,m['orange'],'X')
    cylinder(stage,path+'/Airlock/WindowSeat',(1.694,0,1.49),.149,.025,m['aluminium'],'X')
    cylinder(stage,path+'/Airlock/WindowGlass',(1.713,0,1.49),.116,.012,m['glass'],'X')
    ring(stage,path+'/Airlock/WindowGasket',(1.721,0,1.49),.120,.008,m['gasket'],'X')
    _beam(stage,path+'/Airlock/HatchLeverA',[1.721,-.17,1.20],[1.721,.17,1.20],.018,m['aluminium'])
    for i,yy in enumerate([-.17,.17]):cylinder(stage,path+f'/Airlock/LeverPivot{i}',(1.713,yy,1.20),.031,.033,m['steel'],'X')
    for i,zz in enumerate([.98,1.57]):
        box(stage,path+f'/Airlock/HingeBlock{i}',(1.57,.545,zz),(.17,.08,.105),m['aluminium'],.012)
        cylinder(stage,path+f'/Airlock/HingePin{i}',(1.57,.574,zz),.027,.15,m['steel'])
    box(stage,path+'/Airlock/ControlHousing',(1.27,-.687,1.29),(.28,.12,.36),m['dark'],.025)
    box(stage,path+'/Airlock/ControlScreen',(1.27,-.751,1.34),(.19,.009,.15),m['glass'],.008)
    for i in range(3):cylinder(stage,path+f'/Airlock/ControlKey{i}',(1.20+i*.065,-.756,1.19),.012,.012,m['aluminium'],'Y')
    cylinder(stage,path+'/Airlock/ReadyLamp',(1.27,-.758,1.47),.010,.012,m['status'],'Y')

    # Guarded stair landing: functional-looking tread pitch and non-slip bars.
    for i,(xx,zz,length) in enumerate([(1.98,.075,.38),(1.85,.205,.34),(1.70,.335,.30),(1.54,.465,.26)]):
        box(stage,path+f'/Access/Step{i}',(xx,0,zz),(length,.68,.13),m['aluminium'],.009,collision=True)
        for j in range(8):
            yy=-.28+j*.08
            box(stage,path+f'/Access/Grip{i}_{j}',(xx,yy,zz+.068),(length-.025,.009,.006),m['steel'],.001)
    for k,side in enumerate([-1,1]):
        tube(stage,path+f'/Access/Handrail{k}',[(2.10,side*.425,.65),(1.92,side*.425,.91),
            (1.70,side*.425,1.15),(1.42,side*.425,1.15)],.026,m['aluminium'])
        for j,(xx,top,bottom) in enumerate([(1.95,.90,.14),(1.57,1.15,.50)]):
            _beam(stage,path+f'/Access/HandrailPost{k}_{j}',[xx,side*.425,bottom],[xx,side*.425,top],.018,m['aluminium'])

    contacts=[]
    for i,xx in enumerate([-.93,.84]):
        for j,side in enumerate([-1,1]):
            yy=side*.68;contacts.append([xx,yy,0.])
            cylinder(stage,path+f'/Undercarriage/Foot{i}_{j}',(xx,yy,.035),.195,.07,m['aluminium'],collision=True)
            cylinder(stage,path+f'/Undercarriage/Pad{i}_{j}',(xx,yy,.012),.173,.024,m['gasket'])
            cylinder(stage,path+f'/Undercarriage/InnerLeg{i}_{j}',(xx,yy,.20),.045,.30,m['steel'],collision=True)
            cylinder(stage,path+f'/Undercarriage/OuterLeg{i}_{j}',(xx,yy,.40),.072,.25,m['aluminium'],collision=True)
            cylinder(stage,path+f'/Undercarriage/Collar{i}_{j}',(xx,yy,.30),.089,.055,m['dark'])
            _beam(stage,path+f'/Undercarriage/Diagonal{i}_{j}',[xx-.16,yy*.86,.57],[xx,yy,.15],.021,m['steel'])
            for k in range(3):
                theta=2*math.pi*k/3
                cylinder(stage,path+f'/Undercarriage/FootBolt{i}_{j}_{k}',(xx+.13*math.cos(theta),yy+.13*math.sin(theta),.075),.010,.017,m['steel'])
    for i,xx in enumerate([-.93,.84]):
        _beam(stage,path+f'/Undercarriage/Crossmember{i}',[xx,-.72,.52],[xx,.72,.52],.045,m['aluminium'])

    # Pressure windows are small, layered rectangular ports rather than large
    # decorative circles. Each has a shallow protective shutter and fasteners.
    for k,side in enumerate([-1,1]):
        for j,xx in enumerate([-.60,.35]):
            yy=side*.825
            box(stage,path+f'/Windows/Outer{k}_{j}',(xx,yy,1.55),(.36,.075,.30),m['aluminium'],.042)
            box(stage,path+f'/Windows/Gasket{k}_{j}',(xx,yy+side*.042,1.55),(.303,.015,.243),m['gasket'],.032)
            box(stage,path+f'/Windows/Glass{k}_{j}',(xx,yy+side*.052,1.55),(.277,.009,.215),m['glass'],.026)
            for bolt,(dx,dz) in enumerate([(-.147,-.105),(.147,-.105),(-.147,.105),(.147,.105),(0.,-.132),(0.,.132)]):
                cylinder(stage,path+f'/Windows/Bolt{k}_{j}_{bolt}',(xx+dx,yy+side*.046,1.55+dz),.0075,.014,m['steel'],'Y')
            box(stage,path+f'/Windows/Shutter{k}_{j}',(xx,yy,1.77),(.35,.027,.115),m['shell'],.012,rot=(side*25.,0,0))

    for k,side in enumerate([-1,1]):
        for j,xx in enumerate([-.66,.32]):
            _beam(stage,path+f'/Power/HingeArm{k}_{j}',[xx,side*.72,.75],[xx,side*1.13,.89],.025,m['aluminium'],collision=True)
        _solar_panel(stage,path+f'/Power/SolarWing{k}',(-.19,side*1.22,.91),(1.72,.64),m,side*9.)
        cylinder(stage,path+f'/Power/Hinge{k}',(-.19,side*.89,.91),.040,1.35,m['steel'],'X')
        tube(stage,path+f'/Power/PowerHarness{k}',[(-.73,side*.94,.91),(-.88,side*.83,.76),
            (-.88,side*.64,.57),(-.50,side*.62,.43)],.017,m['gasket'])
    box(stage,path+'/Power/BatteryEnclosure',(-.34,0,.37),(.65,.42,.24),m['gold'],.025,collision=True)
    for i,xx in enumerate([-.59,-.39,-.19,.01]):
        box(stage,path+f'/Power/BatteryStrap{i}',(xx,0,.37),(.024,.434,.25),m['aluminium'],.008)

    box(stage,path+'/Thermal/RadiatorFrame',(-.27,0,2.395),(1.26,.59,.035),m['aluminium'],.012)
    for i in range(12):
        box(stage,path+f'/Thermal/RadiatorTile{i}',(-.82+i*.10,0,2.422),(.087,.54,.013),m['radiator'],.004)
    for k,side in enumerate([-1,1]):
        tube(stage,path+f'/Thermal/CoolantLine{k}',[(-.94,side*.25,2.43),(-1.08,side*.25,2.29),
            (-1.12,side*.45,1.98)],.012,m['aluminium'])
    box(stage,path+'/Thermal/ServiceCassette',(-.72,-.84,1.0),(.48,.16,.28),m['dark'],.017)
    for i in range(7):
        box(stage,path+f'/Thermal/ServiceVent{i}',(-.90+i*.059,-.925,1.0),(.026,.012,.19),m['steel'],.003)
    for i,xx in enumerate([-.9,-.54]):
        tube(stage,path+f'/Thermal/GrabHandle{i}',[(xx,-.96,1.13),(xx,-1.02,1.13),
            (xx,-1.02,1.23),(xx,-.96,1.23)],.012,m['aluminium'])

    cylinder(stage,path+'/Comms/Mast',(-.80,.17,2.59),.019,.32,m['aluminium'])
    _dish(stage,path+'/Comms/Dish',(-.80,.17,2.75),m)
    cylinder(stage,path+'/Comms/OmniAntenna',(.64,.27,2.62),.012,.30,m['dark'])
    cylinder(stage,path+'/Comms/OmniCap',(.64,.27,2.781),.023,.025,m['shell'])
    return _metadata(stage,path,contacts)


def _metadata(stage,path,contacts):
    prim=stage.GetPrimAtPath(path)
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
    bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange();low=np.asarray(bounds.GetMin(),float);high=np.asarray(bounds.GetMax(),float)
    radius=0.;collision=[]
    for item in Usd.PrimRange(prim):
        if item.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Habitat must remain static')
        if item.HasAPI(UsdPhysics.CollisionAPI):collision.append(str(item.GetPath()))
        if item.IsA(UsdGeom.Boundable):
            bb=cache.ComputeWorldBound(item).ComputeAlignedRange()
            if bb.IsEmpty():continue
            lo,hi=np.asarray(bb.GetMin(),float),np.asarray(bb.GetMax(),float)
            radius=max(radius,float(np.linalg.norm(np.maximum(abs(lo[:2]),abs(hi[:2])))))
    if not np.isfinite(np.r_[low,high,radius]).all() or not collision:raise ValueError('Invalid habitat bounds or collision')
    if radius+.025>2.4 or high[2]>3.:raise ValueError('Compact habitat exceeds intended placement bounds')
    view=np.asarray([1.,-.76]);view/=np.linalg.norm(view)
    return dict(prim_path=path,class_name='lunar_base',label_zh='小型月球基地',asset_revision=10,
        local_ground_footprint_radius_m=float(radius+.025),height_m=float(high[2]),
        local_bounds_m=[low.tolist(),high.tolist()],ground_contact_points_local_m=contacts,
        preferred_view_direction_local_xy=view.tolist(),camera_target_height_m=1.26,
        collision_scope='static pressure hull, airlock, stair treads, structural feet and solar arrays; trim is visual',
        static_collision=True,collision_prim_count=len(collision),collision_prim_paths=collision,
        rigid_body=False,articulation=False,external_assets_used=False,
        provenance='original compact engineering-inspired lunar habitat; not a certified or flight-specific design',
        professional_details=['panelized wrinkled insulation','faceted pressure shell and ring frames',
            'latching airlock with pressure window','guarded non-slip steps','telescopic landing feet',
            'individual solar cells and harnesses','roof radiator and service cassette','parabolic communications reflector'])
