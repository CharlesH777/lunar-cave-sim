"""Locally authored USD environment with CC0 scanned PBR surfaces."""
from pathlib import Path
import numpy as np
import trimesh
from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade,UsdPhysics,Vt
from .geometry import build_shell,floor_height,ceiling_height,formation_mesh,concatenate,inside_cave
from .props import build_props,build_rover

ROOT=Path(__file__).resolve().parents[1]


def material(stage,name,color,roughness=.75,metallic=0.,texture=None):
    m=UsdShade.Material.Define(stage,'/World/Materials/'+name)
    s=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Surface');s.CreateIdAttr('UsdPreviewSurface')
    s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
    s.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metallic)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
    if texture:
        reader=UsdShade.Shader.Define(stage,str(m.GetPath())+'/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        for suffix,out,input_,space in [('diff','rgb','diffuseColor','sRGB'),('rough','r','roughness','raw'),('nor_gl','rgb','normal','raw')]:
            file=ROOT/'assets/textures'/f'{texture}_{suffix}_2k.jpg'
            if not file.exists():continue
            tex=UsdShade.Shader.Define(stage,str(m.GetPath())+'/'+suffix);tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(file)))
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set(space)
            tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat');tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
            if suffix=='nor_gl':
                tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
                tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))
            dtype=Sdf.ValueTypeNames.Color3f if input_=='diffuseColor' else Sdf.ValueTypeNames.Normal3f if input_=='normal' else Sdf.ValueTypeNames.Float
            s.CreateInput(input_,dtype).ConnectToSource(tex.ConnectableAPI(),out)
    return m


def mesh(stage,path,v,f,mat,collision=True,tile=2.4):
    v=np.asarray(v,dtype=np.float32);f=np.asarray(f,dtype=np.int32)
    m=UsdGeom.Mesh.Define(stage,path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v));m.CreateFaceVertexCountsAttr([3]*len(f))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
    m.CreateSubdivisionSchemeAttr('none');m.CreateDoubleSidedAttr(True)
    # Per-face dominant-axis planar UVs: floor, walls and roof all have equal texel density.
    vv=v[f];norm=np.cross(vv[:,1]-vv[:,0],vv[:,2]-vv[:,0]);axis=np.abs(norm).argmax(1)
    uv=np.empty((len(f),3,2),np.float32)
    for a,axes in [(0,[1,2]),(1,[0,2]),(2,[0,1])]:
        take=axis==a;uv[take]=vv[take][:,:,axes]/tile
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set(Vt.Vec2fArray.FromNumpy(uv.reshape(-1,2)))
    tm=trimesh.Trimesh(v,f,process=False)
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(tm.vertex_normals,dtype=np.float32)))
    m.SetNormalsInterpolation('vertex')
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('none')
    return m


def camera(stage,path,eye,target,focal=23):
    c=UsdGeom.Camera.Define(stage,path)
    c.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,target)),Gf.Vec3d(0,0,1)).GetInverse())
    c.CreateFocalLengthAttr(focal);c.CreateHorizontalApertureAttr(36);c.CreateVerticalApertureAttr(20.25)
    c.CreateClippingRangeAttr(Gf.Vec2f(.03,120))
    return c


def set_camera(c,eye,target):
    c.GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,target)),Gf.Vec3d(0,0,1)).GetInverse())


def sphere_light(stage,path,pos,color,intensity=1000,radius=.15):
    l=UsdLux.SphereLight.Define(stage,path);l.AddTranslateOp().Set(Gf.Vec3d(*map(float,pos)))
    l.CreateColorAttr(Gf.Vec3f(*color));l.CreateIntensityAttr(intensity);l.CreateRadiusAttr(radius)
    l.CreateExposureAttr(6.)
    return l


def build_scene(stage,seed=7):
    rng=np.random.default_rng(seed)
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    w=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(w.GetPrim())
    for n in ['Materials','Cave','Lights','Cameras']:UsdGeom.Xform.Define(stage,'/World/'+n)
    rock=material(stage,'ScannedLimestone',(.34,.29,.22),texture='rock_face')
    gravel=material(stage,'ScannedGravel',(.23,.20,.16),texture='gravel_floor_04')
    calcite=material(stage,'Calcite',(.40,.31,.21),.52,texture='rock_face')
    wet=material(stage,'Water',(.013,.035,.038),.10,.45)
    v,f,floor=build_shell()
    mesh(stage,'/World/Cave/Limestone',v,f[~floor],rock)
    mesh(stage,'/World/Cave/Gravel',v,f[floor],gravel,tile=2.5)
    formations=[];obstacles=[]
    for _ in range(205):
        x=float(rng.uniform(.0,30));y=float(rng.uniform(-6.8,6.8))
        if not inside_cave(x,y,1):continue
        ceil=ceiling_height(x,y)
        length=float(rng.uniform(.24,1.65));radius=float(rng.uniform(.10,.43))
        formations.append(formation_mesh(x,y,ceil+.10,length,radius,True,rng))
        # Keep centre route traversable; side columns create search occlusions.
        if abs(y) > 1.65 and rng.random()<.54:
            h=float(rng.uniform(.25,1.35))
            formations.append(formation_mesh(x,y,float(floor_height(x,y))-.04,h,radius*1.35,False,rng))
            obstacles.append({'x':x,'y':y,'radius':radius*1.35,'height':h})
    fv,ff=concatenate(formations);formation_triangles=len(ff);mesh(stage,'/World/Cave/Speleothems',fv,ff,calcite)
    boulders=[];pebbles=[]
    prototype=trimesh.creation.icosphere(subdivisions=1)
    for i in range(1600):
        x=float(rng.uniform(-.5,30.5));y=float(rng.uniform(-7,7))
        if not inside_cave(x,y,.35):continue
        large=i<65 and abs(y)>1.4
        size=float(rng.uniform(.23,.7) if large else rng.uniform(.025,.13))
        vv=np.array(prototype.vertices)*rng.uniform(.78,1.20,(len(prototype.vertices),1))
        vv*=np.array([size,size*rng.uniform(.65,1.4),size*.65])
        vv+=np.array([x,y,float(floor_height(x,y))+size*.23])
        (boulders if large else pebbles).append((vv,prototype.faces))
        if large:obstacles.append({'x':x,'y':y,'radius':size*1.2,'height':size})
    for name,parts,mat in [('Boulders',boulders,rock),('LooseStones',pebbles,gravel)]:
        bv,bf=concatenate(parts);mesh(stage,'/World/Cave/'+name,bv,bf,mat,collision=name=='Boulders')
    # Shallow still pools in side chambers, with a naturally irregular shoreline.
    for i,(x,y,rx,ry) in enumerate([(12,-4.7,2.2,1.25),(22.8,4,2.3,1.4)]):
        a=np.linspace(0,2*np.pi,100,endpoint=False);r=1+.07*np.sin(a*7)+.04*np.cos(a*11)
        vv=np.vstack([[x,y,.045],np.column_stack([x+rx*r*np.cos(a),y+ry*r*np.sin(a),np.full(len(a),.045)])])
        ff=np.array([[0,j+1,(j+1)%len(a)+1] for j in range(len(a))])
        mesh(stage,f'/World/Cave/Pool{i}',vv,ff,wet,False)
        obstacles.append({'x':x,'y':y,'radius':max(rx,ry),'height':.2,'kind':'pool'})
    targets=build_props(stage)
    for t in targets:
        obstacles.append({'x':t['position'][0],'y':t['position'][1],'radius':t['radius'],'height':.75,'kind':'target','id':t['id']})
    instructions={'first_aid_kit':'找到洞穴深处的红色急救箱，并停在它旁边。','backpack':'寻找左侧洞室里的橙色背包。','canteen':'找到银色水壶。','toolbox':'在洞穴尽头寻找黄色工具箱。','drill':'寻找蓝色电钻。'}
    for t in targets:t['instruction']=instructions.get(t['id'],instructions.get(t['category'],'寻找指定物品。'))
    build_rover(stage)
    # Practical lamps provide warm pools of light, with cool reflected cave fill.
    dome=UsdLux.DomeLight.Define(stage,'/World/Lights/Environment');dome.CreateIntensityAttr(32);dome.CreateColorAttr(Gf.Vec3f(.32,.43,.58))
    for i,(pos,col,power) in enumerate([
        ((1,0,3.8),(.58,.73,1.),1800),((7,1.5,2.3),(1.,.65,.30),950),
        ((11,-3.2,2.3),(.28,.68,.75),1100),((15,0,3.9),(1.,.78,.49),1200),
        ((21,3.0,3.2),(.30,.64,.75),1700),((24,-1,3.0),(1.,.65,.30),1000),
        ((29,0,3.5),(.40,.58,.84),950)]):
        sphere_light(stage,f'/World/Lights/Practical{i}',pos,col,power,.22)
    # Visible expedition lanterns anchor the light design.
    brass=material(stage,'LanternMetal',(.16,.17,.14),.30,.75)
    glow=material(stage,'LanternGlow',(1.,.58,.16),.25)
    s=UsdShade.Shader.Get(stage,str(glow.GetPath())+'/Surface');s.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(5,2.2,.5))
    for i,(x,y) in enumerate([(6,1.7),(15,-1.8),(25,-2.5)]):
        for part,z,r,h,mat in [('Base',.10,.13,.08,brass),('Glass',.27,.085,.26,glow),('Top',.43,.13,.06,brass)]:
            c=UsdGeom.Cylinder.Define(stage,f'/World/Cave/Lantern{i}{part}');c.CreateRadiusAttr(r);c.CreateHeightAttr(h)
            c.AddTranslateOp().Set(Gf.Vec3d(x,y,z));UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
        sphere_light(stage,f'/World/Lights/Lantern{i}',(x,y,.28),(1,.58,.22),110,.08)
    camera(stage,'/World/Cameras/Agent',(2,0,.95),(4,0,.8))
    camera(stage,'/World/Cameras/Showcase',(.8,-.6,1.6),(12,0,1.8))
    UsdPhysics.Scene.Define(stage,'/World/Physics').CreateGravityMagnitudeAttr(9.81)
    return {'seed':seed,'targets':targets,'obstacles':obstacles,'shell_vertices':len(v),'shell_triangles':len(f),
            'formation_triangles':formation_triangles,'units':'m','up_axis':'Z','bounds_m':[[-1.2,-7.5,0],[31.3,7,7]],
            'agent_model':'collision-checked kinematic differential drive; no wheel dynamics'}
