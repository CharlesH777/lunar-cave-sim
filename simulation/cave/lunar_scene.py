"""Dry lunar lava-tube-inspired USD scene, with a true skylight and task lights."""
import numpy as np
import trimesh
from pxr import Gf,Sdf,UsdGeom,UsdLux,UsdShade,UsdPhysics
from .scene import material,mesh,camera,sphere_light
from .geometry import concatenate
from .lunar_geometry import build_shell,floor_height,surface_height,cave_field,get_layout
from .props import build_rover,box,cylinder
from .lunar_props import build_lunar_props
from .signage import build_signage


def build_scene(stage,seed=7):
    layout=get_layout()
    rng=np.random.default_rng(seed)
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    w=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(w.GetPrim())
    for n in ['Materials','Cave','Lights','Cameras']:UsdGeom.Xform.Define(stage,'/World/'+n)
    basalt=material(stage,'FracturedDarkRockProxy',(.08,.08,.08),.91,texture='dark_rock')
    dust=material(stage,'LunarRegolithAnalogue',(.28,.28,.28),.98,texture='moon_03')
    # Dark scanned rock is a visual proxy; these are not measured lunar BRDFs.
    v,f,ground=build_shell()
    mesh(stage,'/World/Cave/LavaTubeAndSkylight',v,f[~ground],basalt)
    mesh(stage,'/World/Cave/Regolith',v,f[ground],dust,tile=3.2)
    # Continue the landscape to the horizon instead of exposing a rectangular tile.
    xs=np.r_[np.arange(-250,-5,2),np.arange(-5,36.01,.5),np.arange(36.5,250,2)]
    ys=np.r_[np.arange(-250,-15,2),np.arange(-15,15.01,.5),np.arange(15.5,250,2)]
    xx,yy=np.meshgrid(xs,ys,indexing='ij');sv=np.column_stack([xx.ravel(),yy.ravel(),surface_height(xx,yy).ravel()])
    ii,jj=np.meshgrid(np.arange(len(xs)-1),np.arange(len(ys)-1),indexing='ij')
    aa=(ii*len(ys)+jj).ravel();bb=aa+len(ys)
    sf=np.concatenate([np.column_stack([aa,bb,bb+1]),np.column_stack([aa,bb+1,aa+1])])
    sc=sv[sf].mean(1);outside=(sc[:,0]<-4.9)|(sc[:,0]>35.8)|(sc[:,1]<-14.9)|(sc[:,1]>14.8)
    mesh(stage,'/World/Cave/SurroundingMoonSurface',sv,sf[outside],dust,False,tile=3.2)
    parts=[];obstacles=[]
    routes=[layout['main']]+layout['branches']
    def centerline_distance(x,y):
        best=1e9
        for route in routes:
            for a,b in zip(route[:-1],route[1:]):
                aa=np.asarray(a);d=np.asarray(b)-aa;t=np.clip((np.array([x,y])-aa)@d/(d@d),0,1)
                best=min(best,float(np.linalg.norm(np.array([x,y])-aa-t*d)))
        return best
    proto=trimesh.creation.icosphere(subdivisions=1)
    for i in range(1350):
        x=float(rng.uniform(-.7,32.));y=float(rng.uniform(-12.5,12.5))
        if cave_field(x,y,.4)<.1:continue
        # Collapse blocks cluster off the main driving line and near the skylight.
        large=i<90 and (abs(y)>1.5 or x<5 and abs(y)>.8)
        size=float(rng.uniform(.24,.60) if large else rng.uniform(.018,.10))
        if large and np.linalg.norm(np.array([x,y])-np.array(layout['spawn'][:2]))<size*1.35+.8:large=False;size=.08
        if large and any(np.linalg.norm(np.array([x,y])-np.array(p[:2]))<size*1.35+.7 for p in layout['targets'].values()):large=False;size=.06
        # Keep every authored route passable after scattering collapse blocks.
        if large and centerline_distance(x,y)<size*1.35+.80:large=False;size=.07
        vv=np.array(proto.vertices)*rng.uniform(.64,1.3,(len(proto.vertices),1))
        vv*=np.array([size,size*rng.uniform(.7,1.35),size*.7]);vv+=[x,y,float(floor_height(x,y))+size*.3]
        parts.append((vv,proto.faces))
        if large:obstacles.append({'x':x,'y':y,'radius':size*1.35,'height':size*1.1,'kind':'collapse_block'})
    bv,bf=concatenate(parts);mesh(stage,'/World/Cave/CollapseRubble',bv,bf,basalt)
    # Surface rubble surrounding the genuine hole; black sky is not a skybox image.
    parts=[]
    for i in range(260):
        x=float(rng.uniform(-4.5,35));y=float(rng.uniform(-10.5,10.5))
        if (x-3.2)**2+(y-.15)**2<2.35**2:continue
        s=float(rng.uniform(.035,.32));vv=np.array(proto.vertices)*s*rng.uniform(.7,1.3,(len(proto.vertices),1))
        vv+=[x,y,float(surface_height(x,y))+s*.25];parts.append((vv,proto.faces))
    bv,bf=concatenate(parts);mesh(stage,'/World/Cave/SurfaceRubble',bv,bf,basalt)
    targets=build_lunar_props(stage,positions=layout['targets'])
    # Keep entrance instruments in the skylight alcove after narrowing the tube.
    staging=[('/World/LunarArtifacts/CommunicationsTripod',3.9,-1.1,.53,1.55),
             ('/World/LunarArtifacts/SolarPanel',2.7,-1.15,.58,.4)]
    for path,x,y,r,h in staging:
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()[0].Set(Gf.Vec3d(x,y,float(floor_height(x,y))+.035))
    signage=build_signage(stage,junctions=layout['junctions'])
    obstacles.extend(signage['obstacles'] if isinstance(signage,dict) else signage)
    obstacles.extend([{'x':x,'y':y,'radius':r,'height':h,'kind':'equipment'} for _,x,y,r,h in staging])
    instructions={'first_aid_kit':'在月球熔岩管中寻找红色应急箱，靠近并停止。','backpack':'寻找左侧洞室里的橙色设备包。','sample_canister':'找到银色密封月壤样本罐。','toolbox':'寻找洞穴深处的黄色工具箱。','drill':'找到蓝色月壤取样电钻。'}
    for t in targets:
        t['instruction']=instructions[t['id']]
        obstacles.append({'x':t['position'][0],'y':t['position'][1],'radius':t['radius'],'height':.75,'kind':'target','id':t['id']})
    build_rover(stage)
    # Real rover-mounted cones; pitch down to illuminate both floor and signs.
    for i,y in enumerate([-.19,.19]):
        lamp=UsdLux.SphereLight.Define(stage,f'/World/Rover/Illumination/Headlight{i}')
        lamp.AddTranslateOp().Set(Gf.Vec3d(.455,y,.38));lamp.AddRotateXYZOp().Set(Gf.Vec3f(0,-77,0))
        lamp.CreateRadiusAttr(.06);lamp.CreateNormalizeAttr(True);lamp.CreateIntensityAttr(75.);lamp.CreateExposureAttr(8.5)
        lamp.CreateColorAttr(Gf.Vec3f(.96,.98,1.))
        shaping=UsdLux.ShapingAPI.Apply(lamp.GetPrim());shaping.CreateShapingConeAngleAttr(52.)
        shaping.CreateShapingConeSoftnessAttr(.28)
    # Mast lamp reaches route markers while bumper lamps show close terrain.
    lamp=UsdLux.SphereLight.Define(stage,'/World/Rover/Illumination/MastLight')
    lamp.AddTranslateOp().Set(Gf.Vec3d(.15,0,.92));lamp.AddRotateXYZOp().Set(Gf.Vec3f(0,-81,0))
    lamp.CreateRadiusAttr(.045);lamp.CreateNormalizeAttr(True);lamp.CreateIntensityAttr(275.);lamp.CreateExposureAttr(8.)
    lamp.CreateColorAttr(Gf.Vec3f(.96,.98,1.))
    shaping=UsdLux.ShapingAPI.Apply(lamp.GetPrim());shaping.CreateShapingConeAngleAttr(42.);shaping.CreateShapingConeSoftnessAttr(.3)
    # No atmosphere, fog, water or sky fill. Sunlight only enters via the hole.
    dome=UsdLux.DomeLight.Define(stage,'/World/Lights/BlackSpace');dome.CreateIntensityAttr(0.)
    sun=UsdLux.DistantLight.Define(stage,'/World/Lights/Sun');sun.CreateIntensityAttr(1800.)
    sun.CreateAngleAttr(.53);sun.CreateColorAttr(Gf.Vec3f(1.,.98,.95));sun.AddRotateXYZOp().Set(Gf.Vec3f(24.,-18.,-28.))
    metal=material(stage,'TaskLightHousing',(.10,.11,.12),.6,.7)
    lens=material(stage,'TaskLightLens',(.80,.84,.87),.2)
    s=UsdShade.Shader.Get(stage,str(lens.GetPath())+'/Surface')
    s.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(5,5.4,6))
    # Human-deployed temporary work lights illuminate caches and preserve hard shadows.
    for i,(x,y) in enumerate([(3.5,-1.6)]):
        z=float(floor_height(x,y))
        cylinder(stage,f'/World/Cave/WorkLight{i}/Foot',(x,y,z+.035),.16,.05,metal)
        cylinder(stage,f'/World/Cave/WorkLight{i}/Post',(x,y,z+.43),.02,.8,metal)
        box(stage,f'/World/Cave/WorkLight{i}/Head',(x,y,z+.85),(.20,.16,.12),metal,.018)
        box(stage,f'/World/Cave/WorkLight{i}/Lens',(x,y-.083,z+.85),(.15,.008,.075),lens,.005)
        light=sphere_light(stage,f'/World/Lights/Work{i}',(x,y,z+1.),(.91,.94,1.),850,.20)
        light.CreateExposureAttr(6.)
        obstacles.append({'x':x,'y':y,'radius':.18,'height':1.,'kind':'outpost_lamp'})
    camera(stage,'/World/Cameras/Agent',(2,0,.95),(5,0,.5))
    camera(stage,'/World/Cameras/Showcase',(1.4,-.6,1.7),(12,0,1.5))
    UsdPhysics.Scene.Define(stage,'/World/Physics').CreateGravityMagnitudeAttr(1.62)
    return {'environment':'lunar_lava_tube','revision':'narrow_v2_sparse_signs','map_id':layout['id'],'map_name':layout['name'],
            'layout':layout,'spawn':layout['spawn'],'seed':seed,'targets':targets,'obstacles':obstacles,
            'typical_cross_sections':{'main_clear_width_m':[2.4,2.9],'branch_clear_width_m':[1.9,2.3],
                                      'measurement_height_m':.30,'exclude':'junctions and skylight'},
            'wayfinding_sign_count':sum(o.get('kind')=='wayfinding_sign' for o in obstacles),
            'shell_vertices':len(v),'shell_triangles':len(f),'formation_triangles':0,'units':'m','up_axis':'Z',
            'gravity_mps2':1.62,'atmosphere':'vacuum; no atmospheric scattering or aerodynamic model',
            'lighting':'Sun through skylight; one outpost lamp; unlit branches explored using rover-mounted spotlights',
            'asset_scope':'moon_03 lab regolith analogue, dark_rock terrestrial visual proxy; not measured lunar BRDF',
            'scientific_scope':'procedural lunar lava-tube-inspired demo, not a surveyed lunar reconstruction',
            'agent_model':'collision-checked kinematic rover; lunar wheel/soil dynamics not modeled'}
