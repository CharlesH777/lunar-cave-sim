"""Build, render and collect an actual Isaac Sim cave search demonstration."""
from __future__ import annotations
import argparse,json,sys,os,time,traceback,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path)
p.add_argument('--environment',choices=['lunar','earth'],default='lunar')
p.add_argument('--map',dest='map_id',default='01',choices=[f'{i:02d}' for i in range(1,9)])
p.add_argument('--planner',choices=['astar','hybrid'],default='hybrid')
p.add_argument('--scene-dir',type=Path,help='Reuse a saved cave.usdc and scene_manifest.json without changing the environment')
p.add_argument('--gui',action='store_true')
p.add_argument('--stay-open',action='store_true')
p.add_argument('--preview-only',action='store_true')
p.add_argument('--video-frames',type=int,default=300)
p.add_argument('--skip-video',action='store_true')
p.add_argument('--skip-episode',action='store_true')
p.add_argument('--seed',type=int)
p.add_argument('--target',default='first_aid_kit')
p.add_argument('--resolution',type=int,default=1280)
a=p.parse_args();sys.argv=[sys.argv[0]]
if a.seed is None:a.seed=70+int(a.map_id) if a.environment=='lunar' else 7
if a.output is None:a.output=ROOT/'outputs'/((('maps_smooth_tuned' if a.planner=='hybrid' else 'maps_astar')+'/map_'+a.map_id) if a.environment=='lunar' else 'earth_demo')
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)

def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n');tmp.replace(path)

def status(phase,**kw):
    value=dict(phase=phase,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),output=str(a.output),**kw)
    dump(ROOT/'status.json',value);dump(a.output/'status.json',value)
    print('CAVE_STATUS',phase,kw,flush=True)

status('starting_isaac')
from isaacsim import SimulationApp
app=SimulationApp({'headless':not a.gui,'renderer':'RayTracedLighting','width':a.resolution,'height':a.resolution*9//16,
    'multi_gpu':False,'active_gpu':0,'physics_gpu':0,'sync_loads':True,
    'extra_args':['--ext-folder','/home/charles/.local/share/ov/data/exts/v2','--/app/extensions/registryEnabled=false',
                  '--/renderer/multiGpu/enabled=false','--/rtx/indirectDiffuse/enabled=true','--/rtx/ambientOcclusion/enabled=true',
                  '--/exts/omni.replicator.core/Orchestrator/enabled=true']},
    experience='/home/charles/ISAAC_SIM/IsaacLab/apps/isaaclab.python.kit' if a.gui else '/home/charles/ISAAC_SIM/IsaacLab/apps/isaaclab.python.headless.rendering.kit')

import numpy as np
import cv2
import carb
import omni.usd
import omni.replicator.core as rep
from pxr import Gf,UsdGeom,UsdLux
from PIL import Image,ImageDraw,ImageFont
from cave.scene import set_camera,sphere_light
if a.environment=='lunar':
    from cave.lunar_scene import build_scene
    from cave.lunar_geometry import cave_field,floor_height
    from cave.lunar_geometry import configure_layout,get_layout
    configure_layout(a.map_id)
else:
    from cave.scene import build_scene
    from cave.geometry import cave_field,floor_height
from cave.navigation import SearchTask
from cave.sensors import CaveSensors


def safe_info(d):
    return json.loads(json.dumps(d,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))


def annotate(frame,title,subtitle,index=None):
    im=Image.fromarray(frame[:,:,:3].copy());draw=ImageDraw.Draw(im,'RGBA');w,h=im.size
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    big=ImageFont.truetype(fontpath,max(18,w//42));small=ImageFont.truetype(fontpath,max(12,w//70))
    draw.rectangle((0,h-104,w,h),fill=(7,13,16,205))
    draw.text((30,h-88),title,font=big,fill=(241,238,223))
    draw.text((30,h-43),subtitle,font=small,fill=(169,190,190))
    draw.text((24,22),'SELENE / LAVA TUBE' if a.environment=='lunar' else 'KARST / 01',font=big,fill=(240,236,218))
    if index is not None:draw.text((w-160,28),f'{index:04d} / SIM',font=small,fill=(208,217,215))
    return np.asarray(im)


def main():
    status('building_scene')
    if a.scene_dir:
        source=a.scene_dir.resolve()
        meta=json.loads((source/'scene_manifest.json').read_text())
        if str(meta.get('map_id',a.map_id))!=a.map_id:
            raise ValueError('--map must match the saved scene map_id')
        omni.usd.get_context().open_stage(str(source/'cave.usdc'))
        stage=omni.usd.get_context().get_stage()
        from pxr import Sdf,Usd
        # Resolve textures against the source layer before exporting elsewhere.
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                value=attr.Get()
                if isinstance(value,Sdf.AssetPath) and value.resolvedPath:
                    attr.Set(Sdf.AssetPath(value.resolvedPath))
        meta['source_scene_directory']=str(source)
        meta['source_scene_sha256']=hashlib.sha256((source/'cave.usdc').read_bytes()).hexdigest()
    else:
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        meta=build_scene(stage,a.seed)
    meta['demonstration_planner']=a.planner
    # Correct world elevation for every target without modifying its model.
    for t in meta['targets']:
        pos=t['position'];z=float(floor_height(*pos[:2]))+.045
        UsdGeom.Xformable(stage.GetPrimAtPath(t['prim_path'])).GetOrderedXformOps()[0].Set(Gf.Vec3d(pos[0],pos[1],z))
        t['position'][2]=z+.18  # object centre used for aiming and planning
    dump(a.output/'scene_manifest.json',meta)
    settings=carb.settings.get_settings()
    settings.set_bool('/rtx/post/aa/op',True) if False else None
    settings.set('/rtx/post/dlss/execMode',2)
    settings.set('/rtx/raytracing/fractionalCutoutOpacity',True)
    settings.set('/rtx/sceneDb/ambientLightIntensity',.1)
    camera=UsdGeom.Camera.Get(stage,'/World/Cameras/Showcase')
    agent=UsdGeom.Camera.Get(stage,'/World/Cameras/Agent')
    rover=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Rover'))
    rop=rover.GetOrderedXformOps()[0]
    yawop=next((op for op in rover.GetOrderedXformOps() if op.GetOpName()=='xformOp:rotateZ'),None)
    if yawop is None:yawop=rover.AddRotateZOp()
    rop.Set(Gf.Vec3d(3,-.2,float(floor_height(3,-.2))))
    # Camera-mounted work lamp follows the agent. Kept separate from RGB labels.
    lamp=UsdLux.SphereLight.Get(stage,'/World/Lights/AgentLamp')
    if not lamp:lamp=sphere_light(stage,'/World/Lights/AgentLamp',(2,0,1),(.89,.93,1),180,.12)
    if a.environment=='lunar':lamp.CreateIntensityAttr(0.)
    lpos=lamp.GetOrderedXformOps()[0]
    rep.orchestrator.set_capture_on_play(False)
    rp=rep.create.render_product(str(camera.GetPath()),(a.resolution,a.resolution*9//16))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([rp])
    for _ in range(35):app.update()
    def capture(c,eye,target,subframes=8):
        set_camera(c,eye,target)
        rep.orchestrator.step(rt_subframes=subframes,pause_timeline=True)
        return np.asarray(rgb.get_data()).copy()[:,:,:3]
    status('rendering_previews',triangles=meta['shell_triangles'])
    shots=[('hero',(1.4,-.7,1.65),(11,-.1,1.7)),
           ('gallery',(9,-1.6,1.6),(15,-3.6,1.15)),
           ('underground_pool',(18,1.6,1.8),(23,4.0,1.5)),
           ('rescue_cache',(18.7,-.2,1.2),(21,1.2,.5))]
    if a.environment=='lunar':
        shots=[('hero',(4.8,0.,1.2),(12,0,1.1)),
               ('skylight',(1.5,1.2,1.6),(3.2,.15,8.5)),
               ('surface',(8,-6,10.2),(3.2,.15,7.3)),
               ('rescue_cache',(18.7,-.2,1.2),(21,1.2,.5))]
        t=next(t for t in meta['targets'] if t['id']=='first_aid_kit');tx,ty,tz=t['position']
        for angle in np.r_[np.pi,np.linspace(0,2*np.pi,24,endpoint=False)]:
            ex,ey=tx+1.65*np.cos(angle),ty+1.65*np.sin(angle)
            if min(cave_field(ex,ey,.3),cave_field(ex,ey,1.0))>.65:
                shots[-1]=('rescue_cache',(ex,ey,1.15),(tx,ty,tz));break
    for name,eye,target in shots:
        if a.environment=='lunar':
            if name=='surface':
                UsdGeom.Imageable(stage.GetPrimAtPath('/World/Rover')).CreateVisibilityAttr().Set('invisible')
            else:
                UsdGeom.Imageable(stage.GetPrimAtPath('/World/Rover')).CreateVisibilityAttr().Set('inherited')
                d=np.asarray(target)[:2]-np.asarray(eye)[:2];d/=max(np.linalg.norm(d),1e-6)
                bx,by=np.asarray(eye)[:2]-d*.25
                rop.Set(Gf.Vec3d(float(bx),float(by),float(floor_height(bx,by))));yawop.Set(float(np.degrees(np.arctan2(d[1],d[0]))))
        frame=capture(camera,eye,target,16)
        Image.fromarray(frame).save(a.output/(name+'.png'))
        print('SHOT',name,'mean',float(frame.mean()),flush=True)
    set_camera(camera,shots[0][1],shots[0][2])
    if a.environment=='lunar':
        UsdGeom.Imageable(stage.GetPrimAtPath('/World/Rover')).CreateVisibilityAttr().Set('inherited')
        rop.Set(Gf.Vec3d(2,.3,float(floor_height(2,.3))));yawop.Set(0.)
    # Export while local asset paths are valid; then make the saved USD portable.
    scene_path=a.output/'cave.usdc';stage.GetRootLayer().Export(str(scene_path))
    from pxr import Usd,Sdf
    saved=Usd.Stage.Open(str(scene_path))
    for prim in saved.Traverse():
        for attr in prim.GetAttributes():
            value=attr.Get()
            if isinstance(value,Sdf.AssetPath) and value.path.startswith(str(ROOT)):
                attr.Set(Sdf.AssetPath(os.path.relpath(value.path,a.output)))
    saved.GetRootLayer().Save();del saved
    if a.preview_only:
        status('preview_complete',scene=str(scene_path));return
    if not a.skip_video:
        status('rendering_showcase_video',frames=a.video_frames)
        writer=cv2.VideoWriter(str(a.output/'cave_showcase.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),24,(a.resolution,a.resolution*9//16))
        for i in range(a.video_frames):
            if (a.output/'REQUEST_STOP').exists():break
            t=i/max(1,a.video_frames-1)
            x=1.3+25*t
            y=.08*np.sin(t*2*np.pi) if a.environment=='lunar' else -.4+.6*np.sin(t*2*np.pi)
            eye=(x,y,1.2 if a.environment=='lunar' else 1.5+.15*np.sin(t*np.pi))
            look=(min(x+5,30.5),.08*np.sin(x*.3) if a.environment=='lunar' else .5*np.sin(x*.3),1.1 if a.environment=='lunar' else 1.3)
            if a.environment=='lunar':
                rop.Set(Gf.Vec3d(float(x+1.2),float(y),float(floor_height(x+1.2,y))));yawop.Set(0.)
            frame=capture(camera,eye,look,3)
            title=('LUNAR LAVA TUBE / Skylight and basalt' if t<.40 else 'SELENE / Subsurface object search') if a.environment=='lunar' else ('LIMESTONE / Connected chambers' if t<.40 else 'SUBTERRANEAN / Search and discovery')
            frame=annotate(frame,title,'Isaac Sim 5.1  |  Original cave meshes + scanned CC0 PBR  |  Scripted camera',i)
            writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            if i%50==0:status('rendering_showcase_video',frame=i,total=a.video_frames)
        writer.release()
    rgb.detach([rp.path]);rp.destroy()
    episode_result=None
    if not a.skip_episode:
        episode_result=collect_episode(stage,meta,agent,rop,yawop,lpos)
    dump(ROOT/'outputs/latest.json',{'output':str(a.output),'scene':str(scene_path),'preview':str(a.output/'hero.png')})
    status('complete',scene=str(scene_path),search_success=episode_result.get('success') if episode_result else None)
    if a.gui and a.stay_open:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([4.8,0,1.2]),target=np.array([12,0,1.1]),camera_prim_path='/OmniverseKit_Persp')
        while app.is_running() and not (a.output/'REQUEST_STOP').exists():app.update()


def collect_episode(stage,meta,cam,rop,yawop,lpos):
    status('planning_search',target=a.target)
    # Quantized continuous-field cache keeps conservative swept clearance fast.
    obstacles=meta['obstacles']
    from functools import lru_cache
    @lru_cache(maxsize=250000)
    def free_cached(ix,iy,iz):
        x,y,z=ix*.04,iy*.04,iz*.04
        if cave_field(x,y,z)<.035:return False
        for o in obstacles:
            if (o.get('kind')=='pool' or z <= o['height']+.05) and (x-o['x'])**2+(y-o['y'])**2 < (o['radius']+.04)**2:return False
        return True
    def free(x,y,z):return free_cached(round(x/.04),round(y/.04),round(z/.04))
    env=SearchTask(meta['targets'],free_space=free,floor_height=floor_height,seed=a.seed,max_steps=1200,planner=a.planner,
                   bounds=(-.5,34.,-14.,14.) if a.environment=='lunar' else (0.,29.,-8.,8.),
                   start_pose=tuple(meta.get('spawn',[2.,.3,0.])) if a.environment=='lunar' else (2.,0.,0.))
    obs=env.reset(a.target)
    # Planning target centre is above the ground and no target GT enters RGB input.
    path=env.plan_path()
    if a.planner=='hybrid':
        expert_name='Hybrid A* + '+('collision-checked C2 spline + ' if env.planning_diagnostics.get('smoothing',{}).get('accepted') else '')+'regulated pure pursuit'
    else:expert_name='grid A* + heading P'
    dump(a.output/'expert_path.json',{'path_xy_m':np.asarray(path).tolist(),
         'path_poses':env.path_poses,'source':expert_name+'; privileged oracle, not a trained policy',
         'diagnostics':env.planning_diagnostics})
    sensors=CaveSensors(stage,str(cam.GetPath()),(640,360))
    ep=a.output/'episode_000';ep.mkdir(exist_ok=True)
    writer=cv2.VideoWriter(str(a.output/'search_episode.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,360))
    records=[];initial_pose=None
    for i in range(env.max_steps):
        if (a.output/'REQUEST_STOP').exists():break
        x,y,yaw=map(float,env.pose);z=float(floor_height(x,y));forward=np.array([np.cos(yaw),np.sin(yaw),0])
        eye=np.array([x,y,z+.90])+forward*.17
        # Fixed downward tilt, no GT-based camera aiming in the policy observation.
        target=eye+forward*3+np.array([0,0,-.70])
        set_camera(cam,eye,target);rop.Set(Gf.Vec3d(x,y,z));yawop.Set(float(np.degrees(yaw)));lpos.Set(Gf.Vec3d(*map(float,eye+forward*.3)))
        sensor_frame=sensors.capture(target_prim_path=env.target['prim_path'])
        frame=sensor_frame['rgb'];depth=sensor_frame['depth_m'];mask=sensor_frame['instance_id'];info=sensor_frame['instance_mapping']
        visible=sensor_frame['evaluation']['target_visible_pixels']
        if i==0:
            dump(ep/'instance_mapping_initial.json',info)
            dump(ep/'camera.json',{**sensor_frame['camera'],'dt_s':env.dt,
                 'action':['linear_mps','yaw_radps','stop'],'controller':expert_name+'; ground-truth oracle, not a learned VLA'})
        action=env.expert_action()
        Image.fromarray(frame).save(ep/f'{i:05d}_rgb.jpg',quality=94)
        np.savez_compressed(ep/f'{i:05d}_sensors.npz',depth_m=depth,depth_valid=sensor_frame['depth_valid'],instance_id=mask,
                            world_from_camera=sensor_frame['camera']['world_from_camera'],K=sensor_frame['camera']['K'])
        dump(ep/f'{i:05d}_instances.json',info)
        before=obs
        obs,reward,term,trunc,metrics=env.step(action,visible_pixel_count=visible)
        records.append({'step':i,'observation':before,'action':np.asarray(action).tolist(),'reward':reward,'terminated':term,'truncated':trunc,
                        'privileged_info':metrics,'controller_diagnostics':safe_info(env.controller_diagnostics),'rgb':f'{i:05d}_rgb.jpg','sensors':f'{i:05d}_sensors.npz'})
        title='FIND / Red first aid kit' if a.target=='first_aid_kit' else 'FIND / '+a.target.replace('_',' ')
        subtitle=f'{"Hybrid A* + RPP" if a.planner=="hybrid" else "A* + P"} | 2x | d={metrics["distance_to_target_m"]:.1f}m | visible={visible}px | collisions={env.collisions}'
        writer.write(cv2.cvtColor(annotate(frame,title,subtitle,i),cv2.COLOR_RGB2BGR))
        if i%40==0:status('collecting_search_episode',step=i,distance_m=round(metrics['distance_to_target_m'],2),visible_pixels=visible)
        if term or trunc:
            Image.fromarray(frame).save(a.output/'found_target.png')
            dump(ep/'result.json',dict(metrics,frames=i+1,policy=expert_name+'; scripted privileged oracle',pose=env.pose.tolist(),
                 planning_diagnostics=env.planning_diagnostics,controller_diagnostics=env.controller_diagnostics))
            print('EPISODE_RESULT',metrics,flush=True);break
    writer.release()
    with (ep/'transitions.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    sensors.release()
    if not env.done:dump(ep/'result.json',{**(records[-1]['privileged_info'] if records else {}),'success':False,'reason':'interrupted','frames':len(records)})
    return json.loads((ep/'result.json').read_text())


try:
    main()
except Exception as exc:
    status('failed',error=str(exc));traceback.print_exc();raise
finally:
    app.close()
