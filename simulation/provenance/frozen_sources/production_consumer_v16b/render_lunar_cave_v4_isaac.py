"""Render the revised pit, portal and irregular cavern in actual Isaac RTX."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=1600)
    parser.add_argument('--subframes',type=int,default=32)
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    source=args.source.resolve(strict=True);out=args.output.absolute();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();app=None
    protected=[source/'cave.usdc',source/'scene_config.json',ROOT/'outputs/latest.json',ROOT/'status.json']
    hashes={str(p):digest(p) for p in protected}

    def status(phase,**kw):
        record=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.monotonic()-started,2),renderer='Isaac Sim 5.1 RTX',**kw)
        dump(out/'status.json',record);print('LUNAR_V4_RENDER',json.dumps(record),flush=True)

    try:
        status('starting_isaac')
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,renderer='RayTracedLighting',width=args.width,height=args.width*9//16,
            multi_gpu=False,active_gpu=0,physics_gpu=0,sync_loads=True,
            extra_args=['--ext-folder','/home/charles/.local/share/ov/data/exts/v2',
                '--/app/extensions/registryEnabled=false','--/renderer/multiGpu/enabled=false',
                '--/rtx/indirectDiffuse/enabled=true','--/rtx/ambientOcclusion/enabled=true',
                '--/exts/omni.replicator.core/Orchestrator/enabled=true']),
            experience='/home/charles/ISAAC_SIM/IsaacLab/apps/isaaclab.python.headless.rendering.kit')
        import numpy as np
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf,Sdf,UsdGeom,UsdLux,UsdShade,Vt
        from PIL import Image,ImageDraw,ImageFont
        from cave.scene import camera,set_camera,sphere_light,mesh
        from cave.props import build_rover
        from cave.lunar_pit_v4 import surface_height
        from cave.lunar_morphology_v4 import floor_height_at
        from cave.lunar_visual_v4 import _continuous_terrain_color
        from build_lunar_cave_v4_cpu import evaluate_scene_field

        status('loading_revised_asset')
        context=omni.usd.get_context()
        if not context.open_stage(str(source/'cave.usdc')):raise RuntimeError('Cannot load revised asset')
        stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                val=attr.Get()
                if isinstance(val,Sdf.AssetPath) and val.resolvedPath:attr.Set(Sdf.AssetPath(val.resolvedPath))
        config=json.loads((source/'scene_config.json').read_text())
        def render_floor(x,y):
            return float(floor_height_at(config['cavern'],x,y))
        if config.get('ground_profile_v7'):
            from cave.lunar_gravel_v5 import _FloorProjector
            with np.load(source/'mesh.npz') as data:
                render_projector=_FloorProjector(data['vertices'],data['faces'])
            def render_floor(x,y):
                heights,_,_,_=render_projector.project(np.asarray([[x,y]],float))
                if not np.isfinite(heights[0]):raise ValueError('No rendered floor support at '+str((x,y)))
                return float(heights[0])
        settings=carb.settings.get_settings()
        settings.set('/rtx/post/dlss/execMode',2)
        settings.set('/rtx/sceneDb/ambientLightIntensity',0.)
        UsdGeom.Xform.Define(stage,'/World/RevisionReview')
        # A visual surrounding landscape extends the finite terrain patch; it adds no cave geometry.
        ground=UsdShade.Material.Get(stage,'/World/Materials/Regolith')
        geometry_report=json.loads((source/'geometry_report.json').read_text())
        low,high=geometry_report['grid_bounds_m'];x0,x1,y0,y1=low[0],high[0],low[1],high[1]
        spacing=geometry_report['voxel_spacing_m'];shape=geometry_report['grid_shape']
        inner_x=low[0]+np.arange(shape[0])*spacing;inner_y=low[1]+np.arange(shape[1])*spacing
        def left_axis(bound):
            return np.unique(np.r_[np.linspace(-1200.,bound-10.,80),np.arange(bound-10.,bound,spacing),bound])
        def right_axis(bound):
            return np.unique(np.r_[bound,np.arange(bound+spacing,bound+10.,spacing),np.linspace(bound+10.,1200.,80)])
        all_y=np.r_[left_axis(y0)[:-1],inner_y,right_axis(y1)[1:]]
        # Inner boundary vertices match the source grid exactly; no open seam at the terrain extension.
        strips=[(left_axis(x0),all_y),(right_axis(x1),all_y),
                (inner_x,left_axis(y0)),(inner_x,right_axis(y1))]
        for i,(xs,ys) in enumerate(strips):
            xx,yy=np.meshgrid(xs,ys,indexing='ij');zz=surface_height(config['pit'],xx,yy)
            v=np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])
            ii,jj=np.meshgrid(np.arange(len(xs)-1),np.arange(len(ys)-1),indexing='ij')
            a=(ii*len(ys)+jj).ravel();b=a+len(ys)
            f=np.r_[np.column_stack([a,b,b+1]),np.column_stack([a,b+1,a+1])]
            extension=mesh(stage,f'/World/RevisionReview/SurroundingTerrain{i}',v,f,ground,collision=False,
                           tile=float(config.get('floor_texture_tile_m',.65)))
            eps=.175
            gx=(surface_height(config['pit'],xx+eps,yy)-surface_height(config['pit'],xx-eps,yy))/(2*eps)
            gy=(surface_height(config['pit'],xx,yy+eps)-surface_height(config['pit'],xx,yy-eps))/(2*eps)
            n=np.column_stack([-gx.ravel(),-gy.ravel(),np.ones(xx.size)]);n/=np.linalg.norm(n,axis=1,keepdims=True)
            extension.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
            extension.CreateDisplayColorPrimvar('vertex').Set(Vt.Vec3fArray.FromNumpy(_continuous_terrain_color(v,n)))
            if int(config.get('surface_detail_revision',0))>=5:
                from cave.lunar_gravel_material_v5 import set_gravel_primvars
                set_gravel_primvars(extension,tile_m=2.5,texture_resolution='4k')
        sun=UsdLux.DistantLight.Define(stage,'/World/RevisionReview/Sun')
        sun.CreateIntensityAttr(3500.);sun.CreateAngleAttr(.53)
        sun.CreateColorAttr(Gf.Vec3f(1.,.99,.97));sun.AddRotateXYZOp().Set(Gf.Vec3f(24.,-24.,-25.))
        dome=UsdLux.DomeLight.Define(stage,'/World/RevisionReview/BlackSpace');dome.CreateIntensityAttr(0.)
        cam=camera(stage,'/World/RevisionReview/Camera',(-12,-38,115),(-20,0,12),focal=24.)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.03,4000.))
        lamps=[];powers=[1200.,1200.,1600.,1100.]
        for i,power in enumerate(powers):
            light=sphere_light(stage,f'/World/RevisionReview/InspectionLamp{i}',(0,0,5),(.94,.97,1.),power,.30 if i<2 else .65)
            light.CreateNormalizeAttr(True);light.CreateExposureAttr(6.);lamps.append(light)
        rover=build_rover(stage,'/World/RevisionReview/ScaleRover')
        rover_prim=stage.GetPrimAtPath(rover['prim_path']);rx,ry=18.,-3.
        if config.get('rover_route'):
            from cave.rover_route_v6 import route_y
            ry=float(route_y(config['rover_route'],rx))
        rz=render_floor(rx,ry)
        xform=UsdGeom.Xformable(rover_prim);xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(rx,ry,rz+.03))
        dx=(render_floor(rx+.25,ry)-render_floor(rx-.25,ry))/.5
        dy=(render_floor(rx,ry+.25)-render_floor(rx,ry-.25))/.5
        if config.get('rover_route'):
            derivative=float(route_y(config['rover_route'],rx+.05)-route_y(config['rover_route'],rx-.05))/.1
            yaw=float(np.arctan(derivative));c,s=np.cos(yaw),np.sin(yaw)
            forward_grade=dx*c+dy*s;side_grade=-dx*s+dy*c
            xform.AddRotateXYZOp().Set(Gf.Vec3f(float(np.degrees(np.arctan(side_grade))),
                float(-np.degrees(np.arctan(forward_grade))),float(np.degrees(yaw))))
        else:
            xform.AddRotateXYZOp().Set(Gf.Vec3f(float(np.degrees(np.arctan(dy))),float(-np.degrees(np.arctan(dx))),0.))

        views=[
            dict(name='01_surface_collapse_pit',title='LUNAR PIT / collapse rim, steep walls and floor',
                 eye=[-16,-48,130],target=[-22,0,29],focal=25.,lamp_gain=0.,rover=False),
            dict(name='02_pit_rim_downlook',title='PIT RIM / overhang and debris floor',
                 eye=[-10,-23,62],target=[-8,0,1],focal=22.,lamp_gain=0.,rover=False),
            dict(name='03_floor_to_side_cavern',title='PIT FLOOR / lateral cavern beneath the overhang',
                 eye=[-17,-5,2.8],target=[16,1,6],focal=22.,lamp_gain=3.,rover=False),
            dict(name='04_cavern_toward_pit',title='CAVERN / daylight from the collapse entrance',
                 eye=[7,-2,3.5],target=[-24,-2,4.5],focal=20.,lamp_gain=12.,rover=False),
            dict(name='05_irregular_cavern',title='IRREGULAR CAVERN / roof sag, residual rock and deposits',
                 eye=[11,-6.5,3],target=[35,0,5.5],focal=20.,lamp_gain=8.,rover=True),
        ]
        if int(config.get('surface_detail_revision',0))>=5:
            g0=render_floor(14.,0.)
            g1=render_floor(16.8,0.)
            views.append(dict(name='06_gravel_closeup',title='GROUND DETAIL / scanned gravel and real 3D fragments',
                eye=[14.,0.,g0+.75],target=[16.8,0.,g1+.05],focal=30.,lamp_gain=.15,rover=False,closeup=True))
        if config.get('ground_profile_v7'):
            views.append(dict(name='07_uneven_ground_obstacles',title='UNEVEN FLOOR / retained rock obstacles and local relief',
                eye=[13.,-8.,render_floor(13.,-8.)+1.15],
                target=[26.,-4.,render_floor(26.,-4.)+.45],focal=22.,lamp_gain=8.,rover=True))
        if config.get('search_objects_revision'):
            search_targets=json.loads((source/config['search_targets_file']).read_text())['targets']
            from isaacsim.core.utils.semantics import add_labels,get_labels
            views=[]
            for index,obj in enumerate(search_targets):
                prim=stage.GetPrimAtPath(obj['prim_path'])
                add_labels(prim,[obj['class_name']],instance_name='class')
                add_labels(prim,[obj['id']],instance_name='object_id')
                semantic_labels=get_labels(prim)
                if semantic_labels.get('class')!=[obj['class_name']] or semantic_labels.get('object_id')!=[obj['id']]:
                    raise ValueError('Search object semantic class/id labels did not persist')
                center=np.asarray(obj['xyz_m'],float);height=float(obj['height_m']);radius=float(obj['footprint_radius_m'])
                preferred=np.asarray(obj.get('preferred_view_direction_local_xy',[1.,-1.]),float)
                preferred/=np.linalg.norm(preferred)
                yaw=np.radians(float(obj['yaw_deg']));c,s=np.cos(yaw),np.sin(yaw)
                direction_xy=np.array([[c,-s],[s,c]])@preferred
                aim=center+[0,0,float(obj.get('camera_target_height_m',height*.5))]
                chosen=None
                for delta in [0.,20.,-20.,40.,-40.,65.,-65.,90.,-90.,135.,-135.,180.]:
                    angle=np.radians(delta);a,b=np.cos(angle),np.sin(angle)
                    direction=np.array([[a,-b],[b,a]])@direction_xy
                    distance=max(3.8,height*2.45,radius*2.8)
                    eye=center+np.r_[direction*distance,height*1.05+.2]
                    try:eye[2]=max(eye[2],render_floor(eye[0],eye[1])+.65)
                    except ValueError:continue
                    ray=eye[None]+np.linspace(0,1,25)[:,None]*(aim-eye)[None]
                    if np.min(evaluate_scene_field(config,ray))<=.12:continue
                    chosen=eye;break
                if chosen is None:raise ValueError('No unobstructed review camera for '+obj['id'])
                views.append(dict(name=f'{index+1:02d}_search_'+obj['id'],
                    title='SEARCH TARGET / '+obj['class_name'].replace('_',' ').upper(),
                    eye=chosen.tolist(),target=aim.tolist(),focal=22.,lamp_gain=.65,rover=False,
                    search_object=True,target_id=obj['id'],seed=int(config['search_layout_seed'])))
        rep.orchestrator.set_capture_on_play(False)
        product=rep.create.render_product(str(cam.GetPath()),(args.width,args.width*9//16))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
        for _ in range(35):app.update()
        records=[]
        for view in views:
            if (out/'REQUEST_STOP').exists():status('requested_stop',captured=len(records));return 0
            status('rendering',view=view['name'],captured=len(records))
            eye=np.asarray(view['eye'],float);target=np.asarray(view['target'],float)
            eye_margin=float(evaluate_scene_field(config,eye[None])[0])
            if eye_margin<=0:raise ValueError('Camera is in rock: '+view['name']+' '+str(eye_margin))
            set_camera(cam,eye,target);cam.CreateFocalLengthAttr(view['focal'])
            UsdGeom.Imageable(rover_prim).CreateVisibilityAttr().Set('inherited' if view['rover'] else 'invisible')
            direction=target-eye;direction/=np.linalg.norm(direction)
            lateral=np.cross([0.,0.,1.],direction);lateral/=np.linalg.norm(lateral)
            lamp_points=[eye-.3*direction+.25*lateral,eye-.3*direction-.25*lateral,
                         eye+9*direction+4*lateral+[0,0,2.5],eye+18*direction-3*lateral+[0,0,2.]]
            local_powers=powers
            if view.get('closeup') or view.get('search_object'):
                lamp_points=[eye+[-.3,-.5,.2],eye+[.2,.7,.1],target+[.4,1.8,1.],target+[-1.,-1.7,.8]]
                local_powers=[p*m for p,m in zip(powers,[1.,.16,.25,.12] if view.get('closeup') else [1.,.6,.4,.3])]
            for i,light in enumerate(lamps):
                light.CreateRadiusAttr((.035 if i<2 else .08) if view.get('closeup') else (.30 if i<2 else .65))
            for light,point in zip(lamps,lamp_points):
                if evaluate_scene_field(config,np.asarray(point)[None])[0]<=0:point=eye+direction*3+[0,0,.6]
                light.GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float,point)))
            gain=float(view['lamp_gain']);meter=[]
            for attempt in range(3):
                for light,power in zip(lamps,local_powers):light.CreateIntensityAttr(power*gain)
                # Extra update flushes the moved camera before the annotated frame.
                for _ in range(2):app.update()
                rep.orchestrator.step(rt_subframes=args.subframes,pause_timeline=True)
                frame=np.asarray(rgb.get_data()).copy()[:,:,:3]
                avg=float(frame.mean());meter.append(dict(mean_rgb=avg,lamp_gain=gain))
                if gain==0 or attempt==2 or 28.<=avg<=125.:break
                gain*=2. if avg<28. else .5
            if frame.shape!=(args.width*9//16,args.width,3) or frame.std()<1.:raise RuntimeError('Invalid RGB '+view['name'])
            Image.fromarray(frame).save(out/(view['name']+'.png'))
            record=dict(view,file=view['name']+'.png',applied_lamp_gain=gain,mean_rgb=float(frame.mean()),
                        std_rgb=float(frame.std()),p99_rgb=float(np.percentile(frame,99)),camera_air_margin_not_clearance_m=eye_margin,
                        metering=meter,lighting='directional Sun; inspection lamps on interior views only')
            records.append(record);dump(out/'views.json',records);print('SHOT',json.dumps(record),flush=True)
        stage.Export(str(out/'review_scene.usdc'))
        rgb.detach([product.path]);product.destroy()
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
        small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
        sheet=Image.new('RGB',(1600,1028),(9,13,18));draw=ImageDraw.Draw(sheet)
        sheet_indices=[0,1,2,3] if config.get('search_objects_revision') else ([2,4,6,5] if len(records)>6 else ([0,2,4,5] if len(records)>5 else [0,2,3,4]))
        for i,idx in enumerate(sheet_indices):
            record=records[idx];x=(i%2)*800;y=(i//2)*500
            im=Image.open(out/record['file']).convert('RGB').resize((800,450),Image.Resampling.LANCZOS)
            sheet.paste(im,(x,y));draw.text((x+12,y+462),record['title'],font=font,fill=(230,236,242))
        draw.text((12,1005),'Actual Isaac Sim RTX / Marius-scale synthetic pit + Zhou-inspired cavity / unmeasured details are authored hypotheses',font=small,fill=(170,185,198))
        sheet.save(out/'contact_sheet.png')
        changed=[path for path,h in hashes.items() if digest(path)!=h]
        if changed:raise RuntimeError('Protected source changed: '+str(changed))
        dump(out/'render_report.json',dict(renderer='Isaac Sim 5.1 / RayTracedLighting',actual_renderer=True,
            source=str(source),source_files_unchanged=True,protected_hashes=hashes,views=records,
            rover_scope='static 1m visual scale model, not dynamics',
            lighting_scope='Sun-inspired directional light and inspection lamps, not photometrically calibrated lunar observation',
            surrounding_terrain_scope='visual procedural extension outside the validated finite geometry patch',
            training_started=False,navigation_evaluated=False,elapsed_seconds=time.monotonic()-started))
        status('complete',captured=len(records),report='render_report.json',contact_sheet='contact_sheet.png',source_files_unchanged=True)
        return 0
    except BaseException as exc:
        dump(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()));status('failed',error=repr(exc));raise
    finally:
        if app is not None:app.close()


if __name__=='__main__':raise SystemExit(main())
