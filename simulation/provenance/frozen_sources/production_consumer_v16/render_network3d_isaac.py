"""Actual Isaac Sim RTX stills of an existing network; source USD stays read-only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Candidate directory with cave.usdc and network.json')
    parser.add_argument('--output', type=Path, required=True, help='NEW render directory')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--subframes', type=int, default=32)
    parser.add_argument('--lamp-gain', type=float, default=64., help='Inspection illumination multiplier')
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    source = args.source.resolve(strict=True)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protected = [source/'cave.usdc', source/'network.json', ROOT/'outputs/latest.json', ROOT/'status.json']
    baseline = {str(p): sha(p) for p in protected}
    app = None

    def status(phase, **kw):
        value = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=round(time.monotonic()-started, 2),
                     renderer='Isaac Sim 5.1 / RayTracedLighting', output=str(output), **kw)
        write_json(output/'status.json', value)
        print('NETWORK_RENDER_STATUS', json.dumps(value), flush=True)

    try:
        status('starting_isaac')
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer='RayTracedLighting', width=args.width,
            height=args.width*9//16, multi_gpu=False, active_gpu=0, physics_gpu=0, sync_loads=True,
            extra_args=['--ext-folder', '/home/charles/.local/share/ov/data/exts/v2',
                        '--/app/extensions/registryEnabled=false', '--/renderer/multiGpu/enabled=false',
                        '--/rtx/indirectDiffuse/enabled=true', '--/rtx/ambientOcclusion/enabled=true',
                        '--/exts/omni.replicator.core/Orchestrator/enabled=true']),
            experience='/home/charles/ISAAC_SIM/IsaacLab/apps/isaaclab.python.headless.rendering.kit')
        import numpy as np
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt
        from PIL import Image, ImageDraw, ImageFont
        from cave.scene import camera, set_camera, sphere_light
        from cave.props import build_rover
        from cave.network3d import evaluate_field
        from cave.network3d_render_utils import smooth_height_cutaway

        status('loading_existing_candidate')
        context = omni.usd.get_context()
        if not context.open_stage(str(source/'cave.usdc')):
            raise RuntimeError('Isaac failed to open the validated candidate USD')
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        # Resolve source-relative PBR assets in the session before flattening review stages elsewhere.
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath) and value.resolvedPath:
                    attr.Set(Sdf.AssetPath(value.resolvedPath))
        network = json.loads((source/'network.json').read_text())
        settings = carb.settings.get_settings()
        settings.set('/rtx/post/dlss/execMode', 2)
        settings.set('/rtx/raytracing/fractionalCutoutOpacity', True)
        settings.set('/rtx/sceneDb/ambientLightIntensity', 0.)
        preview = UsdGeom.Xform.Define(stage, '/World/InspectionPreview')
        preview.GetPrim().CreateAttribute('cave:scope', Sdf.ValueTypeNames.String, custom=True).Set(
            'temporary inspection lighting and visual scale rover; not a navigation evaluation')
        cam = camera(stage, '/World/InspectionPreview/Camera', (35,-2,-145), (70,2,-141), focal=20.)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.05, 2500.))
        lights = []
        base_powers = [1200.,1200.,1800.,1500.]
        for i, (color, power, exposure, radius) in enumerate([
            ((.94,.97,1.),1200.,6.,.13), ((.94,.97,1.),1200.,6.,.13),
            ((1.,.91,.78),1800.,6.,.35), ((.74,.85,1.),1500.,6.,.35)]):
            lamp = sphere_light(stage, f'/World/InspectionPreview/Lights/Lamp{i}', (0,0,-140), color, power*args.lamp_gain, radius)
            lamp.CreateNormalizeAttr(True)
            lamp.CreateExposureAttr(exposure)
            lights.append(lamp)
        rover = build_rover(stage, '/World/InspectionPreview/ScaleRover')
        rover_prim = stage.GetPrimAtPath(rover['prim_path'])
        rover_xform = UsdGeom.Xformable(rover_prim)
        rover_xform.ClearXformOpOrder()
        rover_xform.AddTranslateOp().Set(Gf.Vec3d(24.,-24.*4/35.,-146.-24./35.+.02))
        rover_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.,1.626,-6.52))

        views = [
            dict(name='01_main_with_scale_rover', title='MAIN TUBE / 1 m rover for scale',
                 eye=[18,-8,-144.5], target=[32,0,-142.5], show_rover=True),
            dict(name='02_asymmetric_arch', title='ASYMMETRIC ARCH / full uncut geometry',
                 eye=[35,-2,-145], target=[70,2,-141], show_rover=False),
            dict(name='03_rejoining_junction', title='JUNCTION / main tube and rejoining branch',
                 eye=[94,4,-143], target=[110,17,-142], show_rover=False),
            dict(name='04_lower_level', title='LOWER LEVEL / descending passage',
                 eye=[278,21,-182], target=[263.5,5.7625,-184], show_rover=False, lamp_gain=8.),
        ]
        rep.orchestrator.set_capture_on_play(False)
        product = rep.create.render_product(str(cam.GetPath()), (args.width,args.width*9//16))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach([product])
        for _ in range(35): app.update()
        records = []

        def capture(view, lighting, cutaway=False):
            eye, target = np.asarray(view['eye'],float), np.asarray(view['target'],float)
            set_camera(cam, eye, target)
            metering=[]
            for attempt in range(3):
                rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=True)
                frame = np.asarray(rgb.get_data()).copy()[:,:,:3]
                mean=float(frame.mean())
                metering.append(dict(mean_rgb=mean,lamp_gain=view.get('lamp_gain')))
                if cutaway or attempt==2 or 30. <= mean <= 135.:break
                factor=4. if mean<30. else .25
                view['lamp_gain']*=factor
                for lamp,power in zip(lights,base_powers):
                    lamp.CreateIntensityAttr(power*view['lamp_gain'])
            if frame.shape != (args.width*9//16,args.width,3):
                raise RuntimeError('RGB shape is incorrect: '+str(frame.shape))
            Image.fromarray(frame).save(output/(view['name']+'.png'))
            record = dict(view, file=view['name']+'.png', lighting=lighting,
                          upper_shell_cutaway=cutaway, resolution=[args.width,args.width*9//16],
                          mean_rgb=float(frame.mean()), std_rgb=float(frame.std()),
                          p99_rgb=float(np.percentile(frame,99)),
                          metering=metering,
                          black_pixel_fraction=float(np.mean(np.max(frame,axis=-1)<4)))
            records.append(record)
            write_json(output/'views.json', records)
            print('SHOT',json.dumps(record),flush=True)
            if float(frame.std()) < 1. or np.max(frame)<10:
                raise RuntimeError('Capture is blank or almost uniform: '+view['name'])

        for view in views:
            if (output/'REQUEST_STOP').exists():
                status('requested_stop', captured=len(records));return 0
            status('rendering_interior', view=view['name'], captured=len(records))
            eye,target = np.asarray(view['eye'],float),np.asarray(view['target'],float)
            direction = target-eye;direction/=np.linalg.norm(direction)
            lateral = np.cross([0.,0.,1.],direction);lateral/=np.linalg.norm(lateral)
            points = [eye-.3*direction+.23*lateral, eye-.3*direction-.23*lateral,
                      eye+10.*direction+4.*lateral+[0,0,3.],eye+23.*direction-4.*lateral+[0,0,3.]]
            gain=float(view.get('lamp_gain',args.lamp_gain))
            view['lamp_gain']=gain
            for lamp,point,power in zip(lights,points,base_powers):
                lamp.CreateIntensityAttr(power*gain)
                if evaluate_field(network,np.asarray([point]))[0] <= 0:
                    point = eye+direction*4.+[0,0,1.]
                lamp.GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float,point)))
            UsdGeom.Imageable(rover_prim).CreateVisibilityAttr().Set('inherited' if view['show_rover'] else 'invisible')
            capture(view,'temporary white inspection lamps; vacuum / no ambient atmosphere')
        stage.Export(str(output/'inspection_full_geometry.usdc'))

        status('rendering_upper_shell_cutaway', captured=len(records))
        for lamp in lights: lamp.CreateIntensityAttr(0.)
        UsdGeom.Imageable(rover_prim).CreateVisibilityAttr().Set('invisible')
        shell = UsdGeom.Mesh.Get(stage,'/World/Cave/AirBoundary')
        vertices = np.asarray(shell.GetPointsAttr().Get(),float)
        faces = np.asarray(shell.GetFaceVertexIndicesAttr().Get(),np.int32).reshape(-1,3)
        uv = UsdGeom.PrimvarsAPI(shell).GetPrimvar('st')
        values = np.asarray(uv.Get(),np.float32).reshape(-1,3,2)
        normals=np.asarray(shell.GetNormalsAttr().Get(),np.float32)
        cut=smooth_height_cutaway(network,vertices,faces,values,normals,height_m=6.)
        shell.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(cut['vertices'],np.float32)))
        shell.GetFaceVertexCountsAttr().Set([3]*len(cut['faces']))
        shell.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.asarray(cut['faces'],np.int32).ravel()))
        shell.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(cut['normals'],np.float32)))
        uv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(cut['uv'],np.float32).reshape(-1,2)))
        for i, (rotation,intensity) in enumerate([((25,-35,-20),2800.),((-30,20,135),1800.)]):
            sun=UsdLux.DistantLight.Define(stage,f'/World/InspectionPreview/StudioKey{i}')
            sun.CreateIntensityAttr(intensity);sun.CreateAngleAttr(8.)
            sun.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        dome=UsdLux.DomeLight.Define(stage,'/World/InspectionPreview/StudioFill')
        dome.CreateIntensityAttr(350.);dome.CreateColorAttr(Gf.Vec3f(.63,.72,.85))
        overview = dict(name='05_upper_shell_cutaway', title='UPPER SHELL CUTAWAY / display variant',
                        eye=[465,-345,195], target=[170,25,-159], show_rover=False)
        cam.CreateFocalLengthAttr(35.)
        capture(overview,'studio illumination for geometry inspection; not lunar ambient lighting',True)
        stage.Export(str(output/'inspection_cutaway.usdc'))
        rgb.detach([product.path]);product.destroy()
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
        sheet = Image.new('RGB',(1280,820),(9,13,18));draw=ImageDraw.Draw(sheet)
        for i,record in enumerate(records[:4]):
            x=(i%2)*640;y=(i//2)*405
            image=Image.open(output/record['file']).convert('RGB').resize((640,360),Image.Resampling.LANCZOS)
            sheet.paste(image,(x,y));draw.text((x+12,y+366),record['title'],font=font,fill=(230,235,240))
        draw.text((12,806),'Actual Isaac Sim 5.1 RTX renders / synthetic lunar cave candidate / temporary inspection lamps',font=small,fill=(170,185,200))
        sheet.save(output/'contact_sheet.png')
        changed = [p for p,h in baseline.items() if sha(p)!=h]
        if changed:raise RuntimeError('Protected source files changed: '+repr(changed))
        report = dict(renderer='Isaac Sim 5.1 / RayTracedLighting', actual_renderer=True,
            original_candidate=str(source), original_source_sha256=baseline[str(source/'cave.usdc')],
            source_files_unchanged=True, protected_hashes=baseline, views=records,
            original_triangle_count=len(faces),cutaway_triangle_count=len(cut['faces']),
            cutaway_definition='Clip interpolated vertex height at 6m above nearest 3D local floor; session-layer display only',
            cutaway_geometry_report=cut['report'],
            training_started=False,navigation_evaluated=False,
            rover_scope='static visual scale shell, approximately 1m tall; no dynamics',
            elapsed_seconds=time.monotonic()-started)
        write_json(output/'render_report.json',report)
        status('complete',captured=len(records),contact_sheet='contact_sheet.png',
               report='render_report.json',source_files_unchanged=True,training_started=False)
        return 0
    except BaseException as exc:
        write_json(output/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc));raise
    finally:
        if app is not None:app.close()


if __name__=='__main__':
    raise SystemExit(main())
