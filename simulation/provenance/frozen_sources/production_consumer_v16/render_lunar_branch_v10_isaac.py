"""Native-resolution Isaac RTX review of the branched cave and detailed props.

Perspective images use the accepted fixed vehicle lights. The final, explicitly
labelled orthographic cutaway hides the ceiling only in the review session;
it is neither a policy observation nor a replacement of the source geometry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    path = Path(path);temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _validated_paths(source):
    """Use real, passed geometry paths for this exact scene, never a centreline."""
    import numpy as np
    found = {}
    candidates = set(source.rglob('obstacle_map_report.json'))
    candidates.update(source.parent.rglob('obstacle_map_report.json'))
    for file in sorted(candidates, key=lambda p: p.stat().st_mtime):
        record = json.loads(file.read_text())
        if record.get('passed') is not True or Path(record.get('scene', '')).resolve() != source:
            continue
        path = np.asarray(record.get('path_xyz_m'), float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 4 or not np.isfinite(path).all():
            continue
        kind = 'branch' if path[-1, 1] > 12. else 'main'
        found[kind] = (path, str(file))
    if set(found) != {'main', 'branch'}:
        raise ValueError('Expected passed main and branch path reports bound to this source scene')
    return found


def _path_pose(path, kind, floor):
    import numpy as np
    index = int(np.argmin(abs(path[:, 0]-26.))) if kind == 'main' else int(np.argmin(
        np.linalg.norm(path[:, :2]-[45., 17.], axis=1)))
    index = min(index, len(path)-3)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    ahead = min(len(path)-1, int(np.searchsorted(arc, arc[index]+12.)))
    direction = path[ahead, :2]-path[index, :2]
    if np.linalg.norm(direction) < .1:
        direction = path[min(index+3, len(path)-1), :2]-path[index, :2]
    direction /= np.linalg.norm(direction)
    xy = path[index, :2]
    eye = np.r_[xy, floor(*xy)+1.5]
    target = eye+np.r_[direction*12., 0.]
    return eye, target, index


def _boulder_rows(source):
    """Conservative actor-camera guards, not a new navigation benchmark."""
    import numpy as np
    rows = []
    for item in json.loads((source/'boulders.json').read_text()):
        center = np.asarray(item['center_xyz_m'], float)
        if center[2] >= 22.:
            continue
        scale = np.asarray(item['scale_xyz_m'], float)
        rows.append(dict(center=center, radius=1.6*float(np.linalg.norm(scale[:2])),
                         z_min=center[2]-1.6*scale[2], z_max=center[2]+1.6*scale[2]))
    return rows


def _slice_below_plane(vertices, faces, cut_z):
    """Exact triangle/plane clipping, deliberately without a fabricated cap.

    Trimesh's slice_mesh_plane imports Shapely even for cap=False; the Isaac
    runtime has no Shapely. Keep inside faces verbatim and clip only crossing
    triangles at their true edge intersections, preserving source winding.
    """
    import numpy as np
    import trimesh
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    z = vertices[faces, 2]
    inside = z <= float(cut_z)
    retained = faces[np.all(inside, axis=1)]
    crossing = faces[np.any(inside, axis=1) & ~np.all(inside, axis=1)]
    added_vertices, added_faces = [], []
    for face in crossing:
        triangle = vertices[face]
        polygon = []
        previous = triangle[-1];previous_inside = previous[2] <= cut_z
        for current in triangle:
            current_inside = current[2] <= cut_z
            if current_inside != previous_inside:
                t = (cut_z-previous[2])/(current[2]-previous[2])
                point = previous+t*(current-previous);point[2] = cut_z
                polygon.append(point)
            if current_inside:
                polygon.append(current)
            previous, previous_inside = current, current_inside
        unique = []
        for point in polygon:
            if not unique or np.linalg.norm(point-unique[-1]) > 1e-10:
                unique.append(point)
        if len(unique) > 1 and np.linalg.norm(unique[-1]-unique[0]) <= 1e-10:
            unique.pop()
        if len(unique) < 3:
            continue
        offset = len(vertices)+len(added_vertices)
        added_vertices.extend(unique)
        for index in range(1, len(unique)-1):
            a, b, c = unique[0], unique[index], unique[index+1]
            if np.linalg.norm(np.cross(b-a, c-a)) > 1e-12:
                added_faces.append([offset, offset+index, offset+index+1])
    all_vertices = np.vstack([vertices, np.asarray(added_vertices).reshape(-1, 3)])
    all_faces = np.vstack([retained, np.asarray(added_faces, np.int64).reshape(-1, 3)])
    if not len(all_faces):
        raise ValueError('No source triangles below the review slice')
    result = trimesh.Trimesh(all_vertices, all_faces, process=False)
    result.remove_unreferenced_vertices();result.merge_vertices(digits_vertex=7)
    return result


def _hero_view(obj, floor, field, boulders, reverse=False):
    import numpy as np
    center = np.asarray(obj['xyz_m'], float)
    height, radius = float(obj['height_m']), float(obj['footprint_radius_m'])
    direction = np.asarray(obj.get('preferred_view_direction_local_xy', [1., -.4]), float)
    direction /= np.linalg.norm(direction)
    yaw = np.radians(float(obj['yaw_deg']))
    c, s = np.cos(yaw), np.sin(yaw)
    direction = np.array([[c, -s], [s, c]])@direction
    if reverse:
        direction = -direction
    aim = center+[0., 0., float(obj.get('camera_target_height_m', height*.55))]
    distance = max(3.0, 1.95*height, 2.7*radius)
    for multiplier in [1., .84, 1.15]:
        for angle in [0., 20., -20., 40., -40., 65., -65., 90., -90., 135., -135., 180.]:
            angle = np.radians(angle);c, s = np.cos(angle), np.sin(angle)
            vector = np.array([[c, -s], [s, c]])@direction
            xy = center[:2]+vector*(distance*multiplier)
            try:
                ground = floor(*xy)
            except ValueError:
                continue
            eye = np.r_[xy, max(ground+1.4, aim[2]+height*.22)]
            ray = eye+np.linspace(0., 1., 39)[:, None]*(aim-eye)
            if np.min(field(ray)) <= .12:
                continue
            occluded = False
            for rock in boulders:
                inside = (np.linalg.norm(ray[:, :2]-rock['center'][:2], axis=1) < rock['radius'])
                inside &= (ray[:, 2] >= rock['z_min']) & (ray[:, 2] <= rock['z_max'])
                if inside[:-3].any():
                    occluded = True;break
            if not occluded:
                return eye, aim
    approach = obj.get('approach_xy_m')
    if approach is not None:
        eye = np.r_[approach, floor(*approach)+max(1.5, height*.7)]
        ray = eye+np.linspace(0., 1., 39)[:, None]*(aim-eye)
        if np.min(field(ray)) > .12:
            return eye, aim
    raise ValueError('No unobstructed review camera for '+obj['id'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--subframes', type=int, default=32)
    parser.add_argument('--experience', type=Path)
    parser.add_argument('--ext-folder', type=Path, action='append', default=[])
    args = parser.parse_args(argv);sys.argv = [sys.argv[0]]
    if args.width < 1280 or args.width % 16 or not 1 <= args.subframes <= 64:
        raise ValueError('Use width>=1280 divisible by16 and subframes1..64; default is native1920x1080')
    width, height = args.width, args.width*9//16
    source = args.source.resolve(strict=True);out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False);started = time.monotonic()
    app = None;rgb = None;product = None
    protected = [source/n for n in ['cave.usdc', 'scene_config.json', 'mesh.npz', 'boulders.json', 'targets.json']]
    protected += [ROOT/'outputs/latest.json', ROOT/'status.json']
    hashes = {str(path): _sha(path) for path in protected if path.is_file()}

    def status(phase, **extra):
        row = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=round(time.monotonic()-started, 3),
                   renderer='Isaac Sim 5.1 / native RTX', native_resolution=[width, height], **extra)
        _write(out/'status.json', row);print(json.dumps(row), flush=True)

    try:
        status('starting_isaac')
        from isaacsim import SimulationApp
        from collect_lunar_trajectory_v11_isaac import experience_path
        import os
        ext_folders = list(args.ext_folder)
        ext_folders += [Path(p) for p in os.environ.get('ISAAC_EXT_FOLDER', '').split(os.pathsep) if p]
        extra_args=['--/app/extensions/registryEnabled=false', '--/renderer/multiGpu/enabled=false',
            '--/rtx/indirectDiffuse/enabled=true', '--/rtx/ambientOcclusion/enabled=true',
            '--/exts/omni.replicator.core/Orchestrator/enabled=true']
        for folder in ext_folders:
            extra_args += ['--ext-folder', str(folder.expanduser().resolve(strict=True))]
        experience = experience_path(args.experience)
        app = SimulationApp(dict(headless=True, renderer='RayTracedLighting', width=width, height=height, anti_aliasing=2,
            multi_gpu=False, active_gpu=0, physics_gpu=0, sync_loads=True,
            extra_args=extra_args), **({'experience': experience} if experience else {}))
        import numpy as np
        import trimesh
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
        from PIL import Image, ImageDraw, ImageFont
        from cave.scene import camera, sphere_light, mesh
        from cave.lunar_gravel_v5 import _FloorProjector
        from cave.lunar_visual_v4 import _continuous_terrain_color, _boulder_mesh, _author_mesh
        from cave.lunar_gravel_material_v5 import set_gravel_primvars
        from build_lunar_cave_v4_cpu import evaluate_scene_field

        config = json.loads((source/'scene_config.json').read_text())
        objects = json.loads((source/'targets.json').read_text())['targets']
        by_id = {item['id']: item for item in objects}
        if not {'astronaut', 'lunar_base'}.issubset(by_id):
            raise ValueError('Detailed astronaut and lunar_base must exist in targets.json')
        paths = _validated_paths(source)
        with np.load(source/'mesh.npz') as data:
            vertices, faces = np.array(data['vertices']), np.array(data['faces'])
        projector = _FloorProjector(vertices, faces)

        def floor(x, y):
            values, _, _, _ = projector.project(np.asarray([[x, y]], float))
            if not np.isfinite(values[0]):
                raise ValueError('No actual floor support at '+str((x, y)))
            return float(values[0])

        field = lambda points: np.asarray(evaluate_scene_field(config, points), float)
        status('loading_source')
        context = omni.usd.get_context()
        if not context.open_stage(str(source/'cave.usdc')):
            raise RuntimeError('Cannot open source cave')
        stage = context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath) and value.resolvedPath:
                    attr.Set(Sdf.AssetPath(value.resolvedPath))
        settings = carb.settings.get_settings()
        # Replicator requires limitedOps=false to select FXAA instead of DLSS.
        native_settings = {
            '/rtx-transient/post/aa/limitedOps': False,
            '/rtx/post/aa/op': 2,
            '/rtx/post/dlss/enabled': False,
            '/rtx/post/scaling/staticRatio': 1.0,
            '/rtx-transient/dlssg/enabled': False,
            '/rtx/post/motionblur/enabled': False,
        }
        settings.set('/rtx/sceneDb/ambientLightIntensity', 0.)
        prefix = '/World/RevisionReviewV10'
        UsdGeom.Xform.Define(stage, prefix)
        sun = UsdLux.DistantLight.Define(stage, prefix+'/Sun')
        sun.CreateIntensityAttr(3500.);sun.CreateAngleAttr(.53)
        sun.CreateColorAttr(Gf.Vec3f(1., .99, .97))
        sun_rotation = sun.AddRotateXYZOp();sun_rotation.Set(Gf.Vec3f(24., -24., -25.))
        dome = UsdLux.DomeLight.Define(stage, prefix+'/GentleFill')
        dome.CreateIntensityAttr(45.);dome.CreateColorAttr(Gf.Vec3f(.70, .75, .80))
        cam = camera(stage, prefix+'/Camera', (0, 0, 3), (1, 0, 3), focal=32.)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.03, 4000.))
        lamps = []
        for name, intensity, radius in [('Headlamp', 24000., .28), ('WorkLeft', 8000., .45), ('WorkRight', 8000., .45)]:
            light = sphere_light(stage, prefix+'/'+name, (0, 0, 3), (.94, .97, 1.), intensity, radius)
            light.CreateNormalizeAttr(True);light.CreateExposureAttr(6.)
            lamps.append((light, intensity))
        boulders = _boulder_rows(source)
        views = []
        for name, title, reverse in [('01_astronaut_detail', 'ASTRONAUT / detailed suit and equipment', False),
                                     ('02_astronaut_backpack', 'ASTRONAUT / life-support backpack and joints', True)]:
            eye, aim = _hero_view(by_id['astronaut'], floor, field, boulders, reverse=reverse)
            views.append(dict(name=name, title=title, eye=eye.tolist(), target=aim.tolist(), focal=32., scope='asset review'))
        eye, aim = _hero_view(by_id['lunar_base'], floor, field, boulders)
        views.append(dict(name='03_habitat_detail', title='LUNAR HABITAT / airlock and external systems',
                          eye=eye.tolist(), target=aim.tolist(), focal=32., scope='asset review'))
        for kind, name, title in [('main', '04_main_fork_obstacles', 'MAIN CAVE / fork and large obstacle groups'),
                                  ('branch', '05_side_branch_obstacles', 'SIDE BRANCH / usable offshoot and obstacles')]:
            eye, aim, index = _path_pose(paths[kind][0], kind, floor)
            views.append(dict(name=name, title=title, eye=eye.tolist(), target=aim.tolist(), focal=18.,
                              path_report=paths[kind][1], path_pose_index=index, scope='geometric-path review; no policy'))

        # Prepare a genuine slice of the source mesh, initially invisible.
        cut_z = 3.5;cut_root = UsdGeom.Xform.Define(stage, prefix+'/Cutaway')
        cut_root.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        cut = _slice_below_plane(vertices, faces, cut_z)
        if not len(cut.faces) or np.max(cut.vertices[:, 2]) > cut_z+1e-5:
            raise ValueError('Actual source ceiling slice failed')
        regolith = UsdShade.Material.Get(stage, '/World/Materials/Regolith')
        cut_mesh = mesh(stage, prefix+'/Cutaway/TerrainBelow35m', cut.vertices, cut.faces,
                        regolith, collision=False, tile=2.5)
        normals = np.asarray(cut.vertex_normals, np.float32)
        cut_mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
        cut_mesh.CreateDisplayColorPrimvar('vertex').Set(Vt.Vec3fArray.FromNumpy(
            _continuous_terrain_color(cut.vertices, normals)))
        set_gravel_primvars(cut_mesh, texture_resolution='4k')
        floor_boulders = [item for item in json.loads((source/'boulders.json').read_text())
                          if float(item['center_xyz_m'][2]) < 22.]
        if floor_boulders:
            v, f, n = _boulder_mesh(floor_boulders, hard=True, detail_revision=10)
            rock_material = UsdShade.Material.Get(stage, '/World/Materials/Basalt')
            blocks = _author_mesh(stage, prefix+'/Cutaway/ActualFloorBoulders', v, f, n, rock_material,
                                  tile=2.4, normal_interpolation='faceVarying')
            # Basalt has the identical source fallback and the authored UVs.
            # Preserve hard face-varying normals; terrain-only primvar helpers
            # require a vertex normal for every point and are not used here.
        # Ground-height screening prevents fragments formerly supported by a
        # removed upper ledge from floating in this ceiling-hidden review.
        instancer = UsdGeom.PointInstancer.Get(stage, '/World/SurfaceDetail/Gravel')
        cut_hidden_ids = []
        if instancer:
            positions = np.asarray(instancer.GetPositionsAttr().Get(), float)
            matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(instancer.GetPrim()), float)
            world = np.column_stack([positions, np.ones(len(positions))])@matrix
            supported, _, _, _ = projector.project(world[:, :2])
            ids = instancer.GetIdsAttr().Get()
            ids = np.arange(len(positions), dtype=np.int64) if ids is None else np.asarray(ids, np.int64)
            cut_hidden_ids = ids[(~np.isfinite(supported)) | (supported > cut_z-.10)].tolist()
        views.append(dict(name='06_orthographic_cutaway', title='ORTHOGRAPHIC MAP / ceiling hidden for review',
            eye=[16., 14., 120.], target=[16., 14., 0.], up=[0., 1., 0.], projection='orthographic',
            horizontal_aperture_usd=1660., scope='source mesh sliced at Z3.5m; not a policy observation'))

        rep.orchestrator.set_capture_on_play(False)
        product = rep.create.render_product(str(cam.GetPath()), (width, height))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
        for _ in range(35):app.update()
        records = []
        for view in views:
            if (out/'REQUEST_STOP').exists():
                status('requested_stop', captured=len(records));return 0
            status('rendering', view=view['name'], captured=len(records))
            eye, aim = np.asarray(view['eye'], float), np.asarray(view['target'], float)
            overview = view.get('projection') == 'orthographic'
            if overview:
                cave = stage.GetPrimAtPath('/World/Cave')
                if not cave:
                    raise ValueError('Expected /World/Cave to hide the full roof and original merged rocks')
                UsdGeom.Imageable(cave).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                cut_root.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
                if instancer and cut_hidden_ids:
                    original = instancer.GetInvisibleIdsAttr().Get() or []
                    instancer.CreateInvisibleIdsAttr(Vt.Int64Array(sorted(set(map(int, original)) | set(cut_hidden_ids))))
                cam.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
                cam.CreateHorizontalApertureAttr(1660.);cam.CreateVerticalApertureAttr(1660.*height/width)
                sun_rotation.Set(Gf.Vec3f(0., 0., 0.));sun.CreateIntensityAttr(1800.)
                dome.CreateIntensityAttr(80.)
                for light, _ in lamps:light.CreateIntensityAttr(0.)
                eye_margin = None
            else:
                eye_margin = float(field(eye[None])[0])
                if eye_margin <= .05:
                    raise ValueError('Review camera is in rock: '+view['name'])
                cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
                cam.CreateHorizontalApertureAttr(36.);cam.CreateVerticalApertureAttr(36.*height/width)
                cam.CreateFocalLengthAttr(view['focal'])
                forward = aim-eye;forward /= np.linalg.norm(forward)
                left = np.cross([0., 0., 1.], forward);left /= np.linalg.norm(left)
                positions = [eye+forward*.12, eye+forward*.22-left*.45+[0, 0, -.25],
                             eye+forward*.22+left*.45+[0, 0, -.25]]
                for (light, intensity), position in zip(lamps, positions):
                    light.GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float, position)))
                    light.CreateIntensityAttr(intensity)
            up = view.get('up', [0., 0., 1.])
            transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, aim)),
                                               Gf.Vec3d(*map(float, up))).GetInverse()
            cam.GetOrderedXformOps()[0].Set(transform)
            for _ in range(2):app.update()
            for key, value in native_settings.items():
                settings.set(key, value)
            rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=True)
            renderer_readback = {key: settings.get(key) for key in native_settings}
            if renderer_readback != native_settings:
                raise RuntimeError('Renderer settings changed during capture: '+view['name'])
            frame = np.asarray(rgb.get_data()).copy()[:, :, :3]
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8 or frame.std() < 1.:
                raise ValueError('Invalid native RGB for '+view['name'])
            Image.fromarray(frame).save(out/(view['name']+'.png'))
            record = dict(view, file=view['name']+'.png', native_resolution=[width, height],
                renderer_settings=renderer_readback,
                camera_air_margin_not_clearance=eye_margin, mean_rgb=float(frame.mean()),
                std_rgb=float(frame.std()), bright_pixel_fraction=float(np.mean(np.max(frame, axis=2) >= 250)),
                lighting='overhead review illumination' if overview else 'fixed vehicle headlamp24000 + two8000 work lights, exposure6, dome45')
            records.append(record);_write(out/'views.json', records)
            print('SHOT '+json.dumps(record), flush=True)
        if not stage.Export(str(out/'review_scene.usdc')):
            raise RuntimeError('Review USD export failed')
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
        sheet = Image.new('RGB', (1920, 1220), (11, 16, 21));draw = ImageDraw.Draw(sheet)
        for i, index in enumerate([0, 2, 3, 5]):
            row = records[index];x, y = (i % 2)*960, (i // 2)*590
            image = Image.open(out/row['file']).convert('RGB').resize((960, 540), Image.Resampling.LANCZOS)
            sheet.paste(image, (x, y));draw.text((x+12, y+551), row['title'], font=font, fill=(232, 239, 245))
        draw.text((12, 1190), 'Actual native Isaac RTX / review views, not policy observations / orthographic map hides only the ceiling in this review session',
                  font=small, fill=(170, 189, 204))
        sheet.save(out/'contact_sheet.png')
        if any(_sha(path) != value for path, value in hashes.items()):
            raise RuntimeError('A protected source/default file changed')
        _write(out/'render_report.json', dict(passed=True, actual_renderer=True,
            renderer='Isaac Sim 5.1 / RayTracedLighting', source=str(source), protected_sha256=hashes,
            source_files_unchanged=True, native_resolution=[width, height], source_rgb_upscaled=False,
            antialiasing='FXAA, DLSS disabled and staticRatio 1 confirmed by setting readback',
            internal_rtx_resolution_measured=False,
            views=records, validated_path_reports={kind: file for kind, (_, file) in paths.items()},
            vehicle_lights=dict(main_intensity=24000., work_intensity_each=8000., exposure=6., dome=45., target_independent=True),
            cutaway=dict(ceiling_hidden_in_review_only=True, source_mesh_true_plane_slice=True, cut_z_m=cut_z,
                plane_slice_method='exact triangle-edge intersections, no cap; no Shapely dependency',
                source_cave_subtree_hidden=True, replacement_floor_boulders=len(floor_boulders),
                upper_ledge_gravel_hidden_ids=len(cut_hidden_ids), horizontal_orthographic_extent_m=166.,
                vertical_orthographic_extent_m=166.*height/width, not_policy_observation=True),
            navigation_policy_evaluated=False, physics_evaluated=False, training_started=False,
            elapsed_seconds=time.monotonic()-started))
        status('complete', passed=True, captured=len(records), contact_sheet='contact_sheet.png', source_files_unchanged=True)
        return 0
    except BaseException as exc:
        _write(out/'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        status('failed', error=repr(exc));raise
    finally:
        if rgb is not None and product is not None:
            rgb.detach([product.path]);product.destroy()
        if app is not None:app.close()


if __name__ == '__main__':
    raise SystemExit(main())
