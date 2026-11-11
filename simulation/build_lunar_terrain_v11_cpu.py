"""Build a portable V11 fork terrain from raw configuration, without Isaac RTX.

The optional layout correction moves one whole, measured gravel object next to
an existing collapse pile. Counts, shape and all terrain remain unchanged. A
fresh collider package and both full vehicle-footprint checks are mandatory.
REQUEST_STOP is cooperative at stage boundaries; a running CPU stage finishes.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
FINGERPRINT_FILES = [
    'build_lunar_cave_v4_cpu.py', 'build_lunar_surface_detail_v5_cpu.py',
    'build_lunar_obstacle_scene_v7_cpu.py', 'validate_lunar_obstacle_map_v7.py',
    'cave/lunar_junction_v11.py', 'cave/lunar_branch_v10.py',
    'cave/lunar_ground_v7.py', 'cave/lunar_pit_v4.py',
    'cave/lunar_morphology_v4.py', 'cave/lunar_visual_v4.py',
    'cave/lunar_gravel_v5.py', 'cave/lunar_gravel_material_v5.py',
    'cave/lunar_obstacles_v7.py', 'assets/ASSET_SOURCES.json',
    'assets/GRAVEL_V5_SOURCES.json']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def link_tree(source, output):
    """Copy package paths, hardlinking immutable payloads on the same volume."""
    def link_or_copy(a, b):
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)
    shutil.copytree(source, output, copy_function=link_or_copy)


def propose_one_object_layout(source, grid_path, main_report, branch_goal):
    """Find a measured single-object barrier; never erase a corridor of rocks."""
    import numpy as np
    from scipy.ndimage import binary_dilation, label
    from cave.lunar_gravel_v5 import _FloorProjector
    from validate_lunar_obstacle_map_v7 import _read_obstacles, _obstacle_masks, _cell, _astar
    report = read(main_report)
    with np.load(grid_path) as data:
        x, y, height, slope, terrain, free = [data[k] for k in
            ['x', 'y', 'height', 'slope_deg', 'terrain_free', 'free']]
    start = tuple(report['start']['grid_index'])
    goals = [_cell(report['goal']['declared_xy_m'], x, y), _cell(branch_goal, x, y)]
    if not free[start]:
        raise ValueError('Declared start is blocked; no automatic endpoint relocation')
    labels, _ = label(free)
    reached = labels == labels[start]
    boundary = binary_dilation(reached, iterations=2)
    obstacles, _ = _read_obstacles(source)
    inflation = report['vehicle_inflation_radius_m']
    candidates = []
    # Only an actual collidable gravel entity can be corrected automatically.
    # A larger structural-layout problem is reported rather than silently cut.
    for item in obstacles:
        if item['kind'] != 'gravel':
            continue
        _, mask, _ = _obstacle_masks([item], x, y, inflation)
        if not (mask & boundary).any():
            continue
        others = [o for o in obstacles if o['id'] != item['id']]
        _, blocked, _ = _obstacle_masks(others, x, y, inflation)
        possible = terrain & ~blocked
        joined, _ = label(possible)
        region = joined[start]
        if region and all(g is not None and joined[g] == region for g in goals):
            candidates.append((int((joined == region).sum()-reached.sum()), item))
    candidates.sort(key=lambda pair: (-pair[0], pair[1]['id']))
    if not candidates:
        raise ValueError('No single measured gravel barrier connects both declared goals; inspect baseline maps')
    with np.load(source/'mesh.npz') as data:
        projector = _FloorProjector(data['vertices'], data['faces'])
    trials = []
    for gain, item in candidates:
        center = np.asarray(item['center_xy_m'], float)
        piles = sorted((o for o in obstacles if o['kind'] == 'macro'),
            key=lambda o: float(np.linalg.norm(np.asarray(o['center_xy_m'])-center)))
        # Positions lie at the shoulder of existing piles, not along a path.
        for pile in piles[:5]:
            pc = np.asarray(pile['center_xy_m'], float)
            for direction in [[1., .45], [-1., .45], [1., -.45], [-1., -.45]]:
                vector = np.asarray(direction, float)
                vector /= np.linalg.norm(vector)
                destination = pc + vector*(pile['xy_radius_m']*.9+item['xy_radius_m'])
                floor, normal, face, _ = projector.project(destination[None])
                if face[0] < 0 or not np.isfinite(floor[0]) or normal[0, 2] < .9:
                    continue
                modified = [dict(o, center_xy_m=destination.tolist()) if o['id'] == item['id'] else o for o in obstacles]
                _, blocked, _ = _obstacle_masks(modified, x, y, inflation)
                possible = terrain & ~blocked
                joined, _ = label(possible)
                region = joined[start]
                if not region or not all(g is not None and joined[g] == region for g in goals):
                    continue
                found = []
                for goal in goals:
                    path, _ = _astar(possible, height, slope, start, goal,
                        report['settings']['grid_resolution_m'], report['settings'])
                    found.append(path is not None)
                trials.append(dict(id=item['id'], destination_xy_m=destination.tolist(), path_found=found))
                if all(found):
                    return dict(schema_version='measured_obstacle_relocation/11.0',
                        gravel_id=int(item['id'].split('_')[1]),
                        source_index=int(item['source_index']), before_obstacle=item,
                        destination_xy_m=destination.tolist(), neighboring_pile_id=pile['id'],
                        diagnostic_connected_cell_gain=gain, trials=trials,
                        removed_obstacle_count=0, macro_objects_moved=0,
                        final_validation_pending=True)
    raise ValueError('Single-object bridge found, but no tested supported pile-side relocation retains both paths')


def relocate_surface(source, reference, proposal, output):
    """Translate one real USD instance; leave its geometry and terrain intact."""
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, Vt
    from cave.lunar_gravel_v5 import _FloorProjector, INSTANCER_PATH
    from build_lunar_surface_detail_v5_cpu import _absolute_assets, _collect_assets, _set_relative_assets
    protected = {n: sha(source/n) for n in
        ['cave.usdc', 'scene_config.json', 'mesh.npz', 'boulders.json', 'geometry_report.json']}
    with np.load(source/'mesh.npz') as data:
        projector = _FloorProjector(data['vertices'], data['faces'])
    xy = np.asarray(proposal['destination_xy_m'], float)
    floor, normal, face, _ = projector.project(xy[None])
    if face[0] < 0 or not np.isfinite(floor[0]):
        raise ValueError('Chosen relocation has no actual mesh floor')
    stage = Usd.Stage.Open(str(source/'cave.usdc'))
    stage.SetEditTarget(stage.GetSessionLayer())
    _absolute_assets(stage, Sdf)
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(INSTANCER_PATH))
    positions = np.asarray(instancer.GetPositionsAttr().Get(), float).copy()
    before = positions.copy()
    ids = np.asarray(instancer.GetIdsAttr().Get(), np.int64)
    rows = np.flatnonzero(ids == proposal['gravel_id'])
    if len(rows) != 1 or int(rows[0]) != proposal['source_index']:
        raise ValueError('Current measured gravel ID/index binding changed')
    index = int(rows[0])
    item = next(o for o in read(reference/'obstacle_report.json')['obstacles']
        if int(o['instance_id']) == proposal['gravel_id'])
    if not np.allclose(item['center_xyz_m'][:2], proposal['before_obstacle']['center_xy_m'], atol=1e-8):
        raise ValueError('Current actual entity position no longer matches diagnostic')
    offset = float(item['z_min_m'])-positions[index, 2]
    thickness = float(item['z_max_m'])-float(item['z_min_m'])
    positions[index] = [*xy, float(floor[0]-offset-min(.02, .2*thickness))]
    if np.count_nonzero(np.any(before != positions, axis=1)) != 1:
        raise ValueError('Relocation must translate exactly one actual object')
    instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
    output.mkdir(parents=True, exist_ok=False)
    for name in ['mesh.npz', 'boulders.json', 'ASSET_LICENSES.json', 'GRAVEL_V5_SOURCES.json',
                 'BASE_ASSET_SOURCES.json', 'surface_detail_report.json']:
        if (source/name).is_file():
            try:
                os.link(source/name, output/name)
            except OSError:
                shutil.copy2(source/name, output/name)
    config = read(source/'scene_config.json')
    config['obstacle_layout_revision_v11'] = 'one_measured_gravel_repositioned_counts_and_shapes_preserved'
    write(output/'scene_config.json', config)
    mapping, assets = _collect_assets(stage, output, Sdf)
    _set_relative_assets(stage, mapping, Sdf)
    if not stage.Export(str(output/'cave.usdc')):
        raise RuntimeError('Relocated USD export failed')
    reopened = Usd.Stage.Open(str(output/'cave.usdc'))
    _set_relative_assets(reopened, mapping, Sdf)
    reopened.GetRootLayer().Save()
    del reopened
    reopened = Usd.Stage.Open(str(output/'cave.usdc'))
    exported = UsdGeom.PointInstancer(reopened.GetPrimAtPath(INSTANCER_PATH))
    for name in ['ProtoIndices', 'Scales', 'Orientations', 'Ids']:
        if list(getattr(exported, 'Get'+name+'Attr')().Get()) != list(getattr(instancer, 'Get'+name+'Attr')().Get()):
            raise ValueError('Gravel property changed: '+name)
    if len(exported.GetPositionsAttr().Get()) != len(before):
        raise ValueError('Gravel instance count changed')
    if any(sha(source/n) != digest for n, digest in protected.items()):
        raise ValueError('Source package was modified')
    report = dict(passed=True, source_sha256=protected, source_unchanged=True,
        terrain_mesh_unchanged=True, macro_objects_moved=0, boulder_count=len(read(source/'boulders.json')),
        gravel_count=len(before), gravel_id=proposal['gravel_id'],
        gravel_position_before=before[index].tolist(), gravel_position_after=positions[index].tolist(),
        dimensions_or_orientations_changed=False, removed_obstacle_count=0,
        proposal=proposal, texture_assets=assets, final_path_validation_pending=True, GPU_work=False)
    write(output/'relocation_report.json', report)
    geometry = read(source/'geometry_report.json')
    geometry['obstacle_relocation_v11'] = report
    write(output/'geometry_report.json', geometry)
    write(output/'status.json', dict(phase='complete', passed=True, path_validated=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/lunar_fork_v11/terrain_final')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--surface-config', type=Path)
    parser.add_argument('--spacing', type=float, default=.35)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '-1'):
        raise RuntimeError('CPU build requires CUDA_VISIBLE_DEVICES empty')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output = args.output.expanduser().absolute()
    work = (args.work_dir or output.parent/'terrain_build').expanduser().absolute()
    if output == work or output.is_relative_to(work) or work.is_relative_to(output):
        raise ValueError('--output and --work-dir must be disjoint')
    if (work.exists() or output.exists()) and not args.resume:
        raise FileExistsError('Existing build requires explicit --resume; no overwrite/retry is automatic')
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def status(phase, **extra):
        write(work/'status.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.monotonic()-started, 3), GPU_work=False, **extra))
        print(json.dumps(dict(phase=phase, **extra)), flush=True)

    def stop():
        flags = [output/'REQUEST_STOP', work/'REQUEST_STOP', output.parent/'REQUEST_STOP']
        if any(p.exists() for p in flags):
            status('requested_stop', passed=False)
            raise InterruptedError('REQUEST_STOP found at a CPU stage boundary')

    def run_stage(name, script, arguments, destination):
        stop()
        if destination.exists():
            saved = destination/'status.json'
            if args.resume and saved.is_file() and read(saved).get('phase') == 'complete' and read(saved).get('passed') is True:
                status(name+'_reused')
                return
            raise FileExistsError('Incomplete or failed stage requires inspection: '+str(destination))
        status(name)
        with (work/(name+'.log')).open('w') as log:
            result = subprocess.run([sys.executable, str(ROOT/script), *map(str, arguments)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; inspect {work/(name+".log")}')
        if read(destination/'status.json').get('passed') is not True:
            raise RuntimeError(name+' did not record a passing completed package')

    def validation(scene, destination, branch=False):
        stop()
        saved = destination/'obstacle_map_report.json'
        if args.resume and saved.is_file():
            result = read(saved)
            if result.get('scene') != str(scene.resolve()) or any(not Path(p).is_file() or sha(p) != digest
                    for p, digest in result.get('source_hashes', {}).items()):
                raise ValueError('Validation source hashes changed: '+str(destination))
            return result
        from validate_lunar_obstacle_map_v7 import validate_obstacle_map
        return validate_obstacle_map(scene, destination, goal_override_xy_m=branch_goal if branch else None)

    try:
        from make_lunar_fork_v11_configs import make_scene_config, make_surface_config
        config = read(args.config) if args.config else make_scene_config(args.seed)
        surface_config = read(args.surface_config) if args.surface_config else make_surface_config(args.seed)
        if not config.get('junction_v11') or not config.get('branch_v10'):
            raise ValueError('V11 terrain needs explicit junction_v11 and branch_v10')
        branch_goal = config['task_branch_goal']['xy_m']
        for name, value in [('scene.json', config), ('surface.json', surface_config)]:
            path = work/'config'/name
            if path.exists() and read(path) != value:
                raise ValueError('Resume configuration changed: '+name)
            if not path.exists():
                write(path, value)
        identity = dict(schema_version='terrain_build_identity/11.0', spacing_m=args.spacing,
            scene_config_sha256=sha(work/'config/scene.json'), surface_config_sha256=sha(work/'config/surface.json'),
            code_sha256={p: sha(ROOT/p) for p in FINGERPRINT_FILES})
        if (work/'input_identity.json').exists() and read(work/'input_identity.json') != identity:
            raise ValueError('Resume recipe/code hashes changed; use a fresh work/output directory')
        write(work/'input_identity.json', identity)
        stop()
        base, surface, before = [work/n for n in ['base', 'surface', 'colliders_before']]
        run_stage('base', 'build_lunar_cave_v4_cpu.py', ['--config', work/'config/scene.json', '--output', base, '--spacing', args.spacing], base)
        run_stage('surface', 'build_lunar_surface_detail_v5_cpu.py', ['--source', base, '--output', surface, '--config', work/'config/surface.json'], surface)
        run_stage('colliders_before', 'build_lunar_obstacle_scene_v7_cpu.py', ['--source', surface, '--output', before], before)
        status('baseline_main')
        main_before = validation(before, work/'baseline_main')
        status('baseline_branch')
        branch_before = validation(before, work/'baseline_branch', True)
        relocation = None
        if main_before['passed'] and branch_before['passed']:
            if not output.exists():
                stop()
                link_tree(before, output)
        else:
            stop()
            proposal_path = work/'relocation_proposal.json'
            if args.resume and proposal_path.is_file():
                relocation = read(proposal_path)
            else:
                relocation = propose_one_object_layout(before, work/'baseline_main/grid_data.npz',
                    work/'baseline_main/obstacle_map_report.json', branch_goal)
                write(proposal_path, relocation)
            moved = work/'surface_relocated'
            if not moved.exists():
                status('relocating_one_measured_object')
                relocate_surface(surface, before, relocation, moved)
            elif not args.resume or not (moved/'status.json').is_file() or read(moved/'status.json').get('passed') is not True:
                raise ValueError('Incomplete relocated source; inspect before retry')
            run_stage('colliders_final', 'build_lunar_obstacle_scene_v7_cpu.py', ['--source', moved, '--output', output], output)
            if not (output/'relocation_report.json').exists():
                shutil.copy2(moved/'relocation_report.json', output/'relocation_report.json')
        status('final_main')
        main_final = validation(output, output/'validation')
        status('final_branch')
        branch_final = validation(output, output/'validation_branch', True)
        passed = main_final['passed'] and branch_final['passed']
        final = dict(schema_version='complete_lunar_fork_terrain/11.0', passed=bool(passed),
            raw_recipe_identity=identity, factory_sha256=sha(ROOT/'make_lunar_fork_v11_configs.py'),
            wrapper_sha256=sha(__file__), baseline_passed=dict(main=main_before['passed'], branch=branch_before['passed']),
            final_passed=dict(main=main_final['passed'], branch=branch_final['passed']),
            path_length_m=dict(main=main_final['path_length_m'], branch=branch_final['path_length_m']),
            actual_gravel_relocation=relocation, original_macro90_preserved=True,
            boulder_count=len(read(output/'boulders.json')),
            gravel_instance_count=read(surface/'surface_detail_report.json')['gravel']['instance_count'],
            obstacle_count=read(output/'obstacle_report.json')['obstacle_count'],
            terrain_mesh_sha256=sha(output/'mesh.npz'), final_scene_sha256=sha(output/'cave.usdc'),
            GPU_work=False, physics_or_policy_success_evaluated=False)
        write(output/'terrain_build_report.json', final)
        if not passed:
            raise ValueError('Final real geometry failed one or both full vehicle checks; reports retained')
        status('complete', passed=True, output=str(output))
        return 0
    except InterruptedError:
        return 3
    except BaseException as exc:
        status('failed', passed=False, error=repr(exc))
        write(work/'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
