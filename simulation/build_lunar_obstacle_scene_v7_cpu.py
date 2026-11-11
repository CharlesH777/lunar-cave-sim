"""Package retained gravel as static collision geometry, preserving visible assets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from build_lunar_surface_detail_v5_cpu import (
    _absolute_assets, _collect_assets, _set_relative_assets, _macro_snapshot,
    _verify_macro, _asset_values, sha, write)

ROOT = Path(__file__).resolve().parent


def _instancer_snapshot(stage, UsdGeom):
    result = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.PointInstancer):
            result[str(prim.GetPath())] = {
                attr.GetName(): hashlib.sha256(str(attr.Get()).encode()).hexdigest()
                for attr in prim.GetAttributes()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source.resolve(strict=True)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def status(phase, **extra):
        value = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=round(time.monotonic()-started, 3), GPU_work=False,
                     physics_evaluated=False, navigation_evaluated=False, **extra)
        write(output/'status.json', value)
        print(json.dumps(value), flush=True)

    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '-1'):
            raise RuntimeError('CPU asset build requires CUDA_VISIBLE_DEVICES empty')
        import numpy as np
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        from cave.lunar_obstacles_v7 import add_gravel_collision
        status('loading_source')
        source_names = ['cave.usdc', 'scene_config.json', 'geometry_report.json', 'mesh.npz',
                        'boulders.json', 'surface_detail_report.json']
        source_hashes = {name: sha(source/name) for name in source_names}
        config = json.loads((source/'scene_config.json').read_text())
        if not config.get('ground_profile_v7') or 'rover_route' in json.dumps(config):
            raise ValueError('Expected v7 uneven ground with no reserved route')
        base_report = json.loads((source/'geometry_report.json').read_text())
        if base_report.get('passed') is not True:
            raise ValueError('Source geometry validation did not pass')
        config['obstacle_revision'] = 7
        write(output/'scene_config.json', config)
        for name in ['mesh.npz', 'boulders.json', 'surface_detail_report.json', 'ASSET_LICENSES.json',
                     'GRAVEL_V5_SOURCES.json', 'BASE_ASSET_SOURCES.json']:
            if (source/name).is_file():
                try:
                    os.link(source/name, output/name)
                except OSError:
                    shutil.copy2(source/name, output/name)
        stage = Usd.Stage.Open(str(source/'cave.usdc'))
        stage.SetEditTarget(stage.GetSessionLayer())
        _absolute_assets(stage, Sdf)
        macro = _macro_snapshot(stage, np, UsdGeom, UsdPhysics)
        instancers = _instancer_snapshot(stage, UsdGeom)
        status('adding_static_gravel_colliders')
        obstacles = add_gravel_collision(stage, config)
        if obstacles.get('passed') is not True:
            raise ValueError('Gravel collision authoring did not pass')
        write(output/'obstacle_report.json', obstacles)
        mapping, assets = _collect_assets(stage, output, Sdf)
        _set_relative_assets(stage, mapping, Sdf)
        if not stage.Export(str(output/'cave.usdc')):
            raise RuntimeError('USD export failed')
        reopened = Usd.Stage.Open(str(output/'cave.usdc'))
        _set_relative_assets(reopened, mapping, Sdf)
        reopened.GetRootLayer().Save()
        del reopened
        reopened = Usd.Stage.Open(str(output/'cave.usdc'))
        preserved = _verify_macro(reopened, macro, np, UsdGeom, UsdPhysics)
        if _instancer_snapshot(reopened, UsdGeom) != instancers:
            raise ValueError('Visible gravel instancing changed during collider packaging')
        added = []
        for prim in reopened.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or str(prim.GetPath()) in macro['meshes']:
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), float)
            valid, reason = UsdGeom.Mesh.ValidateTopology(mesh.GetFaceVertexIndicesAttr().Get(),
                mesh.GetFaceVertexCountsAttr().Get(), len(points))
            collision = UsdPhysics.CollisionAPI(prim)
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            hidden = UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
            if not (valid and np.isfinite(points).all() and collision.GetCollisionEnabledAttr().Get()
                    and hidden and approximation == 'none' and not prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                raise ValueError('Invalid static collision mesh '+str(prim.GetPath())+': '+reason)
            added.append(dict(path=str(prim.GetPath()), points=len(points),
                faces=len(mesh.GetFaceVertexCountsAttr().Get()), hidden=True, approximation='none'))
        if not added:
            raise ValueError('No collision geometry added')
        for prim in reopened.Traverse():
            for attr in prim.GetAttributes():
                values, _ = _asset_values(attr, Sdf)
                for value in values:
                    if not value.path:
                        continue
                    resolved = Path(value.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(
                        reopened.GetRootLayer(), value.path)).resolve(strict=True)
                    if Path(value.path).is_absolute() or not resolved.is_relative_to(output.resolve()):
                        raise ValueError('Nonportable texture path '+value.path)
        for name, digest in source_hashes.items():
            if sha(source/name) != digest:
                raise ValueError('Source file changed: '+name)
        report = dict(passed=True, obstacle_revision=7, base_scene=str(source),
            source_files_sha256=source_hashes, source_unchanged=True,
            visible_geometry_preservation=preserved, visible_instancing_unchanged=True,
            added_static_collision_meshes=added, texture_assets=assets,
            textures_relative_and_local=True, gravel_obstacles=obstacles,
            collision_geometry_authored=True, physx_cooking_evaluated=False,
            wheel_dynamics_evaluated=False, navigation_evaluated=False, GPU_work=False)
        write(output/'obstacle_package_report.json', report)
        combined = dict(base_report)
        combined['obstacle_package'] = report
        combined['passed'] = True
        write(output/'geometry_report.json', combined)
        status('complete', passed=True, usd='cave.usdc', obstacle_report='obstacle_report.json')
        return 0
    except BaseException as exc:
        status('failed', error=repr(exc))
        write(output/'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
