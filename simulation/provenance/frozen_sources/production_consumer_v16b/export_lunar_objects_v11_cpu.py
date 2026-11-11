"""Export the four search objects as independent, portable USD assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from build_lunar_surface_detail_v5_cpu import _collect_assets, _set_relative_assets, sha, write
from cave.lunar_astronaut_v10 import build_astronaut
from cave.lunar_habitat_v10 import build_lunar_base
from cave.lunar_vehicle_base_v8 import build_target_rover
from cave.lunar_human_flag_v8 import build_chinese_flag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '-1'):
        raise ValueError('CPU exporter requires CUDA_VISIBLE_DEVICES empty')
    out = args.output.absolute(); out.mkdir(parents=True, exist_ok=False)
    records = []
    for name, builder in [('astronaut', build_astronaut), ('lunar_base', build_lunar_base),
                          ('lunar_rover', build_target_rover), ('chinese_flag', build_chinese_flag)]:
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, '/World'); stage.SetDefaultPrim(world.GetPrim())
        metadata = builder(stage, '/World/Object')
        meshes = 0; colliders = 0
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Unexpected rigid body in static search object')
            colliders += int(prim.HasAPI(UsdPhysics.CollisionAPI))
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim); points = np.asarray(mesh.GetPointsAttr().Get(), float)
                valid, reason = mesh.ValidateTopology(mesh.GetFaceVertexIndicesAttr().Get(),
                    mesh.GetFaceVertexCountsAttr().Get(), len(points))
                if not valid or not np.isfinite(points).all():
                    raise ValueError('Invalid asset mesh: '+reason)
                meshes += 1
        mapping, assets = _collect_assets(stage, out, Sdf)
        _set_relative_assets(stage, mapping, Sdf)
        path = out/(name+'.usdc')
        if not stage.Export(str(path)):
            raise RuntimeError('Asset export failed')
        reopened = Usd.Stage.Open(str(path))
        if not reopened.GetDefaultPrim() or not reopened.GetPrimAtPath('/World/Object'):
            raise ValueError('Exported object cannot be reopened')
        records.append(dict(id=name, file=path.name, sha256=sha(path), bytes=path.stat().st_size,
            metadata=metadata, meshes=meshes, static_colliders=colliders,
            textures=assets, unit='metres', up_axis='Z', default_prim='/World'))
    write(out/'OBJECT_ASSETS.json', dict(passed=True, objects=records, source='original procedural project assets',
        placement='local origin only; final cave positions and yaw are randomized per layout seed',
        scope='static search objects; no astronaut/vehicle/habitat dynamics'))
    (out/'README.md').write_text('# 月洞搜索物品资产\n\n'
        '四个独立 USD：宇航员、小型月球基地、小型探月车、中国国旗。单位米、Z-up。'
        '引用每个文件的默认 /World prim，可同时带入其材质；模型对象在 /World/Object。\n\n'
        '均为原始程序化模型，包含静态碰撞体；此处原点摆放只为导出资产。'
        '洞穴任务中位置和朝向由布局 seed 随机生成。\n')
    print(json.dumps(dict(passed=True, objects=len(records), bytes=sum(r['bytes'] for r in records))))


if __name__ == '__main__':
    main()
