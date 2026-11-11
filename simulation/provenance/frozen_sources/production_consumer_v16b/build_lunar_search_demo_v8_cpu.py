"""Add seeded, labelled search objects to the unchanged lunar-cave obstacle scene."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from build_lunar_surface_detail_v5_cpu import (
    _absolute_assets, _collect_assets, _set_relative_assets, _macro_snapshot,
    _verify_macro, _asset_values, sha, write)

ROOT=Path(__file__).resolve().parent


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/lunar_obstacle_v7/scene_03')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261118)
    parser.add_argument('--asset-revision',type=int,choices=[8,10],default=8)
    args=parser.parse_args(argv)
    source=args.source.resolve(strict=True);output=args.output.absolute()
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()

    def status(phase,**kw):
        record=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.monotonic()-started,3),GPU_work=False,**kw)
        write(output/'status.json',record);print(json.dumps(record),flush=True)

    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1'):
            raise RuntimeError('Use an empty CUDA_VISIBLE_DEVICES for CPU authoring')
        import numpy as np
        from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdSemantics
        from cave.lunar_human_flag_v8 import build_astronaut,build_chinese_flag
        from cave.lunar_vehicle_base_v8 import build_target_rover,build_lunar_base
        if args.asset_revision>=10:
            from cave.lunar_astronaut_v10 import build_astronaut
            from cave.lunar_habitat_v10 import build_lunar_base
        from cave.lunar_search_placement_v8 import sample_target_layout
        from build_lunar_obstacle_scene_v7_cpu import _instancer_snapshot
        from validate_lunar_obstacle_map_v7 import validate_obstacle_map
        source_hashes={name:sha(source/name) for name in ['cave.usdc','scene_config.json',
            'mesh.npz','boulders.json','obstacle_report.json','geometry_report.json']}
        config=json.loads((source/'scene_config.json').read_text())
        base_report=json.loads((source/'geometry_report.json').read_text())
        if base_report.get('passed') is not True:raise ValueError('Source geometry not accepted')
        for name in ['mesh.npz','boulders.json','obstacle_report.json','surface_detail_report.json',
                     'obstacle_package_report.json','ASSET_LICENSES.json','GRAVEL_V5_SOURCES.json','BASE_ASSET_SOURCES.json']:
            if (source/name).is_file():
                try:os.link(source/name,output/name)
                except OSError:shutil.copy2(source/name,output/name)
        status('authoring_search_objects')
        stage=Usd.Stage.Open(str(source/'cave.usdc'));stage.SetEditTarget(stage.GetSessionLayer())
        _absolute_assets(stage,Sdf)
        macro=_macro_snapshot(stage,np,UsdGeom,UsdPhysics)
        old_instancing=_instancer_snapshot(stage,UsdGeom)
        UsdGeom.Xform.Define(stage,'/World/SearchTargets')
        definitions=[('astronaut',build_astronaut),('lunar_rover',build_target_rover),
                     ('lunar_base',build_lunar_base),('chinese_flag',build_chinese_flag)]
        specifications=[];metadata={}
        for identifier,builder in definitions:
            path='/World/SearchTargets/'+identifier
            item=builder(stage,path);metadata[identifier]=item
            specifications.append(dict(id=identifier,class_name=item['class_name'],
                footprint_radius_m=float(item['local_ground_footprint_radius_m']),height_m=float(item['height_m'])))
        status('sampling_seeded_layout',seed=args.seed)
        layout=sample_target_layout(source,specifications,args.seed)
        target_rows=[];collision_rows=[]
        for item in layout['targets']:
            identifier=item['id'];meta=metadata[identifier]
            prim=stage.GetPrimAtPath(meta['prim_path']);pose=UsdGeom.Xformable(prim)
            pose.ClearXformOpOrder();pose.AddTranslateOp().Set(Gf.Vec3d(*map(float,item['xyz_m'])))
            pose.AddRotateZOp().Set(float(item['yaw_deg']))
            for name,value in [('category',item['class_name']),('object_id',identifier),('label_zh',meta['label_zh'])]:
                prim.CreateAttribute('cave:'+name,Sdf.ValueTypeNames.String,custom=True).Set(value)
            prim.CreateAttribute('cave:placementSeed',Sdf.ValueTypeNames.Int64,custom=True).Set(args.seed)
            prim.CreateAttribute('cave:isSearchTarget',Sdf.ValueTypeNames.Bool,custom=True).Set(True)
            UsdSemantics.LabelsAPI.Apply(prim,'class').CreateLabelsAttr([item['class_name']])
            UsdSemantics.LabelsAPI.Apply(prim,'object_id').CreateLabelsAttr([identifier])
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
            bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            low=np.asarray(bounds.GetMin(),float);high=np.asarray(bounds.GetMax(),float)
            if not np.isfinite([low,high]).all():raise ValueError('Nonfinite target bounds')
            # Rectangular bounds overapproximate the XY projection; retain the
            # declared circumcircle checked by each asset's CPU validation.
            row=dict(item,**{k:v for k,v in meta.items() if k not in item},world_aabb_m=[low.tolist(),high.tolist()],
                instruction_zh='找到'+meta['label_zh']+'并靠近。')
            target_rows.append(row)
            collision_rows.append(dict(id=identifier,kind='search_target',class_name=item['class_name'],
                center_xy_m=item['xyz_m'][:2],xy_radius_m=float(item['footprint_radius_m']),
                z_min_m=float(low[2]),z_max_m=float(high[2]),source_radius='full authored object XY circumcircle'))
        config.update(search_objects_revision=args.asset_revision,search_layout_seed=args.seed,
            search_targets_file='targets.json',target_obstacles_file='task_obstacles.json',
            search_target_classes=[r['class_name'] for r in target_rows])
        write(output/'scene_config.json',config)
        targets=dict(schema_version='lunar_search_targets/8.0',seed=args.seed,targets=target_rows,
            placement=layout['report'],positions_randomized=True,flag_position_randomized=True,
            labels_in_scene_metadata_only=True,road_signs_added=False)
        write(output/'targets.json',targets);write(output/'task_obstacles.json',collision_rows)
        mapping,assets=_collect_assets(stage,output,Sdf);_set_relative_assets(stage,mapping,Sdf)
        if not stage.Export(str(output/'cave.usdc')):raise RuntimeError('USD export failed')
        reopened=Usd.Stage.Open(str(output/'cave.usdc'));_set_relative_assets(reopened,mapping,Sdf)
        reopened.GetRootLayer().Save();del reopened
        reopened=Usd.Stage.Open(str(output/'cave.usdc'))
        preserved=_verify_macro(reopened,macro,np,UsdGeom,UsdPhysics)
        if _instancer_snapshot(reopened,UsdGeom)!=old_instancing:raise ValueError('Existing gravel changed')
        target_checks=[]
        for row in target_rows:
            root_prim=reopened.GetPrimAtPath(row['prim_path']);colliders=[];meshes=0
            if list(UsdSemantics.LabelsAPI(root_prim,'class').GetLabelsAttr().Get())!=[row['class_name']]:
                raise ValueError('Native semantic class missing after export')
            for prim in Usd.PrimRange(root_prim):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Target must remain static')
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                        raise ValueError('Disabled target collider')
                    colliders.append(str(prim.GetPath()))
                if prim.IsA(UsdGeom.Mesh):
                    mesh=UsdGeom.Mesh(prim);pts=np.asarray(mesh.GetPointsAttr().Get(),float)
                    valid,reason=UsdGeom.Mesh.ValidateTopology(mesh.GetFaceVertexIndicesAttr().Get(),
                        mesh.GetFaceVertexCountsAttr().Get(),len(pts))
                    if not valid or not np.isfinite(pts).all():raise ValueError('Invalid prop topology '+str(prim.GetPath())+reason)
                    meshes+=1
            if not colliders:raise ValueError('Missing target collision '+row['id'])
            target_checks.append(dict(id=row['id'],mesh_count=meshes,collision_paths=colliders,static=True))
        for prim in reopened.Traverse():
            for attr in prim.GetAttributes():
                values,_=_asset_values(attr,Sdf)
                for value in values:
                    if not value.path:continue
                    resolved=Path(value.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(reopened.GetRootLayer(),value.path)).resolve(strict=True)
                    if Path(value.path).is_absolute() or not resolved.is_relative_to(output.resolve()):
                        raise ValueError('Asset is not portable '+value.path)
        package=dict(passed=True,seed=args.seed,target_count=len(target_rows),target_checks=target_checks,
            source=str(source),source_files_sha256=source_hashes,base_geometry_preservation=preserved,
            original_gravel_unchanged=True,target_collision_enabled=True,texture_assets=assets,
            geometry_and_assets_self_authored=True,physics_evaluated=False,navigation_policy_evaluated=False)
        combined=dict(base_report);combined['search_objects']=package;write(output/'geometry_report.json',combined)
        write(output/'search_objects_report.json',package)
        status('checking_target_obstacles_and_ground_route')
        checked=validate_obstacle_map(output,output/'validation',extra_obstacles=collision_rows)
        if not checked['passed']:raise ValueError('Added search objects blocked the checked main path; see validation report')
        if config.get('task_branch_goal'):
            branch_checked=validate_obstacle_map(output,output/'validation_branch',extra_obstacles=collision_rows,
                goal_override_xy_m=config['task_branch_goal']['xy_m'])
            if not branch_checked['passed']:raise ValueError('Branch route failed geometry checks; see validation_branch')
        for name,digest in source_hashes.items():
            if sha(source/name)!=digest:raise RuntimeError('Source asset changed '+name)
        status('complete',passed=True,target_count=4,seed=args.seed,geometric_path_m=checked['path_length_m'],
            source_assets_unchanged=True,USD='cave.usdc',targets='targets.json')
        return 0
    except BaseException as exc:
        status('failed',error=repr(exc));write(output/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':raise SystemExit(main())
