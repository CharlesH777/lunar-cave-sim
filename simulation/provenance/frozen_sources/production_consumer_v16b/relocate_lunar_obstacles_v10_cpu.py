"""Move two actual obstacles while preserving counts, dimensions and terrain.

The proposed positions were tested as a whole-map layout, without clearing a
prescribed route. This exports real geometry; final reachability is rechecked
from the resulting scene, not inferred from the proposal.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

from build_lunar_surface_detail_v5_cpu import (
    _absolute_assets,_collect_assets,_set_relative_assets,sha,write)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--obstacle-reference',type=Path,required=True)
    parser.add_argument('--proposal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve(strict=True)
    reference=args.obstacle_reference.resolve(strict=True);output=args.output.absolute()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1'):raise RuntimeError('CPU only')
    import numpy as np
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade,Vt
    from cave.lunar_gravel_v5 import _FloorProjector,INSTANCER_PATH
    from cave.lunar_visual_v4 import _author_mesh,_boulder_mesh,_basalt_color
    proposal=json.loads(args.proposal.read_text())
    if proposal.get('chosen_macro90_xy') is None:raise ValueError('No layout proposal accepted')
    names=['cave.usdc','scene_config.json','mesh.npz','boulders.json','geometry_report.json']
    protected={n:sha(source/n) for n in names}
    config=json.loads((source/'scene_config.json').read_text());boulders=json.loads((source/'boulders.json').read_text())
    original=copy.deepcopy(boulders)
    with np.load(source/'mesh.npz') as data:projector=_FloorProjector(data['vertices'],data['faces'])
    xy=np.asarray([proposal['chosen_macro90_xy'],proposal['gravel17170_xy']],float)
    floor,normal,face,_=projector.project(xy)
    if not np.isfinite(floor).all() or np.any(face<0):raise ValueError('Relocation has no actual mesh floor')
    boulders[90]['center_xyz_m']=[float(xy[0,0]),float(xy[0,1]),float(floor[0]+boulders[90]['scale_xyz_m'][2]*.5)]
    config['additional_boulders_v10'][10]['xy_m']=xy[0].tolist()
    config['obstacle_layout_revision_v10']='same_counts_and_sizes_two_entities_repositioned'
    output.mkdir(parents=True,exist_ok=False)
    for name in ['mesh.npz','ASSET_LICENSES.json','GRAVEL_V5_SOURCES.json','BASE_ASSET_SOURCES.json','surface_detail_report.json']:
        if (source/name).is_file():
            try:os.link(source/name,output/name)
            except OSError:shutil.copy2(source/name,output/name)
    write(output/'scene_config.json',config);write(output/'boulders.json',boulders)
    stage=Usd.Stage.Open(str(source/'cave.usdc'));stage.SetEditTarget(stage.GetSessionLayer())
    _absolute_assets(stage,Sdf)
    v,f,n=_boulder_mesh(boulders,hard=True,detail_revision=10)
    mat=UsdShade.Material.Get(stage,'/World/Materials/Basalt')
    _author_mesh(stage,'/World/Cave/AngularCollapseBlocks',v,f,n,mat,.65,
                 colors=_basalt_color(v),normal_interpolation='faceVarying')
    instancer=UsdGeom.PointInstancer(stage.GetPrimAtPath(INSTANCER_PATH))
    positions=np.asarray(instancer.GetPositionsAttr().Get(),float).copy();before=positions.copy()
    ids=np.asarray(instancer.GetIdsAttr().Get(),np.int64)
    rows=np.flatnonzero(ids==17170)
    if len(rows)!=1:raise ValueError('Gravel id17170 not uniquely bound')
    index=int(rows[0])
    source_obstacles=json.loads((reference/'obstacle_report.json').read_text())
    item=next(o for o in source_obstacles['obstacles'] if int(o.get('instance_id',o.get('instance_index',-1)))==17170)
    # Pure translation keeps the authored half quaternion and shape exactly.
    bottom_offset=float(item['z_min_m'])-positions[index,2]
    thickness=float(item['z_max_m'])-float(item['z_min_m'])
    new_z=float(floor[1]-bottom_offset-min(.02,thickness*.2))
    positions[index]=[float(xy[1,0]),float(xy[1,1]),new_z]
    instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
    if np.count_nonzero(np.any(before!=positions,axis=1))!=1:raise ValueError('Unexpected gravel edits')
    mapping,assets=_collect_assets(stage,output,Sdf);_set_relative_assets(stage,mapping,Sdf)
    if not stage.Export(str(output/'cave.usdc')):raise RuntimeError('USD export failed')
    reopened=Usd.Stage.Open(str(output/'cave.usdc'));_set_relative_assets(reopened,mapping,Sdf);reopened.GetRootLayer().Save()
    del reopened
    reopened=Usd.Stage.Open(str(output/'cave.usdc'))
    exported=UsdGeom.PointInstancer(reopened.GetPrimAtPath(INSTANCER_PATH))
    assert len(exported.GetPositionsAttr().Get())==len(before)
    assert list(exported.GetProtoIndicesAttr().Get())==list(instancer.GetProtoIndicesAttr().Get())
    assert list(exported.GetScalesAttr().Get())==list(instancer.GetScalesAttr().Get())
    assert list(exported.GetOrientationsAttr().Get())==list(instancer.GetOrientationsAttr().Get())
    for n,digest in protected.items():
        if sha(source/n)!=digest:raise RuntimeError('Source changed: '+n)
    report=dict(passed=True,source=str(source),source_sha256=protected,source_unchanged=True,
        terrain_mesh_unchanged=True,boulder_count=len(boulders),gravel_count=len(before),
        boulder90_before=original[90],boulder90_after=boulders[90],
        gravel_id=17170,gravel_position_before=before[index].tolist(),gravel_position_after=positions[index].tolist(),
        dimensions_or_orientations_changed=False,removed_obstacle_count=0,
        relocation_reason='Two objects formed a complete barrier; reposition as natural rubble near existing collapse blocks',
        proposal=proposal,texture_assets=assets,final_path_validation_pending=True,GPU_work=False)
    write(output/'relocation_report.json',report)
    geometry=json.loads((source/'geometry_report.json').read_text());geometry['obstacle_relocation']=report
    write(output/'geometry_report.json',geometry)
    write(output/'status.json',dict(phase='complete',passed=True,geometry_authored=True,path_validated=False))
    print(json.dumps({k:report[k] for k in ['passed','boulder_count','gravel_count','removed_obstacle_count']}))


if __name__=='__main__':main()
