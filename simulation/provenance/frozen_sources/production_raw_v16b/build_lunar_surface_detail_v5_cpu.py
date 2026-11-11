"""Add portable scanned gravel materials and render-only fragments on CPU.

The validated v4 macro geometry and collision are preserved. All authored
changes live in a session layer until flattened into a new output package.
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
import time
import traceback

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _default(value):
    if hasattr(value,'tolist'):return value.tolist()
    if hasattr(value,'item'):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError('Not JSON serializable: '+type(value).__name__)


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=_default)+'\n')
    temp.replace(path)


def _asset_values(attr,Sdf):
    value=attr.Get()
    if isinstance(value,Sdf.AssetPath):return [value],False
    if attr.GetTypeName()==Sdf.ValueTypeNames.AssetArray and value is not None:return list(value),True
    return [],False


def _absolute_assets(stage,Sdf):
    """Capture original-layer resolution before session edits/export relocation."""
    count=0
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            values,array=_asset_values(attr,Sdf)
            if not values:continue
            result=[]
            for value in values:
                if not value.path:result.append(value);continue
                resolved=value.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(),value.path)
                path=Path(resolved).expanduser().resolve(strict=True)
                if not path.is_file():raise FileNotFoundError(path)
                result.append(Sdf.AssetPath(str(path)));count+=1
            attr.Set(result if array else result[0])
    return count


def _macro_snapshot(stage,np,UsdGeom,UsdPhysics):
    meshes={};collision={};xforms={};physics={}
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        transforms={a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith('xformOp')}
        if transforms:xforms[path]=transforms
        if prim.IsA(UsdPhysics.Scene):
            physics[path]={a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith('physics:')}
        if prim.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(prim);values={}
            for name,attr in [('points',mesh.GetPointsAttr()),('counts',mesh.GetFaceVertexCountsAttr()),
                              ('indices',mesh.GetFaceVertexIndicesAttr()),('normals',mesh.GetNormalsAttr())]:
                value=attr.Get()
                array=np.asarray(value) if value is not None else np.empty(0)
                values[name]=dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())
            meshes[path]=values
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            collision[path]=dict(schemas=list(prim.GetAppliedSchemas()),
                collision_enabled=prim.GetAttribute('physics:collisionEnabled').Get(),
                approximation=prim.GetAttribute('physics:approximation').Get(),
                rigid_body_enabled=prim.GetAttribute('physics:rigidBodyEnabled').Get())
    return dict(meshes=meshes,collision=collision,xforms=xforms,physics=physics,
                stage=dict(meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
                           up_axis=str(UsdGeom.GetStageUpAxis(stage)),default_prim=str(stage.GetDefaultPrim().GetPath())))


def _verify_macro(stage,snapshot,np,UsdGeom,UsdPhysics):
    after=_macro_snapshot(stage,np,UsdGeom,UsdPhysics)
    missing=[p for p in snapshot['meshes'] if p not in after['meshes']]
    changed=[p for p,v in snapshot['meshes'].items() if after['meshes'].get(p)!=v]
    changed_collision=[p for p,v in snapshot['collision'].items() if after['collision'].get(p)!=v]
    changed_xforms=[p for p,v in snapshot['xforms'].items() if after['xforms'].get(p)!=v]
    if missing or changed or changed_collision or changed_xforms or after['physics']!=snapshot['physics'] or after['stage']!=snapshot['stage']:
        raise ValueError('Frozen v4 geometry/collision changed: '+repr(dict(missing=missing,geometry=changed,collision=changed_collision,
                                                                        transforms=changed_xforms,physics=after['physics']!=snapshot['physics'],
                                                                        stage=after['stage']!=snapshot['stage'])))
    return dict(passed=True,base_mesh_count=len(snapshot['meshes']),
                preserved_collision_prim_count=len(snapshot['collision']),points_faces_normals_unchanged=True,
                base_transforms_units_and_gravity_unchanged=True)


def _collect_assets(stage,output,Sdf):
    target=output/'textures';target.mkdir(exist_ok=True)
    mapping={};files={}
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            values,array=_asset_values(attr,Sdf)
            if not values:continue
            relative=[]
            for value in values:
                if not value.path:relative.append('');continue
                raw=value.resolvedPath or value.path
                source=Path(raw).expanduser()
                if not source.is_absolute():source=Path(Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(),str(source)))
                source=source.resolve(strict=True)
                digest=sha(source);destination=target/source.name
                if destination.exists() and sha(destination)!=digest:
                    destination=target/(source.stem+'_'+digest[:12]+source.suffix)
                if destination.exists():
                    if sha(destination)!=digest:raise FileExistsError(destination)
                else:shutil.copy2(source,destination)
                rel=destination.relative_to(output).as_posix();relative.append(rel)
                files[rel]=dict(relative_path=rel,source_path=str(source),sha256=digest,bytes=destination.stat().st_size)
            mapping[str(attr.GetPath())]=dict(array=array,paths=relative)
    return mapping,list(files.values())


def _set_relative_assets(stage,mapping,Sdf):
    for path,record in mapping.items():
        attr=stage.GetAttributeAtPath(path)
        if not attr:raise ValueError('Flattened stage lost asset attribute '+path)
        values=[Sdf.AssetPath(p) for p in record['paths']]
        attr.Set(values if record['array'] else values[0])


def _validate_uv(mesh,np,UsdGeom):
    uv=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
    if not uv or not uv.HasValue():return dict(present=False)
    values=np.asarray(uv.Get(),dtype=float)
    if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():raise ValueError('Invalid UV values: '+str(mesh.GetPath()))
    interpolation=str(uv.GetInterpolation());points=len(mesh.GetPointsAttr().Get())
    counts=mesh.GetFaceVertexCountsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get()
    expected={'vertex':points,'varying':points,'faceVarying':len(indices),'uniform':len(counts),'constant':1}.get(interpolation)
    if expected is None:raise ValueError('Unsupported UV interpolation '+interpolation)
    uv_indices=uv.GetIndices()
    if uv_indices is not None and len(uv_indices):
        ix=np.asarray(uv_indices,dtype=np.int64)
        if len(ix)!=expected or ix.min()<0 or ix.max()>=len(values):raise ValueError('Invalid indexed UV: '+str(mesh.GetPath()))
    elif len(values)!=expected:raise ValueError('UV length mismatch: '+str(mesh.GetPath()))
    return dict(present=True,interpolation=interpolation,value_count=len(values),expected_elements=int(expected),finite=True)


def _validate(stage,output,mapping,np,Sdf,UsdGeom,UsdShade,UsdPhysics):
    from PIL import Image
    mesh_rows=[];instancers=[];material_paths=[];shader_paths=[];images={}
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
            counts=mesh.GetFaceVertexCountsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get()
            valid,reason=UsdGeom.Mesh.ValidateTopology(indices,counts,len(points))
            if not valid or not np.isfinite(points).all():raise ValueError('USD topology invalid '+path+': '+reason)
            uv=_validate_uv(mesh,np,UsdGeom)
            bound,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not bound:raise ValueError('Unbound mesh material '+path)
            mesh_rows.append(dict(path=path,points=len(points),faces=len(counts),topology_valid=True,
                                  uv=uv,material=str(bound.GetPath())))
        if prim.IsA(UsdGeom.PointInstancer):
            item=UsdGeom.PointInstancer(prim);positions=np.asarray(item.GetPositionsAttr().Get(),dtype=float)
            proto=np.asarray(item.GetProtoIndicesAttr().Get(),dtype=np.int64)
            targets=item.GetPrototypesRel().GetTargets();n=len(positions)
            scales=np.asarray(item.GetScalesAttr().Get(),dtype=float)
            orientations=item.GetOrientationsAttr().Get()
            if positions.shape!=(n,3) or not n or not np.isfinite(positions).all():raise ValueError('Bad instancer positions '+path)
            if len(proto)!=n or not len(targets) or proto.min()<0 or proto.max()>=len(targets):raise ValueError('Bad prototype indices '+path)
            if scales.shape!=(n,3) or not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Bad instance scales '+path)
            if orientations is None or len(orientations)!=n:raise ValueError('Bad instance orientation count '+path)
            quat=np.asarray([[q.GetReal(),*q.GetImaginary()] for q in orientations],dtype=float)
            if not np.isfinite(quat).all() or np.any(np.linalg.norm(quat,axis=1)<.9):raise ValueError('Invalid instance quaternions '+path)
            for target in targets:
                if not stage.GetPrimAtPath(target):raise ValueError('Missing prototype '+str(target))
            instancers.append(dict(path=path,instance_count=n,prototype_count=len(targets),positions_indices_scales_orientations_equal=True,
                                   prototype_indices_valid=True,prototype_paths=list(map(str,targets))))
        if prim.IsA(UsdShade.Material):
            material=UsdShade.Material(prim);surface=material.GetSurfaceOutput()
            if not surface or not surface.HasConnectedSource():raise ValueError('Material has no surface shader '+path)
            material_paths.append(path)
        if prim.IsA(UsdShade.Shader):shader_paths.append(path)
        if path.startswith('/World/SurfaceDetail') and (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
            raise ValueError('Render-only gravel must not have collision or rigid bodies: '+path)
    for attrpath,record in mapping.items():
        attr=stage.GetAttributeAtPath(attrpath);values,_=_asset_values(attr,Sdf)
        for value in values:
            if not value.path:continue
            if Path(value.path).is_absolute():raise ValueError('Exported texture remains absolute: '+value.path)
            resolved=Path(value.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(),value.path)).resolve(strict=True)
            if not resolved.is_relative_to(output.resolve()):raise ValueError('Asset escaped output package: '+str(resolved))
            if resolved.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff','.webp','.bmp'):
                if str(resolved) not in images:
                    with Image.open(resolved) as image:resolution=list(image.size);image.verify()
                    images[str(resolved)]=dict(relative_path=resolved.relative_to(output).as_posix(),resolution=resolution,readable=True)
    if not instancers or not material_paths or not shader_paths:raise ValueError('Missing gravel instancer/material/shader content')
    return dict(passed=True,stage_reopened=True,meshes=mesh_rows,instancers=instancers,
                material_count=len(material_paths),shader_count=len(shader_paths),material_paths=material_paths,
                image_assets=list(images.values()),all_assets_relative_and_local=True,
                gravel_collision_enabled=False,physics_evaluated=False,renders_executed=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/lunar_morphology_v4/full_scene_02')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/lunar_surface_detail_v5/scene_01')
    parser.add_argument('--config',type=Path,help='Optional surface-detail option JSON, merged over a copy of scene config')
    args=parser.parse_args(argv);source=args.source.expanduser().resolve(strict=True);output=args.output.expanduser().absolute()
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def status(phase,**kw):
        record=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-start,3),
                    GPU_work=False,simulation_started=False,training_started=False,navigation_evaluated=False,**kw)
        write(output/'status.json',record);print(json.dumps(record),flush=True)
    try:
        status('loading_frozen_v4')
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1'):raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for CPU build")
        import numpy as np
        from pxr import Sdf,Usd,UsdGeom,UsdShade,UsdPhysics
        from cave.lunar_gravel_material_v5 import apply_gravel_materials
        from cave.lunar_gravel_v5 import add_gravel
        base_files={name:sha(source/name) for name in ['cave.usdc','scene_config.json','geometry_report.json','mesh.npz']}
        base_config=json.loads((source/'scene_config.json').read_text())
        base_report=json.loads((source/'geometry_report.json').read_text())
        if base_report.get('passed') is not True:raise ValueError('Base v4 geometry has not passed validation')
        config=copy.deepcopy(base_config)
        if args.config:
            options=json.loads(args.config.read_text())
            if not isinstance(options,dict):raise ValueError('Surface config must be a JSON object')
            protected=set(base_config)-{'surface_detail_revision','gravel'}
            if protected.intersection(options):raise ValueError('Surface override may not replace v4 scene fields: '+repr(sorted(protected.intersection(options))))
            config.update(options)
        config['surface_detail_revision']=5
        shutil.copy2(source/'scene_config.json',output/'base_scene_config.json')
        shutil.copy2(source/'geometry_report.json',output/'base_geometry_report.json')
        write(output/'scene_config.json',config)
        try:os.link(source/'mesh.npz',output/'mesh.npz');mesh_copy_method='hardlink_readonly_use'
        except OSError:shutil.copy2(source/'mesh.npz',output/'mesh.npz');mesh_copy_method='copy'
        for name in ['boulders.json']:
            if (source/name).is_file():shutil.copy2(source/name,output/name)
        stage=Usd.Stage.Open(str(source/'cave.usdc'))
        if not stage:raise RuntimeError('Could not open frozen v4 stage')
        stage.SetEditTarget(stage.GetSessionLayer())
        absolute_count=_absolute_assets(stage,Sdf)
        macro=_macro_snapshot(stage,np,UsdGeom,UsdPhysics)
        status('applying_cc0_gravel_materials')
        material_report=apply_gravel_materials(stage,output,config)
        if not isinstance(material_report,dict) or material_report.get('passed') is not True:raise ValueError('Material authoring checks failed')
        status('adding_render_only_gravel')
        gravel_report=add_gravel(stage,source,config)
        if not isinstance(gravel_report,dict) or gravel_report.get('passed') is not True:raise ValueError('Fragment authoring checks failed')
        _verify_macro(stage,macro,np,UsdGeom,UsdPhysics)
        status('collecting_relative_textures')
        asset_mapping,asset_files=_collect_assets(stage,output,Sdf)
        _set_relative_assets(stage,asset_mapping,Sdf)
        usd=output/'cave.usdc'
        if not stage.Export(str(usd)):raise RuntimeError('Flattened USD export failed')
        # Normalize again in the real output layer: anonymous session-layer
        # resolution may otherwise re-anchor a relative path during flattening.
        reopened=Usd.Stage.Open(str(usd))
        if not reopened:raise RuntimeError('Could not reopen surface-detail USD')
        _set_relative_assets(reopened,asset_mapping,Sdf);reopened.GetRootLayer().Save()
        del reopened
        reopened=Usd.Stage.Open(str(usd))
        validation=_validate(reopened,output,asset_mapping,np,Sdf,UsdGeom,UsdShade,UsdPhysics)
        preserved=_verify_macro(reopened,macro,np,UsdGeom,UsdPhysics)
        for name,digest in base_files.items():
            if sha(source/name)!=digest:raise RuntimeError('Frozen base file changed: '+name)
        provider_manifest=json.loads((ROOT/'assets/ASSET_SOURCES.json').read_text())
        shutil.copy2(ROOT/'assets/ASSET_SOURCES.json',output/'BASE_ASSET_SOURCES.json')
        licenses=dict(texture_assets=asset_files,material_texture_sources=material_report.get('texture_sources'),
            provider_sources=provider_manifest['assets'],provider='Poly Haven',license='CC0-1.0',
            license_url='https://creativecommons.org/publicdomain/zero/1.0/',
            provider_license_url='https://polyhaven.com/license',
            procedural_fragments='locally generated display geometry; no downloaded third-party rock mesh',
            appearance_scope='terrestrial scanned surface proxies, not measured lunar albedo or BRDF')
        detail_sources=ROOT/'assets/GRAVEL_V5_SOURCES.json'
        if detail_sources.is_file():
            shutil.copy2(detail_sources,output/'GRAVEL_V5_SOURCES.json')
            licenses['surface_detail_download_sources']=json.loads(detail_sources.read_text())
        write(output/'ASSET_LICENSES.json',licenses)
        report=dict(passed=True,surface_detail_revision=5,base_scene=str(source),base_source_sha256=base_files,
            base_source_files_unchanged=True,base_macro_preservation=preserved,
            original_asset_paths_resolved_before_edit=absolute_count,materials=material_report,gravel=gravel_report,
            usd_validation=validation,texture_assets=asset_files,mesh_copy_method=mesh_copy_method,
            macro_collision_unchanged=True,micro_gravel_render_only=True,micro_gravel_collision=False,
            physics_evaluated=False,navigation_evaluated=False,GPU_work=False,simulation_started=False,training_started=False,
            elapsed_seconds=time.monotonic()-start)
        write(output/'surface_detail_report.json',report)
        combined=copy.deepcopy(base_report);combined['base_geometry']=base_report;combined['surface_details']=report
        combined['passed']=bool(base_report['passed'] and report['passed']);write(output/'geometry_report.json',combined)
        status('complete',passed=True,usd='cave.usdc',report='surface_detail_report.json',
               instance_count=gravel_report.get('instance_count'),macro_collision_unchanged=True,micro_gravel_collision=False)
        return 0
    except BaseException as exc:
        status('failed',error=repr(exc));write(output/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':raise SystemExit(main())
