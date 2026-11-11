"""Pure-PXR materials/export for an open synthetic lunar pit and cave patch.

The caller owns the terrain field, geometry orientation, boundary acceptance,
and scientific assumptions. This module does not start Kit or simulate physics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .scene import ROOT, mesh as write_mesh


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _basalt_color(points):
    """Low-frequency linear colors; no tiled crack albedo or rock normal map."""
    p = np.asarray(points, dtype=np.float64)
    x, y, z = p.T
    broad = .013*np.sin(.13*x+.09*y+.17*z) + .010*np.sin(.31*x-.19*y+.12*z)
    layers = .006*np.sin(1.8*z+.025*x+.018*y)
    value = np.clip(.113+broad+layers, .07, .18)
    return np.clip(value[:, None]*np.array([.97, .99, 1.03]), .07, .18).astype(np.float32)


def _value_noise(points, spacing, seed):
    p = np.asarray(points, dtype=np.float64)/float(spacing)
    cell = np.floor(p).astype(np.int64)
    frac = p-cell
    frac = frac*frac*(3.-2.*frac)
    result = np.zeros(len(p), dtype=np.float64)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                q = cell+np.array([a,b,c], np.int64)
                h = (q[:,0]*73856093) ^ (q[:,1]*19349663) ^ (q[:,2]*83492791) ^ int(seed)
                h = (h ^ (h >> 13))*1274126177
                random = (h & 65535)/32767.5-1.
                weight = ((frac[:,0] if a else 1.-frac[:,0]) *
                          (frac[:,1] if b else 1.-frac[:,1]) *
                          (frac[:,2] if c else 1.-frac[:,2]))
                result += weight*random
    return result


def _continuous_terrain_color(points, global_normals):
    """One smooth albedo field, evaluated before any material splitting."""
    n = np.asarray(global_normals, dtype=np.float64)
    nz = n[:,2]/np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    blend = np.clip((nz-.12)/.76, 0., 1.)
    blend = blend*blend*(3.-2.*blend)
    basalt = np.array([.092, .099, .111])
    dust = np.array([.125, .128, .134])
    variation = .005*_value_noise(points,9.,503) + .003*_value_noise(points,3.1,937)
    color = basalt+(dust-basalt)*blend[:,None]+variation[:,None]
    return np.clip(color,.07,.18).astype(np.float32)


def _vertex_material(stage, path, fallback, roughness=.95):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    reader = UsdShade.Shader.Define(stage, path+'/WorldColor')
    reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
    reader.CreateInput('fallback', Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*fallback))
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), 'result')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def _basalt_material(stage):
    return _vertex_material(stage, '/World/Materials/Basalt', (.105,.108,.115))


def _revision_two_materials(stage, output, config):
    basalt = _basalt_material(stage)
    regolith = _vertex_material(stage, '/World/Materials/Regolith', (.125,.128,.134), .97)
    texture_dir = output.parent/'textures'
    texture_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    strengths = dict(wall=float(config.get('wall_normal_strength', .12)),
                     floor=float(config.get('floor_normal_strength', .28)))
    if any(not 0. <= s <= .35 for s in strengths.values()):
        raise ValueError('Micro normal strengths must be in [0,.35]')
    for material, name, strength in [(basalt,'dark_rock',strengths['wall']),
                                      (regolith,'moon_03',strengths['floor'])]:
        path = str(material.GetPath())
        source = ROOT/'assets/textures'/f'{name}_nor_gl_2k.jpg'
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = texture_dir/source.name
        if destination.exists():
            if _sha(destination) != _sha(source):
                raise FileExistsError('Refusing to replace different texture '+str(destination))
        else:
            shutil.copy2(source,destination)
        uv = UsdShade.Shader.Define(stage,path+'/UV')
        uv.CreateIdAttr('UsdPrimvarReader_float2')
        uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage,path+'/MicroNormal')
        texture.CreateIdAttr('UsdUVTexture')
        relative = 'textures/'+source.name
        texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(relative))
        texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
        texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw')
        texture.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
        texture.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
        texture.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2*strength,2*strength,2*strength,1.))
        texture.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-strength,-strength,1-2*strength,0.))
        surface = UsdShade.Shader.Get(stage,path+'/Surface')
        surface.CreateInput('normal',Sdf.ValueTypeNames.Normal3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
        assets.append(dict(relative_asset=relative,source=str(source),sha256=_sha(source),
                           shader_path=str(texture.GetPath()),normal_strength=strength))
    return basalt,regolith,assets,dict(diffuse='shared continuous vertex displayColor',
        albedo_texture=False,normal_strength=strengths['floor'],roughness=.97,
        texture_is_laboratory_lunar_regolith_analogue=True)


def _floor_material(stage, output, config):
    path = '/World/Materials/Regolith'
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.97)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    uv = UsdShade.Shader.Define(stage, path+'/UV')
    uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    uv.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    texture_dir = output.parent/'textures'
    texture_dir.mkdir(parents=True, exist_ok=True)
    diffuse_scale = float(config.get('floor_diffuse_scale', .20))
    normal_strength = float(config.get('floor_normal_strength', .18))
    if not 0. < diffuse_scale <= .35 or not 0. <= normal_strength <= .35:
        raise ValueError('Floor diffuse/normal strengths must stay within muted inspection defaults')
    assets = []
    for suffix, channel, target, color_space in [
            ('diff', 'rgb', 'diffuseColor', 'sRGB'),
            ('rough', 'r', 'roughness', 'raw'),
            ('nor_gl', 'rgb', 'normal', 'raw')]:
        source = ROOT/'assets/textures'/f'moon_03_{suffix}_2k.jpg'
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = texture_dir/source.name
        if destination.exists():
            if _sha(destination) != _sha(source):
                raise FileExistsError('Refusing to replace different texture '+str(destination))
        else:
            shutil.copy2(source, destination)
        texture = UsdShade.Shader.Define(stage, path+'/'+suffix)
        texture.CreateIdAttr('UsdUVTexture')
        relative = 'textures/'+source.name
        texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(relative))
        texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
        texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(color_space)
        for wrap in ['wrapS', 'wrapT']:
            texture.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set('repeat')
        if suffix == 'diff':
            texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(diffuse_scale, diffuse_scale, diffuse_scale, 1.))
            dtype = Sdf.ValueTypeNames.Color3f
        elif suffix == 'rough':
            texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.08, .08, .08, 1.))
            texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.92, .92, .92, 0.))
            dtype = Sdf.ValueTypeNames.Float
        else:
            s = normal_strength
            texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2*s, 2*s, 2*s, 1.))
            texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-s, -s, 1-2*s, 0.))
            dtype = Sdf.ValueTypeNames.Normal3f
        shader.CreateInput(target, dtype).ConnectToSource(texture.ConnectableAPI(), channel)
        assets.append(dict(relative_asset=relative, source=str(source), sha256=_sha(source),
                           shader_path=str(texture.GetPath())))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material, assets, dict(diffuse_scale=diffuse_scale, normal_strength=normal_strength,
                                  roughness_range=[.92, 1.], texture_is_laboratory_lunar_regolith_analogue=True)


def _compact(vertices, faces, normals):
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3).astype(np.int32), normals[used], used


def _author_mesh(stage, path, vertices, faces, normals, material, tile, colors=None, normal_interpolation='vertex'):
    result = write_mesh(stage, path, vertices, faces, material, collision=True, tile=tile)
    result.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    result.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
    result.SetNormalsInterpolation(normal_interpolation)
    if colors is None:
        colors = _basalt_color(vertices)
    result.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(colors))
    result.CreateExtentAttr(Vt.Vec3fArray([
        Gf.Vec3f(*map(float, vertices.min(0))), Gf.Vec3f(*map(float, vertices.max(0)))]))
    UsdPhysics.CollisionAPI(result.GetPrim()).CreateCollisionEnabledAttr(True)
    return result


def _boulder_mesh(boulders, hard=False, detail_revision=0):
    prototype = trimesh.creation.icosphere(subdivisions=2 if detail_revision>=10 else (0 if hard else 1), radius=1.)
    vertices, faces = [], []
    offset = 0
    for index, item in enumerate(boulders):
        center = np.asarray(item['center_xyz_m'], float)
        scale = np.asarray(item['scale_xyz_m'], float)
        if center.shape != (3,) or scale.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Boulder centers/scales must be finite xyz with positive radii')
        rng = np.random.default_rng(int(item.get('seed', index)))
        v = np.asarray(prototype.vertices)*rng.uniform(.65 if hard else .78, 1.18, (len(prototype.vertices), 1))
        if hard:
            v += rng.normal(0.,.035 if detail_revision>=10 else .07,v.shape)
            if detail_revision>=10:
                lengths=np.linalg.norm(v,axis=1)
                v*=np.minimum(1.,1.24/np.maximum(lengths,1e-12))[:,None]
            rotation,_ = np.linalg.qr(rng.normal(size=(3,3)))
            v = v@rotation
            hull = trimesh.convex.convex_hull(v*scale)
            v = np.asarray(hull.vertices)+center
            local_faces = np.asarray(hull.faces)
        else:
            v = v*scale+center
            local_faces = np.asarray(prototype.faces)
        vertices.append(v)
        faces.append(local_faces+offset)
        offset += len(v)
    v = np.asarray(np.concatenate(vertices), np.float32)
    f = np.asarray(np.concatenate(faces), np.int32)
    tm=trimesh.Trimesh(v,f,process=False)
    n = (np.repeat(np.asarray(tm.face_normals,np.float32),3,axis=0) if hard
         else np.asarray(tm.vertex_normals,np.float32))
    return v, f, n


def write_scene(output_file, vertices, faces, config, normals=None, boulders=None):
    """Write/reopen an open static terrain patch; scale_xyz_m are boulder radii.

    Faces/normals of terrain must point into air. Upward faces use muted fine
    regolith; other faces use vertex-colored basalt without a crack texture.
    """
    output = Path(output_file).expanduser().absolute()
    if output.exists():
        raise FileExistsError('Refusing to overwrite scene '+str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError('vertices must be finite Nx3')
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('faces must be nonempty integer Fx3')
    faces = faces.astype(np.int32)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('face index outside vertex range')
    tri = vertices[faces].astype(float)
    fn = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    length = np.linalg.norm(fn, axis=1)
    if not np.isfinite(length).all() or np.any(length <= 0):
        raise ValueError('Final float32 mesh contains zero-area or nonfinite faces')
    face_normals = fn/length[:, None]
    if normals is None:
        normals = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
    normals = np.asarray(normals, np.float32)
    if normals.shape != vertices.shape or not np.isfinite(normals).all():
        raise ValueError('global normals must be finite Nx3 matching vertices')
    used = np.unique(faces)
    if np.any(np.linalg.norm(normals[used], axis=1) < .9):
        raise ValueError('used global normals must be approximately unit vectors')
    floor = face_normals[:, 2] > .5
    appearance_revision = int(config.get('appearance_revision',1))
    continuous_albedo = appearance_revision >= 2
    global_colors = (_continuous_terrain_color(vertices,normals) if continuous_albedo
                     else _basalt_color(vertices))
    stage = Usd.Stage.CreateNew(str(output))
    if not stage:
        raise RuntimeError('Cannot create USD stage')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    for name in ['Materials', 'Cave']:
        UsdGeom.Xform.Define(stage, '/World/'+name)
    world.GetPrim().CreateAttribute('cave:scientific_scope', Sdf.ValueTypeNames.String, custom=True).Set(
        'synthetic pit and Zhou-inspired cavity; not a reconstruction of a single surveyed site')
    world.GetPrim().CreateAttribute('cave:boundary_scope', Sdf.ValueTypeNames.String, custom=True).Set(
        'open terrain patch; perimeter boundaries expected; caller validates geometry')
    world.GetPrim().CreateAttribute('cave:config_json', Sdf.ValueTypeNames.String, custom=True).Set(json.dumps(config, default=_json, ensure_ascii=False))
    if continuous_albedo:
        basalt,regolith,textures,floor_settings = _revision_two_materials(stage,output,config)
        micro_tile=float(config.get('micro_texture_tile_m',.65))
        if not .25 <= micro_tile <= 1.:
            raise ValueError('Revision-two micro texture tile must be in [.25,1]m')
    else:
        basalt = _basalt_material(stage)
        regolith,textures,floor_settings = _floor_material(stage,output,config)
        micro_tile = None
    expected = {}
    for name, selection, material, tile in [
            ('BasaltWalls', ~floor, basalt, micro_tile if continuous_albedo else 4.),
            ('RegolithFloorAndSurface', floor, regolith, micro_tile if continuous_albedo else float(config.get('floor_texture_tile_m', 3.)))]:
        if not np.any(selection):
            continue
        v, f, n, indices = _compact(vertices, faces[selection], normals)
        path = '/World/Cave/'+name
        colors=global_colors[indices]
        _author_mesh(stage,path,v,f,n,material,tile,colors=colors)
        expected[path] = dict(vertices=v,faces=f,normals=n,material=str(material.GetPath()),
                              colors=colors,normal_interpolation='vertex',global_vertex_indices=indices)
    boulders = [] if boulders is None else list(boulders)
    if boulders:
        v, f, n = _boulder_mesh(boulders,hard=continuous_albedo,detail_revision=int(config.get('boulder_detail_revision',0)))
        path = '/World/Cave/AngularCollapseBlocks'
        block_colors=_basalt_color(v)
        interpolation='faceVarying' if continuous_albedo else 'vertex'
        block = _author_mesh(stage,path,v,f,n,basalt,micro_tile if continuous_albedo else 4.,
                             colors=block_colors,normal_interpolation=interpolation)
        block.GetPrim().CreateAttribute('cave:scope', Sdf.ValueTypeNames.String, custom=True).Set(
            'authored static collapse-block geometry; positions supplied by caller; no dynamics')
        expected[path] = dict(vertices=v,faces=f,normals=n,material=str(basalt.GetPath()),
                              colors=block_colors,normal_interpolation=interpolation)
    physics = UsdPhysics.Scene.Define(stage, '/World/Physics')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(1.62)
    stage.GetRootLayer().customLayerData = dict(authoring='pure_cpu_pxr', boundary_scope='open_terrain_patch',
        scientific_scope='synthetic_visual_hypothesis', material_scope='geometry_color_basalt_and_lab_regolith_analogue')
    stage.GetRootLayer().Save()
    del stage
    reopened = Usd.Stage.Open(str(output))
    if not reopened:
        raise RuntimeError('Saved USD cannot reopen')
    mesh_reports = []
    for path, wanted in expected.items():
        saved = UsdGeom.Mesh.Get(reopened, path)
        p = np.asarray(saved.GetPointsAttr().Get(), np.float32)
        raw_counts = saved.GetFaceVertexCountsAttr().Get()
        raw_indices = saved.GetFaceVertexIndicesAttr().Get()
        counts = np.asarray(raw_counts)
        indices = np.asarray(raw_indices, np.int32)
        topology_ok, reason = UsdGeom.Mesh.ValidateTopology(raw_indices, raw_counts, len(p))
        n = np.asarray(saved.GetNormalsAttr().Get(), np.float32)
        binding, _ = UsdShade.MaterialBindingAPI(saved.GetPrim()).ComputeBoundMaterial()
        if not topology_ok or not np.all(counts == 3) or not np.array_equal(p, wanted['vertices']) or not np.array_equal(indices.reshape(-1, 3), wanted['faces']):
            raise ValueError('Serialized USD mesh differs from input: '+path+' '+reason)
        if not np.array_equal(n,wanted['normals']) or saved.GetNormalsInterpolation()!=wanted['normal_interpolation']:
            raise ValueError('Global-normal slices were not preserved: '+path)
        if not binding or str(binding.GetPath()) != wanted['material'] or not saved.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Missing material/collision binding: '+path)
        colors = np.asarray(saved.GetDisplayColorPrimvar().Get(), np.float32)
        if colors.shape != p.shape or not np.isfinite(colors).all() or colors.min() < .069 or colors.max() > .181:
            raise ValueError('Dark basalt displayColor is invalid: '+path)
        if not np.array_equal(colors,wanted['colors']):
            raise ValueError('Serialized continuous colors differ from global field: '+path)
        mesh_reports.append(dict(path=path, vertices=len(p), triangles=len(counts), ValidateTopology=True,
                                 preserved_global_normals=True,normal_interpolation=str(saved.GetNormalsInterpolation()),
                                 static_collision_authored=True,material=wanted['material']))
    resolved_textures = []
    for item in textures:
        texture = UsdShade.Shader.Get(reopened, item['shader_path'])
        value = texture.GetInput('file').Get()
        resolved = Path(Sdf.ComputeAssetPathRelativeToLayer(reopened.GetRootLayer(), value.path)).resolve()
        if not resolved.is_file() or _sha(resolved) != item['sha256']:
            raise ValueError('Texture reference did not resolve to copied source')
        resolved_textures.append(dict(item, resolved_path=str(resolved)))
    if UsdGeom.GetStageMetersPerUnit(reopened) != 1. or str(UsdGeom.GetStageUpAxis(reopened)) != 'Z':
        raise ValueError('Incorrect stage units/up axis')
    if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in reopened.Traverse()):
        raise ValueError('Static display scene must not create rigid bodies')
    basalt_shader = UsdShade.Shader.Get(reopened, '/World/Materials/Basalt/Surface')
    reader, source_name, _ = basalt_shader.GetInput('diffuseColor').GetConnectedSource()
    if str(reader.GetPrim().GetPath()) != '/World/Materials/Basalt/WorldColor' or source_name != 'result':
        raise ValueError('Basalt color reader is not connected')
    if continuous_albedo:
        shader=UsdShade.Shader.Get(reopened,'/World/Materials/Regolith/Surface')
        reader,source_name,_=shader.GetInput('diffuseColor').GetConnectedSource()
        if str(reader.GetPrim().GetPath())!='/World/Materials/Regolith/WorldColor' or source_name!='result':
            raise ValueError('Regolith must also use continuous vertex albedo')
    return dict(passed=True, output=str(output), bytes=output.stat().st_size,
        stage_reopened=True, meters_per_unit=1., up_axis='Z', gravity_mps2=1.62, gravity_direction=[0,0,-1],
        mesh_reports=mesh_reports, textures=resolved_textures,
        material_paths=dict(wall='/World/Materials/Basalt', floor_and_surface='/World/Materials/Regolith'),
        terrain_triangle_count=len(faces), floor_and_surface_triangles=int(floor.sum()), wall_triangles=int((~floor).sum()),
        boulder_count=len(boulders), boulder_scale_semantics='unit-radius ico three-axis radii, before deterministic radial perturbation',
        appearance_revision=appearance_revision,
        wall_material=dict(diffuse='shared continuous vertex displayColor' if continuous_albedo else 'world-coordinate low-frequency vertex displayColor',
                           linear_color_bounds=[.07,.18],normal_texture=continuous_albedo,
                           normal_strength=float(config.get('wall_normal_strength',.12)) if continuous_albedo else 0.,
                           crack_albedo_texture=False,roughness=.95),
        floor_material=floor_settings, terrain_normals_shared_before_material_split=True,
        albedo_shared_before_material_split=continuous_albedo,micro_texture_tile_m=micro_tile,
        boulder_normals='flat faceVarying convex fragments' if continuous_albedo else 'smooth vertex normals',
        boundary_scope='open terrain patch; no watertightness assertion',
        validation_scope='USD serialization/topology/material references only; no PhysX cooking, dynamics, navigation or structural test',
        GPU_work=False, simulation_started=False, rigid_body_count=0)
