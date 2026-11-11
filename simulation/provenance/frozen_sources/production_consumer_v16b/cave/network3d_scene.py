"""CPU-only USD authoring for a hypothesized closed lunar cave-air boundary.

This module does not import Kit, SimulationApp, a renderer, or a robot controller.
"""
from pathlib import Path
import hashlib
import shutil

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from .scene import ROOT, material, mesh

MESH_PATH = '/World/Cave/AirBoundary'
MATERIAL_PATH = '/World/Materials/DarkRockBasaltAppearanceProxy'
ASSUMPTIONS = {
    'artifact_status': 'candidate_hypothesized',
    'entrance_status': 'no_observed_entrance',
    'endpoint_status': 'artificial_caps_not_geological_endpoints',
    'material_status': 'terrestrial_dark_rock_visual_proxy_not_verified_lunar_basalt',
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _signed_volume(vertices, faces):
    # Centering improves precision without changing a closed shell's signed volume.
    center = (vertices.min(0).astype(np.float64) + vertices.max(0)) * .5
    volume = 0.
    for start in range(0, len(faces), 100000):
        triangles = vertices[faces[start:start + 100000]].astype(np.float64) - center
        volume += np.einsum('ij,ij->i', triangles[:, 0],
                            np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.
    return float(volume)


def orient_air_boundary(vertices, faces):
    """Require one closed consistent shell and make its normals face the air."""
    v = np.asarray(vertices, dtype=np.float32)
    original_faces = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError('Require finite Vx3 vertices for a closed cave')
    if original_faces.ndim != 2 or original_faces.shape[1] != 3 or len(original_faces) < 4:
        raise ValueError('Require triangular Fx3 faces')
    if not np.issubdtype(original_faces.dtype, np.integer):
        raise ValueError('Face indices must be integers')
    if len(v) > np.iinfo(np.int32).max or original_faces.min() < 0 or original_faces.max() >= len(v):
        raise ValueError('Face indices are outside the USD mesh point range')
    f = np.asarray(original_faces, dtype=np.int32)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if not tm.is_watertight or not tm.is_winding_consistent or tm.body_count != 1:
        raise ValueError('Cave must have one connected, watertight, consistently wound air boundary')
    if not np.isfinite(tm.area_faces).all() or np.any(tm.area_faces <= 0):
        raise ValueError('Degenerate or nonfinite triangles cannot define the air boundary')
    volume = _signed_volume(v, f)
    if not np.isfinite(volume) or abs(volume) <= 1e-9:
        raise ValueError('Cave boundary has no finite enclosed volume')
    flipped = volume > 0
    if flipped:
        f = f[:, [0, 2, 1]].copy()
    return v, f, dict(watertight=True, winding_consistent=True, connected_bodies=1,
        input_signed_volume_m3=volume, exported_signed_volume_m3=-abs(volume),
        faces_flipped=bool(flipped), normals_direction='toward_enclosed_air',
        convention='negative signed volume for the single closed air-domain boundary')


def _copy_textures(output_dir):
    target = output_dir / 'textures'
    target.mkdir(exist_ok=False)
    sources = ROOT / 'assets/textures'
    result = []
    for suffix in ['diff', 'rough', 'nor_gl']:
        source = sources / ('dark_rock_' + suffix + '_2k.jpg')
        if not source.is_file():
            raise FileNotFoundError('Missing existing dark-rock PBR texture: ' + str(source))
        destination = target / source.name
        shutil.copy2(source, destination)
        source_sha = _sha(source)
        if _sha(destination) != source_sha:
            raise IOError('Copied texture differs from its existing source')
        result.append(dict(suffix=suffix, source=str(source),
            relative_asset='textures/' + source.name, bytes=destination.stat().st_size,
            sha256=source_sha))
    return result


def _author_stage(output_file, network, vertices, faces, textures):
    stage = Usd.Stage.CreateNew(str(output_file))
    if not stage:
        raise RuntimeError('USD could not create the new stage')
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    for name in ['Materials', 'Cave']:
        UsdGeom.Xform.Define(stage, '/World/' + name)
    for name, value in ASSUMPTIONS.items():
        world.GetPrim().CreateAttribute('cave:' + name, Sdf.ValueTypeNames.String, custom=True).Set(value)
    world.GetPrim().CreateAttribute('cave:no_observed_entrance', Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    world.GetPrim().CreateAttribute('cave:artificial_caps_not_geological_endpoints',
        Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    world.GetPrim().CreateAttribute('cave:network_id', Sdf.ValueTypeNames.String, custom=True).Set(
        str(network.get('candidate_id', network.get('id', network.get('name', 'candidate_network')))))
    stage.GetRootLayer().customLayerData = dict(ASSUMPTIONS, authoring='pure_cpu_pxr',
        no_observed_entrance=True, artificial_caps_not_geological_endpoints=True)
    rock = material(stage, 'DarkRockBasaltAppearanceProxy', (.08, .08, .08), .91, texture='dark_rock')
    for item in textures:
        shader = UsdShade.Shader.Get(stage, MATERIAL_PATH + '/' + item['suffix'])
        shader.GetInput('file').Set(Sdf.AssetPath(item['relative_asset']))
    shell = mesh(stage, MESH_PATH, vertices, faces, rock, collision=True, tile=2.4)
    shell.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    shell.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, vertices.min(0))),
                                       Gf.Vec3f(*map(float, vertices.max(0)))]))
    UsdPhysics.CollisionAPI(shell.GetPrim()).CreateCollisionEnabledAttr(True)
    physics = UsdPhysics.Scene.Define(stage, '/World/Physics')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
    physics.CreateGravityMagnitudeAttr(1.62)
    stage.GetRootLayer().Save()


def validate_network_usd(output_file, expected_vertices, expected_faces, textures):
    """Reopen serialized USD; validate authored data/bindings, not a simulation."""
    output_file = Path(output_file)
    stage = Usd.Stage.Open(str(output_file))
    if not stage:
        raise RuntimeError('Reopening the saved USD stage failed')
    shell = UsdGeom.Mesh.Get(stage, MESH_PATH)
    if not shell:
        raise RuntimeError('Reopened stage does not contain the cave boundary mesh')
    points = np.asarray(shell.GetPointsAttr().Get(), dtype=np.float32)
    counts = shell.GetFaceVertexCountsAttr().Get()
    indices = shell.GetFaceVertexIndicesAttr().Get()
    topology_ok, topology_reason = UsdGeom.Mesh.ValidateTopology(indices, counts, len(points))
    if not topology_ok or not np.all(np.asarray(counts) == 3):
        raise ValueError('USD ValidateTopology rejected the cave: ' + topology_reason)
    faces = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
    if not np.array_equal(points, expected_vertices) or not np.array_equal(faces, expected_faces):
        raise ValueError('Saved USD points/faces differ from the authored cave')
    normal = np.asarray(shell.GetNormalsAttr().Get(), dtype=np.float32)
    normals_ok = normal.shape == points.shape and np.isfinite(normal).all() and np.all(np.linalg.norm(normal, axis=1) > .9)
    signed_volume = _signed_volume(points, faces)
    if not normals_ok or shell.GetNormalsInterpolation() != 'vertex' or signed_volume >= 0:
        raise ValueError('Saved cave normals do not follow the inward air-boundary convention')
    meters = float(UsdGeom.GetStageMetersPerUnit(stage));up = str(UsdGeom.GetStageUpAxis(stage))
    collision = UsdPhysics.CollisionAPI(shell.GetPrim())
    approximation = UsdPhysics.MeshCollisionAPI(shell.GetPrim()).GetApproximationAttr().Get()
    rigid_bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    physics = UsdPhysics.Scene.Get(stage, '/World/Physics')
    gravity = float(physics.GetGravityMagnitudeAttr().Get())
    direction = list(physics.GetGravityDirectionAttr().Get())
    bound_material, _ = UsdShade.MaterialBindingAPI(shell.GetPrim()).ComputeBoundMaterial()
    if meters != 1. or up != 'Z' or abs(gravity - 1.62) > 1e-6 or direction != [0., 0., -1.]:
        raise ValueError('Stage units/up-axis/lunar gravity are incorrect')
    if not shell.GetDoubleSidedAttr().Get() or not shell.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('Cave must be double-sided with static collision authored')
    if not collision.GetCollisionEnabledAttr().Get() or approximation != 'none' or rigid_bodies:
        raise ValueError('Expected enabled static triangle-mesh collision without rigid bodies')
    if not bound_material or str(bound_material.GetPath()) != MATERIAL_PATH:
        raise ValueError('Dark-rock PBR material binding is missing')
    assets = []
    for item in textures:
        shader = UsdShade.Shader.Get(stage, MATERIAL_PATH + '/' + item['suffix'])
        asset = shader.GetInput('file').Get()
        if not isinstance(asset, Sdf.AssetPath) or Path(asset.path).is_absolute():
            raise ValueError('Texture must be an authored relative asset path')
        resolved = Path(Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(), asset.path)).resolve()
        if not resolved.is_file() or resolved != (output_file.parent / item['relative_asset']).resolve():
            raise ValueError('Relative texture does not resolve inside the new scene package')
        assets.append(dict(asset_path=asset.path, resolved_path=str(resolved), exists=True,
                           source_sha256=item['sha256']))
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        stage.GetPrimAtPath('/World')).ComputeAlignedRange()
    bounds = np.array([list(bbox.GetMin()), list(bbox.GetMax())], dtype=float)
    expected_bounds = np.array([expected_vertices.min(0), expected_vertices.max(0)])
    if not np.allclose(bounds, expected_bounds, rtol=0., atol=1e-5):
        raise ValueError('USD world bounds differ from the authored mesh extent')
    return dict(passed=True, stage_reopened=True, usd_version=list(Usd.GetVersion()),
        ValidateTopology=dict(passed=bool(topology_ok), reason=topology_reason),
        vertex_count=len(points), triangle_count=len(faces), bounds_m=bounds.tolist(),
        meters_per_unit=meters, up_axis=up, gravity_mps2=gravity, gravity_direction=direction,
        double_sided=True, normals_interpolation='vertex', normals_toward_air=True,
        signed_volume_m3=signed_volume, static_collision=True, collision_approximation=approximation,
        rigid_body_count=0, material_binding=MATERIAL_PATH, relative_texture_assets=assets,
        physics_steps_executed=0, renders_executed=0, navigation_evaluated=False)


def export_network_usd(network, geometry, output_file):
    """Write one new standalone USD plus copied existing PBR textures; never overwrite."""
    output_file = Path(output_file).absolute()
    if output_file.exists():
        raise FileExistsError('Refusing to overwrite USD: ' + str(output_file))
    if not output_file.parent.is_dir():
        raise FileNotFoundError('USD output parent must be the new CLI output directory')
    vertices, faces, orientation = orient_air_boundary(geometry['vertices'], geometry['faces'])
    textures = _copy_textures(output_file.parent)
    _author_stage(output_file, network, vertices, faces, textures)
    validation = validate_network_usd(output_file, vertices, faces, textures)
    return dict(vertices=vertices, faces=faces, report=dict(assumptions=ASSUMPTIONS,
        surface_orientation=orientation, usd_validation=validation,
        usd_file=output_file.name, usd_bytes=output_file.stat().st_size,
        texture_source_manifest=str(ROOT / 'assets/ASSET_SOURCES.json')))
