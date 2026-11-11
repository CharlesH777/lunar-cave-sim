"""Add static collision geometry for existing, visible lunar gravel instances.

The visible PointInstancer is read-only. Its composed USD instance matrices
(including authored half-precision quaternions and prototype transforms) are
baked into one hidden triangle mesh. This is geometry validation, not a PhysX
cooking or wheel-dynamics test.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial import ConvexHull
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt


INSTANCER_PATH = '/World/SurfaceDetail/Gravel'
COLLIDER_PATH = '/World/Collision/GravelObstaclesV7'


def _settings(config):
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise TypeError('Gravel collision configuration must be a dict')
    settings = config.get('gravel_collision_v7', config)
    if not isinstance(settings, dict):
        raise TypeError('gravel_collision_v7 must be a dict')
    threshold = float(settings.get('min_full_diameter_m', .25))
    tolerance = float(settings.get('max_vertex_error_m', 2e-5))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('min_full_diameter_m must be positive and finite')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('max_vertex_error_m must be positive and finite')
    return dict(min_full_diameter_m=threshold, max_vertex_error_m=tolerance)


def _visual_digest(instancer):
    """Detect any accidental authoring to the visible source subtree."""
    digest = hashlib.sha256()
    for prim in Usd.PrimRange(instancer.GetPrim()):
        digest.update(str(prim.GetPath()).encode())
        for attr in sorted(prim.GetAttributes(), key=lambda a: a.GetName()):
            digest.update(str((attr.GetName(), attr.Get())).encode())
        for rel in sorted(prim.GetRelationships(), key=lambda r: r.GetName()):
            digest.update(str((rel.GetName(), rel.GetTargets())).encode())
    return digest.hexdigest()


def _prototype(stage, path):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    if not mesh:
        raise ValueError(f'Gravel prototype must be a Mesh: {path}')
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError(f'Invalid prototype points: {path}')
    if counts.ndim != 1 or not len(counts) or np.any(counts != 3) or len(indices) != 3 * len(counts):
        raise ValueError(f'Gravel prototype must contain triangles: {path}')
    faces = indices.reshape(-1, 3).copy()
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f'Prototype triangle indices are out of range: {path}')
    if mesh.GetSubdivisionSchemeAttr().Get() != UsdGeom.Tokens.none:
        raise ValueError(f'Cannot bake a subdivided prototype as its control mesh: {path}')
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        faces = faces[:, [0, 2, 1]]
    triangles = vertices[faces]
    if np.any(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                     triangles[:, 2] - triangles[:, 0]), axis=1) < 1e-14):
        raise ValueError(f'Degenerate prototype triangle: {path}')
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected, frequency = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    oriented = {tuple(e): 0 for e in undirected}
    for a, b in edges:
        oriented[tuple(sorted((a, b)))] += 1 if a < b else -1
    if np.any(frequency != 2) or any(oriented.values()):
        raise ValueError(f'Gravel prototype must be a closed consistently wound mesh: {path}')
    a, b = np.triu_indices(len(vertices), 1)
    return dict(vertices=vertices, faces=faces,
                pairs=vertices[a] - vertices[b], visible=(
                    mesh.ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible))


def _world_points(vertices, matrices):
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    return np.einsum('vj,ijk->ivk', homogeneous, matrices)[:, :, :3]


def _xy_hull(points):
    xy = np.unique(points[:, :2], axis=0)
    # Actual gravel is volumetric. Keep a deterministic conservative fallback
    # for a degenerate projection, rather than fail a valid static transform.
    if len(xy) < 3 or np.linalg.matrix_rank(xy - xy[0]) < 2:
        return xy.tolist()
    return xy[ConvexHull(xy).vertices].tolist()


def _require_static_ancestors(prim):
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid = UsdPhysics.RigidBodyAPI(prim)
            if rigid.GetRigidBodyEnabledAttr().Get() is not False:
                raise ValueError(f'Collision mesh would inherit an enabled rigid body: {prim.GetPath()}')
        prim = prim.GetParent()


def add_gravel_collision(stage, config=None):
    """Bake >=25cm gravel into one hidden static collider; return JSON data.

    ``config`` may be a direct settings dict or contain ``gravel_collision_v7``.
    Optional keys are ``min_full_diameter_m`` (default .25) and
    ``max_vertex_error_m`` (default 2e-5). Thresholding uses actual longest
    vertex-to-vertex world distance, not a nominal scale or an AABB diagonal.

    This authors into the caller's current USD edit target. It never saves or
    changes source files, and refuses to replace an existing collider. Existing
    macro boulders are not duplicated; report ``macro_boulders_included`` is
    false. All arrays in the return value use world metres.
    """
    settings = _settings(config)
    time = Usd.TimeCode.Default()
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(INSTANCER_PATH))
    if not instancer:
        raise ValueError(f'Missing gravel PointInstancer: {INSTANCER_PATH}')
    if stage.GetPrimAtPath(COLLIDER_PATH):
        raise ValueError(f'Refusing to replace existing gravel collider: {COLLIDER_PATH}')
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError('Gravel collision expects a stage whose metresPerUnit is 1')
    before = _visual_digest(instancer)
    proto_index = np.asarray(instancer.GetProtoIndicesAttr().Get(), dtype=np.int64)
    count = len(proto_index)
    ids_value = instancer.GetIdsAttr().Get()
    ids = np.arange(count, dtype=np.int64) if ids_value is None else np.asarray(ids_value, dtype=np.int64)
    if ids.shape != (count,):
        raise ValueError('Gravel IDs and protoIndices have different lengths')
    paths = instancer.GetPrototypesRel().GetTargets()
    if proto_index.shape != (count,) or (count and (proto_index.min() < 0 or proto_index.max() >= len(paths))):
        raise ValueError('Invalid gravel prototype indices')
    prototypes = [_prototype(stage, path) for path in paths]
    matrices_gf = instancer.ComputeInstanceTransformsAtTime(
        time, time, UsdGeom.PointInstancer.IncludeProtoXform, UsdGeom.PointInstancer.IgnoreMask)
    matrices = np.asarray(matrices_gf, dtype=float).reshape(-1, 4, 4)
    if matrices.shape != (count, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError('USD did not produce one finite matrix per original instance')
    instancer_world_gf = UsdGeom.Xformable(instancer.GetPrim()).ComputeLocalToWorldTransform(time)
    matrices_world = matrices @ np.asarray(instancer_world_gf, dtype=float)
    active = np.asarray(instancer.ComputeMaskAtTime(time), dtype=bool)
    if active.size == 0:
        active = np.ones(count, dtype=bool)
    if active.shape != (count,):
        raise ValueError('USD instance mask length does not match original instance count')
    if instancer.ComputeVisibility(time) == UsdGeom.Tokens.invisible:
        active[:] = False
    finite = np.isfinite(matrices_world).all()
    if not finite:
        raise ValueError('Non-finite world instance matrices')

    # Read actual composed geometry before authoring anything. Prototype face
    # normals are deliberately ignored: topology and winding come from indices.
    selected = []
    gf_error = 0.
    gf_samples = 0
    active_count = 0
    for prototype_index, proto in enumerate(prototypes):
        rows = np.flatnonzero((proto_index == prototype_index) & active & proto['visible'])
        active_count += len(rows)
        if not len(rows):
            continue
        transforms = matrices_world[rows]
        world = _world_points(proto['vertices'], transforms)
        distances = np.linalg.norm(np.einsum('vj,ijk->ivk', proto['pairs'], transforms[:, :3, :3]), axis=2)
        diameters = distances.max(axis=1)
        for local_index in np.flatnonzero(diameters >= settings['min_full_diameter_m']):
            instance_index = int(rows[local_index])
            transform = transforms[local_index]
            if abs(np.linalg.det(transform[:3, :3])) < 1e-20:
                raise ValueError(f'Singular transform on selected gravel instance {instance_index}')
            faces = proto['faces']
            # Mirrored world transforms need the same closed solid outward
            # winding after vertices have been baked into the destination mesh.
            if np.linalg.det(transform[:3, :3]) < 0:
                faces = faces[:, [0, 2, 1]]
            selected.append(dict(instance_index=instance_index, prototype_index=prototype_index,
                                 vertices=world[local_index], faces=faces,
                                 center=transform[3, :3].copy(), diameter=float(diameters[local_index])))
        # Independent Gf evaluation checks row/column and parent composition,
        # including authored half quaternions, without regenerating rotations.
        for row in rows[np.unique(np.linspace(0, len(rows) - 1, min(3, len(rows)), dtype=int))]:
            for vertex_index in sorted({0, len(proto['vertices']) // 2, len(proto['vertices']) - 1}):
                point = proto['vertices'][vertex_index]
                expected = instancer_world_gf.Transform(matrices_gf[int(row)].Transform(Gf.Vec3d(*map(float, point))))
                actual = np.r_[point, 1.] @ matrices_world[int(row)]
                gf_error = max(gf_error, float(np.linalg.norm(np.asarray(expected) - actual[:3])))
                gf_samples += 1
    if gf_error > 1e-10:
        raise ValueError(f'Batched matrices disagree with Gf.Transform by {gf_error} metres')
    selected.sort(key=lambda item: item['instance_index'])

    # Existing /World transforms may be non-identity. Store in the collision
    # parent's coordinates, then compare the authored float32 points in world.
    parent_path = Sdf.Path(COLLIDER_PATH).GetParentPath()
    parent_prim = stage.GetPrimAtPath(parent_path)
    if parent_prim:
        _require_static_ancestors(parent_prim)
        parent_gf = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(time)
    else:
        ancestor = stage.GetPrimAtPath(parent_path.GetParentPath())
        _require_static_ancestors(ancestor)
        parent_gf = (UsdGeom.Xformable(ancestor).ComputeLocalToWorldTransform(time)
                     if ancestor else Gf.Matrix4d(1.))
    parent_world = np.asarray(parent_gf, dtype=float)
    inverse_parent = np.linalg.inv(parent_world)
    chunks = []
    face_chunks = []
    obstacles = []
    vertex_count = 0
    max_vertex_error = 0.
    for item in selected:
        visible_world = item['vertices']
        local = (np.column_stack([visible_world, np.ones(len(visible_world))]) @ inverse_parent)[:, :3].astype(np.float32)
        collision_world = (np.column_stack([local.astype(float), np.ones(len(local))]) @ parent_world)[:, :3]
        error = float(np.linalg.norm(collision_world - visible_world, axis=1).max())
        max_vertex_error = max(max_vertex_error, error)
        if error > settings['max_vertex_error_m']:
            raise ValueError(f'Collision float32 geometry error {error}m exceeds tolerance')
        combined = np.vstack([visible_world, collision_world])
        center = item['center']
        chunks.append(local)
        face_chunks.append(item['faces'] + vertex_count)
        obstacles.append(dict(
            instance_id=int(ids[item['instance_index']]), instance_index=item['instance_index'],
            prototype_index=item['prototype_index'], center_xyz_m=center.tolist(),
            xy_radius_m=float(np.linalg.norm(combined[:, :2] - center[:2], axis=1).max()),
            xy_hull_m=_xy_hull(combined), world_aabb_m=[combined.min(0).tolist(), combined.max(0).tolist()],
            z_min_m=float(combined[:, 2].min()), z_max_m=float(combined[:, 2].max()),
            full_diameter_m=item['diameter'], vertex_start=vertex_count,
            vertex_count=len(local), triangle_count=len(item['faces']),
            max_collision_vertex_error_m=error))
        vertex_count += len(local)
    points = np.concatenate(chunks) if chunks else np.empty((0, 3), dtype=np.float32)
    faces = np.concatenate(face_chunks).astype(np.int32) if face_chunks else np.empty((0, 3), dtype=np.int32)
    if len(faces):
        tri = points.astype(float)[faces]
        if np.any(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) < 1e-14):
            raise ValueError('Float32 collision serialization collapsed a triangle')
    valid, reason = UsdGeom.Mesh.ValidateTopology(Vt.IntArray.FromNumpy(faces.ravel()),
                                                 Vt.IntArray([3] * len(faces)), len(points))
    if not valid:
        raise ValueError(f'Merged collision topology is invalid: {reason}')
    if _visual_digest(instancer) != before:
        raise RuntimeError('Visible gravel changed during collision preparation')

    # All validation above precedes the sole destination edit.
    if not parent_prim:
        UsdGeom.Xform.Define(stage, parent_path)
    collision = UsdGeom.Mesh.Define(stage, COLLIDER_PATH)
    collision.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    collision.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    collision.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    collision.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    collision.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    collision.CreateDoubleSidedAttr(False)
    collision.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    if len(points):
        collision.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0), points.max(0)])))
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr('none')
    collision.GetPrim().CreateAttribute('cave:scope', Sdf.ValueTypeNames.String, custom=True).Set('static_collision_only')
    collision.GetPrim().CreateAttribute('cave:sourceInstancer', Sdf.ValueTypeNames.String, custom=True).Set(INSTANCER_PATH)
    unchanged = _visual_digest(instancer) == before
    if not unchanged or collision.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError('Collision authoring changed visible gravel or introduced a rigid body')

    return dict(
        passed=True, report_version=7, instancer_path=INSTANCER_PATH, collider_path=COLLIDER_PATH,
        threshold_full_diameter_m=settings['min_full_diameter_m'],
        source_instance_count=count, visible_active_instance_count=int(active_count),
        obstacle_count=len(obstacles), excluded_render_detail_count=int(active_count - len(obstacles)),
        merged_collision_mesh_count=1, collision_vertex_count=len(points), collision_triangle_count=len(faces),
        macro_boulders_included=False, static_collision=True, collision_enabled=True,
        approximation='none', rigid_body_api=False, collider_visibility='invisible',
        visible_instancer_unchanged=unchanged, visible_source_digest_sha256=before,
        maximum_collision_vertex_error_m=max_vertex_error,
        maximum_allowed_vertex_error_m=settings['max_vertex_error_m'],
        gf_transform_sample_vertex_count=gf_samples, maximum_gf_transform_error_m=gf_error,
        threshold_method='longest vertex-to-vertex world distance from composed USD matrices',
        transform_method='ComputeInstanceTransformsAtTime IncludeProtoXform IgnoreMask, then instancer local-to-world; authored masks applied by original index',
        footprint_method='convex XY hull and center radius enclosing both visible and serialized collision vertices',
        validation_scope='CPU USD geometry/topology/transform checks only; PhysX cooking and rover dynamics not tested',
        obstacles=obstacles)
