"""Pure NumPy display-only clipping for three-dimensional cave review renders.

The cut surface is an inspection choice, not a modeled collapse or a new
collision asset. No file I/O, renderer or source-asset mutation occurs here.
"""
from __future__ import annotations

import numpy as np


def _floor_relative_vertex_heights(network, vertices):
    """Height in the up frame of the nearest bounded 3D floor-line segment."""
    nearest_squared = np.full(len(vertices), np.inf)
    heights = np.zeros(len(vertices), dtype=np.float64)
    segment_count = 0
    for edge in network.get('edges', []):
        points = np.asarray(edge['floor_centerline_xyz_m'], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Floor centerlines must contain finite three-dimensional points')
        for a, b in zip(points[:-1], points[1:]):
            direction = b-a
            length_squared = float(direction @ direction)
            if length_squared <= 1e-16:
                raise ValueError('Repeated floor-line control points are unsupported')
            tangent = direction / np.sqrt(length_squared)
            up = np.array([0., 0., 1.])-tangent[2]*tangent
            if np.linalg.norm(up) < 1e-8:
                reference = np.eye(3)[int(np.argmin(np.abs(tangent)))]
                up = reference-float(reference @ tangent)*tangent
            up /= np.linalg.norm(up)
            relative = vertices-a
            fraction = np.clip((relative @ direction)/length_squared, 0., 1.)
            delta = relative-fraction[:, None]*direction
            distance_squared = np.einsum('ij,ij->i', delta, delta)
            better = distance_squared < nearest_squared
            heights[better] = delta[better] @ up
            nearest_squared[better] = distance_squared[better]
            segment_count += 1
    if not segment_count:
        raise ValueError('Network has no floor centerline segments')
    return heights, segment_count


def smooth_height_cutaway(network, verticesNx3, facesMx3, uvMx3x2, normalsNx3, height_m=6.0):
    """Clip an open display mesh at interpolated local floor-relative height.

    Returns ``vertices``/``normals`` Nx3 float32, ``faces`` Fx3 int32, ``uv``
    Fx3x2 float32 (face-varying, preserving seams), and a JSON-friendly report.
    Triangle winding is retained. Intersections share global mesh-edge IDs;
    their per-face UVs remain independent. The upper boundary is intentionally
    open and receives no cap. It must not replace the validated collision USD.
    """
    vertices = np.asarray(verticesNx3, dtype=np.float64)
    faces_input = np.asarray(facesMx3)
    uv = np.asarray(uvMx3x2, dtype=np.float64)
    normals = np.asarray(normalsNx3, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('vertices must be a nonempty finite Nx3 array')
    if faces_input.ndim != 2 or faces_input.shape[1] != 3 or not np.issubdtype(faces_input.dtype, np.integer):
        raise ValueError('faces must be an integer Mx3 array')
    faces = faces_input.astype(np.int64, copy=False)
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Face index outside vertex array')
    if uv.shape != (len(faces), 3, 2) or not np.isfinite(uv).all():
        raise ValueError('uv must be finite face-varying Mx3x2')
    if normals.shape != vertices.shape or not np.isfinite(normals).all():
        raise ValueError('normals must be finite Nx3')
    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.any(normal_lengths <= 1e-12):
        raise ValueError('Input normals must have nonzero length')
    normals = normals/normal_lengths[:, None]
    height_m = float(height_m)
    if not np.isfinite(height_m):
        raise ValueError('height_m must be finite')

    heights, segment_count = _floor_relative_vertex_heights(network, vertices)
    inside = heights <= height_m
    face_inside = inside[faces]
    full_inside = np.all(face_inside, axis=1)
    full_outside = ~np.any(face_inside, axis=1)
    crossing_ids = np.flatnonzero(~full_inside & ~full_outside)
    kept_faces = [faces[full_inside]]
    kept_uv = [uv[full_inside]]
    added_vertices, added_normals = [], []
    edge_cache = {}
    boundary_faces, boundary_uv = [], []
    normal_fallbacks = 0

    def intersection(a, b, uv_a, uv_b):
        nonlocal normal_fallbacks
        fraction = float(np.clip((height_m-heights[a])/(heights[b]-heights[a]), 0., 1.))
        texture = uv_a+fraction*(uv_b-uv_a)
        if fraction <= 1e-12:
            return a, texture
        if fraction >= 1.-1e-12:
            return b, texture
        key = (min(a,b), max(a,b))
        index = edge_cache.get(key)
        if index is None:
            index = len(vertices)+len(added_vertices)
            edge_cache[key] = index
            added_vertices.append(vertices[a]+fraction*(vertices[b]-vertices[a]))
            normal = normals[a]+fraction*(normals[b]-normals[a])
            length = float(np.linalg.norm(normal))
            if length <= 1e-12:
                normal = normals[a].copy()
                normal_fallbacks += 1
            else:
                normal = normal/length
            added_normals.append(normal)
        return index, texture

    for face_id in crossing_ids:
        triangle = faces[face_id]
        texture = uv[face_id]
        polygon = []
        for previous, current in [(2,0), (0,1), (1,2)]:
            a,b = int(triangle[previous]),int(triangle[current])
            if inside[a] != inside[b]:
                polygon.append(intersection(a,b,texture[previous],texture[current]))
            if inside[b]:
                polygon.append((b,texture[current]))
        for i in range(1,len(polygon)-1):
            fan = [polygon[0],polygon[i],polygon[i+1]]
            boundary_faces.append([entry[0] for entry in fan])
            boundary_uv.append([entry[1] for entry in fan])
    if boundary_faces:
        kept_faces.append(np.asarray(boundary_faces,dtype=np.int64))
        kept_uv.append(np.asarray(boundary_uv,dtype=np.float64))
    result_faces = np.concatenate(kept_faces,axis=0)
    result_uv = np.concatenate(kept_uv,axis=0)
    result_vertices = np.vstack([vertices,np.asarray(added_vertices).reshape(-1,3)]).astype(np.float32)
    result_normals = np.vstack([normals,np.asarray(added_normals).reshape(-1,3)]).astype(np.float32)
    # Evaluate after the USD float32 conversion. This is an intentionally open
    # display cut; tiny clipped slivers may be discarded and are counted.
    triangles = result_vertices[result_faces].astype(np.float64)
    double_area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    nondegenerate = np.isfinite(double_area) & (double_area > 2e-12)
    repeated = ((result_faces[:,0] == result_faces[:,1]) | (result_faces[:,1] == result_faces[:,2]) |
                (result_faces[:,2] == result_faces[:,0]))
    nondegenerate &= ~repeated
    discarded = int(np.count_nonzero(~nondegenerate))
    result_faces,result_uv = result_faces[nondegenerate],result_uv[nondegenerate]
    used,remap = np.unique(result_faces.ravel(),return_inverse=True)
    result_vertices = result_vertices[used]
    result_normals = result_normals[used]
    result_faces = remap.reshape(-1,3).astype(np.int32)
    result_uv = result_uv.astype(np.float32)
    return dict(vertices=result_vertices,faces=result_faces,uv=result_uv,normals=result_normals,
        report=dict(method='triangle-linear clip by nearest 3D floor-segment relative height',
            display_only=True,not_a_collision_asset=True,not_a_geological_collapse=True,
            upper_boundary_open=True,caps_added=False,height_m=height_m,
            floor_segments=segment_count,input_vertices=len(vertices),input_faces=len(faces),
            fully_retained_input_faces=int(np.count_nonzero(full_inside)),
            fully_removed_input_faces=int(np.count_nonzero(full_outside)),
            crossing_input_faces=len(crossing_ids),shared_edge_intersections=len(edge_cache),
            output_vertices=len(result_vertices),output_faces=len(result_faces),
            discarded_degenerate_output_faces=discarded,interpolated_normal_fallbacks=normal_fallbacks,
            uv_layout='Fx3x2 face-varying; per-face seams retained',
            limitation='Local floor assignment is a nearest-segment display approximation at junctions; scalar is linearly interpolated inside each original triangle'))
