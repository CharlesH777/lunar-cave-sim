"""Synchronized Isaac Sim 5.1 RGB-D and instance rendering for cave search.

Instantiate after SimulationApp, then call capture() after moving the camera.
No learned policy is run here. Instance labels and ``evaluation`` are privileged
supervision: a policy should receive only the modalities selected for its task.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import carb
import omni.replicator.core as rep
import omni.timeline
from pxr import Sdf, Usd, UsdGeom


def _plain(value):
    """Copy annotator metadata into JSON-serializable Python values."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class CaveSensors:
    """Own one render product and synchronized RGB, depth, instance annotators.

    ``camera.K`` maps OpenCV camera coordinates to pixels. Matrices use column
    vectors: ``p_world = world_from_camera @ [x_right,y_down,z_forward,1]``.
    World coordinates and axial depth are metres; the USD scene remains Z-up.
    Frame arrays own their storage and survive subsequent renderer updates.
    ``evaluation.target_visible_pixels`` counts actual rendered instance pixels,
    including only the requested prim and its descendants, with occlusion.
    """

    def __init__(self, stage, camera_path='/World/Cameras/Agent', resolution=(640, 360)):
        if len(resolution) != 2 or any(int(v) != v or v <= 0 for v in resolution):
            raise ValueError('resolution must be positive integer (width,height)')
        self.stage = stage
        self.camera_path = str(camera_path)
        self.resolution = tuple(map(int, resolution))
        self.camera = UsdGeom.Camera.Get(stage, self.camera_path)
        if not self.camera:
            raise ValueError(f'No USD Camera at {self.camera_path}')
        self.meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not np.isfinite(self.meters_per_unit) or self.meters_per_unit <= 0:
            raise ValueError('Invalid stage metersPerUnit')
        self._released = False
        self._render_product = None
        self._render_product_path = None
        self._annotators = {}
        self._fallback_names = {}
        try:
            carb.settings.get_settings().set_bool('/exts/omni.replicator.core/Orchestrator/enabled', True)
            self._render_product = rep.create.render_product(self.camera_path, self.resolution)
            self._render_product_path = str(self._render_product.path)
            for name in ('rgb', 'distance_to_image_plane', 'instance_segmentation'):
                kwargs = {'init_params': {'colorize': False}} if name == 'instance_segmentation' else {}
                annotator = rep.AnnotatorRegistry.get_annotator(name, **kwargs)
                annotator.attach([self._render_product_path])
                self._annotators[name] = annotator
        except Exception:
            self.release()
            raise

    def _camera_metadata(self):
        # Sample parent transforms too; an individual xformOp is not a world pose.
        seconds = float(omni.timeline.get_timeline_interface().get_current_time())
        time_code = Usd.TimeCode(seconds * self.stage.GetTimeCodesPerSecond())
        c = self.camera
        if c.GetProjectionAttr().Get(time_code) != UsdGeom.Tokens.perspective:
            raise ValueError('CaveSensors supports perspective pinhole cameras')
        focal = float(c.GetFocalLengthAttr().Get(time_code))
        aperture = np.array([c.GetHorizontalApertureAttr().Get(time_code),
                             c.GetVerticalApertureAttr().Get(time_code)], dtype=float)
        if focal <= 0 or not np.isfinite(focal) or np.any(aperture <= 0) or not np.all(np.isfinite(aperture)):
            raise ValueError('Invalid focal length or film aperture')
        width, height = self.resolution
        if not np.isclose(aperture[0] / aperture[1], width / height, rtol=1e-4):
            raise ValueError('Camera film and image aspect ratios differ; set the vertical '
                             'aperture to horizontal_aperture * height / width for exact K')
        offsets = np.array([c.GetHorizontalApertureOffsetAttr().Get(time_code),
                            c.GetVerticalApertureOffsetAttr().Get(time_code)], dtype=float)
        fx, fy = np.array([width, height]) * focal / aperture
        cx = width * (.5 - offsets[0] / aperture[0])
        cy = height * (.5 + offsets[1] / aperture[1])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        world_from_usd = np.asarray(UsdGeom.XformCache(time_code).GetLocalToWorldTransform(c.GetPrim()),
                                    dtype=np.float64).T.copy()
        world_from_usd[:3, 3] *= self.meters_per_unit
        # USD cameras: +X right,+Y up,-Z forward. OpenCV: +X right,+Y down,+Z forward.
        world_from_camera = world_from_usd @ np.diag([1., -1., -1., 1.])
        rotation = world_from_camera[:3, :3]
        if not np.all(np.isfinite(world_from_camera)) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError('Camera or camera parent has non-rigid scale/shear; pose is not SE(3)')
        return {'K': K, 'world_from_camera': world_from_camera,
                'world_from_usd_camera': world_from_usd, 'resolution': [width, height],
                'coordinate_convention': 'column vectors; OpenCV +X right,+Y down,+Z forward; world Z-up; metres',
                'depth_convention': 'axial Z in metres, distance_to_image_plane; invalid values retained',
                'pixel_convention': 'top-left origin, u right,v down; principal point (width/2,height/2) at zero aperture offset',
                'focal_length_usd': focal, 'horizontal_aperture_usd': float(aperture[0]),
                'vertical_aperture_usd': float(aperture[1]),
                'lens_units': 'USD camera attributes in tenths of stage length units; K uses their ratios',
                'timeline_seconds': seconds}

    def _target_ids(self, labels, target_path):
        target = Sdf.Path(str(target_path))
        prim = self.stage.GetPrimAtPath(target)
        if not target.IsAbsolutePath() or not prim:
            raise ValueError(f'No target prim at {target_path}')

        def label_paths(value):
            if isinstance(value, str):
                return [value] if value.startswith('/') else []
            if isinstance(value, Mapping):
                return [p for v in value.values() for p in label_paths(v)]
            if isinstance(value, (tuple, list)):
                return [p for v in value for p in label_paths(v)]
            return []

        matches = []
        for identifier, label in labels.items():
            for path in label_paths(label):
                # HasPrefix is prim-boundary aware: kit does not match kit_blue.
                if Sdf.Path.IsValidPathString(path) and Sdf.Path(path).HasPrefix(target):
                    matches.append(int(identifier))
                    break
        if matches:
            return sorted(set(matches))

        # Some adapters provide a bare prim name. Only accept exact, globally
        # unique names; never match category labels or substring object IDs.
        name = prim.GetName()
        if name not in self._fallback_names:
            self._fallback_names[name] = sum(p.GetName() == name for p in self.stage.Traverse()) == 1
        if self._fallback_names[name]:
            exact = [int(k) for k, v in labels.items() if isinstance(v, str) and v == name]
            if len(exact) == 1:
                return exact
        return []

    def capture(self, target_prim_path=None):
        """Render three RTX subframes without advancing physics; return a frame.

        ``target_prim_path`` is optional evaluator input, never used for camera
        aiming or RGB processing. An offscreen or occluded target returns zero.
        Missing annotator data raises RuntimeError instead of saving empty files.
        """
        if self._released:
            raise RuntimeError('CaveSensors has been released')
        rep.orchestrator.step(rt_subframes=3, delta_time=0.)
        metadata = self._camera_metadata()
        width, height = self.resolution
        raw_rgb = self._annotators['rgb'].get_data()
        if raw_rgb is None:
            raise RuntimeError('RGB annotator returned no data')
        rgb = np.asarray(raw_rgb)
        if rgb.shape[:2] != (height, width) or rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8:
            raise RuntimeError(f'Unexpected RGB data: shape={rgb.shape}, dtype={rgb.dtype}')
        raw_depth = self._annotators['distance_to_image_plane'].get_data()
        if raw_depth is None:
            raise RuntimeError('Depth annotator returned no data')
        depth = np.asarray(raw_depth)
        if depth.shape == (height, width, 1):
            depth = depth[..., 0]
        if depth.shape != (height, width) or depth.dtype.kind != 'f':
            raise RuntimeError(f'Unexpected depth data: shape={depth.shape}, dtype={depth.dtype}')
        raw_seg = self._annotators['instance_segmentation'].get_data()
        if not isinstance(raw_seg, Mapping) or 'data' not in raw_seg or 'info' not in raw_seg:
            raise RuntimeError('Instance annotator returned no data/info mapping')
        instance = np.asarray(raw_seg['data'])
        if instance.shape == (height, width, 1):
            instance = instance[..., 0]
        if instance.shape != (height, width) or instance.dtype.kind not in 'ui':
            raise RuntimeError(f'Unexpected instance IDs: shape={instance.shape}, dtype={instance.dtype}')
        info = _plain(raw_seg['info'])
        if not isinstance(info, dict) or 'idToLabels' not in info:
            raise RuntimeError('Instance segmentation is missing idToLabels')
        if isinstance(info['idToLabels'], str):
            info['idToLabels'] = json.loads(info['idToLabels'])
        labels = info['idToLabels']
        if not isinstance(labels, dict):
            raise RuntimeError('Instance idToLabels is not a mapping')
        wanted = self._target_ids(labels, target_prim_path) if target_prim_path is not None else []
        visible = int(np.isin(instance, wanted).sum()) if wanted else 0
        return {'rgb': np.array(rgb[..., :3], copy=True),
                'depth_m': np.array(depth, dtype=np.float32, copy=True),
                'depth_valid': np.isfinite(depth) & (depth > 0),
                'instance_id': np.array(instance, dtype=np.uint32, copy=True),
                'camera': metadata, 'instance_mapping': info,
                'evaluation': {'target_visible_pixels': visible,
                               'target_prim_path': str(target_prim_path) if target_prim_path is not None else None,
                               'matched_instance_ids': wanted}}

    def release(self):
        """Detach owned annotators and remove the render product; idempotent."""
        if self._released:
            return
        self._released = True
        failures = []
        try:
            for annotator in self._annotators.values():
                try:
                    # Replicator 1.12's detach normalizes a scalar HydraTexture,
                    # but misses HydraTexture elements inside a list. Pass paths.
                    annotator.detach([self._render_product_path])
                except Exception as exc:
                    failures.append(exc)
        finally:
            self._annotators.clear()
            if self._render_product is not None:
                self._render_product.destroy()
                self._render_product = None
                self._render_product_path = None
        if failures:
            raise RuntimeError('Failed to detach one or more cave annotators') from failures[0]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
