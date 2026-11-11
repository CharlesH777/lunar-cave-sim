"""Current rendered landmark visibility and strictly past encounter memory."""

from collections.abc import Mapping
import math

import numpy as np


LABELS_ZH = {'astronaut': '宇航员', 'lunar_rover': '月球车',
             'lunar_base': '月球基地', 'chinese_flag': '中国国旗'}


def _label_paths(value):
    """Use the same absolute-prim-path interpretation as CaveSensors."""
    if isinstance(value, str):
        return [value] if value.startswith('/') else []
    if isinstance(value, Mapping):
        return [path for item in value.values() for path in _label_paths(item)]
    if isinstance(value, (list, tuple)):
        return [path for item in value for path in _label_paths(item)]
    return []


class LandmarkEvidence:
    """Own one episode's causal first-encounter state, separate from STOP.

    ``min_pixels`` is a narration threshold. Goal STOP acceptance continues to
    use its own target visibility rule in the collector. Pure class names do
    not establish which scene object generated an instance mask.
    """

    def __init__(self, targets, target_id, min_pixels=48):
        rows = list(targets.values()) if isinstance(targets, Mapping) else list(targets)
        self.targets = {row['id']: {'path': row['prim_path'].rstrip('/'),
                                  'label_zh': row.get('label_zh') or LABELS_ZH.get(row['id'], row['id'])} for row in rows}
        if len(rows) != 4 or set(self.targets) != set(LABELS_ZH) or target_id not in self.targets:
            raise ValueError('Expected the four explicit lunar search objects and a valid goal')
        paths = [row['path'] for row in self.targets.values()]
        if len(set(paths)) != len(paths) or any(not path.startswith('/') for path in paths):
            raise ValueError('Landmark prim paths must be unique absolute paths')
        if not isinstance(min_pixels, int) or isinstance(min_pixels, bool) or min_pixels < 12:
            raise ValueError('Narration visibility threshold must be an integer of at least 12 pixels')
        self.target_id = target_id
        self.min_pixels = min_pixels
        self._seen = {}
        self._last_timestamp = None

    def observe(self, instance_image, instance_mapping, K256, timestamp_s):
        image = np.asarray(instance_image)
        K = np.asarray(K256, dtype=float)
        timestamp = float(timestamp_s)
        if image.ndim != 2 or not image.size or image.dtype.kind not in 'iu':
            raise ValueError('Instance image must be a 2D integer identifier image')
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0. or K[1, 1] <= 0.:
            raise ValueError('Expected finite calibrated camera intrinsics')
        if not math.isfinite(timestamp) or timestamp < 0. or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
            raise ValueError('Landmark observation timestamps must strictly increase')
        if not isinstance(instance_mapping, Mapping):
            raise ValueError('Missing instance identifier to prim-path mapping')
        labels = instance_mapping.get('idToLabels', instance_mapping)
        if not isinstance(labels, Mapping):
            raise ValueError('Missing instance identifier to prim-path mapping')
        identifiers = {name: [] for name in self.targets}
        for identifier, value in labels.items():
            paths = _label_paths(value)
            matches = [name for name, row in self.targets.items()
                       if any(path == row['path'] or path.startswith(row['path'] + '/') for path in paths)]
            if len(matches) == 1:
                try:
                    number = int(identifier)
                except (ValueError, TypeError):
                    continue
                if number >= 0:
                    identifiers[matches[0]].append(number)

        visible = []
        for name, numbers in identifiers.items():
            mask = np.isin(image, numbers) if numbers else np.zeros(image.shape, dtype=bool)
            pixels = int(mask.sum())
            if pixels < self.min_pixels:
                continue
            rows, columns = np.nonzero(mask)
            bbox = [int(columns.min()), int(rows.min()), int(columns.max()+1), int(rows.max()+1)]
            bearing = math.degrees(math.atan2(K[0, 2] - (bbox[0]+bbox[2])*.5, K[0, 0]))
            visible.append(dict(id=name, label_zh=self.targets[name]['label_zh'], is_goal=name == self.target_id,
                pixels=pixels, bbox_xyxy=bbox, bearing_deg=bearing,
                bearing='left' if bearing > 10. else 'right' if bearing < -10. else 'ahead'))
        visible.sort(key=lambda row: row['id'])
        seen_before = sorted(self._seen)
        first = [row['id'] for row in visible if row['id'] not in self._seen]
        for name in first:
            self._seen[name] = timestamp
        self._last_timestamp = timestamp
        return dict(visible_objects=visible, landmark_first_seen=first, landmark_seen_before=seen_before,
            landmark_seen_so_far=sorted(self._seen),
            landmark_events=[dict(type='first_seen', id=name, timestamp_s=timestamp, is_goal=name == self.target_id) for name in first],
            landmark_visibility_threshold_pixels=self.min_pixels,
            landmark_evidence_method='current_instance_mask_with_full_prim_path_or_descendant_match')
