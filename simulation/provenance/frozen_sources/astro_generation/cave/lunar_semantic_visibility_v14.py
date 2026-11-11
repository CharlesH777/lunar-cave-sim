"""Provisional human-salience gates, separate from raw instance detection.

The helper never changes raw detection, STOP acceptance or route geometry.
Passing this geometric image gate is a candidate for human review, not proof
that a person can recognize the object category.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
from PIL import Image

from .lunar_landmark_evidence_v13 import LABELS_ZH, _label_paths

LEGACY_DEFAULT_CONFIG = dict(
    schema_version='lunar_semantic_visibility_config/14.0',
    revision='14.0', provisional=True, observation_stride=5,
    min_visible_pixels_rgb224=256, min_edge_margin_rgb224=4,
    rover_min_visible_pixels_rgb224=700, rover_min_edge_margin_rgb224=8,
    min_valid_depth_fraction=.9,
    rover_min_bbox_rgb224=[32, 24], other_min_bbox_rgb224=[12, 16],
    rover_max_abs_bearing_deg=30.,
    mask_projection='nearest-neighbour 256x256 instance mask to 224x224',
    semantic_scope='candidate recognizability gate; validate global class thresholds against actual RGB before release',
)

# This exact predecessor remains replay-compatible for explicit old-key configs.
LEGACY_MODULE_SHA256 = '4414de2fc63bb62f33d5729cb1b677420d6e2c3c9873c7a2ecec7c5e1bb27a05'
OPTIONAL_CONFIG_KEYS = {'class_overrides', 'terminal_goal_edge_exception'}
DEFAULT_CONFIG = dict(LEGACY_DEFAULT_CONFIG, provisional=False,
    class_overrides=dict(lunar_base=dict(min_visible_pixels_rgb224=800,
        min_bbox_rgb224=[40, 28], min_edge_margin_rgb224=8, max_abs_bearing_deg=30.)),
    terminal_goal_edge_exception=dict(enabled=True, min_visible_pixels_rgb224=2500,
        min_bbox_rgb224=[64, 48], min_valid_depth_fraction=.9, max_abs_bearing_deg=30.))


def config_hash(config):
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def checked_config(config=None):
    value = deepcopy(DEFAULT_CONFIG if config is None else config)
    if (not set(LEGACY_DEFAULT_CONFIG).issubset(value)
            or set(value) - set(LEGACY_DEFAULT_CONFIG) - OPTIONAL_CONFIG_KEYS
            or value['schema_version'] != DEFAULT_CONFIG['schema_version']
            or value['revision'] != '14.0' or not isinstance(value['provisional'], bool)
            or value['observation_stride'] != 5 or value['min_visible_pixels_rgb224'] < 1
            or value['rover_min_visible_pixels_rgb224'] < 1 or value['rover_min_edge_margin_rgb224'] < 0
            or value['min_edge_margin_rgb224'] < 0 or not 0 < value['min_valid_depth_fraction'] <= 1
            or not 0 < value['rover_max_abs_bearing_deg'] <= 90
            or any(len(value[key]) != 2 or any(n < 1 for n in value[key])
                   for key in ('rover_min_bbox_rgb224', 'other_min_bbox_rgb224'))):
        raise ValueError('Invalid provisional V14 semantic visibility configuration')
    allowed_overrides = {'min_visible_pixels_rgb224', 'min_bbox_rgb224', 'min_edge_margin_rgb224',
                         'max_abs_bearing_deg', 'min_valid_depth_fraction'}
    overrides = value.get('class_overrides', {})
    if not isinstance(overrides, Mapping) or set(overrides) - set(LABELS_ZH):
        raise ValueError('Class overrides must name known semantic object categories')
    for override in overrides.values():
        if not isinstance(override, Mapping) or not override or set(override) - allowed_overrides:
            raise ValueError('Unknown or empty semantic class override')
        for key in ('min_visible_pixels_rgb224', 'min_edge_margin_rgb224'):
            if key in override and (type(override[key]) is not int or override[key] < (1 if key == 'min_visible_pixels_rgb224' else 0)):
                raise ValueError('Invalid semantic class pixel/edge threshold')
        if 'min_bbox_rgb224' in override and (not isinstance(override['min_bbox_rgb224'], list)
                or len(override['min_bbox_rgb224']) != 2
                or any(type(n) is not int or n < 1 for n in override['min_bbox_rgb224'])):
            raise ValueError('Invalid semantic class bounding box threshold')
        if 'max_abs_bearing_deg' in override and not 0 < override['max_abs_bearing_deg'] <= 90:
            raise ValueError('Invalid semantic class bearing threshold')
        if 'min_valid_depth_fraction' in override and not 0 < override['min_valid_depth_fraction'] <= 1:
            raise ValueError('Invalid semantic class valid depth threshold')
    if 'terminal_goal_edge_exception' in value:
        rule = value['terminal_goal_edge_exception']
        if (not isinstance(rule, Mapping) or set(rule) != set(DEFAULT_CONFIG['terminal_goal_edge_exception'])
                or type(rule['enabled']) is not bool or type(rule['min_visible_pixels_rgb224']) is not int
                or rule['min_visible_pixels_rgb224'] < 2500
                or not isinstance(rule['min_bbox_rgb224'], list) or len(rule['min_bbox_rgb224']) != 2
                or any(type(n) is not int or n < floor for n, floor in zip(rule['min_bbox_rgb224'], (64, 48)))
                or not .9 <= rule['min_valid_depth_fraction'] <= 1
                or not 0 < rule['max_abs_bearing_deg'] <= 30):
            raise ValueError('Terminal edge exception must retain large-object, depth and central-bearing safeguards')
    config_hash(value)
    return value


def legacy_replay_module_hashes(config):
    """Only this known predecessor may accompany a validated explicit old config.

    Callers still verify config/raw/evidence hashes. New optional groups never
    accept predecessor provenance, even when an optional rule is disabled.
    """
    value = checked_config(config)
    return (LEGACY_MODULE_SHA256,) if set(value) == set(LEGACY_DEFAULT_CONFIG) else ()


def class_gate(config, name):
    """Resolve one class without adding fields/defaults to an explicit config."""
    rover = name == 'lunar_rover'
    gate = dict(min_visible_pixels_rgb224=config['rover_min_visible_pixels_rgb224' if rover else 'min_visible_pixels_rgb224'],
        min_bbox_rgb224=config['rover_min_bbox_rgb224' if rover else 'other_min_bbox_rgb224'],
        min_edge_margin_rgb224=config['rover_min_edge_margin_rgb224' if rover else 'min_edge_margin_rgb224'],
        max_abs_bearing_deg=config['rover_max_abs_bearing_deg'] if rover else None,
        min_valid_depth_fraction=config['min_valid_depth_fraction'])
    gate.update(config.get('class_overrides', {}).get(name, {}))
    return gate


def target_rows(targets):
    rows = list(targets.values()) if isinstance(targets, Mapping) else list(targets)
    result = {row['id']: row for row in rows}
    if len(rows) != 4 or set(result) != set(LABELS_ZH):
        raise ValueError('Semantic visibility requires the four explicit search objects')
    paths = [row['prim_path'].rstrip('/') for row in rows]
    if len(set(paths)) != 4 or any(not path.startswith('/') for path in paths):
        raise ValueError('Search-object full prim paths must be unique and absolute')
    return result


def measure_objects(instance_image, instance_mapping, K256, depth_m, depth_valid, targets, config=None):
    """Measure every category from the actual current mask/depth, including zero."""
    config = checked_config(config)
    targets = target_rows(targets)
    image = np.asarray(instance_image)
    depth, valid = np.asarray(depth_m), np.asarray(depth_valid)
    K = np.asarray(K256, float)
    if (image.shape != (256, 256) or image.dtype.kind not in 'iu' or depth.shape != image.shape
            or valid.shape != image.shape or K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError('Semantic visibility requires actual aligned 256px mask/depth/calibration')
    labels = instance_mapping.get('idToLabels', instance_mapping)
    if not isinstance(labels, Mapping):
        raise ValueError('Missing full-prim instance mapping')
    ids = {name: [] for name in targets}
    for number, value in labels.items():
        paths = _label_paths(value)
        matches = [name for name, row in targets.items() if any(
            path == row['prim_path'].rstrip('/') or path.startswith(row['prim_path'].rstrip('/') + '/') for path in paths)]
        if len(matches) == 1:
            try:
                number = int(number)
            except (TypeError, ValueError):
                continue
            if number >= 0:
                ids[matches[0]].append(number)
    output = []
    for name in sorted(targets):
        mask = np.isin(image, ids[name]) if ids[name] else np.zeros(image.shape, bool)
        count = int(mask.sum())
        projected = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((224, 224), Image.Resampling.NEAREST), bool)
        count224 = int(projected.sum())
        good_depth = mask & valid.astype(bool) & np.isfinite(depth) & (depth > 0)
        depth_count = int(good_depth.sum())
        row = dict(id=name, label_zh=targets[name].get('label_zh') or LABELS_ZH[name],
                   pixels_256=count, pixels_rgb224=count224, bbox_xyxy_256=None, bbox_xyxy_rgb224=None,
                   bbox_width_height_rgb224=[0, 0], edge_margin_rgb224=0, bearing_deg=None,
                   valid_depth_fraction=depth_count / count if count else 0.,
                   median_axial_depth_m=float(np.median(depth[good_depth])) if depth_count else None)
        if count:
            yy, xx = np.nonzero(mask)
            box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
            row['bbox_xyxy_256'] = box
            row['bearing_deg'] = math.degrees(math.atan2(K[0, 2]-(box[0]+box[2])*.5, K[0, 0]))
        if count224:
            yy, xx = np.nonzero(projected)
            box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
            row.update(bbox_xyxy_rgb224=box, bbox_width_height_rgb224=[box[2]-box[0], box[3]-box[1]],
                       edge_margin_rgb224=min(box[0], box[1], 224-box[2], 224-box[3]))
        gate = class_gate(config, name)
        minimum = gate['min_bbox_rgb224']
        reasons = []
        min_area = gate['min_visible_pixels_rgb224']
        min_edge = gate['min_edge_margin_rgb224']
        if count224 < min_area:
            reasons.append('insufficient_visible_area')
        if any(value < floor for value, floor in zip(row['bbox_width_height_rgb224'], minimum)):
            reasons.append('bbox_too_small')
        if row['edge_margin_rgb224'] < min_edge:
            reasons.append('edge_clipped_or_insufficient_margin')
        if gate['max_abs_bearing_deg'] is not None and (row['bearing_deg'] is None or abs(row['bearing_deg']) > gate['max_abs_bearing_deg']):
            reasons.append('outside_rover_central_bearing' if name == 'lunar_rover' else 'outside_class_central_bearing')
        if row['valid_depth_fraction'] < gate['min_valid_depth_fraction']:
            reasons.append('insufficient_current_valid_depth')
        row.update(salient_candidate=not reasons, rejection_reasons=reasons)
        output.append(row)
    return output


class SemanticVisibility:
    """Causal first salient *eligible-grid* cue state for one physical episode."""
    def __init__(self, targets, target_id, config=None):
        self.targets = target_rows(targets)
        if target_id not in self.targets:
            raise ValueError('Unknown semantic goal')
        self.target_id = target_id
        self.config = checked_config(config)
        self._seen = set()
        self._previous_step = -1
        self._previous_time = -math.inf
        self._strict_first_grid = {}

    def observe(self, instance_image, instance_mapping, K256, depth_m, depth_valid,
                timestamp_s, step_index, *, terminal_stop=False):
        if (not isinstance(step_index, int) or step_index <= self._previous_step
                or not math.isfinite(timestamp_s) or timestamp_s <= self._previous_time):
            raise ValueError('Semantic observations must be in strictly causal step/time order')
        measured = measure_objects(instance_image, instance_mapping, K256, depth_m, depth_valid, self.targets, self.config)
        strict_ids = {row['id'] for row in measured if row['salient_candidate']}
        rule = self.config.get('terminal_goal_edge_exception')
        if terminal_stop and rule and rule['enabled'] and self.target_id in self._strict_first_grid:
            goal = next(row for row in measured if row['id'] == self.target_id)
            if (goal['rejection_reasons'] == ['edge_clipped_or_insufficient_margin']
                    and goal['pixels_rgb224'] >= rule['min_visible_pixels_rgb224']
                    and all(n >= floor for n, floor in zip(goal['bbox_width_height_rgb224'], rule['min_bbox_rgb224']))
                    and goal['valid_depth_fraction'] >= rule['min_valid_depth_fraction']
                    and goal['bearing_deg'] is not None and abs(goal['bearing_deg']) <= rule['max_abs_bearing_deg']):
                goal.update(salient_candidate=True, rejection_reasons=[], strict_salient_candidate=False,
                    strict_rejection_reasons=['edge_clipped_or_insufficient_margin'],
                    terminal_goal_edge_exception=dict(applied=True, waived_check='edge_margin_only',
                        prior_strict_salient_grid_step_index=self._strict_first_grid[self.target_id]))
        # Record strict regular-grid visibility after evaluating the current
        # exception. Current STOP or an off-grid sighting cannot provide its own prior.
        if step_index % self.config['observation_stride'] == 0:
            for name in strict_ids:
                self._strict_first_grid.setdefault(name, step_index)
        visible = [dict(row, is_goal=row['id'] == self.target_id) for row in measured if row['salient_candidate']]
        eligible = step_index % self.config['observation_stride'] == 0 or terminal_stop
        now = sorted({row['id'] for row in visible} - self._seen) if eligible else []
        before = sorted(self._seen)
        self._seen.update(now)
        self._previous_step, self._previous_time = step_index, timestamp_s
        return dict(semantic_visibility_revision='14.0', semantic_visibility_config_sha256=config_hash(self.config),
            semantic_observation_eligible=eligible, semantic_visible_objects=visible,
            semantic_cue_first_seen=now, semantic_cue_seen_before=before, semantic_cue_seen_so_far=sorted(self._seen),
            semantic_cue_events=[dict(type='first_salient_eligible_observation', id=name, step_index=step_index,
                timestamp_s=timestamp_s, is_goal=name == self.target_id) for name in now],
            semantic_object_diagnostics=measured,
            semantic_evidence_scope='actual current full-prim mask and valid depth; provisional recognizability; cue history uses eligible grid only')
