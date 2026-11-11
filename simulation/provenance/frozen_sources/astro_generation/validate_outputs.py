#!/usr/bin/env python3
"""CPU validation of saved Isaac cave artifacts; never launches the simulator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


def _json(path):
    return json.loads(path.read_text())


def _image(path):
    with Image.open(path) as image:
        image.load()
        if image.mode not in ('RGB', 'RGBA'):
            raise ValueError(f'Expected RGB/RGBA image, got {image.mode}')
        data = np.asarray(image.convert('RGB'))
    if min(data.shape[:2]) < 32 or not np.any(data) or float(data.std()) == 0:
        raise ValueError('Empty, black, constant, or implausibly small RGB image')
    return data


def _matrix(K, pose, width, height):
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError('K must be a finite 3x3 matrix')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('Invalid pinhole K')
    if not (0 <= K[0, 2] <= width and 0 <= K[1, 2] <= height):
        raise ValueError('Principal point is outside the image')
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError('world_from_camera must be finite 4x4')
    rotation = pose[:3, :3]
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Transform uses wrong row/column convention')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError('Camera rotation is not in SO(3)')


def _path_labels(value):
    if isinstance(value, str):
        return [value] if value.startswith('/') else []
    if isinstance(value, dict):
        return [p for v in value.values() for p in _path_labels(v)]
    if isinstance(value, list):
        return [p for v in value for p in _path_labels(v)]
    return []


def _visible(instance, info, target_path):
    labels = info['idToLabels']
    if isinstance(labels, str):
        labels = json.loads(labels)
    identifiers = [int(k) for k, label in labels.items()
                   if any(p == target_path or p.startswith(target_path + '/')
                          for p in _path_labels(label))]
    return int(np.isin(instance, identifiers).sum()) if identifiers else 0


def validate_outputs(path, *, require_episode=False):
    """Validate one output directory and write its validation_report.json.

    Preview-only output can pass with episode_present=false. With an episode,
    all referenced observations are checked and final success is reconstructed
    from the saved instance pixels, STOP action, pose metrics, and collisions.
    This checks artifacts; process exit and root status remain separate evidence.
    """
    path = Path(path).resolve()
    errors = []
    checks = 0
    report = {'schema_version': 1, 'output': str(path),
              'validated_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'saved artifact integrity and rendered instance evidence; no simulator or learned policy run',
              'episode_present': (path / 'episode_000').is_dir()}

    def check(label, operation):
        nonlocal checks
        checks += 1
        try:
            return operation()
        except Exception as exc:
            errors.append(f'{label}: {exc}')
            return None

    def need(condition, message):
        if not condition:
            raise ValueError(message)

    def stage_check():
        stage = path / 'cave.usdc'
        need(stage.is_file() and stage.stat().st_size > 1024, 'Missing or empty cave.usdc')
        with stage.open('rb') as handle:
            need(handle.read(8) == b'PXR-USDC', 'Saved file is not a USD crate')
        return stage.stat().st_size

    report['scene_bytes'] = check('scene', stage_check)
    manifest = check('scene_manifest.json', lambda: _json(path / 'scene_manifest.json'))
    previews = sorted(path.glob('*.png'))
    check('previews', lambda: need(bool(previews), 'No rendered PNG previews'))
    report['rendered_previews'] = []
    for preview in previews:
        data = check(preview.name, lambda preview=preview: _image(preview))
        if data is not None:
            report['rendered_previews'].append({'file': preview.name,
                                               'resolution': [data.shape[1], data.shape[0]],
                                               'mean_srgb_8bit': float(data.mean())})
    ep = path / 'episode_000'
    if not ep.is_dir():
        if require_episode:
            check('episode', lambda: need(False, 'episode_000 is required'))
    else:
        rows = check('transitions.jsonl', lambda: [json.loads(s) for s in (ep / 'transitions.jsonl').read_text().splitlines() if s.strip()])
        result = check('result.json', lambda: _json(ep / 'result.json'))
        camera = check('camera.json', lambda: _json(ep / 'camera.json'))
        rows = rows or []
        result = result or {}
        report['frames'] = len(rows)
        report['recorded_result'] = result
        check('nonempty trajectory', lambda: need(bool(rows), 'No transition records'))
        check('frame count', lambda: need(result.get('frames') == len(rows), 'Result frame count differs from transitions'))
        rgb_names = {r.get('rgb') for r in rows}
        sensor_names = {r.get('sensors') for r in rows}
        actual_rgb = {p.name for p in ep.glob('*_rgb.jpg')}
        actual_npz = {p.name for p in ep.glob('*_sensors.npz')}
        check('RGB file set', lambda: need(actual_rgb == rgb_names and len(rgb_names) == len(rows), 'Missing, extra, or repeated RGB files'))
        check('sensor file set', lambda: need(actual_npz == sensor_names and len(sensor_names) == len(rows), 'Missing, extra, or repeated sensor files'))
        final_visible = None
        first_K = None
        for index, record in enumerate(rows):
            def frame_check(index=index, record=record):
                need(record['step'] == index, 'Noncontiguous or unordered steps')
                rgb_name, sensor_name = record['rgb'], record['sensors']
                need(Path(rgb_name).name == rgb_name and Path(sensor_name).name == sensor_name, 'Observation path escapes episode directory')
                need(re.fullmatch(r'\d+_rgb.jpg', rgb_name) and re.fullmatch(r'\d+_sensors.npz', sensor_name), 'Unexpected observation filename')
                image = _image(ep / rgb_name)
                height, width = image.shape[:2]
                with np.load(ep / sensor_name, allow_pickle=False) as sensors:
                    need({'depth_m', 'depth_valid', 'instance_id', 'K', 'world_from_camera'} <= set(sensors.files), 'Missing sensor arrays')
                    depth, valid, instance = sensors['depth_m'], sensors['depth_valid'], sensors['instance_id']
                    need(depth.shape == valid.shape == instance.shape == (height, width), 'Sensor/RGB dimensions differ')
                    need(depth.dtype.kind == 'f' and valid.dtype == np.bool_ and instance.dtype.kind in 'ui', 'Wrong sensor dtypes')
                    need(np.array_equal(valid, np.isfinite(depth) & (depth > 0)), 'Depth validity mask disagrees with saved depth')
                    need(bool(valid.any()), 'No valid scene depth')
                    K = sensors['K'].copy()
                    _matrix(K, sensors['world_from_camera'], width, height)
                    if camera:
                        need(camera.get('resolution') == [width, height], 'camera.json resolution mismatch')
                        need(np.allclose(K, np.asarray(camera['K'])), 'K differs from camera.json')
                        need('column' in camera.get('coordinate_convention', ''), 'Camera convention is not explicit column-vector OpenCV')
                    target_id = record.get('privileged_info', {}).get('target_id', result.get('target_id'))
                    candidates = [t for t in (manifest or {}).get('targets', []) if t['id'] == target_id]
                    need(len(candidates) == 1, 'Target ID is absent or ambiguous in scene manifest')
                    mapping = _json(ep / (Path(sensor_name).stem.removesuffix('_sensors') + '_instances.json'))
                    visible = _visible(instance, mapping, candidates[0]['prim_path'])
                    need(visible == record['privileged_info']['visible_pixel_count'], 'Recorded target visibility differs from instance pixels')
                    need(record['privileged_info']['collision_steps'] == 0 and not record['privileged_info']['collision'], 'Trajectory contains a collision')
                    return visible, K
            frame = check(f'frame {index}', frame_check)
            if frame is not None:
                if index == 0:
                    first_K = frame[1]
                if index == len(rows) - 1:
                    final_visible = frame[0]
        report['final_visible_pixels_recomputed'] = final_visible
        report['K'] = first_K.tolist() if first_K is not None else None

        def success_check():
            need(bool(rows), 'No final transition')
            last = rows[-1]
            need(result.get('success') is True, 'Search did not succeed')
            need(result.get('stop_requested') is True and last['action'][2] >= .5, 'Final action did not request STOP')
            need(last.get('terminated') is True and not last.get('truncated'), 'Episode did not terminate successfully')
            need(final_visible is not None and final_visible >= 32, 'Fewer than 32 real target instance pixels at STOP')
            need(result.get('visible_pixel_count') == final_visible, 'Result visibility differs from saved final instance pixels')
            need(result.get('collision_steps') == 0 and not result.get('collision'), 'Collision count is nonzero')
            need(result.get('distance_to_target_m', float('inf')) <= 1.8, 'STOP outside 1.8m target radius')
            need(result.get('facing_error_rad', float('inf')) <= .65, 'STOP heading is outside tolerance')
            need(last.get('privileged_info', {}).get('success') is True, 'Final transition does not report success')
            return True

        report['search_success_verified'] = check('successful STOP', success_check) is True
    report.update(passed=not errors, checks=checks, error_count=len(errors), errors=errors[:50])
    path.mkdir(parents=True, exist_ok=True)
    temp = path / 'validation_report.json.tmp'
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path / 'validation_report.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--require-episode', action='store_true')
    parser.add_argument('--expected-maps', type=int,
                        help='Validate exactly this number of map_XX output subdirectories')
    args = parser.parse_args()
    if args.expected_maps is None:
        report = validate_outputs(args.path, require_episode=args.require_episode)
    else:
        directories = sorted(p for p in args.path.glob('map_*') if p.is_dir())
        maps = [validate_outputs(p, require_episode=args.require_episode) for p in directories]
        report = {'validated_utc': datetime.now(timezone.utc).isoformat(),
                  'expected_maps': args.expected_maps, 'existing_maps': len(maps),
                  'passed': len(maps) == args.expected_maps and all(r['passed'] for r in maps),
                  'maps': [{'path': r['output'], 'passed': r['passed'],
                            'episode_present': r['episode_present'],
                            'search_success_verified': r.get('search_success_verified', False),
                            'error_count': r['error_count']} for r in maps]}
        args.path.mkdir(parents=True, exist_ok=True)
        (args.path / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
