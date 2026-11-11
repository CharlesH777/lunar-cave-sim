#!/usr/bin/env python3
"""Build a fresh hypothesized cave USD using CPU PXR; never starts Isaac/Kit.

Runtime: env_isaaclab Python with the installed omni.usd.libs path in PYTHONPATH,
its bin directory and the conda lib directory in LD_LIBRARY_PATH; hide CUDA.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError('Not JSON serializable: ' + type(value).__name__)


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='A NEW directory; existing paths are rejected')
    parser.add_argument('--spacing', type=float, default=1.0, help='CPU geometry sampling spacing in metres')
    parser.add_argument('--no-mesh-npz', action='store_true', help='Skip optional compressed vertices/faces for review')
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    # mkdir without exist_ok is the atomic no-overwrite boundary, including symlinks.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def status(phase, **values):
        _write(output / 'build_status.json', dict(status=phase,
            updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            GPU_work=False, simulation_app_started=False, renders_executed=0,
            training_started=False, navigation_evaluated=False, **values))
    try:
        status('validating_network')
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '-1'):
            raise RuntimeError("CPU build requires CUDA_VISIBLE_DEVICES='' to keep GPU work disabled")
        import numpy as np
        if not np.isfinite(args.spacing) or args.spacing <= 0:
            raise ValueError('--spacing must be a finite positive number of metres')
        from cave.network3d import validate_network, build_network_geometry
        from cave.network3d_scene import export_network_usd
        source = args.network.expanduser().resolve(strict=True)
        raw = source.read_bytes();network = json.loads(raw)
        if not isinstance(network, dict):raise ValueError('Network JSON root must be an object')
        validation = validate_network(network)
        if not isinstance(validation, dict):raise ValueError('Network validation must return a report object')
        if any(validation.get(k) is False for k in ['passed', 'valid', 'ok']) or validation.get('errors'):
            raise ValueError('Network validation rejected this candidate: ' + json.dumps(validation, default=_json_default))
        (output / 'network.json').write_bytes(raw)
        status('building_geometry')
        geometry = build_network_geometry(network, spacing_m=args.spacing)
        if not isinstance(geometry, dict) or not all(k in geometry for k in ['vertices', 'faces', 'report']):
            raise ValueError('Geometry must supply vertices, faces and a JSON report')
        if geometry['report'].get('passed') is not True:
            rejected = dict(geometry['report'])
            rejected.update(artifact_status='candidate_hypothesized', network_validation=validation,
                source_network=str(source), source_network_sha256=hashlib.sha256(raw).hexdigest(),
                requested_spacing_m=args.spacing, usd_export=None, GPU_work=False,
                simulation_app_started=False, renders_executed=0, navigation_evaluated=False,
                rover_navigation_claim=False, no_observed_entrance=True,
                artificial_caps_not_geological_endpoints=True)
            _write(output / 'geometry_report.json', rejected)
            status('rejected', complete=False, geometry_checks_passed=False,
                   report_file='geometry_report.json', usd_authored=False)
            print(json.dumps(dict(complete=False, rejected=True, output=str(output),
                reason='Geometry acceptance checks did not pass', GPU_work=False)))
            return 2
        status('authoring_and_reopening_usd', vertices=len(geometry['vertices']), triangles=len(geometry['faces']))
        exported = export_network_usd(network, geometry, output / 'cave.usdc')
        report = dict(geometry['report'])
        report.update(schema='candidate_network3d_geometry_and_usd_report_v1',
            artifact_status='candidate_hypothesized', network_validation=validation,
            source_network=str(source), source_network_sha256=hashlib.sha256(raw).hexdigest(),
            requested_spacing_m=args.spacing, usd_export=exported['report'],
            GPU_work=False, simulation_app_started=False, renders_executed=0,
            navigation_evaluated=False, rover_navigation_claim=False,
            no_observed_entrance=True, artificial_caps_not_geological_endpoints=True)
        if not args.no_mesh_npz:
            np.savez_compressed(output / 'mesh.npz', vertices=exported['vertices'], faces=exported['faces'])
            report['mesh_npz'] = dict(path='mesh.npz', bytes=(output / 'mesh.npz').stat().st_size,
                                      keys=['vertices', 'faces'])
        report['build_elapsed_seconds'] = time.monotonic()-started
        _write(output / 'geometry_report.json', report)
        status('complete', complete=True, usd_file='cave.usdc', report_file='geometry_report.json')
        print(json.dumps(dict(complete=True, output=str(output), usd=str(output / 'cave.usdc'),
                              report=str(output / 'geometry_report.json'), GPU_work=False)))
        return 0
    except BaseException as exc:
        status('failed', complete=False, error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    raise SystemExit(main())
