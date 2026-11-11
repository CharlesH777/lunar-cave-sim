"""SHA-verify and atomically publish one immutable dataset unit using SSH/rsync."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cave.lunar_dataset_delivery_v14 import (DeliveryError, _write_json,
    build_unit_manifest, manifest_digest, publish_unit, verify_published_unit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Source stays unchanged. Only REQUEST_STOP and basename *.tmp entries/subtrees are excluded '
        'and listed. State identifies one transaction; use a distinct state directory for each unit. '
        'There is one upload attempt per invocation. Failures preserve incoming data.'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--host')
    parser.add_argument('--remote-root')
    parser.add_argument('--unit', help='Safe relative unit path under the remote root')
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--require-relative', action='append', default=[], metavar='PATH',
                        help='Require a metadata/index/report/payload entry inside this unit; repeatable')
    parser.add_argument('--ssh-option', action='append', default=[],
                        help='Additional ssh argv element; use --ssh-option=-oExample=value')
    parser.add_argument('--inventory-only', action='store_true', help='Local CPU inventory only; no network')
    parser.add_argument('--manifest-out', type=Path, help='Inventory-only manifest path outside the source')
    parser.add_argument('--verify-state', type=Path, help='Rehash an existing remote unit from retained state; no local payload or upload')
    args = parser.parse_args(argv)
    if args.verify_state:
        if args.source or args.inventory_only or args.host or args.remote_root or args.unit or args.state_dir:
            parser.error('--verify-state uses only its retained transaction, not a new source/destination')
    elif not args.inventory_only and not all((args.source, args.host, args.remote_root, args.unit, args.state_dir)):
        parser.error('--host, --remote-root, --unit and --state-dir are required for publication')
    if args.inventory_only and (args.source is None or args.manifest_out is None):
        parser.error('--inventory-only requires --source and --manifest-out')
    try:
        if args.verify_state:
            result = verify_published_unit(state_dir=args.verify_state, ssh_options=args.ssh_option)
        elif args.inventory_only:
            source = args.source.expanduser().resolve(strict=True)
            output = args.manifest_out.expanduser().resolve()
            if output.is_relative_to(source):
                raise DeliveryError('Inventory output must be outside the immutable source tree')
            manifest = build_unit_manifest(source, required_paths=args.require_relative)
            _write_json(output, manifest)
            result = dict(passed=True, inventory_only=True, manifest_sha256=manifest_digest(manifest),
                          manifest_path=str(output), file_count=manifest['file_count'],
                          total_file_bytes=manifest['total_file_bytes'],
                          excluded_control_paths=manifest['excluded_control_paths'])
        else:
            result = publish_unit(args.source, host=args.host, remote_root=args.remote_root,
                unit=args.unit, state_dir=args.state_dir, required_paths=args.require_relative,
                ssh_options=args.ssh_option)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (DeliveryError, OSError, ValueError) as exc:
        print(json.dumps(dict(passed=False, error=type(exc).__name__ + ': ' + str(exc),
                              source_pruned=False, automatic_retry=False), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
