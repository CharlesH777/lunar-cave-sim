"""Export or validate every-fifth-frame training observations at native action dt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cave.lunar_training_stride_v14 import export_training_view, validate_training_view, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--history-frames', type=int, default=8)
    parser.add_argument('--validate-view', type=Path)
    args = parser.parse_args(argv)
    if args.validate_view:
        report = validate_training_view(args.validate_view)
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report['passed'] else 1
    if args.base is None or args.output is None:
        parser.error('--base and --output are required for export')
    manifest, report = export_training_view(args.base, args.output,
        stride=args.stride, history_frames=args.history_frames)
    print(json.dumps(dict(passed=report['passed'], output=str(args.output),
        sampled_physical_frames=manifest['sampled_physical_frame_count'],
        reader_rows=manifest['statistics']['view_reader_rows'],
        retained_cot_rows=manifest['statistics']['retained_cot_reader_rows'],
        observation_stride=args.stride, native_action_dt_s=manifest['native_action_dt_s']), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
