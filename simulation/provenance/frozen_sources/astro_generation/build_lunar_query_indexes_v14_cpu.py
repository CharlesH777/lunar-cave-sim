"""Create isolated query-family training and evaluation indexes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--view', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    from cave.lunar_query_indexes_v14 import build_query_indexes
    manifest, report = build_query_indexes(args.base, args.view,
                                           output=args.output, protocol=args.protocol)
    print(json.dumps(dict(passed=report['passed'], output=str(args.output),
                          schema_version=manifest.get('schema_version')), ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
