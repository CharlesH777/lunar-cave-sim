"""Dependency-free SHA256 delivery verification, also works with Python 3.6."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    errors = []
    for row in manifest['files']:
        path = (root/row['path']).resolve()
        if root not in path.parents or not path.is_file():
            errors.append({'path': row['path'], 'error': 'missing or outside delivery'})
            continue
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024*1024), b''):
                sha.update(block)
        if path.stat().st_size != row['bytes'] or sha.hexdigest() != row['sha256']:
            errors.append({'path': row['path'], 'error': 'size or sha256 mismatch'})
    expected = set(row['path'] for row in manifest['files']) | {'MANIFEST.json'}
    actual = set(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
    extras = sorted(actual-expected)
    result = {'passed': not errors and not extras, 'files_checked': len(manifest['files']),
              'total_bytes': manifest['total_bytes'], 'errors': errors, 'extra_files': extras}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
