"""Adopt complete immutable layout/raw files without rendering or API calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def adopt(source, output, *, kind, expected_seed=None, expected_episode=None,
          expected_target=None, layout=None):
    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Adoption requires a new output directory')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output may not contain one another')
    if kind == 'layout':
        targets = json.loads((source / 'targets.json').read_text())
        geometry = json.loads((source / 'geometry_report.json').read_text())
        if expected_seed is None or targets['seed'] != expected_seed or not geometry['passed']:
            raise ValueError('Adopted layout seed/geometry does not match')
        for required in ('cave.usdc', 'mesh.npz', 'scene_config.json'):
            if not (source / required).is_file():
                raise ValueError('Layout asset missing: ' + required)
    elif kind == 'raw':
        metadata = json.loads((source / 'metadata.json').read_text())
        status = json.loads((source / 'status.json').read_text())
        if status['phase'] != 'complete' or not metadata['terminal_success']:
            raise ValueError('A partial capture cannot be adopted as complete')
        if metadata.get('capture_stride') != 1:
            raise ValueError('Dense raw adoption requires every native observation')
        if (metadata['episode_id'] != expected_episode or metadata['target_id'] != expected_target
                or layout is None):
            raise ValueError('Raw identity or matching layout is missing')
        layout = Path(layout).resolve(strict=True)
        for relative, value in metadata['source_sha256'].items():
            path = (layout / relative).resolve(strict=True)
            if layout not in path.parents or digest(path) != value:
                raise ValueError('Raw scene bytes differ from the adopted layout')
    else:
        raise ValueError('Unknown adoption kind')
    inventory = {}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Immutable adoption does not permit symlinks')
        if path.is_file() and path.name != 'REQUEST_STOP' and not path.name.endswith('.tmp'):
            inventory[path.relative_to(source).as_posix()] = digest(path)
    output.mkdir(parents=True, exist_ok=False)
    for relative in inventory:
        original, destination = source / relative, output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(original, destination)
        except OSError:
            shutil.copy2(original, destination)
        if digest(destination) != inventory[relative]:
            raise ValueError('Source changed during adoption: ' + relative)
    proof = dict(kind=kind, source=str(source), source_files_sha256=inventory,
                 source_unchanged=True, source_and_payloads_not_rewritten=True)
    if (source / 'ADOPTION.json').is_file():
        proof['source_previous_adoption'] = json.loads((source / 'ADOPTION.json').read_text())
        proof['source_previous_adoption_sha256'] = inventory['ADOPTION.json']
    temporary = output / 'ADOPTION.json.tmp'
    temporary.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(output / 'ADOPTION.json')
    return dict(passed=True, kind=kind, files=len(inventory), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['layout', 'raw'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-seed', type=int)
    parser.add_argument('--expected-episode')
    parser.add_argument('--expected-target')
    parser.add_argument('--layout', type=Path)
    args = parser.parse_args()
    print(json.dumps(adopt(args.source, args.output, kind=args.kind,
        expected_seed=args.expected_seed, expected_episode=args.expected_episode,
        expected_target=args.expected_target, layout=args.layout), ensure_ascii=False))


if __name__ == '__main__':
    main()
