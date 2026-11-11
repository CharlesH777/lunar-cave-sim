"""Place task endpoints on nearby free ground without changing terrain or rocks."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

from build_lunar_surface_detail_v5_cpu import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, nargs=2, required=True)
    parser.add_argument('--goal', type=float, nargs=2, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.absolute()
    original = json.loads((source/'scene_config.json').read_text())
    config = copy.deepcopy(original)
    config['task_spawn']['xy_m'] = args.start
    config['task_goal']['xy_m'] = args.goal
    output.mkdir(parents=True, exist_ok=False)
    names = ['cave.usdc', 'mesh.npz', 'boulders.json', 'geometry_report.json',
             'obstacle_report.json', 'obstacle_package_report.json', 'surface_detail_report.json',
             'ASSET_LICENSES.json', 'GRAVEL_V5_SOURCES.json', 'BASE_ASSET_SOURCES.json']
    shared = {}
    for name in names:
        src = source/name
        if not src.is_file():
            continue
        try:
            os.link(src, output/name)
        except OSError:
            shutil.copy2(src, output/name)
        shared[name] = sha(src)
        assert sha(output/name) == shared[name]
    shutil.copytree(source/'textures', output/'textures', copy_function=os.link)
    write(output/'scene_config.json', config)
    write(output/'previous_scene_config.json', original)
    report = dict(passed=True, source=str(source), evidence=str(args.evidence.resolve(strict=True)),
        reason='Initial authored endpoints overlapped preserved gravel; use nearest free suggestions before testing connectivity',
        old_start=original['task_spawn'], new_start=config['task_spawn'],
        old_goal=original['task_goal'], new_goal=config['task_goal'],
        changed_config_keys=['task_spawn.xy_m', 'task_goal.xy_m'],
        shared_assets_sha256=shared, geometry_and_all_obstacles_unchanged=True,
        path_success_claimed=False, GPU_work=False)
    assert json.loads((source/'scene_config.json').read_text()) == original
    write(output/'task_revision_report.json', report)
    write(output/'status.json', dict(phase='task_endpoints_revised', path_validated=False,
                                    task_revision_report='task_revision_report.json'))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
