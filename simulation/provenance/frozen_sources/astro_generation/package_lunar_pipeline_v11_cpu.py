"""Freeze a self-contained dataset generator and locally verified example.

The Isaac Sim SDK, credentials, old runs and navigation model weights are not
part of the delivery. This is a CPU-only file operation, never a training job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def copy_tree(source, target, *, omit=(), link_payloads=False):
    source = source.resolve(strict=True)
    for item in sorted(source.rglob('*')):
        if not item.is_file() or item.is_symlink():
            continue
        relative = item.relative_to(source)
        if any(part in omit for part in relative.parts):
            continue
        destination = target/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if link_payloads and item.suffix.lower() in {'.usdc', '.npz', '.png', '.jpg', '.jpeg', '.webp', '.mp4'}:
            try:
                os.link(item, destination)
                continue
            except OSError:
                pass
        shutil.copy2(item, destination)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terrain', type=Path, required=True)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--map', type=Path)
    parser.add_argument('--revision', type=int, choices=[11, 12, 13, 14, 16], default=11)
    parser.add_argument('--motion-review', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    out = args.output.absolute()
    if out.exists():
        raise FileExistsError('Use a new delivery directory: '+str(out))
    terrain = args.terrain.resolve(strict=True)
    layout = args.layout.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    review = args.review.resolve(strict=True)
    if json.loads((terrain/'geometry_report.json').read_text()).get('passed') is not True:
        raise ValueError('Terrain geometry must have passed')
    if not (layout/'targets.json').is_file():
        raise ValueError('Missing randomized layout targets')
    validation_root = dataset/'dataset' if args.revision >= 14 and (dataset/'dataset/dataset_validation.json').is_file() else dataset
    report = json.loads((validation_root/'dataset_validation.json').read_text())
    if report.get('passed') is not True:
        raise ValueError('Dataset validation must pass before packaging')
    if json.loads((review/'render_report.json').read_text()).get('source_files_unchanged') is not True:
        raise ValueError('Missing real-render/source preservation evidence')
    out.mkdir(parents=True)
    omitted = ('__pycache__', '.git', '.env', 'REQUEST_STOP')
    # All project modules and CPU authoring tools are small. Shipping the
    # dependency closure avoids hidden imports from the author's workspace.
    for pattern in ('*.py', '*.sh'):
        for path in sorted(ROOT.glob(pattern)):
            shutil.copy2(path, out/path.name)
    for directory in ('cave', 'configs', 'assets', 'docs', 'scripts'):
        if (ROOT/directory).is_dir():
            copy_tree(ROOT/directory, out/directory, omit=omitted, link_payloads=directory == 'assets')
    for name in ('requirements-pipeline-v11.txt', '.env.pipeline-v11.example'):
        path = ROOT/name
        if path.is_file():
            shutil.copy2(path, out/name)
    copy_tree(terrain, out/'templates/lunar_fork_v11', omit=omitted, link_payloads=True)
    copy_tree(dataset, out/'examples/dataset_smoke', omit=omitted, link_payloads=True)
    # A visual review is not a policy input. Keep it in its own directory.
    (out/'review').mkdir(exist_ok=True)
    for path in sorted(review.glob('*.png')):
        shutil.copy2(path, out/'review'/path.name)
    shutil.copy2(review/'render_report.json', out/'review/render_report.json')
    if args.map:
        map_path = args.map.resolve(strict=True)
        shutil.copy2(map_path, out/'review/layout_map.png')
        shutil.copy2(map_path.with_suffix('.json'), out/'review/layout_map.json')
    if args.motion_review:
        copy_tree(args.motion_review, out/'review/motion', omit=omitted, link_payloads=True)
    if args.evidence:
        evidence = args.evidence.resolve(strict=True)
        for name in ('EARTH_FORK_EVIDENCE.md', 'LUNAR_EVIDENCE.md', 'REMOTE_2027_PREFLIGHT.json',
                     'junction_validation.json', 'ITERATION_REPORT.md'):
            if (evidence/name).is_file():
                (out/'docs/evidence').mkdir(parents=True, exist_ok=True)
                shutil.copy2(evidence/name, out/'docs/evidence'/name)
    guide = f'PIPELINE_V{args.revision}.md'
    (out/'README.md').write_text(
        f'# Lunar cave dataset pipeline V{args.revision}\n\n'
        f'完整入口与环境说明见 [docs/{guide}](docs/{guide})。\n\n'
        '阶段：地形与障碍生成 → 随机四目标布局 → 连续 RGB-D 轨迹 → '
        + ('真实路线 query 编译 / 当前图像短 CoT → 数据校验 → 独立队列发布。\n\n'
           if args.revision >= 16 else 'PPIO query / 逐步短 CoT → 数据校验。\n\n') +
        '提供 templates/lunar_fork_v11 和 examples/dataset_smoke。'
        + ('示例是完整单元，密集基底位于其 dataset/，稀疏视图和 query_partitions 与其并列。'
           if args.revision >= 14 else '') +
        '数据域为 isaac_continuous；不是现有 129 训练器已接入证明。\n\n'
        '运行前用 Python 3.11 执行 scripts/check_pipeline_runtime_v11.py；'
        '服务器系统默认 Python 3.6 仅可运行下面的完整性校验器。\n\n'
        '```bash\npython3 scripts/verify_delivery_v11.py .\n```\n')
    write(out/'DELIVERY.json', dict(schema_version=f'lunar_pipeline_delivery/{args.revision}.0',
        created_unix_s=time.time(), complete_pipeline_included=True,
        code=True, source_assets=True, terrain_template=True, example_layout=True,
        example_dataset=True, native_review=True, dataset_validation=report,
        credentials_included=False, model_weights_included=False,
        sdk_included=False, remote_runtime_validated=False,
        runtime_scope='Isaac Sim 5.1 / Python 3.11; host environment must be checked independently'))
    manifest = []
    for path in sorted(out.rglob('*')):
        if path.is_symlink():
            raise ValueError('No external symlinks allowed in delivery')
        if not path.is_file():
            continue
        if path.name.startswith('.env') and path.name != '.env.pipeline-v11.example':
            raise ValueError('Unexpected environment file in delivery')
        manifest.append(dict(path=path.relative_to(out).as_posix(), bytes=path.stat().st_size, sha256=digest(path)))
    write(out/'MANIFEST.json', dict(schema_version='sha256_manifest/1', files=manifest,
        total_bytes=sum(x['bytes'] for x in manifest), file_count=len(manifest)))
    print(json.dumps(dict(output=str(out), files=len(manifest), bytes=sum(x['bytes'] for x in manifest))))


if __name__ == '__main__':
    main()
