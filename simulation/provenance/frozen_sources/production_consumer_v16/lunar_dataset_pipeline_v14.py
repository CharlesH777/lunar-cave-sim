"""Reusable dense capture, semantic language and stride-five training unit.

Batch scheduling and remote publication are separate entry points. Every unit
is independently validated and can be moved without the author's workspace.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import threading
from lunar_dataset_pipeline_v11 import (
    ROOT, TARGETS, StopRequested, check_episode, directory_hashes, finalize_dataset,
    project_code_hash, resolve_project, run_stage as _run_stage)
from cave.ppio_client_v11 import object_hash, sha256, write_json
from validate_lunar_dataset_v11 import validate_dataset, validate_episode


def run_stage(name, output, result_dir, identity, command, **kwargs):
    output = Path(output)
    unit = output.parent
    done = threading.Event()
    def forward_stop():
        while not done.is_set():
            if (unit/'REQUEST_STOP').exists():
                (output/'REQUEST_STOP').touch(exist_ok=True)
            done.wait(.25)
    watcher = threading.Thread(target=forward_stop, daemon=True)
    watcher.start()
    try:
        return _run_stage(name, output, result_dir, identity, command, **kwargs)
    finally:
        done.set()
        watcher.join(timeout=1.)


def write_status(unit, output, value):
    write_json(output/'status.json', value)
    write_json(unit/'status.json', value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/lunar_dataset_pipeline_v14.json')
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-llm', action='store_true'); parser.add_argument('--max-episodes', type=int)
    parser.add_argument('--isaac-python'); parser.add_argument('--ppio-env', type=Path)
    parser.add_argument('--build-terrain', action='store_true'); parser.add_argument('--all-frames', action='store_true')
    parser.add_argument('--layout-source', type=Path, help='Adopt one immutable matching prebuilt layout')
    parser.add_argument('--raw-source', type=Path, help='Adopt one complete matching raw episode')
    parser.add_argument('--provider-cache', type=Path, help='Shared durable paid-request ledger across units')
    args = parser.parse_args(argv)
    if args.max_episodes is not None and args.max_episodes < 1: parser.error('--max-episodes must be positive')
    config = json.loads(args.config.resolve(strict=True).read_text())
    unit = args.output.expanduser().absolute()
    unit.mkdir(parents=True, exist_ok=args.resume)
    output = unit/'dataset'
    output.mkdir(parents=True, exist_ok=args.resume)
    view = unit/'training_stride5'
    write_status(unit, output, dict(phase='starting', passed=False, training_ready=False))
    runtime = args.isaac_python or os.environ.get('ISAAC_PYTHON') or config.get('isaac_python')
    if not runtime: raise RuntimeError('Set ISAAC_PYTHON or pass --isaac-python with Isaac Sim 5.1 Python')
    runtime_path = Path(runtime).expanduser()
    if runtime_path.is_absolute() or '/' in runtime:
        runtime = str(runtime_path.resolve(strict=True))
    else:
        import shutil
        found = shutil.which(runtime)
        if not found: raise RuntimeError('Isaac Python executable was not found')
        runtime = found
    layouts = config['layouts']; targets = config.get('targets', TARGETS)
    seed_splits = {}
    for item in layouts:
        if item['split'] not in ('train', 'val', 'test'): raise ValueError('Invalid split')
        if item['seed'] in seed_splits: raise ValueError('Duplicate layout seed')
        layout_targets = item.get('targets', targets)
        if (not isinstance(layout_targets, list) or not layout_targets
                or any(t not in TARGETS for t in layout_targets) or len(set(layout_targets)) != len(layout_targets)):
            raise ValueError('Unknown, duplicate or empty layout target selection')
        seed_splits[item['seed']] = item['split']
    if (args.layout_source or args.raw_source) and sum(len(item.get('targets', targets)) for item in layouts) != 1:
        raise ValueError('External layout/raw adoption requires exactly one configured episode')
    external_hashes = {}
    for name in ('layout_source', 'raw_source'):
        value = getattr(args, name)
        if value:
            external_hashes[name] = directory_hashes(value.expanduser().resolve(strict=True))
    annotation_config = config.get('annotation', {})
    resource_hashes = {}
    for name in ('semantic_config', 'query_protocol', 'query_bank'):
        if annotation_config.get(name):
            resource_hashes[name] = sha256(resolve_project(annotation_config[name]))
    effective = dict(external_input_sha256=external_hashes, external_resource_sha256=resource_hashes, config=config, skip_llm=args.skip_llm, all_frames=args.all_frames, build_terrain=args.build_terrain,
                     max_episodes=args.max_episodes, code_sha256=project_code_hash())
    run_identity = object_hash(effective)
    if (output/'run_config.json').exists():
        if json.loads((output/'run_config.json').read_text())['identity_sha256'] != run_identity:
            raise ValueError('Output belongs to different source/config/options; choose a new directory')
    else: write_json(output/'run_config.json', dict(identity_sha256=run_identity, **effective))
    completed_episodes = []; reports = []; started = time.monotonic()
    template = resolve_project(config['terrain_template']); code_hash = effective['code_sha256']
    try:
        build = args.build_terrain or config.get('terrain_build', {}).get('enabled', False)
        if build:
            template = output/'terrain_template'
            build_settings = config.get('terrain_build', {})
            substitutions = dict(python=runtime, root=str(ROOT), output=str(template))
            raw_command = build_settings.get('command', ['{python}', '{root}/build_lunar_terrain_v11_cpu.py', '--output', '{output}'])
            if not isinstance(raw_command, list) or any(not isinstance(x, str) for x in raw_command): raise ValueError('terrain_build.command must be an argument list, never shell code')
            command = [part.format(**substitutions) for part in raw_command]
            run_stage('terrain', output, template, dict(code_sha256=code_hash, settings=build_settings), command, cpu=True, resume=args.resume)
        if not template.is_dir(): raise FileNotFoundError('Missing packaged terrain template; use --build-terrain: '+str(template))
        if (template/'targets.json').exists(): raise ValueError('Terrain template already has SearchTargets; choose pre-placement terrain')
        terrain_hash = object_hash(directory_hashes(template))
        for item in layouts:
            if args.max_episodes and len(completed_episodes) >= args.max_episodes: break
            seed = item['seed']; split = item['split']; layout = output/'layouts'/f'seed_{seed}'
            if args.layout_source:
                layout_command = [sys.executable, str(ROOT/'stage_lunar_assets_v14_cpu.py'), '--kind', 'layout',
                    '--source', str(args.layout_source.expanduser().resolve(strict=True)), '--output', str(layout), '--expected-seed', str(seed)]
            else:
                layout_command = [runtime, str(ROOT/'build_lunar_search_demo_v8_cpu.py'), '--asset-revision', '10',
                    '--source', str(template), '--seed', str(seed), '--output', str(layout)]
            run_stage(f'layout_{seed}', output, layout, dict(terrain_sha256=terrain_hash, seed=seed,
                asset_revision=10, code_sha256=code_hash, adopted_source=external_hashes.get('layout_source')),
                layout_command, cpu=True, resume=args.resume)
            layout_hash = object_hash(directory_hashes(layout))
            for target in item.get('targets', targets):
                if args.max_episodes and len(completed_episodes) >= args.max_episodes: break
                episode_id = f'seed_{seed}_{target}'; episode_root = output/'episodes'/episode_id
                trajectory = episode_root/'trajectory'; language = episode_root/'language'
                command = [runtime, str(ROOT/'collect_lunar_trajectory_v14_isaac.py'), '--source', str(layout),
                    '--output', str(trajectory), '--episode-id', episode_id, '--target', target, '--seed', str(seed),
                    '--capture-stride', '1', '--max-steps', str(config.get('max_steps', 5000))]
                if config.get('experience'): command += ['--experience', str(resolve_project(config['experience']))]
                for folder in config.get('ext_folders', []): command += ['--ext-folder', str(resolve_project(folder))]
                if 'motion_controller' in config: command += ['--motion-controller', config['motion_controller']]
                landmark_settings = config.get('landmark_pass', {})
                if landmark_settings.get('enabled', False):
                    command += ['--via-landmark', landmark_settings.get('landmark_id', 'auto')]
                    command += ['--landmark-pass-policy', landmark_settings.get('policy', 'required')]
                    if landmark_settings.get('settings'):
                        settings_path = output/'settings/landmark_pass.json'
                        write_json(settings_path, dict(settings=landmark_settings['settings']))
                        command += ['--landmark-pass-config', str(settings_path)]
                if args.raw_source:
                    command = [sys.executable, str(ROOT/'stage_lunar_assets_v14_cpu.py'), '--kind', 'raw',
                        '--source', str(args.raw_source.expanduser().resolve(strict=True)), '--output', str(trajectory),
                        '--layout', str(layout), '--expected-episode', episode_id, '--expected-target', target]
                run_stage('capture_'+episode_id, output, trajectory, dict(layout_sha256=layout_hash,
                    target_id=target, code_sha256=code_hash, capture_settings=command[2:], landmark_pass=landmark_settings,
                    adopted_source=external_hashes.get('raw_source')), command, cpu=bool(args.raw_source), resume=args.resume)
                raw_check = validate_episode(trajectory, layout=layout, require_cot=False)
                write_json(episode_root/'raw_validation.json', raw_check)
                if not raw_check['passed']:
                    raise ValueError('Dense raw integrity failed before language/API: '+episode_id)
                raw_metadata = json.loads((trajectory/'metadata.json').read_text())
                if raw_metadata.get('motion_controller') != config.get('motion_controller', 'smooth_v13'):
                    raise ValueError('Adopted motion controller differs from configured constraints')
                annotation = config.get('annotation', {})
                if annotation.get('instruction_style') == 'route_grounded':
                    derived = episode_root/'route_trajectory'
                    materializer = ('cave.lunar_route_materializer_v14' if annotation.get('require_semantic_visibility') == 'v14'
                                    else 'cave.lunar_route_language_v13')
                    command = [sys.executable, '-m', materializer, '--source', str(trajectory),
                               '--output', str(derived), '--layout', str(layout)]
                    if annotation.get('require_semantic_visibility') == 'v14':
                        command += ['--semantic-config', str(resolve_project(annotation['semantic_config']))]
                    if annotation.get('fork_evidence_revision'):
                        command += ['--fork-evidence-revision', annotation['fork_evidence_revision']]
                    run_stage('route_instruction_'+episode_id, output, derived,
                        dict(raw_capture_sha256=object_hash(directory_hashes(trajectory)), instruction_style='route_grounded',
                             code_sha256=code_hash, semantic_resource_sha256=resource_hashes.get('semantic_config')), command, cpu=True, resume=args.resume)
                    trajectory = derived
                command = [sys.executable, str(ROOT/'annotate_lunar_trajectory_v11_ppio.py'), '--source', str(trajectory),
                    '--output', str(language), '--split', split, '--cache', str(args.provider_cache.expanduser().absolute() if args.provider_cache else output/'provider_cache'),
                    '--budget-cny', str(annotation.get('budget_cny', 2.)),
                    '--max-llm-frames', str(annotation.get('max_llm_frames', 16)),
                    '--model', annotation.get('model', 'qwen/qwen3.8-27b'),
                    '--review-model', annotation.get('review_model', 'qwen/qwen3.8-max')]
                if annotation.get('require_semantic_visibility'):
                    command += ['--require-semantic-visibility', annotation['require_semantic_visibility']]
                if annotation.get('query_count') is not None:
                    command += ['--query-count', str(annotation['query_count'])]
                if annotation.get('query_model'):
                    command += ['--query-model', annotation['query_model']]
                if annotation.get('query_mode'):
                    command += ['--query-mode', annotation['query_mode']]
                if annotation.get('query_bank'):
                    command += ['--query-bank', str(resolve_project(annotation['query_bank']))]
                if annotation.get('query_protocol'):
                    command += ['--query-protocol', str(resolve_project(annotation['query_protocol']))]
                if args.skip_llm: command.append('--skip-llm')
                if args.all_frames or annotation.get('all_frames', False): command.append('--all-frames')
                if 'cot_style' in annotation: command += ['--cot-style', annotation['cot_style']]
                if 'instruction_style' in annotation: command += ['--instruction-style', annotation['instruction_style']]
                env_path = args.ppio_env or annotation.get('env_file')
                if env_path: command += ['--ppio-env', str(Path(env_path).expanduser().absolute())]
                run_stage('language_'+episode_id, output, language, dict(trajectory_sha256=object_hash(directory_hashes(trajectory)),
                    annotation_settings=annotation, skip_llm=args.skip_llm, all_frames=args.all_frames,
                    split=split, code_sha256=code_hash, query_resource_sha256=resource_hashes.get('query_protocol'),
                    query_bank_sha256=resource_hashes.get('query_bank')), command, cpu=True, resume=args.resume)
                entry, report = check_episode(output, episode_id, seed, split, target, require_cot=not args.skip_llm,
                                              trajectory_subdir=trajectory.name)
                for relative in (entry['validation'], str(Path(entry['trajectory'])/'semantic_visibility_config.json'),
                                 str(Path(entry['trajectory'])/'semantic_visibility_enrichment.json')):
                    if (output/relative).is_file():
                        entry['sha256'][relative] = sha256(output/relative)
                completed_episodes.append(entry)
                reports.append(report)
                write_json(output/'progress.json', dict(completed_episodes=len(completed_episodes),
                    elapsed_seconds=time.monotonic()-started, last_episode=episode_id, training_ready=False))
                if (output/'REQUEST_STOP').exists(): raise StopRequested('REQUEST_STOP after episode')
        finalize_dataset(output, completed_episodes, reports, code_hash)
        manifest_path = output/'dataset_manifest.json'
        finalized = json.loads(manifest_path.read_text())
        actual_limits = [json.loads((output/entry['trajectory']/'metadata.json').read_text())
                         for entry in completed_episodes]
        finalized['controls']['max_v_mps'] = max(row['max_v_mps'] for row in actual_limits)
        finalized['controls']['max_abs_omega_radps'] = max(row['max_abs_omega_radps'] for row in actual_limits)
        write_json(manifest_path, finalized)
        validation = validate_dataset(output, require_cot=not args.skip_llm)
        write_json(output/'dataset_validation.json', validation)
        if not validation['passed']:
            raise ValueError('Final dense dataset validation failed')
        training = config.get('training_view', {})
        if training.get('enabled', True):
            command = [sys.executable, str(ROOT/'export_lunar_training_stride_v14_cpu.py'), '--base', str(output),
                '--output', str(view), '--stride', str(training.get('stride', 5)),
                '--history-frames', str(training.get('history_frames', 8))]
            run_stage('training_view', output, view,
                dict(dense_manifest_sha256=sha256(output/'dataset_manifest.json'),
                     reader_sha256=sha256(output/'reader_index.jsonl'), settings=training, code_sha256=code_hash),
                command, cpu=True, resume=args.resume)
            if not args.skip_llm and annotation_config.get('query_protocol'):
                partition = unit/'query_partitions'
                command = [sys.executable, str(ROOT/'build_lunar_query_indexes_v14_cpu.py'),
                    '--base', str(output), '--view', str(view),
                    '--output', str(partition), '--protocol', str(resolve_project(annotation_config['query_protocol']))]
                run_stage('query_partitions', output, partition,
                    dict(view_manifest_sha256=sha256(view/'view_manifest.json'),
                         query_protocol_sha256=resource_hashes.get('query_protocol'), code_sha256=code_hash),
                    command, cpu=True, resume=args.resume)
                write_json(unit/'TRAINING_INPUTS.json', dict(
                    schema_version='lunar_training_inputs/14.0',
                    default_training_index='query_partitions/query_train_index.jsonl',
                    known_route_query_evaluation_index='query_partitions/query_eval_index.jsonl',
                    physical_validation_index='query_partitions/physical_val_index.jsonl',
                    physical_test_index='query_partitions/physical_test_index.jsonl',
                    observation_stride=training.get('stride', 5), native_action_dt_s=.2,
                    query_families_held_out_from_training=True, training_started=False))
        if config.get('review', {}).get('enabled', True):
            command = [sys.executable, str(ROOT/'render_lunar_dataset_review_v14_cpu.py'),
                '--dataset', str(output), '--output', str(unit/'review')]
            if config.get('review', {}).get('font'):
                command += ['--font', str(resolve_project(config['review']['font']))]
            run_stage('review', output, unit/'review',
                dict(dense_manifest_sha256=sha256(output/'dataset_manifest.json'),
                     review_settings=config.get('review', {}), code_sha256=code_hash),
                command, cpu=True, resume=args.resume)
        write_status(unit, output, dict(phase='complete', passed=True, episode_count=len(completed_episodes),
            elapsed_seconds=time.monotonic()-started, domain='isaac_continuous', training_ready=False))
        print(json.dumps(dict(phase='complete', episode_count=len(completed_episodes), manifest=str(output/'dataset_manifest.json'))))
        return 0
    except BaseException as exc:
        stopped = isinstance(exc, StopRequested)
        write_status(unit, output, dict(phase='stopped' if stopped else 'failed', passed=False,
            completed_episodes=len(completed_episodes), error_type=type(exc).__name__,
            # Stage errors contain only generated paths and sanitized client errors.
            error=str(exc), automatic_retry=False, domain='isaac_continuous', training_ready=False))
        print(json.dumps(dict(phase='stopped' if stopped else 'failed', error_type=type(exc).__name__, error=str(exc))))
        return 2 if stopped else 1


if __name__ == '__main__':
    raise SystemExit(main())
