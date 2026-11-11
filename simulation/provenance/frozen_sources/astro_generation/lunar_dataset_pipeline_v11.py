"""Portable layout -> Isaac teacher -> PPIO summaries -> validated dataset.

Each stage runs once, preserves failures and never trains a model. --resume
accepts only stage outputs whose code, inputs and complete artifacts still hash
match. A partial capture is kept for inspection and is not silently restarted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cave.ppio_client_v11 import object_hash, sha256, write_json
from annotate_lunar_trajectory_v11_ppio import read_jsonl, write_jsonl
from validate_lunar_dataset_v11 import validate_episode

ROOT = Path(__file__).resolve().parent
TARGETS = ['astronaut', 'lunar_rover', 'lunar_base', 'chinese_flag']


class StopRequested(RuntimeError): pass


def directory_hashes(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): sha256(path) for path in sorted(directory.rglob('*'))
            if path.is_file() and path.name != 'REQUEST_STOP' and not path.name.endswith('.tmp')}


def project_code_hash():
    # Include the raw cave/junction authoring dependency closure as well as the
    # high-level stages. Hashing these small source files needs no SDK import.
    files = list(ROOT.glob('*.py')) + list((ROOT/'cave').glob('*.py'))
    return object_hash({str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(files)) if path.is_file()})


def resolve_project(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def run_stage(name, output, result_dir, identity, command, *, cpu=False, resume=False):
    """No shell, no retries, no overwriting old/partial simulation outputs."""
    output = Path(output); result_dir = Path(result_dir)
    marker = output/'stage_cache'/(name+'.json'); fingerprint = object_hash(identity)
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['identity_sha256'] != fingerprint: raise ValueError('Stage source/config/code hash mismatch: '+name)
        if not old.get('complete'): raise RuntimeError('Stage previously failed or stopped; inspect and choose a new output: '+name)
        if not result_dir.is_dir() or old['artifact_sha256'] != directory_hashes(result_dir):
            raise ValueError('Stage artifacts changed: '+name)
        if not resume: raise FileExistsError('Use --resume to reuse existing stage '+name)
        return True
    if result_dir.exists(): raise FileExistsError('Uncached/partial result exists; preserved without automatic retry: '+str(result_dir))
    if (output/'REQUEST_STOP').exists(): raise StopRequested('REQUEST_STOP before '+name)
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    log = output/'logs'/(name+'.log'); log.parent.mkdir(parents=True, exist_ok=True)
    write_json(marker, dict(identity_sha256=fingerprint, identity=identity, complete=False, state='starting'))
    environment = os.environ.copy()
    if cpu: environment['CUDA_VISIBLE_DEVICES'] = ''
    write_json(output/'status.json', dict(phase=name, command=command, domain='isaac_continuous', training_ready=False,
        updated_utc=datetime.now(timezone.utc).isoformat()))
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        requested = False
        while process.poll() is None:
            if (output/'REQUEST_STOP').exists() and result_dir.exists():
                (result_dir/'REQUEST_STOP').touch(exist_ok=True); requested = True
            time.sleep(.5)
    code = process.returncode
    completed = code == 0 and result_dir.is_dir()
    write_json(marker, dict(identity_sha256=fingerprint, identity=identity, complete=completed,
        state='complete' if completed else 'partial_or_failed', exit_code=code,
        stop_requested=requested, artifact_sha256=directory_hashes(result_dir) if result_dir.is_dir() else {}))
    if requested or (output/'REQUEST_STOP').exists(): raise StopRequested('REQUEST_STOP after '+name)
    if not completed: raise RuntimeError('Stage failed; no automatic retry: '+name+'; inspect '+str(log))
    return False


def reader_rows(dataset_root, episodes):
    """Reader JSON separates policy inputs from privileged label references."""
    dataset_root = Path(dataset_root)
    for episode in episodes:
        trajectory = dataset_root/episode['trajectory']; language = dataset_root/episode['language']
        frames = read_jsonl(trajectory/'frame_records.jsonl'); annotations = read_jsonl(language/'annotations.jsonl')
        queries = json.loads((language/'queries.json').read_text())
        variants = [dict(language_episode_id=episode['episode_id']+':canonical', instruction=queries['canonical_instruction'])]+queries['variants']
        prefix = Path(episode['trajectory'])
        for variant in variants:
            language_id = variant['language_episode_id']
            for frame, annotation in zip(frames, annotations):
                def observation(record):
                    return dict(observation_id=record['observation_id'], rgb=str(prefix/record['rgb']),
                        depth=str(prefix/record['depth']), timestamp_s=record['timestamp_s'])
                supervision = dict(continuous_controls=str(prefix/'supervision.npz'), step_index=frame['step_index'],
                    cot_valid=annotation['cot_valid'], text=annotation['output'] if annotation['cot_valid'] else None)
                if 'cot_text' in annotation:
                    supervision['cot_text'] = annotation['cot_text']
                yield dict(schema_version='lunar_reader_row/11.0', domain='isaac_continuous',
                    sample_id=language_id+f":{frame['step_index']:06d}", episode_id=language_id,
                    physical_trajectory_id=episode['episode_id'], layout_seed=episode['layout_seed'],
                    split=episode['split'], step_index=frame['step_index'], training_ready=False,
                    inputs=dict(instruction=variant['instruction'], observation=observation(frame['inputs']['observation']),
                        history=[observation(h) for h in frame['inputs']['history']]),
                    supervision=supervision,
                    privileged=dict(teacher_only=str(prefix/'teacher_only.npz'),
                        machine_evidence=str(Path(episode['language'])/'annotations.jsonl')))


def check_episode(output, episode_id, seed, split, target, *, require_cot, trajectory_subdir='trajectory'):
    episode_root = output/'episodes'/episode_id
    trajectory = episode_root/trajectory_subdir; language = episode_root/'language'
    layout = output/'layouts'/f'seed_{seed}'
    report = validate_episode(trajectory, language, layout, require_cot=require_cot)
    validation_path = episode_root/'validation.json'; write_json(validation_path, report)
    if not report['passed']:
        raise RuntimeError('Episode validation failed: '+episode_id+'; see '+str(validation_path))
    relative = lambda path: str(path.relative_to(output))
    files = [trajectory/'metadata.json', trajectory/'supervision.npz', trajectory/'teacher_only.npz',
             trajectory/'frame_records.jsonl', language/'queries.json', language/'annotations.jsonl']
    metadata = json.loads((trajectory/'metadata.json').read_text())
    if metadata.get('instruction_style') == 'route_grounded':
        files += [trajectory/'route_contract.json', trajectory/'instruction_binding.json', language/'protocol.json']
    entry = dict(episode_id=episode_id, layout_seed=seed, split=split, target_id=target,
        trajectory=relative(trajectory), language=relative(language), layout=relative(layout),
        validation=relative(validation_path), sha256={relative(path): sha256(path) for path in files})
    if metadata.get('instruction_style') == 'route_grounded':
        entry.update(instruction_style='route_grounded', raw_trajectory=relative(episode_root/'trajectory'),
                     route_contract=relative(trajectory/'route_contract.json'))
    if metadata.get('motion_controller') == 'smooth_v13':
        entry.update(motion_controller='smooth_v13', canonical_terminal_radius_m=1.0,
                     planned_stop_tolerance_m=.05, stationary_turns_allowed=False)
    return entry, dict(episode_id=episode_id, **report)


def finalize_dataset(output, episodes, reports, code_hash, *, recovery_provenance=None):
    if not episodes or not all(r['passed'] for r in reports):
        raise ValueError('Only validated nonempty datasets can be finalized')
    manifest = dict(schema_version='lunar_dataset_manifest/11.0', annotation_domain='isaac_continuous',
        complete=True, episodes=episodes, training_ready=False, source_code_sha256=code_hash,
        policy_inputs_allowlist=['instruction', 'observation.rgb', 'observation.depth_m', 'history (strictly past same language episode)'],
        coordinates='world Z-up metres; camera OpenCV right/down/forward; depth axial Z metres',
        controls=dict(dt_s=.2, max_v_mps=.4, max_abs_omega_radps=.8, horizon=8, unique_terminal_stop=True),
        split_rule='layout seed -> split; all physical-trajectory language aliases inherit the same split',
        terminal_teacher_m=.05, near_goal_stop_training_threshold_m=1., evaluation_success_distance_m=3.,
        near_goal_stop_labels_provided=False, native_action_labels_provided=False,
        current_129_adapter_implemented=False, current_129_text_loss_supported=False,
        physics_scope='collision-checked kinematic teacher; no tyre/soil dynamics; no learned-policy claim')
    controllers = [json.loads((Path(output)/episode['trajectory']/'metadata.json').read_text()).get('motion_controller', 'segment')
                   for episode in episodes]
    if all(controller == 'smooth_v13' for controller in controllers):
        manifest.pop('terminal_teacher_m')
        manifest.update(motion_controller='smooth_v13', canonical_terminal_radius_m=1.0,
                        planned_stop_tolerance_m=.05, stationary_turns_allowed=False)
    elif any(controller == 'smooth_v13' for controller in controllers):
        manifest.pop('terminal_teacher_m')
        manifest['terminal_policy'] = 'per_episode'
        manifest['episodes'] = [dict(episode, terminal_criterion=(
            dict(canonical_terminal_radius_m=1.0, planned_stop_tolerance_m=.05, stationary_turns_allowed=False)
            if controller == 'smooth_v13' else dict(terminal_teacher_m=.05, stationary_turns_allowed=True)))
            for episode, controller in zip(episodes, controllers)]
    if recovery_provenance:
        manifest['recovery_provenance'] = recovery_provenance
    if all(episode.get('instruction_style') == 'route_grounded' for episode in episodes):
        manifest['instruction_style'] = 'route_grounded'
    write_jsonl(output/'reader_index.jsonl', reader_rows(output, episodes))
    write_json(output/'validation/report.json', dict(passed=True, episodes=reports,
        annotation_domain='isaac_continuous', training_ready=False))
    write_json(output/'dataset_manifest.json', manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/lunar_dataset_pipeline_v11.json')
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-llm', action='store_true'); parser.add_argument('--max-episodes', type=int)
    parser.add_argument('--isaac-python'); parser.add_argument('--ppio-env', type=Path)
    parser.add_argument('--build-terrain', action='store_true'); parser.add_argument('--all-frames', action='store_true')
    args = parser.parse_args(argv)
    if args.max_episodes is not None and args.max_episodes < 1: parser.error('--max-episodes must be positive')
    config = json.loads(args.config.resolve(strict=True).read_text())
    output = args.output.expanduser().absolute(); output.mkdir(parents=True, exist_ok=args.resume)
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
    effective = dict(config=config, skip_llm=args.skip_llm, all_frames=args.all_frames, build_terrain=args.build_terrain,
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
            run_stage(f'layout_{seed}', output, layout, dict(terrain_sha256=terrain_hash, seed=seed,
                asset_revision=10, code_sha256=code_hash), [runtime, str(ROOT/'build_lunar_search_demo_v8_cpu.py'),
                '--asset-revision', '10', '--source', str(template), '--seed', str(seed), '--output', str(layout)], cpu=True, resume=args.resume)
            layout_hash = object_hash(directory_hashes(layout))
            for target in item.get('targets', targets):
                if args.max_episodes and len(completed_episodes) >= args.max_episodes: break
                episode_id = f'seed_{seed}_{target}'; episode_root = output/'episodes'/episode_id
                trajectory = episode_root/'trajectory'; language = episode_root/'language'
                command = [runtime, str(ROOT/'collect_lunar_trajectory_v11_isaac.py'), '--source', str(layout),
                    '--output', str(trajectory), '--episode-id', episode_id, '--target', target, '--seed', str(seed),
                    '--capture-stride', '1', '--max-steps', str(config.get('max_steps', 5000))]
                if config.get('experience'): command += ['--experience', str(resolve_project(config['experience']))]
                for folder in config.get('ext_folders', []): command += ['--ext-folder', str(resolve_project(folder))]
                if 'motion_controller' in config: command += ['--motion-controller', config['motion_controller']]
                run_stage('capture_'+episode_id, output, trajectory, dict(layout_sha256=layout_hash,
                    target_id=target, code_sha256=code_hash, capture_settings=command[2:]), command, resume=args.resume)
                annotation = config.get('annotation', {})
                if annotation.get('instruction_style') == 'route_grounded':
                    derived = episode_root/'route_trajectory'
                    command = [sys.executable, '-m', 'cave.lunar_route_language_v13', '--source', str(trajectory),
                               '--output', str(derived), '--layout', str(layout)]
                    if annotation.get('fork_evidence_revision'):
                        command += ['--fork-evidence-revision', annotation['fork_evidence_revision']]
                    run_stage('route_instruction_'+episode_id, output, derived,
                        dict(raw_capture_sha256=object_hash(directory_hashes(trajectory)), instruction_style='route_grounded',
                             code_sha256=code_hash), command, cpu=True, resume=args.resume)
                    trajectory = derived
                command = [sys.executable, str(ROOT/'annotate_lunar_trajectory_v11_ppio.py'), '--source', str(trajectory),
                    '--output', str(language), '--split', split, '--cache', str(output/'provider_cache'),
                    '--budget-cny', str(annotation.get('budget_cny', 2.)),
                    '--max-llm-frames', str(annotation.get('max_llm_frames', 16)),
                    '--model', annotation.get('model', 'qwen/qwen3.8-27b'),
                    '--review-model', annotation.get('review_model', 'qwen/qwen3.8-max')]
                if args.skip_llm: command.append('--skip-llm')
                if args.all_frames or annotation.get('all_frames', False): command.append('--all-frames')
                if 'cot_style' in annotation: command += ['--cot-style', annotation['cot_style']]
                if 'instruction_style' in annotation: command += ['--instruction-style', annotation['instruction_style']]
                env_path = args.ppio_env or annotation.get('env_file')
                if env_path: command += ['--ppio-env', str(Path(env_path).expanduser().absolute())]
                run_stage('language_'+episode_id, output, language, dict(trajectory_sha256=object_hash(directory_hashes(trajectory)),
                    annotation_settings=annotation, skip_llm=args.skip_llm, all_frames=args.all_frames,
                    split=split, code_sha256=code_hash), command, cpu=True, resume=args.resume)
                entry, report = check_episode(output, episode_id, seed, split, target, require_cot=not args.skip_llm,
                                              trajectory_subdir=trajectory.name)
                completed_episodes.append(entry)
                reports.append(report)
                write_json(output/'progress.json', dict(completed_episodes=len(completed_episodes),
                    elapsed_seconds=time.monotonic()-started, last_episode=episode_id, training_ready=False))
                if (output/'REQUEST_STOP').exists(): raise StopRequested('REQUEST_STOP after episode')
        finalize_dataset(output, completed_episodes, reports, code_hash)
        write_json(output/'status.json', dict(phase='complete', passed=True, episode_count=len(completed_episodes),
            elapsed_seconds=time.monotonic()-started, domain='isaac_continuous', training_ready=False))
        print(json.dumps(dict(phase='complete', episode_count=len(completed_episodes), manifest=str(output/'dataset_manifest.json'))))
        return 0
    except BaseException as exc:
        stopped = isinstance(exc, StopRequested)
        write_json(output/'status.json', dict(phase='stopped' if stopped else 'failed', passed=False,
            completed_episodes=len(completed_episodes), error_type=type(exc).__name__,
            # Stage errors contain only generated paths and sanitized client errors.
            error=str(exc), automatic_retry=False, domain='isaac_continuous', training_ready=False))
        print(json.dumps(dict(phase='stopped' if stopped else 'failed', error_type=type(exc).__name__, error=str(exc))))
        return 2 if stopped else 1


if __name__ == '__main__': raise SystemExit(main())
