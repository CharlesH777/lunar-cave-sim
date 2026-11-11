"""Finish PPIO language and validation for hash-verified completed captures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from annotate_lunar_trajectory_v11_ppio import (
    MAIN_MODEL, OUTPUT_SCHEMA, QUERY_SCHEMA, REVIEW_MODEL, annotate, annotation_images,
    frame_prompt, query_prompt, read_jsonl, select_frames)
from cave.ppio_client_v11 import PPIOClient, object_hash, sha256, write_json
from lunar_dataset_pipeline_v11 import check_episode, directory_hashes, finalize_dataset
from validate_lunar_dataset_v11 import validate_dataset


def read_json(path):
    return json.loads(path.read_text())


def verify_stage(source, name, directory):
    marker_path = source/'stage_cache'/(name+'.json')
    marker = read_json(marker_path)
    if (not marker.get('complete') or marker.get('exit_code') != 0 or
            marker['identity_sha256'] != object_hash(marker['identity'])):
        raise ValueError('Stage did not complete with a valid identity: '+name)
    if directory_hashes(source/directory) != marker['artifact_sha256']:
        raise ValueError('Completed stage artifacts changed: '+name)
    return marker


def copy_verified(source, output, relative, hashes):
    destination = output/relative
    if not destination.exists():
        def link_or_copy(src, dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        shutil.copytree(source/relative, destination, copy_function=link_or_copy)
    if directory_hashes(destination) != hashes:
        raise ValueError('Recovered artifacts do not match completed stage: '+str(relative))


def reconstruct_reservations(client, trajectory, model, maximum, all_frames):
    """Recover only deterministic request audits whose fingerprints match."""
    frames = read_jsonl(trajectory/'frame_records.jsonl')
    first = frames[0]
    selected = select_frames(frames, maximum, all_frames)
    attempts = [(query_prompt(first['target_id'], first['instruction']), QUERY_SCHEMA, (), 'episode_queries')]
    attempts.extend((frame_prompt(frame), OUTPUT_SCHEMA, annotation_images(trajectory, frame), frame['sample_id'])
                    for frame in frames if frame['sample_id'] in selected)
    for prompt, schema, images, item_id in attempts:
        audit = client.request_audit(prompt, schema, model, images)
        fingerprint = audit['request_sha256']
        has_interrupted_artifact = any((client.cache/(fingerprint+f'.{suffix}.json')).exists()
                                       for suffix in ('pending', 'response', 'error'))
        if not has_interrupted_artifact or client._saved_response(fingerprint) is not None:
            continue
        try:
            client._unresolved_reserve(fingerprint)
        except RuntimeError:
            write_json(client.cache/(fingerprint+'.recovered_request.json'),
                       dict(audit, recovered_from='exact prompt, model and image fingerprint', item_id=item_id),
                       durable=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ppio-env', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output == source or output.is_relative_to(source):
        parser.error('Choose a separate recovery output directory')
    run = read_json(source/'run_config.json')
    old_identity = run['identity_sha256']
    if object_hash({k: v for k, v in run.items() if k != 'identity_sha256'}) != old_identity:
        raise ValueError('Original run configuration identity does not match')
    config = run['config']
    splits = {item['seed']: item['split'] for item in config['layouts']}
    settings = config.get('annotation', {})
    model = settings.get('model', MAIN_MODEL)
    review_model = settings.get('review_model', REVIEW_MODEL)
    maximum = settings.get('max_llm_frames', 16)
    all_frames = run.get('all_frames', False) or settings.get('all_frames', False)
    episodes = []
    stages = {}
    for marker_path in sorted((source/'stage_cache').glob('capture_*.json')):
        episode_id = marker_path.stem.removeprefix('capture_')
        relative = Path('episodes')/episode_id/'trajectory'
        if not read_json(marker_path).get('complete'):
            continue
        capture = verify_stage(source, marker_path.stem, relative)
        metadata = read_json(source/relative/'metadata.json')
        seed = metadata['source_layout_seed']
        if metadata['episode_id'] != episode_id or seed not in splits:
            raise ValueError('Completed capture identity differs from the run configuration')
        layout_name = f'layout_{seed}'
        layout_dir = Path('layouts')/f'seed_{seed}'
        if layout_name not in stages:
            stages[layout_name] = (layout_dir, verify_stage(source, layout_name, layout_dir))
        layout = stages[layout_name][1]
        if capture['identity']['layout_sha256'] != object_hash(layout['artifact_sha256']):
            raise ValueError('Capture identity does not bind to the verified layout')
        stages[marker_path.stem] = (relative, capture)
        episodes.append((episode_id, seed, splits[seed], metadata['target_id']))
    if not episodes:
        raise ValueError('No completed captures to recover')
    provenance = dict(source_run_identity_sha256=old_identity, capture_code_sha256=run['code_sha256'],
        capture_stage_identity_sha256={name: marker['identity_sha256'] for name, (_, marker) in stages.items()},
        source_run_config='capture_run_config.json', source_stage_markers='reused_stage_evidence',
        provider_cache_import='byte-preserved public request/response artifacts; incomplete fingerprints reserved and skipped')
    output.mkdir(parents=True, exist_ok=args.resume)
    provenance_path = output/'recovery_provenance.json'
    if provenance_path.exists() and read_json(provenance_path) != provenance:
        raise ValueError('Recovery output belongs to a different capture')
    write_json(provenance_path, provenance)
    shutil.copy2(source/'run_config.json', output/'capture_run_config.json')
    for name, (relative, marker) in stages.items():
        copy_verified(source, output, relative, marker['artifact_sha256'])
        write_json(output/'reused_stage_evidence'/(name+'.json'), marker)

    cache = output/'provider_cache'
    cache.mkdir(exist_ok=True)
    original_cache = source/'provider_cache'
    for original in original_cache.glob('*.json'):
        restored = cache/original.name
        if not restored.exists():
            shutil.copy2(original, restored)
        if sha256(restored) != sha256(original):
            raise ValueError('Imported PPIO artifact differs from source: '+original.name)
    client = PPIOClient(cache, settings.get('budget_cny', 2.), args.ppio_env or settings.get('env_file'))
    for episode_id, *_ in episodes:
        reconstruct_reservations(client, output/'episodes'/episode_id/'trajectory', model, maximum, all_frames)
    print(json.dumps(dict(phase='annotating_recovered_capture', episodes=len(episodes), usage=client.usage())), flush=True)
    entries = []; reports = []; language_summaries = []
    for episode_id, seed, split, target in episodes:
        episode_root = output/'episodes'/episode_id
        language = episode_root/'language'
        report = annotate(episode_root/'trajectory', language, client=client, max_llm_frames=maximum,
                          all_frames=all_frames, model=model, review_model=review_model,
                          resume=args.resume, split=split)
        if not report['complete']:
            raise RuntimeError('Language recovery paused: '+str(report['stop_reason']))
        entry, validation = check_episode(output, episode_id, seed, split, target, require_cot=True)
        entries.append(entry)
        reports.append(validation)
        language_summaries.append({k: report[k] for k in ('frame_count', 'selected_count', 'cot_valid_count',
                                  'rejected_attempt_count', 'generated_queries')})
    code_files = ('recover_lunar_dataset_v11.py', 'lunar_dataset_pipeline_v11.py',
                  'annotate_lunar_trajectory_v11_ppio.py', 'cave/ppio_client_v11.py', 'validate_lunar_dataset_v11.py')
    code_hash = object_hash({name: sha256(Path(__file__).resolve().parent/name) for name in code_files})
    finalize_dataset(output, entries, reports, code_hash, recovery_provenance=provenance_path.name)
    validation = validate_dataset(output, require_cot=True)
    write_json(output/'dataset_validation.json', validation)
    if not validation['passed']:
        raise RuntimeError('Dataset-level validation failed: '+str(output/'dataset_validation.json'))
    status = dict(phase='complete', passed=True, episode_count=len(entries), language=language_summaries,
                  usage=client.usage(), annotation_domain='isaac_continuous', training_ready=False)
    write_json(output/'status.json', status)
    print(json.dumps(status), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
