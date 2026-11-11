"""Independent CPU annotation/publication consumer for immutable raw-ready units.

Default is a local plan, not execution. --run makes one annotation attempt per
new unit. Failed units become needs_review; they are never automatically
retried. This process never sends signals or STOP files to the raw producer.
Optional reclamation removes only this consumer's fully published bulk frames.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from cave.lunar_batch_catalog_v14 import build_catalog, summarize_unit
from cave.lunar_annotation_retention_v16 import reclaim_owned_frames
from cave.lunar_dataset_delivery_v14 import publish_unit, verify_published_unit
from cave.ppio_client_v11 import PPIOClient, object_hash, sha256, write_json


ROOT = Path(__file__).resolve().parent
TARGETS = ('astronaut', 'lunar_rover', 'lunar_base', 'chinese_flag')
HEX = re.compile(r'^[0-9a-f]{64}$')
IDENTITY_KEYS = ('unit_id', 'layout_seed', 'target_id', 'split', 'physical_slot', 'ordinal')


def read(path):
    return json.loads(Path(path).read_text())


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def child(root, relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or '\\' in relative or any(ord(c) < 32 for c in relative)
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError('Unsafe raw-unit relative path')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Raw-unit reference escapes its root')
    return path


def stable_json(path):
    before = sha256(path)
    value = read(path)
    if sha256(path) != before:
        raise ValueError('Atomic input manifest changed while being read')
    return value, before


def code_hashes():
    return {path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted([*ROOT.glob('*.py'), *(ROOT / 'cave').glob('*.py')])}


def raw_binding(ready):
    return dict(unit_id=ready['unit_id'], raw_manifest_sha256=ready['raw_manifest_sha256'],
                raw_delivery_manifest_sha256=ready['raw_delivery']['manifest_sha256'])


def validate_ready(ready, *, producer_root, plan, plan_sha256, host, remote_root):
    """Hash-bound input/control checks only; no simulator, API, or remote call."""
    if ready.get('schema_version') != 'lunar_raw_ready/16.0' or ready.get('complete', True) is not True:
        raise ValueError('Not a complete V16 raw-ready manifest')
    if ready.get('plan_sha256') != plan_sha256:
        raise ValueError('Ready input belongs to another frozen physical plan')
    identity = ready['unit_id']
    if identity != f"seed_{ready['layout_seed']}_{ready['target_id']}" or ready['target_id'] not in TARGETS:
        raise ValueError('Raw-ready physical identity is inconsistent')
    slots = {row['physical_slot']: row for row in plan['slots']}
    slot = slots[ready['physical_slot']]
    if (ready['split'] != slot['split']
            or ready['ordinal'] != ready['physical_slot'] * 4 + TARGETS.index(ready['target_id'])):
        raise ValueError('Ready split/ordinal differs from its physical slot')
    relative = 'raw_inputs/' + identity
    if ready['producer_relative_root'] != relative:
        raise ValueError('Raw-ready source is not its own immutable unit')
    source = child(producer_root, relative)
    if Path(ready['source']['root']).resolve(strict=True) != source:
        raise ValueError('Raw-ready absolute source differs from its bound producer-relative root')
    manifest_path = source / 'raw_manifest.json'
    manifest, manifest_sha = stable_json(manifest_path)
    if manifest_sha != ready['raw_manifest_sha256'] or manifest.get('schema_version') != 'lunar_raw_unit/16.0' or manifest.get('complete') is not True:
        raise ValueError('Raw unit manifest is incomplete or changed')
    if any(manifest[key] != ready[key] for key in IDENTITY_KEYS):
        raise ValueError('Raw manifest identity changed')
    if manifest['native_action_dt_s'] != .2 or manifest['future_horizon'] != 8:
        raise ValueError('Raw native action clock/horizon changed')
    payload = manifest['payload_files']
    if object_hash(payload) != manifest['payload_inventory_sha256']:
        raise ValueError('Raw payload inventory hash changed')
    expected = {}
    for item in payload:
        path = child(source, item['path'])
        if item['path'] in expected or path.is_symlink() or not path.is_file():
            raise ValueError('Raw payload has duplicate/nonregular entries')
        if not HEX.fullmatch(item['sha256']) or path.stat().st_size != item['size_bytes'] or sha256(path) != item['sha256']:
            raise ValueError('Raw payload bytes differ from their accepted inventory: ' + item['path'])
        expected[item['path']] = item
    actual = set()
    for path in source.rglob('*'):
        file_relative = path.relative_to(source)
        if any(part == 'REQUEST_STOP' or part.endswith('.tmp') for part in file_relative.parts):
            continue
        if path.is_symlink():
            raise ValueError('Raw ready trees may not contain symlinks')
        if path.is_file():
            actual.add(file_relative.as_posix())
        elif not path.is_dir():
            raise ValueError('Raw ready trees may not contain special filesystem entries')
    if actual != set(expected) | {'raw_manifest.json'}:
        raise ValueError('Raw payload files are incomplete or contain undeclared additions')
    hashes = {name: item['sha256'] for name, item in expected.items()}
    hashes['raw_manifest.json'] = manifest_sha
    for mapping in (manifest['critical_sha256'], ready['source']['files_sha256']):
        if any(hashes.get(name) != digest for name, digest in mapping.items()):
            raise ValueError('Raw critical metadata/source hashes disagree')
    if ready['source']['files_sha256'].get('raw_manifest.json') != manifest_sha:
        raise ValueError('Ready input omitted its raw manifest hash')
    layout = child(source, ready['source']['layout'])
    raw = child(source, ready['source']['raw'])
    if manifest['layout'] != ready['source']['layout'] or manifest['trajectory'] != ready['source']['raw']:
        raise ValueError('Raw/layout path identity changed')
    metadata = read(raw / 'metadata.json')
    if (metadata['episode_id'] != identity or metadata['target_id'] != ready['target_id']
            or metadata['seed'] != ready['layout_seed'] or metadata.get('capture_stride') != 1
            or metadata.get('terminal_success') is not True or read(raw / 'status.json').get('phase') != 'complete'
            or read(child(source, manifest['raw_validation'])).get('passed') is not True
            or read(layout / 'targets.json')['seed'] != ready['layout_seed']):
        raise ValueError('Raw capture/layout completion identity mismatch')
    for name, digest in metadata['source_sha256'].items():
        if sha256(child(layout, name)) != digest:
            raise ValueError('Raw controls were captured against different layout bytes')
    with np.load(raw / 'supervision.npz', allow_pickle=False) as archive:
        controls = archive['vw_si']; stop = archive['stop']; dt = archive['dt_s']
        future = archive['future_vw_si']; valid = archive['horizon_valid']
        n = len(controls)
        if (controls.shape != (n, 2) or stop.shape != (n,) or dt.shape != (n,)
                or future.shape != (n, 8, 2) or valid.shape != (n, 8)
                or not np.isfinite(controls).all() or not np.isfinite(future).all()
                or not np.allclose(dt, .2, atol=1e-8) or np.count_nonzero(stop) != 1 or stop[-1] != 1
                or not np.array_equal(controls[-1], [0, 0])
                or np.max(np.abs(controls[:, 0])) > .400001 or np.max(np.abs(controls[:, 1])) > .800001):
            raise ValueError('Raw native control/STOP contract failed')
        for offset in range(8):
            count = max(0, n-offset)
            if not np.array_equal(valid[:, offset], np.arange(n)+offset < n) or not np.array_equal(future[:count, offset], controls[offset:]):
                raise ValueError('Raw future-eight labels skip or modify native controls')
        frame_count = 0
        for line in (raw / 'frame_records.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            frame = json.loads(line)
            action = frame['executed_action']
            if (frame['step_index'] != frame_count or frame['episode_id'] != identity or frame['target_id'] != ready['target_id']
                    or abs(frame['inputs']['observation']['timestamp_s'] - frame_count*.2) > 1e-8
                    or not np.allclose([action['v_mps'], action['omega_radps']], controls[frame_count], atol=1e-6)
                    or bool(action['stop']) != bool(stop[frame_count])):
                raise ValueError('Raw frame/control/episode identities diverged')
            frame_count += 1
    with np.load(raw / 'teacher_only.npz', allow_pickle=False) as archive:
        poses = archive['executed_positions']
        if poses.shape != (n+1, 3) or not np.isfinite(poses).all():
            raise ValueError('Raw pose/action count changed')
    if (n != metadata['captured_frame_count'] or n != manifest['frames'] or n != manifest['action_count']
            or n+1 != manifest['pose_count'] or frame_count != n):
        raise ValueError('Raw frame/action/pose inventory counts disagree')
    delivery = ready['raw_delivery']
    receipt = delivery['receipt']
    if (delivery.get('passed') is not True or receipt.get('passed') is not True
            or receipt['manifest_sha256'] != delivery['manifest_sha256']
            or receipt['host'] != host or receipt['remote_root'] != remote_root or receipt['unit'] != relative
            or delivery['remote_destination'] != remote_root + '/' + relative):
        raise ValueError('Raw input has no matching remote SHA publication receipt')
    proof_path = Path(delivery['manifest_path']).resolve(strict=True)
    receipt_path = Path(delivery['receipt_path']).resolve(strict=True)
    if (not proof_path.is_relative_to(producer_root) or not receipt_path.is_relative_to(producer_root)
            or sha256(proof_path) != delivery['manifest_sha256'] or read(receipt_path) != receipt):
        raise ValueError('Retained raw delivery proof changed or escaped its producer')
    delivered_files = {row['path']: row for row in read(proof_path)['entries'] if row['kind'] == 'file'}
    if any(name not in delivered_files or delivered_files[name]['sha256'] != digest for name, digest in hashes.items()):
        raise ValueError('Accepted raw bytes were not covered by the remote delivery manifest')
    return dict(source=source, layout=layout, raw=raw, manifest=manifest, frames=n)


class AnnotationQueue:
    def __init__(self, args, *, publisher=publish_unit, verifier=verify_published_unit, summarizer=summarize_unit,
                 runner=None, usage=None):
        self.args = args
        self.config = read(args.config)
        if self.config.get('schema_version') != 'lunar_annotation_queue/16.0':
            raise ValueError('Unknown annotation queue configuration')
        self.producer = resolve(args.producer_root or self.config['producer_root'])
        self.queue = self.producer / 'queue'
        self.output = resolve(args.output or self.config['output'])
        if self.output == self.producer or self.producer.is_relative_to(self.output) or self.output.is_relative_to(self.producer):
            raise ValueError('Consumer output must be separate from the producer tree')
        self.output.mkdir(parents=True, exist_ok=True)
        self.plan, self.plan_sha = stable_json(self.producer / 'plan.json')
        if (self.plan.get('schema_version') != 'lunar_raw_plan/16.0' or tuple(self.plan['targets']) != TARGETS
                or self.plan['requested_units'] != self.config.get('requested_units', 100)
                or len(self.plan['slots'])*4 != self.plan['requested_units']
                or {row['physical_slot'] for row in self.plan['slots']} != set(range(len(self.plan['slots'])))):
            raise ValueError('Producer plan lacks the expected physical slots/targets')
        expected = self.config.get('expected_layout_splits', dict(train=20,val=3,test=2))
        if {split:sum(row['split']==split for row in self.plan['slots']) for split in expected} != expected:
            raise ValueError('Physical split counts differ from the fixed consumer plan')
        self.python = str(resolve(args.python or self.config.get('python', sys.executable)))
        self.ppio_env = resolve(args.ppio_env or self.config['ppio_env'])
        self.cache = resolve(args.provider_cache or self.config['provider_cache'])
        if not self.ppio_env.is_file() or not self.cache.is_dir():
            raise ValueError('Existing provider environment file and shared ledger directory are required')
        self.host = self.config.get('publish_host', 'lab-a100-20')
        self.remote = self.config['remote_root']
        self.unit_template = read(resolve(self.config['unit_config']))
        annotation = self.unit_template['annotation']
        annotation.update(query_mode='compiled', query_count=20, budget_cny=12., model='qwen/qwen3.8-27b')
        if self.config.get('query_bank'):
            annotation['query_bank'] = self.config['query_bank']
        if not annotation.get('query_bank') or not annotation.get('query_protocol'):
            raise ValueError('Compiled query bank and fixed query protocol are required')
        self.unit_template['training_view'] = dict(self.unit_template.get('training_view', {}), enabled=True, stride=5)
        self.unit_template['terrain_build'] = dict(self.unit_template.get('terrain_build', {}), enabled=False)
        self.pipeline = resolve(self.config.get('pipeline_script', 'lunar_dataset_pipeline_v14.py'))
        self.publisher, self.verifier, self.summarizer = publisher, verifier, summarizer
        self.runner = runner or self.run_pipeline
        self.usage = usage or (lambda: PPIOClient(self.cache, budget_cny=12.).usage())
        self.entries = {}
        self.checked = set()
        self.resources = {str(resolve(annotation[key])): sha256(resolve(annotation[key]))
            for key in ('semantic_config', 'query_protocol', 'query_bank')}
        self.codes = code_hashes()
        self.fingerprint = object_hash(dict(config=self.config, effective_unit_config=self.unit_template,
            producer_plan_sha256=self.plan_sha, producer_root=str(self.producer), resources=self.resources,
            code_sha256=self.codes, python=self.python, provider_cache=str(self.cache), ppio_env=str(self.ppio_env)))
        self.revision = self.config.get('consumer_revision', 'v16')
        for name in ('unit_state', 'unit_configs', 'units', 'delivery_state', 'logs', 'catalog'):
            (self.output/name).mkdir(exist_ok=True)
        (self.queue/'consumer_state').mkdir(parents=True, exist_ok=True)
        (self.queue/'consumer_locks').mkdir(parents=True, exist_ok=True)
        binding = self.output/'consumer_binding.json'
        if binding.exists():
            if read(binding)['identity_sha256'] != self.fingerprint:
                raise ValueError('Consumer source/config changed; use a new revision output')
            if not args.resume:
                raise FileExistsError('Existing consumer requires explicit --resume')
        else:
            write_json(binding, dict(identity_sha256=self.fingerprint, consumer_revision=self.revision,
                                    producer_plan_sha256=self.plan_sha, resources=self.resources, code_sha256=self.codes))

    def frozen(self):
        if code_hashes() != self.codes or sha256(self.producer/'plan.json') != self.plan_sha:
            raise ValueError('Frozen consumer code or producer plan changed')
        if any(sha256(Path(path)) != digest for path,digest in self.resources.items()):
            raise ValueError('Frozen annotation resource changed')

    def stopped(self):
        return (self.output/'REQUEST_STOP').exists()

    def state(self, identity, value):
        value = dict(value, unit_id=identity, consumer_revision=self.revision)
        write_json(self.output/'unit_state'/(identity+'.json'), value)
        write_json(self.queue/'consumer_state'/(identity+'.json'), value)

    def progress(self, phase, **extra):
        value = dict(phase=phase, published_units=len(self.entries), requested_units=self.plan['requested_units'],
            consumer_revision=self.revision, gpu_execution=False, automatic_failed_unit_retry=False,
            producer_stop_requested=False, local_frame_reclamation=bool(self.config.get('retention', {}).get('enabled', False)), **extra)
        write_json(self.output/'progress.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def command(self, ready, source, root, config_path):
        return [self.python, str(self.pipeline), '--config', str(config_path), '--output', str(root),
            '--isaac-python', self.python, '--layout-source', str(source['layout']), '--raw-source', str(source['raw']),
            '--provider-cache', str(self.cache), '--ppio-env', str(self.ppio_env)]

    def run_pipeline(self, command, root, record):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                           PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
        libraries = Path(self.python).parent.parent/'lib'
        environment['PYTHONPATH'] = ':'.join([str(ROOT), *map(str,self.config.get('cpu_pythonpaths',())),
                                              environment.get('PYTHONPATH','')])
        environment['LD_LIBRARY_PATH'] = ':'.join([str(libraries), *map(str,self.config.get('cpu_library_paths',())),
                                                   environment.get('LD_LIBRARY_PATH','')])
        cpp = [libraries/name for name in ('libstdc++.so.6','libgcc_s.so.1')]
        if self.config.get('preload_cpp_runtime',True) and all(path.is_file() for path in cpp):
            environment['LD_PRELOAD'] = ':'.join([*map(str,cpp),environment.get('LD_PRELOAD','')]).rstrip(':')
        with (self.output/'logs'/(record['unit_id']+'.log')).open('ab') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            self.state(record['unit_id'], dict(record, phase='pipeline_running', pid=process.pid, command=command))
            while process.poll() is None:
                if self.stopped() and (root/'status.json').exists() and read(root/'status.json').get('phase') != 'complete':
                    (root/'REQUEST_STOP').touch()  # Own consumer unit only, never the producer.
                time.sleep(.5)
        return process.returncode

    def matching_process(self, pid, root):
        try:
            arguments = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        except OSError:
            return False
        return (b'--output' in arguments and any(value.endswith(b'lunar_dataset_pipeline_v14.py') for value in arguments)
            and Path(os.fsdecode(arguments[arguments.index(b'--output')+1])).resolve() == root)

    def handle(self, ready_path):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', ready_path.stem):
            raise ValueError('Unsafe ready filename')
        # Shared across consumer revisions/output paths: duplicate consumers
        # cannot both start paid work for the same immutable input.
        with (self.queue/'consumer_locks'/(ready_path.stem+'.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return self._handle_locked(ready_path)

    def _handle_locked(self, ready_path):
        ready, ready_sha = stable_json(ready_path)
        identity = ready['unit_id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', identity) or identity != ready_path.stem:
            raise ValueError('Unsafe or mismatched ready filename')
        bindings = raw_binding(ready)
        shared_state = self.queue/'consumer_state'/(identity+'.json')
        old = read(shared_state) if shared_state.exists() else {}
        if old.get('phase') == 'needs_review':
            return False  # Operator must explicitly version/requeue; never retry HTTP.
        record = dict(bindings, ready_sha256=ready_sha, consumer_identity_sha256=self.fingerprint)
        root = self.output/'units'/identity
        try:
            if old and (any(old.get(key) != value for key,value in bindings.items()) or old.get('ready_sha256') != ready_sha):
                raise ValueError('Ready source changed after consumer admission')
            if old.get('phase') == 'final_published':
                if identity in self.checked:
                    return True
                retained_state = Path(old['final_delivery']['state_dir'])
                delivery = self.verifier(state_dir=retained_state,
                                         ssh_options=self.config.get('ssh_options', ()))
                if delivery['manifest_sha256'] != old['final_delivery']['manifest_sha256']:
                    raise ValueError('Final remote SHA differs from retained consumer state')
                self.entries[identity] = old['entry']
                if Path(old.get('output_unit_root', str(root))).resolve() != root:
                    self.entries[identity] = dict(old['entry'], inherited_publication=True)
                self.state(identity, dict(old, phase='final_published', entry=self.entries[identity]))
                self.checked.add(identity)
                return True  # No local raw/derived frame reread after reclamation.
            if old and old.get('consumer_identity_sha256') != self.fingerprint:
                raise ValueError('Existing attempt belongs to another frozen consumer revision')
            self.frozen()
            source = validate_ready(ready, producer_root=self.producer, plan=self.plan, plan_sha256=self.plan_sha,
                                    host=self.host, remote_root=self.remote)
            registry_path = self.output/'physical_slots.json'
            registry = read(registry_path) if registry_path.exists() else {}
            slot_key = str(ready['physical_slot'])
            accepted_slot = dict(layout_seed=ready['layout_seed'], split=ready['split'],
                scene_sha256=read(source['raw']/'metadata.json')['source_sha256'])
            if slot_key in registry and registry[slot_key] != accepted_slot:
                raise ValueError('One physical slot changed accepted seed/split/scene between its four targets')
            registry[slot_key] = accepted_slot
            write_json(registry_path, registry)
            config = deepcopy(self.unit_template)
            config['layouts'] = [dict(seed=ready['layout_seed'], split=ready['split'], targets=[ready['target_id']])]
            config['targets'] = [ready['target_id']]
            config_path = self.output/'unit_configs'/(identity+'.json')
            if config_path.exists() and read(config_path) != config:
                raise ValueError('Existing per-unit annotation configuration changed')
            write_json(config_path, config)
            record.update(unit_config_sha256=sha256(config_path), output_unit_root=str(root))
            if old.get('phase') == 'pipeline_running':
                while self.matching_process(old.get('pid'), root):
                    if self.stopped():
                        (root/'REQUEST_STOP').touch()
                    time.sleep(1)
                if not (root/'status.json').exists() or read(root/'status.json').get('phase') != 'complete':
                    raise RuntimeError('Prior annotation attempt is incomplete; automatic retry forbidden')
            elif old.get('phase') not in ('pipeline_complete', 'publishing'):
                if root.exists():
                    raise RuntimeError('Uncached or partial annotation unit preserved without retry')
                if self.usage().get('committed_cny', 0.) >= 12.:
                    raise RuntimeError('Shared twelve-CNY budget exhausted')
                self.state(identity, dict(record, phase='pipeline_running'))
                if self.runner(self.command(ready, source, root, config_path), root, record):
                    raise RuntimeError('Annotation pipeline failed; paid responses preserved without retry')
            if not (root/'status.json').exists() or read(root/'status.json').get('phase') != 'complete' or not read(root/'status.json').get('passed'):
                raise ValueError('Annotation unit did not finish every validation stage')
            self.frozen()
            run_config = read(root/'dataset/run_config.json')
            if run_config['config'] != config or run_config['code_sha256'] != object_hash(self.codes):
                raise ValueError('Completed annotation unit has different source/config provenance')
            validate_ready(ready, producer_root=self.producer, plan=self.plan, plan_sha256=self.plan_sha,
                           host=self.host, remote_root=self.remote)
            summary = self.summarizer(root, expected_seed=ready['layout_seed'], expected_target=ready['target_id'], expected_split=ready['split'])
            if len(summary['training_family_ids']) != 16 or len(summary['heldout_family_ids']) != 4:
                raise ValueError('Final unit lacks the exact twenty-family sixteen/four query contract')
            self.state(identity, dict(record, phase='pipeline_complete', summary=summary))
            self.state(identity, dict(record, phase='publishing', summary=summary))
            publication_options = {}
            if self.config.get('enable_raw_preseed', False):
                publication_options['preseed_units'] = [dict(unit='raw_inputs/'+identity,
                    manifest_sha256=ready['raw_delivery']['manifest_sha256'])]
            delivered = self.publisher(root, host=self.host, remote_root=self.remote, unit='units/'+identity,
                state_dir=self.output/'delivery_state'/identity, required_paths=summary['required_paths'],
                ssh_options=self.config.get('ssh_options', ()), **publication_options)
            entry = dict(summary, unit_id=identity, ordinal=ready['ordinal'], relative_unit_root='units/'+identity,
                         physical_slot=ready['physical_slot'], delivery=delivered, local_frames_evicted=False,
                         raw_input_binding=bindings, raw_delivery=ready['raw_delivery'])
            self.state(identity, dict(record, phase='final_published', final_delivery=delivered, entry=entry))
            self.entries[identity] = entry
            self.checked.add(identity)
            self.catalog(complete=False)
            self.progress('unit_final_published', unit_id=identity)
            self.reclaim_if_needed()
            return True
        except Exception as exc:
            self.state(identity, dict(record, phase='needs_review', error_type=type(exc).__name__, error=str(exc),
                automatic_retry=False, raw_preserved=True, paid_responses_preserved=True, output_unit_root=str(root)))
            self.progress('unit_needs_review', unit_id=identity, error=str(exc))
            return False

    def catalog(self, *, complete):
        return build_catalog(self.output/'catalog/catalog.json', list(self.entries.values()),
                             requested_units=self.plan['requested_units'], complete=complete)

    def reclaim_if_needed(self):
        settings = self.config.get('retention', {})
        if not settings.get('enabled', False):
            return
        if shutil.disk_usage(self.output).free >= float(settings.get('below_free_gib', 6.))*1024**3:
            return
        for entry in sorted(self.entries.values(), key=lambda value:value['ordinal']):
            if entry.get('inherited_publication') or entry.get('local_frames_evicted') or entry.get('retention_phase') == 'needs_retention_review':
                continue
            identity = entry['unit_id']
            state_path = self.output/'unit_state'/(identity+'.json')
            old = read(state_path)
            final_state = Path(entry['delivery']['state_dir'])
            raw = entry['raw_delivery']
            try:
                retention = reclaim_owned_frames(self.output/entry['relative_unit_root'],
                    owned_units_root=self.output/'units', unit_id=identity, ordinal=entry['ordinal'],
                    final_manifest_path=final_state/'manifest.json', final_receipt_path=final_state/'receipt.json',
                    expected_final_sha256=entry['delivery']['manifest_sha256'],
                    raw_manifest_path=Path(raw['manifest_path']), raw_receipt_path=Path(raw['receipt_path']),
                    expected_raw_sha256=raw['manifest_sha256'], critical_sha256=entry['critical_sha256'],
                    keep_review_units=int(settings.get('keep_review_units', 4)))
                entry['local_frames_evicted'] = retention['local_frames_evicted']
                entry['retention_phase'] = retention['phase']
            except Exception as exc:
                retention = dict(phase='needs_retention_review', error=str(exc), publication_remains_valid=True,
                                 no_unverified_frames_removed=True)
                entry['retention_phase'] = 'needs_retention_review'
            # Publication remains final even when local reclamation needs review.
            self.state(identity, dict(old, phase='final_published', entry=entry, retention=retention,
                                     local_frames_evicted=entry['local_frames_evicted']))
            self.progress('consumer_retention_'+retention['phase'], unit_id=identity)
            if shutil.disk_usage(self.output).free >= float(settings.get('target_free_gib', 8.))*1024**3:
                break

    def cycle(self, max_new=None):
        self.frozen()
        new = 0
        for path in sorted((self.queue/'ready').glob('*.json')):
            if self.stopped() or (max_new is not None and new >= max_new):
                break
            was_known = path.stem in self.checked
            phase = None
            try:
                shared = self.queue/'consumer_state'/(path.stem+'.json')
                phase = read(shared).get('phase') if shared.exists() else None
                if phase == 'needs_review':
                    continue
                self.handle(path)
            except Exception as exc:
                if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', path.stem):
                    self.state(path.stem, dict(phase='needs_review', ready_sha256=sha256(path),
                        error=str(exc), automatic_retry=False, raw_preserved=True, paid_responses_preserved=True))
                self.progress('malformed_ready_needs_review', ready_file=path.name, error=str(exc))
            new += not was_known and phase != 'final_published'
        self.reclaim_if_needed()
        return new

    def run(self):
        while not self.stopped():
            self.cycle(self.args.max_new_units)
            if len(self.entries) == self.plan['requested_units']:
                catalog = self.catalog(complete=True)
                usage_path = self.output/'catalog/provider_usage.json'
                if not usage_path.exists():
                    write_json(usage_path, dict(budget_cny=12., shared_provider_cache=str(self.cache), **self.usage()))
                write_json(self.output/'catalog/producer_plan.json', self.plan)
                delivery = self.publisher(self.output/'catalog', host=self.host, remote_root=self.remote,
                    unit=self.config.get('catalog_unit', 'catalog_v16'), state_dir=self.output/'delivery_state/catalog_v16',
                    required_paths=['catalog.json','provider_usage.json','producer_plan.json'],
                    ssh_options=self.config.get('ssh_options', ()))
                self.progress('complete', counts=catalog['counts'], final_catalog_delivery=delivery)
                return 0
            self.progress('waiting_for_independent_complete_raw_inputs')
            if self.args.once or self.args.max_new_units is not None:
                return 0
            time.sleep(float(self.config.get('poll_seconds', 5.)))
        self.progress('consumer_stopped_without_stopping_producer')
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/lunar_annotation_queue_v16.json')
    parser.add_argument('--producer-root'); parser.add_argument('--output'); parser.add_argument('--python')
    parser.add_argument('--ppio-env'); parser.add_argument('--provider-cache')
    parser.add_argument('--run', action='store_true'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--once', action='store_true'); parser.add_argument('--max-new-units', type=int)
    args = parser.parse_args(argv)
    if args.max_new_units is not None and args.max_new_units < 1:
        parser.error('--max-new-units must be positive')
    queue = AnnotationQueue(args)
    with (queue.output/'consumer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.run:
            queue.progress('prepared_without_api_gpu_or_ssh')
            return 0
        with (queue.queue/'annotation_consumer.lock').open('a') as shared_lock:
            fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return queue.run()


if __name__ == '__main__':
    raise SystemExit(main())
