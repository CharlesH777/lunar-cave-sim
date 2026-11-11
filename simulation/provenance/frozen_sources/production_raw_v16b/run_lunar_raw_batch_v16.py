"""Raw-only physical producer. No provider, language, CoT or stub stage.

Ready events follow complete dense validation and remote SHA publication.
An independently failing language consumer can fill the bounded pending
buffer, but never causes a valid physical capture to be rerun or discarded.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from cave.lunar_raw_physical_v16 import (
    ROOT,TARGETS,GIB,PhysicalBatch,read,write_json,sha256,object_hash,resolve,raw_summary,preflight_layout)
from cave.lunar_dataset_delivery_v14 import publish_unit,verify_published_unit
from stage_lunar_assets_v14_cpu import adopt

FRAME=re.compile(r'^\d{6}_(?:rgb224\.png|rgb256\.png|sensors\.npz|instances\.json)$')


def physical_code_files():
    """Freeze only this physical dependency closure, not evolving query code."""
    pending=[Path(__file__).resolve(),ROOT/'validate_lunar_raw_v16_cpu.py',ROOT/'build_lunar_search_demo_v8_cpu.py']
    result={}
    while pending:
        path=pending.pop()
        if path in result:continue
        result[path]=sha256(path)
        module=path.relative_to(ROOT).with_suffix('').parts
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[item.name for item in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=list(module[:-node.level]) if node.level else []
                name='.'.join([*prefix,*((node.module or '').split('.') if node.module else [])])
                names=[name,*[name+'.'+alias.name for alias in node.names]]
            for name in names:
                candidate=ROOT/Path(*name.split('.')).with_suffix('.py')
                if candidate.is_file() and candidate not in result:pending.append(candidate)
    return {str(path.relative_to(ROOT)):digest for path,digest in sorted(result.items())}


def payload_inventory(root):
    files=[]
    for path in sorted(Path(root).rglob('*')):
        relative=path.relative_to(root)
        if any(part=='REQUEST_STOP' or part.endswith('.tmp') for part in relative.parts):continue
        if path.is_symlink():raise ValueError('Raw publication requires self-contained regular files')
        if path.is_file() and relative.as_posix()!='raw_manifest.json':
            before=path.stat();digest=sha256(path);after=path.stat()
            if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                raise ValueError('Raw payload changed during inventory')
            files.append(dict(path=relative.as_posix(),size_bytes=after.st_size,sha256=digest))
    return files


class RawProducer(PhysicalBatch):
    def __init__(self,args):
        self.args=args;self.config=read(args.config);self.config_path=args.config.resolve(strict=True)
        if self.config.get('schema_version')!='lunar_raw_batch/16.0':raise ValueError('Unknown raw-only configuration')
        self.output=args.output.expanduser().resolve();self.slots=self.config['layouts']
        self.capture_config=self.config['capture'];runtime=args.isaac_python or self.config.get('isaac_python') or os.environ.get('ISAAC_PYTHON')
        if not runtime:raise ValueError('Set --isaac-python or ISAAC_PYTHON')
        self.runtime=str(resolve(runtime))
        self.template=resolve(self.config['terrain_template']);self.host=self.config.get('publish_host','lab-a100-20')
        self.remote=self.config['remote_root'];self.prebuilt={int(seed):resolve(path) for seed,path in self.config.get('prebuilt_layouts',{}).items()}
        self.max_attempts=int(self.config.get('maximum_cpu_layout_attempts',5));self.reserved={row['seed'] for row in self.slots}
        self.pending_limit=int(self.config.get('maximum_unannotated_units',8))
        if not 1<=self.max_attempts<=5 or self.pending_limit<1:raise ValueError('Invalid bounded CPU/pending limits')
        if len(self.reserved)!=len(self.slots) or len(self.slots)*4!=self.config['requested_units']:raise ValueError('Duplicate seeds or non-four-target plan')
        expected=self.config.get('expected_layout_splits',dict(train=20,val=3,test=2))
        if set(expected)!=set(('train','val','test')) or {s:sum(row['split']==s for row in self.slots) for s in expected}!=expected:
            raise ValueError('Physical split counts differ from the raw plan')
        self.lock=threading.RLock();self.admission=threading.Lock();self.active={};self.reservations=set()
        self.preflight_script=Path(__file__).resolve();self.queue=self.output/'queue'
        for directory in ('layouts','layout_state','raw_state','raw_inputs','delivery_state','consumer_receipts','retention','logs','queue/ready','queue/consumer_state'):
            (self.output/directory).mkdir(parents=True,exist_ok=True)

    def stopped(self):return (self.output/'REQUEST_STOP').exists()

    def status(self,phase,**extra):
        with self.lock:
            write_json(self.output/'progress.json',dict(schema_version='lunar_raw_progress/16.0',phase=phase,
                requested_raw_units=self.config['requested_units'],ready_raw_units=len(list((self.queue/'ready').glob('*.json'))),
                active_units=dict(self.active),pending_limit=self.pending_limit,training_ready=False,annotation_started=False,
                updated_unix_s=time.time(),**extra))
            print(phase,extra,flush=True)

    def prepare(self):
        if not self.template.is_dir() or (self.template/'targets.json').exists():raise ValueError('Missing pre-placement fixed-geometry template')
        self.code_hashes=physical_code_files()
        if any('query_' in path or 'ppio' in path or 'annotate_' in path for path in self.code_hashes):
            raise ValueError('Language/provider dependency leaked into the physical producer')
        self.terrain_hashes={name:sha256(self.template/name) for name in ('cave.usdc','mesh.npz','scene_config.json','boulders.json','geometry_report.json')}
        self.prebuilt_hashes={str(seed):{name:sha256(path/name) for name in ('cave.usdc','mesh.npz','scene_config.json','targets.json','task_obstacles.json')}
            for seed,path in self.prebuilt.items()}
        adoptions={}
        for identity,source in self.config.get('adopt_raw',{}).items():
            metadata=read(resolve(source)/'metadata.json');seed=int(identity.split('_')[1]);target=identity.split('_',2)[2]
            if str(seed) not in self.prebuilt_hashes:raise ValueError('An adopted raw requires its exact prebuilt layout')
            summary=raw_summary(resolve(source),self.prebuilt[seed],seed=seed,target=target)
            adoptions[identity]=dict(source=str(resolve(source)),frames=summary['frames'],metadata_sha256=sha256(resolve(source)/'metadata.json'))
        self.config_sha=sha256(self.config_path)
        identity=dict(config=self.config,config_sha256=self.config_sha,physical_code_sha256=self.code_hashes,terrain_sha256=self.terrain_hashes,
            prebuilt_sha256=self.prebuilt_hashes,adoption_metadata=adoptions)
        state=self.output/'run_config.json'
        if state.exists():
            if read(state)['identity_sha256']!=object_hash(identity):raise ValueError('Physical producer source/config changed')
            if not self.args.resume:raise FileExistsError('Existing raw producer requires --resume')
        else:write_json(state,dict(identity_sha256=object_hash(identity),**identity))
        plan=dict(schema_version='lunar_raw_plan/16.0',config_sha256=sha256(self.config_path),
            targets=list(TARGETS),slots=[dict(physical_slot=i,requested_seed=row['seed'],split=row['split']) for i,row in enumerate(self.slots)],
            requested_units=self.config['requested_units'],expected_layout_splits=self.config['expected_layout_splits'])
        write_json(self.output/'plan.json',plan);self.plan_sha=sha256(self.output/'plan.json')
        write_json(self.output/'adoption_inventory.json',dict(complete_sources=adoptions,
            excluded_partial_sources=self.config.get('excluded_partial_sources',[]),new_captures_started=0))
        self.status('prepared_raw_only_without_capture')

    def frozen(self):
        if sha256(self.config_path)!=self.config_sha:raise ValueError('Frozen raw configuration changed')
        if any(sha256(ROOT/path)!=digest for path,digest in self.code_hashes.items()):raise ValueError('Frozen physical code changed')
        if any(sha256(self.template/path)!=digest for path,digest in self.terrain_hashes.items()):raise ValueError('Fixed cave geometry changed')
        for seed,files in self.prebuilt_hashes.items():
            if any(sha256(self.prebuilt[int(seed)]/name)!=digest for name,digest in files.items()):raise ValueError('Prebuilt source layout changed')

    def consumer_done(self,ready):
        """Only an exact raw-bound final annotation publication releases raw."""
        cache=self.output/'consumer_receipts'/(ready['unit_id']+'.json')
        path=self.queue/'consumer_state'/(ready['unit_id']+'.json')
        if not path.is_file() and not cache.is_file():return False
        try:
            value=read(path if path.is_file() else cache);delivery=value['final_delivery'];receipt=delivery['receipt']
            valid=(value.get('phase')=='final_published' and value.get('unit_id')==ready['unit_id']
                and value.get('raw_manifest_sha256')==ready['raw_manifest_sha256']
                and value.get('raw_delivery_manifest_sha256')==ready['raw_delivery']['manifest_sha256']
                and delivery.get('passed') is True and receipt.get('passed') is True
                and delivery.get('manifest_sha256')==receipt.get('manifest_sha256')
                and receipt.get('unit')=='units/'+ready['unit_id'] and receipt.get('host')==self.host
                and receipt.get('remote_root')==self.remote)
            if valid:
                with self.lock:write_json(cache,value)
            return valid
        except (KeyError,ValueError,OSError,TypeError):return False

    def pending_count(self):
        pending=set(self.reservations)
        for path in (self.queue/'ready').glob('*.json'):
            ready=read(path)
            if not self.consumer_done(ready):pending.add(ready['unit_id'])
        return len(pending)

    def reserve(self,identity):
        while True:
            if self.stopped():raise InterruptedError('REQUEST_STOP while waiting for raw pending capacity')
            with self.lock:
                count=self.pending_count()
                if count<self.pending_limit:self.reservations.add(identity);return
            self.evict_if_needed()
            self.status('waiting_for_annotation_consumer',pending_raw_units=count)
            time.sleep(3)

    def evict_if_needed(self):
        if not self.config.get('prune_consumed_raw',True):return
        import shutil
        if shutil.disk_usage(self.output).free>=float(self.config.get('evict_below_free_gib',6.))*GIB:return
        with self.lock:
            for ready_path in sorted((self.queue/'ready').glob('*.json')):
                ready=read(ready_path);identity=ready['unit_id'];record=self.output/'retention'/(identity+'.json')
                if ready['ordinal']<int(self.config.get('keep_review_units',4)) or record.exists() or not self.consumer_done(ready):continue
                if ready['producer_relative_root']!='raw_inputs/'+identity or Path(identity).name!=identity:raise ValueError('Raw retention path is outside its owned unit')
                root=(self.output/ready['producer_relative_root']).resolve(strict=True)
                if not root.is_relative_to((self.output/'raw_inputs').resolve(strict=True)):raise ValueError('Raw retention unit escaped the producer root')
                transaction=self.output/'delivery_state'/identity
                receipt=read(transaction/'receipt.json');published=read(transaction/'manifest.json')
                if (sha256(transaction/'manifest.json')!=ready['raw_delivery']['manifest_sha256']
                        or receipt.get('passed') is not True or receipt.get('manifest_sha256')!=ready['raw_delivery']['manifest_sha256']
                        or receipt.get('unit')!='raw_inputs/'+identity or receipt.get('host')!=self.host or receipt.get('remote_root')!=self.remote
                        or not any(item.get('path')=='raw_manifest.json' and item.get('sha256')==ready['raw_manifest_sha256'] for item in published['entries'])):
                    raise ValueError('Raw publication receipt/manifest no longer binds the retained raw')
                manifest=read(root/'raw_manifest.json')
                if sha256(root/'raw_manifest.json')!=ready['raw_manifest_sha256']:raise ValueError('Raw manifest changed before retention')
                if any(sha256(root/path)!=digest for path,digest in ready['source']['files_sha256'].items()):raise ValueError('Raw critical metadata changed before retention')
                proposed=[]
                for item in manifest['payload_files']:
                    relative=Path(item['path']);path=root/relative
                    if relative.parts[:2]!=('trajectory','frames') or not FRAME.fullmatch(relative.name):continue
                    if not path.resolve().is_relative_to(root):raise ValueError('Raw frame path escaped its owned unit')
                    if path.is_symlink():raise ValueError('Raw frame became a symlink')
                    if not path.exists():continue
                    stat=path.stat();version=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
                    if stat.st_size!=item['size_bytes'] or sha256(path)!=item['sha256']:raise ValueError('Changed/unuploaded raw frame retained')
                    after=path.stat()
                    if version!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('Raw frame changed during retention preflight')
                    proposed.append((path,version))
                for path,version in proposed:
                    stat=path.stat()
                    if version!=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns):raise ValueError('Raw frame changed before retention')
                for path,_ in proposed:path.unlink()
                write_json(record,dict(unit_id=identity,raw_manifest_sha256=ready['raw_manifest_sha256'],
                    final_annotation_receipt_bound=True,removed_owned_frame_files=len(proposed),metadata_and_raw_validation_preserved=True))
                if shutil.disk_usage(self.output).free>=float(self.config.get('evict_target_free_gib',8.))*GIB:break

    def command(self,command,log,*,cpu=False,unit_root=None,on_spawn=None,admit_unit_id=None):
        if self.stopped():raise InterruptedError('REQUEST_STOP before physical stage')
        with Path(log).open('ab') as stream:
            if admit_unit_id is not None:
                with self.admission:
                    self.capacity(admit_unit_id)
                    if self.stopped():raise InterruptedError('REQUEST_STOP before raw GPU launch')
                    process=subprocess.Popen(command,cwd=ROOT,env=self.environment(cpu=cpu),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                    if on_spawn:on_spawn(process.pid)
            else:
                process=subprocess.Popen(command,cwd=ROOT,env=self.environment(cpu=cpu),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                if on_spawn:on_spawn(process.pid)
            while process.poll() is None:
                if unit_root and self.stopped() and (Path(unit_root)/'status.json').is_file():
                    if read(Path(unit_root)/'status.json').get('phase')!='complete':(Path(unit_root)/'REQUEST_STOP').touch()
                time.sleep(1)
        return process.returncode

    def existing_capture(self,trajectory):
        processes=[]
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():continue
            try:arguments=(process/'cmdline').read_bytes().split(b'\0')
            except OSError:continue
            if b'--output' in arguments and any(value.endswith(b'collect_lunar_trajectory_v14_isaac.py') for value in arguments):
                if Path(os.fsdecode(arguments[arguments.index(b'--output')+1])).resolve()==trajectory:processes.append(process)
        if len(processes)>1:raise RuntimeError('Multiple captures claim one raw trajectory')
        if processes:
            while (processes[0]/'cmdline').exists() and (processes[0]/'cmdline').read_bytes():
                if self.stopped() and (trajectory/'status.json').is_file() and read(trajectory/'status.json').get('phase')!='complete':(trajectory/'REQUEST_STOP').touch()
                time.sleep(2)

    def ready_event(self,root,manifest,delivery):
        critical=dict(manifest['critical_sha256']);critical['raw_manifest.json']=sha256(root/'raw_manifest.json')
        value={key:manifest[key] for key in ('unit_id','layout_seed','target_id','split','physical_slot','ordinal')}
        value.update(schema_version='lunar_raw_ready/16.0',plan_sha256=self.plan_sha,
            raw_manifest_sha256=critical['raw_manifest.json'],producer_relative_root='raw_inputs/'+manifest['unit_id'],
            source=dict(root=str(root),layout='layout',raw='trajectory',files_sha256=critical),
            raw_delivery=dict(delivery,receipt_path=str(Path(delivery['state_dir'])/'receipt.json'),
                manifest_path=str(Path(delivery['state_dir'])/'manifest.json')))
        write_json(self.queue/'ready'/(manifest['unit_id']+'.json'),value)
        return value

    def run_unit(self,slot,layout,target):
        seed=layout['seed'];identity=f'seed_{seed}_{target}';ordinal=slot*4+TARGETS.index(target)
        root=self.output/'raw_inputs'/identity;state=self.output/'raw_state'/(identity+'.json')
        ready_path=self.queue/'ready'/(identity+'.json');trajectory=root/'trajectory';adopted=self.config.get('adopt_raw',{}).get(identity)
        try:
            self.frozen()
            if ready_path.exists():
                ready=read(ready_path)
                if (ready['plan_sha256']!=self.plan_sha or ready['physical_slot']!=slot or ready['ordinal']!=ordinal
                        or ready['layout_seed']!=seed or ready['split']!=layout['split']):raise ValueError('Raw ready event differs from its frozen physical plan')
                result=verify_published_unit(state_dir=self.output/'delivery_state'/identity,ssh_options=self.config.get('ssh_options',[]))
                if result['manifest_sha256']!=ready['raw_delivery']['manifest_sha256']:raise ValueError('Published raw input changed')
                return ready
            self.reserve(identity)
            with self.lock:self.active[identity]=dict(root=str(root),adopted=bool(adopted))
            root.mkdir(parents=True,exist_ok=True)
            if not (root/'layout').exists():adopt(layout['layout'],root/'layout',kind='layout',expected_seed=seed)
            if not trajectory.exists():
                if adopted:
                    self.status('adopting_complete_raw',unit_id=identity)
                    raw_summary(resolve(adopted),root/'layout',seed=seed,target=target)
                    adopt(resolve(adopted),trajectory,kind='raw',expected_episode=identity,expected_target=target,layout=root/'layout')
                else:
                    command=[self.runtime,str(ROOT/'collect_lunar_trajectory_v14_isaac.py'),'--motion-controller','smooth_v13',
                        '--source',str(root/'layout'),'--output',str(trajectory),'--episode-id',identity,'--target',target,'--seed',str(seed),
                        '--capture-stride','1','--max-steps',str(self.capture_config.get('max_steps',5000))]
                    if self.capture_config.get('experience'):command+=['--experience',str(resolve(self.capture_config['experience']))]
                    for folder in self.capture_config.get('ext_folders',[]):command+=['--ext-folder',str(resolve(folder))]
                    setting=self.capture_config.get('landmark_pass',{})
                    if setting.get('enabled',True):
                        command+=['--via-landmark',setting.get('landmark_id','auto'),'--landmark-pass-policy',setting.get('policy','prefer')]
                        if setting.get('settings'):
                            path=self.output/'raw_state'/(identity+'.settings.json');write_json(path,dict(settings=setting['settings']));command+=['--landmark-pass-config',str(path)]
                    write_json(state,dict(phase='capture_starting',automatic_gpu_retry=False))
                    def started(pid):
                        with self.lock:self.active[identity]['pid']=pid
                        write_json(state,dict(phase='capturing',pid=pid,automatic_gpu_retry=False))
                    code=self.command(command,self.output/'logs'/(identity+'.capture.log'),unit_root=trajectory,on_spawn=started,admit_unit_id=identity)
                    if code:raise RuntimeError('Raw GPU capture failed; no automatic retry: '+identity)
            self.existing_capture(trajectory)
            summary=raw_summary(trajectory,root/'layout',seed=seed,target=target)
            if not (root/'raw_manifest.json').exists():
                validation=root/'raw_validation.json'
                code=self.command([sys.executable,str(ROOT/'validate_lunar_raw_v16_cpu.py'),'--episode',str(trajectory),
                    '--layout',str(root/'layout'),'--output',str(validation)],self.output/'logs'/(identity+'.validation.log'),cpu=True)
                if code or read(validation).get('passed') is not True:raise ValueError('Dense raw observation/control validation failed')
                binding=dict(schema_version='lunar_raw_source_binding/16.0',unit_id=identity,layout_seed=seed,target_id=target,
                    source_sha256=summary['source_sha256'],adopted=bool(adopted),original_source=str(resolve(adopted)) if adopted else None,
                    producer_and_validation_code_sha256=self.code_hashes,physical_config=self.capture_config,raw_validation_sha256=sha256(validation),
                    original_source_not_modified=True,annotation_started=False)
                write_json(root/'source_binding.json',binding)
                critical={str(path.relative_to(root)):sha256(path) for path in
                    [trajectory/name for name in ('metadata.json','status.json','supervision.npz','teacher_only.npz','frame_records.jsonl','planning_teacher_only.json')]
                    +[root/'raw_validation.json',root/'source_binding.json']+[root/'layout'/name for name in summary['source_sha256']]}
                inventory=payload_inventory(root)
                manifest=dict(schema_version='lunar_raw_unit/16.0',complete=True,training_ready=False,annotation_started=False,
                    annotation_domain='isaac_continuous',unit_id=identity,layout_seed=seed,target_id=target,split=layout['split'],
                    physical_slot=slot,ordinal=ordinal,layout='layout',trajectory='trajectory',raw_validation='raw_validation.json',source_binding='source_binding.json',
                    adoption=dict(adopted=bool(adopted),original_source=str(resolve(adopted)) if adopted else None),
                    **summary,critical_sha256=critical,payload_files=inventory,payload_inventory_sha256=object_hash(inventory))
                write_json(root/'raw_manifest.json',manifest)
            manifest=read(root/'raw_manifest.json')
            self.status('publishing_raw_input',unit_id=identity)
            delivery=publish_unit(root,host=self.host,remote_root=self.remote,unit='raw_inputs/'+identity,
                state_dir=self.output/'delivery_state'/identity,required_paths=['raw_manifest.json','raw_validation.json','source_binding.json',*manifest['critical_sha256']],
                ssh_options=self.config.get('ssh_options',[]))
            ready=self.ready_event(root,manifest,delivery);write_json(state,dict(phase='raw_ready',ready_sha256=sha256(ready_path),delivery=delivery))
            self.status('raw_ready',unit_id=identity,adopted=bool(adopted),frames=summary['frames'])
            return ready
        except BaseException as exc:
            (self.output/'REQUEST_STOP').touch();write_json(state,dict(phase='failed_preserved',error=str(exc),automatic_gpu_retry=False))
            self.status('raw_failed_preserved',unit_id=identity,error=str(exc));raise
        finally:
            with self.lock:self.active.pop(identity,None);self.reservations.discard(identity)

    def run(self):
        futures=[]
        try:
            if self.command([self.runtime,'-c','from pxr import Usd; import numpy,scipy'],self.output/'logs/cpu_dependencies.log',cpu=True):raise RuntimeError('CPU SDK unavailable')
            with ThreadPoolExecutor(max_workers=self.args.capture_lanes) as pool:
                try:
                    adopted_slots=[i for i,row in enumerate(self.slots) if any(f"seed_{row['seed']}_{target}" in self.config.get('adopt_raw',{}) for target in TARGETS)]
                    deferred=[]
                    for slot in adopted_slots:
                        self.frozen();layout=self.prepare_layout(slot,self.slots[slot])
                        for target in TARGETS:
                            if f"seed_{layout['seed']}_{target}" in self.config.get('adopt_raw',{}):futures.append(pool.submit(self.run_unit,slot,layout,target))
                            else:deferred.append((slot,layout,target))
                    for task in deferred:futures.append(pool.submit(self.run_unit,*task))
                    for slot,row in enumerate(self.slots):
                        if slot in adopted_slots:continue
                        for future in futures:
                            if future.done():future.result()
                        if self.stopped():raise InterruptedError('REQUEST_STOP before raw layout')
                        self.frozen();layout=self.prepare_layout(slot,row)
                        for target in TARGETS:futures.append(pool.submit(self.run_unit,slot,layout,target))
                    for future in as_completed(futures):future.result()
                except BaseException:
                    (self.output/'REQUEST_STOP').touch()
                    for future in futures:future.cancel()
                    raise
            self.status('raw_collection_complete')
        except BaseException as exc:
            (self.output/'REQUEST_STOP').touch();self.status('raw_stopped_preserved',error=str(exc));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/lunar_raw_batch_v16.json')
    parser.add_argument('--output',type=Path);parser.add_argument('--isaac-python');parser.add_argument('--run',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--capture-lanes',type=int,choices=(1,2),default=2)
    parser.add_argument('--preflight-layout',type=Path,help=argparse.SUPPRESS);parser.add_argument('--preflight-config',type=Path,help=argparse.SUPPRESS);parser.add_argument('--preflight-report',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.preflight_layout:return preflight_layout(args.preflight_layout,args.preflight_config,args.preflight_report)
    if args.output is None:parser.error('--output is required')
    producer=RawProducer(args)
    with (producer.output/'producer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);producer.prepare()
        if args.run:producer.run()
    return 0


if __name__=='__main__':raise SystemExit(main())
