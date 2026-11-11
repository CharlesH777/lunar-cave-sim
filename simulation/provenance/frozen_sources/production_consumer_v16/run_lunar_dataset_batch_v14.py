"""Bounded layout planning, two-lane unit generation and SHA publication.

The default prepares a plan only. --run starts the authorized batch. Failed
GPU/API units are preserved and never retried. CPU layout rejection may choose
at most five deterministic replacement seeds per layout slot.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from cave.ppio_client_v11 import object_hash, sha256, write_json, PPIOClient
from cave.lunar_batch_catalog_v14 import summarize_unit,build_catalog
from cave.lunar_dataset_delivery_v14 import publish_unit,verify_published_unit

ROOT=Path(__file__).resolve().parent
TARGETS=('astronaut','lunar_rover','lunar_base','chinese_flag')
GIB=1024**3
FRAME=re.compile(r'^\d{6}_(?:rgb224\.png|rgb256\.png|sensors\.npz|instances\.json)$')


def read(path):return json.loads(Path(path).read_text())


def resolve(value):
    path=Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def cpu_preflight(layout,unit_config,report_path):
    """Use exactly the collector's teacher policy; no renderer or paid API."""
    from cave.lunar_trajectory_v11 import TerrainWorld
    from collect_lunar_trajectory_v14_isaac import landmark_plan
    from cave.lunar_smooth_teacher_v13 import precompute_smooth_teacher
    config=read(unit_config);world=TerrainWorld(layout,runtime_collision_mode='actual_geometry')
    setting=config.get('landmark_pass',{})
    setting_path=Path(report_path).with_suffix('.settings.json')
    if setting.get('settings'):write_json(setting_path,dict(settings=setting['settings']))
    reports={}
    try:
        for target in TARGETS:
            if (Path(report_path).parent.parent/'REQUEST_STOP').exists():raise InterruptedError('REQUEST_STOP in CPU preflight')
            if setting.get('enabled',False):
                args=SimpleNamespace(target=target,via_landmark=setting.get('landmark_id','auto'),
                    landmark_pass_policy=setting.get('policy','prefer'),
                    landmark_pass_config=setting_path if setting.get('settings') else None)
                _,planning=landmark_plan(world,args,config.get('max_steps',5000))
            else:_,planning=precompute_smooth_teacher(world,target,config.get('max_steps',5000))
            reports[target]=planning
        result=dict(passed=True,source_sha256=world.source_hashes,targets=reports)
    except Exception as exc:
        result=dict(passed=False,error_type=type(exc).__name__,error=str(exc),
            failure_report=getattr(exc,'report',None),targets=reports)
    write_json(report_path,result)
    return 0 if result['passed'] else 1


class Batch:
    def __init__(self,args):
        self.args=args;self.config=read(args.config);self.output=args.output.expanduser().resolve()
        if self.config.get('schema_version')!='lunar_dataset_batch/14.0':raise ValueError('Unknown batch configuration')
        self.budget=float(self.config.get('budget_cny',10.))
        if not math.isfinite(self.budget) or self.budget<=0:raise ValueError('Batch budget_cny must be finite and positive')
        self.unit_config=read(resolve(self.config['unit_config']))
        self.template=resolve(self.config.get('terrain_template',self.unit_config['terrain_template']))
        runtime=args.isaac_python or self.config.get('isaac_python') or os.environ.get('ISAAC_PYTHON')
        environment_file=args.ppio_env or self.config.get('ppio_env') or os.environ.get('PPIO_ENV_FILE')
        if not runtime or not environment_file:raise ValueError('Set --isaac-python and --ppio-env (or their configuration/environment fields)')
        self.runtime=str(resolve(runtime))
        self.cache=resolve(args.provider_cache or self.config['provider_cache'])
        self.env_file=resolve(environment_file)
        self.host=self.config.get('publish_host','lab-a100-20')
        self.remote=self.config.get('remote_root','/data/zhenghengcao/lunar_cave_query100_20261111_v14')
        self.lock=threading.RLock();self.admission=threading.Lock();self.active={}
        self.slots=self.config['layouts'];self.max_attempts=int(self.config.get('maximum_cpu_layout_attempts',5))
        if not 1<=self.max_attempts<=5:raise ValueError('CPU replacements are bounded to at most five attempts')
        if len({row['seed'] for row in self.slots})!=len(self.slots):raise ValueError('Duplicate physical seed')
        if any(row['split'] not in ('train','val','test') for row in self.slots):raise ValueError('Unknown split')
        if self.config.get('requested_units',100)!=len(self.slots)*4:raise ValueError('Each physical layout must contribute exactly four units')
        expected=self.config.get('expected_layout_splits',{'train':20,'val':3,'test':2})
        if set(expected)!=set(('train','val','test')):raise ValueError('Declare all three physical split counts')
        if {split:sum(row['split']==split for row in self.slots) for split in expected}!=expected:
            raise ValueError('Physical layout split counts differ from the declared plan')
        query_protocol=read(resolve(self.unit_config['annotation']['query_protocol']))
        if self.unit_config.get('annotation',{}).get('query_count')!=query_protocol['query_count']:
            raise ValueError('Unit query count must match its complete prescribed composition protocol')
        self.unit_config['annotation']['budget_cny']=self.budget
        self.unit_config['terrain_template']=str(self.template)
        self.unit_config['landmark_pass']=dict(self.unit_config.get('landmark_pass',{}),enabled=True,landmark_id='auto',policy='prefer')
        self.unit_config['training_view']=dict(self.unit_config.get('training_view',{}),enabled=True,stride=5)
        self.reserved={row['seed'] for row in self.slots};self.entries={}
        self.prebuilt={int(seed):resolve(path) for seed,path in self.config.get('prebuilt_layouts',{}).items()}
        for directory in ('layouts','layout_state','unit_configs','unit_state','units','delivery_state','logs','catalog'):
            (self.output/directory).mkdir(parents=True,exist_ok=True)
        self.state=self.output/'batch_state.json'

    def stopped(self):return (self.output/'REQUEST_STOP').exists()

    def environment(self,*,cpu=False):
        environment=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        lib=Path(self.runtime).parent.parent/'lib'
        environment['LD_LIBRARY_PATH']=':'.join([str(lib),*map(str,self.config.get('cpu_library_paths',[])),environment.get('LD_LIBRARY_PATH','')])
        libraries=[lib/name for name in ('libstdc++.so.6','libgcc_s.so.1')]
        if all(path.exists() for path in libraries):environment['LD_PRELOAD']=':'.join(map(str,libraries))
        environment['PYTHONPATH']=':'.join([str(ROOT),*map(str,self.config.get('cpu_pythonpaths',[])),environment.get('PYTHONPATH','')])
        environment['CUDA_VISIBLE_DEVICES']='' if cpu else str(self.config.get('cuda_visible_devices','0'))
        return environment

    def prepare(self):
        if not self.template.is_dir() or (self.template/'targets.json').exists():raise ValueError('Choose an existing pre-placement terrain template')
        if not self.cache.is_dir():raise ValueError('The shared pre-existing PPIO ledger/cache is required')
        if not self.env_file.is_file():raise ValueError('The existing provider environment file is missing')
        self.code_hashes={str(path.relative_to(ROOT)):sha256(path) for path in sorted([*ROOT.glob('*.py'),*(ROOT/'cave').glob('*.py')])}
        self.resources={str(resolve(self.unit_config['annotation'][name])):sha256(resolve(self.unit_config['annotation'][name]))
            for name in ('semantic_config','query_protocol')}
        self.terrain_hashes={name:sha256(self.template/name) for name in ('cave.usdc','scene_config.json','mesh.npz','boulders.json','geometry_report.json')}
        self.prebuilt_hashes={str(seed):{name:sha256(path/name) for name in ('cave.usdc','scene_config.json','mesh.npz','targets.json','task_obstacles.json')}
            for seed,path in self.prebuilt.items()}
        identity=dict(config=self.config,unit_config=self.unit_config,runtime=self.runtime,
            shared_provider_cache=str(self.cache),ppio_env=str(self.env_file),
            terrain_files_sha256=self.terrain_hashes,resource_sha256=self.resources,prebuilt_layout_sha256=self.prebuilt_hashes,
            code_sha256=self.code_hashes)
        fingerprint=object_hash(identity)
        if self.state.exists():
            if read(self.state)['identity_sha256']!=fingerprint:raise ValueError('Frozen batch source/config changed')
            if not self.args.resume:raise FileExistsError('Existing batch requires explicit --resume')
        else:write_json(self.state,dict(identity_sha256=fingerprint,identity=identity,initial_provider_usage=self.usage()))
        self.identity=fingerprint
        self.status('prepared_without_gpu_or_api')

    def usage(self):return PPIOClient(self.cache,budget_cny=self.budget).usage()

    def frozen(self):
        current={str(path.relative_to(ROOT)):sha256(path) for path in sorted([*ROOT.glob('*.py'),*(ROOT/'cave').glob('*.py')])}
        if current!=self.code_hashes:raise ValueError('Frozen batch source files changed')
        if any(sha256(Path(path))!=digest for path,digest in self.resources.items()):raise ValueError('Frozen semantic/query protocol changed')
        if any(sha256(self.template/path)!=digest for path,digest in self.terrain_hashes.items()):raise ValueError('Frozen terrain geometry changed')
        if any(sha256(self.prebuilt[int(seed)]/name)!=digest for seed,files in self.prebuilt_hashes.items() for name,digest in files.items()):
            raise ValueError('Prebuilt adoption layout changed')

    def status(self,phase,**extra):
        with self.lock:
            value=dict(phase=phase,published_units=len(self.entries),requested_units=len(self.slots)*4,
                free_gib=round(shutil.disk_usage(self.output).free/GIB,3),updated_unix_s=time.time(),
                automatic_gpu_api_retry=False,active_units=deepcopy(self.active),**extra)
            write_json(self.output/'progress.json',value)
            print(json.dumps(value,ensure_ascii=False),flush=True)

    def command(self,command,log,*,cpu=False,unit_root=None,on_spawn=None):
        if self.stopped():raise InterruptedError('REQUEST_STOP before stage')
        with Path(log).open('ab') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=self.environment(cpu=cpu),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            if on_spawn:on_spawn(process.pid)
            while process.poll() is None:
                if self.stopped() and unit_root and (Path(unit_root)/'status.json').exists():
                    if read(Path(unit_root)/'status.json').get('phase')!='complete':(Path(unit_root)/'REQUEST_STOP').touch()
                time.sleep(1)
        return process.returncode

    def layout(self,slot_index,slot):
        state=self.output/'layout_state'/f'{slot_index:03d}.json'
        if state.exists() and read(state).get('passed'):
            result=read(state)
            proof=read(result['preflight'])
            if any(sha256(Path(result['layout'])/name)!=digest for name,digest in proof['source_sha256'].items()):
                raise ValueError('Cached layout changed after its CPU proof')
            self.reserved.add(result['seed']);return result
        previous=read(state) if state.exists() else dict(attempts=[])
        for attempt in range(len(previous['attempts']),self.max_attempts):
            if self.stopped():raise InterruptedError('REQUEST_STOP before CPU layout')
            seed=slot['seed'] if attempt==0 else int.from_bytes(hashlib.sha256(f"{slot['seed']}:{attempt}".encode()).digest()[:4],'big')
            if attempt and seed in self.reserved:
                previous['attempts'].append(dict(seed=seed,passed=False,error='reserved_seed_collision'));write_json(state,previous);continue
            self.reserved.add(seed)
            directory=self.output/'layouts'/f'seed_{seed}'
            proof=self.output/'layout_state'/f'{slot_index:03d}_{attempt}.preflight.json'
            if directory.exists():
                previous['attempts'].append(dict(seed=seed,passed=False,error='prior_incomplete_cpu_candidate_preserved'))
                write_json(state,previous);continue
            self.status('cpu_layout_candidate',slot=slot_index,attempt=attempt+1,seed=seed)
            if seed in self.prebuilt:
                command=[sys.executable,str(ROOT/'stage_lunar_assets_v14_cpu.py'),'--kind','layout','--source',str(self.prebuilt[seed]),
                    '--expected-seed',str(seed),'--output',str(directory)]
            else:
                command=[self.runtime,str(ROOT/'build_lunar_search_demo_v8_cpu.py'),'--asset-revision','10',
                    '--source',str(self.template),'--seed',str(seed),'--output',str(directory)]
            code=self.command(command,self.output/'logs'/f'layout_{seed}.log',cpu=True)
            if code==0:
                config=self.output/'layout_state'/f'{slot_index:03d}_{attempt}.config.json'
                write_json(config,self.unit_config)
                code=self.command([self.runtime,str(Path(__file__).resolve()),'--preflight-layout',str(directory),
                    '--preflight-config',str(config),'--preflight-report',str(proof)],self.output/'logs'/f'preflight_{seed}.log',cpu=True)
                if code and not proof.exists():raise RuntimeError('CPU preflight initialization failed; resampling cannot repair missing dependencies')
            row=dict(seed=seed,passed=code==0,preflight=str(proof),exit_code=code)
            previous['attempts'].append(row);write_json(state,previous)
            if code==0:
                result=dict(previous,passed=True,seed=seed,requested_seed=slot['seed'],split=slot['split'],layout=str(directory),preflight=str(proof))
                write_json(state,result);return result
        raise RuntimeError('CPU candidate limit exhausted; no GPU/API attempted for layout slot '+str(slot_index))

    def collectors(self):
        result=[]
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():continue
            try:args=(process/'cmdline').read_bytes().split(b'\0')
            except OSError:continue
            if b'--output' in args and any(b'collect_lunar_trajectory_' in value for value in args):
                output=Path(os.fsdecode(args[args.index(b'--output')+1]))
                result.append(output)
        return result

    def available_resources(self):
        memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2}
        gpu=str(self.config.get('cuda_visible_devices','0')).split(',')[0]
        free=int(subprocess.check_output(['nvidia-smi','--id',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        return memory['MemAvailable']/1024**2,free/1024

    def capacity(self,unit_id):
        while True:
            if self.stopped():raise InterruptedError('REQUEST_STOP in resource guard')
            self.evict_if_needed()
            if shutil.disk_usage(self.output).free<float(self.config.get('minimum_staging_free_gib',4.))*GIB:
                raise RuntimeError('Insufficient staging space; unuploaded data preserved')
            collectors=self.collectors()
            with self.lock:others=[value for key,value in self.active.items() if key!=unit_id and not value.get('adopted')]
            initialized=all(any(read(path).get('phase') in ('capturing_integrated_trajectory','complete')
                for path in (other/'dataset/episodes').glob('*/trajectory/status.json')) for other in [Path(row['root']) for row in others])
            if len(collectors)<2 and initialized:
                host_free,gpu_free=self.available_resources()
                host_min=float(self.config.get('first_lane_min_host_gib',8.)) if not collectors else 8.
                gpu_min=float(self.config.get('first_lane_min_gpu_gib',6.)) if not collectors else 12.
                if host_free>=host_min and gpu_free>=gpu_min:return
            time.sleep(3)

    def evict_if_needed(self):
        if self.args.no_prune or shutil.disk_usage(self.output).free>=float(self.config.get('evict_below_free_gib',6.))*GIB:return
        with self.lock:
            for entry in sorted(self.entries.values(),key=lambda item:item['ordinal']):
                if entry['ordinal']<int(self.config.get('keep_review_units',4)) or entry.get('local_frames_evicted'):continue
                root=self.output/entry['relative_unit_root'];deleted=0
                # CoT observation pairs are small and retained for local review.
                keep=set()
                base=root/'dataset';manifest=read(base/'dataset_manifest.json')
                for episode in manifest['episodes']:
                    annotations=base/episode['language']/'annotations.jsonl'
                    chosen={json.loads(line)['sample_id'] for line in annotations.read_text().splitlines() if line.strip() and json.loads(line).get('cot_valid')}
                    trajectory=base/episode['trajectory']
                    for line in (trajectory/'frame_records.jsonl').read_text().splitlines():
                        frame=json.loads(line)
                        if frame['sample_id'] not in chosen:continue
                        for index in (frame['step_index'],max(0,frame['step_index']-1)):
                            for path in (trajectory/'frames').glob(f'{index:06d}_*'):
                                if path.exists():keep.add((path.stat().st_dev,path.stat().st_ino))
                delivery=read(self.output/'delivery_state'/entry['unit_id']/'manifest.json')
                transaction=self.output/'delivery_state'/entry['unit_id']
                if (sha256(transaction/'manifest.json')!=entry['delivery']['manifest_sha256']
                        or read(transaction/'receipt.json').get('manifest_sha256')!=entry['delivery']['manifest_sha256']
                        or read(transaction/'receipt.json').get('passed') is not True
                        or any(sha256(root/path)!=digest for path,digest in entry['critical_sha256'].items())):
                    raise ValueError('Eviction prerequisites no longer match the published metadata/receipt')
                candidates=[]
                for item in delivery['entries']:
                    relative=Path(item['path'])
                    if item['kind']!='file' or 'frames' not in relative.parts or not FRAME.fullmatch(relative.name):continue
                    path=root/relative
                    if path.is_symlink():raise ValueError('Published regular frame became a symlink; retained without eviction')
                    if not path.is_file():continue
                    stat=path.stat()
                    if (stat.st_dev,stat.st_ino) in keep:continue
                    version=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
                    if stat.st_size!=item['size_bytes'] or sha256(path)!=item['sha256']:
                        raise ValueError('A changed/unuploaded frame is retained; eviction stopped')
                    after=path.stat()
                    if version!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                        raise ValueError('Frame changed during eviction preflight')
                    candidates.append((path,version))
                # First verify the complete deletion set, then unlink only the
                # exact published versions; a mismatch preserves all candidates.
                for path,version in candidates:
                    stat=path.stat()
                    if version!=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns):
                        raise ValueError('Frame changed after eviction preflight')
                for path,_ in candidates:path.unlink();deleted+=1
                entry['local_frames_evicted']=True
                write_json(self.output/'unit_state'/f"{entry['unit_id']}.json",dict(phase='published',entry=entry,frame_eviction=dict(files=deleted,remote_sha_preceded_eviction=True)))
                self.status('verified_bulk_frames_evicted',unit_id=entry['unit_id'],files=deleted)
                if shutil.disk_usage(self.output).free>=float(self.config.get('evict_target_free_gib',8.))*GIB:break

    def run_unit(self,ordinal,layout,target):
        seed,split=layout['seed'],layout['split'];identity=f'seed_{seed}_{target}'
        root=self.output/'units'/identity;state=self.output/'unit_state'/(identity+'.json')
        try:
            self.frozen()
            old=read(state) if state.exists() else {}
            if old.get('phase')=='published':
                entry=old['entry']
                verification=verify_published_unit(state_dir=self.output/'delivery_state'/identity,ssh_options=self.config.get('ssh_options',[]))
                if verification['manifest_sha256']!=entry['delivery']['manifest_sha256']:raise ValueError('Remote publication changed')
                with self.lock:self.entries[identity]=entry
                return entry
            if old.get('phase') in ('pipeline_running','pipeline_failed'):
                existing=[]
                for process in Path('/proc').iterdir():
                    if not process.name.isdigit():continue
                    try:arguments=(process/'cmdline').read_bytes().split(b'\0')
                    except OSError:continue
                    if (b'--output' in arguments and any(value.endswith(b'lunar_dataset_pipeline_v14.py') for value in arguments)
                            and Path(os.fsdecode(arguments[arguments.index(b'--output')+1])).resolve()==root):existing.append(process)
                if len(existing)>1:raise RuntimeError('Multiple processes claim the same unit')
                if existing:
                    with self.lock:self.active[identity]=dict(root=str(root),adopted=False)
                    while (existing[0]/'cmdline').exists() and (existing[0]/'cmdline').read_bytes():
                        if self.stopped() and (root/'status.json').exists() and read(root/'status.json').get('phase')!='complete':(root/'REQUEST_STOP').touch()
                        time.sleep(2)
                if not (root/'status.json').exists() or read(root/'status.json').get('phase')!='complete':
                    raise RuntimeError('Prior GPU/API execution is preserved; automatic retry forbidden: '+identity)
            configuration=deepcopy(self.unit_config);configuration['layouts']=[dict(seed=seed,split=split,targets=[target])];configuration['targets']=[target]
            config_path=self.output/'unit_configs'/(identity+'.json');write_json(config_path,configuration)
            if not (root/'status.json').exists() or read(root/'status.json').get('phase')!='complete':
                if root.exists():raise RuntimeError('Uncached/partial unit is preserved: '+identity)
                if self.usage()['committed_cny']>=self.budget:raise RuntimeError(f'Shared {self.budget:g}-CNY paid-request budget exhausted')
                adopted=self.config.get('adopt_raw',{}).get(identity)
                command=[sys.executable,str(ROOT/'lunar_dataset_pipeline_v14.py'),'--config',str(config_path),'--output',str(root),
                    '--isaac-python',self.runtime,'--ppio-env',str(self.env_file),'--provider-cache',str(self.cache),'--layout-source',layout['layout']]
                if adopted:command+=['--raw-source',str(resolve(adopted))]
                with self.admission:
                    self.capacity(identity)
                    write_json(state,dict(phase='pipeline_running',command=command,ordinal=ordinal))
                    with self.lock:self.active[identity]=dict(root=str(root),adopted=bool(adopted))
                self.status('unit_pipeline',unit_id=identity)
                def launched(pid):
                    write_json(state,dict(phase='pipeline_running',command=command,ordinal=ordinal,pid=pid))
                    with self.lock:self.active[identity]['pid']=pid
                code=self.command(command,self.output/'logs'/(identity+'.log'),unit_root=root,on_spawn=launched)
                if code:raise RuntimeError('Unit GPU/API pipeline failed without retry: '+identity)
            frozen_run=read(root/'dataset/run_config.json')
            if frozen_run['config']!=configuration or frozen_run['code_sha256']!=object_hash(self.code_hashes):
                raise ValueError('Completed unit has different source/configuration')
            summary=summarize_unit(root,expected_seed=seed,expected_target=target,expected_split=split)
            write_json(state,dict(phase='pipeline_complete',summary=summary,ordinal=ordinal))
            self.status('publishing_unit',unit_id=identity)
            delivered=publish_unit(root,host=self.host,remote_root=self.remote,unit='units/'+identity,
                state_dir=self.output/'delivery_state'/identity,required_paths=summary['required_paths'],ssh_options=self.config.get('ssh_options',[]))
            entry=dict(summary,unit_id=identity,ordinal=ordinal,relative_unit_root='units/'+identity,
                delivery=delivered,local_frames_evicted=False)
            write_json(state,dict(phase='published',entry=entry))
            with self.lock:
                self.entries[identity]=entry
                build_catalog(self.output/'catalog/catalog.json',list(self.entries.values()),requested_units=len(self.slots)*4)
            self.status('unit_published',unit_id=identity)
            self.evict_if_needed()
            return entry
        except BaseException as exc:
            (self.output/'REQUEST_STOP').touch()
            if not state.exists() or read(state).get('phase') not in ('published','pipeline_complete'):
                phase='pipeline_failed' if state.exists() and read(state).get('phase') in ('pipeline_running','pipeline_failed') else 'blocked_before_pipeline'
                write_json(state,dict(phase=phase,error=str(exc),automatic_retry=False))
            self.status('failed_closed',unit_id=identity,error=str(exc))
            raise
        finally:
            with self.lock:self.active.pop(identity,None)

    def run(self):
        futures=[]
        try:
            if self.command([self.runtime,'-c','from pxr import Usd; import numpy,scipy'],self.output/'logs/cpu_dependencies.log',cpu=True):
                raise RuntimeError('CPU SDK dependencies unavailable; layout resampling would not fix this')
            with ThreadPoolExecutor(max_workers=self.args.capture_lanes) as pool:
                try:
                    for index,slot in enumerate(self.slots):
                        if self.stopped():raise InterruptedError('REQUEST_STOP before layout')
                        for future in futures:
                            if future.done():future.result()
                        self.frozen();layout=self.layout(index,slot)
                        for offset,target in enumerate(TARGETS):futures.append(pool.submit(self.run_unit,index*4+offset,layout,target))
                    for future in as_completed(futures):future.result()
                except BaseException:
                    (self.output/'REQUEST_STOP').touch()
                    for future in futures:future.cancel()
                    raise
            catalog=build_catalog(self.output/'catalog/catalog.json',list(self.entries.values()),requested_units=len(self.slots)*4,complete=True)
            if not (self.output/'catalog/provider_usage.json').exists():
                write_json(self.output/'catalog/provider_usage.json',dict(budget_cny=self.budget,shared_cache=str(self.cache),**self.usage()))
            publish_unit(self.output/'catalog',host=self.host,remote_root=self.remote,unit='catalog',
                state_dir=self.output/'delivery_state/catalog',required_paths=['catalog.json','provider_usage.json'],ssh_options=self.config.get('ssh_options',[]))
            self.status('complete',query_counts=catalog['counts'])
        except BaseException as exc:
            (self.output/'REQUEST_STOP').touch();self.status('stopped_or_failed',error=str(exc));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/lunar_dataset_batch_v14.json');parser.add_argument('--output',type=Path)
    parser.add_argument('--isaac-python');parser.add_argument('--ppio-env');parser.add_argument('--provider-cache')
    parser.add_argument('--capture-lanes',type=int,choices=(1,2),default=2)
    parser.add_argument('--run',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--no-prune',action='store_true')
    parser.add_argument('--preflight-layout',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--preflight-config',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--preflight-report',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.preflight_layout:return cpu_preflight(args.preflight_layout,args.preflight_config,args.preflight_report)
    if args.config is None or args.output is None:parser.error('--config and --output are required')
    batch=Batch(args)
    with (batch.output/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        batch.prepare()
        if args.run:batch.run()
    return 0


if __name__=='__main__':raise SystemExit(main())
