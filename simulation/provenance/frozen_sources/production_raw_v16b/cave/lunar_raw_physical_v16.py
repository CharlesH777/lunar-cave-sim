"""Physical-only helpers derived from the reviewed V14 batch scheduler.

No provider client, language stage or query module is imported. Layout search,
rendering admission and native physical bounds retain their original policy.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TARGETS=('astronaut','lunar_rover','lunar_base','chinese_flag')
GIB=1024**3
PHYSICAL_HELPERS_SOURCE_SHA256='eafb3b5b54f08a49b1e7270cf70462d5cebb54fc061c89260f84d58f74ca6828'


def read(path):return json.loads(Path(path).read_text())

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def resolve(value):
    path=Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def raw_summary(raw,layout,*,seed,target):
    raw=Path(raw);layout=Path(layout);metadata=read(raw/'metadata.json');status=read(raw/'status.json')
    identity=f'seed_{seed}_{target}'
    if (status.get('phase')!='complete' or metadata.get('terminal_success') is not True
            or metadata.get('episode_id')!=identity or metadata.get('target_id')!=target
            or metadata.get('capture_stride')!=1 or metadata.get('source_assets_unchanged') is not True
            or metadata.get('motion_controller')!='smooth_v13' or metadata.get('runtime_collision_mode')!='actual_geometry'):
        raise ValueError('Only complete identity-matched dense actual-geometry raw may be adopted')
    wall=metadata.get('body_wall_clearance') or {}
    if (metadata.get('canonical_stop_radius_m')!=1. or wall.get('required_gap_m')!=1.
            or wall.get('passed') is not True or wall.get('expanded_air_envelope_passed') is not True
            or wall.get('expanded_air_envelope_gap_m',0.)<1.05):
        raise ValueError('Missing canonical-region/expanded-body wall proof')
    if read(layout/'targets.json')['seed']!=seed:raise ValueError('Layout seed differs from the raw identity')
    for name,digest in metadata['source_sha256'].items():
        path=(layout/name).resolve(strict=True)
        if not path.is_relative_to(layout.resolve(strict=True)) or sha256(path)!=digest:
            raise ValueError('Raw/layout byte binding differs: '+name)
    with np.load(raw/'supervision.npz',allow_pickle=False) as archive:
        controls=archive['vw_si'].astype(float);stop=archive['stop'];dt=archive['dt_s']
    with np.load(raw/'teacher_only.npz',allow_pickle=False) as archive:
        positions=archive['executed_positions'];goal=archive['original_approach_xy_m'];planned=archive['planned_stop_xy_m']
    n=len(controls)
    if (not n or controls.shape!=(n,2) or positions.shape!=(n+1,3) or not np.isfinite(controls).all()
            or not np.allclose(dt,.2,atol=1e-8) or np.count_nonzero(stop)!=1 or stop[-1]!=1
            or not np.allclose(controls[-1],0.) or metadata['captured_frame_count']!=n
            or metadata['action_count']!=n or metadata['pose_count']!=n+1):
        raise ValueError('Native N/N+1/control/unique-STOP contract failed')
    v,w=controls.T;acceleration=np.diff(np.vstack([[0.,0.],controls]),axis=0)/.2
    pivots=int(np.count_nonzero((v<=1e-8)&(abs(w)>1e-6)))
    error=float(np.linalg.norm(positions[-1,:2]-goal));planned_error=float(np.linalg.norm(positions[-1,:2]-planned))
    if (pivots or v.min() < -1e-7 or v.max()>.400001 or abs(w).max()>.320001
            or abs(acceleration[:,0]).max()>.250001 or abs(acceleration[:,1]).max()>.240001
            or error>1.000001 or planned_error>.050001):
        raise ValueError('Native physical speed/slew/pivot/terminal bounds failed')
    return dict(frames=n,action_count=n,pose_count=n+1,native_action_dt_s=.2,future_horizon=8,
        stationary_turn_frames=pivots,canonical_goal_error_m=error,planned_stop_error_m=planned_error,
        maximum_omega_radps=float(abs(w).max()),maximum_linear_acceleration_mps2=float(abs(acceleration[:,0]).max()),
        maximum_angular_acceleration_radps2=float(abs(acceleration[:,1]).max()),body_wall_clearance=wall,
        source_sha256=metadata['source_sha256'])


def preflight_layout(layout,config,report):
    config=read(config);capture=config['capture'];adoptions=config.get('adopt_raw',{});reports={};world=None
    try:
        seed=read(Path(layout)/'targets.json')['seed']
        for target in TARGETS:
            if (Path(report).parent.parent/'REQUEST_STOP').exists():raise InterruptedError('REQUEST_STOP in CPU physical preflight')
            identity=f'seed_{seed}_{target}'
            if identity in adoptions:
                reports[target]=dict(adopted=True,source=str(resolve(adoptions[identity])),
                    summary=raw_summary(resolve(adoptions[identity]),layout,seed=seed,target=target))
                continue
            if world is None:
                from cave.lunar_trajectory_v11 import TerrainWorld
                world=TerrainWorld(layout,runtime_collision_mode='actual_geometry')
            from collect_lunar_trajectory_v14_isaac import landmark_plan
            from types import SimpleNamespace
            setting=capture.get('landmark_pass',{})
            if setting.get('enabled',True):
                setting_path=Path(report).with_suffix('.settings.json')
                if setting.get('settings'):write_json(setting_path,dict(settings=setting['settings']))
                args=SimpleNamespace(target=target,via_landmark=setting.get('landmark_id','auto'),
                    landmark_pass_policy=setting.get('policy','prefer'),landmark_pass_config=setting_path if setting.get('settings') else None)
                _,planning=landmark_plan(world,args,capture.get('max_steps',5000))
            else:
                from cave.lunar_smooth_teacher_v13 import precompute_smooth_teacher
                _,planning=precompute_smooth_teacher(world,target,capture.get('max_steps',5000))
            reports[target]=dict(adopted=False,planning=planning)
        source_hashes=world.source_hashes if world is not None else next(iter(reports.values()))['summary']['source_sha256']
        result=dict(passed=True,source_sha256=source_hashes,targets=reports,training_ready=False,annotation_started=False)
    except Exception as exc:
        result=dict(passed=False,error_type=type(exc).__name__,error=str(exc),failure_report=getattr(exc,'report',None),targets=reports)
    write_json(report,result)
    return 0 if result['passed'] else 1


class PhysicalBatch:

    def environment(self, *, cpu=False):
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        lib = Path(self.runtime).parent.parent / 'lib'
        environment['LD_LIBRARY_PATH'] = ':'.join([str(lib), *map(str, self.config.get('cpu_library_paths', [])), environment.get('LD_LIBRARY_PATH', '')])
        libraries = [lib / name for name in ('libstdc++.so.6', 'libgcc_s.so.1')]
        if all((path.exists() for path in libraries)):
            environment['LD_PRELOAD'] = ':'.join(map(str, libraries))
        environment['PYTHONPATH'] = ':'.join([str(ROOT), *map(str, self.config.get('cpu_pythonpaths', [])), environment.get('PYTHONPATH', '')])
        environment['CUDA_VISIBLE_DEVICES'] = '' if cpu else str(self.config.get('cuda_visible_devices', '0'))
        return environment

    def prepare_layout(self, slot_index, slot):
        state = self.output / 'layout_state' / f'{slot_index:03d}.json'
        if state.exists() and read(state).get('passed'):
            result = read(state)
            proof = read(result['preflight'])
            if any((sha256(Path(result['layout']) / name) != digest for name, digest in proof['source_sha256'].items())):
                raise ValueError('Cached layout changed after its CPU proof')
            self.reserved.add(result['seed'])
            return result
        previous = read(state) if state.exists() else dict(attempts=[])
        for attempt in range(len(previous['attempts']), self.max_attempts):
            if self.stopped():
                raise InterruptedError('REQUEST_STOP before CPU layout')
            seed = slot['seed'] if attempt == 0 else int.from_bytes(hashlib.sha256(f"{slot['seed']}:{attempt}".encode()).digest()[:4], 'big')
            if attempt and seed in self.reserved:
                previous['attempts'].append(dict(seed=seed, passed=False, error='reserved_seed_collision'))
                write_json(state, previous)
                continue
            self.reserved.add(seed)
            directory = self.output / 'layouts' / f'seed_{seed}'
            proof = self.output / 'layout_state' / f'{slot_index:03d}_{attempt}.preflight.json'
            if directory.exists():
                previous['attempts'].append(dict(seed=seed, passed=False, error='prior_incomplete_cpu_candidate_preserved'))
                write_json(state, previous)
                continue
            self.status('cpu_layout_candidate', slot=slot_index, attempt=attempt + 1, seed=seed)
            if seed in self.prebuilt:
                command = [sys.executable, str(ROOT / 'stage_lunar_assets_v14_cpu.py'), '--kind', 'layout', '--source', str(self.prebuilt[seed]), '--expected-seed', str(seed), '--output', str(directory)]
            else:
                command = [self.runtime, str(ROOT / 'build_lunar_search_demo_v8_cpu.py'), '--asset-revision', '10', '--source', str(self.template), '--seed', str(seed), '--output', str(directory)]
            code = self.command(command, self.output / 'logs' / f'layout_{seed}.log', cpu=True)
            if code == 0:
                config = self.output / 'layout_state' / f'{slot_index:03d}_{attempt}.config.json'
                write_json(config, dict(capture=self.capture_config, adopt_raw=self.config.get('adopt_raw', {})))
                code = self.command([self.runtime, str(self.preflight_script), '--preflight-layout', str(directory), '--preflight-config', str(config), '--preflight-report', str(proof)], self.output / 'logs' / f'preflight_{seed}.log', cpu=True)
                if code and (not proof.exists()):
                    raise RuntimeError('CPU preflight initialization failed; resampling cannot repair missing dependencies')
            row = dict(seed=seed, passed=code == 0, preflight=str(proof), exit_code=code)
            previous['attempts'].append(row)
            write_json(state, previous)
            if code == 0:
                result = dict(previous, passed=True, seed=seed, requested_seed=slot['seed'], split=slot['split'], layout=str(directory), preflight=str(proof))
                write_json(state, result)
                return result
        raise RuntimeError('CPU candidate limit exhausted; no GPU/API attempted for layout slot ' + str(slot_index))

    def collectors(self):
        result = []
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():
                continue
            try:
                args = (process / 'cmdline').read_bytes().split(b'\x00')
            except OSError:
                continue
            if b'--output' in args and any((b'collect_lunar_trajectory_' in value for value in args)):
                output = Path(os.fsdecode(args[args.index(b'--output') + 1]))
                result.append(output)
        return result

    def available_resources(self):
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
        gpu = str(self.config.get('cuda_visible_devices', '0')).split(',')[0]
        free = int(subprocess.check_output(['nvidia-smi', '--id', gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
        return (memory['MemAvailable'] / 1024 ** 2, free / 1024)

    def capacity(self, unit_id):
        while True:
            if self.stopped():
                raise InterruptedError('REQUEST_STOP in resource guard')
            self.evict_if_needed()
            if shutil.disk_usage(self.output).free < float(self.config.get('minimum_staging_free_gib', 4.0)) * GIB:
                self.status('waiting_for_raw_storage', unuploaded_data_preserved=True)
                time.sleep(3)
                continue
            collectors = self.collectors()
            with self.lock:
                others = [value for key, value in self.active.items() if key != unit_id and (not value.get('adopted')) and value.get('pid')]
            initialized = all(((Path(other['root']) / 'trajectory/status.json').is_file() and read(Path(other['root']) / 'trajectory/status.json').get('phase') in ('capturing_integrated_trajectory', 'complete') for other in others))
            if len(collectors) < 2 and initialized:
                host_free, gpu_free = self.available_resources()
                host_min = float(self.config.get('first_lane_min_host_gib', 8.0)) if not collectors else 8.0
                gpu_min = float(self.config.get('first_lane_min_gpu_gib', 6.0)) if not collectors else 12.0
                if host_free >= host_min and gpu_free >= gpu_min:
                    return
            time.sleep(3)
