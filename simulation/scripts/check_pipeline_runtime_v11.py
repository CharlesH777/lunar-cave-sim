#!/usr/bin/env python3
"""Read-only CPU preflight. Does not import Isaac, create a GPU context or install.

This diagnostic intentionally parses on Python 3.6 so an old system interpreter
can report why it cannot run the pipeline. The pipeline itself requires >=3.10;
Isaac Sim 5.1 specifically requires its Python 3.11 ABI environment.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REQUIREMENTS_URL = 'https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html'
PYTHON_URL = 'https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html'
DEPENDENCIES = ['numpy', 'scipy', 'trimesh', 'PIL', 'pxr', 'skimage', 'matplotlib']
SECRET_NAMES = ['PPIO_API_KEY', 'ANTHROPIC_API_KEY']


def version_tuple(value):
    try:
        return tuple(int(part) for part in str(value).strip().split('.'))
    except (TypeError, ValueError):
        return ()


def os_release():
    result = {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()}
    path = Path('/etc/os-release')
    if path.is_file():
        for line in path.read_text().splitlines():
            key, separator, value = line.partition('=')
            if separator and key in ['ID', 'VERSION_ID', 'PRETTY_NAME']:
                result[key.lower()] = value.strip().strip('"').strip("'")
    result['glibc_name'], result['glibc_version'] = platform.libc_ver()
    return result


def dependency_checks():
    # Separate subprocess imports CPU libraries only. Never echo its stdout or
    # exception text: third-party diagnostics must not accidentally expose env.
    code = '''import importlib,json
result={}
for name in ['numpy','scipy','trimesh','PIL','pxr','skimage','matplotlib']:
    try:
        module=importlib.import_module(name)
        if name=='pxr':
            importlib.import_module('pxr.Usd')
            importlib.import_module('pxr.Gf')
        result[name]={'import_ok':True,'version':str(getattr(module,'__version__','unknown'))}
    except BaseException as exc:
        result[name]={'import_ok':False,'error_type':type(exc).__name__}
print('PREFLIGHT_CPU_RESULT='+json.dumps(result))
'''
    env = dict(os.environ); env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        process = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True, timeout=45, env=env)
        lines = [line for line in process.stdout.splitlines() if line.startswith('PREFLIGHT_CPU_RESULT=')]
        if lines:
            return json.loads(lines[-1].split('=', 1)[1])
        return {name: {'import_ok': False, 'error_type': 'ChildProcessExit'+str(process.returncode)} for name in DEPENDENCIES}
    except BaseException as exc:
        return {name: {'import_ok': False, 'error_type': type(exc).__name__} for name in DEPENDENCIES}


def isaac_package():
    result = {'discoverable_without_import': False, 'version': None, 'imported': False}
    try:
        result['discoverable_without_import'] = importlib.util.find_spec('isaacsim') is not None
    except (ImportError, ValueError, AttributeError):
        pass
    try:
        from importlib import metadata
        result['version'] = metadata.version('isaacsim')
    except Exception:
        pass
    return result


def gpu_inventory(disabled):
    result = {'query': 'read-only nvidia-smi inventory; no CUDA/RTX initialization', 'available': False,
              'gpus': [], 'renderer_verified': False}
    executable = shutil.which('nvidia-smi')
    if disabled or executable is None:
        result['reason'] = 'disabled_by_cli' if disabled else 'nvidia_smi_not_found'; return result
    try:
        process = subprocess.run([executable, '--query-gpu=name,driver_version,memory.total',
                                  '--format=csv,noheader,nounits'], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True, timeout=10)
        if process.returncode:
            result['reason'] = 'nvidia_smi_exit_'+str(process.returncode); return result
        for line in process.stdout.splitlines():
            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 3:
                continue
            name, driver, memory = parts
            tested = (580, 88) if platform.system() == 'Windows' else (580, 65, 6)
            result['gpus'].append({'name': name, 'driver_version': driver,
                'memory_total_mib': float(memory) if memory.replace('.', '', 1).isdigit() else None,
                'known_non_rt_gpu': any(token in name.upper() for token in ['A100', 'H100']),
                'driver_at_or_above_documented_tested_platform_version': version_tuple(driver) >= tested,
                'driver_is_documented_tested_linux_version': driver == '580.65.06'})
        result['available'] = bool(result['gpus'])
    except BaseException as exc:
        result['reason'] = type(exc).__name__
    return result


def secret_presence(env_path):
    result = {'environment': {name: bool(os.environ.get(name)) for name in SECRET_NAMES},
              'file_configured': env_path is not None, 'file_exists': False,
              'file': {name: False for name in SECRET_NAMES}}
    if env_path is not None:
        path = Path(env_path).expanduser()
        result['file_exists'] = path.is_file()
        if result['file_exists']:
            try:
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith('export '):
                        line = line[7:].lstrip()
                    key, separator, value = line.partition('=')
                    if separator and key.strip() in SECRET_NAMES:
                        value = value.strip().strip('"').strip("'")
                        result['file'][key.strip()] = bool(value) and not value.startswith('#')
            except (OSError, UnicodeError):
                result['file_readable'] = False
    result['any_present'] = any(result['environment'].values()) or any(result['file'].values())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--json', type=Path, help='Optional JSON report; stdout is always JSON')
    parser.add_argument('--mode', choices=['cpu', 'isaac', 'all'], default='all')
    parser.add_argument('--experience', type=Path)
    parser.add_argument('--ext-folder', type=Path, action='append', default=[])
    parser.add_argument('--isaac-root', type=Path)
    parser.add_argument('--ppio-env', type=Path)
    parser.add_argument('--api-required', action='store_true')
    parser.add_argument('--no-nvidia-smi', action='store_true')
    args = parser.parse_args(argv)
    operating = os_release(); dependencies = dependency_checks(); isaac = isaac_package()
    inventory = gpu_inventory(args.no_nvidia_smi); secrets = secret_presence(args.ppio_env)
    python_ok = sys.version_info[:2] >= (3, 10); abi_ok = sys.version_info[:2] == (3, 11)
    experience = args.experience or os.environ.get('ISAAC_EXPERIENCE')
    if not experience and os.environ.get('ISAACLAB_PATH'):
        experience = Path(os.environ['ISAACLAB_PATH'])/'apps/isaaclab.python.headless.rendering.kit'
    experience_result = {'explicitly_configured': experience is not None,
        'exists': None if experience is None else Path(experience).expanduser().is_file(),
        'selection': 'SimulationApp automatic default, not launch-verified' if experience is None else str(Path(experience).expanduser()),
        'launch_verified': False}
    folders = list(args.ext_folder) + [Path(p) for p in os.environ.get('ISAAC_EXT_FOLDER', '').split(os.pathsep) if p]
    folder_checks = [{'path': str(p.expanduser()), 'exists': p.expanduser().is_dir()} for p in folders]
    cpu_blockers = []
    if not python_ok:
        cpu_blockers.append('pipeline_python_requires_at_least_3_10')
    cpu_blockers += ['cpu_dependency_import_failed:'+name for name, check in dependencies.items() if not check['import_ok']]
    render_blockers = list(cpu_blockers)
    if not abi_ok:
        render_blockers.append('isaac_5_1_requires_python_3_11_exact_abi')
    if not isaac['discoverable_without_import']:
        render_blockers.append('isaacsim_not_discoverable_in_selected_interpreter')
    if isaac['version'] and not isaac['version'].startswith('5.1.'):
        render_blockers.append('selected_isaacsim_package_is_not_5_1')
    if experience is not None and not experience_result['exists']:
        render_blockers.append('configured_experience_file_not_found')
    if any(not check['exists'] for check in folder_checks):
        render_blockers.append('configured_extension_folder_not_found')
    if operating['system'] == 'Linux':
        if operating.get('id') != 'ubuntu' or operating.get('version_id') not in ['22.04', '24.04']:
            render_blockers.append('os_not_in_documented_x86_64_ubuntu_matrix')
        if operating['glibc_name'] == 'glibc' and version_tuple(operating['glibc_version']) < (2, 35):
            render_blockers.append('linux_glibc_below_5_1_pip_wheel_requirement_2_35')
    elif operating['system'] != 'Windows':
        render_blockers.append('os_not_in_documented_x86_64_support_matrix')
    if operating['machine'].lower() not in ['x86_64', 'amd64']:
        render_blockers.append('this_preflight_targets_x86_64_pipeline_not_dgx_spark_aarch64')
    if not inventory['available']:
        render_blockers.append('gpu_inventory_unavailable_render_compatibility_unverified')
    if any(gpu['known_non_rt_gpu'] for gpu in inventory['gpus']):
        render_blockers.append('inventory_contains_documented_unsupported_non_rt_a100_h100_select_compatible_rtx_host')
    if inventory['gpus'] and not any(gpu['driver_at_or_above_documented_tested_platform_version'] for gpu in inventory['gpus']):
        render_blockers.append('driver_below_documented_tested_baseline_compatibility_not_accepted_without_runtime_validation')
    requested = cpu_blockers if args.mode == 'cpu' else render_blockers
    if args.api_required and not secrets['any_present']:
        requested = requested + ['requested_api_annotation_has_no_configured_secret']
    report = {'schema_version': 'lunar_pipeline_runtime_preflight/11.0',
        'checked_utc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode,
        'read_only': True, 'gpu_context_initialized': False, 'isaac_imported': False,
        'packages_installed': False, 'network_api_called': False,
        'python': {'executable': sys.executable, 'version': platform.python_version(),
                   'pipeline_at_least_3_10': python_ok, 'isaac_5_1_exact_python_3_11': abi_ok},
        'os': operating, 'cpu_dependencies': dependencies, 'isaacsim': isaac,
        'experience': experience_result, 'extension_folders': folder_checks,
        'isaac_root': {'configured': args.isaac_root is not None,
                       'exists': None if args.isaac_root is None else args.isaac_root.expanduser().is_dir()},
        'gpu_inventory': inventory, 'secrets': secrets,
        'official_5_1_requirements': {'sources': [REQUIREMENTS_URL, PYTHON_URL],
            'python': '3.11', 'x86_64_os': ['Ubuntu 22.04', 'Ubuntu 24.04', 'Windows 10', 'Windows 11'],
            'linux_pip_glibc_minimum': '2.35', 'documented_tested_linux_driver': '580.65.06',
            'documented_tested_windows_driver': '580.88',
            'driver_scope': 'Documentation lists tested versions, not a proof that every older driver fails or every newer driver works.'},
        'cpu_ready': not cpu_blockers, 'render_preflight_eligible': not render_blockers,
        'render_compatibility_verified': False, 'api_secret_present': secrets['any_present'],
        'cpu_blockers': cpu_blockers, 'render_blockers': render_blockers,
        'passed': not requested, 'blockers': requested, 'training_ready': False,
        'next_action': 'Use a compatible isolated interpreter/RTX runtime, then run the documented short sensor smoke test; this script does not alter the host.'}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+'\n'
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.expanduser().with_suffix(args.output.suffix+'.tmp')
        temporary.write_text(encoded); temporary.replace(args.output.expanduser())
    sys.stdout.write(encoded)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
