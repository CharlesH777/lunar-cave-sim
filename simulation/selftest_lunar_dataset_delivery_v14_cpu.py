"""Isolated publication tests: real verifier/rsync, no SSH or remote server."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from cave.lunar_dataset_delivery_v14 import (DeliveryError, REMOTE_HELPER,
    SSHRsyncTransport, build_unit_manifest, canonical_bytes, manifest_digest,
    publish_unit, verify_published_unit)


class LocalTransport:
    """Execute the exact remote helper locally, and use local rsync paths."""
    def __init__(self, fail_transfer=False, corrupt_transfer=False, mutate_source=False):
        self.transfers = 0
        self.requests = []
        self.fail_transfer = fail_transfer
        self.corrupt_transfer = corrupt_transfer
        self.mutate_source = mutate_source

    def invoke(self, request):
        self.requests.append(request['operation'])
        result = subprocess.run([sys.executable, '-c', REMOTE_HELPER], input=canonical_bytes(request),
                                capture_output=True)
        reply = json.loads(result.stdout.decode('utf-8'))
        if result.returncode or reply.get('passed') is not True:
            raise DeliveryError(reply.get('error', result.stderr.decode('utf-8', 'replace')))
        return reply

    def transfer(self, source, remote_payload):
        self.transfers += 1
        payload = Path(remote_payload)
        if self.fail_transfer:
            partial = payload / 'reports/validation.json'
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b'{"pa')
            raise DeliveryError('Injected interruption after one partial file')
        subprocess.run(['rsync', '--archive', '--no-owner', '--no-group', '--checksum', '--partial',
            '--exclude=REQUEST_STOP', '--exclude=*.tmp', '--', str(source) + '/', str(payload) + '/'],
            check=True, capture_output=True)
        if self.corrupt_transfer:
            target = payload / 'reports/validation.json'
            original = target.stat()
            target.write_bytes(b'X' * original.st_size)
            os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        if self.mutate_source:
            (source / 'reports/validation.json').write_bytes(b'{"passed":false}\n')


def fixture(root):
    files = {
        'dense/dataset_manifest.json': b'{"complete":true}\n',
        'dense/reader_index.jsonl': b'{"step_index":0}\n',
        'dense/episodes/demo/metadata.json': b'{"terminal_success":true}\n',
        'dense/episodes/demo/supervision.npz': bytes(range(256)) * 20,
        'dense/episodes/demo/frames/000000_rgb224.png': b'example-frame-bytes\x00\xff',
        'language/annotations.jsonl': '{"cot_text":"我已进入洞穴"}\n'.encode(),
        'stride5/view_manifest.json': b'{"base_dataset":{"relative_path":"../dense"}}\n',
        'stride5/reader_index.jsonl': b'{"native_step_index":0}\n',
        'reports/validation.json': b'{"passed":true}\n',
        "README 'quote';$(touch INJECTION_MARKER).md": b'argv quotation fixture\n',
        '中文路径/说明.json': '{"示例":true}\n'.encode(),
        'REQUEST_STOP': b'control-not-payload\n',
        'scratch.tmp': b'transient\n',
        'cache.tmp/nested/not_payload.bin': b'transient-directory\n',
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / 'empty_directory').mkdir()
    (root / 'frame_alias.png').symlink_to('dense/episodes/demo/frames/000000_rgb224.png')
    (root / 'dense_alias').symlink_to('dense', target_is_directory=True)


def must_fail(call, contains=None):
    try:
        call()
    except (DeliveryError, OSError, ValueError) as error:
        if contains:
            assert contains.lower() in str(error).lower(), str(error)
        return
    raise AssertionError('Expected a closed failure')


def snapshot(root):
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            result[path.relative_to(root).as_posix()] = ('symlink', os.readlink(path))
        elif path.is_file():
            result[path.relative_to(root).as_posix()] = ('file', path.read_bytes())
        else:
            result[path.relative_to(root).as_posix()] = ('directory',)
    return result


def main():
    # The embedded verifier uses no syntax introduced after remote Python 3.6.
    ast.parse(REMOTE_HELPER, feature_version=(3, 6))
    with tempfile.TemporaryDirectory(prefix='lunar_delivery_v14_test_') as temporary:
        root = Path(temporary)
        source = root / "source tree 'quoted'"
        fixture(source)
        before = snapshot(source)
        remote = root / "remote root 'quoted';$(not_a_command)"
        state = root / 'state'
        required = ('dense/dataset_manifest.json', 'dense/reader_index.jsonl',
                    'language/annotations.jsonl', 'stride5/reader_index.jsonl', 'reports/validation.json')
        manifest = build_unit_manifest(source, required_paths=required)
        entries = {entry['path']: entry for entry in manifest['entries']}
        assert entries['reports/validation.json']['sha256'] == hashlib.sha256(b'{"passed":true}\n').hexdigest()
        assert manifest['file_count'] == 11
        assert manifest['excluded_control_paths'] == ['REQUEST_STOP', 'cache.tmp', 'scratch.tmp']
        assert entries['dense_alias']['kind'] == 'symlink'
        must_fail(lambda: build_unit_manifest(source, required_paths=['missing/index.json']), 'required')
        transport = LocalTransport()
        kwargs = dict(host='fake-a100', remote_root=str(remote), unit="units/demo 'quoted'", state_dir=state,
                      required_paths=required, transport=transport)
        result = publish_unit(source, **kwargs)
        destination = Path(result['remote_destination'])
        assert result['passed'] and not result['reused'] and transport.transfers == 1
        assert not (destination / 'REQUEST_STOP').exists() and not (destination / 'cache.tmp').exists()
        assert (destination / 'dense_alias').is_symlink()
        assert (destination / 'empty_directory').is_dir()
        assert json.loads((destination / '.delivery_manifest.json').read_text()) == manifest
        assert result['manifest_sha256'] == manifest_digest(manifest)
        again = publish_unit(source, **kwargs)
        assert again['reused'] and transport.transfers == 1
        # Exact source relocation keeps the same inventory and reusable remote unit.
        moved_source = root / 'relocated_source'
        shutil.copytree(source, moved_source, symlinks=True)
        assert publish_unit(moved_source, **kwargs)['reused']
        assert verify_published_unit(state_dir=state, transport=transport)['passed']
        (source / 'changed.txt').write_text('different source')
        count = len(transport.requests)
        must_fail(lambda: publish_unit(source, **kwargs), 'different source')
        assert len(transport.requests) == count
        must_fail(lambda: publish_unit(source, **dict(kwargs, state_dir=root / 'new_state')), 'different manifest')
        (source / 'changed.txt').unlink()
        must_fail(lambda: publish_unit(source, **dict(kwargs, unit='units/different')), 'destination')
        # Corrupt a published byte while retaining size: verification must reject it.
        report = destination / 'reports/validation.json'
        saved = report.read_bytes()
        report.write_bytes(b'X' * len(saved))
        must_fail(lambda: verify_published_unit(state_dir=state, transport=transport), 'SHA/size')
        assert report.read_bytes() == b'X' * len(saved)  # Failure never deletes or repairs it.
        report.write_bytes(saved)
        assert verify_published_unit(state_dir=state, transport=transport)['passed']
        (destination / 'unexpected.bin').write_bytes(b'untracked')
        must_fail(lambda: verify_published_unit(state_dir=state, transport=transport), 'Unexpected')
        assert (destination / 'unexpected.bin').read_bytes() == b'untracked'
        (destination / 'unexpected.bin').unlink()  # Explicit isolated-test repair, never tool behavior.
        index = destination / 'dense/reader_index.jsonl'
        saved_index = index.read_bytes()
        index.unlink()
        must_fail(lambda: verify_published_unit(state_dir=state, transport=transport), 'inventory')
        assert not index.exists()
        index.write_bytes(saved_index)
        # A failed partial transfer is unpublished; the exact same state resumes.
        partial_kwargs = dict(kwargs, unit='units/partial', state_dir=root / 'partial_state',
                              transport=LocalTransport(fail_transfer=True))
        must_fail(lambda: publish_unit(source, **partial_kwargs), 'interruption')
        assert not (remote / 'units/partial').exists()
        partial_stages = list((remote / '.incoming').glob('*/payload/reports/validation.json'))
        assert any(path.read_bytes() == b'{"pa' for path in partial_stages)
        assert publish_unit(source, **dict(partial_kwargs, transport=LocalTransport()))['passed']
        # Same-size / same-mtime transfer corruption proves SHA validation and checksum resume.
        corrupt_kwargs = dict(kwargs, unit='units/corrupt', state_dir=root / 'corrupt_state',
                              transport=LocalTransport(corrupt_transfer=True))
        must_fail(lambda: publish_unit(source, **corrupt_kwargs), 'SHA/size')
        assert not (remote / 'units/corrupt').exists()
        assert publish_unit(source, **dict(corrupt_kwargs, transport=LocalTransport()))['passed']
        # Source mutation during transfer cannot expose a final unit.
        changing = root / 'changing_source'
        shutil.copytree(source, changing, symlinks=True)
        changing_kwargs = dict(kwargs, unit='units/changing', state_dir=root / 'changing_state',
                               transport=LocalTransport(mutate_source=True))
        must_fail(lambda: publish_unit(changing, **changing_kwargs), 'Source changed')
        assert not (remote / 'units/changing').exists()
        # Retained state can reverify after separately authorized local frame eviction.
        shutil.rmtree(moved_source / 'dense/episodes/demo/frames')
        assert verify_published_unit(state_dir=state, transport=transport)['passed']
        # External/absolute/cyclic symlinks, special files, and unsafe paths fail locally.
        (source / 'external').symlink_to(root / 'state')
        must_fail(lambda: build_unit_manifest(source), 'relative')
        (source / 'external').unlink()
        (source / 'external').symlink_to('../state')
        must_fail(lambda: build_unit_manifest(source), 'escapes')
        (source / 'external').unlink()
        (source / 'cycle').symlink_to('cycle')
        must_fail(lambda: build_unit_manifest(source), 'cyclic')
        (source / 'cycle').unlink()
        os.mkfifo(source / 'pipe')
        must_fail(lambda: build_unit_manifest(source), 'Special')
        (source / 'pipe').unlink()
        must_fail(lambda: publish_unit(source, **dict(kwargs, unit='../escape')), 'relative')
        must_fail(lambda: publish_unit(source, **dict(kwargs, state_dir=source / 'bad_state')), 'outside')
        assert snapshot(source) == before
        assert not (root / 'INJECTION_MARKER').exists()
        # SSH command is a single shell-quoted script; paths remain JSON/stdin or argv.
        captured = []
        def fake_run(command, **options):
            captured.append((command, options))
            return subprocess.CompletedProcess(command, 0, stdout=b'{"passed":true,"phase":"staging"}', stderr=b'')
        with patch('cave.lunar_dataset_delivery_v14.subprocess.run', fake_run):
            ssh = SSHRsyncTransport('test-alias')
            ssh.invoke({'unit': "quotes ';$(ignored)"})
            ssh.transfer(source, str(remote / "space 'payload'"))
        assert shlex.split(captured[0][0][-1]) == ['python3', '-c', REMOTE_HELPER]
        assert captured[0][1]['input'] == canonical_bytes({'unit': "quotes ';$(ignored)"})
        rsync_command = captured[1][0]
        assert '--protect-args' in rsync_command and '--delete' not in rsync_command
        assert rsync_command[-2] == str(source) + '/'
        print(json.dumps(dict(passed=True, payload_files=manifest['file_count'],
            remote_python36_syntax=True, exact_reuse=True, relocation=True,
            partial_resume=True, corrupt_payload_rejected=True, checksum_resume=True,
            source_mutation_rejected=True, changed_destination_rejected=True,
            remote_verify_without_local_frames=True, symlink_and_path_guards=True,
            source_unchanged=True, ssh_quoting=True, actual_ssh_used=False)))


if __name__ == '__main__':
    main()
