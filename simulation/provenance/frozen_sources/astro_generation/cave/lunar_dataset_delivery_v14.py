"""Immutable, resumable CPU publication of self-contained dataset units.

Every source entry is inventoried. Only entries named ``REQUEST_STOP`` or
ending in ``.tmp`` (including their directory subtrees) are excluded and
explicitly listed. Relative internal symlinks are retained; external,
absolute, dangling, and cyclic symlinks fail closed. No source is modified or
deleted. A caller should finish its dataset validation before publishing.

The remote verifier is embedded and needs only Python >=3.6, ssh, and rsync.
It verifies payload SHA/size/inventory, then atomically renames its incoming
directory. Publication receipts are integrity records, not signatures.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import uuid
from pathlib import Path, PurePosixPath


SCHEMA = 'lunar_delivery_unit/14.0'
RESERVED = {'.delivery_manifest.json', '.delivery_receipt.json'}
CONTROL_POLICY = ['basename REQUEST_STOP', 'basename suffix .tmp (including directory subtrees)']


class DeliveryError(RuntimeError):
    pass


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def manifest_digest(manifest):
    return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def _relative(value):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value) or '\\' in value:
        raise DeliveryError('Unsafe relative path')
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise DeliveryError('Paths must be normalized, safe and relative: ' + value)
    return parsed.as_posix()


def _remote_root(value):
    if (not isinstance(value, str) or not value.startswith('/') or value == '/'
            or any(ord(c) < 32 for c in value) or '\\' in value
            or any(part in ('.', '..', '') for part in value.split('/')[1:])):
        raise DeliveryError('Remote root must be a normalized absolute directory other than /')
    return value


def _control(name):
    return name == 'REQUEST_STOP' or name.endswith('.tmp')


def _hash_file(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise DeliveryError('File changed while hashing: ' + str(path))
    return digest.hexdigest(), after.st_size


def build_unit_manifest(source, *, required_paths=()):
    """Inventory a completed tree without changing it or depending on its location."""
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise DeliveryError('Source must be a directory')
    entries, excluded = [], []
    for directory, directories, files in os.walk(source, followlinks=False):
        parent = Path(directory)
        for name in sorted(directories + files):
            path = parent / name
            relative = _relative(path.relative_to(source).as_posix())
            if relative in RESERVED:
                raise DeliveryError('Source collides with reserved publication metadata: ' + relative)
            info = path.lstat()
            if _control(name):
                excluded.append(relative)
                if name in directories:
                    directories.remove(name)
                continue
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                if Path(target).is_absolute() or any(ord(c) < 32 for c in target) or '\\' in target:
                    raise DeliveryError('Symlink target must be relative and safe: ' + relative)
                try:
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise DeliveryError('Dangling or cyclic symlink: ' + relative) from exc
                if not resolved.is_relative_to(source) or not (resolved.is_file() or resolved.is_dir()):
                    raise DeliveryError('Symlink escapes its publication unit: ' + relative)
                if any(_control(part) for part in resolved.relative_to(source).parts):
                    raise DeliveryError('Symlink targets an excluded control entry: ' + relative)
                entries.append(dict(path=relative, kind='symlink', target=target))
                if name in directories:
                    directories.remove(name)
            elif stat.S_ISDIR(info.st_mode):
                entries.append(dict(path=relative, kind='directory'))
            elif stat.S_ISREG(info.st_mode):
                digest, size = _hash_file(path)
                entries.append(dict(path=relative, kind='file', size_bytes=size, sha256=digest))
            else:
                raise DeliveryError('Special filesystem entry is not a dataset file: ' + relative)
        directories.sort()
    entries.sort(key=lambda entry: entry['path'])
    by_path = {entry['path']: entry for entry in entries}
    required = sorted({_relative(path) for path in required_paths})
    if any(path not in by_path for path in required):
        raise DeliveryError('A required metadata/index/report/payload path is absent or excluded')
    return dict(schema_version=SCHEMA, hash_algorithm='sha256', entries=entries,
        entry_count=len(entries), file_count=sum(entry['kind'] == 'file' for entry in entries),
        total_file_bytes=sum(entry.get('size_bytes', 0) for entry in entries),
        excluded_control_policy=CONTROL_POLICY, excluded_control_paths=sorted(excluded),
        required_paths=required)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


# Keep this code compatible with Python 3.6; it runs without project imports.
REMOTE_HELPER = r'''
import datetime
import fcntl
import hashlib
import json
import os
import stat
import sys

SCHEMA = 'lunar_delivery_unit/14.0'
RESERVED = {'.delivery_manifest.json', '.delivery_receipt.json'}

def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def relative(value):
    if (not isinstance(value, str) or not value or value.startswith('/') or '\\' in value
            or any(ord(c) < 32 for c in value) or any(p in ('', '.', '..') for p in value.split('/'))):
        raise ValueError('Unsafe relative path')
    return value

def inside(path, root):
    return os.path.commonpath([path, root]) == root

def ensure_directory(path):
    # Refuse symlink parents as well as final symlink destinations.
    current = '/'
    for component in path.split('/')[1:]:
        current = os.path.join(current, component)
        if os.path.lexists(current):
            if os.path.islink(current) or not os.path.isdir(current):
                raise ValueError('Remote directory path contains a symlink or non-directory')
        else:
            try:
                os.mkdir(current)
            except FileExistsError:
                # Different unit transactions share parent directories.
                # Accept a concurrently created real directory only.
                if os.path.islink(current) or not os.path.isdir(current):
                    raise ValueError('Concurrent remote directory is a symlink or non-directory')

def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)

def write_json(path, value, temporary_directory=None):
    temporary = os.path.join(temporary_directory or os.path.dirname(path), '.' + os.path.basename(path) + '.tmp')
    with open(temporary, 'wb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def file_digest(path):
    before = os.stat(path)
    hasher = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    after = os.stat(path)
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Remote file changed during verification')
    return hasher.hexdigest(), after.st_size

def verify_tree(payload, manifest):
    if manifest.get('schema_version') != SCHEMA or manifest.get('hash_algorithm') != 'sha256':
        raise ValueError('Unsupported unit manifest')
    entries = manifest['entries']
    expected = {}
    for entry in entries:
        name = relative(entry['path'])
        if name in RESERVED or name in expected:
            raise ValueError('Reserved or duplicate manifest entry')
        expected[name] = entry
    seen = set()
    files = total = 0
    for directory, directories, names in os.walk(payload, followlinks=False):
        for name in sorted(directories + names):
            path = os.path.join(directory, name)
            rel = os.path.relpath(path, payload).replace(os.sep, '/')
            if rel in RESERVED:
                if not stat.S_ISREG(os.lstat(path).st_mode):
                    raise ValueError('Publisher metadata is not a regular file')
                continue
            if rel not in expected:
                raise ValueError('Unexpected payload entry: ' + rel)
            entry = expected[rel]
            seen.add(rel)
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                if entry['kind'] != 'symlink' or os.readlink(path) != entry['target'] or os.path.isabs(entry['target']):
                    raise ValueError('Symlink identity changed: ' + rel)
                resolved = os.path.realpath(path)
                if not inside(resolved, payload) or not os.path.exists(resolved):
                    raise ValueError('Symlink escapes or is dangling: ' + rel)
                if name in directories:
                    directories.remove(name)
            elif stat.S_ISDIR(mode):
                if entry['kind'] != 'directory':
                    raise ValueError('Entry type changed: ' + rel)
            elif stat.S_ISREG(mode):
                actual_digest, size = file_digest(path)
                if entry['kind'] != 'file' or size != entry['size_bytes'] or actual_digest != entry['sha256']:
                    raise ValueError('Payload SHA/size mismatch: ' + rel)
                files += 1
                total += size
            else:
                raise ValueError('Unsupported payload entry: ' + rel)
        directories.sort()
    if (seen != set(expected) or len(entries) != manifest['entry_count'] or files != manifest['file_count']
            or total != manifest['total_file_bytes']):
        raise ValueError('Payload inventory is incomplete or has incorrect counts')
    if any(relative(path) not in expected for path in manifest['required_paths']):
        raise ValueError('Required metadata/index/report path is absent')
    return dict(entry_count=len(seen), file_count=files, total_file_bytes=total)

def seed_payload(root, payload, manifest, request):
    """Reuse verified immutable raw files without modifying their contents or attrs."""
    references = request.get('preseed_units', [])
    if not references:
        return dict(files=0, logical_bytes=0, verified_sources=[])
    if not isinstance(references, list) or len(references) > 8:
        raise ValueError('Invalid preseed source list')
    attributes = request.get('preseed_target_attributes', {})
    candidates = {}
    verified = []
    for reference in references:
        source_unit = relative(reference['unit'])
        if source_unit == request['unit'] or source_unit.split('/')[0] in ('.incoming', '.delivery_locks'):
            raise ValueError('Unsafe preseed source unit')
        source = os.path.join(root, source_unit)
        current = '/'
        for component in source.split('/')[1:]:
            current = os.path.join(current, component)
            if not os.path.isdir(current) or os.path.islink(current):
                raise ValueError('Preseed source must be an existing regular directory')
        for name in RESERVED:
            if not stat.S_ISREG(os.lstat(os.path.join(source, name)).st_mode):
                raise ValueError('Preseed source metadata is not regular')
        original = read_json(os.path.join(source, '.delivery_manifest.json'))
        receipt = read_json(os.path.join(source, '.delivery_receipt.json'))
        source_sha = digest(original)
        expected = dict(host=request['host'], remote_root=root, unit=source_unit,
                        manifest_sha256=reference['manifest_sha256'])
        if source_sha != reference['manifest_sha256'] or not receipt.get('passed') or any(receipt.get(k) != v for k,v in expected.items()):
            raise ValueError('Preseed source receipt/manifest binding mismatch')
        stats = {}
        for item in original['entries']:
            if item['kind'] == 'file':
                name = relative(item['path'])
                info = os.lstat(os.path.join(source, name))
                stats[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, stat.S_IMODE(info.st_mode))
        counts = verify_tree(source, original)
        if any(receipt.get(k) != v for k,v in counts.items()):
            raise ValueError('Preseed source verification differs from its receipt')
        for item in original['entries']:
            if item['kind'] == 'file':
                candidates.setdefault((item['sha256'], item['size_bytes']), []).append(
                    (os.path.join(source, item['path']), stats[item['path']]))
        verified.append(dict(unit=source_unit, manifest_sha256=source_sha, **counts))
    files = total = 0
    for item in manifest['entries']:
        if item['kind'] != 'file':
            continue
        name = relative(item['path'])
        destination = os.path.join(payload, name)
        if os.path.lexists(destination):
            if not stat.S_ISREG(os.lstat(destination).st_mode):
                raise ValueError('Preseed destination is not a regular partial file')
            continue
        wanted = attributes.get(name)
        if not wanted:
            continue
        for source, expected_stat in candidates.get((item['sha256'], item['size_bytes']), []):
            info = os.lstat(source)
            actual_stat = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, stat.S_IMODE(info.st_mode))
            if not stat.S_ISREG(info.st_mode) or actual_stat != expected_stat:
                raise ValueError('Preseed source changed after complete verification')
            # rsync --archive must have no timestamp/permission reason to mutate
            # the inode shared with the existing immutable raw publication.
            if info.st_mtime_ns != wanted['mtime_ns'] or stat.S_IMODE(info.st_mode) != wanted['mode']:
                continue
            ensure_directory(os.path.dirname(destination))
            os.link(source, destination)
            files += 1
            total += item['size_bytes']
            break
    return dict(files=files, logical_bytes=total, verified_sources=verified)

def main(request):
    root = request['remote_root']
    if (not isinstance(root, str) or not root.startswith('/') or root == '/' or '\\' in root
            or any(ord(c) < 32 for c in root) or any(p in ('', '.', '..') for p in root.split('/')[1:])):
        raise ValueError('Unsafe remote root')
    unit = relative(request['unit'])
    if unit.split('/')[0] in ('.incoming', '.delivery_locks'):
        raise ValueError('Unit collides with publisher control directories')
    manifest = request['manifest']
    manifest_sha = digest(manifest)
    if manifest_sha != request['manifest_sha256']:
        raise ValueError('Unit manifest digest mismatch')
    ensure_directory(root)
    incoming = os.path.join(root, '.incoming')
    locks = os.path.join(root, '.delivery_locks')
    ensure_directory(incoming)
    ensure_directory(locks)
    unit_id = hashlib.sha256(unit.encode('utf-8')).hexdigest()
    stage = os.path.join(incoming, unit_id)
    destination = os.path.join(root, unit)
    binding = dict(host=request['host'], remote_root=root, unit=unit, manifest_sha256=manifest_sha,
                   client_token=request['client_token'])
    with open(os.path.join(locks, unit_id + '.lock'), 'a') as operation_lock:
        fcntl.flock(operation_lock.fileno(), fcntl.LOCK_EX)
        # Check the full destination parent chain before reading or writing it.
        ensure_directory(os.path.dirname(destination))
        if os.path.lexists(destination):
            if os.path.islink(destination) or not os.path.isdir(destination):
                raise ValueError('Destination is not a regular published unit directory')
            existing = read_json(os.path.join(destination, '.delivery_manifest.json'))
            receipt = read_json(os.path.join(destination, '.delivery_receipt.json'))
            if digest(existing) != manifest_sha:
                raise ValueError('Existing destination has a different manifest; refusing overwrite')
            if any(receipt.get(key) != binding[key] for key in ('host', 'remote_root', 'unit', 'manifest_sha256')):
                raise ValueError('Existing receipt is bound to a different destination')
            counts = verify_tree(destination, existing)
            if not receipt.get('passed') or any(receipt.get(key) != value for key, value in counts.items()):
                raise ValueError('Existing publication receipt is inconsistent')
            return dict(passed=True, phase='verified', reused=True, receipt=receipt)
        if request['operation'] not in ('prepare', 'finalize'):
            raise ValueError('Unknown publication operation')
        if os.path.lexists(stage):
            if os.path.islink(stage) or not os.path.isdir(stage):
                raise ValueError('Unsafe incoming stage')
            if read_json(os.path.join(stage, 'binding.json')) != binding:
                raise ValueError('Incoming stage belongs to a different source, destination, or client state')
            if digest(read_json(os.path.join(stage, 'manifest.json'))) != manifest_sha:
                raise ValueError('Incoming manifest changed')
        elif request['operation'] == 'prepare':
            os.mkdir(stage)
            write_json(os.path.join(stage, 'binding.json'), binding)
            write_json(os.path.join(stage, 'manifest.json'), manifest)
        else:
            raise ValueError('No prepared incoming stage exists')
        payload = os.path.join(stage, 'payload')
        if request['operation'] == 'prepare':
            ensure_directory(payload)
            seeded = seed_payload(root, payload, manifest, request)
            return dict(passed=True, phase='staging', reused=False, stage=stage, payload=payload, preseed=seeded)
        if not os.path.isdir(payload) or os.path.islink(payload):
            raise ValueError('Incoming payload is absent or unsafe')
        counts = verify_tree(payload, manifest)
        receipt = dict(schema_version='lunar_delivery_receipt/14.0', passed=True,
            host=request['host'], remote_root=root, unit=unit, manifest_sha256=manifest_sha,
            publication='verified_then_atomic_directory_rename',
            verified_at_utc=datetime.datetime.utcnow().isoformat() + 'Z', **counts)
        original_mode = stat.S_IMODE(os.stat(payload).st_mode)
        os.chmod(payload, original_mode | stat.S_IWUSR)
        try:
            write_json(os.path.join(payload, '.delivery_manifest.json'), manifest, stage)
            write_json(os.path.join(payload, '.delivery_receipt.json'), receipt, stage)
        finally:
            os.chmod(payload, original_mode)
        # Same-filesystem rename: the final unit is never partially visible.
        if os.path.lexists(destination):
            raise ValueError('Destination appeared during publication')
        os.rename(payload, destination)
        write_json(os.path.join(stage, 'published.json'), receipt)
        return dict(passed=True, phase='verified', reused=False, receipt=receipt)

try:
    result = main(json.loads(sys.stdin.buffer.read().decode('utf-8')))
    sys.stdout.buffer.write((json.dumps(result, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
except Exception as error:
    sys.stdout.buffer.write((json.dumps(dict(passed=False, error=type(error).__name__ + ': ' + str(error)), ensure_ascii=False) + '\n').encode('utf-8'))
    sys.exit(1)
'''


class SSHRsyncTransport:
    """Commands use argv plus SSH shell quoting; no local shell interpolation."""
    def __init__(self, host, *, ssh_options=(), ssh_binary='ssh', rsync_binary='rsync'):
        if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host):
            raise DeliveryError('Host must be a safe SSH alias or user@hostname')
        self.host = host
        self.ssh_options = ('-oBatchMode=yes', '-oConnectTimeout=20',
                            '-oServerAliveInterval=30', '-oServerAliveCountMax=2', *tuple(ssh_options))
        self.ssh_binary, self.rsync_binary = ssh_binary, rsync_binary

    def invoke(self, request):
        command = [self.ssh_binary, '-T', *self.ssh_options, self.host,
                   'python3 -c ' + shlex.quote(REMOTE_HELPER)]
        completed = subprocess.run(command, input=canonical_bytes(request), capture_output=True)
        try:
            result = json.loads(completed.stdout.decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as exc:
            raise DeliveryError('Remote verifier did not return valid JSON: ' + completed.stderr.decode('utf-8', 'replace')[-4000:]) from exc
        if completed.returncode or result.get('passed') is not True:
            raise DeliveryError(result.get('error', 'Remote verifier failed'))
        return result

    def transfer(self, source, remote_payload):
        ssh_command = shlex.join([self.ssh_binary, '-T', *self.ssh_options])
        command = [self.rsync_binary, '--archive', '--hard-links', '--no-owner', '--no-group', '--checksum',
            '--partial', '--protect-args', '--timeout=120', '--exclude=REQUEST_STOP', '--exclude=*.tmp',
            '-e', ssh_command, '--', str(source) + '/', self.host + ':' + remote_payload + '/']
        completed = subprocess.run(command, capture_output=True)
        if completed.returncode:
            raise DeliveryError('rsync failed (partial data preserved), exit ' + str(completed.returncode)
                                + ': ' + completed.stderr.decode('utf-8', 'replace')[-4000:])


@contextlib.contextmanager
def _state_lock(path):
    import fcntl
    with path.open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DeliveryError('Another publisher is using this local unit state') from exc
        yield


def _accept_receipt(result, binding, manifest, state):
    receipt = result['receipt']
    wanted = {key: binding[key] for key in ('host', 'remote_root', 'unit', 'manifest_sha256')}
    wanted.update({key: manifest[key] for key in ('entry_count', 'file_count', 'total_file_bytes')})
    if receipt.get('passed') is not True or any(receipt.get(key) != value for key, value in wanted.items()):
        raise DeliveryError('Remote receipt does not match the requested unit and manifest')
    _write_json(state / 'receipt.json', receipt)
    answer = dict(passed=True, reused=bool(result.get('reused')), manifest_sha256=binding['manifest_sha256'],
        remote_destination=binding['remote_root'] + '/' + binding['unit'], host=binding['host'],
        state_dir=str(state), receipt=receipt)
    _write_json(state / 'status.json', dict(answer, phase='verified'))
    return answer


def verify_published_unit(*, state_dir, ssh_options=(), transport=None):
    """Rehash an existing remote unit from retained state, without local payload.

    This is useful after a separate caller has authorized bounded local frame
    retention. It cannot upload, recreate, overwrite, or prune any payload.
    """
    state = Path(state_dir).expanduser().resolve(strict=True)
    with _state_lock(state / 'publish.lock'):
        binding = json.loads((state / 'binding.json').read_text())
        manifest = json.loads((state / 'manifest.json').read_text())
        if manifest_digest(manifest) != binding['manifest_sha256']:
            raise DeliveryError('Retained local manifest no longer matches the transaction')
        transport = transport or SSHRsyncTransport(binding['host'], ssh_options=ssh_options)
        try:
            result = transport.invoke(dict(binding, manifest=manifest, operation='verify'))
            return _accept_receipt(result, binding, manifest, state)
        except Exception as exc:
            _write_json(state / 'status.json', dict(phase='failed', failed_phase='remote_reverify',
                passed=False, manifest_sha256=binding['manifest_sha256'], error=str(exc),
                partial_data_preserved=True))
            raise


def publish_unit(source, *, host, remote_root, unit, state_dir, required_paths=(),
                 ssh_options=(), transport=None, preseed_units=()):
    """Publish one immutable unit exactly once, or explicitly resume its state.

    ``state_dir`` names one transaction and must be outside ``source``. Reusing
    that transaction with different source contents or destination is
    rejected. The transport hook is for isolated CPU integration tests.
    There is one transfer attempt per call; failures never prune either side.
    """
    source = Path(source).expanduser().resolve(strict=True)
    remote_root, unit = _remote_root(remote_root), _relative(unit)
    if unit.split('/')[0] in {'.incoming', '.delivery_locks'}:
        raise DeliveryError('Unit collides with publisher control directories')
    transport = transport or SSHRsyncTransport(host, ssh_options=ssh_options)
    manifest = build_unit_manifest(source, required_paths=required_paths)
    digest = manifest_digest(manifest)
    destination = dict(host=host, remote_root=remote_root, unit=unit)
    seeds = []
    for reference in preseed_units:
        if not isinstance(reference, dict) or set(reference) != {'unit', 'manifest_sha256'}:
            raise DeliveryError('Preseed references require unit and manifest_sha256')
        name = _relative(reference['unit'])
        value = reference['manifest_sha256']
        if (name == unit or name.split('/')[0] in {'.incoming', '.delivery_locks'}
                or not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None):
            raise DeliveryError('Unsafe preseed unit or digest')
        seeds.append(dict(unit=name, manifest_sha256=value))
    seeds.sort(key=lambda value: value['unit'])
    if len(seeds) > 8 or len({value['unit'] for value in seeds}) != len(seeds):
        raise DeliveryError('Duplicate or excessive preseed references')
    state = Path(state_dir).expanduser().resolve()
    if state.is_relative_to(source):
        raise DeliveryError('Publisher state must be outside the immutable source tree')
    state.mkdir(parents=True, exist_ok=True)
    with _state_lock(state / 'publish.lock'):
        binding_path = state / 'binding.json'
        if binding_path.exists():
            binding = json.loads(binding_path.read_text())
            if any(binding.get(key) != value for key, value in dict(destination, manifest_sha256=digest).items()):
                _write_json(state / 'status.json', dict(phase='failed', failed_phase='binding_check',
                    passed=False, manifest_sha256=binding.get('manifest_sha256'),
                    attempted_manifest_sha256=digest, error='Different source contents or destination',
                    partial_data_preserved=True))
                raise DeliveryError('Local transaction is bound to different source contents or destination')
        else:
            binding = dict(destination, manifest_sha256=digest, client_token=uuid.uuid4().hex)
            _write_json(binding_path, binding)
        _write_json(state / 'manifest.json', manifest)
        request = dict(binding, manifest=manifest)
        optimization = state/'preseed_sources.json'
        if optimization.exists():
            previous = json.loads(optimization.read_text())
            if seeds and previous != seeds:
                raise DeliveryError('Preseed references changed for an existing transaction')
            seeds = previous
        elif seeds:
            _write_json(optimization, seeds)
        if seeds:
            request['preseed_units'] = seeds
            request['preseed_target_attributes'] = {
                item['path']: dict(mode=stat.S_IMODE((source/item['path']).stat().st_mode),
                                   mtime_ns=(source/item['path']).stat().st_mtime_ns)
                for item in manifest['entries'] if item['kind'] == 'file'}
        phase = 'prepare'
        try:
            _write_json(state / 'status.json', dict(phase=phase, passed=False, manifest_sha256=digest))
            prepared = transport.invoke(dict(request, operation='prepare'))
            if prepared.get('preseed'):
                _write_json(state/'preseed_result.json', prepared['preseed'])
            if prepared['phase'] == 'verified':
                result = prepared
            else:
                phase = 'transfer'
                _write_json(state / 'status.json', dict(phase=phase, passed=False, manifest_sha256=digest))
                transport.transfer(source, prepared['payload'])
                phase = 'source_recheck'
                if manifest_digest(build_unit_manifest(source, required_paths=required_paths)) != digest:
                    raise DeliveryError('Source changed during transfer; incoming data preserved without publication')
                phase = 'remote_verify_and_publish'
                _write_json(state / 'status.json', dict(phase=phase, passed=False, manifest_sha256=digest))
                result = transport.invoke(dict(request, operation='finalize'))
            if manifest_digest(build_unit_manifest(source, required_paths=required_paths)) != digest:
                raise DeliveryError('Source changed before receipt acceptance; no source files were removed')
            return _accept_receipt(result, binding, manifest, state)
        except Exception as exc:
            _write_json(state / 'status.json', dict(phase='failed', failed_phase=phase, passed=False,
                manifest_sha256=digest, error=str(exc), partial_data_preserved=True))
            raise
