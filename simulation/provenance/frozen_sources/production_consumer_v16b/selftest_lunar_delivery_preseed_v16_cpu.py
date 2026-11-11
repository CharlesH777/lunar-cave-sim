"""Exercise exact remote preseed helper locally; no network/GPU/provider calls."""
from pathlib import Path
import json
import os
import shutil
import tempfile
from unittest.mock import patch

from cave.lunar_dataset_delivery_v14 import REMOTE_HELPER, publish_unit, verify_published_unit
from selftest_lunar_dataset_delivery_v14_cpu import LocalTransport, must_fail


def main():
    with tempfile.TemporaryDirectory(prefix='lunar_preseed_') as temporary:
        root = Path(temporary)
        # Deterministically inject another unit creating the shared parent
        # between lexists and mkdir; regular dirs pass, symlinks do not.
        namespace = {}
        exec(REMOTE_HELPER.split('\ntry:\n    result = main(')[0], namespace)
        mkdir = os.mkdir
        race = root/'concurrent_parent'
        def concurrent_directory(path):
            if str(path) == str(race):
                mkdir(path)
            return mkdir(path)
        with patch('os.mkdir', concurrent_directory):
            namespace['ensure_directory'](str(race/'child'))
        assert (race/'child').is_dir()
        bad_race = root/'concurrent_symlink'
        def concurrent_symlink(path):
            if str(path) == str(bad_race):
                os.symlink(race, path)
            return mkdir(path)
        with patch('os.mkdir', concurrent_symlink):
            must_fail(lambda: namespace['ensure_directory'](str(bad_race/'child')), 'symlink')
        raw = root/'raw'
        (raw/'trajectory').mkdir(parents=True)
        frame = raw/'trajectory/frame.bin'
        frame.write_bytes(bytes(range(256))*4096)
        (raw/'metadata.json').write_text('{"complete":true}\n')
        remote = root/'remote'
        def publish(source, unit, state, **kwargs):
            return publish_unit(source, host='fixture', remote_root=str(remote),
                unit=unit, state_dir=root/state, transport=LocalTransport(), **kwargs)
        raw_receipt = publish(raw, 'raw_inputs/demo', 'raw_state')
        ref = dict(unit='raw_inputs/demo', manifest_sha256=raw_receipt['manifest_sha256'])
        remote_frame = remote/'raw_inputs/demo/trajectory/frame.bin'
        before = remote_frame.stat()
        final = root/'final'
        (final/'dataset/frames').mkdir(parents=True)
        local_final = final/'dataset/frames/renamed.bin'
        os.link(frame, local_final)
        (final/'validation.json').write_text('{"passed":true}\n')
        receipt = publish(final, 'units/demo', 'final_state', preseed_units=[ref])
        linked = remote/'units/demo/dataset/frames/renamed.bin'
        assert linked.stat().st_ino == remote_frame.stat().st_ino
        assert (before.st_mode, before.st_mtime_ns, before.st_size) == (
            remote_frame.stat().st_mode, remote_frame.stat().st_mtime_ns, remote_frame.stat().st_size)
        seeded = json.loads((root/'final_state/preseed_result.json').read_text())
        assert seeded['files'] == 1 and seeded['logical_bytes'] == frame.stat().st_size
        assert receipt['passed'] and verify_published_unit(state_dir=root/'final_state', transport=LocalTransport())['passed']
        # Timestamp mismatch must transfer normally and preserve the raw inode attrs.
        separate = root/'different_mtime'
        separate.mkdir()
        copied = separate/'frame.bin'
        shutil.copy2(frame, copied)
        os.utime(copied, ns=(before.st_atime_ns, before.st_mtime_ns+2_000_000_000))
        publish(separate, 'units/mtime', 'mtime_state', preseed_units=[ref])
        assert (remote/'units/mtime/frame.bin').stat().st_ino != before.st_ino
        assert remote_frame.stat().st_mtime_ns == before.st_mtime_ns
        assert json.loads((root/'mtime_state/preseed_result.json').read_text())['files'] == 0
        # Ref validation and source full integrity happen before transfer.
        invalid = [dict(unit='../outside', manifest_sha256=ref['manifest_sha256']),
                   dict(unit='units/new', manifest_sha256=ref['manifest_sha256']),
                   dict(unit=ref['unit'], manifest_sha256='invalid')]
        for i, bad in enumerate(invalid):
            must_fail(lambda bad=bad, i=i: publish(final, 'units/new', 'invalid_'+str(i), preseed_units=[bad]))
        must_fail(lambda: publish(final, 'units/dup', 'duplicate', preseed_units=[ref, ref]), 'Duplicate')
        must_fail(lambda: publish(final, 'units/wrong', 'wrong_digest', preseed_units=[dict(ref, manifest_sha256='0'*64)]), 'binding')
        remote_frame.write_bytes(b'X'*before.st_size)
        must_fail(lambda: publish(final, 'units/corrupt', 'corrupt_state', preseed_units=[ref]), 'SHA/size')
        must_fail(lambda: verify_published_unit(state_dir=root/'final_state', transport=LocalTransport()), 'SHA/size')
        assert not (remote/'units/corrupt').exists()
    print('PASS: concurrent directory race/symlink guard, preseed reuse/rename, immutable attrs, timestamp skip, references, source/final SHA guards')


if __name__ == '__main__':
    main()
