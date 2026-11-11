"""Guarded reclamation of only a consumer's SHA-published frame aliases."""
from __future__ import annotations

import json
from pathlib import Path
import re

from .ppio_client_v11 import sha256


FRAME = re.compile(r'^\d{6}_(?:rgb224\.png|rgb256\.png|sensors\.npz|instances\.json)$')


def read(path):
    return json.loads(Path(path).read_text())


def safe_path(root, relative):
    parts = Path(relative).parts
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or '\\' in relative or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError('Unsafe consumer retention path')
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Consumer retention refuses file or ancestor symlinks')
    if not current.resolve().is_relative_to(root):
        raise ValueError('Retention path escapes the owned unit')
    return current


def identity(path):
    value = path.lstat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def proof(manifest_path, receipt_path, expected_sha):
    if sha256(manifest_path) != expected_sha:
        raise ValueError('Retained publication manifest hash changed')
    manifest, receipt = read(manifest_path), read(receipt_path)
    if (receipt.get('passed') is not True or receipt.get('manifest_sha256') != expected_sha
            or any(receipt.get(key) != manifest.get(key) for key in ('entry_count','file_count','total_file_bytes'))):
        raise ValueError('A matching verified publication receipt is required before reclamation')
    return manifest, receipt


def reclaim_owned_frames(unit, *, owned_units_root, unit_id, ordinal, final_manifest_path,
                         final_receipt_path, expected_final_sha256, raw_manifest_path,
                         raw_receipt_path, expected_raw_sha256, critical_sha256,
                         keep_review_units=4):
    """Preflight every deletion before unlinking any path, keeping all CoT pairs.

    Both raw and final receipt bindings are mandatory. Metadata and non-frame
    payloads must still match the final manifest. Missing already-reclaimed
    frame aliases are harmless; a changed/replaced frame aborts the whole pass.
    No source in the producer tree is ever unlinked by this function.
    """
    owned = Path(owned_units_root).resolve(strict=True)
    unit = Path(unit).absolute()
    if unit.is_symlink() or unit.resolve(strict=True) != owned / unit_id or unit.parent.resolve() != owned:
        raise ValueError('Retention only owns one direct consumer unit directory')
    unit = unit.resolve(strict=True)
    final, final_receipt = proof(final_manifest_path, final_receipt_path, expected_final_sha256)
    _, raw_receipt = proof(raw_manifest_path, raw_receipt_path, expected_raw_sha256)
    if (final_receipt.get('unit') != 'units/' + unit_id or raw_receipt.get('unit') != 'raw_inputs/' + unit_id
            or any(final_receipt.get(key) != raw_receipt.get(key) for key in ('host','remote_root'))):
        raise ValueError('Raw and final receipts do not refer to this same published physical unit')
    if ordinal < keep_review_units:
        return dict(phase='retained_review', files_deleted=0, local_frames_evicted=False, kept_native_indices=[])
    by_path = {entry['path']:entry for entry in final['entries']}
    for relative, digest in critical_sha256.items():
        entry = by_path.get(relative)
        if not entry or entry.get('sha256') != digest:
            raise ValueError('Critical metadata differs from its publication manifest')
    # Bind the annotations used to decide which local review pairs to keep.
    for entry in final['entries']:
        relative = Path(entry['path'])
        bulk = 'frames' in relative.parts and FRAME.fullmatch(relative.name)
        path = safe_path(unit, entry['path'])
        if entry['kind'] == 'file' and not bulk:
            if not path.is_file() or path.stat().st_size != entry['size_bytes'] or sha256(path) != entry['sha256']:
                raise ValueError('Published consumer metadata/non-frame payload changed: ' + entry['path'])
        elif entry['kind'] == 'symlink':
            raise ValueError('Reclamation requires regular consumer payload paths')
    base = unit / 'dataset'
    keep, kept_indices = set(), set()
    for episode in read(base/'dataset_manifest.json')['episodes']:
        annotations = base / episode['language'] / 'annotations.jsonl'
        selected = {value['sample_id'] for value in map(json.loads,annotations.read_text().splitlines()) if value.get('cot_valid')}
        trajectory = base / episode['trajectory']
        for frame in map(json.loads,(trajectory/'frame_records.jsonl').read_text().splitlines()):
            if frame['sample_id'] not in selected:
                continue
            for index in (frame['step_index'], max(0,frame['step_index']-1)):
                kept_indices.add(index)
                for path in (trajectory/'frames').glob(f'{index:06d}_*'):
                    safe_path(unit,path.relative_to(unit).as_posix())
                    if path.is_file():
                        value = path.stat()
                        keep.add((value.st_dev,value.st_ino))
    candidates = []
    for entry in final['entries']:
        relative = Path(entry['path'])
        if entry['kind'] != 'file' or 'frames' not in relative.parts or not FRAME.fullmatch(relative.name):
            continue
        path = safe_path(unit,entry['path'])
        if not path.exists():
            continue
        before = identity(path)
        if before[:2] in keep:
            continue
        if not path.is_file() or before[2] != entry['size_bytes'] or sha256(path) != entry['sha256'] or identity(path) != before:
            raise ValueError('Current unuploaded/changed frame preserved; no candidates removed: ' + entry['path'])
        candidates.append((path,before))
    # Re-stat the entire preflight set before the first deletion. Link counts
    # are deliberately excluded: the producer can unlink its own aliases.
    for path,before in candidates:
        safe_path(unit,path.relative_to(unit).as_posix())
        if identity(path) != before:
            raise ValueError('A proposed frame changed after preflight; no candidates removed')
    for path,_ in candidates:
        path.unlink()
    return dict(phase='complete', files_deleted=len(candidates), local_frames_evicted=True,
                kept_native_indices=sorted(kept_indices), raw_and_final_sha_preceded_reclamation=True)
