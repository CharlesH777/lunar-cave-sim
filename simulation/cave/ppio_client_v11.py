"""Portable one-shot PPIO client: public responses only, content-addressed cache.

No provider internal reasoning, request headers, credentials or image data URLs
are persisted. Failed requests are not retried automatically.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import urllib.error
import urllib.request

ENDPOINT = 'https://api.ppio.com/openai/v1/chat/completions'
PRICES = {'qwen/qwen3.8-27b': (3., 12.), 'qwen/qwen3.8-max': (12., 36.)}


def write_json(path, value, *, durable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
    temporary.replace(path)
    if durable:
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_key(env_file=None):
    """Parse a dotenv subset without executing shell code; never return extra vars."""
    settings = {}
    if env_file:
        for line in Path(env_file).expanduser().read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'): continue
            if line.startswith('export '): line = line[7:].lstrip()
            name, separator, value = line.partition('=')
            if not separator or name.strip() not in ('PPIO_API_KEY', 'ANTHROPIC_API_KEY'): continue
            parts = shlex.split(value, comments=True, posix=True)
            if len(parts) != 1: raise ValueError('Invalid PPIO dotenv value')
            settings[name.strip()] = parts[0]
    for name in ('PPIO_API_KEY', 'ANTHROPIC_API_KEY'):
        if os.environ.get(name): return os.environ[name]
    for name in ('PPIO_API_KEY', 'ANTHROPIC_API_KEY'):
        if settings.get(name): return settings[name]
    raise RuntimeError('Set PPIO_API_KEY or pass --ppio-env; credentials are never bundled')


class BudgetExceeded(RuntimeError): pass


class UncertainRequest(RuntimeError):
    def __init__(self, fingerprint):
        super().__init__('Interrupted request may have been billed; this fingerprint was skipped')
        self.fingerprint = fingerprint


def cache_record(path, fingerprint):
    try:
        row = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return row if isinstance(row, dict) and row.get('request_sha256') == fingerprint else None


def valid_charge(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


class PPIOClient:
    def __init__(self, cache_dir, budget_cny=2., env_file=None, endpoint=ENDPOINT,
                 prices=None, transport=None, request_timeout=120):
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.budget = float(budget_cny); self.env_file = env_file
        if not endpoint.startswith('https://'): raise ValueError('PPIO endpoint must use HTTPS')
        self.endpoint = endpoint; self.prices = dict(PRICES, **(prices or {}))
        self.transport = transport; self.timeout = request_timeout
        if self.budget <= 0: raise ValueError('Cost cap must be positive')

    def _saved_response(self, fingerprint):
        row = cache_record(self.cache/(fingerprint+'.response.json'), fingerprint)
        if row is None or not valid_charge(row.get('estimated_cny')) or not isinstance(row.get('content'), str):
            return None
        usage = row.get('usage')
        if not isinstance(usage, dict) or any(not isinstance(usage.get(key, 0), int) or usage.get(key, 0) < 0
                                              for key in ('prompt_tokens', 'completion_tokens')):
            return None
        return row

    def _unresolved_reserve(self, fingerprint):
        candidates = []
        for suffix, field in (('pending', 'reserved_cny'), ('request', 'reserved_maximum_cny'),
                              ('recovered_request', 'reserved_maximum_cny')):
            record = cache_record(self.cache/(fingerprint+f'.{suffix}.json'), fingerprint)
            if record and valid_charge(record.get(field)):
                candidates.append(record[field])
        if not candidates:
            raise RuntimeError('No charge bound for interrupted PPIO fingerprint '+fingerprint)
        return max(candidates)

    def usage(self):
        totals = dict(prompt_tokens=0, completion_tokens=0, estimated_cny=0., requests=0,
                      unresolved_requests=0, unresolved_reserved_cny=0.)
        fingerprints = {path.name.split('.', 1)[0] for suffix in ('response', 'pending', 'error')
                        for path in self.cache.glob(f'*.{suffix}.json')}
        for fingerprint in sorted(fingerprints):
            row = self._saved_response(fingerprint)
            if row is None:
                totals['unresolved_requests'] += 1
                totals['unresolved_reserved_cny'] += self._unresolved_reserve(fingerprint)
                continue
            totals['prompt_tokens'] += int(row['usage'].get('prompt_tokens', 0))
            totals['completion_tokens'] += int(row['usage'].get('completion_tokens', 0))
            totals['estimated_cny'] += float(row['estimated_cny']); totals['requests'] += 1
        totals['committed_cny'] = totals['estimated_cny']+totals['unresolved_reserved_cny']
        return totals

    def request_audit(self, prompt, schema, model, images, max_tokens=700):
        images = [Path(path).resolve(strict=True) for path in images]
        if len(images) > 2: raise ValueError('Only current RGB and one actual past RGB are allowed')
        if model not in self.prices: raise ValueError('Configure explicit prices before using a different model')
        if max_tokens < 1: raise ValueError('max_tokens must be positive')
        options = dict(model=model, max_tokens=max_tokens, temperature=.2, enable_thinking=False)
        identity = dict(endpoint=self.endpoint, prompt=prompt, schema=schema, options=options,
                        image_sha256=[sha256(path) for path in images], protocol='lunar_ppio_public/11.0')
        # Input is bounded conservatively by UTF-8 bytes plus 16k tokens/image.
        # Refuse oversized messages before HTTP, reserve maximum output tokens.
        input_bound = len(prompt.encode()) + len(json.dumps(schema).encode()) + 16384*len(images) + 512
        if input_bound > 60000: raise ValueError('Annotation prompt exceeds conservative 60k input-token bound')
        ip, op = self.prices[model]
        return dict(identity, request_sha256=object_hash(identity), input_token_upper_bound=input_bound,
                    reserved_maximum_cny=(input_bound*ip + max_tokens*op)/1_000_000, automatic_retries=False)

    def complete(self, prompt, schema, model='qwen/qwen3.8-27b', images=(), max_tokens=700):
        audit = self.request_audit(prompt, schema, model, images, max_tokens)
        # Serialize the budget check and request in each cache. A dead process
        # releases its lock, while its durable reservation stays on disk.
        with (self.cache/'.requests.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._complete(audit, images)

    def _complete(self, audit, images):
        fingerprint = audit['request_sha256']; response_path = self.cache/(fingerprint+'.response.json')
        saved = self._saved_response(fingerprint)
        if saved is not None:
            return saved, True
        pending = self.cache/(fingerprint+'.pending.json')
        if response_path.exists() or pending.exists() or (self.cache/(fingerprint+'.error.json')).exists():
            raise UncertainRequest(fingerprint)
        reserve = audit['reserved_maximum_cny']
        current_usage = self.usage()
        if current_usage['committed_cny'] + reserve > self.budget + 1e-12:
            raise BudgetExceeded('PPIO cost cap would be exceeded by the next request reservation')
        # A missing local key is preflight failure, not a possibly billed call.
        key = None if self.transport else read_key(self.env_file)
        content = [dict(type='text', text=audit['prompt'])]
        for path in map(Path, images):
            mime = 'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
            content.append(dict(type='image_url', image_url=dict(
                url='data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode(), detail='high')))
        payload = dict(**audit['options'], response_format=dict(type='json_schema', json_schema=dict(
            name='lunar_navigation_annotation', strict=True, schema=audit['schema'])),
            messages=[dict(role='user', content=content)])
        write_json(self.cache/(fingerprint+'.request.json'), audit, durable=True)
        write_json(pending, dict(request_sha256=fingerprint, state='request_started', reserved_cny=reserve), durable=True)
        model = audit['options']['model']; ip, op = self.prices[model]
        try:
            if self.transport:
                response = self.transport(payload)
            else:
                request = urllib.request.Request(self.endpoint, json.dumps(payload).encode(),
                    headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=self.timeout) as stream:
                    response = json.load(stream)
            choice = response['choices'][0]; message = choice['message']
            public = message.get('content') or ''
            if not isinstance(public, str): raise ValueError('Provider public content is not a string')
            usage = response.get('usage') or {}
            usage = {k: max(0, int(usage.get(k, 0))) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            usage_missing = not usage['prompt_tokens']
            cost = reserve if usage_missing else (usage['prompt_tokens']*ip+usage['completion_tokens']*op)/1_000_000
            saved = dict(request_sha256=fingerprint, requested_model=model, model=response.get('model'),
                request_id=response.get('id'), content=public, finish_reason=choice.get('finish_reason'),
                usage=usage, usage_missing=usage_missing, estimated_cny=cost,
                price_cny_per_million=dict(input=ip, output=op),
                created_utc=datetime.now(timezone.utc).isoformat())
            write_json(response_path, saved, durable=True)
            pending.unlink()
        except BaseException as exc:
            # No exception body, provider body, headers or key are logged.
            write_json(self.cache/(fingerprint+'.error.json'), dict(request_sha256=fingerprint,
                error_type=type(exc).__name__, http_status=getattr(exc, 'code', None),
                automatic_retry=False, may_have_been_billed=True))
            raise RuntimeError('PPIO request failed; sanitized cache error saved; automatic retry disabled') from None
        if self.usage()['committed_cny'] > self.budget+1e-12:
            raise BudgetExceeded('Provider reported usage above the reserved estimate; budget exhausted')
        return saved, False


def parse_public(saved):
    if saved.get('finish_reason') != 'stop': raise ValueError('Incomplete public structured response')
    value = json.loads(saved['content']); unwrapped = False
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]; unwrapped = True
    if not isinstance(value, dict): raise ValueError('Expected a JSON object')
    return value, unwrapped
