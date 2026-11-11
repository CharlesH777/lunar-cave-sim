"""Grounded query instantiation from an audited provider phrase bank.

Compiled instructions are not represented as fresh provider responses. Their
movement phrases have explicit provider-span or safe-phrase authorship, while
logical forms come from the fixed protocol and actual semantic route bindings.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .lunar_query_composition_v15 import (
    allowed_logic,check_queries,compact,load_protocol,operator_context,parse_public_response,protocol_hash,
)
from .lunar_route_language_v13 import OBJECT_NAMES,object_digest

VERSION='lunar_grounded_query_compiler/1.0'
BANK_SCHEMA='lunar_query_template_bank/15.1'

ENTRY_CANDIDATES=['按提示进入洞穴','进洞','驶入洞穴内部','进入洞穴','沿提示入洞','按提示进洞','进洞行驶','驶入洞穴']
FORK_CANDIDATES=['在第一个分叉向{side}转','遇首个岔口往{side}拐','到第一处分叉选{side}侧',
    '首个分叉处{side}转','遇第一个岔路向{side}','在第一分叉转向{side}边','首个分叉走{side}侧',
    '到第一个岔口{side}转','遇首处分叉向{side}转','于第一分叉选{side}','首个岔路向{side}拐','在第一分叉处{side}转']
WAYPOINT_CANDIDATES=['看到{landmark}之后继续前进','路过{landmark}接着走','经过{landmark}后保持前行',
    '见到{landmark}仍向前开','看到{landmark}后继续行进','途经{landmark}后往前开',
    '经过{landmark}后保持移动','见{landmark}后继续前行','经过{landmark}后继续走']
SAFE_ENDPOINTS=['前往任务目标附近','接着走向任务目标','继续向任务目标前进','随后前往终点',
    '继续沿路前往任务目标','接近所指的任务目标','再向任务目标所在处前进','沿当前路线前往终点',
    '继续去任务目标附近','接着前往终点附近','再去任务目标附近','继续前往任务目标所在位置',
    '朝任务目标继续行进','再继续走向终点','沿着路线接近任务目标','随后去终点附近',
    '继续前进至任务目标附近','再往任务目标所在的位置走','继续朝终点行进','前往所指的任务目标']


def build_bank(saved,source_contract,protocol,*,source_response_file_sha256):
    protocol=load_protocol(protocol)
    value,_=parse_public_response(saved)
    check_queries(value,source_contract,protocol)
    context=operator_context(source_contract)
    side='左' if context['fork_side']=='left' else '右'
    slot_values=dict(side=side,landmark=context['waypoint_name'])
    cores=[next(segment['text'] for segment in row['segments'] if segment['kind']=='route_core') for row in value['queries']]
    groups={}
    for kind,candidates in [('entry',ENTRY_CANDIDATES),('fork',FORK_CANDIDATES),('waypoint',WAYPOINT_CANDIDATES)]:
        rows=[]
        for template in candidates:
            literal=template.format(**slot_values)
            match=next(((index,core.index(literal)) for index,core in enumerate(cores) if literal in core),None)
            if match is None:continue
            index,start=match
            rows.append(dict(template_id=f'{kind}.{len(rows):02d}',text=template,
                authorship='provider_authored_movement_fragment',
                source=dict(query_index=index+1,family_id=value['queries'][index]['family_id'],
                            role='route_core',span=[start,start+len(literal)],literal=literal)))
        if len(rows)<2:raise ValueError('Approved provider source lacks sufficient '+kind+' phrases')
        groups[kind]=rows
    endpoints=[];seen=set()
    for index,core in enumerate(cores):
        parts=[part for part in re.split(r'[，。；]',core) if part]
        phrase=parts[-1]
        if phrase in seen or not re.search(r'任务目标|终点|目的地',phrase):continue
        if any(name in phrase for name in OBJECT_NAMES.values()):continue
        seen.add(phrase);start=core.rfind(phrase)
        endpoints.append(dict(template_id=f'endpoint.{len(endpoints):02d}',text=phrase,
            authorship='provider_authored_movement_fragment',source=dict(query_index=index+1,
            family_id=value['queries'][index]['family_id'],role='route_core',span=[start,start+len(phrase)],literal=phrase)))
    for phrase in SAFE_ENDPOINTS:
        if phrase in seen:continue
        seen.add(phrase)
        endpoints.append(dict(template_id=f'endpoint.{len(endpoints):02d}',text=phrase,
            authorship='protocol_safe_anonymous_motion_phrase',
            source=dict(author='grounded compiler phrase bank',scope='anonymous endpoint motion only; no new route event or STOP rule')))
    if len(endpoints)<20:raise ValueError('Compiler needs twenty distinct anonymous endpoint phrases')
    groups['endpoint']=endpoints
    return dict(schema_version=BANK_SCHEMA,compiler_version=VERSION,lexical_seed=0,
        query_protocol_sha256=protocol_hash(protocol),phrases=groups,
        source_provider=dict(request_sha256=saved['request_sha256'],requested_model=saved['requested_model'],
            original_response_file_sha256=source_response_file_sha256,public_content=saved['content'],finish_reason=saved['finish_reason'],
            public_value_sha256=object_digest(value),source_route_contract=deepcopy(source_contract)),
        logic_authorship='protocol-defined truth templates returned verbatim and validated in the same provider response',
        actor_authorship='provider-authored phrase templates plus explicit safe phrases; grounded deterministic slot instantiation, not fresh provider output')


def bank_hash(bank):
    return object_digest(bank)


def load_bank(value,protocol):
    bank=json.loads(Path(value).read_text()) if isinstance(value,(str,Path)) else deepcopy(value)
    protocol=load_protocol(protocol)
    if (bank.get('schema_version')!=BANK_SCHEMA or bank.get('compiler_version')!=VERSION
            or type(bank.get('lexical_seed')) is not int or bank.get('query_protocol_sha256')!=protocol_hash(protocol)
            or set(bank.get('phrases',{}))!={'entry','fork','waypoint','endpoint'}):
        raise ValueError('Invalid grounded query bank/version/protocol')
    source=bank['source_provider']
    if not re.fullmatch('[0-9a-f]{64}',source['request_sha256']):raise ValueError('Invalid source provider identity')
    value,_=parse_public_response(dict(content=source['public_content'],finish_reason=source['finish_reason']))
    if object_digest(value)!=source['public_value_sha256']:raise ValueError('Provider source text changed')
    check_queries(value,source['source_route_contract'],protocol)
    context=operator_context(source['source_route_contract'])
    bindings=dict(side='左' if context['fork_side']=='left' else '右',landmark=context['waypoint_name'])
    ids=set()
    for kind,rows in bank['phrases'].items():
        if len(rows)<(20 if kind=='endpoint' else 2):raise ValueError('Insufficient compiler phrase diversity')
        for row in rows:
            if row['template_id'] in ids or not row['text']:raise ValueError('Duplicate/empty compiler phrase')
            ids.add(row['template_id'])
            slots=set(re.findall(r'{([^{}]+)}',row['text']))
            if slots!=({'side'} if kind=='fork' else {'landmark'} if kind=='waypoint' else set()):
                raise ValueError('Unexpected typed slot in '+kind)
            if row['authorship']=='provider_authored_movement_fragment':
                witness=row['source'];record=value['queries'][witness['query_index']-1]
                core=next(s['text'] for s in record['segments'] if s['kind']=='route_core')
                start,end=witness['span'];literal=row['text'].format(**bindings)
                if (record['family_id']!=witness['family_id'] or core[start:end]!=literal or witness['literal']!=literal):
                    raise ValueError('Phrase no longer matches its exact provider source span')
            elif (kind!='endpoint' or row['authorship']!='protocol_safe_anonymous_motion_phrase'
                  or row['text'] not in SAFE_ENDPOINTS):
                raise ValueError('Unapproved non-provider phrase')
    if len({compact(row['text']) for row in bank['phrases']['endpoint']})<20:
        raise ValueError('Anonymous endpoint bank does not guarantee twenty unique cores')
    return bank


def compile_queries(contract,protocol,bank):
    protocol=load_protocol(protocol);bank=load_bank(bank,protocol)
    context=operator_context(contract)
    events=[event for event in contract['events'] if event['type']!='goal_stop']
    if any(event['type'] not in ('enter_cave','first_fork','landmark_continue') for event in events):
        raise ValueError('Unknown semantic route event cannot be compiled')
    if len([e for e in events if e['type']=='first_fork'])>1:
        raise ValueError('Protocol15 supports one actual first-fork cue')
    if any(e['side'] not in ('left','right') for e in events if e['type']=='first_fork'):
        raise ValueError('An unknown branch choice must not be rendered as a direction')
    if any(e['id']==contract['goal']['id'] for e in events if e['type']=='landmark_continue'):
        raise ValueError('Goal cannot be inserted as a nonterminal waypoint')
    if [e['step_index'] for e in events]!=sorted(e['step_index'] for e in events):
        raise ValueError('Route events must retain actual chronological order')
    seed_identity=dict(lexical_seed=bank['lexical_seed'],physical_trajectory_id=contract['physical_trajectory_id'],
                       actual_route_contract_sha256=object_digest(contract),query_bank_sha256=bank_hash(bank))
    seed_key=object_digest(seed_identity)
    rank=lambda text:hashlib.sha256((seed_key+'|'+text).encode()).hexdigest()
    endpoints=sorted(bank['phrases']['endpoint'],key=lambda row:rank('endpoint|'+row['template_id']))
    rows=[];traces=[]
    for index,spec in enumerate(protocol['families']):
        clauses=[];used=[]
        for event_index,event in enumerate(events):
            kind={'enter_cave':'entry','first_fork':'fork','landmark_continue':'waypoint'}[event['type']]
            choices=bank['phrases'][kind]
            choice=choices[int(rank(f'{spec["family_id"]}|{event_index}|{kind}'),16)%len(choices)];slots={}
            if kind=='fork':slots['side']='左' if event['side']=='left' else '右'
            if kind=='waypoint':slots['landmark']=OBJECT_NAMES[event['id']]
            clauses.append(choice['text'].format(**slots));used.append(dict(template_id=choice['template_id'],event_type=event['type'],slots=slots))
        endpoint=endpoints[index];clauses.append(endpoint['text']);used.append(dict(template_id=endpoint['template_id'],event_type='anonymous_endpoint',slots={}))
        core=dict(kind='route_core',text='，'.join(clauses)+'。')
        logic=dict(kind='logic',operator_ids=spec['operator_ids'],text=allowed_logic(spec,context)[0])
        parts=dict(route_core=core,logic=logic)
        rows.append(dict(family_id=spec['family_id'],segments=[parts[k] for k in spec['segment_order']]))
        traces.append(dict(family_id=spec['family_id'],movement_templates=used,operator_ids=spec['operator_ids'],
                           logical_binding=deepcopy(context)))
    public=dict(queries=rows)
    check_queries(public,contract,protocol)  # Compile failures happen before any provider request.
    code_path=Path(__file__)
    provenance=dict(schema_version='lunar_compiled_query_provenance/1.0',generator=VERSION,
        generation_method='grounded_template_instantiation',fresh_provider_output=False,new_query_HTTP_requests=0,
        query_bank_sha256=bank_hash(bank),query_protocol_sha256=protocol_hash(protocol),
        compiler_code_sha256=hashlib.sha256(code_path.read_bytes()).hexdigest(),
        source_provider_template_request_sha256=bank['source_provider']['request_sha256'],
        source_provider_template_response_sha256=bank['source_provider']['original_response_file_sha256'],
        authorship=bank['actor_authorship'],seed_identity=seed_identity,slot_binding_sha256=object_digest(context),
        family_trace=traces)
    return public,provenance


def validate_compiled_record(record,contract,protocol,bank):
    from .lunar_query_composition_v15 import public_value
    expected,provenance=compile_queries(contract,protocol,bank)
    if (record.get('generation_method')!='compiled' or record.get('provider_generated_queries') is not False
            or public_value(record['variants'],protocol)!=expected):
        raise ValueError('Compiled record is not the exact grounded bank instantiation')
    actual=record['provenance']
    for key,value in provenance.items():
        if key=='compiler_code_sha256':continue  # Historical source hash; exact recompilation checks behavior.
        if actual.get(key)!=value:raise ValueError('Compiled authorship/slot provenance mismatch: '+key)
    if not re.fullmatch('[0-9a-f]{64}',actual.get('compiler_code_sha256','')):
        raise ValueError('Compiled generation-source SHA is absent')
    for row,trace in zip(record['variants'],provenance['family_trace']):
        if (row.get('generation_method')!='grounded_template_instantiation' or row.get('compiler_trace')!=trace
                or row.get('query_bank_sha256')!=provenance['query_bank_sha256']):
            raise ValueError('A compiled variant lost its per-family authored template/slot binding')
    return True
