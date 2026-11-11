"""Partition unchanged stride-view rows by prescribed query composition family."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path

from .lunar_query_families_v14 import check_queries, load_protocol, protocol_hash, public_value
from .lunar_training_stride_v14 import jsonl_rows, sha256, write_json

FILES = dict(query_train='query_train_index.jsonl',query_eval='query_eval_index.jsonl',
             physical_val='physical_val_index.jsonl',physical_test='physical_test_index.jsonl')


def _families(base,protocol):
    manifest=json.loads((base/'dataset_manifest.json').read_text())
    lookup,source_files={},{}
    for episode in manifest['episodes']:
        language=base/episode['language']
        query=json.loads((language/'queries.json').read_text())
        actual_protocol=load_protocol(language/'query_protocol.json')
        contract=json.loads((base/episode['trajectory']/'route_contract.json').read_text())
        if (not query['generated'] or protocol_hash(actual_protocol) != protocol_hash(protocol)
                or query.get('query_protocol_sha256') != protocol_hash(protocol)
                or len(query['variants']) != 20 or query['split'] != episode['split']):
            raise ValueError('A physical episode lacks its validated fixed twenty-family query batch')
        value=public_value(query['variants'],protocol)
        check_queries(value,contract,protocol)
        if query.get('generation_method')=='compiled':
            from .lunar_query_compiler_v15 import load_bank,validate_compiled_record
            validate_compiled_record(query,contract,protocol,load_bank(language/'query_bank.json',protocol))
        for variant,spec in zip(query['variants'],protocol['families']):
            identity=variant['language_episode_id']
            if (identity in lookup or variant['physical_trajectory_id'] != episode['episode_id']
                    or variant['split'] != episode['split'] or variant['family_id'] != spec['family_id']
                    or variant['query_partition'] != spec['query_partition']):
                raise ValueError('Query identity/physical split/global family partition mismatch')
            lookup[identity]=dict(variant,layout_seed=episode['layout_seed'])
        for filename in ('queries.json','query_protocol.json','protocol.json'):
            path=language/filename
            source_files[path.relative_to(base).as_posix()]=sha256(path)
        if query.get('generation_method')=='compiled':
            path=language/'query_bank.json'
            source_files[path.relative_to(base).as_posix()]=sha256(path)
    return lookup,source_files


def _destination(row,lookup):
    identity=row['episode_id']
    if identity.endswith(':canonical'):
        return None
    if identity not in lookup:
        raise ValueError('View contains an unknown, unpartitioned query alias')
    family=lookup[identity]
    if (row['physical_trajectory_id'] != family['physical_trajectory_id'] or row['split'] != family['split']
            or row['layout_seed'] != family['layout_seed'] or row['inputs']['instruction'] != family['instruction']):
        raise ValueError('Stride row differs from its fixed query family/physical identity')
    if row['split']=='train':
        return 'query_train' if family['query_partition']=='train' else 'query_eval'
    if family['query_partition']=='train' and row['split'] in ('val','test'):
        return 'physical_'+row['split']
    return None


def build_query_indexes(dense_base,stride_view,output=None,protocol=None):
    base,view=Path(dense_base).resolve(strict=True),Path(stride_view).resolve(strict=True)
    output=Path(output).absolute() if output is not None else view/'query_partitions'
    protocol=load_protocol(protocol)
    view_manifest=json.loads((view/'view_manifest.json').read_text())
    if ((view/view_manifest['base_dataset']['relative_path']).resolve(strict=True) != base
            or view_manifest['observation_stride'] != 5
            or view_manifest['reader_index_sha256'] != sha256(view/'reader_index.jsonl')):
        raise ValueError('Query indexes require the actual hash-bound stride5 view of this dense base')
    lookup,sources=_families(base,protocol)
    output.mkdir(parents=True,exist_ok=False)
    counts={key:0 for key in FILES}
    aliases={key:set() for key in FILES}
    physical={key:set() for key in FILES}
    excluded_canonical=0
    with ExitStack() as stack:
        streams={key:stack.enter_context((output/filename).open('w')) for key,filename in FILES.items()}
        for row in jsonl_rows(view/'reader_index.jsonl'):
            if row['episode_id'].endswith(':canonical'):
                excluded_canonical+=1
            destination=_destination(row,lookup)
            if destination is None:continue
            streams[destination].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            counts[destination]+=1;aliases[destination].add(row['episode_id']);physical[destination].add(row['physical_trajectory_id'])
    manifest=dict(schema_version='lunar_query_generalization_indexes/14.0',complete=True,training_ready=False,
        dense_base_relative_path=os.path.relpath(base,output),stride_view_relative_path=os.path.relpath(view,output),
        dense_manifest_sha256=sha256(base/'dataset_manifest.json'),stride_manifest_sha256=sha256(view/'view_manifest.json'),
        stride_reader_sha256=sha256(view/'reader_index.jsonl'),query_protocol_sha256=protocol_hash(protocol),
        query_source_files_sha256=sources,query_families=lookup,
        training_family_ids=[r['family_id'] for r in protocol['families'] if r['query_partition']=='train'],
        heldout_family_ids=[r['family_id'] for r in protocol['families'] if r['query_partition']=='heldout'],
        physical_split_unchanged=True,query_eval_uses_known_physical_train_routes=True,
        primary_dense_and_stride_readers_unchanged=True,canonical_rows_excluded_from_family_indexes=excluded_canonical,
        row_rule='exact unmodified stride reader rows; family assignment joined by fixed language_episode_id',
        default_training_index=FILES['query_train'],heldout_query_eval_index=FILES['query_eval'],
        files={key:dict(path=filename,sha256=sha256(output/filename),rows=counts[key],
                       language_episodes=len(aliases[key]),physical_episodes=len(physical[key])) for key,filename in FILES.items()})
    write_json(output/'query_protocol.json',protocol)
    write_json(output/'query_partition_manifest.json',manifest)
    report=validate_query_indexes(output)
    write_json(output/'validation.json',report)
    if not report['passed']:
        raise ValueError('Query family indexes failed independent unchanged-row/partition validation')
    return manifest,report


def validate_query_indexes(output,*,dense_base=None,stride_view=None,protocol=None):
    output=Path(output).resolve(strict=True)
    manifest=json.loads((output/'query_partition_manifest.json').read_text())
    base=Path(dense_base).resolve(strict=True) if dense_base is not None else (output/manifest['dense_base_relative_path']).resolve(strict=True)
    view=Path(stride_view).resolve(strict=True) if stride_view is not None else (output/manifest['stride_view_relative_path']).resolve(strict=True)
    protocol=load_protocol(protocol or output/'query_protocol.json')
    errors=[];checks=0;counts={key:0 for key in FILES};seen={key:set() for key in FILES}
    def check(condition,message):
        nonlocal checks
        checks+=1
        if not condition and len(errors)<100:errors.append(message)
    try:
        lookup,sources=_families(base,protocol)
        check(manifest['query_protocol_sha256']==protocol_hash(protocol),'Query protocol changed')
        check(manifest['query_families']==lookup and manifest['query_source_files_sha256']==sources,'Query identity/provenance changed')
        check(not set(manifest['training_family_ids']) & set(manifest['heldout_family_ids']),'Heldout family enters training taxonomy')
        check(manifest['dense_manifest_sha256']==sha256(base/'dataset_manifest.json'),'Dense manifest changed')
        check(manifest['stride_manifest_sha256']==sha256(view/'view_manifest.json'),'Stride manifest changed')
        check(manifest['stride_reader_sha256']==sha256(view/'reader_index.jsonl'),'Source stride rows changed')
        streams={key:iter(jsonl_rows(output/filename)) for key,filename in FILES.items()}
        for row in jsonl_rows(view/'reader_index.jsonl'):
            destination=_destination(row,lookup)
            if destination is None:continue
            actual=next(streams[destination],None)
            check(actual==row,'Query index changed RGB-D, history, native controls, CoT or fixed alias identity')
            check(row['sample_id'] not in seen[destination],'Duplicate query index sample')
            seen[destination].add(row['sample_id']);counts[destination]+=1
            if destination=='query_eval':
                check(row['split']=='train' and lookup[row['episode_id']]['query_partition']=='heldout',
                      'Query eval uses a foreign physical split or trained family')
            if destination=='query_train':
                check(row['split']=='train' and lookup[row['episode_id']]['query_partition']=='train',
                      'Heldout family/physical route leaked into training')
        for key,filename in FILES.items():
            check(next(streams[key],None) is None,'Query index contains unexpected extra rows')
            check(manifest['files'][key]['rows']==counts[key],'Query index count changed')
            check(manifest['files'][key]['sha256']==sha256(output/filename),'Query index SHA changed')
    except (OSError,KeyError,ValueError,TypeError) as exc:
        errors.append(type(exc).__name__+': '+str(exc))
    return dict(schema_version='lunar_query_generalization_validation/14.0',passed=not errors,
                checks=checks,errors=errors,rows=counts,physical_split_unchanged=True,new_API_requests=0)


def load_policy_inputs(output,row):
    from .lunar_training_stride_v14 import load_policy_inputs as load_view
    output=Path(output).resolve(strict=True)
    manifest=json.loads((output/'query_partition_manifest.json').read_text())
    return load_view((output/manifest['stride_view_relative_path']).resolve(strict=True),row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path)
    parser.add_argument('--view',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    if args.validate:
        result=validate_query_indexes(args.output,dense_base=args.base,stride_view=args.view,protocol=args.protocol)
    else:
        if args.base is None or args.view is None:parser.error('--base and --view are required to build')
        _,result=build_query_indexes(args.base,args.view,args.output,args.protocol)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
