"""A catalog over immutable self-contained units, preserving every native row."""
from __future__ import annotations
import json
from pathlib import Path

from .lunar_training_stride_v14 import jsonl_rows, sha256, write_json
from .lunar_query_indexes_v14 import FILES

QUERY_ROOT='query_partitions'
CRITICAL=('dataset/dataset_manifest.json','dataset/dataset_validation.json','TRAINING_INPUTS.json',
    'training_stride5/view_manifest.json','training_stride5/validation.json',
    QUERY_ROOT+'/query_partition_manifest.json',QUERY_ROOT+'/validation.json')


def safe_child(root,relative):
    if (not isinstance(relative,str) or not relative or Path(relative).is_absolute()
            or any(part in ('','..','.') for part in relative.split('/'))):
        raise ValueError('Unsafe catalog reference')
    path=(Path(root)/relative).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError('Catalog reference escapes its batch')
    return path


def summarize_unit(unit,*,expected_seed,expected_target,expected_split):
    unit=Path(unit).resolve(strict=True)
    for filename in CRITICAL:
        safe_child(unit,filename)
    base=unit/'dataset'
    dense=json.loads((base/'dataset_manifest.json').read_text())
    query=json.loads((unit/QUERY_ROOT/'query_partition_manifest.json').read_text())
    training=json.loads((unit/'TRAINING_INPUTS.json').read_text())
    status=json.loads((unit/'status.json').read_text())
    if status.get('phase')!='complete' or not status.get('passed') or not dense.get('complete') or len(dense['episodes'])!=1:
        raise ValueError('A unit must contain one completed physical episode')
    episode=dense['episodes'][0]
    if (episode['layout_seed'],episode['target_id'],episode['split'])!=(expected_seed,expected_target,expected_split):
        raise ValueError('Physical unit identity/split differs from its planned layout')
    if (not query.get('complete') or query.get('physical_split_unchanged') is not True
            or query.get('query_eval_uses_known_physical_train_routes') is not True
            or query['dense_manifest_sha256']!=sha256(base/'dataset_manifest.json')
            or training.get('observation_stride')!=5 or training.get('native_action_dt_s')!=.2
            or training['default_training_index']!=QUERY_ROOT+'/'+FILES['query_train']):
        raise ValueError('Unit does not have the exact stride-five/native-action/query-family contract')
    for filename in ('dataset/dataset_validation.json','training_stride5/validation.json',QUERY_ROOT+'/validation.json'):
        if json.loads((unit/filename).read_text()).get('passed') is not True:
            raise ValueError('A unit semantic/stride/query check failed')
    training_families=set(query['training_family_ids']); heldout=set(query['heldout_family_ids'])
    if not training_families or not heldout or training_families&heldout:
        raise ValueError('The global query taxonomy is not disjoint and nonempty')
    aliases=query['query_families']
    if len(aliases)!=len(training_families)+len(heldout):
        raise ValueError('One prescribed query alias per global family is required')
    for alias in aliases.values():
        family=alias['family_id']
        if (alias['layout_seed']!=expected_seed or alias['split']!=expected_split
                or alias['physical_trajectory_id']!=episode['episode_id']
                or alias['query_partition']!=('train' if family in training_families else 'heldout')
                or family not in training_families|heldout):
            raise ValueError('A query alias leaks its physical or query partition')
    files={}
    if set(query['files'])!=set(FILES):raise ValueError('Missing query/physical partition index')
    for partition,item in query['files'].items():
        if partition not in FILES or item['path']!=FILES[partition]:
            raise ValueError('Unknown partition/index path')
        relative=QUERY_ROOT+'/'+item['path']
        if sha256(safe_child(unit,relative))!=item['sha256']:
            raise ValueError('A query index differs from its validation manifest')
        if expected_split!='train' and partition in ('query_train','query_eval') and item['rows']:
            raise ValueError('Physical heldout routes entered query training/evaluation')
        if partition in ('physical_val','physical_test') and partition!='physical_'+expected_split and item['rows']:
            raise ValueError('Physical evaluation split leaked')
        expected_aliases=(len(training_families) if partition=='query_train' else len(heldout)) if expected_split=='train' and partition in ('query_train','query_eval') else 0
        if expected_split in ('val','test') and partition=='physical_'+expected_split:expected_aliases=len(training_families)
        if item['language_episodes']!=expected_aliases or bool(item['rows'])!=bool(expected_aliases):
            raise ValueError('Missing or unexpected query aliases in a partition')
        files[partition]=dict(path=relative,**{k:v for k,v in item.items() if k!='path'})
    required=list(CRITICAL)+[item['path'] for item in files.values()]
    for filename in ('metadata.json','frame_records.jsonl','supervision.npz','teacher_only.npz'):
        required.append('dataset/'+episode['trajectory']+'/'+filename)
    metadata=json.loads((base/episode['trajectory']/'metadata.json').read_text())
    if (not metadata.get('terminal_success') or metadata.get('motion_controller')!='smooth_v13'
            or metadata.get('runtime_collision_mode')!='actual_geometry'
            or metadata.get('canonical_stop_radius_m')!=1.
            or metadata.get('stationary_turns_allowed') is not False
            or metadata.get('body_wall_clearance',{}).get('required_gap_m')!=1.
            or metadata.get('body_wall_clearance',{}).get('expanded_air_envelope_gap_m',0.)<1.05
            or metadata.get('body_wall_clearance',{}).get('expanded_air_envelope_passed') is not True):
        raise ValueError('Missing physical V13-family actual-geometry/wall acceptance')
    return dict(episode_id=episode['episode_id'],layout_seed=expected_seed,target_id=expected_target,
        split=expected_split,query_indexes=files,query_protocol_sha256=query['query_protocol_sha256'],
        training_family_ids=sorted(training_families),heldout_family_ids=sorted(heldout),
        physical_frames=metadata['captured_frame_count'],required_paths=required,
        critical_sha256={path:sha256(unit/path) for path in required},
        native_action_dt_s=.2,observation_stride=5,training_ready=False)


def build_catalog(path,entries,*,requested_units,complete=False):
    entries=sorted(entries,key=lambda row:row['unit_id'])
    seeds={}; seen=set(); protocol=None
    totals={key:dict(rows=0,language_episodes=0,physical_episodes=0) for key in FILES}
    for entry in entries:
        if entry['unit_id'] in seen or entry['delivery']['passed'] is not True:
            raise ValueError('Catalog units must be unique and SHA-verified')
        seen.add(entry['unit_id']); seed,split=entry['layout_seed'],entry['split']
        if seed in seeds and seeds[seed]!=split:
            raise ValueError('The same physical layout leaked across splits')
        seeds[seed]=split
        if protocol is not None and protocol!=entry['query_protocol_sha256']:
            raise ValueError('Global query composition taxonomy changed between units')
        protocol=entry['query_protocol_sha256']
        for name,index in entry['query_indexes'].items():
            for key in totals[name]:totals[name][key]+=index[key]
    if complete and len(entries)!=requested_units:
        raise ValueError('A complete catalog must have exactly the requested units')
    if complete and len(seeds)*4!=requested_units:
        raise ValueError('A complete catalog must retain four distinct target units per layout')
    manifest=dict(schema_version='lunar_unit_catalog/14.0',complete=bool(complete),
        requested_units=requested_units,published_units=len(entries),batch_root_relative_path='..',
        default_partition='query_train',native_rows_unchanged=True,training_ready=False,
        query_protocol_sha256=protocol,counts=totals,
        physical_layouts_by_split={split:sum(value==split for value in seeds.values()) for split in ('train','val','test')},
        physical_frames=sum(row['physical_frames'] for row in entries),units=entries)
    write_json(path,manifest)
    return manifest


def iter_rows(catalog_path,partition='query_train',*,batch_root=None,allow_partial=False):
    """Yield unit context plus an unchanged row; do not prepend/rebase its paths."""
    if partition not in FILES:raise ValueError('Unknown query/physical partition')
    catalog_path=Path(catalog_path).resolve(strict=True)
    catalog=json.loads(catalog_path.read_text())
    if not catalog.get('complete') and not allow_partial:raise ValueError('Training requires a complete catalog; partial reads must be explicit')
    root=Path(batch_root).resolve(strict=True) if batch_root else (catalog_path.parent/'..').resolve(strict=True)
    for entry in catalog['units']:
        unit=safe_child(root,entry['relative_unit_root'])
        item=entry['query_indexes'][partition]
        index=safe_child(unit,item['path'])
        if sha256(index)!=item['sha256']:raise ValueError('Catalog query index changed')
        for row in jsonl_rows(index):
            if row['split']!=entry['split'] or row['layout_seed']!=entry['layout_seed']:
                raise ValueError('Catalog row has a different physical layout/split')
            yield dict(unit_id=entry['unit_id'],unit_root=str(unit),partition_root=str(unit/QUERY_ROOT),row=row)


def load_policy_inputs(example):
    from .lunar_query_indexes_v14 import load_policy_inputs as load_unit
    return load_unit(Path(example['partition_root']),example['row'])
