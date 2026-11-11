"""Globally partitioned query composition families over a verified route."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .lunar_route_language_v13 import canonical_instruction, check_route_queries, object_digest


def family(identity, prefix, style, required=(), *, partition='train', min_sentences=1, max_sentences=12):
    return dict(family_id=identity, query_partition=partition, prefix=prefix,
                composition=style, required_connectors=list(required),
                min_sentences=min_sentences, max_sentences=max_sentences)


DEFAULT_PROTOCOL = dict(schema_version='lunar_query_composition_protocol/14.0', query_count=20,
    max_output_tokens=10000, physical_split_unchanged=True,
    partition_scope='global composition families; heldout families never enter query training indexes',
    families=[
        family('train.goal_first', '找到', '目标先行，两句表达任务与完整路线', min_sentences=2),
        family('train.task_sequence', '任务是', '任务说明后用先、再、最后组织实际路线', ('先','最后')),
        family('train.polite_compound', '请找到', '礼貌请求开头，单句复合指令', max_sentences=1),
        family('train.route_first', '先进入', '先说明实际路线，再明确找到何物；目标名称出现在进洞提示之后'),
        family('train.purpose_clause', '为了找到', '目的从句引出实际路线'),
        family('train.when_clause', '当你', '以实际进入洞穴的当…后从句连接任务', ('后',)),
        family('train.objective_constraints', '目标是', '明确任务目标，再用依次组织约束', ('依次',)),
        family('train.requirements', '需要找到', '简洁需求说明，保留全部实际路标条件'),
        family('train.you_need', '你需要', '第二人称任务简报，随后给出路线'),
        family('train.short_sentences', '先要', '至少三个短句分别表达目标、路线、靠近后停止', min_sentences=3),
        family('train.search_one_sentence', '寻找', '以寻找目标开头，将路线嵌在一个复合句里', max_sentences=1),
        family('train.polite_route_first', '请先进入', '礼貌请求先走路线，再点明最终寻找目标'),
        family('train.first_person_task', '我要', '第一人称任务叙述，陈述自己应遵循的实际路线'),
        family('train.you_goal', '你要', '直接对执行者说明目标与路线'),
        family('train.task_goal_closure', '任务目标是', '目标先行，结尾再次点明目标并靠近停止'),
        family('train.this_mission', '这次要', '本次任务简报，用随后连接实际路线阶段', ('随后',)),
        family('heldout.nested_if_purpose', '如果要找到', '目的条件与路线后置结合；用就连接执行条件', ('就',), partition='heldout'),
        family('heldout.route_brief_goal_last', '先按这条路线', '先完整路线简报，最后再明确要找到的目标', partition='heldout'),
        family('heldout.before_terminal', '在靠近目标停下来之前', '先提出终止前提，再依实际顺序描述路线，结尾明确靠近具体目标停止', partition='heldout'),
        family('heldout.only_then_mission', '为完成这次寻找任务', '任务目的与只有…才…条件组合，完整说明实际路线', ('只有','才'), partition='heldout'),
    ])


def load_protocol(value=None):
    if value is None:
        result=deepcopy(DEFAULT_PROTOCOL)
    elif isinstance(value,(str,Path)):
        result=json.loads(Path(value).read_text())
    else:
        result=deepcopy(value)
    if result.get('schema_version') == 'lunar_query_composition_protocol/15.0':
        from .lunar_query_composition_v15 import load_protocol as load_v15
        return load_v15(result)
    if (result.get('schema_version') != DEFAULT_PROTOCOL['schema_version'] or result.get('query_count') != 20
            or result.get('physical_split_unchanged') is not True or not 2000 <= result.get('max_output_tokens',0) <= 10000
            or not isinstance(result.get('families'),list) or len(result['families']) != 20):
        raise ValueError('Invalid V14 twenty-query composition protocol')
    expected={row['family_id']:row['query_partition'] for row in DEFAULT_PROTOCOL['families']}
    seen=set()
    for row in result['families']:
        if (set(row) != set(DEFAULT_PROTOCOL['families'][0]) or row['family_id'] in seen
                or row['family_id'] not in expected or row['query_partition'] != expected[row['family_id']]
                or not row['prefix'] or not row['composition'] or not 1 <= row['min_sentences'] <= row['max_sentences']
                or not isinstance(row['required_connectors'],list)):
            raise ValueError('Query family identity, composition or global partition changed')
        seen.add(row['family_id'])
    if seen != set(expected):
        raise ValueError('Query taxonomy must contain all sixteen train and four heldout families')
    return result


def query_schema(protocol):
    protocol=load_protocol(protocol)
    if protocol['schema_version'].endswith('/15.0'):
        from .lunar_query_composition_v15 import query_schema as schema_v15
        return schema_v15(protocol)
    return dict(type='object',additionalProperties=False,required=['queries'],properties={
        'queries':dict(type='array',minItems=20,maxItems=20,items=dict(type='object',additionalProperties=False,
            required=['family_id','instruction'],properties={
                'family_id':dict(type='string',enum=[row['family_id'] for row in protocol['families']]),
                'instruction':dict(type='string',minLength=4,maxLength=240)}))})


def query_prompt(contract,protocol):
    protocol=load_protocol(protocol)
    if protocol['schema_version'].endswith('/15.0'):
        from .lunar_query_composition_v15 import query_prompt as prompt_v15
        return prompt_v15(contract,protocol)
    family_specs=[{key:row[key] for key in ('family_id','prefix','composition','required_connectors','min_sentences','max_sentences')}
                  for row in protocol['families']]
    return ('将已核验的月洞导航任务写成二十条结构不同的中文指令，一次返回JSON对象queries列表。'
            '每项只有family_id和instruction，按给定family顺序各生成一条。family_id是输出元数据，不要写进指令。'
            '每条必须保留原任务的目标、洞穴深处提示（如果有）、进洞、每个实际分叉方向、'
            '每个实际地标后继续前进、这些提示的顺序、最后靠近目标停止。原任务没有的条件不得添加。'
            '只允许改表达结构，不能加入其他地标、另一分叉、绕路、精确距离、坐标或未来画面。'
            '每条单行且最多240字；不同family要体现从句/主句结构差异，不能只换同义词。'
            '必须逐字以该family的prefix开头，包含required_connectors，并满足中文句号划分的句数范围。'
            '路线先行的指令也要明确说找到哪个具体目标，不只写“目标”。'
            '\n已核验的固定任务：'+canonical_instruction(contract)+
            '\n结构规范：'+json.dumps(family_specs,ensure_ascii=False,separators=(',',':')))


def check_queries(value,contract,protocol):
    protocol=load_protocol(protocol)
    if protocol['schema_version'].endswith('/15.0'):
        from .lunar_query_composition_v15 import check_queries as check_v15
        return check_v15(value,contract,protocol)
    if not isinstance(value,dict) or set(value) != {'queries'} or not isinstance(value['queries'],list) or len(value['queries']) != 20:
        raise ValueError('The query batch must contain exactly twenty prescribed family records')
    texts=[]
    for item,spec in zip(value['queries'],protocol['families']):
        if (not isinstance(item,dict) or set(item) != {'family_id','instruction'}
                or item['family_id'] != spec['family_id'] or not isinstance(item['instruction'],str)):
            raise ValueError('Query family IDs/order differ from the prescribed global taxonomy')
        text=item['instruction']
        sentences=len([part for part in text.split('。') if part.strip()])
        if (not text.startswith(spec['prefix']) or not all(word in text for word in spec['required_connectors'])
                or not spec['min_sentences'] <= sentences <= spec['max_sentences']):
            raise ValueError('Query composition does not match family '+spec['family_id'])
        texts.append(text)
    check_route_queries(dict(queries=texts),contract,expected_count=20)
    return True


def variants(value,episode,split,protocol):
    protocol=load_protocol(protocol)
    if protocol['schema_version'].endswith('/15.0'):
        from .lunar_query_composition_v15 import variants as variants_v15
        return variants_v15(value,episode,split,protocol)
    return [dict(language_episode_id=episode+f':zh:{i:03d}',instruction=item['instruction'],
                 physical_trajectory_id=episode,split=split,family_id=spec['family_id'],query_partition=spec['query_partition'])
            for i,(item,spec) in enumerate(zip(value['queries'],protocol['families']))]


def protocol_hash(protocol):
    return object_digest(load_protocol(protocol))


def public_value(records,protocol):
    protocol=load_protocol(protocol)
    if protocol['schema_version'].endswith('/15.0'):
        from .lunar_query_composition_v15 import public_value as public_v15
        return public_v15(records,protocol)
    return dict(queries=[dict(family_id=row['family_id'],instruction=row['instruction']) for row in records])
