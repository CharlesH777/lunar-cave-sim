"""Distributed target binding and held-out logical-operator compositions.

The actor receives only the concatenated public segment texts. The movement
core cannot name the goal or specify STOP; its goal/termination binding must be
read from the logical segment. Operator metadata is supervision/provenance.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from .lunar_route_language_v13 import (
    OBJECT_ALIASES, OBJECT_NAMES, canonical_instruction, check_route_queries, object_digest,
)

OPERATORS = {
    'waypoint_role': '途经地标不是任务终点，真正终点由逻辑段绑定',
    'nonterminal_no_stop': '不能仅因为看到途经点就停止；目标附近才结束',
    'near_goal_only_stop': '靠近真实目标是停止的必要条件',
    'visible_goal_approach': '看见目标但尚未靠近时先接近，不因首次看见而立即停止',
    'until_near_goal': '持续执行前进任务直到靠近目标，再停止',
    'arrival_conjunction': '路线前提与靠近目标两个条件都满足才停止',
    'target_contrast': '区分真正搜索目标与实际途经点/入口',
    'route_gate': '实际第一分叉方向对立选择；无分叉时使用实际入口前提',
}

GOAL_CORE_ALIASES = {
    'chinese_flag':('红旗','国旗','旗帜','旗子','五星'),
    'astronaut':('宇航员','航天员','太空人','宇航服','航天服'),
    'lunar_rover':('月球车','探月车','月面车','漫游车','月球探测车','六轮车','小车'),
    'lunar_base':('月球基地','月面基地','基地','空间站','月球站','居住舱'),
}


def _families():
    rows=[]
    for index,operator in enumerate(OPERATORS):
        for form in ('statement','conditional'):
            rows.append(dict(family_id=f'train.{operator}.{form}',query_partition='train',
                operator_ids=[operator],surface_form=form,composition='ATOM',
                segment_order=['logic','route_core'] if (index+(form=='conditional'))%2 else ['route_core','logic']))
    for index,(name,operators) in enumerate([
        ('role_and_stop',['waypoint_role','near_goal_only_stop']),
        ('nonterminal_and_appearance',['nonterminal_no_stop','visible_goal_approach']),
        ('until_and_preconditions',['until_near_goal','arrival_conjunction']),
        ('target_and_route_gate',['target_contrast','route_gate']),
    ]):
        rows.append(dict(family_id='heldout.'+name,query_partition='heldout',operator_ids=operators,
            surface_form='combined',composition='AND',
            segment_order=['logic','route_core'] if index%2 else ['route_core','logic']))
    return rows


DEFAULT_PROTOCOL=dict(schema_version='lunar_query_composition_protocol/15.0',query_count=20,
    max_output_tokens=10000,physical_split_unchanged=True,
    actor_input_rule='concatenate public segment texts; no operator metadata or resolved canonical answer',
    anonymous_route_goal=True,goal_and_stop_must_be_bound_in_logic=True,
    min_unique_route_cores=10,max_identical_normalized_route_core=2,
    operators=OPERATORS,families=_families())


def load_protocol(value=None):
    result=deepcopy(DEFAULT_PROTOCOL) if value is None else (
        json.loads(Path(value).read_text()) if isinstance(value,(str,Path)) else deepcopy(value))
    if (set(result)!=set(DEFAULT_PROTOCOL) or result['schema_version']!=DEFAULT_PROTOCOL['schema_version']
            or result['query_count']!=20 or result['max_output_tokens']!=10000
            or result['physical_split_unchanged'] is not True or result['anonymous_route_goal'] is not True
            or result['goal_and_stop_must_be_bound_in_logic'] is not True
            or result['operators']!=OPERATORS or result['families']!=_families()
            or not 10<=result['min_unique_route_cores']<=20
            or not 1<=result['max_identical_normalized_route_core']<=2):
        raise ValueError('Invalid V15 operator/partition/distributed-binding protocol')
    train=[row for row in result['families'] if row['query_partition']=='train']
    held=[row for row in result['families'] if row['query_partition']=='heldout']
    if (len(train)!=16 or len(held)!=4 or any(len(row['operator_ids'])!=1 for row in train)
            or any(len(row['operator_ids'])!=2 for row in held)
            or {op for row in held for op in row['operator_ids']} != set(OPERATORS)):
        raise ValueError('Every held-out combination must consist of previously trained atomic operators')
    return result


def protocol_hash(protocol):
    return object_digest(load_protocol(protocol))


def operator_context(contract):
    goal=contract['goal']['id']
    if goal not in OBJECT_NAMES:
        raise ValueError('Unknown operator goal')
    milestones=contract['intermediate_landmarks']
    entry=next((row for row in contract['events'] if row['type']=='enter_cave'),None)
    fork=next((row for row in contract['events'] if row['type']=='first_fork'),None)
    waypoint=milestones[0]['name'] if milestones else ('洞口' if entry else '第一个分叉' if fork else '途中位置')
    waypoint_id=milestones[0]['id'] if milestones else None
    if waypoint_id==goal:
        raise ValueError('A task endpoint cannot be assigned the nonterminal waypoint role')
    if entry:
        prerequisite='已按提示进入洞穴';imperative='按提示进入洞穴'
    elif fork:
        prerequisite='已按提示通过第一个分叉';imperative='按提示通过第一个分叉'
    elif milestones:
        prerequisite='已按提示经过'+waypoint;imperative='按提示经过'+waypoint
    else:
        prerequisite='已经看见'+OBJECT_NAMES[goal];imperative='确认看见'+OBJECT_NAMES[goal]
    return dict(goal_id=goal,goal_name=OBJECT_NAMES[goal],
        goal_description=('洞穴深处的' if contract['deep_target_hint'] else '')+OBJECT_NAMES[goal],
        waypoint_name=waypoint,waypoint_id=waypoint_id,route_prerequisite=prerequisite,
        route_imperative=imperative,fork_side=fork['side'] if fork else None,
        route_gate_realization='fork_choice' if fork else 'entry_precondition' if entry else 'goal_visibility_precondition')


def allowed_logic(spec,context):
    """Auditable grammatical realizations; changing a bound negation fails."""
    g,d,w=context['goal_name'],context['goal_description'],context['waypoint_name']
    p,verb=context['route_prerequisite'],context['route_imperative']
    side='左' if context['fork_side']=='left' else '右'
    other='右' if side=='左' else '左'
    statements={
        'waypoint_role':f'{w}只是路标，不是任务终点；真正要靠近并停下的是{d}。',
        'nonterminal_no_stop':f'不能因为看到了{w}就停；本次要找{d}，靠近{g}后再停下来。',
        'near_goal_only_stop':f'任务目标是{d}；只有靠近{g}才允许停下。',
        'visible_goal_approach':f'要找的是{d}。看到{g}但尚未靠近时先继续接近，靠近后再停，不能刚看到就停。',
        'until_near_goal':f'终点是{d}附近；在靠近{g}之前持续前进，直到靠近后再停下。',
        'arrival_conjunction':f'把{d}作为任务目标；{p}与靠近{g}两个条件都满足后才停。',
        'target_contrast':f'本次找的是{d}，不是{w}；靠近真正的任务目标并停下才结束。',
        'route_gate':(f'第一个分叉应走{side}侧，{other}侧不是提示方向；真正终点是{d}，靠近它才停。'
                      if context['fork_side'] else f'先{verb}，再接近{d}；目标是{g}，靠近后才能停。'),
    }
    conditionals={
        'waypoint_role':f'即使看到了{w}也只是经过路标；要靠近并停车的任务终点是{d}。',
        'nonterminal_no_stop':f'如果看到{w}，不能把它当作停止理由；要找的是{d}，靠近{g}以后再停。',
        'near_goal_only_stop':f'要找{d}，未靠近{g}时不能停；靠近后才停下来。',
        'visible_goal_approach':f'目标是{d}；若看见{g}但还没靠近，就继续接近而非马上停，靠近后再停车。',
        'until_near_goal':f'只要尚未靠近{g}就继续前进，直到靠近{d}再停车；它才是任务目标。',
        'arrival_conjunction':f'目标是{d}；若{p}且已靠近{g}，这两个条件同时成立才可以停下。',
        'target_contrast':f'不要把{w}当成要找的目标；本次选择{d}作为终点，靠近{g}才停下来。',
        'route_gate':(f'若来到第一个分叉，就走{side}边而不是{other}边；要找的是{d}，靠近{g}之后停止。'
                      if context['fork_side'] else f'若尚未满足{verb}的要求，就继续执行路线；本次找{d}，靠近{g}以后再停下。'),
    }
    combinations={
        'heldout.role_and_stop':f'虽然{w}只是途经路标而非终点，但只有靠近{d}才可以停下来；{g}才是本次要找的目标。',
        'heldout.nonterminal_and_appearance':f'不能因看到{w}就停；要找的是{d}，如果只是看到{g}但还未靠近，就继续接近，靠近后再停。',
        'heldout.until_and_preconditions':f'任务目标是{d}；继续前进，直到{p}并靠近{g}这两个条件都满足，再停下来。',
        'heldout.target_and_route_gate':(f'要找的是{d}而不是{w}；第一个分叉走{side}不走{other}，最后靠近真正的任务目标停下。'
            if context['fork_side'] else f'要找的是{d}而不是{w}；先{verb}，再靠近真正的任务目标并停下来。'),
    }
    if spec['composition']=='AND':
        return [combinations[spec['family_id']]]
    operator=spec['operator_ids'][0]
    return [(statements if spec['surface_form']=='statement' else conditionals)[operator]]


def compact(text):
    return re.sub(r'[\s，。、；：,.!?！？]','',text)


def assemble(segments):
    return ''.join(segment['text'].strip() for segment in segments)


ANONYMOUS_ENDPOINT = r'任务目标|终点|目的地'
ENTRY_PATTERN = r'(?:(?:进入|走进|步入|踏入|驶入|深入|潜入)(?:洞穴|洞内|熔岩洞)|进洞|入洞|穿过洞口)'
FORK_ORDINAL = r'(?:第一个|第一处|第一次|第一条|最初的|首个|首处|首条|第一)'
FORK_NOUN = r'(?:分叉口|分叉处|分岔口|分岔点|岔路口|分叉|分岔|岔道|岔路|岔口|路口|分支)'
CONTINUATION = (r'(?:继续|接着|保持|维持|持续|仍|依然)[^，。；]{0,8}'
                r'(?:前进|前行|行进|行驶|移动|走|开|深入|探索|推进|赶路)'
                r'|(?:往|向)(?:前|里)(?:走|开|前进|行进|移动)?'
                r'|不停留|不要停留|不停歇|不中断(?:行程|行进)'
                r'|前往|去往|驶向|走向|赶往')


def analyse_anonymous_route(text,contract):
    """Bounded Chinese event grammar, separate from logical role/STOP clauses.

    This accepts only observed route-event synonyms. It does not rewrite public
    provider text or infer extra routes from a language model.
    """
    errors=[];events=[]
    def fail(code,detail):
        errors.append(dict(code=code,detail=detail))
    if not isinstance(text,str) or not 4<=len(text)<=220 or '\n' in text:
        return dict(valid=False,errors=[dict(code='invalid_text',detail='Expected a short single-line movement core')],events=[])
    if any(alias in text for alias in GOAL_CORE_ALIASES[contract['goal']['id']]):
        fail('goal_leak','Concrete goal identity occurs in the movement core')
    if re.search(r'颜色|外观|模样|形状|红色|黄色|白色|穿着|轮子|车轮|天线|舱体',text):
        fail('appearance_leak','Movement core adds an object-appearance description')
    references=list(re.finditer(ANONYMOUS_ENDPOINT,text))
    if not references:fail('missing_anonymous_endpoint','No anonymous task endpoint')
    if re.search(r'\d|坐标|米|公里|度|秒|向东|向西|向南|向北|往东|往西|往南|往北|掉头|返回|后退|倒退|绕道|改道|跳过|捷径|加速|减速',text):
        fail('unverified_route_condition','An unverified metric, detour or motion condition was added')
    if re.search(r'不改变方向|保持(?:原有)?(?:朝向|方向)|径直|直线|始终直行|一直直走|不要转弯|不能转弯',text):
        fail('unverified_straight_constraint','Keeping one heading/going straight is not an observed route condition')
    if not contract['deep_target_hint'] and '深处' in text:
        fail('unsupported_deep_hint','No verified deep-cave goal hint')
    allowed={event['id'] for event in contract['intermediate_landmarks']}
    for identifier,aliases in OBJECT_ALIASES.items():
        if identifier not in allowed and identifier!=contract['goal']['id'] and any(alias in text for alias in aliases):
            fail('unsupported_landmark','Core introduces '+identifier)
    # "After passing W, do not linger / keep going" is a movement continuation,
    # not a goal STOP rule. Waive only these idioms in the actual W clause.
    stop_probe=text
    for token in re.finditer(r'不要停留|不停留|不停歇',text):
        before=text[:token.start()]
        clause=re.split(r'[，。；]',before)[-1]
        has_waypoint=any(alias in clause for event in contract['intermediate_landmarks'] for alias in OBJECT_ALIASES[event['id']])
        if has_waypoint and not re.search(ANONYMOUS_ENDPOINT,clause):
            stop_probe=stop_probe.replace(token.group(),'继续移动',1)
    if re.search(r'停|结束|终止|完成任务|刹车|制动|驻车|驻足|等候|等待|速度为零|零速|stop|hold|(?:目标|终点|目的地)(?:是|为|指)',stop_probe,re.I):
        fail('stop_or_binding_leak','Core contains an actual terminal rule or target-identity binding')
    entry=next((row for row in contract['events'] if row['type']=='enter_cave'),None)
    entry_match=re.search(ENTRY_PATTERN,text)
    if entry:
        if not entry_match:fail('missing_entry','Required entry event is absent')
        else:
            prefix=text[max(0,entry_match.start()-4):entry_match.start()]
            if re.search(r'不|别|不要|不能|无需',prefix):fail('negated_entry','Required entry is negated')
            events.append(dict(type='enter_cave',position=entry_match.start(),surface=entry_match.group()))
    elif entry_match:fail('extra_entry','Core introduces an entrance event not in the route')
    fork=next((row for row in contract['events'] if row['type']=='first_fork'),None)
    fork_match=re.search(FORK_ORDINAL+FORK_NOUN,text)
    if fork:
        side='左' if fork['side']=='left' else '右';opposite='右' if side=='左' else '左'
        if opposite in text:fail('wrong_or_extra_direction','Opposite branch direction occurs in movement core')
        if not fork_match:fail('missing_first_fork','Required first-fork cue is absent')
        else:
            decision=re.split(r'[，。；]',text[fork_match.end():])[0][:14]
            if side not in decision:fail('missing_fork_choice','First fork has no actual branch choice')
            if re.search(r'(?:不|别|不要|不能|禁止)[^，。；]{0,4}'+side,decision):
                fail('negated_actual_branch','Actual branch choice is negated')
            events.append(dict(type='first_fork',position=fork_match.start(),surface=fork_match.group(),side=fork['side']))
    elif fork_match or re.search(r'左|右|分叉|分岔|岔道|岔路|岔口',text):
        fail('extra_fork_or_direction','Core adds a fork/direction absent from the contract')
    if re.search(r'(?:第二|第三|下个|下一个|另一个|再次|后续)[^，。；]{0,3}'+FORK_NOUN,text):
        fail('extra_route_event','Core adds an unverified later intersection')
    if len(list(re.finditer(FORK_NOUN,text)))>1:
        fail('extra_intersection_mention','Core contains an additional unverified intersection clause')
    for event in contract['intermediate_landmarks']:
        aliases='|'.join(map(re.escape,OBJECT_ALIASES[event['id']]))
        matches=list(re.finditer(aliases,text));found=None
        for match in matches:
            before=text[max(0,match.start()-12):match.start()]
            prefix=re.search(r'(?:看到|看见|见到|目睹|确认|经过|途经|路过|通过|越过|掠过|过|见)[^，。；]{0,4}$',before)
            if not prefix:continue
            if re.search(r'(?:不|别|不要|不能)(?:看到|看见|见到|经过|途经|路过|通过|过|见)',before):continue
            next_goal=re.search(ANONYMOUS_ENDPOINT,text[match.end():])
            end=match.end()+next_goal.start() if next_goal else len(text)
            continuation=text[match.end():end]
            if re.search(CONTINUATION,continuation):
                found=dict(type='landmark_continue',id=event['id'],position=match.start()-len(before)+prefix.start(),surface=text[match.start():match.end()])
                break
        if found is None:fail('missing_landmark_continue','No observing/passing then continuing cue for '+event['id'])
        else:events.append(found)
    # Compare with the observed event graph, not a fixed entry/fork/landmark
    # template: a verified landmark can occur before the first fork.
    observed={(row['type'],row.get('id')):row for row in events}
    events=[observed[(row['type'],row.get('id'))] for row in contract['events']
            if (row['type'],row.get('id')) in observed]
    positions=[row['position'] for row in events]
    if positions!=sorted(positions) or len(set(positions))!=len(positions):
        fail('route_order_changed','Entry, first fork and actual waypoint continuation changed order')
    if references and positions and references[-1].start()<max(positions):
        fail('endpoint_before_milestones','Anonymous endpoint occurs before required route milestones')
    return dict(valid=not errors,errors=errors,events=events,
                anonymous_endpoint=references[-1].group() if references else None)


def check_anonymous_route(text,contract):
    report=analyse_anonymous_route(text,contract)
    if not report['valid']:
        # Preserve the existing review-context identity for already-billed
        # requests. Detailed per-family diagnostics remain available above.
        raise ValueError('Route query omitted or changed a verified route hint')


def query_schema(protocol):
    protocol=load_protocol(protocol)
    route=dict(type='object',additionalProperties=False,required=['kind','text'],properties={
        'kind':dict(type='string',enum=['route_core']),'text':dict(type='string',minLength=4,maxLength=220)})
    logic=dict(type='object',additionalProperties=False,required=['kind','operator_ids','text'],properties={
        'kind':dict(type='string',enum=['logic']),
        'operator_ids':dict(type='array',minItems=1,maxItems=2,items=dict(type='string',enum=list(OPERATORS))),
        'text':dict(type='string',minLength=4,maxLength=220)})
    return dict(type='object',additionalProperties=False,required=['queries'],properties={
        'queries':dict(type='array',minItems=20,maxItems=20,items=dict(type='object',additionalProperties=False,
            required=['family_id','segments'],properties={
                'family_id':dict(type='string',enum=[row['family_id'] for row in protocol['families']]),
                'segments':dict(type='array',minItems=2,maxItems=2,items=dict(anyOf=[route,logic]))}))})


def query_prompt(contract,protocol):
    protocol=load_protocol(protocol);context=operator_context(contract)
    specs=[dict(family_id=spec['family_id'],operator_ids=spec['operator_ids'],composition=spec['composition'],
        segment_order=spec['segment_order'],allowed_logic_texts=allowed_logic(spec,context)) for spec in protocol['families']]
    return ('一次生成二十条真正需要理解逻辑的月洞导航指令，JSON只有queries列表。按规定family顺序，'
        '每项只有family_id与segments；两个segment分别为route_core和logic，按segment_order排列。'
        '不要输出另外的instruction，最终输入将由两段原文直接拼接，总长不超过240字。'
        'route_core必须自然改写完整真实移动路线，按实际顺序保留进洞、已有的第一分叉方向、'
        '每个实际途经物后继续前进，最后前往“任务目标”或“终点”；不得出现具体任务目标名称，'
        '不得含停车、停止、结束或目标身份绑定。因此单看route_core不能知道要找哪个物体。'
        'logic必须逐字使用该family给出的allowed_logic_texts之一，正确填写operator_ids，'
        '这里负责真实目标身份、靠近后停车规则，以及指定的单逻辑算子或未见算子组合。'
        '不要把方向否定/角色对比抄到route_core。不能增加其他地标、分叉、距离、坐标或路线条件。'
        '二十条route_core至少十种不同写法，同一改写最多用两次；避免只换开头而复制共同尾句。'
        '每段单行，标点自然，不能用换行或数字编号。'
        '\n实际完整任务（仅用于理解；具体目标不能泄漏到route_core）：'+canonical_instruction(contract)+
        '\n实际逻辑绑定：'+json.dumps(context,ensure_ascii=False,separators=(',',':'))+
        '\n逐项结构规范：'+json.dumps(specs,ensure_ascii=False,separators=(',',':')))


def check_queries(value,contract,protocol):
    protocol=load_protocol(protocol);context=operator_context(contract)
    if not isinstance(value,dict) or set(value)!={'queries'} or not isinstance(value['queries'],list) or len(value['queries'])!=20:
        raise ValueError('V15 requires twenty prescribed typed query records')
    instructions=[];cores=[]
    for item,spec in zip(value['queries'],protocol['families']):
        if (not isinstance(item,dict) or set(item)!={'family_id','segments'} or item['family_id']!=spec['family_id']
                or not isinstance(item['segments'],list) or len(item['segments'])!=2
                or [segment.get('kind') for segment in item['segments']]!=spec['segment_order']):
            raise ValueError('V15 family, composition or segment order differs from prescribed metadata')
        by_kind={segment['kind']:segment for segment in item['segments']}
        core,logic=by_kind['route_core'],by_kind['logic']
        if (set(core)!={'kind','text'} or set(logic)!={'kind','operator_ids','text'}
                or logic['operator_ids']!=spec['operator_ids'] or not isinstance(logic['text'],str)
                or '\n' in logic['text'] or len(logic['text'])>220):
            raise ValueError('Wrong logical operator IDs or typed segment schema')
        allowed=allowed_logic(spec,context)
        if compact(logic['text']) not in {compact(text) for text in allowed}:
            raise ValueError('Logical truth/goal/STOP binding differs from '+spec['family_id'])
        check_anonymous_route(core['text'],contract)
        instruction=assemble(item['segments'])
        if len(instruction)>240:
            raise ValueError('Combined V15 actor instruction exceeds 240 characters')
        instructions.append(instruction);cores.append(compact(core['text']))
    counts=Counter(cores)
    if len(set(instructions))!=20 or len(counts)<protocol['min_unique_route_cores'] or max(counts.values())>protocol['max_identical_normalized_route_core']:
        raise ValueError('V15 repeats movement cores instead of supplying diverse complete expressions')
    return True


def variants(value,episode,split,protocol):
    protocol=load_protocol(protocol)
    return [dict(language_episode_id=episode+f':zh:{index:03d}',instruction=assemble(item['segments']),
        physical_trajectory_id=episode,split=split,family_id=spec['family_id'],query_partition=spec['query_partition'],
        segments=deepcopy(item['segments']),operator_ids=spec['operator_ids'],composition=spec['composition'],
        query_composition_version='15.0') for index,(item,spec) in enumerate(zip(value['queries'],protocol['families']))]


def public_value(records,protocol):
    protocol=load_protocol(protocol)
    for record,spec in zip(records,protocol['families']):
        if (record['instruction']!=assemble(record['segments']) or record['operator_ids']!=spec['operator_ids']
                or record['composition']!=spec['composition'] or record.get('query_composition_version')!='15.0'):
            raise ValueError('V15 actor input differs from the exact public segment concatenation')
    return dict(queries=[dict(family_id=row['family_id'],segments=deepcopy(row['segments'])) for row in records])


def parse_public_response(saved):
    """Normalize only an exact twenty-record array container, never its facts."""
    from .ppio_client_v11 import parse_public
    try:
        value,unwrapped=parse_public(saved)
        return value,dict(single_record_array_normalized=unwrapped,flat_query_array_wrapped=False)
    except ValueError:
        value=json.loads(saved['content'])
        if (not isinstance(value,list) or len(value)!=20
                or any(not isinstance(row,dict) or set(row)!={'family_id','segments'} for row in value)):
            raise ValueError('Expected a V15 query object or exact twenty typed-record array')
        return dict(queries=value),dict(single_record_array_normalized=False,flat_query_array_wrapped=True)


def completed_review_response(client,prompt,schema,model,images,max_tokens,checker):
    """Reuse only validated completed responses to the exact base task/schema.

    An old guard-feedback suffix changes the HTTP fingerprint. Preserve that
    original request identity rather than issuing a new equivalent request.
    No cache writes, reservations or network operations occur here.
    """
    expected=client.request_audit(prompt,schema,model,images,max_tokens)
    prefix=prompt+'\n前一次输出未通过本地一致性审核：'
    candidates=[]
    direct=client.cache/(expected['request_sha256']+'.request.json')
    paths=([direct] if direct.is_file() else [])+[path for path in sorted(client.cache.glob('*.request.json')) if path!=direct]
    for path in paths:
        try:
            audit=json.loads(path.read_text())
            if (audit.get('options')!=expected['options'] or audit.get('schema')!=schema
                    or audit.get('image_sha256')!=expected['image_sha256']
                    or audit.get('endpoint')!=expected['endpoint'] or audit.get('protocol')!=expected['protocol']
                    or not isinstance(audit.get('prompt'),str)
                    or audit['prompt']!=prompt and not audit['prompt'].startswith(prefix)):
                continue
            fingerprint=audit['request_sha256']
            if (path.name!=fingerprint+'.request.json'
                    or client.request_audit(audit['prompt'],schema,model,images,max_tokens)['request_sha256']!=fingerprint):
                continue
            saved=client._saved_response(fingerprint)
            if saved is None or saved['requested_model']!=model:continue
            value,_=parse_public_response(saved)
            checker(value)
            return saved,dict(completed_request_revalidated=True,
                completed_review_context_reused=audit['prompt']!=prompt,
                base_request_sha256=expected['request_sha256'],actual_request_sha256=fingerprint)
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return None
