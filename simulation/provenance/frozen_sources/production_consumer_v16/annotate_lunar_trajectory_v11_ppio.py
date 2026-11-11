"""Annotate real continuous trajectories with sparse, causal public summaries.

The teacher, mask and camera are ground truth. The LLM never receives a map,
absolute goal, future RGB, future controls or exact remaining distance.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

from cave.ppio_client_v11 import PPIOClient, BudgetExceeded, UncertainRequest, object_hash, parse_public, sha256, write_json

ROOT = Path(__file__).resolve().parent
MAIN_MODEL = 'qwen/qwen3.8-27b'
REVIEW_MODEL = 'qwen/qwen3.8-max'
COT_STYLES = ('legacy', 'first_person')
INSTRUCTION_STYLES = ('legacy', 'route_grounded')
QUERY_MODES = ('provider', 'compiled', 'provider_with_compiled_fallback')
PROGRESS = ['到达洞口', '进入洞穴', '洞内搜索', '接近目标', '准备停止', '暂无法确认']
INTENTS = ['move_forward', 'turn_left', 'turn_right', 'explore', 'stop', 'hold']
ALIASES = {'astronaut': ['宇航员'], 'lunar_rover': ['探月车', '月球车'],
           'lunar_base': ['月球基地', '基地'], 'chinese_flag': ['中国国旗', '国旗', '红旗']}
OUTPUT_SCHEMA = dict(type='object', additionalProperties=False,
    required=['progress', 'observation', 'decision_reason', 'action_intent'], properties={
        'progress': dict(type='string', enum=PROGRESS),
        'observation': dict(type='string', minLength=2, maxLength=60),
        'decision_reason': dict(type='string', minLength=2, maxLength=80),
        'action_intent': dict(type='string', enum=INTENTS)})
QUERY_SCHEMA = dict(type='object', additionalProperties=False, required=['queries'], properties={
    'queries': dict(type='array', minItems=3, maxItems=3, items=dict(type='string', minLength=4, maxLength=100))})
TARGET_NAMES = {'astronaut': '宇航员', 'lunar_rover': '探月车',
                'lunar_base': '月球基地', 'chinese_flag': '红旗'}
FIRST_PERSON_TERRAIN = ('我看到了灰色月壤', '我看到了碎石', '我看到了岩壁',
                        '我看到了岩石', '我看到了灰色月壤和碎石', '我暂时看不清前方')
FIRST_PERSON_ACTION = {'move_forward': '我应该继续前进', 'turn_left': '我应该向左转',
                       'turn_right': '我应该向右转', 'hold': '我应该保持当前位置',
                       'stop': '我应该停下来'}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows))
    temporary.replace(path)


def safe_path(source, value):
    path = (Path(source)/value).resolve(strict=True)
    if not path.is_relative_to(Path(source).resolve()): raise ValueError('Source path escapes trajectory')
    return path


def teacher_intent(action):
    from cave.lunar_action_intent_v12 import action_intent
    return action_intent(action['v_mps'], action['omega_radps'], action['stop'],
                         motion_controller=action.get('motion_controller', 'segment'))


def check_queries(value, target, *, instruction_style='legacy', route_contract=None, query_protocol=None):
    if query_protocol is not None:
        if instruction_style != 'route_grounded' or route_contract is None or route_contract['goal']['id'] != target:
            raise ValueError('Composition queries require their actual route goal/contract')
        from cave.lunar_query_families_v14 import check_queries as check_families
        return check_families(value, route_contract, query_protocol)
    if instruction_style == 'route_grounded':
        from cave.lunar_route_language_v13 import check_route_queries
        if route_contract is None or route_contract['goal']['id'] != target:
            raise ValueError('Route query goal differs from its contract')
        return check_route_queries(value, route_contract)
    if instruction_style != 'legacy': raise ValueError('Unknown instruction style')
    if set(value) != {'queries'} or not isinstance(value['queries'], list) or len(value['queries']) != 3:
        raise ValueError('Query response must contain exactly three queries')
    if len(set(value['queries'])) != 3: raise ValueError('Query variants are duplicated')
    for query in value['queries']:
        if not isinstance(query, str) or not 4 <= len(query) <= 100 or '\n' in query:
            raise ValueError('Invalid query string')
        if not any(alias in query for alias in ALIASES[target]): raise ValueError('Query changed target identity')
        if re.search(r'左|右|坐标|\d|往东|向东|往西|向西|往南|向南|往北|向北', query):
            raise ValueError('Query contains a route/direction/metric answer')
        if any(alias in query for key, aliases in ALIASES.items() if key != target for alias in aliases):
            raise ValueError('Query adds another task target')
    return True


def positive(pattern, text):
    """Small auditable Chinese claim guard, with local explicit negation handling."""
    for match in re.finditer(pattern, text):
        prefix = text[max(0, match.start()-8):match.start()]
        if not re.search(r'(?:未|没|没有|尚未|尚无|无法|不能|不曾|并非|不是|无|不)[^，。；]{0,5}$', prefix):
            return True
    return False


def check_output(value, frame, *, cot_style='legacy', instruction_style='legacy', route_contract=None):
    if cot_style not in COT_STYLES: raise ValueError('Unknown CoT style')
    if instruction_style not in INSTRUCTION_STYLES: raise ValueError('Unknown instruction style')
    if not isinstance(value, dict) or set(value) != set(OUTPUT_SCHEMA['required']): raise ValueError('Wrong output fields')
    if any(not isinstance(v, str) or '\n' in v for v in value.values()): raise ValueError('Output must contain short single-line strings')
    if value['progress'] not in PROGRESS or value['action_intent'] not in INTENTS: raise ValueError('Unknown progress/intent')
    if not 2 <= len(value['observation']) <= 60 or not 2 <= len(value['decision_reason']) <= 80: raise ValueError('Invalid summary length')
    evidence = frame['machine_evidence']; expected = teacher_intent(frame['executed_action'])
    if value['action_intent'] != expected: raise ValueError('Intent disagrees with actually executed teacher control')
    if evidence.get('expected_action_intent') != expected: raise ValueError('Source evidence and control disagree')
    text = value['observation']+'。'+value['decision_reason']
    if re.search(r'\d+(?:\.\d+)?\s*(?:米|m\b|度|秒|m/s)|坐标|下一帧|未来路线|后面将会|接下来会看到', text):
        raise ValueError('Unsupported precise metric or future observation')
    for phrase in ('雪地', '积雪', '下雪', '冰雪', '白雪', '草地', '树林', '水池', '河流'):
        if positive(re.escape(phrase), text): raise ValueError('Unsupported dry-lunar-scene claim')
    if positive(r'已经(?:安全)?停(?:止|下|稳)|已停稳|完成停车|已(?:经)?到达目标|已经找到目标|已完成任务', text):
        raise ValueError('Commanded STOP is not evidence of a previously achieved stop')
    event = evidence.get('entry_event', 'unknown')
    if value['progress'] == '进入洞穴' and event != 'entered_cave': raise ValueError('No actual prior-to-current entrance crossing')
    if value['progress'] == '到达洞口' and event not in ('entered_cave', 'at_cave_entrance'): raise ValueError('No current entrance event')
    semantic_entry = (evidence.get('semantic_visibility_revision') == '14.0'
                      and evidence.get('semantic_entry_support') is not None
                      and evidence['semantic_entry_support'].get('current_inside_verified') is True
                      and evidence['semantic_entry_support'].get('step_index', frame['step_index']+1) <= frame['step_index'])
    if positive(r'刚(?:刚)?进入洞穴|已经进入洞穴|已进入洞穴|刚进洞', text) and event != 'entered_cave' and not semantic_entry:
        raise ValueError('Invented temporal entry event')
    if positive(r'已到(?:达)?洞(?:穴入)?口|到达洞(?:穴入)?口|位于洞口', text) and event not in ('entered_cave', 'at_cave_entrance'):
        raise ValueError('Unsupported entrance claim')
    if value['progress'] == '准备停止' and expected != 'stop': raise ValueError('Premature stop progress')
    visible = evidence.get('target_visible')
    if value['progress'] == '接近目标' and visible is not True:
        raise ValueError('Target-approach progress has no current visible evidence')
    names = '(?:目标|'+ '|'.join(map(re.escape, ALIASES[frame['target_id']]))+')'
    seeing = r'(?:看到|看见|见到|可见|发现)[^，。；]{0,12}'+names+'|'+names+r'[^，。；]{0,8}(?:清晰可见|出现在|已显现)'
    if visible is not True and positive(seeing, text): raise ValueError('Current target visibility not confirmed by current mask')
    if visible is None and re.search(r'没有看到|未看到|看不到|目标不可见', text):
        raise ValueError('Unknown visibility cannot be asserted false')
    if visible is True:
        bearing = evidence.get('target_bearing')
        for side, pattern in [('left', r'左侧|左前|左边|偏左|在左'), ('right', r'右侧|右前|右边|偏右|在右')]:
            # Check target location, not a teacher turn in an obstacle detour.
            location_pattern = names+r'[^，。；]{0,8}(?:'+pattern+')|(?:'+pattern+r')[^，。；]{0,8}'+names
            if positive(location_pattern, value['observation']) and bearing != side:
                raise ValueError('Target direction contradicts calibrated current mask bbox')
        # Target bearing need not match motion while following a safe route.
        if expected == 'turn_left' and bearing == 'right' and re.search(r'对准|朝向目标|向目标', value['decision_reason']):
            raise ValueError('Left turn cannot be justified as aligning to a right-side target')
        if expected == 'turn_right' and bearing == 'left' and re.search(r'对准|朝向目标|向目标', value['decision_reason']):
            raise ValueError('Right turn cannot be justified as aligning to a left-side target')
    if expected == 'turn_left' and positive(r'向右转|右转|往右转', value['decision_reason']): raise ValueError('Text turn contradicts actual left action')
    if expected == 'turn_right' and positive(r'向左转|左转|往左转', value['decision_reason']): raise ValueError('Text turn contradicts actual right action')
    if visible is not True and positive(names+r'[^，。；]{0,8}(?:偏左|偏右|左侧|右侧|左前|右前|在前方)|(?:左侧|右侧|左前|右前)[^，。；]{0,8}'+names, text):
        raise ValueError('Unseen target direction is privileged')
    if re.search(r'为(?:了)?(?:避开|绕过)|因[^。]{0,12}(?:挡路|阻挡)|(?:石块|巨石|障碍物)[^。]{0,6}(?:挡路|阻挡)', text):
        if not evidence.get('local_obstacle_verified', False):
            raise ValueError('Specific obstacle avoidance explanation lacks independent local obstacle evidence')
    if cot_style == 'first_person' or instruction_style == 'route_grounded':
        progress, pairs = grounded_pairs(frame, instruction_style=instruction_style, route_contract=route_contract)
        pair = (compact_clause(value['observation']), compact_clause(value['decision_reason']))
        if value['progress'] not in progress or pair not in pairs:
            raise ValueError('First-person observation/action pair is not supported by the current frame')
    return True


def verified_fork_side(frame):
    evidence = frame['machine_evidence']
    side = evidence.get('fork_choice_side')
    return side if evidence.get('fork_visible_verified') is True and side in ('left', 'right') else None


def compact_clause(value):
    return value.strip().rstrip('。')


def render_cot_text(value):
    return compact_clause(value['observation'])+'，'+compact_clause(value['decision_reason'])+'。'


def first_person_grounding(frame):
    """Build a closed set of truthful, concise observation/action pairs."""
    evidence = frame['machine_evidence']
    intent = teacher_intent(frame['executed_action'])
    decision = FIRST_PERSON_ACTION[intent]
    visible = evidence.get('target_visible') is True
    target_observations = []
    if visible:
        target = TARGET_NAMES[frame['target_id']]
        target_observations.append('我看到了'+target)
        bearing = {'left': '左侧', 'right': '右侧', 'ahead': '前方'}.get(evidence.get('target_bearing'))
        if bearing:
            target_observations.append('我看到了'+bearing+'的'+target)
    if intent == 'stop':
        if evidence.get('terminal_accepted') is not True or not visible:
            return [], []
        return ['准备停止'], [(observation, decision) for observation in target_observations]

    progress = ['暂无法确认']
    fork_side = verified_fork_side(frame)
    if evidence.get('progress', {}).get('inside_cave') is True or fork_side:
        progress.append('洞内搜索')
    if visible:
        progress.append('接近目标')
    pairs = [(observation, decision) for observation in (*FIRST_PERSON_TERRAIN, *target_observations)]
    entry = evidence.get('entry_event')
    if intent == 'move_forward' and entry in ('at_cave_entrance', 'entered_cave'):
        progress.append('到达洞口')
        if entry == 'entered_cave':
            progress.append('进入洞穴')
        pairs = [('我看到了洞口', '我应该进去')]
    if fork_side and intent in ('move_forward', 'turn_left', 'turn_right'):
        side = '左' if fork_side == 'left' else '右'
        pairs = [('我看到了分叉', f'我应该走{side}边的通道')]
    return progress, pairs


def grounded_pairs(frame, *, instruction_style='legacy', route_contract=None):
    if frame['machine_evidence'].get('semantic_visibility_revision') == '14.0':
        if instruction_style != 'route_grounded' or route_contract is None:
            raise ValueError('V14 semantic text requires its verified route contract')
        from cave.lunar_route_materializer_v14 import route_frame_pair
        from cave.lunar_route_language_v13 import canonical_instruction
        if frame['instruction'] != canonical_instruction(route_contract) or frame['episode_id'] != route_contract['physical_trajectory_id']:
            raise ValueError('V14 fixed instruction/physical identity mismatch')
        pair = route_frame_pair(frame, route_contract)
        return ([pair[0]], [pair[1]]) if pair else ([], [])
    progress, pairs = first_person_grounding(frame)
    if instruction_style == 'legacy':
        return progress, pairs
    if instruction_style != 'route_grounded' or route_contract is None:
        raise ValueError('Route-conditioned text requires its verified semantic contract')
    from cave.lunar_route_language_v13 import canonical_instruction, route_frame_pair
    if (frame['instruction'] != canonical_instruction(route_contract)
            or frame['episode_id'] != route_contract['physical_trajectory_id']):
        raise ValueError('CoT frame instruction differs from its route contract')
    pair = route_frame_pair(frame, route_contract)
    if pair is not None:
        return [pair[0]], [pair[1]]
    # Only the contract's actual first fork can be narrated as that route cue.
    pairs = [(o, a) for o, a in pairs if '分叉' not in o]
    if not pairs:
        intent = teacher_intent(frame['executed_action'])
        pairs = [(observation, FIRST_PERSON_ACTION[intent]) for observation in FIRST_PERSON_TERRAIN]
    return progress, pairs


def check_annotation_cot(row, frame, *, cot_style='legacy', instruction_style='legacy', route_contract=None):
    if cot_style not in COT_STYLES or row.get('cot_style', 'legacy') != cot_style:
        raise ValueError('Annotation CoT style differs from its protocol')
    if row['cot_valid']:
        check_output(row['output'], frame, cot_style=cot_style,
                     instruction_style=instruction_style, route_contract=route_contract)
    elif row['output'] is not None:
        raise ValueError('Invalid sparse frame contains a fabricated text target')
    if cot_style == 'first_person':
        expected = render_cot_text(row['output']) if row['cot_valid'] else None
        if 'cot_text' not in row or row['cot_text'] != expected:
            raise ValueError('First-person cot_text differs from its structured output')
    if row.get('instruction_style', 'legacy') != instruction_style:
        raise ValueError('Annotation instruction style differs from its protocol')
    if instruction_style == 'route_grounded':
        from cave.lunar_route_language_v13 import object_digest
        if row.get('route_contract_sha256') != object_digest(route_contract):
            raise ValueError('CoT row refers to a different semantic route contract')


def select_frames(frames, maximum=16, all_frames=False, *, cot_style='legacy',
                  instruction_style='legacy', route_contract=None):
    if frames and frames[0]['machine_evidence'].get('semantic_visibility_revision') == '14.0':
        if all_frames or instruction_style != 'route_grounded' or route_contract is None:
            raise ValueError('V14 text selection requires sparse task-relevant exact-grid route cues')
        from cave.lunar_route_materializer_v14 import select_frames as select_v14
        return select_v14(frames, route_contract, maximum)
    if all_frames: return {r['sample_id']: ['all_frames'] for r in frames}
    selected = {}; previous_intent = None; previously_visible = False; previous_entry = False; previous_fork = None
    route_landmarks = set()
    if instruction_style == 'route_grounded':
        if route_contract is None: raise ValueError('Route frame selection requires its contract')
        route_landmarks = {event['step_index'] for event in route_contract['events'] if event['type'] == 'landmark_continue'}
    for frame in frames:
        evidence = frame['machine_evidence']; intent = teacher_intent(frame['executed_action']); events = []
        entry = evidence.get('entry_event')
        if entry == 'entered_cave': events.append('entered_cave')
        elif entry == 'at_cave_entrance' and not previous_entry: events.append('entrance')
        if evidence.get('target_visible') is True and not previously_visible: events.append('target_first_visible')
        if intent in ('turn_left', 'turn_right') and intent != previous_intent: events.append('turn_change')
        if intent == 'stop': events.append('terminal_stop')
        if frame['step_index'] in route_landmarks: events.append('route_landmark')
        if (cot_style == 'first_person' or instruction_style == 'route_grounded') and intent in ('move_forward', 'turn_left', 'turn_right'):
            fork = verified_fork_side(frame)
            if fork is not None and fork != previous_fork:
                events.append('verified_fork')
                previous_fork = fork
        if not selected: events.append('initial')
        if events: selected[frame['sample_id']] = events
        previously_visible = previously_visible or evidence.get('target_visible') is True; previous_intent = intent
        previous_entry = entry in ('entered_cave', 'at_cave_entrance')
    if len(selected) <= maximum: return selected
    by_id = {r['sample_id']: r for r in frames}
    priority = {'terminal_stop': 0, 'target_first_visible': 1, 'entered_cave': 2, 'initial': 3, 'entrance': 4, 'turn_change': 5}
    if cot_style == 'first_person' or instruction_style == 'route_grounded':
        priority = {'terminal_stop': 0, 'target_first_visible': 1, 'entered_cave': 2, 'verified_fork': 3,
                    'entrance': 4, 'initial': 5, 'turn_change': 6}
    if instruction_style == 'route_grounded': priority['route_landmark'] = 3
    ordered = sorted(selected, key=lambda key: (min(priority[e] for e in selected[key]), by_id[key]['step_index']))
    chosen = set(ordered[:maximum])
    return {key: events for key, events in selected.items() if key in chosen}


def frame_prompt(frame, *, cot_style='legacy', instruction_style='legacy', route_contract=None):
    if cot_style == 'first_person' or instruction_style == 'route_grounded':
        progress, pairs = grounded_pairs(frame, instruction_style=instruction_style, route_contract=route_contract)
        intent = teacher_intent(frame['executed_action'])
        facts = dict(progress=progress, action_intent=intent,
                     allowed_pairs=[dict(observation=o, decision_reason=a) for o, a in pairs])
        preferred = next(((o, a) for o, a in pairs if o in ('我看到了洞口', '我看到了分叉')), None)
        if intent == 'stop' and pairs:
            preferred = pairs[0]
        if instruction_style == 'route_grounded' and pairs:
            preferred = pairs[0]
        example = ('\n该帧已核验的优先说法：'+render_cot_text(dict(observation=preferred[0], decision_reason=preferred[1]))
                   if preferred else '')
        return ('给真实月洞导航的当前帧写一句第一人称的观察与动作说明。返回四字段 JSON，'
                'progress 从当前可用选项中选，action_intent 逐字照抄当前实际动作。'
                'observation 与 decision_reason 从 allowed_pairs 选择同一对短句，'
                '依当前图像选真实观察，保持“我看到了…，我应该…”的口吻。'
                '图片先当前帧，若有第二张则是真实上一帧。不得把上一帧或未来画面当作当前事实。'
                '不要添加物品、方向、坐标、距离或动作解释。停下来指当前停止命令。'
                '\n固定任务：'+frame['instruction']+'\n当前可用说法：'+json.dumps(facts, ensure_ascii=False)+example)
    if cot_style != 'legacy': raise ValueError('Unknown CoT style')
    evidence = frame['machine_evidence']
    facts = {key: evidence.get(key) for key in ('target_visible', 'target_bearing', 'entry_event', 'terminal_accepted')}
    facts['action_intent'] = teacher_intent(frame['executed_action'])
    facts['history_present'] = bool(frame['inputs'].get('history'))
    return ('为真实月洞导航轨迹的当前帧生成简短中文外部决策说明，不输出内部思考。'
        '场景是干燥灰色月壤、熔岩洞与碎石，没有雪地、水或植被。严格返回四字段JSON。'
        'instruction在同一episode固定，禁止改写任务；图片依次是当前帧和可选的真实上一帧。'
        'progress仅可选到达洞口/进入洞穴/洞内搜索/接近目标/准备停止/暂无法确认。'
        '进入洞穴仅当entry_event=entered_cave；到达洞口仅当at_cave_entrance或entered_cave。'
        '目标可见性只依据当前mask事实，方向left=相机左侧，right=右侧，ahead=前方；未知时不能编目标方向。'
        'action_intent必须逐字等于给定实际teacher意图。真实绕障动作可以与目标所在方向不同，不能硬说是向目标对准；'
        '缺少已核验局部障碍标签，不能编造某块石头挡路或具体绕障原因，可说按当前路线调整朝向、继续搜索。'
        'observation只描述当前图像和事实，≤60字；decision_reason简短≤80字。'
        'stop表示当前请求停止，只能说准备停止，不能声称之前已经停车/已完成任务。'
        '禁止精确距离/坐标/未来画面/不可见地标，不出现mask、teacher、标注、机器事实等术语。'
        '\n固定instruction：'+frame['instruction']+'\n当前因果核验事实：'+json.dumps(facts, ensure_ascii=False))


def query_prompt(target, canonical, *, instruction_style='legacy', route_contract=None, query_protocol=None):
    if query_protocol is not None:
        if instruction_style != 'route_grounded' or route_contract is None:
            raise ValueError('Composition query prompting requires a verified route')
        from cave.lunar_query_families_v14 import query_prompt as family_prompt
        from cave.lunar_route_language_v13 import canonical_instruction
        if canonical != canonical_instruction(route_contract):
            raise ValueError('Composition query fixed task differs from actual route')
        return family_prompt(route_contract, query_protocol)
    if instruction_style == 'route_grounded':
        from cave.lunar_route_language_v13 import canonical_instruction, route_query_prompt
        if route_contract is None or canonical != canonical_instruction(route_contract):
            raise ValueError('Route query prompt differs from its canonical instruction')
        return route_query_prompt(route_contract)
    if instruction_style != 'legacy': raise ValueError('Unknown instruction style')
    return ('生成三个不同的中文导航任务改写，JSON只有queries列表。目标保持不变：'+ALIASES[target][0]+
        '。任务是找到该物品并在附近安全停止。不能加入世界坐标、局部左右、具体路线、其他目标或距离答案。'
        '每条4至100字。原始任务：'+canonical)


def annotation_images(source, frame):
    images = [safe_path(source, frame['rgb_224'])]
    history = frame['inputs'].get('history', [])
    if history:
        previous = history[-1]
        if previous['timestamp_s'] >= frame['inputs']['observation']['timestamp_s']:
            raise ValueError('Noncausal history')
        images.append(safe_path(source, previous['rgb']))
    return images


def annotate(source, output, *, client=None, skip_llm=False, max_llm_frames=16, all_frames=False,
             model=MAIN_MODEL, review_model=REVIEW_MODEL, resume=False, split='train', cot_style='legacy',
             instruction_style='legacy', query_count=3, query_protocol=None, query_model=None,
             query_mode='provider', query_bank=None):
    source = Path(source).resolve(strict=True); output = Path(output).absolute()
    if max_llm_frames < 1: raise ValueError('max_llm_frames must be positive')
    if cot_style not in COT_STYLES: raise ValueError('Unknown CoT style')
    if instruction_style not in INSTRUCTION_STYLES: raise ValueError('Unknown instruction style')
    if query_mode not in QUERY_MODES: raise ValueError('Unknown query generation mode')
    output.mkdir(parents=True, exist_ok=resume)
    metadata = json.loads((source/'metadata.json').read_text()); frames = read_jsonl(source/'frame_records.jsonl')
    if metadata.get('semantic_visibility_revision') == '14.0':
        max_llm_frames = min(max_llm_frames, 8)
    if not frames: raise ValueError('No real captured frames')
    route_contract = None
    query_schema = QUERY_SCHEMA
    if instruction_style == 'route_grounded':
        from cave.lunar_route_language_v13 import ROUTE_QUERY_SCHEMA, validate_materialized_episode
        route_contract = validate_materialized_episode(source, metadata, frames)
        query_schema = ROUTE_QUERY_SCHEMA
    elif metadata.get('instruction_style') == 'route_grounded':
        raise ValueError('A derived route episode requires --instruction-style route_grounded')
    if query_count not in (3, 20) or query_count == 3 and query_protocol is not None:
        raise ValueError('Use the legacy three-query protocol or the explicit twenty-query composition protocol')
    if query_count == 20:
        if instruction_style != 'route_grounded':
            raise ValueError('Twenty complex queries require route-grounded instructions')
        from cave.lunar_query_families_v14 import load_protocol, query_schema as family_schema
        query_protocol = load_protocol(query_protocol or ROOT/'configs/lunar_query_protocol_v14.json')
        query_schema = family_schema(query_protocol)
    compiled_value=compiled_provenance=loaded_bank=None
    if query_mode!='provider':
        if query_protocol is None or not query_protocol['schema_version'].endswith('/15.0'):
            raise ValueError('Compiled queries require the unchanged V15 twenty-family protocol')
        from cave.lunar_query_compiler_v15 import load_bank,compile_queries,bank_hash,VERSION
        loaded_bank=load_bank(query_bank or ROOT/'configs/lunar_query_bank_v15.json',query_protocol)
        compiled_value,compiled_provenance=compile_queries(route_contract,query_protocol,loaded_bank)
    if not skip_llm and client is None:
        raise ValueError('A configured provider client is required for actual image CoT labels')
    source_files = {'metadata.json', 'supervision.npz', 'teacher_only.npz', 'frame_records.jsonl'}
    for frame in frames:
        source_files.update(frame[key] for key in ('rgb_224', 'rgb_256', 'sensors_file', 'instances_file'))
    if route_contract is not None:
        source_files.update(('route_contract.json', 'instruction_binding.json', 'raw_capture/manifest.json',
                             'raw_capture/metadata.json', 'raw_capture/frame_records.jsonl'))
        if (source/'fork_evidence_enrichment.json').is_file():
            source_files.add('fork_evidence_enrichment.json')
        if metadata.get('semantic_visibility_revision') == '14.0':
            source_files.update(('semantic_visibility_config.json', 'semantic_visibility_enrichment.json'))
    source_hash = object_hash({name: sha256(safe_path(source, name)) for name in sorted(source_files)})
    protocol = dict(schema_version='lunar_annotation_protocol/11.0', source_sha256=source_hash,
        model=model, review_model=review_model, sparse=not all_frames, max_llm_frames=max_llm_frames,
        split=split, skip_llm=skip_llm, annotation_domain='isaac_continuous', training_ready=False,
        code_sha256=object_hash({name: sha256(ROOT/name) for name in ('annotate_lunar_trajectory_v11_ppio.py', 'cave/ppio_client_v11.py')}))
    if query_model is not None:
        protocol['query_model']=query_model
    if loaded_bank is not None:
        protocol.update(query_mode=query_mode,query_bank_sha256=bank_hash(loaded_bank),query_compiler_version=VERSION,
                        query_compiler_code_sha256=sha256(ROOT/'cave/lunar_query_compiler_v15.py'))
    if cot_style != 'legacy': protocol['cot_style'] = cot_style
    if route_contract is not None:
        protocol.update(instruction_style=instruction_style, route_contract_sha256=object_hash(route_contract),
                        instruction_binding_sha256=sha256(source/'instruction_binding.json'),
                        route_code_sha256=sha256(ROOT/'cave/lunar_route_language_v13.py'))
        if (source/'fork_evidence_enrichment.json').is_file():
            protocol['fork_evidence_enrichment_sha256'] = sha256(source/'fork_evidence_enrichment.json')
        if metadata.get('semantic_visibility_revision') == '14.0':
            protocol.update(semantic_visibility_revision='14.0',
                semantic_visibility_config_sha256=metadata['semantic_visibility_config_sha256'],
                semantic_enrichment_sha256=sha256(source/'semantic_visibility_enrichment.json'))
    if query_protocol is not None:
        from cave.lunar_query_families_v14 import protocol_hash
        query_code = ('cave/lunar_query_composition_v15.py' if query_protocol['schema_version'].endswith('/15.0')
                      else 'cave/lunar_query_families_v14.py')
        protocol.update(query_count=20, query_protocol_sha256=protocol_hash(query_protocol),
                        query_family_code_sha256=sha256(ROOT/query_code))
    protocol['settings_sha256'] = object_hash(protocol)
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text()) != protocol:
        raise ValueError('Resume settings/source hash mismatch; use a new output directory')
    if query_protocol is not None:
        write_json(output/'query_protocol.json', query_protocol)
    if loaded_bank is not None:
        write_json(output/'query_bank.json',loaded_bank)
    write_json(output/'protocol.json', protocol)
    canonical = frames[0]['instruction']; target = metadata['target_id']; episode = metadata['episode_id']
    if any(f['instruction'] != canonical or f['episode_id'] != episode or f['target_id'] != target for f in frames):
        raise ValueError('Frame identity or fixed episode instruction changed')
    query_record = dict(schema_version='lunar_query_variants/11.0', physical_trajectory_id=episode,
        split=split, canonical_instruction=canonical, variants=[], generated=False, training_ready=False)
    if route_contract is not None:
        query_record.update(instruction_style=instruction_style, route_contract_sha256=object_hash(route_contract))
    if query_protocol is not None:
        query_record.update(query_count=20, query_protocol_sha256=protocol_hash(query_protocol),
                            family_partition_counts=dict(train=16, heldout=4))
        if query_protocol['schema_version'].endswith('/15.0'):
            from cave.lunar_query_composition_v15 import operator_context
            query_record.update(query_composition_version='15.0',
                operator_context_sha256=object_hash(operator_context(route_contract)),
                goal_binding_source='public logic segment; movement core has anonymous endpoint')
    rejected = []; selected = select_frames(frames, max_llm_frames, all_frames, cot_style=cot_style,
                                           instruction_style=instruction_style, route_contract=route_contract)
    requests = []; stop_reason = None

    def generate_checked(prompt, schema, images, checker, item_id):
        review_reason = None
        initial_model=(query_model or model) if item_id=='episode_queries' else model
        review_models=([review_model] if review_model and review_model != initial_model else [])
        if item_id=='episode_queries' and query_mode=='provider_with_compiled_fallback':
            review_models=[]  # Quality failure compiles; never buys another query retry.
        for attempt_model in [initial_model]+review_models:
            if (output/'REQUEST_STOP').exists(): raise InterruptedError('REQUEST_STOP')
            attempt_prompt = prompt if review_reason is None else prompt+'\n前一次输出未通过本地一致性审核：'+review_reason+'。请严格按同一事实重新生成。'
            reuse=None
            try:
                if (item_id=='episode_queries' and query_protocol is not None
                        and query_protocol['schema_version'].endswith('/15.0')):
                    from cave.lunar_query_composition_v15 import completed_review_response
                    reuse=completed_review_response(client,prompt,schema,attempt_model,images,
                                                    query_protocol['max_output_tokens'],checker)
                if reuse is not None:
                    saved,reuse_info=reuse
                    cached=True
                elif item_id == 'episode_queries' and query_protocol is not None:
                    saved, cached = client.complete(attempt_prompt, schema, attempt_model, images,
                                                    max_tokens=query_protocol['max_output_tokens'])
                else:
                    saved, cached = client.complete(attempt_prompt, schema, attempt_model, images)
            except BudgetExceeded:
                if review_reason is None:
                    raise
                # Review is optional. A larger review reservation must not
                # prevent the primary model from covering later observations.
                rejected.append(dict(item_id=item_id, model=attempt_model,
                                     reason='Optional review exceeds remaining budget',
                                     human_review_required=True, request_sent=False))
                return None, None
            except UncertainRequest as exc:
                rejected.append(dict(item_id=item_id, model=attempt_model, request_sha256=exc.fingerprint,
                                     reason=str(exc), human_review_required=True, possibly_billed=True))
                return None, None
            requests.append(dict(item_id=item_id, request_sha256=saved['request_sha256'], model=attempt_model,
                cached=cached, estimated_cny=saved['estimated_cny']))
            if reuse is not None:
                requests[-1].update(reuse_info)
            try:
                if (item_id=='episode_queries' and query_protocol is not None
                        and query_protocol['schema_version'].endswith('/15.0')):
                    from cave.lunar_query_composition_v15 import parse_public_response
                    value,normalization=parse_public_response(saved)
                else:
                    value,unwrapped=parse_public(saved)
                    normalization=dict(single_record_array_normalized=unwrapped)
                checker(value)
                if reuse is not None:
                    normalization.update(reuse_info)
                return value, dict(request_sha256=saved['request_sha256'], model=attempt_model,
                    **normalization, public_content_only=True)
            except (ValueError, TypeError, KeyError) as exc:
                review_reason = str(exc)
                rejected.append(dict(item_id=item_id, model=attempt_model, request_sha256=saved['request_sha256'],
                                     reason=str(exc), human_review_required=True))
        return None, None

    def accept_query_value(value,provenance,*,generation_method):
        if query_protocol is None:
            query_variants=[dict(language_episode_id=episode+f':zh:{i:03d}',instruction=q,
                physical_trajectory_id=episode,split=split) for i,q in enumerate(value['queries'])]
        else:
            from cave.lunar_query_families_v14 import variants
            query_variants=variants(value,episode,split,query_protocol)
        if generation_method=='compiled':
            for variant,trace in zip(query_variants,provenance['family_trace']):
                variant.update(generation_method='grounded_template_instantiation',compiler_trace=trace,
                               query_bank_sha256=provenance['query_bank_sha256'])
        query_record.update(variants=query_variants,generated=True,provenance=provenance)
        if loaded_bank is not None:
            query_record.update(query_mode=query_mode,generation_method=generation_method,
                                provider_generated_queries=generation_method=='provider')

    if query_mode=='compiled' or skip_llm and query_mode=='provider_with_compiled_fallback':
        accept_query_value(compiled_value,compiled_provenance,generation_method='compiled')
    elif not skip_llm:
        if client is None: raise ValueError('A configured PPIO client is required')
        try:
            value, provenance = generate_checked(query_prompt(target, canonical, instruction_style=instruction_style,
                route_contract=route_contract, query_protocol=query_protocol), query_schema, (),
                lambda v: check_queries(v, target, instruction_style=instruction_style, route_contract=route_contract,
                                        query_protocol=query_protocol), 'episode_queries')
            if value:
                accept_query_value(value,provenance,generation_method='provider')
            elif query_mode=='provider_with_compiled_fallback':
                fallback=copy.deepcopy(compiled_provenance)
                fallback.update(fallback_reason='provider_query_quality_rejected',
                    rejected_provider_request_sha256=[r['request_sha256'] for r in requests if r['item_id']=='episode_queries'],
                    prior_provider_query_attempts=sum(r['item_id']=='episode_queries' for r in requests),
                    prior_provider_query_new_HTTP_requests=sum(r['item_id']=='episode_queries' and not r['cached'] for r in requests))
                accept_query_value(compiled_value,fallback,generation_method='compiled')
            elif query_protocol is not None:
                stop_reason = 'QueryBatchRejected'
        except (RuntimeError, BudgetExceeded, InterruptedError) as exc:
            stop_reason = type(exc).__name__
    write_json(output/'queries.json', query_record)
    rows = []
    for frame in frames:
        row = dict(schema_version='lunar_step_cot/11.0', sample_id=frame['sample_id'], episode_id=episode,
            physical_trajectory_id=episode, split=split, target_id=target, step_index=frame['step_index'],
            annotation_domain='isaac_continuous', annotation_mode='trajectory_step', annotation_status='annotation_proto',
            training_ready=False, inputs=copy.deepcopy(frame['inputs']), machine_evidence=copy.deepcopy(frame['machine_evidence']),
            cot_valid=False, output=None, selection_events=selected.get(frame['sample_id'], []), path_base='trajectory')
        if frame['sample_id'] in selected and not skip_llm and not stop_reason:
            try:
                value, provenance = generate_checked(frame_prompt(frame, cot_style=cot_style,
                    instruction_style=instruction_style, route_contract=route_contract), OUTPUT_SCHEMA,
                    annotation_images(source, frame), lambda v: check_output(v, frame, cot_style=cot_style,
                    instruction_style=instruction_style, route_contract=route_contract), frame['sample_id'])
                if value: row.update(cot_valid=True, output=value, provenance=provenance)
            except (RuntimeError, BudgetExceeded, InterruptedError) as exc:
                stop_reason = type(exc).__name__
            except (ValueError, TypeError, KeyError) as exc:
                rejected.append(dict(item_id=frame['sample_id'], reason=str(exc), human_review_required=True))
        if cot_style != 'legacy':
            row.update(cot_style=cot_style, cot_text=render_cot_text(row['output']) if row['cot_valid'] else None)
        if route_contract is not None:
            row.update(instruction_style=instruction_style, route_contract_sha256=object_hash(route_contract))
        rows.append(row)
        if (metadata.get('semantic_visibility_revision') != '14.0' or frame['sample_id'] in selected
                or len(rows) % 64 == 0 or len(rows) == len(frames)):
            # Resolved paid responses are already fsync-cached. Avoid quadratic
            # rewrites of dense unlabeled rows in the large V14 collection.
            write_jsonl(output/'annotations.jsonl', rows)
    write_json(output/'rejected.json', rejected)
    write_json(output/'human_review.json', dict(instructions='Review against source RGB and machine evidence; do not modify machine facts to fit generated text.',
        edits_file='human_review_edits.jsonl (optional; record sample_id,before,after,reason,reviewer)', items=rejected))
    report = dict(schema_version='lunar_annotation_report/11.0', complete=stop_reason is None,
        source_sha256=source_hash, frame_count=len(frames), selected_count=len(selected),
        cot_valid_count=sum(r['cot_valid'] for r in rows), rejected_attempt_count=len(rejected),
        generated_queries=query_record['generated'], llm_skipped=skip_llm, stop_reason=stop_reason,
        requests=requests, shared_cache_usage=None if client is None else client.usage(),
        annotation_domain='isaac_continuous', training_ready=False,
        current_129_text_loss_supported=False, adapter_required=True,
        policy_inputs=['fixed instruction', 'current RGB-D', 'strictly past same-episode observations'])
    if cot_style != 'legacy': report['cot_style'] = cot_style
    if route_contract is not None: report['instruction_style'] = instruction_style
    if query_protocol is not None:
        report.update(query_count=20, query_protocol_sha256=protocol_hash(query_protocol))
    if query_model is not None:
        report['query_model']=query_model
    if loaded_bank is not None:
        report.update(query_mode=query_mode,query_generation_method=query_record.get('generation_method'),
            query_bank_sha256=bank_hash(loaded_bank),provider_generated_queries=query_record.get('provider_generated_queries',False),
            fresh_query_HTTP_requests=sum(r['item_id']=='episode_queries' and not r['cached'] for r in requests))
    write_json(output/'report.json', report)
    write_json(output/'status.json', dict(phase='complete' if report['complete'] else 'partial', **report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--skip-llm', action='store_true')
    parser.add_argument('--model', default=MAIN_MODEL); parser.add_argument('--review-model', default=REVIEW_MODEL)
    parser.add_argument('--ppio-env', type=Path); parser.add_argument('--cache', type=Path)
    parser.add_argument('--budget-cny', type=float, default=2.); parser.add_argument('--max-llm-frames', type=int, default=16)
    parser.add_argument('--all-frames', action='store_true'); parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--cot-style', choices=COT_STYLES, default='legacy')
    parser.add_argument('--instruction-style', choices=INSTRUCTION_STYLES, default='legacy')
    parser.add_argument('--require-semantic-visibility', choices=['v14'])
    parser.add_argument('--query-count', type=int, choices=[3,20], default=3)
    parser.add_argument('--query-protocol', type=Path)
    parser.add_argument('--query-model')
    parser.add_argument('--query-mode',choices=QUERY_MODES,default='provider')
    parser.add_argument('--query-bank',type=Path)
    args = parser.parse_args(argv)
    if args.require_semantic_visibility and json.loads((args.source/'metadata.json').read_text()).get('semantic_visibility_revision') != '14.0':
        raise ValueError('The requested semantic V14 derived source is required')
    client = None if args.skip_llm else PPIOClient(args.cache or args.output/'provider_cache', args.budget_cny, args.ppio_env)
    report = annotate(args.source, args.output, client=client, skip_llm=args.skip_llm, resume=args.resume,
        max_llm_frames=args.max_llm_frames, all_frames=args.all_frames, model=args.model,
        review_model=args.review_model, split=args.split, cot_style=args.cot_style, instruction_style=args.instruction_style,
        query_count=args.query_count, query_protocol=args.query_protocol, query_model=args.query_model,
        query_mode=args.query_mode,query_bank=args.query_bank)
    print(json.dumps({k: v for k, v in report.items() if k != 'requests'}, ensure_ascii=False))
    return 0 if report['complete'] else 1


if __name__ == '__main__': raise SystemExit(main())
