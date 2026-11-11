"""Grounded short navigation explanations from real probes; not train-ready episodes.

Uses the existing local PPIO credential reader and one-shot HTTP client. No
secret, Authorization header or model-internal reasoning is written to disk.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import re
import time

from build_lunar_surface_detail_v5_cpu import sha,write

ROOT=Path(__file__).resolve().parent
MODEL='qwen/qwen3.8-flash'
ENDPOINT='https://api.ppio.com/openai/v1/chat/completions'
INPUT_CNY_PER_M=0.8
OUTPUT_CNY_PER_M=2.7
KEYS=['instruction','progress','observation','decision_reason','action_intent']
RESPONSE_SCHEMA=dict(type='object',additionalProperties=False,required=KEYS,properties={
    'instruction':dict(type='string'), 'progress':dict(type='string',enum=['到达洞口','进入洞穴','洞内搜索','接近目标','暂无法确认']),
    'observation':dict(type='string'),'decision_reason':dict(type='string'),
    'action_intent':dict(type='string',enum=['turn_left','turn_right','move_forward','explore'])})


def existing_client():
    path=ROOT.parents[1]/'3d_encoder/multimodal_bbox_annotator/batch_pipeline.py'
    spec=importlib.util.spec_from_file_location('existing_ppio_client',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def check_output(value,frame):
    if not isinstance(value,dict) or set(value)!=set(KEYS) or any(not isinstance(value[k],str) for k in KEYS):
        raise ValueError('Response fields do not match the annotation schema')
    limits={'instruction':(4,100),'progress':(2,100),'observation':(2,60),'decision_reason':(2,80)}
    for key,(low,high) in limits.items():
        if not low<=len(value[key])<=high:raise ValueError('Invalid annotation length: '+key)
    evidence=frame['machine_evidence']
    aliases={'astronaut':['宇航员'],'lunar_rover':['探月车','月球车'],
             'lunar_base':['基地'],'chinese_flag':['国旗','红旗']}
    if not any(word in value['instruction'] for word in aliases[evidence['target_id']]):
        raise ValueError('Generated language query changed the requested target')
    if re.search(r'左转|右转|向左|向右|左侧|右侧|坐标',value['instruction']):
        raise ValueError('Task query must not include privileged local direction labels')
    if value['action_intent']!=evidence['action_intent']:
        raise ValueError('Text action disagrees with the verified geometric probe intent')
    if value['progress']=='进入洞穴':raise ValueError('Independent probe cannot establish a temporal entry event')
    if value['progress']=='到达洞口' and not evidence['at_cave_entrance']:
        raise ValueError('Cave-entry progress without the current entry geometry case')
    if not evidence['target_visible']:
        observation=value['observation']
        if re.search(r'看到|看见|可见|见到',observation) and not re.search(r'未|没|无法|看不|不见|尚无',observation):
            raise ValueError('Positive visibility claim without a rendered target mask')
    text=' '.join(value.values())
    direction_text=value['observation']+' '+value['decision_reason']
    intent=evidence['action_intent']
    if intent=='turn_left' and re.search(r'右转|向右|右侧|偏右|在右|右前|右边',direction_text):
        raise ValueError('Explanation direction disagrees with the left-turn evidence')
    if intent=='turn_right' and re.search(r'左转|向左|左侧|偏左|在左|左前|左边',direction_text):
        raise ValueError('Explanation direction disagrees with the right-turn evidence')
    if intent=='explore' and re.search(r'左转|右转|向左|向右',direction_text):
        raise ValueError('Unseen target cannot justify a directed target-alignment turn')
    for match in re.finditer(r'雪地|积雪|下雪|冰雪|白雪|草地|树林|水池|河流',direction_text):
        prefix=direction_text[max(0,match.start()-8):match.start()]
        if not re.search(r'(?:无|没有|未见|不是|并非|不存在)[^，。；]{0,5}$',prefix):
            raise ValueError('Invented snow, vegetation or water in the dry lunar cave')
    if re.search(r'\d+\s*(米|m\b|度|degrees)',text):raise ValueError('Invented precise distance/angle in explanation')
    if not evidence['at_cave_entrance'] and re.search(r'已到.*洞口|到达.*洞口|位于.*洞口',value['progress']):
        raise ValueError('Cave-entry progress without a corresponding geometry case')
    return True


def main():
    global MODEL,INPUT_CNY_PER_M,OUTPUT_CNY_PER_M
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=9)
    parser.add_argument('--model',choices=['qwen/qwen3.8-flash','qwen/qwen3.8-27b','qwen/qwen3.8-max'],default='qwen/qwen3.8-27b')
    parser.add_argument('--sample-ids',nargs='+',help='Review only the named rejected probes')
    parser.add_argument('--keep-going',action='store_true',help='Record and exclude inconsistent labels while finishing the bounded batch')
    parser.add_argument('--generate',action='store_true',help='Explicitly execute the paid annotation requests')
    parser.add_argument('--resume',action='store_true',help='Reuse already saved responses; never pay for the same saved frame twice')
    args=parser.parse_args();source=args.source.resolve(strict=True);output=args.output.absolute()
    MODEL=args.model
    INPUT_CNY_PER_M,OUTPUT_CNY_PER_M=(12.,36.) if MODEL.endswith('-max') else ((3.,12.) if MODEL.endswith('-27b') else (.8,2.7))
    output.mkdir(parents=True,exist_ok=args.resume)
    frames=[json.loads(line) for line in (source/'frame_records.jsonl').read_text().splitlines() if line.strip()]
    if args.sample_ids:
        wanted=set(args.sample_ids);frames=[r for r in frames if r['sample_id'] in wanted]
        if {r['sample_id'] for r in frames}!=wanted:raise ValueError('Unknown requested probe id')
    if not 1<=args.limit<=12:raise ValueError('Prototype request limit must be within 1..12')
    frames=frames[:args.limit]
    if not frames:raise ValueError('No real sensor frames to annotate')
    client=existing_client();settings=client.read_export_env(Path('/home/charles/.config/claude-ppio.env'))
    import jsonschema
    annotation_schema=json.loads((ROOT/'schemas/step_query_cot_v9.schema.json').read_text())
    validator=jsonschema.Draft202012Validator(annotation_schema)
    key=settings.get('ANTHROPIC_API_KEY') or settings.get('PPIO_API_KEY')
    if not key:raise RuntimeError('Local PPIO credential unavailable')
    write(output/'protocol.json',dict(model=MODEL,endpoint=ENDPOINT,request_limit=len(frames),
        max_output_tokens_per_request=700,automatic_retries=False,max_estimated_total_cny=.20,
        output_scope='brief externally verifiable navigation summaries, not internal reasoning',
        independent_sensor_probes=True,continuous_controls_executed=False,training_ready=False,
        input_cny_per_m=INPUT_CNY_PER_M,output_cny_per_m=OUTPUT_CNY_PER_M,
        inputs_are_causal=True,absolute_goal_coordinates_sent_to_model=False))
    completed=[];rejected=[];usage_total=dict(prompt_tokens=0,completion_tokens=0);cost=0.;started=time.monotonic()
    for frame in frames:
        evidence=frame['machine_evidence'];image=(source/frame['rgb_224']).resolve(strict=True)
        if not image.is_relative_to(source):raise ValueError('RGB path escaped the captured dataset')
        facts={k:evidence[k] for k in ['target_id','target_class','target_visible','target_visible_pixels',
            'bearing_deg','at_cave_entrance','action_intent','action_intent_source']}
        if not evidence['target_visible']:facts['bearing_deg']=None
        prompt=(
            '为月洞导航生成中文语言query和当前帧短决策说明。这是干燥的月球熔岩洞，灰色月壤和碎石不是雪地，无积雪、水体或植被。'
            '图片是真实当前传感器RGB；机器事实是监督核验信息，不是策略输入。'
            '输出严格JSON，只有instruction/progress/observation/decision_reason/action_intent五字段。'
            'instruction将给定任务自然改写，保持同一个目标，不包含路线答案、方向、坐标或距离。'
            'progress只选到达洞口/洞内搜索/接近目标/暂无法确认；at_cave_entrance=true必须选到达洞口且observation可说“我已到洞穴入口”。'
            '独立静态probe禁止选进入洞穴；at_cave_entrance=false时也绝对禁止选到达洞口，即使背景看得到洞口。'
            '当前目标的bearing为正必须说目标在左侧，负值必须说目标在右侧，不得把背景方位当目标方位。'
            'observation不超过60字，decision_reason不超过80字。'
            'observation仅写图片与实例mask证实的现象；target_visible=false必须说尚未确认目标，不能说看到了目标。'
            'decision_reason只用一两句简短可核验的行动理由。若目标可见且偏左/右，可说“我看到了目标，目标偏左/右，应先向左/右转对准”。'
            '这些话直接面向导航用户，不要出现bearing、mask、像素、机器事实或标注流程等字段名。'
            '若目标不可见，应继续观察寻找，不能根据隐藏目标坐标转弯。action_intent必须严格等于机器给定值。'
            '左右按当前车体/相机局部坐标，bearing正是左，负是右。不要写内部推理、精确角度、米数、未来经历或不存在的障碍。'
            '这是独立静态观测probe，方向是几何对准建议，尚无真实执行控制轨迹，不要声称已执行动作。'
            '\n原始任务：'+frame['instruction']+'\n当前机器事实：'+json.dumps(facts,ensure_ascii=False))
        payload=dict(model=MODEL,temperature=.35,max_tokens=700,enable_thinking=False,
            response_format=dict(type='json_schema',json_schema=dict(name='navigation_step_annotation',strict=True,schema=RESPONSE_SCHEMA)),
            messages=[dict(role='user',content=[dict(type='text',text=prompt),
                dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode(),detail='high'))])])
        identifier=frame['sample_id']
        audit_file=output/(identifier+'_request_audit.json')
        if audit_file.exists():
            old_audit=json.loads(audit_file.read_text())
            if old_audit['model']!=MODEL or old_audit['image_sha256']!=sha(image):
                raise ValueError('Saved response belongs to a different model or image')
        else:write(audit_file,dict(model=MODEL,prompt=prompt,
            image_path=str(image),image_sha256=sha(image),response_schema=RESPONSE_SCHEMA,
            max_tokens=700,enable_thinking=False))
        if not args.generate:continue
        if cost>=.18:raise RuntimeError('Prototype cost guard reached before the next request')
        response_file=output/(identifier+'_response.json')
        if args.resume and response_file.exists():
            saved=json.loads(response_file.read_text())
        else:
            try:
                response=client.post_json(ENDPOINT,payload,{'Authorization':'Bearer '+key},timeout=120)
            except Exception as exc:
                write(output/'error.json',dict(sample_id=identifier,error_type=type(exc).__name__,
                    http_status=getattr(exc,'code',None),automatic_retry=False))
                raise RuntimeError('PPIO request failed; see sanitized error.json') from None
            choice=response['choices'][0]
            # Deliberately omit reasoning_content, request headers and image base64.
            saved=dict(model=response.get('model'),content=choice['message'].get('content') or '',
                usage=response.get('usage',{}),finish_reason=choice.get('finish_reason'))
            write(response_file,saved)
        content=saved['content'];usage=saved['usage']
        for field in usage_total:usage_total[field]+=int(usage.get(field,0))
        cost=(usage_total['prompt_tokens']*INPUT_CNY_PER_M+usage_total['completion_tokens']*OUTPUT_CNY_PER_M)/1_000_000
        if saved.get('finish_reason')!='stop':raise ValueError('Incomplete structured annotation')
        try:
            value=json.loads(content)
            array_normalized=isinstance(value,list) and len(value)==1 and isinstance(value[0],dict)
            if array_normalized:value=value[0]
            check_output(value,frame)
            jsonschema.Draft202012Validator(annotation_schema['properties']['output']).validate(
                {k:value[k] for k in KEYS if k!='instruction'})
        except (ValueError,jsonschema.ValidationError) as exc:
            rejected.append(dict(sample_id=identifier,reason=str(exc),response_file=response_file.name))
            write(output/'rejected.json',rejected)
            print(json.dumps(dict(sample_id=identifier,accepted=False,rejection_reason=str(exc))),flush=True)
            if args.keep_going:continue
            raise
        visible=bool(evidence['target_visible']);bearing=float(evidence['bearing_deg']) if visible else None
        normalized=dict(status='verified_causal_observations',method='instance_mask_and_calibrated_camera',
            source_observation_ids=[identifier],target_visible=visible,
            visible_pixel_count=int(evidence['target_visible_pixels']),target_bearing_deg=bearing,
            target_bearing='unknown' if not visible else ('left' if bearing>10 else ('right' if bearing < -10 else 'ahead')),
            entry_event='at_cave_entrance' if evidence['at_cave_entrance'] else 'none',
            expected_action_intent=evidence['action_intent'])
        row=dict(schema_version='step_query_cot_v9',annotation_status='annotation_proto',
            annotation_mode='independent_probe',training_ready=False,sample_id=identifier,
            episode_id=None,target_id=evidence['target_id'],step_index=0,
            inputs=dict(instruction=value['instruction'],observation=dict(observation_id=identifier,
                rgb=str(image),depth=str(source/frame['sensors_file']),timestamp_s=None),history=[]),
            machine_evidence=normalized,output={k:value[k] for k in KEYS if k!='instruction'})
        validator.validate(row);completed.append(row)
        write(output/(identifier+'_provenance.json'),dict(provider='PPIO',model=MODEL,
            real_sensor_record=str(source/'frame_records.jsonl'),real_rgb_sha256=sha(image),
            single_record_array_envelope_normalized=array_normalized,
            scope='observed progress and geometric alignment intent, not executed action GT'))
        (output/'annotations.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in completed))
        write(output/'status.json',dict(phase='annotating',completed=len(completed),requested=len(frames),
            usage=usage_total,estimated_cny=cost,training_ready=False))
        print(json.dumps(dict(sample_id=identifier,completed=len(completed),estimated_cny=round(cost,6))),flush=True)
    write(output/'report.json',dict(passed=bool(args.generate and len(completed)==len(frames)),
        generated=args.generate,model=MODEL,sample_count=len(completed),usage=usage_total,
        requested_sample_count=len(frames),rejected_sample_count=len(rejected),rejected_ids=[r['sample_id'] for r in rejected],
        estimated_cny=cost,price_scope='listed per-token estimate, not a billing receipt',
        real_observation_grounded=True,continuous_teacher_controls_present=False,
        text_cot_loss_currently_supported_by_129=False,training_ready=False,
        elapsed_seconds=time.monotonic()-started,updated_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
