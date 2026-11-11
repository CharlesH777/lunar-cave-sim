"""CPU regressions for class gates, causal STOP exception and legacy replay."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from cave import lunar_semantic_visibility_v14 as helper


LEGACY_GOLDEN_SHA256 = '5ca7335f11ab67da35b52f6c1cb6ef1d0c864317aecf568e83761a0369f68798'
NAMES = ('astronaut', 'lunar_rover', 'lunar_base', 'chinese_flag')
TARGETS = [dict(id=name, prim_path='/Search/' + name) for name in NAMES]
MAPPING = {'idToLabels': {str(i+1): row['prim_path'] for i, row in enumerate(TARGETS)}}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def payload(name='lunar_base', box=(100, 105, 150, 140), *, invalid_depth=False, fx=128):
    image = np.zeros((256, 256), np.int32)
    x0,y0,x1,y1 = box
    image[y0:y1,x0:x1] = NAMES.index(name) + 1
    depth = np.full((256, 256), 3., np.float32)
    valid = np.ones((256, 256), bool)
    if invalid_depth:
        depth[image != 0] = np.nan
    K = np.asarray([[fx,0,128],[0,128,128],[0,0,1]], float)
    return image, MAPPING, K, depth, valid


def observe(observer, step, data, terminal=False):
    return observer.observe(*data, step*.2, step, terminal_stop=terminal)


def row(state, name='lunar_base'):
    return next(value for value in state['semantic_object_diagnostics'] if value['id'] == name)


def legacy_states(module):
    config = deepcopy(helper.LEGACY_DEFAULT_CONFIG)
    result = []
    for goal in NAMES:
        observer = module.SemanticVisibility(TARGETS, goal, config)
        result.append([observe(observer, 0, payload(goal)),
                       observe(observer, 1, payload(goal, (0,80,160,220))),
                       observe(observer, 5, payload(goal, (0,80,160,220)), terminal=True)])
    return result


def synthetic_tests(legacy_reference=None):
    config = helper.checked_config()
    old = deepcopy(helper.LEGACY_DEFAULT_CONFIG)
    assert helper.checked_config(old) == old and set(helper.checked_config(old)) == set(old)
    assert helper.config_hash(old) == '1aa1077b24c5d53a74a1e88379f2a9b1c1458cdb24ed0e2c2df2ad273f2a5035'
    assert helper.legacy_replay_module_hashes(old) == (helper.LEGACY_MODULE_SHA256,)
    assert helper.legacy_replay_module_hashes(config) == ()
    legacy = legacy_states(helper)
    assert canonical_sha(legacy) == LEGACY_GOLDEN_SHA256
    if legacy_reference:
        assert hashlib.sha256(legacy_reference.read_bytes()).hexdigest() == helper.LEGACY_MODULE_SHA256
        spec = importlib.util.spec_from_file_location('cave._legacy_semantic_visibility_qa',legacy_reference)
        prior = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prior)
        assert legacy == legacy_states(prior)
    large = payload(box=(0,80,160,220))
    observer = helper.SemanticVisibility(TARGETS,'lunar_base',config)
    first = observe(observer,0,payload())
    assert row(first)['salient_candidate'] and first['semantic_cue_first_seen'] == ['lunar_base']
    stop = observe(observer,5,large,terminal=True)
    accepted = row(stop)
    assert accepted['salient_candidate'] and accepted['strict_salient_candidate'] is False
    assert accepted['strict_rejection_reasons'] == ['edge_clipped_or_insufficient_margin']
    assert accepted['terminal_goal_edge_exception']['prior_strict_salient_grid_step_index'] == 0
    assert not stop['semantic_cue_first_seen'] and 'lunar_base' in stop['semantic_cue_seen_before']
    rejected = []
    def case(name,*,prior=True,prior_step=0,terminal=True,goal='lunar_base',data=large,cfg=config):
        observer = helper.SemanticVisibility(TARGETS,goal,cfg)
        if prior:
            observe(observer,prior_step,payload(goal))
        result = observe(observer,5,data,terminal=terminal)
        assert not row(result)['salient_candidate'], name
        assert 'terminal_goal_edge_exception' not in row(result), name
        rejected.append(name)
    case('no_prior_strict_cue',prior=False)
    case('offgrid_prior_cannot_authorize',prior_step=1)
    case('nonterminal_cannot_waive_edge',terminal=False)
    case('non_goal_cannot_receive_exception',goal='astronaut')
    case('invalid_current_depth',data=payload(box=(0,80,160,220),invalid_depth=True))
    case('small_partial_target',data=payload(box=(100,220,148,256)))
    case('wrong_current_bearing',data=payload(box=(0,80,160,220),fx=64))
    # Even a large object may not waive a stricter normal bounding-box rule.
    stricter = deepcopy(config)
    stricter['class_overrides']['lunar_base']['min_bbox_rgb224'] = [160,100]
    observer = helper.SemanticVisibility(TARGETS,'lunar_base',stricter)
    assert row(observe(observer,0,payload(box=(20,20,236,236))))['salient_candidate']
    result = observe(observer,5,large,terminal=True)
    assert not row(result)['salient_candidate'] and 'bbox_too_small' in row(result)['rejection_reasons']
    rejected.append('non_edge_bbox_check_retained')
    disabled = deepcopy(config)
    disabled['terminal_goal_edge_exception']['enabled'] = False
    case('disabled_exception',cfg=disabled)
    legacy_observer = helper.SemanticVisibility(TARGETS,'lunar_base',old)
    observe(legacy_observer,0,payload())
    assert not row(observe(legacy_observer,5,large,terminal=True))['salient_candidate']
    assert helper.checked_config(old) == old
    return dict(passed=True,legacy_exact_replay_sha256=LEGACY_GOLDEN_SHA256,
                negative_cases=rejected,large_clipped_actual_stop_passed=True)


def real_episode(source,layout,audit):
    config = helper.checked_config()
    stats = json.loads(audit.read_text())
    target = stats['goal']
    def passes(value):
        w,h = value['bbox_width_height_rgb224']
        return value['pixels_rgb224'] >= 800 and w >= 40 and h >= 28 and value['edge_margin_rgb224'] >= 8 and abs(value['bearing_deg'] or 0) <= 30 and value['valid_depth_fraction'] >= .9
    first = min((value for value in stats['all_grid_measurements'][target] if passes(value)),key=lambda value:value['step_index'])
    frames = {value['step_index']:value for value in map(json.loads,(source/'frame_records.jsonl').read_text().splitlines())}
    targets = json.loads((layout/'targets.json').read_text())['targets']
    observer = helper.SemanticVisibility(targets,target,config)
    results = []
    for step in (first['step_index']-5,first['step_index'],stats['frame_count']-1):
        frame = frames[step]
        with np.load(source/frame['sensors_file'],allow_pickle=False) as sensor:
            result = observer.observe(sensor['instance_id'],json.loads((source/frame['instances_file']).read_text()),
                sensor['K256'],sensor['depth_m'],sensor['depth_valid'],frame['inputs']['observation']['timestamp_s'],step,
                terminal_stop=bool(frame['executed_action']['stop']))
        results.append(row(result,target))
    assert not results[0]['salient_candidate']
    assert results[1]['salient_candidate'] and results[1]['pixels_rgb224'] == first['pixels_rgb224']
    assert results[2]['salient_candidate'] and results[2]['terminal_goal_edge_exception']['applied']
    assert results[2]['terminal_goal_edge_exception']['prior_strict_salient_grid_step_index'] == first['step_index']
    return dict(passed=True,first_step=first['step_index'],first_pixels=results[1]['pixels_rgb224'],
        terminal_step=stats['frame_count']-1,terminal_pixels=results[2]['pixels_rgb224'],
        terminal_waived_only=results[2]['terminal_goal_edge_exception']['waived_check'],sensors_read_count=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-reference',type=Path)
    parser.add_argument('--real-source',type=Path)
    parser.add_argument('--real-layout',type=Path)
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args()
    report=synthetic_tests(args.legacy_reference)
    if any((args.real_source,args.real_layout,args.audit)):
        if not all((args.real_source,args.real_layout,args.audit)):parser.error('Provide all real episode options')
        report['real_episode']=real_episode(args.real_source,args.real_layout,args.audit)
    report['config_sha256']=helper.config_hash(helper.checked_config())
    report['module_sha256']=hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
