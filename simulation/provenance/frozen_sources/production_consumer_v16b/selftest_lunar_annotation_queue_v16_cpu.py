"""Isolated queue/retention regressions with no HTTP, renderer or SSH."""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cave.lunar_annotation_retention_v16 import reclaim_owned_frames
from cave.lunar_dataset_delivery_v14 import build_unit_manifest, canonical_bytes, manifest_digest
from cave.ppio_client_v11 import object_hash, sha256, write_json
from run_lunar_annotation_queue_v16 import AnnotationQueue, TARGETS, read, validate_ready


def proof(source, state, unit):
    state.mkdir(parents=True,exist_ok=True)
    manifest=build_unit_manifest(source)
    digest=manifest_digest(manifest)
    (state/'manifest.json').write_bytes(canonical_bytes(manifest))
    receipt=dict(passed=True,host='fake-a100',remote_root='/fake/remote',unit=unit,manifest_sha256=digest,
        **{key:manifest[key] for key in ('entry_count','file_count','total_file_bytes')})
    write_json(state/'receipt.json',receipt)
    return dict(passed=True,manifest_sha256=digest,host='fake-a100',remote_destination='/fake/remote/'+unit,
        state_dir=str(state),manifest_path=str(state/'manifest.json'),receipt_path=str(state/'receipt.json'),receipt=receipt)


class Fixture:
    def __init__(self,root):
        self.root=root;self.producer=root/'producer';self.producer.mkdir()
        self.output=root/'consumer';self.cache=root/'cache';self.cache.mkdir()
        self.env=root/'provider.env';self.env.write_text('TEST_ONLY=1\n')
        self.plan=dict(schema_version='lunar_raw_plan/16.0',config_sha256='test',
            slots=[dict(physical_slot=0,requested_seed=123,split='train')],targets=list(TARGETS),
            requested_units=4,expected_layout_splits=dict(train=1,val=0,test=0))
        write_json(self.producer/'plan.json',self.plan)
        resources={}
        for name in ('semantic_config','query_protocol','query_bank'):
            path=root/(name+'.json');write_json(path,{'fixture':name});resources[name]=str(path)
        template=root/'unit_template.json'
        write_json(template,dict(schema_version='lunar_dataset_pipeline/14.0',terrain_template='unused',
            annotation=dict(**resources,query_count=20),training_view=dict(enabled=True,stride=5)))
        config=root/'queue.json'
        write_json(config,dict(schema_version='lunar_annotation_queue/16.0',consumer_revision='fixture16',
            producer_root=str(self.producer),output=str(self.output),ppio_env=str(self.env),provider_cache=str(self.cache),
            unit_config=str(template),publish_host='fake-a100',remote_root='/fake/remote',requested_units=4,
            expected_layout_splits=dict(train=1,val=0,test=0),retention=dict(enabled=False)))
        self.args=SimpleNamespace(config=config,producer_root=None,output=None,python=None,ppio_env=None,
            provider_cache=None,resume=False,max_new_units=None,once=True)
        self.calls=[];self.published=[];self.verified=[];self.publication_options=[]

    def ready(self,target,*,bad_future=False,capture_seed=123,source_layout_seed=None):
        identity='seed_123_'+target
        source=self.producer/'raw_inputs'/identity;layout=source/'layout';raw=source/'trajectory'
        layout.mkdir(parents=True);raw.mkdir()
        write_json(layout/'targets.json',dict(seed=123))
        write_json(layout/'scene_config.json',dict(fixture=True))
        controls=np.array([[.1,0],[.1,.1],[0,0]],np.float32);n=len(controls)
        future=np.zeros((n,8,2),np.float32);valid=np.zeros((n,8),bool)
        for index in range(n):
            count=min(8,n-index);future[index,:count]=controls[index:index+count];valid[index,:count]=True
        if bad_future:future[0,1,0]=.3
        np.savez(raw/'supervision.npz',vw_si=controls,stop=np.array([0,0,1],np.uint8),dt_s=np.full(n,.2,np.float32),future_vw_si=future,horizon_valid=valid)
        np.savez(raw/'teacher_only.npz',executed_positions=np.zeros((n+1,3),np.float32))
        frames=[]
        for index,control in enumerate(controls):
            frames.append(dict(step_index=index,episode_id=identity,target_id=target,
                inputs=dict(observation=dict(timestamp_s=index*.2)),executed_action=dict(v_mps=float(control[0]),
                    omega_radps=float(control[1]),stop=int(index==n-1))))
        (raw/'frame_records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in frames))
        write_json(raw/'metadata.json',dict(episode_id=identity,target_id=target,seed=capture_seed,
            source_layout_seed=source_layout_seed,terminal_success=True,
            capture_stride=1,captured_frame_count=n,source_sha256={name:sha256(layout/name) for name in ('targets.json','scene_config.json')}))
        write_json(raw/'status.json',dict(phase='complete'))
        write_json(source/'raw_validation.json',dict(passed=True))
        write_json(source/'source_binding.json',dict(unit_id=identity))
        files=[dict(path=p.relative_to(source).as_posix(),size_bytes=p.stat().st_size,sha256=sha256(p))
               for p in sorted(source.rglob('*')) if p.is_file()]
        critical={row['path']:row['sha256'] for row in files}
        manifest=dict(schema_version='lunar_raw_unit/16.0',complete=True,annotation_domain='isaac_continuous',
            unit_id=identity,layout_seed=123,target_id=target,split='train',physical_slot=0,ordinal=TARGETS.index(target),
            layout='layout',trajectory='trajectory',raw_validation='raw_validation.json',source_binding='source_binding.json',
            frames=n,action_count=n,pose_count=n+1,native_action_dt_s=.2,future_horizon=8,
            critical_sha256=critical,payload_files=files,payload_inventory_sha256=object_hash(files))
        write_json(source/'raw_manifest.json',manifest)
        delivery=proof(source,self.producer/'delivery_state'/identity,'raw_inputs/'+identity)
        ready=dict(schema_version='lunar_raw_ready/16.0',unit_id=identity,layout_seed=123,target_id=target,split='train',
            physical_slot=0,ordinal=TARGETS.index(target),plan_sha256=sha256(self.producer/'plan.json'),
            raw_manifest_sha256=sha256(source/'raw_manifest.json'),producer_relative_root='raw_inputs/'+identity,
            source=dict(root=str(source),layout='layout',raw='trajectory',files_sha256=dict(critical,**{'raw_manifest.json':sha256(source/'raw_manifest.json')})),
            raw_delivery=delivery)
        path=self.producer/'queue/ready'/(identity+'.json');write_json(path,ready)
        return path

    def runner(self,command,root,record):
        self.calls.append(record['unit_id'])
        if record['unit_id'].endswith('astronaut'):
            root.mkdir(parents=True);write_json(root/'status.json',dict(phase='failed',passed=False))
            (self.cache/'paid_response_preserved.json').write_text('{"fixture":"paid response"}\n')
            return 1
        root.mkdir(parents=True);(root/'dataset').mkdir()
        configuration=read(Path(command[command.index('--config')+1]))
        write_json(root/'dataset/run_config.json',dict(config=configuration,code_sha256=object_hash(self.queue.codes)))
        write_json(root/'status.json',dict(phase='complete',passed=True))
        return 0

    def summary(self,root,**expected):
        return dict(episode_id=root.name,layout_seed=expected['expected_seed'],target_id=expected['expected_target'],
            split=expected['expected_split'],training_family_ids=['train.'+str(i) for i in range(16)],
            heldout_family_ids=['heldout.'+str(i) for i in range(4)],query_protocol_sha256='same_protocol',
            query_indexes={name:dict(path='query_partitions/'+name+'.jsonl',rows=1,language_episodes=1,physical_episodes=1)
                for name in ('query_train','query_eval','physical_val','physical_test')},physical_frames=3,
            required_paths=['status.json','dataset/run_config.json'],
            critical_sha256={'status.json':sha256(root/'status.json')})

    def publisher(self,source,**kwargs):
        self.published.append(kwargs['unit'])
        self.publication_options.append(kwargs)
        return proof(source,kwargs['state_dir'],kwargs['unit'])

    def verifier(self,**kwargs):
        self.verified.append(kwargs['state_dir'])
        state=kwargs['state_dir'];manifest=read(state/'manifest.json');receipt=read(state/'receipt.json')
        return dict(passed=True,manifest_sha256=manifest_digest(manifest),state_dir=str(state),receipt=receipt)

    def consumer(self):
        self.queue=AnnotationQueue(self.args,runner=self.runner,publisher=self.publisher,verifier=self.verifier,
                                   summarizer=self.summary,usage=lambda:dict(committed_cny=0.))
        return self.queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='annotation_queue_v16_');self.root=Path(self.temp.name)
        self.fixture=Fixture(self.root)

    def tearDown(self):self.temp.cleanup()

    def test_failed_paid_unit_preserved_other_inputs_continue_once(self):
        f=self.fixture;failed=f.ready('astronaut');good=f.ready('lunar_rover');queue=f.consumer()
        queue.cycle()
        self.assertEqual(f.calls,['seed_123_astronaut','seed_123_lunar_rover'])
        self.assertEqual(read(f.producer/'queue/consumer_state'/failed.name)['phase'],'needs_review')
        state=read(f.producer/'queue/consumer_state'/good.name)
        self.assertEqual(state['phase'],'final_published')
        self.assertEqual(state['raw_manifest_sha256'],read(good)['raw_manifest_sha256'])
        self.assertEqual(state['raw_delivery_manifest_sha256'],read(good)['raw_delivery']['manifest_sha256'])
        self.assertTrue((f.cache/'paid_response_preserved.json').exists())
        self.assertFalse((f.producer/'REQUEST_STOP').exists())
        queue.cycle();self.assertEqual(len(f.calls),2)
        self.assertTrue((f.producer/'raw_inputs/seed_123_astronaut/trajectory/supervision.npz').exists())

    def test_native_control_corruption_rejected_before_paid_work(self):
        f=self.fixture;path=f.ready('lunar_rover',bad_future=True);queue=f.consumer();queue.cycle()
        self.assertEqual(f.calls,[])
        self.assertEqual(read(f.producer/'queue/consumer_state'/path.name)['phase'],'needs_review')
        self.assertFalse((f.producer/'REQUEST_STOP').exists())

    def test_capture_rng_seed_is_independent_of_layout_seed(self):
        f=self.fixture;path=f.ready('lunar_base',capture_seed=7,source_layout_seed=123);queue=f.consumer()
        accepted=validate_ready(read(path),producer_root=f.producer,plan=f.plan,
            plan_sha256=sha256(f.producer/'plan.json'),host='fake-a100',remote_root='/fake/remote')
        self.assertEqual(accepted['capture_rng_seed'],7);self.assertEqual(accepted['source_layout_seed'],123)
        queue.cycle();self.assertEqual(len(f.calls),1)

    def test_explicit_wrong_source_layout_seed_rejected_before_paid_work(self):
        f=self.fixture;path=f.ready('lunar_base',capture_seed=123,source_layout_seed=999);queue=f.consumer()
        queue.cycle();self.assertEqual(f.calls,[])
        self.assertEqual(read(f.producer/'queue/consumer_state'/path.name)['phase'],'needs_review')

    def test_legacy_layout_binding_does_not_fall_back_to_rng_seed(self):
        f=self.fixture;path=f.ready('lunar_base',capture_seed=7,source_layout_seed=None);queue=f.consumer()
        queue.cycle();self.assertEqual(len(f.calls),1)

    def test_hash_changed_input_rejected_and_partial_ready_ignored(self):
        f=self.fixture;path=f.ready('lunar_base')
        (f.producer/'queue/ready/not_ready.json.tmp').write_text('{')
        source=Path(read(path)['source']['root']);(source/'trajectory/metadata.json').write_text('{}')
        queue=f.consumer();queue.cycle();self.assertEqual(f.calls,[])
        self.assertEqual(read(f.producer/'queue/consumer_state'/path.name)['phase'],'needs_review')

    def test_resume_final_remote_verification_does_not_read_pruned_raw(self):
        f=self.fixture;path=f.ready('lunar_rover');queue=f.consumer();queue.cycle()
        shutil.rmtree(f.producer/'raw_inputs/seed_123_lunar_rover/trajectory')
        f.args.resume=True;queue=f.consumer();queue.cycle()
        self.assertEqual(len(f.calls),1);self.assertEqual(len(f.verified),1)
        self.assertIn('seed_123_lunar_rover',queue.entries)

    def test_cpu_adoption_command_compiled_twenty_queries_budget12(self):
        f=self.fixture;path=f.ready('lunar_base');queue=f.consumer();queue.cycle()
        config=read(f.output/'unit_configs/seed_123_lunar_base.json')
        self.assertEqual(config['annotation']['query_mode'],'compiled')
        self.assertEqual(config['annotation']['query_count'],20)
        self.assertEqual(config['annotation']['budget_cny'],12.)
        source=validate_ready(read(path),producer_root=f.producer,plan=f.plan,
            plan_sha256=sha256(f.producer/'plan.json'),host='fake-a100',remote_root='/fake/remote')
        command=queue.command(read(path),source,f.output/'units/seed_123_lunar_base',f.root/'unit.json')
        self.assertIn('--layout-source',command);self.assertIn('--raw-source',command)
        self.assertNotIn('--resume',command)

    def test_shared_unit_claim_blocks_duplicate_paid_work(self):
        f=self.fixture;path=f.ready('lunar_base');queue=f.consumer()
        with (f.producer/'queue/consumer_locks'/(path.stem+'.lock')).open('a') as claim:
            fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertFalse(queue.handle(path));self.assertEqual(f.calls,[])
        self.assertTrue(queue.handle(path));self.assertEqual(len(f.calls),1)

    def test_default_process_environment_disables_gpu(self):
        f=self.fixture;path=f.ready('lunar_base');queue=f.consumer();captured=[]
        class Process:
            pid=12345;returncode=0
            def poll(self):return 0
        def spawn(command,**kwargs):captured.append(kwargs);return Process()
        with patch('run_lunar_annotation_queue_v16.subprocess.Popen',spawn):
            queue.run_pipeline(['fake-cpu-command'],f.output/'units'/path.stem,dict(unit_id=path.stem))
        self.assertEqual(captured[0]['env']['CUDA_VISIBLE_DEVICES'],'')
        self.assertNotIn('shell',captured[0])

    def test_preseed_reference_is_bound_to_ready_raw_delivery(self):
        f=self.fixture;config=read(f.args.config);config['enable_raw_preseed']=True;write_json(f.args.config,config)
        path=f.ready('lunar_base');queue=f.consumer();queue.cycle()
        self.assertEqual(f.publication_options[0]['preseed_units'],[dict(unit='raw_inputs/'+path.stem,
            manifest_sha256=read(path)['raw_delivery']['manifest_sha256'])])

    def test_retention_failure_keeps_valid_final_publication_phase(self):
        f=self.fixture;config=read(f.args.config)
        config['retention']=dict(enabled=True,below_free_gib=1000000000,target_free_gib=1000000001,keep_review_units=0)
        write_json(f.args.config,config);path=f.ready('lunar_base');queue=f.consumer()
        with patch('run_lunar_annotation_queue_v16.reclaim_owned_frames',side_effect=ValueError('changed frame')):
            queue.cycle()
        state=read(f.producer/'queue/consumer_state'/path.name)
        self.assertEqual(state['phase'],'final_published');self.assertTrue(state['final_delivery']['receipt']['passed'])
        self.assertEqual(state['retention']['phase'],'needs_retention_review')


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='annotation_retention_v16_');self.root=Path(self.temp.name)
        self.identity='seed_123_lunar_base';self.raw=self.root/'producer/raw_inputs'/self.identity
        self.unit=self.root/'consumer/units'/self.identity
        self.raw.mkdir(parents=True);self.unit.mkdir(parents=True)
        raw_frames=self.raw/'trajectory/frames';raw_frames.mkdir(parents=True)
        trajectory=self.unit/'dataset/episodes/demo/route_trajectory';trajectory.mkdir(parents=True)
        original=self.unit/'dataset/episodes/demo/trajectory';original.mkdir()
        for directory in (trajectory/'frames',original/'frames'):directory.mkdir()
        for index in range(6):
            for suffix in ('rgb224.png','rgb256.png','sensors.npz','instances.json'):
                name=f'{index:06d}_'+suffix;source=raw_frames/name
                source.write_bytes(bytes([index+1])*8192)
                os.link(source,trajectory/'frames'/name);os.link(source,original/'frames'/name)
        frames=[dict(sample_id=f'{self.identity}:{i:06d}',step_index=i) for i in range(6)]
        for directory in (trajectory,original):
            (directory/'frame_records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in frames))
        language=self.unit/'dataset/episodes/demo/language';language.mkdir()
        (language/'annotations.jsonl').write_text(''.join(json.dumps(dict(sample_id=row['sample_id'],cot_valid=row['step_index'] in (4,5)))+'\n' for row in frames))
        write_json(self.unit/'dataset/dataset_manifest.json',dict(episodes=[dict(trajectory='episodes/demo/route_trajectory',language='episodes/demo/language')]))
        (self.unit/'review').mkdir();(self.unit/'review/preview.mp4').write_bytes(b'retain-video')
        write_json(self.unit/'status.json',dict(phase='complete',passed=True))
        self.raw_delivery=proof(self.raw,self.root/'raw_state','raw_inputs/'+self.identity)
        self.final_delivery=proof(self.unit,self.root/'final_state','units/'+self.identity)
        self.kwargs=dict(owned_units_root=self.unit.parent,unit_id=self.identity,ordinal=4,
            final_manifest_path=self.root/'final_state/manifest.json',final_receipt_path=self.root/'final_state/receipt.json',
            expected_final_sha256=self.final_delivery['manifest_sha256'],raw_manifest_path=self.root/'raw_state/manifest.json',
            raw_receipt_path=self.root/'raw_state/receipt.json',expected_raw_sha256=self.raw_delivery['manifest_sha256'],
            critical_sha256={'status.json':sha256(self.unit/'status.json')})

    def tearDown(self):self.temp.cleanup()

    def aliases(self):return sorted(self.unit.glob('dataset/episodes/demo/*/frames/*'))

    def test_both_alias_owners_needed_to_release_blocks_cot_pairs_kept(self):
        result=reclaim_owned_frames(self.unit,**self.kwargs)
        self.assertEqual(result['files_deleted'],24)
        self.assertEqual(result['kept_native_indices'],[3,4,5])
        producer_bulk=self.raw/'trajectory/frames/000000_rgb224.png'
        self.assertTrue(producer_bulk.exists());self.assertEqual(producer_bulk.stat().st_nlink,1)
        for index in (3,4,5):
            self.assertTrue((self.unit/f'dataset/episodes/demo/route_trajectory/frames/{index:06d}_rgb224.png').exists())
        self.assertTrue((self.unit/'review/preview.mp4').exists())
        blocks=producer_bulk.stat().st_blocks
        producer_bulk.unlink()  # Simulate the other owner's separately verified unlink.
        self.assertGreater(blocks,0);self.assertFalse(producer_bulk.exists())

    def test_changed_late_candidate_preserves_all_proposed_deletions(self):
        aliases=self.aliases();changed=self.unit/'dataset/episodes/demo/trajectory/frames/000002_sensors.npz'
        changed.write_bytes(b'new unuploaded bytes')
        with self.assertRaises(ValueError):reclaim_owned_frames(self.unit,**self.kwargs)
        self.assertTrue(all(path.exists() for path in aliases))
        self.assertEqual(changed.read_bytes(),b'new unuploaded bytes')

    def test_missing_raw_or_final_receipt_cannot_prune(self):
        for key in ('raw_receipt_path','final_receipt_path'):
            with self.assertRaises(FileNotFoundError):reclaim_owned_frames(self.unit,**dict(self.kwargs,**{key:self.root/'missing'}))
        self.assertEqual(len(self.aliases()),48)

    def test_first_four_review_units_kept_and_metadata_change_rejected(self):
        result=reclaim_owned_frames(self.unit,**dict(self.kwargs,ordinal=3))
        self.assertFalse(result['local_frames_evicted']);self.assertEqual(len(self.aliases()),48)
        (self.unit/'dataset/episodes/demo/language/annotations.jsonl').write_text('changed')
        with self.assertRaises(ValueError):reclaim_owned_frames(self.unit,**self.kwargs)
        self.assertEqual(len(self.aliases()),48)

    def test_external_frame_ancestor_refused(self):
        frames=self.unit/'dataset/episodes/demo/trajectory/frames'
        external=self.root/'external_frames';shutil.move(frames,external);frames.symlink_to(external,target_is_directory=True)
        with self.assertRaises(ValueError):reclaim_owned_frames(self.unit,**self.kwargs)
        self.assertEqual(len(list(external.iterdir())),24)


if __name__=='__main__':unittest.main()
