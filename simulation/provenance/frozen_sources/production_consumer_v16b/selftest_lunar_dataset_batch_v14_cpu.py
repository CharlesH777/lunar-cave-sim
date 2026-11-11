"""Isolated catalog/stop/retention tests; no renderer, API or SSH."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from cave.lunar_batch_catalog_v14 import build_catalog,iter_rows
from cave.lunar_training_stride_v14 import sha256,write_json
from run_lunar_dataset_batch_v14 import Batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
    def tearDown(self):self.temporary.cleanup()

    def entry(self,identity='one',seed=1,split='train'):
        return dict(unit_id=identity,layout_seed=seed,split=split,delivery=dict(passed=True),
            query_protocol_sha256='fixed',relative_unit_root='units/'+identity,physical_frames=1,
            query_indexes={key:dict(rows=0,language_episodes=0,physical_episodes=0) for key in
                ('query_train','query_eval','physical_val','physical_test')})

    def test_physical_layout_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError,'leaked'):
            build_catalog(self.root/'catalog.json',[self.entry(),self.entry('two',1,'test')],requested_units=4)

    def test_catalog_preserves_native_row_and_unit_base(self):
        unit=self.root/'units/one';(unit/'query_partitions').mkdir(parents=True)
        row=dict(split='train',layout_seed=1,native_step_index=10,
            supervision=dict(native_step_index=10,continuous_controls='episodes/physical/supervision.npz'),
            inputs=dict(observation=dict(depth='episodes/physical/frames/000010_sensors.npz')))
        index=unit/'query_partitions/query_train_index.jsonl';index.write_text(json.dumps(row)+'\n')
        entry=self.entry();entry['query_indexes']['query_train']=dict(path='query_partitions/query_train_index.jsonl',sha256=sha256(index),rows=1,language_episodes=1,physical_episodes=1)
        path=self.root/'catalog/catalog.json';build_catalog(path,[entry],requested_units=4)
        actual=list(iter_rows(path,allow_partial=True))[0]
        self.assertEqual(actual['row'],row)
        self.assertEqual(Path(actual['unit_root']),unit)
        self.assertEqual(actual['row']['supervision']['continuous_controls'],'episodes/physical/supervision.npz')

    def retention(self,passed):
        batch=object.__new__(Batch);batch.output=self.root;batch.lock=threading.RLock()
        batch.args=SimpleNamespace(no_prune=False)
        batch.config=dict(evict_below_free_gib=10**7,evict_target_free_gib=10**7,keep_review_units=0)
        batch.status=lambda *args,**kwargs:None
        unit=self.root/'units/one';base=unit/'dataset';trajectory=base/'episodes/one/route_trajectory';language=base/'episodes/one/language'
        (trajectory/'frames').mkdir(parents=True);language.mkdir()
        write_json(base/'dataset_manifest.json',dict(episodes=[dict(trajectory='episodes/one/route_trajectory',language='episodes/one/language')]))
        (language/'annotations.jsonl').write_text(json.dumps(dict(sample_id='one:0',cot_valid=True))+'\n')
        (trajectory/'frame_records.jsonl').write_text(json.dumps(dict(sample_id='one:0',step_index=0))+'\n')
        selected=trajectory/'frames/000000_rgb224.png';selected.write_bytes(b'keep actual CoT pair')
        bulk=trajectory/'frames/000001_rgb224.png';bulk.write_bytes(b'verified bulk')
        other=self.root/'units/unuploaded/dataset/frames/000001_rgb224.png';other.parent.mkdir(parents=True);other.write_bytes(b'never delete')
        state=self.root/'delivery_state/one';state.mkdir(parents=True)
        write_json(state/'manifest.json',dict(entries=[dict(path=str(path.relative_to(unit)),kind='file',size_bytes=path.stat().st_size,sha256=sha256(path)) for path in (selected,bulk)]))
        digest=sha256(state/'manifest.json');write_json(state/'receipt.json',dict(passed=passed,manifest_sha256=digest))
        entry=dict(unit_id='one',ordinal=0,relative_unit_root='units/one',delivery=dict(passed=True,manifest_sha256=digest),
            critical_sha256={'dataset/dataset_manifest.json':sha256(base/'dataset_manifest.json')})
        batch.entries={'one':entry}
        return batch,selected,bulk,other

    def test_no_eviction_without_complete_receipt(self):
        batch,selected,bulk,other=self.retention(False)
        with self.assertRaisesRegex(ValueError,'prerequisites'):batch.evict_if_needed()
        self.assertTrue(selected.exists() and bulk.exists() and other.exists())

    def test_only_verified_bulk_evicted_cot_and_unuploaded_kept(self):
        batch,selected,bulk,other=self.retention(True);batch.evict_if_needed()
        self.assertTrue(selected.exists() and other.exists());self.assertFalse(bulk.exists())
        self.assertTrue(batch.entries['one']['local_frames_evicted'])

    def test_changed_unuploaded_frame_preserves_whole_deletion_set(self):
        batch,selected,bulk,other=self.retention(True)
        bulk.write_bytes(b'new unuploaded version')
        with self.assertRaisesRegex(ValueError,'changed/unuploaded'):batch.evict_if_needed()
        self.assertTrue(selected.exists() and bulk.exists() and other.exists())

    def test_failed_gpu_api_unit_is_never_relaunched(self):
        batch=object.__new__(Batch);batch.output=self.root;batch.lock=threading.RLock();batch.active={}
        batch.frozen=lambda:None;batch.status=lambda *args,**kwargs:None
        identity='seed_1_astronaut';root=self.root/'units'/identity;root.mkdir(parents=True)
        write_json(root/'status.json',dict(phase='failed'))
        state=self.root/'unit_state'/(identity+'.json');write_json(state,dict(phase='pipeline_failed'))
        calls=[];batch.command=lambda *args,**kwargs:calls.append(args)
        with self.assertRaisesRegex(RuntimeError,'retry forbidden'):
            batch.run_unit(0,dict(seed=1,split='train'),'astronaut')
        self.assertEqual(calls,[]);self.assertEqual(json.loads(state.read_text())['phase'],'pipeline_failed')

    def resource_batch(self):
        batch=object.__new__(Batch);batch.output=self.root;batch.lock=threading.RLock()
        batch.config=dict(minimum_staging_free_gib=0.);batch.active={}
        batch.collectors=lambda:[];batch.evict_if_needed=lambda:None
        return batch

    def test_first_lane_low_host_or_gpu_waits_and_honors_stop(self):
        for resources in ((7.99,24.),(32.,5.99)):
            with self.subTest(resources=resources):
                (self.root/'REQUEST_STOP').unlink(missing_ok=True)
                batch=self.resource_batch();batch.available_resources=Mock(return_value=resources)
                with patch('run_lunar_dataset_batch_v14.time.sleep',side_effect=lambda _: (self.root/'REQUEST_STOP').touch()) as sleep:
                    with self.assertRaisesRegex(InterruptedError,'REQUEST_STOP'):batch.capacity('new')
                self.assertEqual(batch.available_resources.call_count,1);sleep.assert_called_once_with(3)

    def test_first_lane_starts_at_default_resource_thresholds(self):
        batch=self.resource_batch();batch.available_resources=Mock(return_value=(8.,6.))
        with patch('run_lunar_dataset_batch_v14.time.sleep') as sleep:batch.capacity('new')
        sleep.assert_not_called();batch.available_resources.assert_called_once_with()

    def test_first_lane_configurable_thresholds(self):
        batch=self.resource_batch();batch.config.update(first_lane_min_host_gib=12.,first_lane_min_gpu_gib=10.)
        batch.available_resources=Mock(side_effect=[(8.,6.),(12.,10.)])
        with patch('run_lunar_dataset_batch_v14.time.sleep') as sleep:batch.capacity('new')
        sleep.assert_called_once_with(3);self.assertEqual(batch.available_resources.call_count,2)

    def test_second_lane_keeps_startup_and_twelve_gib_gpu_gate(self):
        batch=self.resource_batch();other=self.root/'other'
        status=other/'dataset/episodes/other/trajectory/status.json'
        write_json(status,dict(phase='starting_isaac_continuous_capture'))
        batch.active={'other':dict(root=str(other),adopted=False)};batch.collectors=lambda:[status.parent]
        batch.available_resources=Mock(side_effect=[(8.,11.99),(8.,12.)])
        def advance(_):write_json(status,dict(phase='capturing_integrated_trajectory'))
        with patch('run_lunar_dataset_batch_v14.time.sleep',side_effect=advance) as sleep:batch.capacity('new')
        self.assertEqual(sleep.call_count,2);self.assertEqual(batch.available_resources.call_count,2)

    def test_budget_forwarded_to_unit_and_existing_ledger_without_reset(self):
        protocol=self.root/'query_protocol.json';write_json(protocol,dict(query_count=20))
        unit=self.root/'unit.json';write_json(unit,dict(terrain_template=str(self.root/'terrain'),
            annotation=dict(query_count=20,query_protocol=str(protocol),budget_cny=2.,model='image-fixture',query_model='text-fixture')))
        for configured,expected in ((12.,12.),(None,10.)):
            with self.subTest(configured=configured):
                config=dict(schema_version='lunar_dataset_batch/14.0',unit_config=str(unit),
                    isaac_python='/unused/python',provider_cache=str(self.root/'existing_ledger'),ppio_env=str(self.root/'existing.env'),
                    layouts=[dict(seed=1,split='train')],requested_units=4,expected_layout_splits=dict(train=1,val=0,test=0))
                if configured is not None:config['budget_cny']=configured
                path=self.root/'batch.json';write_json(path,config)
                args=SimpleNamespace(config=path,output=self.root/'output',isaac_python=None,provider_cache=None,ppio_env=None)
                prior_usage=dict(committed_cny=.328,requests=9)
                client=Mock();client.usage.return_value=prior_usage
                with patch('run_lunar_dataset_batch_v14.PPIOClient',return_value=client) as provider:
                    batch=Batch(args)
                    self.assertEqual(batch.budget,expected)
                    self.assertEqual(batch.unit_config['annotation']['budget_cny'],expected)
                    self.assertEqual(batch.usage(),prior_usage)
                    provider.assert_called_once_with(batch.cache,budget_cny=expected)
                self.assertEqual(batch.cache,self.root/'existing_ledger')
                self.assertEqual(batch.unit_config['annotation']['model'],'image-fixture')
                self.assertEqual(batch.unit_config['annotation']['query_model'],'text-fixture')

    def test_budget_must_be_finite_and_positive(self):
        for budget in (0.,-1.,float('inf'),float('-inf'),float('nan')):
            with self.subTest(budget=budget):
                path=self.root/'invalid_budget.json';path.write_text(json.dumps(dict(schema_version='lunar_dataset_batch/14.0',budget_cny=budget)))
                args=SimpleNamespace(config=path,output=self.root/'unused')
                with self.assertRaisesRegex(ValueError,'finite and positive'):Batch(args)

    def test_cpu_failure_cancels_queued_gpu_units_before_pool_wait(self):
        batch=object.__new__(Batch);batch.output=self.root;batch.args=SimpleNamespace(capture_lanes=2)
        batch.slots=[{},{}];batch.config={};batch.frozen=lambda:None;batch.command=lambda *args,**kwargs:0
        batch.runtime='unused';batch.status=lambda *args,**kwargs:None
        (self.root/'logs').mkdir();calls=[]
        def layout(index,slot):
            if index:raise RuntimeError('bounded CPU failure')
            return {}
        def unit(*args):
            calls.append(args)
            while not batch.stopped():time.sleep(.005)
        batch.layout=layout;batch.run_unit=unit
        before=time.monotonic()
        with self.assertRaisesRegex(RuntimeError,'bounded CPU failure'):batch.run()
        self.assertLess(time.monotonic()-before,2.);self.assertLessEqual(len(calls),2)
        self.assertTrue(batch.stopped())


if __name__=='__main__':unittest.main()
