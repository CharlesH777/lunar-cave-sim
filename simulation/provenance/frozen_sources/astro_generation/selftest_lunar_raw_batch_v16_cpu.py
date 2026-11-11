"""Raw producer fixtures: no renderer, provider or real transport."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from run_lunar_raw_batch_v16 import RawProducer,physical_code_files,payload_inventory
from cave.lunar_raw_physical_v16 import write_json,sha256


class RawTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def producer(self):
        p=object.__new__(RawProducer);p.output=self.root;p.queue=self.root/'queue';p.host='test-host';p.remote='/data/test'
        p.lock=threading.RLock();p.admission=threading.Lock();p.active={};p.reservations=set();p.pending_limit=2
        p.config=dict(requested_units=4,maximum_unannotated_units=2,minimum_staging_free_gib=0.,prune_consumed_raw=True,
            evict_below_free_gib=10**7,evict_target_free_gib=10**7,keep_review_units=0)
        p.frozen=lambda:None;p.status=lambda *a,**k:None
        (p.queue/'ready').mkdir(parents=True);(p.queue/'consumer_state').mkdir()
        return p

    def ready(self,p):
        ready=dict(schema_version='lunar_raw_ready/16.0',unit_id='seed_1_astronaut',ordinal=0,
            raw_manifest_sha256='a'*64,raw_delivery=dict(manifest_sha256='b'*64),producer_relative_root='raw_inputs/seed_1_astronaut')
        write_json(p.queue/'ready/seed_1_astronaut.json',ready)
        return ready

    def final(self,p,ready,**overrides):
        result=dict(phase='final_published',unit_id=ready['unit_id'],raw_manifest_sha256=ready['raw_manifest_sha256'],
            raw_delivery_manifest_sha256=ready['raw_delivery']['manifest_sha256'],final_delivery=dict(passed=True,manifest_sha256='f'*64,
                receipt=dict(passed=True,manifest_sha256='f'*64,unit='units/'+ready['unit_id'],host=p.host,remote_root=p.remote)))
        result.update(overrides);return result

    def test_physics_dependency_closure_has_no_language_or_provider(self):
        files=physical_code_files()
        self.assertIn('collect_lunar_trajectory_v14_isaac.py',files)
        self.assertIn('validate_lunar_raw_v16_cpu.py',files)
        self.assertFalse([name for name in files if any(x in name for x in ('ppio','query_','annotate_'))])

    def test_inventory_covers_raw_bytes_and_excludes_only_controls_and_self(self):
        root=self.root/'unit';(root/'trajectory/frames').mkdir(parents=True)
        (root/'trajectory/frames/000000_rgb224.png').write_bytes(b'actual bytes fixture')
        (root/'raw_manifest.json').write_text('{}');(root/'REQUEST_STOP').touch();(root/'partial.tmp').touch()
        inventory=payload_inventory(root)
        self.assertEqual([x['path'] for x in inventory],['trajectory/frames/000000_rgb224.png'])
        self.assertEqual(inventory[0]['sha256'],sha256(root/inventory[0]['path']))

    def test_language_needs_review_only_backpressures_until_exact_receipt(self):
        p=self.producer();ready=self.ready(p);p.reservations={'inflight'};p.evict_if_needed=lambda:None
        path=p.queue/'consumer_state'/(ready['unit_id']+'.json')
        write_json(path,dict(phase='needs_review',unit_id=ready['unit_id']))
        calls=[]
        def change(_):
            calls.append(1)
            write_json(path,self.final(p,ready,raw_manifest_sha256='wrong') if len(calls)==1 else self.final(p,ready))
        with patch('run_lunar_raw_batch_v16.time.sleep',side_effect=change):p.reserve('next')
        self.assertEqual(len(calls),2);self.assertEqual(p.pending_count(),2);self.assertFalse(p.stopped())

    def test_pending_buffer_wait_honors_user_stop_without_capture(self):
        p=self.producer();p.reservations={'one','two'};p.evict_if_needed=lambda:None
        with patch('run_lunar_raw_batch_v16.time.sleep',side_effect=lambda _: (self.root/'REQUEST_STOP').touch()):
            with self.assertRaisesRegex(InterruptedError,'REQUEST_STOP'):p.reserve('third')
        self.assertEqual(p.reservations,{'one','two'})

    def test_partial_gpu_output_preserved_without_retry(self):
        p=self.producer();p.config['adopt_raw']={};p.plan_sha='plan';p.reserve=lambda unit:p.reservations.add(unit)
        p.existing_capture=lambda path:None;p.command=Mock(side_effect=AssertionError('must not launch'))
        root=self.root/'raw_inputs/seed_1_astronaut';(root/'layout').mkdir(parents=True);(root/'trajectory/frames').mkdir(parents=True)
        write_json(root/'trajectory/metadata.json',dict(terminal_success=False));write_json(root/'trajectory/status.json',dict(phase='partial'))
        frame=root/'trajectory/frames/000000_rgb224.png';frame.write_bytes(b'preserve stopped raw')
        with self.assertRaisesRegex(ValueError,'Only complete'):p.run_unit(0,dict(seed=1,split='train'),'astronaut')
        p.command.assert_not_called();self.assertTrue(frame.is_file());self.assertFalse((p.queue/'ready/seed_1_astronaut.json').exists())

    def retention(self,*,matched=True):
        p=self.producer();ready=self.ready(p);root=self.root/ready['producer_relative_root'];(root/'trajectory/frames').mkdir(parents=True)
        frame=root/'trajectory/frames/000001_rgb224.png';frame.write_bytes(b'published native frame')
        write_json(root/'raw_validation.json',dict(passed=True));manifest=dict(payload_files=[dict(path='trajectory/frames/'+frame.name,size_bytes=frame.stat().st_size,sha256=sha256(frame))])
        write_json(root/'raw_manifest.json',manifest);ready['raw_manifest_sha256']=sha256(root/'raw_manifest.json')
        ready['source']=dict(files_sha256={'raw_validation.json':sha256(root/'raw_validation.json'),'raw_manifest.json':ready['raw_manifest_sha256']})
        state=self.root/'delivery_state'/ready['unit_id'];write_json(state/'manifest.json',dict(entries=[dict(path='raw_manifest.json',sha256=ready['raw_manifest_sha256'])]))
        ready['raw_delivery']['manifest_sha256']=sha256(state/'manifest.json')
        write_json(state/'receipt.json',dict(passed=True,manifest_sha256=ready['raw_delivery']['manifest_sha256'],unit='raw_inputs/'+ready['unit_id'],host=p.host,remote_root=p.remote))
        write_json(p.queue/'ready'/(ready['unit_id']+'.json'),ready)
        write_json(p.queue/'consumer_state'/(ready['unit_id']+'.json'),self.final(p,ready,raw_manifest_sha256=ready['raw_manifest_sha256'] if matched else 'stale'))
        return p,ready,frame

    def test_no_raw_eviction_before_matching_final_annotation_receipt(self):
        p,ready,frame=self.retention(matched=False);p.evict_if_needed();self.assertTrue(frame.exists())
        self.assertFalse((self.root/'retention'/(ready['unit_id']+'.json')).exists())

    def test_bound_raw_and_final_receipts_release_only_owned_frames(self):
        p,ready,frame=self.retention();outside=self.root/'other_owned_task/frames/000001_rgb224.png';outside.parent.mkdir(parents=True);outside.write_bytes(b'never touch')
        p.evict_if_needed();self.assertFalse(frame.exists());self.assertTrue(outside.exists())
        self.assertTrue((frame.parents[2]/'raw_validation.json').is_file())

    def test_modified_unuploaded_raw_version_not_deleted(self):
        p,ready,frame=self.retention();frame.write_bytes(b'new unuploaded bytes')
        with self.assertRaisesRegex(ValueError,'Changed/unuploaded'):p.evict_if_needed()
        self.assertTrue(frame.exists())

    def test_disk_guard_waits_without_stopping_other_work(self):
        p=self.producer();p.config['minimum_staging_free_gib']=1.;p.evict_if_needed=lambda:None
        p.collectors=lambda:[];p.available_resources=Mock(return_value=(32.,24.))
        with patch('cave.lunar_raw_physical_v16.shutil.disk_usage',return_value=SimpleNamespace(free=0)),patch('cave.lunar_raw_physical_v16.time.sleep',side_effect=lambda _: (self.root/'REQUEST_STOP').touch()):
            with self.assertRaisesRegex(InterruptedError,'REQUEST_STOP'):p.capacity('new')
        p.available_resources.assert_not_called()

    def test_capture_admission_lock_is_released_during_capture(self):
        p=self.producer();p.capacity=lambda unit:None;p.environment=lambda **kwargs:{}
        released=threading.Event();started=[]
        class Process:
            def __init__(self,*args,**kwargs):
                self.pid=len(started)+1;self.returncode=0;started.append(self.pid)
                if len(started)==2:released.set()
            def poll(self):return 0 if released.is_set() else None
        errors=[]
        def launch(identity):
            try:p.command(['fake'],self.root/(identity+'.log'),admit_unit_id=identity)
            except Exception as exc:errors.append(exc)
        with patch('run_lunar_raw_batch_v16.subprocess.Popen',Process),patch('run_lunar_raw_batch_v16.time.sleep',side_effect=lambda _:released.wait(.01)):
            threads=[threading.Thread(target=launch,args=(name,),daemon=True) for name in ('one','two')]
            for thread in threads:thread.start()
            for thread in threads:thread.join(2.)
            self.assertEqual(started,[1,2]);self.assertTrue(all(not thread.is_alive() for thread in threads));self.assertFalse(errors)


if __name__=='__main__':unittest.main()
