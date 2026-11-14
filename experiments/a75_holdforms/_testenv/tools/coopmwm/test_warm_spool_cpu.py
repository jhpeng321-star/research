"""Failure-path qualification of disconnected durable inference, no GPU/SUMO."""
import json
from pathlib import Path
import tempfile
import unittest
from .warm_spool import SpoolInput,canonical,immutable,submit


class Idle(Exception):pass


class DurableInference(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        immutable(self.root/'GPU_BINDING.json',{'test':True})
        self.record=dict(state_id='a'*64,history=[],rows=[],commands={})

    def stream(self):
        def idle(_):raise Idle()
        return iter(SpoolInput(self.root,100,sleep=idle,clock=lambda:1))

    def test_disconnect_after_submit_and_after_result_never_recomputes(self):
        # Losing the acknowledgement does not authorize a second inference.
        submit(self.root,self.record);submit(self.root,self.record)
        source=self.stream();line=next(source)
        self.assertEqual(json.loads(line),self.record)
        immutable(self.root/'forecasts'/(self.record['state_id']+'.json'),
            dict(key=self.record['state_id'],result={'saving':1}))
        submit(self.root,self.record)
        with self.assertRaises(Idle):next(source)
        # A fresh adapter adopts the completed forecast; no second model call.
        with self.assertRaises(Idle):next(self.stream())
        self.assertEqual(len(list((self.root/'inference_claims').glob('*.json'))),1)

    def test_crash_mid_search_is_fatal_not_replayed(self):
        submit(self.root,self.record);source=self.stream();next(source);source.close()
        with self.assertRaisesRegex(ValueError,'interrupted inference'):next(self.stream())

    def test_conflicting_request_cannot_replace_original(self):
        submit(self.root,self.record)
        with self.assertRaisesRegex(ValueError,'conflicting immutable'):
            submit(self.root,dict(self.record,history=[{'time_s':1}]))
        self.assertEqual(json.loads((self.root/'inbox'/('a'*64+'.json')).read_text()),self.record)

    def test_changed_completed_request_is_fatal(self):
        submit(self.root,self.record);source=self.stream();next(source)
        immutable(self.root/'forecasts'/('a'*64+'.json'),dict(key='a'*64,result={}))
        source.close()
        (self.root/'inbox'/('a'*64+'.json')).write_bytes(canonical(dict(self.record,history=[1])))
        with self.assertRaisesRegex(ValueError,'input changed'):next(self.stream())

    def test_missing_result_stops_before_next_inference(self):
        submit(self.root,self.record);source=self.stream();next(source)
        with self.assertRaisesRegex(ValueError,'without durable forecast'):next(source)

    def test_wrong_result_identity_is_fatal(self):
        submit(self.root,self.record);source=self.stream();next(source);source.close()
        immutable(self.root/'forecasts'/('a'*64+'.json'),dict(key='b'*64,result={}))
        with self.assertRaisesRegex(ValueError,'identity mismatch'):next(self.stream())

    def test_unclaimed_forecast_is_not_adopted(self):
        submit(self.root,self.record);(self.root/'forecasts').mkdir()
        immutable(self.root/'forecasts'/('a'*64+'.json'),dict(key='a'*64,result={}))
        with self.assertRaisesRegex(ValueError,'lacks durable'):next(self.stream())

    def test_invalid_identity_cannot_escape_spool(self):
        for key in ('../x','a/../../x','a'*63,'A'*64):
            with self.assertRaises(ValueError):submit(self.root,dict(self.record,state_id=key))

    def test_deadline_and_stop_are_respected(self):
        with self.assertRaisesRegex(ValueError,'deadline'):
            next(iter(SpoolInput(self.root,0,clock=lambda:1)))
        immutable(self.root/'SPOOL_STOP.json',dict(test=True))
        with self.assertRaises(StopIteration):next(self.stream())


if __name__=='__main__':unittest.main()
