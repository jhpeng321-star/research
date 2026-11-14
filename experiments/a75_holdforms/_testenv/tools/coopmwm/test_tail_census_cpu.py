import json
from pathlib import Path
import tempfile
import unittest
from .tail_ops import query_census


class CensusTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_completed_capacity_decline_and_pending_are_distinct(self):
        self.put('quota_requests/u002/a_REQUEST.json',dict(records=[dict(state_id=k) for k in ('done','declined','pending')]))
        self.put('broker_completed/done.json',dict(slot='gpuA@0'))
        self.put('responses/done.json',dict(accepted=True))
        self.put('responses/declined.json',dict(accepted=False,query_admission=dict(available=False)))
        self.assertEqual(query_census(self.root,2),dict(update=2,bound_queries=3,completed_inferences=1,
            accepted_forecasts=1,resolved_capacity_declines=1,pending_responses=1,responses_without_inference_receipt=0))

    def test_missing_inference_metadata_never_imputed_as_zero_compute(self):
        self.put('quota_requests/u002/a_REQUEST.json',dict(state_id='unknown'))
        self.put('responses/unknown.json',dict(accepted=False))
        value=query_census(self.root,2)
        self.assertEqual(value['responses_without_inference_receipt'],1)
        self.assertEqual(value['resolved_capacity_declines'],0)
        self.assertEqual(value['pending_responses'],0)

    def test_other_updates_and_unbound_diagnostics_excluded_and_ids_deduplicated(self):
        self.put('quota_requests/u002/a_REQUEST.json',dict(state_id='same'))
        self.put('quota_requests/u002/b_REQUEST.json',dict(records=[dict(state_id='same')]))
        self.put('quota_requests/u003/a_REQUEST.json',dict(state_id='other'))
        self.put('requests/diagnostic.json',dict(state_id='diagnostic'))
        self.assertEqual(query_census(self.root,2)['bound_queries'],1)


if __name__=='__main__':unittest.main()
