import unittest
from .live_telemetry import summarize


class TelemetryTests(unittest.TestCase):
    def report(self,queries,admitted=None):
        p=dict(unix=1000,queries=queries,executed_starts=[],grader={},charged_sim_s=0)
        monitors=dict(monitors={h:dict(receipts={}) for h in ('gpuA','gpuR')})
        if admitted is not None:
            monitors['monitors']['gpuA']['receipts']['WARM_REMOTE_ENDPOINT_ADMISSION_gpuA_SECOND.json']=dict(unix=admitted)
        return summarize(p,monitors,{h:dict(unix=100) for h in ('gpuA','gpuR')})

    def q(self,slot,end,wall):
        return dict(slot=slot,unix=end,inference_wall_s=wall,group_size=2,accepted=True,futures=584)

    def test_qualification_cost_is_not_mislabeled_as_two_worker_throughput(self):
        v=self.report([self.q('gpuA@0',90,50),self.q('gpuA@0',200,150),
                       self.q('gpuA@1',400,100)],admitted=300)
        self.assertEqual(len(v['rates']),3)
        self.assertEqual(v['rates']['gpuA|qualification_or_boundary_overlap|n2']['queries'],1)

    def test_concurrent_worker_seconds_are_not_aggregate_wall_throughput(self):
        v=self.report([self.q('gpuA@0',500,100),self.q('gpuA@1',500,100)],admitted=300)
        g=v['rates']['gpuA|two_workers_admitted|n2']
        self.assertEqual(g['queries_per_inference_worker_hour'],36)
        self.assertEqual(g['completed_queries_per_observed_span_hour'],72)
        self.assertEqual(g['imagined_futures'],1168)

    def test_empty_progress_cannot_manufacture_a_rate_or_benefit(self):
        v=self.report([])
        self.assertEqual(v['rates'],{})
        self.assertEqual(v['executed_starts'],0)
        self.assertFalse(v['outcomes_opened'])
        self.assertEqual(v['new_sumo_sim_s'],0)


if __name__=='__main__':unittest.main()
