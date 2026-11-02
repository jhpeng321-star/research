import tempfile
import unittest
from pathlib import Path
from .ls_fast_worker import configured_search, drain_applies, pending_claims


class FastWorkerTests(unittest.TestCase):
    def test_successor_adopts_original_receipts_before_any_retirement(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        import tools.coopmwm as package
        from . import warm_service,ls_fast_worker
        delegated=Mock()
        fake=SimpleNamespace(capture_search=lambda fn,**kw:fn,run=delegated)
        with patch.dict(sys.modules,{'ls_gpu':fake}),patch.object(sys,'path',list(sys.path)), \
             patch.object(package,'__path__',list(package.__path__)),patch.object(warm_service,'main'):
            ls_fast_worker.run('/root/test',dict(pid=12),100.,'sha','/root/overlay')
        self.assertTrue(delegated.call_args.kwargs['recover'])
        self.assertEqual(delegated.call_args.kwargs['generation'],'_FASTK2V5')

    def test_config_is_explicit_and_preserves_other_parameters(self):
        seen=[]
        def search(*args,**kwargs):
            seen.append(kwargs['config']);return dict(accepted=False)
        cfg=dict(top_k=8,finalists_per_k=1,pop_per_dim=2,screen_H_min=120)
        out=configured_search(search)(config=cfg)
        self.assertEqual(seen,[dict(cfg,iterations=2,screen_K=2)])
        self.assertNotIn('iterations',cfg)
        self.assertEqual(out['search_config']['final_K'],8)
        self.assertEqual(out['search_config']['screen_H'],300)
        self.assertFalse(out['search_config']['mirror_active'])

    def test_actual_search_keeps_full_population_and_unfiltered_eight(self):
        from . import mppi_search as ms
        from .ls_gpu import capture_search
        rows=[dict(vid='a',is_cav=True,lane=0,s_m=10.,v_mps=20.),
              dict(vid='b',is_cav=True,lane=1,s_m=12.,v_mps=20.)]
        group=ms.group_components(rows)[0]
        record=dict(rows=rows,group=group['members'],commands={'a':0.,'b':0.},
            policy_residuals={'a':0.,'b':0.})
        calls=[]
        def evaluate(rec,actions,k,h):
            calls.append((len(actions),k,h))
            return {a['id']:[100. if a['id']=='policy' else 101. if a['id']=='MAPPO' else 102.+i/100.
                for _ in range(k)] for i,a in enumerate(actions)}
        result=configured_search(capture_search(ms.search,label_list8=True))(
            record,group,[[0.]*300]*2,(-3.,3.),evaluate,'ordinary',seed=20261101)
        self.assertEqual(len(result['iterations']),2)
        self.assertEqual(len(result['collection_top8']),8)
        self.assertTrue(all(p['saving_vs_policy']<0 for p in result['collection_top8']))
        screen=[x for x in calls if x[1]==2]
        final=[x for x in calls if x[1]==8]
        self.assertEqual(len(screen),2)
        self.assertEqual(len(final),1)
        self.assertTrue(all(h==300 for _,_,h in screen))
        self.assertTrue(all(h>=300 for _,_,h in final))
        self.assertEqual(screen[0][0]-2,result['scaling']['population'])

    def test_drain_is_identity_scoped(self):
        old=dict(boot_id='boot',pid=12,start_ticks='34')
        self.assertTrue(drain_applies(dict(owner=old),old))
        self.assertFalse(drain_applies(dict(owner=old),dict(old,pid=13)))
        self.assertFalse(drain_applies(dict(owner=old),dict(old,start_ticks='35')))
        self.assertFalse(drain_applies(None,old))

    def test_all_submitted_requests_must_finish(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'inbox').mkdir();(r/'forecasts').mkdir()
            (r/'inbox/a.json').write_text('{}')
            self.assertEqual(pending_claims(r),['a.json'])
            (r/'forecasts/a.json').write_text('{}')
            self.assertEqual(pending_claims(r),[])


if __name__=='__main__':unittest.main()
