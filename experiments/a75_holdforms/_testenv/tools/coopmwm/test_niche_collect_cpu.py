import unittest
from . import niche_collect as nc,mppi_search as ms
from .ls_gpu import capture_search


class NicheCollectionTests(unittest.TestCase):
    def test_unregistered_request_configuration_never_reaches_the_model(self):
        from .plan_service import request_memory
        record=dict(state_id='s',world_id=986000,time_s=300,rows=[],group=[],commands={},
            policy_residuals={},search_config=nc.search_config(0))
        with self.assertRaisesRegex(ValueError,'not bound'):request_memory(None,record,{})
        record['search_config']=dict(iterations=2,screen_K=1,niches=4)
        with self.assertRaisesRegex(ValueError,'registered'):request_memory(None,record,dict(niche_comparison=True))

    def test_registered_admission_alternation(self):
        self.assertEqual([nc.search_config(i)['niches'] for i in range(6)],[1,4,1,4,1,4])
        self.assertTrue(all(nc.search_config(i)['screen_K']==2 for i in range(6)))
        for bad in (dict(iterations=2,screen_K=1,niches=4),dict(iterations=2,screen_K=2,niches=2)):
            with self.assertRaises(ValueError):nc.validate(bad)

    def run_search(self,mode):
        rows=[dict(vid='a',is_cav=True,lane=0,s_m=10.,v_mps=20.),dict(vid='b',is_cav=True,lane=1,s_m=12.,v_mps=20.)]
        rec=dict(rows=rows,group=['a','b'],commands={'a':0.,'b':0.},policy_residuals={'a':0.,'b':0.})
        calls=[]
        def evaluate(record,actions,k,h):
            calls.append((len(actions),k,h))
            return {a['id']:[100. if a['id']=='policy' else 101. if a['id']=='MAPPO' else 102.+i/100.
                for _ in range(k)] for i,a in enumerate(actions)}
        out=capture_search(ms.search,label_list8=True)(rec,ms.group_components(rows)[0],
            [[0.]*300]*2,(-3.,3.),evaluate,'ordinary',seed=20261101,config=nc.search_config(0 if mode==1 else 1))
        return out,calls

    def test_negative_finalists_keep_niche_metadata_and_all_eight_labels(self):
        out,calls=self.run_search(4)
        self.assertEqual(len(out['collection_top8']),8)
        self.assertTrue(all(p['saving_vs_policy']<0 for p in out['collection_top8']))
        self.assertTrue(all('niche' in p for p in out['collection_top8']))
        self.assertEqual(out['variation']['occupied_niches'],4)
        self.assertEqual(len(calls),3)
        self.assertEqual([k for _,k,_ in calls],[2,2,8])
        self.assertTrue(all(h>=300 for _,_,h in calls))

    def test_comparison_keeps_identical_evaluation_budget(self):
        plain,pc=self.run_search(1);niche,nc_calls=self.run_search(4)
        self.assertEqual(pc,nc_calls)
        self.assertEqual(plain['imagined_futures'],niche['imagined_futures'])
        self.assertEqual(plain['variation']['niches'],1)
        self.assertEqual(niche['variation']['niches'],4)


if __name__=='__main__':unittest.main()
