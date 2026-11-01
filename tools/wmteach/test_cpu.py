import unittest

from tools.wmteach.labels import actions, label, compare, quality


class TeacherTests(unittest.TestCase):
    def record(self):
        return dict(state_id='s981000_t300',world_id=981000,role='QUALITY_ONLY',
                    time_s=300,setting=0,horizon_s=120,rows=[
                        dict(vid='a',s_m=-20.,v_mps=10.,lane=0,is_cav=True),
                        dict(vid='b',s_m=-10.,v_mps=10.,lane=1,is_cav=True)])

    def futures(self):
        return {a['id']:[dict(draw_id=str(i),team_delay=(100. if a['id']=='O0' else
                    101. if i<6 else 90.),breakdown=None) for i in range(8)]
                for a in actions(self.record()['rows'],0)}

    def test_mean_and_probability_gate_are_distinct(self):
        result=label(self.record(),self.futures())
        self.assertEqual(result['selected'],'O1')
        self.assertEqual(result['paired_gate']['selected'],'O0')
        self.assertEqual(result['effects']['O1'],1.75)

    def test_unpaired_missing_or_failed_draw_rejected(self):
        for mode in ('unpaired','missing','failed'):
            futures=self.futures()
            if mode=='unpaired': futures['O1'][0]['draw_id']='other'
            if mode=='missing': futures['O1'].pop()
            if mode=='failed': futures['O1'][0]['team_delay']=float('nan')
            with self.assertRaises(ValueError): label(self.record(),futures)

    def test_exact_regret_and_sign_not_overall_accuracy_only(self):
        result=label(self.record(),self.futures())
        exact=dict(state_id=result['state_id'],horizon_s=120,effects=dict(O0=0.,O1=-20.,O2=8.,O4=-1.))
        row=compare(result,exact)
        self.assertEqual(row['regret'],28.)
        self.assertEqual(row['exact_saving'],-20.)
        self.assertEqual(row['sign_count'],3)
        self.assertEqual(row['sign_matches'],1)
        self.assertEqual(quality([row],300)['states'],1)
        exact['horizon_s']=300
        with self.assertRaises(ValueError): compare(result,exact)

    def test_missing_exact_option_not_imputed(self):
        result=label(self.record(),self.futures())
        with self.assertRaises(ValueError):
            compare(result,dict(state_id=result['state_id'],horizon_s=120,effects={'O0':0.}))


if __name__=='__main__': unittest.main()
