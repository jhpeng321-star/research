import copy
import unittest
from .warm_history import ObservableHistory, arrays, validate_history


def row(vid='a', speed=10.):
    return dict(vid=vid,s_m=-100.,v_mps=speed,lane=0,is_cav=False,length_m=5.)


class HistoryTests(unittest.TestCase):
    def test_observable_whitelist_and_owned_copy(self):
        h=ObservableHistory();rows=[row()];h.push(0,rows)
        rows[0]['v_mps']=99
        self.assertEqual(h.snapshot(0)[0]['rows'][0]['v_mps'],10)
        for field in ('hidden_setting','future_speed','vType'):
            with self.assertRaises(ValueError):h.push(1,[dict(row(),**{field:1})])

    def test_rolling_clock_and_future_noninterference(self):
        h=ObservableHistory()
        for t in range(121):h.push(t,[row()])
        f=h.snapshot(120)
        self.assertEqual([x['time_s'] for x in f],list(range(60,121)))
        f.append(dict(time_s=121,rows=[row()],entrance_counts=None))
        with self.assertRaises(ValueError):validate_history(f,120)
        with self.assertRaises(ValueError):h.push(120,[row()])

    def test_empty_and_counter_contract(self):
        self.assertIsNone(arrays(dict(time_s=300,rows=[],history=[])))
        h=ObservableHistory()
        with self.assertRaises(ValueError):h.push(300,[row()],[0,1.5])
        h.push(300,[row()],[1,2]);self.assertEqual(h.snapshot(300)[0]['entrance_counts'],[1,2])

    def test_identity_continuity_and_backward_acceleration(self):
        h=ObservableHistory()
        h.push(300,[row('old')]);h.push(301,[row('new'),row('old',12)])
        record=dict(time_s=301,rows=[row('new'),row('old',12)],history=h.snapshot(301))
        a=arrays(record)
        self.assertEqual(a['ids'],['new','old'])
        self.assertFalse(a['mask'][0,0]);self.assertEqual(a['kin'][1,0,5],0.)
        self.assertEqual(a['kin'][1,1,5],2.)
        bad=copy.deepcopy(record);bad['rows'][0]['v_mps']=11
        with self.assertRaises(ValueError):arrays(bad)


if __name__=='__main__':unittest.main()
