"""CPU-only scientific/data boundary and observable policy regression tests."""
import unittest
from unittest.mock import patch

import numpy as np

from tools.coopbandit.protocol import observable
from tools.coopplan.predictor import ObservableState
from .data import (KEYS, OPTIONS, arrays, encode, load_exam, planner_records,
                   training_role, checked)
from .model import (Student, choose, confidence, fit_values, predict_values,
                    teacher, residuals, fit_classifier, classify)
from .train import splits, world_mean, evaluate


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x),self.value)


ROWS = (
    dict(vid='a',is_cav=True,lane=0,s_m=-10.,v_mps=10.),
    dict(vid='b',is_cav=True,lane=1,s_m=-20.,v_mps=11.),
    dict(vid='r',is_cav=False,lane=2,s_m=-15.,v_mps=6.),
)


class Tests(unittest.TestCase):
    def test_mean_rule_acts_without_lower_bound(self):
        p = np.array([[0.,2.,1.,900.,-1.],[0.,-1.,0.,-1.,-2.]])
        av = np.array([[1,1,1,0,1],[1,1,1,1,1]],bool)
        # Arbitrarily broad errors do NOT veto the mean policy at confidence0.
        e = [np.array([1e6]) for _ in range(5)]
        np.testing.assert_array_equal(choose(p,av,e,0),[1,0])
        np.testing.assert_array_equal(choose(p,av,e,.5),[0,0])

    def test_no_lower_ranked_fallback(self):
        p = np.array([[0.,2.,1.,-1.,-2.]])
        e = [np.array([0.]),np.array([3.]),np.array([0.]),np.array([0.]),np.array([0.])]
        self.assertEqual(choose(p,np.ones_like(p,bool),e,.5)[0],0)

    def test_signed_probability_strictness(self):
        p = np.array([[0.,2.,3.,0.,0.]])
        e = [np.array([-2.,2.]) for _ in range(5)]
        np.testing.assert_array_equal(confidence(p,e)[0,1:3],[.5,1.])

    def test_residuals_available_only(self):
        p = np.ones((2,5))
        y = np.zeros_like(p)
        av = np.ones_like(p,bool)
        av[1,1] = False
        e = residuals(p,y,av)
        self.assertEqual(len(e[1]),1)
        self.assertEqual(len(e[2]),2)

    def test_teacher_zero_ties_default(self):
        np.testing.assert_array_equal(teacher(np.zeros((2,5)),np.ones((2,5),bool)),[0,0])

    def test_world_splits_and_equal_weighting(self):
        g = np.repeat(np.arange(12),[1,2,3,4,5,6,7,8,9,10,11,12])
        x = np.zeros((len(g),1))
        seen = []
        for tr,te in splits(x,g,4):
            self.assertFalse(set(g[tr]) & set(g[te]))
            seen.extend(te.tolist())
        self.assertEqual(sorted(seen),list(range(len(x))))
        self.assertEqual(world_mean(np.array([2.,0.,0.]),np.array([0,1,1])),1.)

    def test_nontrain_refused(self):
        for seed in (981080,983100,983200,983300,964000,962000,977000):
            with self.assertRaises(ValueError):
                training_role(seed)
        with self.assertRaises(ValueError):
            arrays([dict(role='DEV_NEVER_FIT',world_id=981000)])

    def test_paths_before_read(self):
        for segment in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1',
                        'hetdata_truth','vehicle_params','driver_params','vType'):
            with self.assertRaises(ValueError):
                checked('E:/unused/'+segment+'/file.json')

    def test_inference_causal_schema(self):
        a = encode(ROWS,300,0)
        rows = [{**r,'demand':99999,'driver_secret':100} for r in ROWS]
        np.testing.assert_array_equal(a,encode(rows,300,0))
        self.assertEqual(set(KEYS),set(observable(ROWS,300,0)))
        self.assertFalse({'world_id','seed','demand','cell','penetration'} & set(KEYS))

    def test_student_option_and_cadence(self):
        payload = dict(keys=KEYS,models=[Constant(v) for v in (1.,3.,999.,2.)],
                       errors=[np.array([9999.]) for _ in range(5)],confidence=0.)
        student = Student(payload)
        with patch('subprocess.Popen',side_effect=AssertionError('no physics')):
            event = student.step(ObservableState(300,ROWS,'synthetic'))
        self.assertEqual(event['executed_action']['id'],'O2')
        self.assertEqual(event['group'],['a','b'])
        self.assertEqual(event['executed_action']['magnitude'],3.)
        self.assertEqual(event['executed_action']['window'],30)
        with self.assertRaises(ValueError):
            student.step(ObservableState(300,ROWS,'synthetic'))

    def test_available_only_training(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(20,3))
        av = np.ones((20,5),bool)
        av[10:,1] = False
        y = np.zeros((20,5)); y[:10,1] = x[:10,0]
        altered = y.copy(); altered[10:,1] = 1e12
        for family in ('gbt','mlp'):
            a = predict_values(fit_values(x,y,av,family),x)
            b = predict_values(fit_values(x,altered,av,family),x)
            np.testing.assert_array_equal(a,b)

    def test_classifier_masks_unavailable(self):
        x = np.arange(40).reshape(20,2)
        y = np.zeros((20,5)); y[:,1] = 10.
        av = np.ones_like(y,bool)
        model = fit_classifier(x,y,av)
        av[:,1] = False
        np.testing.assert_array_equal(classify(model,x,av),np.zeros(20,int))

    def test_exam_complete_builder_and_hash_parity(self):
        with patch('subprocess.Popen',side_effect=AssertionError('no physics')):
            records, provenance = load_exam()
        x,y,av,g = arrays(records)
        self.assertEqual(x.shape,(300,len(KEYS)))
        self.assertEqual(y.shape,(300,5))
        self.assertEqual(len(set(g)),60)
        self.assertEqual(provenance['states'],300)
        self.assertTrue(np.all(y[~av] == 0))

    def test_all_planner_dev_extraction(self):
        with patch('subprocess.Popen',side_effect=AssertionError('no physics')):
            records,sources = planner_records()
        self.assertEqual(len(records),720)
        self.assertEqual(len(sources),3)
        self.assertEqual({r['role'] for r in records},{'DEV_NEVER_FIT'})
        self.assertTrue(any(r['unmeasured'] for r in records))
        with self.assertRaises(ValueError):
            arrays(records)

    def test_metrics_are_label_savings_not_prediction(self):
        y = np.array([[0.,-100.,2.,3.,4.],[0.,2.,3.,4.,5.]])
        metrics = evaluate(y,np.ones_like(y,bool),np.array([0,1]),np.array([1,0]))
        self.assertEqual(metrics['saving']['mean'],-50.)
        self.assertEqual(metrics['moves'],1)
        self.assertEqual(metrics['action_counts'],{'O1':1,'O0':1})


if __name__ == '__main__':
    unittest.main()
