import unittest
import numpy as np

from tools.distill.data import KEYS,OPTIONS
from tools.distill.model import fit_values,predict_values,residuals,choose as original_choose
from tools.wmteach.portable import export,predict,choose


class PortableTests(unittest.TestCase):
    def test_frozen_families_and_unseen_options_keep_exact_choices(self):
        rng=np.random.default_rng(4132)
        x=rng.normal(size=(48,len(KEYS))); x[:,0]=300.
        y=np.column_stack((np.zeros(48),8*x[:,2]-x[:,4],x[:,3]**2-1.,-x[:,5],x[:,6]+2))
        av=np.ones_like(y,dtype=bool); av[:,3]=False  # unseen option must remain zero/no move
        for family in ('gbt','mlp'):
            models=fit_values(x,y,av,family)
            pred=predict_values(models,x)
            errors=residuals(pred,y,av)
            payload=dict(family=family,confidence=.65,label_horizon_s=300,
                         train_worlds=list(range(969400,969432)),keys=KEYS,models=models,errors=errors)
            frozen=export(payload)
            query=np.concatenate([x,rng.normal(size=(80,len(KEYS)))])
            query[:,0]=300.
            expected=predict_values(models,query)
            actual=predict(frozen,query)
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            available=np.ones_like(expected,dtype=bool); available[::2,1]=False
            np.testing.assert_array_equal(choose(frozen,actual,available)[0],
                original_choose(expected,available,errors,.65))
            self.assertEqual(frozen['label_horizon_s'],300)
            self.assertEqual(frozen['family'],family)

    def test_zero_positive_ties_and_unavailable_masks(self):
        policy=dict(confidence=0.,errors=[[]]*5)
        pred=np.array([[0,10,11,-1,-1],[0,0,0,0,0],[0,2,2,2,2]],float)
        av=np.ones_like(pred,dtype=bool); av[0,2]=False
        np.testing.assert_array_equal(choose(policy,pred,av)[0],[1,0,1])

    def test_no_family_substitution(self):
        with self.assertRaises(ValueError): export(dict(keys=KEYS,family='unregistered'))


if __name__=='__main__': unittest.main()
