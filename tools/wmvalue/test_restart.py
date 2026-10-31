import unittest

import numpy as np

from .core import features
from .reconstruct import BOUNDARIES, OBS_KEYS, observe_prefix
from .restart import (align_history, causal_queue_prior, current_names, entry_records,
                      imagined_observation, prefix, road_rectangle)


def tape():
    times = np.arange(0., 361., .5)
    shape = (len(times), 3)
    t = dict(time_s=times, s_m=np.zeros(shape), lane=np.zeros(shape, int),
             v_mps=np.full(shape, 10.), a_mps2=np.zeros(shape),
             present=np.zeros(shape, bool), is_cav=np.zeros(shape, bool),
             vid_index=np.full(shape, -1, int), vehicle_ids=np.array(['mainline.0', 'ramp.0', 'unused']))
    for i, start in enumerate((0, 500)):
        t['present'][start:, i] = True
        t['vid_index'][start:, i] = i
        t['lane'][start:, i] = 0 if i == 0 else 2
        t['s_m'][start:, i] = BOUNDARIES[0 if i == 0 else 2]+10.*(times[start:]-times[start])
    t['is_cav'][:, 0] = True
    return t


class RestartTests(unittest.TestCase):
    def test_suffix_and_unseen_registry_inert(self):
        t = tape()
        a = prefix(t, 300)
        h = align_history(a, sorted(current_names(a)))
        records = entry_records(a)
        for k in OBS_KEYS:
            if k != 'time_s':
                t[k][601:] = 0
        t['vehicle_ids'][2] = 'poison'
        b = prefix(t, 300)
        hh = align_history(b, sorted(current_names(b)))
        for k in ('kin', 'mask', 'target_mask'):
            np.testing.assert_array_equal(h[k], hh[k])
        self.assertEqual(records, entry_records(b))
        self.assertEqual(len(h['kin']), 121)

    def test_id_alignment_across_slot_moves_and_birth(self):
        t = tape()
        for k in OBS_KEYS[1:]:
            t[k][550:, [0, 1]] = t[k][550:, [1, 0]]
        p = prefix(t, 300)
        h = align_history(p, ['ramp.0', 'mainline.0', 'absent_current_union'])
        self.assertTrue(h['mask'][-1, :2].all())
        self.assertFalse(h['mask'][:, 2].any())
        self.assertEqual(h['mask'][:, 0].sum(), 101)  # birth t250 in history240..300
        self.assertEqual(h['lane'][-1, 0], 2)
        self.assertAlmostEqual(float(h['kin'][-1, 1, 0]), t['s_m'][600, 1], places=4)
        self.assertFalse(h['target_mask'][19, 0])

    def test_real_imagined_sensing_exact_parity(self):
        t = tape()
        p = prefix(t, 300)
        frame = {k: p[k][-1] for k in ('s_m', 'lane', 'v_mps', 'a_mps2', 'present', 'is_cav')}
        events = np.array([[r['time'], int(r['lane'] == 2)] for r in entry_records(p)])
        a = imagined_observation(frame, events, 300)
        np.testing.assert_array_equal(features(a), features(observe_prefix(t, 300)))

    def test_wm_queue_prior_never_uses_real_request_schedule(self):
        r = [dict(time=10., lane=0, is_cav=True), dict(time=330., lane=2, is_cav=False)]
        q = causal_queue_prior(r, [9., 5.], 360)
        np.testing.assert_allclose(q[:2], [11, 5])
        np.testing.assert_allclose(q[2:4], [10/300, 5/300])
        np.testing.assert_array_equal(causal_queue_prior(r, [9., 5.], 300)[:2], [9, 5])

    def test_birth_exit_union_and_endstep_source_clock(self):
        before = dict(present=np.array([[1, 0, 1]], bool), v_mps=np.array([[10., 0., 5.]]))
        after = dict(present=np.array([[1, 1, 0]], bool), v_mps=np.array([[20., 15., 1000.]]))
        actual = road_rectangle(before, after)
        expected = .5*((1-20/31.29)+(1-15/31.29)+(1-5/31.29))
        np.testing.assert_allclose(actual, [expected])
        # Source end-step rectangle is added separately, not fractional request age.
        np.testing.assert_allclose(actual+.5*np.array([3.]), [expected+1.5])


if __name__ == '__main__':
    unittest.main()
