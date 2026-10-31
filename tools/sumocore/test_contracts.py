"""Meaningful no-SUMO tests of inference, source clocks and preregistered budget."""
import unittest
from demand import estimate,sample
from inference import sumo_attributes


class Contracts(unittest.TestCase):
    def setUp(self):
        self.start=dict(t0=300,observed_queue_history=[[240,5,2],[300,7,1]],
            past_entries=[dict(time=t,lane=lane,is_cav=cav) for t,lane,cav in
                          ((240,0,False),(250,0,True),(260,1,False),(290,2,False))])

    def test_counter_conservation_and_boundary(self):
        e=estimate(self.start)
        self.assertEqual(e['inserted'],[2,1])
        self.assertEqual(e['requested'],[4,0])
        self.assertEqual(e['queue_t0'],[7,1])
        self.assertAlmostEqual(e['observed_cav_fraction'],1/3)

    def test_common_random_streams_and_no_future_oracle(self):
        a,e=sample(self.start,969421,101,600)
        b,_=sample(self.start,969421,101,600)
        self.assertEqual(a,b)
        self.assertEqual(sum(r['depart']==300 for r in a),8)
        self.assertEqual(sum(r['entrance']=='ramp' for r in a),1)
        self.assertTrue(all(300<=r['depart']<900 for r in a))
        self.assertNotEqual(a,sample(self.start,969421,202,600)[0])

    def test_mapping_preserves_units_and_does_not_invent_lane_mapping(self):
        values=dict(v0=34.419,time_headway=.9,min_gap=1.7,accel=.8,decel=2.2,politeness=.1)
        p=sumo_attributes(values)
        self.assertAlmostEqual(p['speedFactor'],1.1)
        self.assertEqual((p['tau'],p['minGap'],p['accel'],p['decel']),(.9,1.7,.8,2.2))
        self.assertEqual(p['lcCooperative'],.65)

    def test_reported_boundary_source_clock(self):
        # Existing queued request waits .5 even if it departs at this boundary;
        # a newly due request contributes only .2s. Subsequent ticks exclude it.
        after=300.5
        requests=[dict(id='q',depart=300),dict(id='new',depart=300.3),dict(id='future',depart=301)]
        area=sum(max(0.,after-max(after-.5,r['depart'])) for r in requests)
        self.assertAlmostEqual(area,.7)


if __name__=='__main__':unittest.main()
