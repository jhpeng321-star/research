import unittest
from .capacity_broker import new_item_order


class CentreFairTests(unittest.TestCase):
    def setUp(self):
        self.items=[('plan',dict(state_id='dense')),('plan',dict(state_id='ordinary')),
                    ('centre',dict(state_id='r1')),('centre',dict(state_id='r2'))]

    def test_alternation_bounds_both_queues_without_changing_requests(self):
        pending=list(self.items);sequence=[];last='plan'
        while pending:
            chosen=new_item_order(pending,last,True)[0]
            self.assertTrue(any(chosen is p for p in self.items))
            sequence.append(chosen[1]['state_id']);last=chosen[0];pending.remove(chosen)
        self.assertEqual(sequence,['r1','dense','r2','ordinary'])

    def test_idle_capacity_never_waits_for_an_empty_queue(self):
        self.assertEqual(new_item_order(self.items[:2],'plan',True),self.items[:2])
        self.assertEqual(new_item_order(self.items[2:],'centre',True),self.items[2:])

    def test_disabled_keeps_original_object_and_order(self):
        self.assertIs(new_item_order(self.items,'plan',False),self.items)

if __name__=='__main__':unittest.main()
