"""Publication transport/immutability gates; no SUMO/Torch or remote dispatch."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import coopfree_publish as p


class PublicationTests(unittest.TestCase):
    def test_transport_corruption_and_traversal_rejected(self):
        data=b'checkpoint'
        item=dict(base64=base64.b64encode(data).decode(),sha256=p.sha(data))
        self.assertEqual(p.unpack({'policy_u001.pt':item}),{'policy_u001.pt':data})
        with self.assertRaises(ValueError):
            p.unpack({'../policy_u001.pt':item})
        with self.assertRaises(ValueError):
            p.unpack({'..\\policy_u001.pt':item})
        with self.assertRaises(ValueError):
            p.unpack({'policy_u001.pt':dict(item,sha256='0'*64)})

    def test_existing_artifact_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint'
            p.immutable(path,b'original')
            p.immutable(path,b'original')
            with self.assertRaises(ValueError):
                p.immutable(path,b'new')
            self.assertEqual(path.read_bytes(),b'original')

    def test_handoff_preserves_surrounding_user_content(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'HANDOFF.md'
            path.write_text('owner text\n')
            p.replace_block(path,'first')
            path.write_text(path.read_text()+'new user text\n')
            p.replace_block(path,'second')
            text=path.read_text()
            self.assertIn('owner text',text)
            self.assertIn('new user text',text)
            self.assertEqual(text.count(p.BEGIN),1)
            self.assertNotIn('first',text)

    def test_manifest_tampering_blocks_before_remote_read(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'policy_u001.pt').write_bytes(b'tampered')
            (root/'MANIFEST.json').write_text(json.dumps(dict(files={'policy_u001.pt':p.sha(b'original')})))
            with patch.object(p,'PUBLIC',root),patch.object(p,'collect') as collect:
                with self.assertRaisesRegex(ValueError,'existing local publication hash'):
                    p.once()
                collect.assert_not_called()

    def test_collector_contains_no_physics_dispatch(self):
        def fake_remote(host,source,timeout):
            self.assertEqual(host,'cpuQ')
            self.assertIn("assert dispatch['pid']==339173",source)
            self.assertIn('checkpoint/update binding',source)
            self.assertNotIn('Popen',source)
            self.assertNotIn('backend.create',source)
            self.assertNotIn('write_text',source)
            return '{}'
        with patch.object(p,'remote',side_effect=fake_remote):
            self.assertEqual(p.collect(),{})

    def test_report_does_not_call_positive_synergy_a_helpful_group(self):
        credits={h:dict(D_G=-10.,sum_D_i=-30.,C_G=20.) for h in ('120','300','600')}
        report=dict(update=1,limitations='descriptive subset',groups=[dict(joint_id='g',credit=credits)],
                    analysis=dict(events=[dict(joint_id='g',events=[dict(behaviour='yield')])],
                                  clusters=dict(status='insufficient groups',k=None),note='Not causal proof.'))
        text='\n'.join(p.discovery_text(report))
        self.assertIn('C_G>0 1/1, D_G>0 and C_G>0 0/1',text)
        self.assertIn('Events in groups with positive D_G and C_G at600s: none',text)

    def test_paired_dev_sign_and_uncertainty_are_reported(self):
        metrics={}
        for h in ('600','900'):
            for name in ('team_delay_vs','collisions','teleports','ttc_below_3_vs','hard_brake'):
                metrics[h+'_'+name]=dict(mean=0.,ci95=[-1.,1.])
        metrics['600_team_delay_vs']=dict(mean=-2.,ci95=[-3.,-1.])
        report=dict(update=1,worlds=24,learned_minus_comparator=dict(mappo=metrics))
        text='\n'.join(p.dev_text(report))
        self.assertIn('mappo J600: -2.000 [-3.000, -1.000], supported reduction',text)
        self.assertIn('mappo J900: 0.000 [-1.000, 1.000], CI includes zero',text)


if __name__=='__main__':
    unittest.main()
