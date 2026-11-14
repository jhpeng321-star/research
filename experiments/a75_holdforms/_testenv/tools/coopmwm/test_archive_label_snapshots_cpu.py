import json
from pathlib import Path
import tempfile
import unittest
from .archive_label_snapshots import final_snapshot_paths


class FinalSnapshotSelection(unittest.TestCase):
    def test_only_successful_final_train_input_copies_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'labels').mkdir()
            for i, (status, seed) in enumerate((('accepted',986016), ('rejected',986017),
                                               ('failed',986018), ('accepted',984000))):
                key = 'a'+str(i); name = key+'_SUMOENV1.json'
                (root/'labels'/name).write_text(json.dumps(dict(search_id=key,world_id=seed,
                    verification=dict(status=status,verification_id='sv_'+key))))
            paths = final_snapshot_paths(root)
            self.assertEqual(paths, {root/'label_attempts/SUMOENV1/branches/sv_a0/SNAPSHOT.pt',
                                     root/'label_attempts/SUMOENV1/branches/sv_a1/SNAPSHOT.pt'})
            self.assertTrue(all('snapshots' not in p.parts for p in paths))

    def test_unrecognized_generation_and_path_traversal_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'labels').mkdir()
            for key, generation, vid in (('a','SUMOENV1','../bad'),('b','LIVE','sv_b')):
                (root/'labels'/(key+'_'+generation+'.json')).write_text(json.dumps(dict(
                    search_id=key,world_id=986016,verification=dict(status='accepted',verification_id=vid))))
            self.assertEqual(final_snapshot_paths(root), set())

    def test_runner_returns_selected_but_canonical_results_never_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'labels').mkdir()
            (root/'labels/a_SUMOENV1.json').write_text(json.dumps(dict(search_id='a',world_id=986016,
                verification=dict(status='accepted',verification_id='sv_a',branches=[dict(branch='policy'),dict(branch='candidate_0')]))))
            paths = final_snapshot_paths(root)
            self.assertEqual({p.name for p in paths}, {'SNAPSHOT.pt','policy.result.json','candidate_0.result.json'})
            self.assertTrue(all(p.name not in ('RESULT.json','RESULT.json.gz','observable.jsonl.gz') for p in paths))


if __name__ == '__main__': unittest.main()
