# P5_FIX — review BLOCKER 7 (J4 split trusted batch name alone)

Status: COMPLETED — fixed, tested, counts verified. Branch `wms/p5`, fix
commit **bf3f330** (parent 5bc66f1); report commit follows.

Fix (`output/wd01/wm_n1/p5splits/splits.py`): `is_j4` now requires all three
frozen J4 identities — campaign label (or chunk-4 wall-arm shape) AND the
wall action actually selected (`wall_action_selected`; explicit non-wall
policy beats a wall-shaped arm name) AND frozen wall law v2.1
(`has_frozen_j4_law`; explicit v2.2/J4B identity always fails, absent law
field defers to the J4 campaign identity). Native/PI rows of
`walls_pilot_chunk4` are NOT J4. Metadata only; no sealed file opened.
Tests: `test_j4_split_requires_wall_policy_not_only_batch_name` added;
10/10 pass; ruff clean. BLOCKER 7 is the only REVIEW.md item in wms/p5.

Counts — regenerated from the metadata-only inventory (2476 runs);
`_out/p5_splits.json` re-materialised byte-identical ⇒ counts UNCHANGED:
i: train 156 (J4 v2.1), excluded 1743 keys/2009 windows, dev 1743, reg 156;
ii: train 1847, excluded 210, dev 210, reg 11 (CAL-tune PI);
iii: train 2087, excluded 78, dev 78, reg 6 (CAL-tune W-mid); leak 0 all.
