# REGISTRATION_WM_N1 — AMENDMENT 2 (disclosure): J4 wall-pilot data used as-is under wall law v2.1, with road-end relabelling

**Status:** 2026-10-01, world model session. A disclosure only. It changes no
target, split, test or variant.

- Amends `REGISTRATION_WM_N1.md`, LF-sha256 `666233aa…93246`, together with
  Amendment 1, `decb9656…`.

## Disclosure

1. **What J4 is.** The WM-N1 TRAIN/CAL corpus includes the J4 wall-pilot
   branches (`runs/campaign_walls_pilot_chunk4`, run under wall law v2.1).
   These are "walls held through the merge". They are not the owner's
   intended v2.2 design (walls release at the merge start; see
   `REGISTRATION_WALL_V22`).
2. **The bug.** In J4 the registered road-end truncation never fired
   (0 `road_end_release` events). The cause was an edge-blind `road_left_m`,
   `wall_request.py:119`, root-caused in `zcode/wallrt` eee755f.
   - In the 176 completed J4 runs there are 28 `member_left_network`
     aborts (the RUN_RECEIPT `wall_summary.abort_reason` count), all in
     hold: hold 60: 1, 120: 12, 240: 15.
   - The root-cause census (eee755f) counted 31. That count scanned all 200
     `wall_events.jsonl` files, including the 24 archived `.interrupted`
     attempts, which hold the other 3 (1 each). Those attempts are NOT in
     the WM-N1 corpus.
   - In every one of these aborts, both members were last on the exit edge
     1487606178.
3. **The relabel.** For WM-N1 training labels and wall tokens only, these
   `member_left_network` aborts are RELABELLED as road-end events
   (`road_end_relabelled=true` in the corpus metadata).
   - The J4 receipts keep their original labels.
   - The WM-59 ledger entry states that the truncation never fired.
4. **Physics unaffected.** The vehicle trajectories themselves are valid
   simulation data under the v2.1 law actually run. Only the event label
   differs.
5. **The chunk-5 TEST wall arms** will run under law v2.2, the rental payload
   after the patch. The J4-trained wall tokens therefore describe a
   different release rule than the TEST arms.
   - This train/test law shift is disclosed here.
   - WM-N1 wall-arm results on chunk 5 are reported separately, alongside
     the non-wall arms.
