# FULL-v5 engineering freeze / teacher availability

Source `b2df7458a41f5f4d3a048532bc40d8d4435ec381` and recipe
`configs/wm_phys_v5.toml` SHA256
`023fa74a4d3059115ec0a5f28d39030a76137d871db5798eb1d23a13fc1e32c5`
are frozen before any new final outcome or score. Candidate combines the four
reviewed source families and pre-existing development-fitted physical inputs;
no final-panel preference selected this combination.

Runtime: gpuA `/root/autodl-tmp/c4/rb2int20261101/phys_code_b2df7458a41f`.
Archive SHA83cfe685d21c975a7776eb06817e31beaee9d152a32f4ca5f26af647f4128d1c.
WM2 checkpoint/encoder SHA remainsb8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436.

Verification: all-four-source CPU135PASS; closed60 dev969420/969421 PASS42.93s;
allfivearm/H600/K1 same worlds PASS220.56/232.72s with finite/conserved source
ledger, original-ID60s endpoints and learned-correction guard. Completion snapshot
`CANARY_V5_B2DF7458_STATUS03.json` contains immutable task/identity bindings.
Own446421 exited. Do not repeat completed canaries or count K1 as final K8.

Staged parameter fit/selection separation, inherited hashes and406 source/artifact
manifest PASS: `EXPOSURE_B2DF7458_GPUA_V2.json` (SHA5b2bf147b63127fc5efafa60776add8309457b9aac2246eaaa77a2fca4d936a8).
Declared fits only969400–969431. Historical aggregate exposure is disclosed and
the full12-cell precision limitation remains. This is an engineering availability
signal, not distributional qualification or switching/policy-adoption evidence.

S-WMTEACH notified through shared `RB2_INT_FULL_V5_READY.md` and HANDOFF.md.
Its FIT path already uses candidate_future. Roster-only diagnostic_future needs
an adapter update to retain LC population, calibrated arrival wrapper and CAV
settings; copying the recipe alone would silently omit these new behaviors.
Unknown history remains cold/prior, explicitly weaker than authentic conditioning.

LP recipe `configs/wm_phys_v5_lp.toml` SHA95802e9bc6d4661cd86dd738db0853cb0ce6055ebcf23194943a5c745f945c15
is fixed alongside v5; only the physical departure distribution differs. Its separate
full canary449208 has now PASSED both dev worlds218.12/215.52s and EXITED;
`CANARY_LP_B2DF7458_STATUS01.json`. Both variants are engineering READY. Final
input/bridge/dispatch and full evaluation remain.
