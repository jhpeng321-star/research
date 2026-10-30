# Candidate B — build and qualification

No training is dispatched by staging. `launch.sh` is a prospective, single-card
gpuA entry point; MAIN must wait for S-WMREQUAL to release the card.

## Preset

`configs/c4_policy14_candB.toml` extends the complete longgrad_skip chain. Every
override has an inline source citation. CAPREL supplies the CTRL0-relative law;
ESLONG supplies M=2 antithetic head credit and exploration/Jacobian sigma;
GATEMEAS supplies CTRL0 reference, matched scoring and observable headroom starts;
SETUPAUDIT supplies actual entrant-mix execution and objective-match refusal;
LONGGRAD/LONGSKIP supply balance and exact skips; ZEROSTART zeroes the output
after explicit new-phase resume. The owner-required ranking switch is opt-in
in this preset only. Other presets/defaults retain their original settings.

## Required pre-training ranking check

From the immutable runtime root (Linux; CPU works; no SUMO):
```sh
export PYTHONPATH=src:. C4_ALLOWED_ROOTS=/root/autodl-tmp/c4
export WMS_DATA_V5_NATIVE_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/v5_native_q
export WMS_DATA_GENRAND_PLAIN_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/genrand_plain_q
L=/root/autodl-tmp/c4/restart20261026/arm/loop
python -m tools.candB.rankcheck --wm "$L/wm/c02/wm_s.pt" \
 --wm-config /root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml \
 --mappo /root/autodl-tmp/c4/mappores20261028/inputs/mappo_v5_final.pt \
 --current-policy /root/autodl-tmp/c4/candA_prod_keyerr/policy_000191.pt \
 --horizon-gate "$L/longint/gate_wm2/HORIZON_GATE_WM2_REG.json" \
 --boundary authority_gen/BOUNDARY_WM2.json \
 --pairing-receipt "$L/returns/pairing_c03.a1.json" \
 --cct2 authority/CCT2_b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436_mix.json \
 --n-starts 16 --device cpu --out authority_gen/CANDB_RANKCHECK.json
```

The CCT-2 must qualify **this** WM2/config, mix ON, all present/future requested
vehicles, road deficit + source waiting, CTRL0 reference, K8 paired joint q10.
Historical t0-HDV/K1 qualification is refused, not upgraded by this tool.
The existing objective contract and mix flag are checked before scoring; the
WM rollout wrapper also checks that the actual birth sampler is connected.
The same paired-return development exclusions, saved TRAIN sampler (if the
groups still match), focus proposal, headroom rejection path, update-index
entry/group/future seeds and boundary horizon regime are used. No VAL panel is
substituted. Every sampled start is retained, including low-observed-share starts.

Arms: native, CTRL0, frozen aecd7d70 MAPPO with its lane head ignored and native
lanes, CIRCLES, PI external-U, current residual policy, fresh-zero candidate B.
MAPPO absolute execution uses the setup-audit cap/Euler/native-background adapter.
PI/damping reuse `wms.train.cav_drive.TimelineControllers` from the verdict bridge,
including their state and native abstentions. All controller comparisons run
closed-loop in the same WM, entry and keyed K8 streams. Group/solo/native/CTRL0
rows pass through the **actual** Policy14Trainer actor objective, scale=1;
reported physical J is distinct from suffix/joint-q10/gain-gated training credit.
J90 and J300 come from the same 300s rollout. Low-share dense training caps are
retained and explicitly labelled `dense_horizon_s` in each raw row.

Output: durable partial/final JSON plus `.json.txt` text table; raw per-future J,
per-start objective loss/credits, actual mix assignments, full config/source/input
hashes, 5,000-draw seed-20261030 WORLD-cluster paired CIs. Repeated starts and K
draws are averaged inside a world before resampling. No optimizer updates occur.
Exit 0 requires owner R2's upper95 speed-only MAPPO-minus-CTRL0 J300 <=0, or a
CI-supported baseline ranking matching explicit real speed-only evidence via
`--real-evidence`. Lane-on historical MAPPO evidence is insufficient for that
alternative. Missing evidence/qualification or a failed gate cannot launch.

`policy14_cli --rankcheck FILE` independently recomputes the full runtime/config,
source manifest, boundary and pairing hashes before scoring/updates. Candidate B
requires the flag. Launch authority additionally binds actor/WM/config/qualification,
rankcheck and the fresh cap-relative parity receipt. Requalification after any
objective/mix/law change is mandatory; no old score baseline is relabelled.
