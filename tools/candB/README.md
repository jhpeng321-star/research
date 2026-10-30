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

For WM-only diagnostic scoring with unqualified historical CCT-2, explicitly add
`--measure-only`. This retains the exact candidate-B loss, K8/CTRL0/mix/config/source
bindings but marks the receipt `measure_only=true`, `passed=false` and unlaunchable.
The numerical R2 outcome is separately `ranking_passed`; no authority is generated.
`--mappo-speed PATH` adds a hash/iteration-bound, partially trained native-lane export.
CUDA measurements enforce `--gpu-memory-fraction .40` (maximum); no worker pool is
started by rankcheck. `objective90_loss` is the resolved dense-prefix actor loss
(low-share caps retained), `objective300_loss` the exact dense+gain-gated long loss.
If an old checkpoint binds CRLF driver bytes, measurement-only
`--current-policy-driver ORIGINAL` verifies exact CRLF-to-LF equality, loads under
the original checkpoint contract, then restores the staged LF execution driver.
Both driver hashes and this normalization are retained in the measurement receipt.

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
Qualification must also bind `wm_config_sha256`; a matching checkpoint alone
does not qualify a different inference configuration.
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
The current head is evaluated under the candidate-B execution law; `candB_zero`
uses the same loaded representations with only its output layer zeroed.

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

## Staging and prospective launch

```sh
# Local source-only archive. Optional `gpuA FRESH_RUNTIME` arguments only copy/extract.
PYTHONPATH=src:. bash tools/candB/stage.sh /existing/dir/candB_SOURCE.tar.gz
# In the staged immutable runtime, after matching requalification + ranking + parity:
python -m tools.candB.authority build --wm "$WM" --wm-config "$WM_CONFIG" \
 --current-policy "$CURRENT" --cct2 "$CCT2" --horizon-gate "$HGATE" \
 --boundary authority_gen/BOUNDARY_WM2.json --pairing-receipt "$PAIRING" \
 --real-returns "$RETURNS" --native-returns "$NATIVE_RETURNS" --gate-panel "$PANEL" \
 --parity authority_gen/CAPREL_PARITY.json --rankcheck authority_gen/CANDB_RANKCHECK.json \
 --out authority_gen/CANDB_AUTHORITY.json
python -m tools.candB.authority verify authority_gen/CANDB_AUTHORITY.json
RUNTIME=/root/autodl-tmp/c4/candB_SOURCE bash tools/candB/launch.sh /root/autodl-tmp/c4/candB_dry dry
RUNTIME=/root/autodl-tmp/c4/candB_SOURCE bash tools/candB/launch.sh /root/autodl-tmp/c4/candB_pilot pilot
```

These commands are **not executed by S-INTEG**. Archive creation refuses dirty
trees; LF bytes come from committed Git blobs and every staged file is hash-bound.
All input artifacts (including real returns and gate panel) are verified. Launch
refuses any running GPU compute PID, so it cannot overlap S-WMREQUAL. Two serial
forward workers use `.24` each, one coordinator uses `.40`, future budget `.90`;
memory/concurrency remain prospective, not GPU-qualified by the CPU suite.
`dry` means two real-start optimizer updates (not a no-training validation).
MAIN must apply the registered phase-10 raw-dJ monitoring rule and STOP if it fails.
