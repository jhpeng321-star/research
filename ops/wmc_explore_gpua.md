# WM-C creative exploration on gpuA (strategy_explore, 2026-11-07)

WM = proposal generator only (it failed qualification as a predictor). This run gives SUMO (WMVERIFY) a varied set
of new cooperation patterns. It does not rank anything. The new module `tools/wzprop/strategy_explore.py` imports
`strategy_search` unchanged (sha256 763e7ae6...) and uses the same WM backend construction as
`python -m tools.wzprop strategy`.

Preconditions: (G) GPU slot agreed (WM-C <= 8 processes on gpuA, nice 10, mem-fraction 0.12 each); WM-E final
checkpoint `$FINAL` known; nothing in this run touches SUMO.

## 1. Deploy to a NEW code dir `code_v6x` (code_v6 stays as it is)

Local (Windows). Export the committed LF bytes, never the CRLF working copy:

```bash
C=<commit sha of 'C4 WM-C explore: ...'>        # from git log in E:/agent-work/c4-wz-wm-explore
cd E:/agent-work/c4-wz-wm-explore
mkdir -p /e/Temp/wmc_explore_deploy
git show $C:tools/wzprop/strategy_explore.py          > /e/Temp/wmc_explore_deploy/strategy_explore.py
git show $C:tools/wzprop/test_strategy_explore_cpu.py > /e/Temp/wmc_explore_deploy/test_strategy_explore_cpu.py
sha256sum /e/Temp/wmc_explore_deploy/*.py
#   expected strategy_explore.py          9b4045c9abf30a9c1ebbe9dc7d1c825ce1bf5a57bce534746ffae87b4596ec1d
#   expected test_strategy_explore_cpu.py 8838de3d56ae51d8d073b0f9ab931156fcb34891eee8916def7efc729dcc5ea6
scp /e/Temp/wmc_explore_deploy/*.py gpuA:/root/autodl-fs/wmc_rankcheck_20261106/   # use your usual gpuA ssh alias/port
```

On gpuA:

```bash
B=/root/autodl-fs/wmc_rankcheck_20261106
test ! -e $B/code_v6x || { echo "code_v6x exists - STOP"; exit 1; }
cp -a $B/code_v6 $B/code_v6x
cp $B/strategy_explore.py          $B/code_v6x/tools/wzprop/strategy_explore.py
cp $B/test_strategy_explore_cpu.py $B/code_v6x/tools/wzprop/test_strategy_explore_cpu.py
cd $B/code_v6x
# The new module depends on the strategy_search API at 65aeefc3. Check it before launching:
sha256sum tools/wzprop/strategy_search.py tools/wzprop/__main__.py tools/wzprop/wm_backend.py \
          tools/wzprop/strategy_explore.py | tee $B/code_v6x/SHA256SUMS_explore.txt
#   strategy_search.py must be 763e7ae692bf8813051740750a64684ed4b749f8f2dbfbf493d6a4e26698b306 (65aeefc3)
#   __main__.py at 65aeefc3:  f22ff2c1279340f3a52124367c39149448b6f0b66d496e672ef0b9e3aa7a0361
#   wm_backend.py at 65aeefc3: 7edd48d4e3f41bad2d302cb338ddc6a7fc1e3d81980160052be65881f9aa0686
#   If strategy_search.py differs: STOP and report. Do not overlay files by hand.
#   If only wm_backend.py differs (a later WM-A port on gpuA), that is the backend in use: record it and continue.
date -u +%FT%TZ >> $B/code_v6x/SHA256SUMS_explore.txt
```

CPU checks (stub WM only, about 3 min):

```bash
cd $B/code_v6x
PYTHONPATH=src:. /root/miniconda3/bin/python -B -m pytest -q tools/wzprop/test_strategy_explore_cpu.py
#   test_protected_modules_unchanged checks strategy_search/__main__ sha (CRLF-normalised); the lanes.py
#   cross-check runs if torch + src/wms import (it does on gpuA).
PYTHONPATH=src:. /root/miniconda3/bin/python -B -m tools.wzprop.strategy_explore \
  --states data/regime_STATES.json --cells D3200_p60 --out $B/explore_smoke_stub --wm stub \
  --config '{"n_init":4,"generations":1,"n_children":4,"n_screen_states":3,"K_screen":1,"finalists":2,"n_final_states":3,"K_final":1}'
```

## 2. Launch: one cell per run, a NEW `--out` per cell, 8 workers, run in order

Cell order is the same as strategy_v3_final: the SUMO verification cell goes first. All four cells run one after
another in a single detached chain. Only 8 WM processes run at any time.

```bash
B=/root/autodl-fs/wmc_rankcheck_20261106
FINAL=<WM-E final wm_s.pt>          # same checkpoint as strategy_v3_final; record its sha256 below
sha256sum $FINAL >> $B/code_v6x/SHA256SUMS_explore.txt
cd $B/code_v6x
CFG='{"n_init":150,"generations":6,"n_children":48,"n_elite":4,"p_mutate":0.6,"lam":0.0,"n_screen_states":16,"K_screen":3,"finalists":12,"n_final_states":30,"K_final":6,"immigrants":0.15,"min_feasible":0.25,"novelty_k":10,"seed":20261108,"min_gate_pass":0.3,"novel_harm_s":5.0,"select_n":6}'
export B FINAL CFG                   # the detached shell reads these (no nested quoting)
setsid nohup nice -n 10 bash -c 'set -u
for CELL in D3200_p60 D2800_p60 D2400_p60 D3200_p40; do
  PYTHONPATH=src:. /root/miniconda3/bin/python -B -m tools.wzprop.strategy_explore \
    --states data/regime_STATES.json --cells "$CELL" --out "$B/explore_v1_$CELL" \
    --wm wma:WMA_READY.json --wm-arg checkpoint="$FINAL" --wm-arg map_spec=maps/wz3to2/MAP_SPEC.json \
    --wm-arg device=cuda --wm-arg max_rows=32 --workers 8 --mem-fraction 0.12 --k 4 \
    --config "$CFG" > "$B/explore_v1_$CELL.log" 2>&1 || exit 1
done' > $B/explore_v1_chain.log 2>&1 < /dev/null &
echo $! > $B/explore_v1_chain.pid
```

Monitor: `tail -f $B/explore_v1_D3200_p60/LOG_D3200_p60.jsonl` (one line per screened batch: evaluated,
rollouts, archive sizes incl. `archive_novel`). Stop: `kill -- -$(cat $B/explore_v1_chain.pid)` (process group).

Outputs per cell, in `$B/explore_v1_<cell>/`:
- `STRATEGY_SEARCH_<cell>.json`: the SS v3 schema, plus `explore_schema` = `wmc-strategy-explore-v1`, the `novel`
  archive (fine descriptor, novelty, gate_pass_rate), per-final `gate_pass_rate` / `gate_requests` / `support`,
  and the `selection` block.
- `ARMS.json`: the usual groups plus `wm_novel` (specs only, which pass the WMVERIFY rules) and `wm_novel_tags`
  (support in/out, cluster, predicted score, worst-group increase, gate pass, distance to the nearest comparator).
  Do not rank in-support picks against out-of-support picks. WMVERIFY's GROUPS does not list `wm_novel`, so pass
  the group to WMVERIFY explicitly (its `extra=` path) or merge it by hand.

## 3. Runtime estimate (per cell)

- Screening covers 17 seeds + 150 initial + 6 x 48 children = 455 strategies on 16 states with K = 3. At the owner's
  figure of ~25 WM rollouts per screened strategy that is ~11.4k rollouts. The CPU stub dry run (40 states, full
  defaults) counted ~9.5k screening rollouts.
- Finals: up to 4 x 12 + 2 comparators = 50 strategies on 30 states with K = 6. At ~64 rollouts per final that is
  ~3.2k. Every strategy activates on every state in the stub, so the stub upper bound is ~9k.
- Total: ~14.6k rollouts (owner figures) to ~18.5k (stub count). At 1.0-1.3 rollouts/s with 8 workers that is
  about 3.1-4.1 h per cell, or up to ~5.1 h. All four cells take about 13-20 h.
- CPU overhead outside the WM (novelty, gate, clustering) is under 1 min per cell.
- To shorten the run: `"finalists": 8` gives at most 34 finals; `"generations": 4` drops 96 screened strategies.

## 4. Notes

- `disagreement` is still reported but is never used for ranking (lam 0.0). The `bonus` hook defaults to 0.
- Lane gate: `gate_pass_rate` applies the exact LaneRules own-sensing gate to the OBSERVABLE start state. For
  onset > 0 it uses constant-velocity extrapolation. This is a proxy: the WM exposes no imagined per-vehicle states.
  Lane-change plans need a rate >= 0.3 to enter `novel` / `wm_novel`. Hold-only plans are not affected.
- The in/out-of-support tag is a heuristic: per activation <= 3 roles and <= 1 lane changer. The WM backends expose
  no OOD signal.
