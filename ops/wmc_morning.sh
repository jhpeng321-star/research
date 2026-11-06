#!/bin/bash
# WM-C morning chain (2026-11-07). Steps are run BY HAND in order; each line is one step.
# Preconditions:
#   (G) GPU steps only after WM-E's fine-tune has finished (ft_r1/STATUS.json state != training).
#   (Q) SUMO verification (WMVERIFY) only after WM_QUAL_GATE PASS on the WM-E final checkpoint.
# Sharing (agreed with WMEV): WM-C <= 8 processes total on gpuA (nice 10, mem-fraction 0.12 each), WMEV <= 6;
# WM-D's qualification run has priority; halve if WM-E/WM-D slow down.
set -u
B=/root/autodl-fs/wmc_rankcheck_20261106          # gpuA work root; code_v6 = port 7138ba9a (prog + fairness trace) + wzprop 9e1a8d01
FINAL=${FINAL:?set FINAL=<WM-E final wm_s.pt (sha from WM-E)>}
cd $B/code_v6

# 1) (G) strategy-level search (owner 04:03: 15 m/s banned, 10 s/veh group bound; incumbent wall_g_solo2), three archives
#    delay / bounded / fairer (+ strict fair flag), per-vehicle trace fairness; cell order = SUMO verification cell first.
#    Prefer starting after WM-A's CRN fix (rows in one call CRN-paired); otherwise note it in the report.
#    6 parallel WM workers; finals on 30 states per cell, K 6.
PYTHONPATH=src:. nohup nice -n 10 /root/miniconda3/bin/python -B -m tools.wzprop strategy \
  --states data/regime_STATES.json --cells D3200_p60,D2800_p60,D2400_p60,D3200_p40 --out $B/strategy_v3_final \
  --wm wma:WMA_READY.json --wm-arg checkpoint=$FINAL --wm-arg map_spec=maps/wz3to2/MAP_SPEC.json \
  --wm-arg device=cuda --wm-arg max_rows=48 --workers 6 --mem-fraction 0.12 \
  --config '{"n_final_states": 30, "K_final": 6}' > $B/strategy_v3_final.log 2>&1 &
#    -> $B/strategy_v3_final/ARMS.json  {cell: incumbent, banned_comparator, reference[2], wm[4], wm_bounded[4], wm_fairer[4],
#       wm_fair, structured[4], random[4], best per archive, comparators, fairness}
#       STRATEGY_SEARCH_<cell>.json (finals: gains vs u0 / incumbent + LCBs, group_inc s/veh + UCB, CVaR90 change, flags)
#    Hand ARMS.json to WMVERIFY (a42a3d5c086a6e51b) when the run ends (cells run in the order above).

# 2) (G) final rank check, full protocol, 2 workers (PLANGEN K8 + solos for S-labelled), after step 1 or in spare slots
#    bash $B/run_ckpt.sh wme_final $FINAL 2 0
#    metrics: python -m tools.wzprop rankmetrics --summary PG_SUMMARY.json --scores <RANK_part*.jsonl concatenated> \
#             --label WME-final --out RANKCHECK_WME_final.json  -> update WMC_RANKCHECK.md + CLEAN1_STATUS WMPROP

# 3) phase-A CPU rank check (running since 23:17): concatenate $B/wmeA_cpu_T_K2/SCORES_part*.jsonl -> rankmetrics

# 4) (Q) SUMO verification: WMVERIFY runs its wmverify_hook on the u0 plant with ARMS.json (registration: S-WMSEARCH
#    amendment 1 / WMVERIFY's own registration). Fallback (only if WMVERIFY is not running): tools.wzprop.wmsearch_u0
#    plan/run/analyze on cpuV, see WMSEARCH_AMENDMENT1.md.
