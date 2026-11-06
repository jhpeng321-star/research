# OPEN_SEARCH: a roaming WM search over cooperation programs (WM-C, design note)

Owner direction (2026-11-06): once the WM qualifies, the search must start broad and roam the cooperation
space instead of only ranking known families. This note describes the mode `tools/wzprop/open_search.py`.
CLI: `python -m tools.wzprop opensearch --moments M --out OS.jsonl --wm auto [--config JSON]`. It uses zero SUMO and
is tested on CPU with the stub WM (`tools/wzprop/test_wzprop_cpu.py`, open_search tests).

## 1. What is searched
A program covers a group of 2 to 6 CAVs that sit in one V2V-local cluster (chained within 100 m) in the member
zone [X_B-1800, X_B-400] on `up`. Each member gets its own program:
- an optional lane change to an adjacent lane (the closing lane included), starting at `lc_from`;
- a speed action: hold toward `v_w`, which accelerates or slows the vehicle, or accelerate at amax;
- `dmax`, a duration, a start offset `sp_from`, and an own time gap `tau`;
- release at the end of the duration or at the drop.

Every value comes from `lcm_plan` constants. Those constants are the registered NEWMAP2, COOPKINDS, PLANGEN2 and
WMDATA ranges:
- R_GRID: v_w 15/18/22 m/s, hold 90/120/180 s, dmax 1.0/1.5;
- OFFSETS: 0/20/40/60 s;
- TAU_SET: none/0.6/2.0.

Programs use one of two executors:
- A program with no lane change, no tau and no accel is written as nm2 `hold` members. These run on nm2_core and
  are scored by the current WM planning API.
- Any other program is written as LCMOVES `prog` members (`lcm_plan.prog`). These run on lcm_core. They are
  searched only when the WM declares `supported_roles` that include `prog`. WM-A's API is hold-only today, which is
  an interface need.

Both executors use the same variant format as N3/PLANGEN/LCMOVES. The SUMO admission rule is unchanged. The plug-in
re-derives every member spec and checks the registered ranges before any SUMO time is spent.

## 2. Algorithm per moment (in imagination, batched)
1. **Broad initialisation.** Draw `n_init` random programs (default 48) over random local groups and sizes. Add the
   known families as seeds: structured walls and pairs, plus the LCMOVES builders when `prog` is supported.
2. **Score.** For each candidate, the margin M = min(J_base, min_i J_solo_i) - J_group is computed per WM draw,
   paired by common random numbers. The base, the group and every solo go in the same WM call, which is batched on
   the GPU through `max_rows`. The score is `LCB(M) - lambda * sd(M)`, where LCB = mean - 1.645·sd/√K and
   lambda = 0.5. The sd penalty keeps the optimiser away from plans the WM is unsure about, so it cannot exploit WM
   errors. With a WM ensemble, sd is taken across members × draws.
3. **Diversity archive (MAP-Elites).** A cell is the tuple (number of CAVs, lanes used after the programs, number
   of lane changes, mean speed-change bin, distance-to-drop bin, spread bin). The archive keeps the best-scoring
   plan per cell.
4. **Iterative refinement.** This runs for `generations` rounds (default 3). Parents are the global top-`n_elite`
   plus the archive top-`n_elite`. Children are split into thirds:
   - mutation: each gene is re-drawn with probability 0.35;
   - uniform crossover of two parents on the same group;
   - cross-entropy sampling: a per-gene categorical fitted on that group's elites, with 0.5 smoothing toward
     uniform.
5. **Final.** Re-score with K_final = 8 draws: the top `finalists` plus archive elites from other cells.

## 3. Choosing what SUMO verifies (budgeted, later)
Each moment has k slots, and the share is configurable:
- **50 % top score**;
- **30 % diverse archive elites**, from cells not yet covered;
- **20 % high uncertainty**: the largest sd(M) among candidates with a positive mean M. This is the active-learning
  share.

Each proposal records its slot, its cell and its origin (`init`, `mutation`, `crossover`, `cem`, or a family seed).
Every SUMO verdict goes back to the WM as training data through `open_search.feedback_rows`. Each row holds the
state reference (the tape path and t0), the action, the measured per-replicate T and S, the admission, and the WM
prediction.

## 4. Reporting
`compare --moments ...` lists every verified plan of every arm, with its family or program origin and its
descriptor cell. This feeds the matched WM vs structured vs random comparison (S-WMSEARCH) and shows whether the WM
finds plans outside the known families.

## 5. Limits and needs
- WM-A's `wms.planapi` executes `hold` members only. Lane-change programs need a `prog` executor in the WM. Until it
  exists, the open search roams over per-member heterogeneous hold programs: members, onset, duration, speed target
  and dmax.
- Search cost per moment at the defaults is about 1,500-2,500 action-draws (H = 300 s). This needs the GPU. CPU
  rollouts are about 100 s per step locally.
- The archive and the slot split are deterministic given the moment, which is seeded by the sid hash. That makes a
  run reproducible.
