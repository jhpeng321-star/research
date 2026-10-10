# Experiment B/C execution runbook

Once `runs/rllib/shepherd_hybrid_meta_v1/train.out` shows `Iter 99 | steps
594,000/600,000` (near completion), run the orchestrators in order. All
scripts live in `D:\research-dense`.

## Env setup (every shell)

```bash
cd /d/research-dense
source /c/Users/jpeng/Miniconda3/etc/profile.d/conda.sh
conda activate rl_driving
export SUMO_HOME='C:\Program Files (x86)\Eclipse\Sumo'
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="D:/research-dense;$PYTHONPATH"
```

## Step 1 — after B training finishes

```bash
bash after_expB_finish.sh
```

Does (SEQUENTIAL, to avoid the eval/RLlib port-stall seen on 2026-08-12):
1. Evaluate B candidates (best / latest iter_* / final) on 2700 (8 seeds,
   shared no-op cache) — logs under `runs/eval/orch_evalB_*.log`.
2. Launch Experiment C training (`launch_expC.sh`, run dir
   `runs/rllib/shepherd_hybrid_meta_v2`, ~4h).

## Step 2 — while C trains: multi-density eval of B best

Pick B's best checkpoint from Step 1 (compare wins in
`runs/eval/orch_evalB_best.log`), then:

```bash
python -u _eval_hybrid_meta_v1.py --ckpt <B_BEST_CKPT> --main 2000 --workers 4
python -u _eval_hybrid_meta_v1.py --ckpt <B_BEST_CKPT> --main 2400 --workers 4
# 2700 already done in step 1; these reuse shared no-op caches.
```

## Step 3 — after C training finishes

```bash
bash after_expC_finish.sh
```

Does:
1. Evaluate C bare + C+health-mask on 2700 (8 seeds).
2. Dynamics probes (seed 105/119, mask ON + mask OFF control) + Spearman
   analysis -> `runs/eval/expC_dynamics/`.
3. B vs C comparison table (`_compare_expBC.py`).

## Step 4 — multi-density eval of C best

```bash
python -u _eval_hybrid_meta_v1.py --ckpt <C_BEST_CKPT> --main 2000 --workers 4
python -u _eval_hybrid_meta_v1.py --ckpt <C_BEST_CKPT> --main 2400 --workers 4
python -u _eval_hybrid_meta_v1.py --ckpt <C_BEST_CKPT> --main 2700 --workers 4
python -u _eval_hybrid_meta_v1.py --ckpt <C_BEST_CKPT> --main 2700 --workers 4 --health-mask
```

## Step 5 — fill the report

`docs/expC_metastable_health_mask.md` — fill §4/§5/§6 with the numbers from
the eval logs and `_compare_expBC.py` output.

## Completion criteria mapping

- (1) 2700 win rule: `_compare_expBC.py` output (C >= B, >= 50%).
- (2) dynamic relationship: `after_expC_finish.sh` step 2 Spearman output
  (mask ON: coast~+h, metering~-h; mask OFF control for attribution).
- (3) multi-density: steps 2/4 per-density summaries — no systematic TTC
  regression vs no-op (noop caches: `runs/eval/noop_cache_{light_l2000,
  light_m2400r400, light_m2700r400}/`).

## Key files

- Eval driver: `_eval_hybrid_meta_v1.py` (`--main 2000|2400|2700`,
  `--health-mask`, `--only-noop`)
- Dynamics: `_probe_expC_dynamics.py` (`--no-mask` control),
  `_analyze_expC_dynamics.py`
- Compare: `_compare_expBC.py`
- C training: `launch_expC.sh` (`--metastable-health-mask`)
