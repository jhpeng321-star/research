"""Build the before/after (pre-LC1 vs post-LC1-fixes) A2 metrics table
from the frozen eval_a2 summary and the post-fix re-run summary."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
pre = json.loads(
    (HERE / "results" / "eval_a2" / "summary.json").read_text(
        encoding="utf-8"))["summary"]["per_model_h"]
post = json.loads(
    (HERE / "results" / "eval_a2_post_lc1" / "summary.json").read_text(
        encoding="utf-8"))["summary"]["per_model_h"]

KEYS = ["n", "s_rmse_mean", "v_rmse_mean", "delay_err_mean",
        "delay_err_mae", "ttc3_err_mean", "hard_err_mean",
        "lc_merge_pred", "lc_merge_true", "lc_disc_pred", "lc_disc_true",
        "collisions_per_1000", "n_tracked_mean", "n_cav_mean",
        "n_free_road_true_mean"]
print(f"{'model|H':<22}{'metric':<24}{'pre-LC1':>14}{'post-LC1':>14}"
      f"{'delta':>12}")
for k in sorted(pre):
    a, b = pre[k], post[k]
    for m in KEYS:
        va, vb = a.get(m), b.get(m)
        if va is None or vb is None:
            continue
        d = vb - va
        if abs(d) < 1e-9:
            continue
        fa = f"{va:.4g}" if isinstance(va, float) else str(va)
        fb = f"{vb:.4g}" if isinstance(vb, float) else str(vb)
        fd = f"{d:.4g}" if isinstance(d, float) else str(d)
        print(f"{k:<22}{m:<24}{fa:>14}{fb:>14}{fd:>12}")
    print()
