"""PROBE ONLY (never for training): memory/time of real round-9 train steps (forward+backward, no optimizer step).

Runs the real `python -m wms train ...` entry with two probe-only patches: (1) the loss total is nan_to_num'd so
zero-mass events (inf NLL, a support issue fixed elsewhere) do not abort before backward; (2) Trainer.train_batch
only does forward_backward, records torch.cuda.max_memory_allocated / seconds, and the process exits after
R9_PROBE_STEPS steps. R9_PROBE_OLD=1 swaps in the pre-change O(T^2) hybrid_log_masses (tests/test_merge_pivot_memory.py).
Result JSON lines go to R9_PROBE_OUT.
"""
import json, os, sys, time
import torch
import wms.model.merge_pivot as mp
import wms.train.losses.merge_pivot as lp
from wms.train.losses.compose import LossComposer
from wms.train.trainer import WmSTrainer

OUT = os.environ["R9_PROBE_OUT"]
STEPS = int(os.environ.get("R9_PROBE_STEPS", "2"))
if os.environ.get("R9_PROBE_OLD") == "1":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from test_merge_pivot_memory import _reference
    def old(c, tau, w, order=32, node_budget=None):
        return _reference(c, tau, w, order)
    mp.hybrid_log_masses = old
    lp.hybrid_log_masses = old
_orig_call = LossComposer.__call__
calls = {"n": 0, "nonfinite": 0}
def call(self, ctx):
    b = _orig_call(self, ctx)
    calls["n"] += 1
    if not bool(torch.isfinite(b.total)):
        calls["nonfinite"] += 1
        b.total = torch.nan_to_num(b.total, nan=0.0, posinf=1e6, neginf=-1e6)
    return b
LossComposer.__call__ = call
done = {"n": 0}
def train_batch(self, batch, *, unroll_steps, truncation_steps=None):
    dev = self.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    self.optimizer.zero_grad(set_to_none=True)
    err = None
    try:
        parts = self.forward_backward(batch, unroll_steps=unroll_steps, truncation_steps=truncation_steps)
    except torch.cuda.OutOfMemoryError as e:
        import traceback; parts, err = {}, "CUDA_OOM: " + str(e)[:300] + " || " + " | ".join(" ".join(l.split()) for l in traceback.format_tb(e.__traceback__)[-9:])
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    row = dict(step=done["n"], unroll_steps=unroll_steps, truncation_steps=truncation_steps,
               seconds=time.perf_counter() - t0, composer_calls=calls["n"], nonfinite_masked=calls["nonfinite"],
               cuda_max_allocated_mb=(torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None),
               cuda_max_reserved_mb=(torch.cuda.max_memory_reserved(dev) / 2**20 if dev.type == "cuda" else None),
               error=err, loss=float(parts.get("loss", float("nan"))) if parts else None)
    try:
        import psutil; mi = psutil.Process().memory_info(); row['peak_rss_gb'] = getattr(mi, 'peak_wset', mi.rss) / 2**30
    except Exception:
        pass
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    done["n"] += 1
    if err or done["n"] >= STEPS:
        os._exit(0)
    return {**parts, "seconds": row["seconds"], "finite": True}
WmSTrainer.train_batch = train_batch
from wms.cli.__main__ import main
sys.exit(main(sys.argv[1:]))
