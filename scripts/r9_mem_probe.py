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
if os.environ.get("R9_PROBE_FINITE") == "1":
    # PROBE ONLY: zero-mass (-inf) event masses become a constant -50 so the value/gradient comparison is finite.
    _hl = mp.hybrid_log_masses
    def _finite_hl(c, tau, w, order=32, node_budget=None):
        e, sv = _hl(c, tau, w, order)
        return (torch.where(torch.isneginf(e), torch.full_like(e, -50.0), e),
                torch.where(torch.isneginf(sv), torch.full_like(sv, -50.0), sv))
    mp.hybrid_log_masses = _finite_hl
    lp.hybrid_log_masses = _finite_hl
_orig_call = LossComposer.__call__
calls = {"n": 0, "nonfinite": 0}
def call(self, ctx):
    b = _orig_call(self, ctx)
    calls["n"] += 1
    if os.environ.get("R9_PROBE_COMPARE") == "1":
        return b
    if not b.total.requires_grad:   # PROBE ONLY: chunks without a scored merge term still run a backward
        anchor = sum(p.sum() for p in ctx.targets.prepared["merge_pivot"]["model"].merge_pivot.parameters())
        b.total = b.total + 0.0 * anchor
    if not bool(torch.isfinite(b.total)):
        calls["nonfinite"] += 1
        b.total = torch.nan_to_num(b.total, nan=0.0, posinf=1e6, neginf=-1e6)
    return b
LossComposer.__call__ = call
stages = []
def _mark(tag):
    if torch.cuda.is_available():
        stages.append((tag, round(torch.cuda.max_memory_allocated() / 2**20), round(torch.cuda.memory_allocated() / 2**20)))
_orig_prepare = LossComposer.prepare
def prepare(self, targets):
    _mark("before_prepare"); r = _orig_prepare(self, targets); _mark("after_prepare"); return r
LossComposer.prepare = prepare
from wms.model.wms import WmSModel
_orig_rollout = WmSModel.rollout
def rollout(self, *a, **k):
    _mark("rollout_start"); r = _orig_rollout(self, *a, **k); _mark("rollout_end"); return r
WmSModel.rollout = rollout
done = {"n": 0}
def train_batch(self, batch, *, unroll_steps, truncation_steps=None):
    dev = self.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    self.optimizer.zero_grad(set_to_none=True)
    err = None
    stages.clear()
    try:
        parts = self.forward_backward(batch, unroll_steps=unroll_steps, truncation_steps=truncation_steps)
    except RuntimeError as e:   # CUDA OOM, or a batch without any scored merge term (no grad_fn): memory still recorded
        if not isinstance(e, torch.cuda.OutOfMemoryError) and 'does not require grad' not in str(e):
            raise
        import traceback; parts, err = {}, ("CUDA_OOM: " if isinstance(e, torch.cuda.OutOfMemoryError) else "NOGRAD: ") + str(e)[:300] + " || " + " | ".join(" ".join(l.split()) for l in traceback.format_tb(e.__traceback__)[-30:])
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    row = dict(step=done["n"], unroll_steps=unroll_steps, truncation_steps=truncation_steps,
               seconds=time.perf_counter() - t0, composer_calls=calls["n"], stages=list(stages), nonfinite_masked=calls["nonfinite"],
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
    if (err and err.startswith("CUDA_OOM")) or done["n"] >= STEPS:
        os._exit(0)
    return {**parts, "seconds": row["seconds"], "finite": True}

cmpstat = {"tried": 0, "compared": 0, "skipped_nonfinite_zero_mass": 0, "skipped_no_scored_term": 0, "true_oom": 0}
def train_batch_compare(self, batch, *, unroll_steps, truncation_steps=None):
    # PROBE ONLY: same batch and RNG with merge_pivot_state_st False vs True. Batches whose loss is non-finite
    # (zero-mass events) or that have no scored merge term are SKIPPED and counted, never masked.
    cmpstat["tried"] += 1
    res = {}
    names = [n for n, p in self.model.named_parameters() if p.requires_grad]
    skip = None
    for flag in (False, True):
        object.__setattr__(self.model.spec, "merge_pivot_state_st", flag)
        totals.clear(); torch.manual_seed(4321)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(4321); torch.cuda.reset_peak_memory_stats()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            self.forward_backward(batch, unroll_steps=unroll_steps, truncation_steps=truncation_steps)
        except torch.cuda.OutOfMemoryError:
            if flag:
                cmpstat["true_oom"] += 1; skip = "true_oom"
            else:
                raise
            break
        except RuntimeError as e:
            m = str(e)
            if "non-finite first_passage loss" in m or "non-finite WM-S loss" in m:
                skip = "skipped_nonfinite_zero_mass"
            elif "does not require grad" in m:
                skip = "skipped_no_scored_term"
            else:
                raise
            cmpstat[skip] += 1
            break
        res[flag] = dict(totals=list(totals),
                         grads={n: (p.grad.detach().clone() if p.grad is not None else None)
                                for n, p in self.model.named_parameters() if p.requires_grad},
                         peak_mb=(torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if skip is None:
        cmpstat["compared"] += 1
        row = dict(mode="compare", batch_try=cmpstat["tried"], unroll_steps=unroll_steps, truncation_steps=truncation_steps,
                   totals_true=res[True]["totals"], totals_false=res[False]["totals"],
                   max_abs_total_diff=max(abs(x - y) for x, y in zip(res[True]["totals"], res[False]["totals"])),
                   peak_true_mb=res[True]["peak_mb"], peak_false_mb=res[False]["peak_mb"], grads={})
        va, vb = [], []
        for n in names:
            a, b = res[True]["grads"][n], res[False]["grads"][n]
            if a is None or b is None:
                row["grads"][n] = dict(true=None if a is None else float(a.norm()), false=None if b is None else float(b.norm()))
                continue
            a, b = a.double().flatten(), b.double().flatten(); va.append(a); vb.append(b)
            na, nb = float(a.norm()), float(b.norm())
            row["grads"][n] = dict(true_norm=na, false_norm=nb, cosine=float(a @ b) / max(na * nb, 1e-300),
                                   rel_diff=float((a - b).norm()) / max(na, 1e-300))
        if va:
            A, B = torch.cat(va), torch.cat(vb)
            row["global"] = dict(true_norm=float(A.norm()), false_norm=float(B.norm()),
                                 cosine=float(A @ B) / max(float(A.norm()) * float(B.norm()), 1e-300),
                                 rel_diff=float((A - B).norm()) / max(float(A.norm()), 1e-300))
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + chr(10))
    with open(OUT + ".summary.json", "w") as f:
        json.dump(dict(cmpstat, last_skip=skip), f)
    done["n"] = cmpstat["compared"]
    if cmpstat["compared"] >= STEPS or cmpstat["tried"] >= int(os.environ.get("R9_PROBE_MAXTRY", "40")):
        os._exit(0)
    return {"seconds": 0.0, "finite": True}
totals = []
_prev_call = LossComposer.__call__
def call2(self, ctx):
    b = _prev_call(self, ctx); totals.append(float(b.total.detach())); return b
LossComposer.__call__ = call2
WmSTrainer.train_batch = train_batch_compare if os.environ.get("R9_PROBE_COMPARE") == "1" else train_batch
from wms.cli.__main__ import main
sys.exit(main(sys.argv[1:]))
