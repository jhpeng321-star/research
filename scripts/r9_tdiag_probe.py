"""PROBE ONLY (never for training): diagnose non-finite merge-pivot batches and the grad path.

Runs the real `python -m wms train ...` entry; patches (diagnosis only, nothing here enters training code):
  * `score_chunk` is wrapped: after the real call, every slot with a zero-mass event is re-scored ALONE
    (same tensors sliced to the slot, real carry) and described (path tf/on, window index, veto class);
  * the loss total is nan_to_num'd so a non-finite batch still reaches backward;
  * `Trainer.train_batch` runs forward_backward, records finiteness, requires_grad of the 8 merge_pivot
    tensors and their gradient norms, and writes one JSON line per batch to R9_TD_OUT; exits after R9_TD_STEPS.
"""
import json, os, sys, time
import torch
import wms.train.losses.merge_pivot as lp
from wms.train.losses.compose import LossComposer
from wms.train.trainer import WmSTrainer

OUT = os.environ["R9_TD_OUT"]
STEPS = int(os.environ.get("R9_TD_STEPS", "10"))
CUR = {"prep": None, "ctx": None, "recs": [], "calls": 0}
_orig_score = lp.score_chunk


def _slice_kw(kw, key, b, k):
    v = kw.get(key)
    return None if v is None else v[b:b + 1, :, k:k + 1]


def traced(c, tau, risk, event, **kw):
    out = _orig_score(c, tau, risk, event, **kw)
    stats = out[2]
    if stats["zero_mass_events"] or stats["zero_survival"] or stats["physical_fail_events"]:
        path = "on" if kw.get("support") is not None else "tf"
        carry = kw.get("carry") or {}
        with torch.no_grad():
            ev = event.bool() & risk.bool()
            for b, i, k in torch.nonzero(ev).tolist():
                sub = {}
                for key in ("known_entry", "reset", "eligible", "support", "stop"):
                    sub[key] = _slice_kw({**kw, "known_entry": kw["known_entry"], "reset": kw["reset"]}, key, b, k)
                sub["w"] = kw.get("w"); sub["quadrature_order"] = kw.get("quadrature_order", 32)
                sub["initial_maximum"] = (None if kw.get("initial_maximum") is None
                                          else kw["initial_maximum"][b:b + 1, k:k + 1])
                sub["initial_history"] = (None if kw.get("initial_history") is None
                                          else kw["initial_history"][b:b + 1, k:k + 1])
                sub["carry"] = {key: val[b:b + 1, k:k + 1] for key, val in carry.items()} or None
                try:
                    _, _, st = _orig_score(c[b:b + 1, :, k:k + 1], tau, risk[b:b + 1, :, k:k + 1],
                                           event[b:b + 1, :, k:k + 1], **sub)
                except Exception as e:  # noqa
                    st = {"error": str(e)[:100]}
                if st.get("zero_mass_events") or st.get("physical_fail_events") or st.get("zero_survival") or "error" in st:
                    prep, ctx = CUR["prep"], CUR["ctx"]
                    g = prep["geometry"]
                    c0 = ctx.c0
                    t = c0 + i
                    rec = dict(path=path, call=CUR["calls"], c0=c0, c1=ctx.c1, b=b, k=k, t=t, i=i,
                               st={x: y for x, y in st.items() if y},
                               known=bool(prep["known"][b, t, k]),
                               rec_eligible=bool(g["eligible"][b, t, k]),
                               rec_phys=bool(g["physical_eligible"][b, t, k]),
                               rec_resv=bool(g["reservation_veto"][b, t, k]),
                               rec_overlap=bool(g["overlap"][b, t, k]),
                               rec_fixed=bool(g["fixed_overlap"][b, t, k]),
                               rec_scene=bool(g["scene_valid"][b, t, k]),
                               rec_c=float(model_c(prep, g, b, t, k)),
                               x_route=float(g["route_s"][b, t, k]), v=float(g["v"][b, t, k]),
                               has_f=bool(g["has_f"][b, t, k]), has_l=bool(g["has_l"][b, t, k]),
                               gap_f=float(g["gap_f"][b, t, k]), gap_l=float(g["gap_l"][b, t, k]),
                               in_window_pre=bool(prep["known_before"][b, t, k]))
                    if path == "on":
                        pr = ctx.pred
                        rec.update(on_eligible=bool(pr["merge_pivot_eligible"][b, i, k]),
                                   on_phys=bool(pr["merge_pivot_physical_eligible"][b, i, k]),
                                   on_resv=bool(pr["merge_pivot_reservation_veto"][b, i, k]),
                                   on_support=bool(kw["support"][b, i, k]), on_c=float(c[b, i, k]),
                                   wm_pos=float(pr["positions"][b, i, k]), wm_lane=int(pr["lanes"][b, i, k]),
                                   rec_pos=float(ctx.targets.states[b, t, k, 0]))
                    CUR["recs"].append(rec)
    return out


def model_c(prep, g, b, t, k):
    with torch.no_grad():
        geo = {key: val[b:b + 1, t:t + 1, k:k + 1] for key, val in g.items()}
        return prep["model"].merge_pivot(geo)[0, 0, 0]


lp.score_chunk = traced
_orig_term = lp.MergePivotTerm.__call__


def term_call(self, ctx):
    CUR["prep"] = ctx.targets.prepared[self.name]; CUR["ctx"] = ctx; CUR["calls"] += 1
    return _orig_term(self, ctx)


lp.MergePivotTerm.__call__ = term_call
_orig_call = LossComposer.__call__
info = {"nonfinite": 0, "chunks": 0, "total_requires_grad": []}


def call(self, ctx):
    b = _orig_call(self, ctx)
    info["chunks"] += 1
    info["total_requires_grad"].append(bool(torch.is_tensor(b.total) and b.total.requires_grad))
    if torch.is_tensor(b.total) and not bool(torch.isfinite(b.total)):
        info["nonfinite"] += 1
        b.total = torch.nan_to_num(b.total, nan=0.0, posinf=1e6, neginf=-1e6)   # PROBE ONLY
    return b


LossComposer.__call__ = call
done = {"n": 0}


def train_batch(self, batch, *, unroll_steps, truncation_steps=None):
    t0 = time.perf_counter()
    self.optimizer.zero_grad(set_to_none=True)
    CUR["recs"].clear(); CUR["calls"] = 0
    info.update(nonfinite=0, chunks=0, total_requires_grad=[])
    err = None
    pv = {n: p.requires_grad for n, p in self.model.named_parameters() if n.startswith("merge_pivot")}
    try:
        parts = self.forward_backward(batch, unroll_steps=unroll_steps, truncation_steps=truncation_steps)
    except RuntimeError as e:
        import traceback
        err = str(e)[:300] + " || " + " | ".join(" ".join(l.split()) for l in traceback.format_tb(e.__traceback__)[-6:]); parts = {}
    metrics = {}
    try:
        from wms.train.losses.merge_pivot import merge_pivot_metrics
        if CUR["prep"] is not None:
            metrics = merge_pivot_metrics(CUR["prep"], 0.5)
    except Exception as e:  # noqa
        metrics = {"metrics_error": str(e)}
    grads = {n: (None if p.grad is None else float(p.grad.norm()))
             for n, p in self.model.named_parameters() if n.startswith("merge_pivot")}
    row = dict(step=done["n"], unroll=unroll_steps, trunc=truncation_steps, seconds=time.perf_counter() - t0,
               nonfinite_chunks=info["nonfinite"], chunks=info["chunks"], total_requires_grad=info["total_requires_grad"],
               requires_grad=pv, grad_norms=grads, error=err, loss=parts.get("loss"),
               metrics={k: v for k, v in {**metrics, **parts}.items() if k.startswith("merge_pivot_") and not isinstance(v, (list, dict))},
               zero_mass=list(CUR["recs"]),
               batch_keys=[k for k in batch.keys()] if done["n"] == 0 else None,
               window=_window_id(batch))
    with open(OUT, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    done["n"] += 1
    if done["n"] >= STEPS:
        os._exit(0)
    return {**parts, "seconds": row["seconds"], "finite": True}


def _window_id(batch):
    out = {}
    for key in ("episode", "episodes", "world", "t0", "start", "window_start", "t_start", "source", "name", "index"):
        if key in batch:
            v = batch[key]
            out[key] = v.tolist() if torch.is_tensor(v) else v
    return out


WmSTrainer.train_batch = train_batch
from wms.cli.__main__ import main
sys.exit(main(sys.argv[1:]))
