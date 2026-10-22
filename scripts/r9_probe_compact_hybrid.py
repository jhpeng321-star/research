"""LOCAL DIAGNOSTIC PROBE (not for gpuR): monkeypatch wms.model.merge_pivot.hybrid_log_masses so that columns that are
-inf in EVERY row are dropped before the quadrature and re-expanded after (events -inf, survival carried forward,
survival 0 before the first finite column). Vetoed steps have zero hazard, so this is exact (self-test: max abs diff
~1e-15, identical gradients). It turns the O(T^2) quadrature memory of long F2 pre-window prefixes into O(T_finite^2).

    python scripts/r9_probe_compact_hybrid.py --selftest
    python scripts/r9_probe_compact_hybrid.py train --config ... (any wms CLI args; set R9_PROBE_LOG=path for shapes)
"""
import sys, time
import torch
import wms.model.merge_pivot as mp
import wms.train.losses.merge_pivot as lp
orig = mp.hybrid_log_masses
import os
LOG = os.environ.get("R9_PROBE_LOG", os.devnull)

def compact(c, tau, w, order=32):
    if c.ndim < 1 or c.shape[-1] == 0:
        return orig(c, tau, w, order)
    keep = (c != -torch.inf).any(dim=tuple(range(c.ndim - 1))) if c.ndim > 1 else (c != -torch.inf)
    if bool(keep.all()) or not bool(keep.any()):
        return orig(c, tau, w, order)
    with open(LOG, "a") as f:
        rows = c.reshape(-1, c.shape[-1])
        per_row = (rows != -torch.inf).sum(-1).tolist()
        f.write(f"shape={tuple(c.shape)} union_finite_cols={int(keep.sum())} per_row_finite={per_row[:8]}\n")
    ek, sk = orig(c[..., keep], tau, w, order)
    idx = keep.long().cumsum(0) - 1
    e = torch.full(c.shape, -torch.inf, dtype=ek.dtype, device=ek.device)
    e[..., keep] = ek
    s = torch.where(idx >= 0, sk[..., idx.clamp(min=0)], torch.zeros_like(sk[..., :1]).expand(*sk.shape[:-1], c.shape[-1]))
    return e, s

mp.hybrid_log_masses = compact
lp.hybrid_log_masses = compact
if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
    torch.manual_seed(1)
    c = torch.randn(3, 40, dtype=torch.float64) * 0.5
    c[:, :12] = -torch.inf; c[:, 20:23] = -torch.inf; c[1, 30:] = -torch.inf; c[2, 35] = torch.inf
    tau = torch.tensor(0.5, requires_grad=True); w = torch.tensor(1.0, requires_grad=True)
    e0, s0 = orig(c, tau, w, 16); e1, s1 = compact(c, tau, w, 16)
    print("events max|diff|", float(((e0 - e1).abs()[torch.isfinite(e0) & torch.isfinite(e1)]).max()),
          "same -inf pattern", bool((torch.isinf(e0) == torch.isinf(e1)).all()),
          "surv max|diff|", float((s0 - s1).abs()[torch.isfinite(s0) & torch.isfinite(s1)].max()),
          "surv inf match", bool((torch.isinf(s0) == torch.isinf(s1)).all()))
    g0 = torch.autograd.grad((s0[:, -1][torch.isfinite(s0[:, -1])]).sum(), [tau, w], retain_graph=True)
    g1 = torch.autograd.grad((s1[:, -1][torch.isfinite(s1[:, -1])]).sum(), [tau, w])
    print("grad diff", [float((a - b).abs()) for a, b in zip(g0, g1)])
    sys.exit(0)
from wms.cli.__main__ import main
sys.exit(main(sys.argv[1:]))
