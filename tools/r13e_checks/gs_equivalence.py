"""CPU-only differential check against the actual GAPACC install_gate function.

Default: synthetic boundary/invalid-input cases, not empirical evidence.
Optional --raw-npz EXACT_PATH (repeatable): existing GAPACC raw trajectories,
using rec_* insertion geometry and that file's dp_full learned slot parameters.
No rollouts, checkpoint loads, data discovery, or output files are performed.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wms.model.merge_core import load_init  # noqa: E402
from wms.model.merge_pivot import FirstPassageNet  # noqa: E402

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "hetdata_truth", "vtype")


def checked_path(value):
    path = Path(value)
    for text in (str(path), str(path.absolute())):
        if any(word in text.lower() for word in FORBIDDEN):
            raise ValueError("restricted path")
        if any(964000 <= int(n) <= 964199 for n in re.findall(r"\d+", text)):
            raise ValueError("restricted evaluation seed")
    resolved = path.resolve()
    if str(resolved) != str(path.absolute()):
        # Recheck symlink targets before opening them.
        for word in FORBIDDEN:
            if word in str(resolved).lower():
                raise ValueError("restricted target")
        if any(964000 <= int(n) <= 964199 for n in re.findall(r"\d+", str(resolved))):
            raise ValueError("restricted target seed")
    return resolved


def module_at(path, name):
    path = checked_path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case(label, *, v=8., vf=10., vl=12., gf=100., gl=100.,
         has_f=True, has_l=True, eligible=True, dp=None):
    # Local slots are R,F,L; heterogeneous parameters make incorrect gathers visible.
    if dp is None:
        dp = [[30., 1.1, 2., 1., 2.], [30., .7, 1.5, 1., 1.],
              [30., 1.8, 3., 1., 5.]]
    return dict(label=label, v=v, vf=vf, vl=vl, gf=gf, gl=gl,
                has_f=has_f, has_l=has_l, eligible=eligible, dp=dp)


def tensors(cases, dtype):
    n = len(cases)
    v = torch.tensor([[c["v"], c["vf"], c["vl"]] for c in cases], dtype=dtype)
    g = {"v": v, "eligible": torch.zeros((n, 3), dtype=torch.bool)}
    g["eligible"][:, 0] = torch.tensor([c["eligible"] for c in cases])
    for side, slot in (("f", 1), ("l", 2)):
        g[f"has_{side}"] = torch.zeros((n, 3), dtype=torch.bool)
        g[f"has_{side}"][:, 0] = torch.tensor([c[f"has_{side}"] for c in cases])
        g[f"gap_{side}"] = torch.zeros_like(v)
        g[f"gap_{side}"][:, 0] = torch.tensor([c[f"g{side}"] for c in cases], dtype=dtype)
        g[f"v_{side}"] = torch.zeros_like(v)
        g[f"v_{side}"][:, 0] = v[:, slot]
        g[f"index_{side}"] = torch.full((n, 3), -1, dtype=torch.long)
        g[f"index_{side}"][:, 0] = slot
    dp = torch.tensor(np.asarray([c["dp"] for c in cases]), dtype=dtype)
    settings = {name: dp[..., i] for name, i in
                (("time_headway", 1), ("min_gap", 2), ("decel", 4))}
    return g, dp, settings


def gapacc_boundary(reference, net, g, dp, *, gate_only=False):
    # Execute GAPACC's real wrapper, with exactly the same underlying boundary.
    fp = SimpleNamespace(forward=(lambda geometry: torch.zeros_like(geometry["v"]))
                         if gate_only else net.boundary)
    model = SimpleNamespace(merge_pivot=fp, _gate_dp=dp, _gate_out={},
                            _apply_merge_pivot=lambda *a, **kw: None)
    reference.install_gate(model, use_T=True)
    return fp.forward(g)


def oracle(g, dp, *, add_s0):
    ok = torch.ones_like(g["v"], dtype=torch.bool)
    for side in ("f", "l"):
        idx = g[f"index_{side}"].clamp_min(0)
        neighbor = dp.gather(-2, idx[..., None].expand(*idx.shape, dp.shape[-1]))
        rear, front = (neighbor, dp) if side == "f" else (dp, neighbor)
        vb, vf = (g[f"v_{side}"], g["v"]) if side == "f" else (g["v"], g[f"v_{side}"])
        stopping = torch.relu(
            vb.square() / (2 * rear[..., 4]) - vf.square() / (2 * front[..., 4]))
        need = ((rear[..., 2] + vb * rear[..., 1] + stopping) if add_s0 else
                rear[..., 1] * vb + stopping)
        ok &= ~g[f"has_{side}"] | (g[f"gap_{side}"] >= need)
    return ok


def compare(reference, cases, dtype, net=None):
    g, dp, settings = tensors(cases, dtype)
    net = (FirstPassageNet() if net is None else copy.deepcopy(net)).to(dtype).eval()
    with torch.no_grad():
        ga = gapacc_boundary(reference, net, g, dp, gate_only=True)[:, 0] != -torch.inf
        r13 = net.secure_gap_margin(g, settings)[:, 0] >= 0
        base = net(g)[:, 0]
        ga_c = gapacc_boundary(reference, net, g, dp)[:, 0]
        r13_c = net(g, settings)[:, 0]
        # Neighbour gathers must be invariant to changing every slot's position.
        perm = torch.tensor([2, 0, 1])
        inverse = torch.argsort(perm)
        gp = {k: val[:, perm].clone() for k, val in g.items()}
        for side in ("f", "l"):
            idx = gp[f"index_{side}"]
            gp[f"index_{side}"] = torch.where(idx >= 0, inverse[idx.clamp_min(0)], idx)
        sp = {k: val[:, perm] for k, val in settings.items()}
        assert torch.equal(net(gp, sp)[:, 1], r13_c), "R13E slot-order dependence"
        assert torch.equal(gapacc_boundary(reference, net, gp, dp[:, perm])[:, 1], ga_c)
        # tau and the global willingness parameters do not scale the physical veto.
        for tau in (.01, 2.):
            net.tau_raw.fill_(tau + math.log(-math.expm1(-tau)))
            assert torch.equal(net.secure_gap_margin(g, settings)[:, 0] >= 0, r13)
        finite = torch.isfinite(dp).all(dim=(-1, -2)) & (dp[..., 4] > 0).all(-1)
        finite &= (g["v"] >= 0).all(-1) & torch.isfinite(g["v"]).all(-1)
        finite &= (dp[..., 1] >= 0).all(-1) & (dp[..., 2] > 0).all(-1)
        finite &= g["eligible"][:, 0]
        assert torch.equal(ga[finite], oracle(g, dp, add_s0=False)[:, 0][finite])
        assert not bool((r13 & ~ga & finite).any()), "unexpected non-conservative difference"
        # Count log-rounding differences against direct comparisons separately.
        direct = oracle(g, dp, add_s0=True)[:, 0]
        positive_gaps = torch.ones_like(finite)
        for side in ("f", "l"):
            gap = g[f"gap_{side}"][:, 0]
            positive_gaps &= ~g[f"has_{side}"][:, 0] | (torch.isfinite(gap) & (gap > 0))
        valid = finite & positive_gaps
    differences = torch.nonzero(ga != r13).flatten().tolist()
    rounding = torch.nonzero((r13 != direct) & valid).flatten().tolist()
    return dict(n=len(cases), gate_equal=int((ga == r13).sum()),
                gapacc_only=int((ga & ~r13).sum()), r13e_only=int((r13 & ~ga).sum()),
                both_accept=int((ga & r13).sum()), both_reject=int((~ga & ~r13).sum()),
                boundary_equal=int((ga_c == r13_c).sum()),
                boundary_different=int((ga_c != r13_c).sum()),
                ungated_boundary_neg_inf=int(torch.isneginf(base).sum()),
                log_vs_direct_differences=int(((r13 != direct) & valid).sum()),
                log_vs_direct_examples=[dict(label=cases[i]["label"],
                                             direct=bool(direct[i]), r13e=bool(r13[i]))
                                        for i in rounding[:8]],
                examples=[dict(label=cases[i]["label"], gapacc=bool(ga[i]), r13e=bool(r13[i]),
                               gap_f=cases[i]["gf"], gap_l=cases[i]["gl"])
                          for i in differences[:8]])


def edge_cases(dtype):
    cases = []
    for side in ("f", "l"):
        base = case("threshold", has_f=side == "f", has_l=side == "l")
        g, dp, _ = tensors([base], dtype)
        rear, front = (1, 0) if side == "f" else (0, 2)
        vb, vf = (g["v_f"][0, 0], g["v"][0, 0]) if side == "f" else (
            g["v"][0, 0], g["v_l"][0, 0])
        need = dp[0, rear, 1] * vb + torch.relu(
            vb.square() / (2 * dp[0, rear, 4]) - vf.square() / (2 * dp[0, front, 4]))
        for name, threshold in (("gapacc", need), ("r13e", dp[0, rear, 2] + need)):
            for direction in (-1, 0, 1):
                gap = threshold if not direction else torch.nextafter(
                    threshold, threshold.new_tensor(math.copysign(math.inf, direction)))
                c = dict(base, label=f"{side}:{name}:ulp{direction:+}")
                c[f"g{side}"] = float(gap)
                cases.append(c)
    cases.extend([
        case("rest-small-clearance", v=0., vf=0., vl=0., gf=.5, gl=.5),
        case("rest-zero-clearance", v=0., vf=0., vl=0., gf=0., gl=0.),
        case("rest-exact-s0", v=0., vf=0., vl=0., gf=1.5, gl=2.),
        case("no-neighbors", has_f=False, has_l=False, vf=math.nan, vl=math.nan),
        case("absent-follower-padding", has_f=False, vf=math.nan, gf=math.nan),
        case("ineligible", eligible=False),
        case("overlap", gf=-1.), case("touching", gf=0.),
        case("slower-follower-unequal-b", v=10., vf=8., gf=20.),
        case("fast-follower", vf=30., gf=20.),
        case("leader-veto", gf=1000., gl=1.),
        case("negative-speed", vf=-1.),
        case("nan-gap", gf=math.nan), case("infinite-gap", gf=math.inf),
    ])
    for slot, name in ((1, "rear"), (0, "front")):
        for col, field in ((1, "T"), (2, "s0"), (4, "b")):
            for value in (0., -1., math.nan, math.inf):
                c = case(f"invalid-{name}-{field}-{value}", has_l=False)
                c["dp"][slot][col] = value
                cases.append(c)
    # Exact counterexample: identical geometry and driver parameters, finite base.
    c = case("minimal-counterexample", v=2., vf=2., gf=2., has_l=False)
    c["dp"][0][4] = c["dp"][1][4] = 2.
    c["dp"][1][1] = 1.
    c["dp"][1][2] = 1.5
    cases.append(c)
    return cases


def random_cases(n=4096):
    rng = np.random.default_rng(130013)
    cases = []
    for i in range(n):
        dp = np.tile([30., 1., 2., 1., 2.], (3, 1))
        dp[:, 1] = rng.uniform(.1, 3., 3)
        dp[:, 2] = rng.uniform(.1, 5., 3)
        dp[:, 4] = rng.uniform(.3, 6., 3)
        v, vf, vl = rng.uniform(0., 35., 3)
        rf = dp[1, 1]*vf + max(0., vf*vf/(2*dp[1, 4]) - v*v/(2*dp[0, 4]))
        rl = dp[0, 1]*v + max(0., v*v/(2*dp[0, 4]) - vl*vl/(2*dp[2, 4]))
        cases.append(case(f"random-{i}", v=v, vf=vf, vl=vl, dp=dp,
                          gf=max(.001, rf + rng.uniform(-2., 7.)),
                          gl=max(.001, rl + rng.uniform(-2., 7.)),
                          has_f=bool(rng.integers(0, 2)), has_l=bool(rng.integers(0, 2))))
    return cases


def index_shape_edges(reference, dtype):
    net = FirstPassageNet().to(dtype)
    g, dp, settings = tensors([case("index", has_l=False)], dtype)
    outcomes = []
    for index in (-1, 3):
        g["index_f"][0, 0] = index
        actual = bool(net.secure_gap_margin(g, settings)[0, 0] >= 0)
        assert not actual
        try:
            ref = bool(gapacc_boundary(reference, net, g, dp, gate_only=True)[0, 0] != -torch.inf)
        except RuntimeError:
            ref = "RuntimeError: index out of bounds"
        outcomes.append(dict(index=index, gapacc=ref, r13e=actual))
    bad = {**settings, "decel": settings["decel"][:, :2]}
    try:
        net.secure_gap_margin(g, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch was accepted")
    empty_g = {k: value[:, :0] for k, value in g.items()}
    empty_s = {k: value[:, :0] for k, value in settings.items()}
    assert net(empty_g, empty_s).shape == (1, 0)
    return dict(indices=outcomes, bad_settings_shape="ValueError", empty_slots="passed")


def recorded_cases(path, extractor, which):
    path = checked_path(path)
    with np.load(path, allow_pickle=False) as z:
        prefix = "rec_" if which == "rec" else ""
        dp = z["dp_full"]
        pos, lane, mask, v = (z[prefix + key] for key in ("pos", "lane", "mask", "v"))
        if dp.ndim != 2 or dp.shape[0] != v.shape[1] or dp.shape[1] < 5:
            raise ValueError("dp_full must contain the actual learned parameters for every slot")
        out = []
        for event in extractor.events(pos, lane, mask, v, z["is_cav"]):
            r, f, leader = (int(event[key]) for key in ("r", "f", "l"))
            local_dp = dp[[r, max(f, 0), max(leader, 0)]].copy()
            c = case(f"{path.name}:{which}:t{event['tm']}:r{r}",
                     v=float(event["vR"]), vf=float(event["vF"]), vl=float(event["vL"]),
                     gf=float(event["gf"]), gl=float(event["gl"]),
                     has_f=f >= 0, has_l=leader >= 0, dp=local_dp)
            c.update(hdv=bool(event["hdvR"]), window=40 <= event["tm"] <= 216)
            out.append(c)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gapacc-script", default="E:/agent-work/c4-gapacc/scripts/factworker.py")
    parser.add_argument("--raw-npz", action="append", default=[])
    parser.add_argument("--which", choices=("rec", "wm"), default="rec")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.name == "posix":
        os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    reference = module_at(args.gapacc_script, "gapacc_reference")
    report = {"reference": str(checked_path(args.gapacc_script)),
              "learned_veto_scale": "none: hard requirement has implicit fixed multiplier 1",
              "tau_initial_formula": "alpha_log_sigma * sqrt(3) / pi; softplus(raw)",
              "tau_initial_expected": load_init()["alpha_log_sigma"] * math.sqrt(3) / math.pi,
              "tau_initial_actual": float(FirstPassageNet().params()["tau"].detach()),
              "synthetic": {}}
    random = random_cases()
    for dtype in (torch.float32, torch.float64):
        edges = edge_cases(dtype)
        result = compare(reference, edges, dtype)
        assert result["gapacc_only"] > 0
        g, dp, settings = tensors([edges[-1]], dtype)
        net = FirstPassageNet().to(dtype)
        with torch.no_grad():
            assert torch.isfinite(gapacc_boundary(reference, net, g, dp)[0, 0])
            assert torch.isneginf(net(g, settings)[0, 0])
        report["synthetic"][str(dtype)] = {
            "edge_cases": result, "random_heterogeneous_cases": compare(reference, random, dtype),
            "index_shape_edges": index_shape_edges(reference, dtype)}
    all_recorded = []
    if args.raw_npz:
        extractor = module_at(Path(args.gapacc_script).with_name("ins.py"), "gapacc_ins")
        for path in args.raw_npz:
            all_recorded.extend(recorded_cases(path, extractor, args.which))
    report["recorded_events"] = len(all_recorded)
    if all_recorded:
        report["recorded"] = {}
        for label, cases in (("all", all_recorded),
                             ("hdv", [c for c in all_recorded if c["hdv"]]),
                             ("hdv_window40_216", [c for c in all_recorded if c["hdv"] and c["window"]])):
            report["recorded"][label] = {str(dt): compare(reference, cases, dt)
                                         for dt in (torch.float32, torch.float64)} if cases else {}
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
