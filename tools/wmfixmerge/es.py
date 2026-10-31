"""S-WMFIX-ES: derivative-free calibration of existing WM2 physical scalars.

Owner 12:50 (CPU route): the analytic gradient of the paired TRAIN objective is
unreliable (dense hard-event breakpoints, WMFIX_FEAS_BLOCKED), so this tool
optimises with antithetic ES over a small set of EXISTING physical
merge/admission/discharge parameters of WM2 (option (b); the 1,473-dim head of
option (a) needs popsize >= 16 which exceeds the measured CPU budget).  Values
are OPTIMISED, never hand-set: the incumbent vector is the live resolved WM2
config, and ES moves it inside generous a-priori physical bounds.

Fitness: the SAME paired training objective (paired road/upstream/crossing
columns + discharge/road absolute anchors, equal-world weighting; source and
paired J reported separately, never optimised) on deterministic CRN rollouts
of the 12-group TRAIN corpus, fitness horizon/groups capped by the measured
budget.  Everything else stays frozen-bitwise (asserted per candidate cell).
Cells are sticky to one host for the whole run (cross-build numerics).

Modes: ``--mode sweep`` (single-process worker; one manifest, threads=1) and
``--mode drive`` (generation loop: sample, dispatch sweepers on gpuA and cpuQ,
aggregate, ES update, immutable receipts; fail-closed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time

import numpy as np

from tools.wmfixmerge.data import WM2_SHA, digest, install_guard, safe

#: Existing physical scalars actually consumed by the entry-path rollout under
#: the resolved WM2 config (lane_learnable + courtesy + secure_gap_veto +
#: emergency driver_b + sumo_like_v2 learned admission; see WMFIX_ES.md).
PARAMS = (
    dict(name="accel_min_mps2", where="physics", lo=-15.0, hi=-3.0),
    dict(name="accel_max_mps2", where="physics", lo=1.0, hi=6.0),
    dict(name="lane_physical_margin_m", where="physics", lo=0.0, hi=3.0),
    dict(name="mainline_courtesy_zone_m", where="physics", lo=0.0, hi=600.0),
    dict(name="free_road_gap_m", where="physics", lo=50.0, hi=500.0),
    dict(name="speed_limit_mps", where="physics", lo=25.0, hi=40.0),
    dict(name="sumo_min_speed_mps", where="entry", lo=0.0, hi=5.0),
)

J, UPSTREAM, SOURCE, DISCHARGE, ROAD = range(5)
GRADIENT_COLUMNS = (UPSTREAM, DISCHARGE, ROAD)
ANCHOR_COLUMNS = (DISCHARGE, ROAD)

#: PhysicsConfig/EntryConfig dataclass defaults of the optimised scalars
#: (src/wms/physics/config.py, src/wms/model/entries.py); every worker
#: verifies them against the live resolved model, fail-closed.
DEFAULTS = dict(accel_min_mps2=-9.0, accel_max_mps2=3.0, lane_physical_margin_m=0.0,
                mainline_courtesy_zone_m=300.0, free_road_gap_m=250.0,
                speed_limit_mps=31.29, sumo_min_speed_mps=0.0)


def to_unit(values):
    return np.clip([(v - p["lo"]) / (p["hi"] - p["lo"]) for v, p in zip(values, PARAMS)], 0., 1.)


def from_unit(x):
    return [float(p["lo"] + float(np.clip(x[i], 0., 1.)) * (p["hi"] - p["lo"]))
            for i, p in enumerate(PARAMS)]


def params_doc(values):
    return {p["name"]: float(v) for p, v in zip(PARAMS, values)}


def es_update(x, eps, losses, lr, sigma):
    """Antithetic rank-shaped ES step in unit space (deterministic losses).

    ``eps``: [P/2, d] standard-normal draws; ``losses``: [P] with candidate
    order (pair k plus at 2k, minus at 2k+1).  Returns the new centre.
    """
    losses = np.asarray(losses, float)
    n = losses.size
    order = losses.argsort()
    ranks = np.empty(n)
    ranks[order] = 0.5 - np.arange(n) / max(n - 1, 1)  # best (lowest) loss -> +0.5
    grad = np.zeros_like(x, dtype=float)
    for k, draw in enumerate(eps):
        for sign, u in ((+1., ranks[2 * k]), (-1., ranks[2 * k + 1])):
            grad += u * sign * draw
    grad /= (n / 2.) * sigma
    return np.clip(np.asarray(x, float) + lr * grad, 0., 1.)


def fitness_from_cells(cells, scales, anchor_scales):
    """Equal-world aggregation of the SAME paired objective over panel cells.

    Each cell holds predicted/recorded [rows, components] of one (group, arm,
    horizon) microbatch (native reference row 0, arm row 1).  Returns
    (fitness, reports); paired J and source contrasts are reports only.
    """
    scales = np.asarray(scales, float)
    anchor = np.asarray(anchor_scales, float)
    pair_terms, anchor_terms, per_group = [], [], {}
    for cell in cells:
        predicted = np.asarray(cell["predicted"], float)
        recorded = np.asarray(cell["recorded"], float)
        error = (predicted[1] - predicted[0]) - (recorded[1] - recorded[0])
        pair = float(np.mean([np.square(error[c] / scales[c]) for c in GRADIENT_COLUMNS]))
        absolute = float(np.mean(np.square((predicted - recorded)[:, list(ANCHOR_COLUMNS)]
                                           / anchor[list(ANCHOR_COLUMNS)])))
        pair_terms.append(pair)
        anchor_terms.append(absolute)
        per_group.setdefault((cell["group"], cell["horizon"]), []).append(
            dict(pair=pair, absolute=absolute,
                 j=abs(float(error[J])), source=abs(float(error[SOURCE]))))
    fitness = float(np.mean([np.mean([r["pair"] + r["absolute"] for r in v])
                             for v in per_group.values()]))
    reports = dict(cells=len(cells),
                   paired_J_MAE=float(np.mean([np.mean([r["j"] for r in v]) for v in per_group.values()])),
                   source_contrast_MAE=float(np.mean([np.mean([r["source"] for r in v])
                                                      for v in per_group.values()])),
                   per_group={f"g{g}_h{h}": dict(paired_J_MAE=float(np.mean([r["j"] for r in v])),
                                                 source_contrast_MAE=float(np.mean([r["source"] for r in v])))
                              for (g, h), v in sorted(per_group.items())})
    return fitness, reports


def state_hash(model):
    import torch
    sha = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("merge_response."):
            continue
        sha.update(name.encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def sweep(argv=None):
    parser = argparse.ArgumentParser(description="worker sweep over cells")
    for name in ("prepared", "config", "checkpoint", "controllers", "manifest", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prepared-wmdata", default=None,
                        help="prepared doc for corpus 'wm' cells (required if any present)")
    parser.add_argument("--wmdata-controllers")
    args = parser.parse_args(argv)
    install_guard()
    import torch
    torch.set_num_threads(1)
    import tools.wmfixmerge.runtime as runtime_mod
    from tools.wmfixmerge.runtime import Runtime
    manifest = json.loads(safe(args.manifest).read_text())
    corpora = {cell["corpus"] for cell in manifest["cells"]}
    if "wm" in corpora and not args.prepared_wmdata:
        raise ValueError("wm cells need --prepared-wmdata")
    runtimes, bases = {}, {}

    def runtime_for(corpus):
        if corpus not in runtimes:
            prepared = args.prepared if corpus == "fit" else args.prepared_wmdata
            # wm corpus: native + ctrl0 only (registered paired contrast); the
            # designed-controller arms never enter this rollout source.
            rt = Runtime(prepared, args.config, args.checkpoint, args.controllers, "cpu",
                         wmdata_controllers=args.wmdata_controllers)
            live = {p["name"]: float(getattr(rt.model.physics, p["name"]))
                    for p in PARAMS if p["where"] == "physics"}
            for name, value in live.items():
                if abs(value - manifest["incumbents"][name]) > 1e-9:
                    raise ValueError(f"live {corpus} physics incumbent differs for {name}: "
                                     f"{value} vs {manifest['incumbents'][name]}")
            bases[corpus] = state_hash(rt.model)
            if len(set(bases.values())) != 1:
                raise ValueError("frozen WM2 tensors differ across corpora")
            runtimes[corpus] = rt
        return runtimes[corpus]

    rt0 = runtime_for(manifest["cells"][0]["corpus"])
    base = bases[manifest["cells"][0]["corpus"]]
    live = {p["name"]: float(getattr(rt0.model.physics, p["name"]))
            for p in PARAMS if p["where"] == "physics"}
    out = safe(args.out)
    env_path = out / f"ENV_{manifest['sweep_id']}.json"
    if not env_path.is_file():  # idempotent across driver relaunches
        with env_path.open("x") as stream:
            json.dump(dict(host=socket.gethostname(), pid=os.getpid(), torch=torch.__version__,
                           corpora=sorted(corpora), base_state_sha256=base,
                           physics_incumbents=live), stream)
    original_replace = runtime_mod.replace
    for cell in manifest["cells"]:
        path = out / cell_name(manifest["gen"], cell["cand"], cell["corpus"],
                               cell["group"], cell["arm"], cell["horizon"])
        if path.is_file():
            continue
        rt = runtime_for(cell["corpus"])
        values = cell["params"]
        entry = {p["name"]: values[i] for i, p in enumerate(PARAMS) if p["where"] == "entry"}
        for i, p in enumerate(PARAMS):
            if p["where"] == "physics":
                setattr(rt.model.physics, p["name"], float(values[i]))
        seen_incumbents = {}

        def patched_replace(cfg, _entry=entry, **kw):
            cfg = original_replace(cfg, **kw)
            if "conserve_queue" in kw and hasattr(cfg, "sumo_min_speed_mps"):
                if "sumo_min_speed_mps" not in seen_incumbents:
                    seen_incumbents["sumo_min_speed_mps"] = float(cfg.sumo_min_speed_mps)
                cfg = original_replace(cfg, **_entry)
            return cfg

        began = time.monotonic()
        runtime_mod.replace = patched_replace
        try:
            with torch.no_grad():
                predicted, recorded, _, receipt = rt.rollout(
                    cell["group"], cell["horizon"], arm_index=cell["arm"])
        finally:
            runtime_mod.replace = original_replace
        if receipt["pair_key"] != cell["pair_key"]:
            raise ValueError(f"{cell['corpus']} group order shifted: {receipt['pair_key']}"
                             f" != {cell['pair_key']}")
        if len(receipt["arms"]) != 2 or receipt["arms"][0] != "native":
            raise ValueError("unexpected microbatch rows: " + repr(receipt["arms"]))
        if cell.get("arm_name") not in (None, "registered_single") and receipt["arms"][1] != cell["arm_name"]:
            raise ValueError(f"arm order differs: {receipt['arms'][1]} != {cell['arm_name']}")
        if state_hash(rt.model) != bases[cell["corpus"]]:
            raise ValueError("frozen WM2 tensors changed under candidate " + str(cell["cand"]))
        if seen_incumbents and abs(seen_incumbents["sumo_min_speed_mps"]
                                   - manifest["incumbents"]["sumo_min_speed_mps"]) > 1e-9:
            raise ValueError("live entry incumbent differs: " + repr(seen_incumbents))
        row = dict(gen=manifest["gen"], cand=cell["cand"], corpus=cell["corpus"],
                   group=cell.get("group_label", str(cell["group"])), arm=cell["arm"],
                   horizon=cell["horizon"], seconds=round(time.monotonic() - began, 1),
                   host=socket.gethostname(), params_sha256=cell.get("params_sha256"),
                   predicted=predicted.detach().cpu().tolist(),
                   recorded=recorded.detach().cpu().tolist(), rollout=receipt)
        with path.open("x") as stream:
            json.dump(row, stream, allow_nan=False)
        print(json.dumps(dict(cell=path.name, seconds=row["seconds"])), flush=True)
    with (out / f"SWEEP_DONE_{manifest['sweep_id']}").open("x") as stream:
        stream.write(str(time.time()))
    return 0


def pull(cells_dir_remote, cells_dir_local, host):
    """Pull CELL/ENV files from a remote sweep dir (per-file scp; resume-safe)."""
    listing = subprocess.run(["ssh", host, "ls " + cells_dir_remote], capture_output=True,
                             text=True, check=True).stdout.split()
    brought = 0
    for name in listing:
        if not (name.startswith("CELL_") or name.startswith("ENV_")
                or name.startswith("SWEEP_DONE_")):
            continue
        local = safe(cells_dir_local / name)
        if local.is_file():
            continue
        subprocess.run(["scp", "-q", f"{host}:{cells_dir_remote}/{name}", str(local)],
                       check=True)
        brought += 1
    return brought


def incumbents_from_resolved(resolved_path):
    resolved = json.loads(safe(resolved_path).read_text())
    physics = resolved.get("physics", {})
    values = []
    for p in PARAMS:
        if p["name"] in physics:
            values.append(float(physics[p["name"]]))
        else:
            values.append(float(DEFAULTS[p["name"]]))
    return values


def cell_name(gen, cand, corpus, group, arm, horizon):
    return f"CELL_g{gen:03d}_c{cand:03d}_{corpus}_grp{group}_arm{arm}_h{horizon}.json"


def drive(argv=None):
    parser = argparse.ArgumentParser(description="antithetic-ES generation driver")
    for name in ("prepared", "config", "checkpoint", "controllers", "resolved-config", "out"):
        parser.add_argument("--" + name, required=True)
    for name in ("prepared-wmdata", "cpuq-prepared-wmdata", "registry"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=140014)
    parser.add_argument("--fitness-horizon", type=int, default=300)
    parser.add_argument("--gpuA-workers", type=int, default=18)
    parser.add_argument("--cpuq-workers", type=int, default=28)
    parser.add_argument("--gpuA-fit-groups", default="0-7")
    parser.add_argument("--cpuq-fit-groups", default="8-11")
    parser.add_argument("--cpuq-root", default="/home/ubuntu/wmfix_es")
    parser.add_argument("--cpuq-prepared", default="/home/ubuntu/wmfix_es/TRAIN_PREPARED.json")
    parser.add_argument("--max-seconds", type=float, default=12600.)
    parser.add_argument("--stop-patience", type=int, default=3)
    parser.add_argument("--poll", type=float, default=30.)
    parser.add_argument("--gen-timeout", type=float, default=5400.)
    args = parser.parse_args(argv)
    install_guard()
    out = safe(args.out)
    if not out.is_dir() or any(out.iterdir()):
        raise ValueError("out must be an existing empty directory")

    def range_spec(spec):
        a, b = (int(v) for v in spec.split("-"))
        return set(range(a, b + 1))

    gpuA_fit = range_spec(args.gpuA_fit_groups)
    cpuq_fit = range_spec(args.cpuq_fit_groups)
    if gpuA_fit & cpuq_fit or gpuA_fit | cpuq_fit != set(range(12)):
        raise ValueError("FIT group host pinning must partition 0-11")

    from tools.wmfixmerge.prepare_combined import check_registry
    registry = check_registry(args.registry, args.prepared)
    doc = json.loads(safe(args.prepared).read_text())
    scales, anchor_scales = doc["scales"], doc["anchor_scales"]
    groups = registry["fitness_groups"]
    if {fg["group"] for fg in groups if fg["corpus"] == "fit"} != set(range(12)):
        raise ValueError("registry must cover all 12 FIT groups")

    def host_of(fg):
        if fg["corpus"] == "wm":
            return "cpuQ"
        return "gpuA" if fg["group"] in gpuA_fit else "cpuQ"

    incumbent_values = incumbents_from_resolved(args.resolved_config)
    x = to_unit(incumbent_values)
    began = time.monotonic()
    with (out / "RUN.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started=time.time(), host=socket.gethostname(),
                        schema="wmfix_es_run/2", popsize=args.popsize, sigma=args.sigma,
                        lr=args.lr, seed=args.seed,
                        fitness_groups=[dict(g, host=host_of(g)) for g in groups],
                        fitness_horizon=args.fitness_horizon, max_seconds=args.max_seconds,
                        stop_patience=args.stop_patience,
                        prepared_sha256=digest(args.prepared),
                        registry_sha256=digest(args.registry),
                        wmdata_prepared_sha256=registry["parts"]["wm"]["sha256"],
                        resolved_config_sha256=digest(args.resolved_config),
                        base_sha256=WM2_SHA, implementation_params=[p["name"] for p in PARAMS],
                        incumbents=params_doc(incumbent_values)), stream, indent=2)
    best = None
    stale = 0
    gen = 0
    eps = None
    try:
        for gen in range(args.gens + 1):
            if time.monotonic() - began > args.max_seconds:
                break
            gen_began = time.monotonic()
            if gen == 0:
                cand_params = {0: incumbent_values}
            else:
                rng = np.random.default_rng(args.seed + gen)
                eps = rng.standard_normal((args.popsize // 2, len(PARAMS)))
                cand_params = {}
                for k, draw in enumerate(eps):
                    for sign, cand in ((+1, 2 * k), (-1, 2 * k + 1)):
                        cand_params[cand] = from_unit(x + sign * args.sigma * draw)
                with (out / f"EPS_{gen:03d}.json").open("x") as stream:
                    json.dump(eps.tolist(), stream)
            with (out / f"CANDS_{gen:03d}.json").open("x") as stream:
                json.dump({str(c): params_doc(v) for c, v in cand_params.items()}, stream,
                          indent=2, allow_nan=False)
            jobs = []
            for host, workers in (("gpuA", args.gpuA_workers), ("cpuQ", args.cpuq_workers)):
                todo = [(c, fg) for c in cand_params for fg in groups
                        if host_of(fg) == host
                        and not (out / cell_name(gen, c, fg["corpus"], fg["group"],
                                                fg["arm"], args.fitness_horizon)).is_file()]
                chunks = [todo[i::workers] for i in range(workers)]
                for w, chunk in enumerate(chunks):
                    if not chunk:
                        continue
                    sweep_id = f"g{gen:03d}_{host}_w{w}"
                    cells = [dict(cand=c, corpus=fg["corpus"], group=fg["group"],
                                  group_label=f"{fg['corpus']}:{fg['group']}",
                                  arm=fg["arm"], arm_name=fg.get("arm_name"),
                                  horizon=args.fitness_horizon, pair_key=fg["pair_key"],
                                  params=cand_params[c],
                                  params_sha256=hashlib.sha256(
                                      json.dumps(cand_params[c]).encode()).hexdigest())
                             for c, fg in chunk]
                    manifest = dict(gen=gen, sweep_id=sweep_id, cells=cells,
                                    incumbents={p["name"]: incumbent_values[i]
                                                for i, p in enumerate(PARAMS)})
                    with (out / f"MANIFEST_{sweep_id}.json").open("x") as stream:
                        json.dump(manifest, stream, allow_nan=False)
                    jobs.append((host, sweep_id))
            expected = [(c, fg) for c in cand_params for fg in groups]
            children = []
            for host, sweep_id in jobs:
                if host == "gpuA":
                    cmd = ["/root/miniconda3/bin/python", "-m", "tools.wmfixmerge.es",
                           "--mode", "sweep", "--prepared", args.prepared, "--config", args.config,
                           "--checkpoint", args.checkpoint, "--controllers", args.controllers,
                           "--manifest", str(out / f"MANIFEST_{sweep_id}.json"),
                           "--out", str(out)]
                    env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1")
                    log_path = out / f"SWEEP_{sweep_id}.log"
                    children.append((sweep_id, subprocess.Popen(
                        cmd, stdout=log_path.open("xb"), stderr=subprocess.STDOUT,
                        env=env, cwd=str(safe(Path_root())))))
                else:
                    remote = (f"cd {args.cpuq_root}/wt && "
                              f"nohup sudo env PYTHONPATH=src:. OMP_NUM_THREADS=1 "
                              f"MKL_NUM_THREADS=1 /home/ubuntu/miniconda3/bin/python "
                              f"-m tools.wmfixmerge.es --mode sweep "
                              f"--prepared {args.cpuq_prepared} "
                              f"--prepared-wmdata {args.cpuq_prepared_wmdata} "
                              f"--config {args.config} --checkpoint {args.checkpoint} "
                              f"--controllers {args.controllers} "
                              f"--manifest {args.cpuq_root}/es_run_combined/MANIFEST_{sweep_id}.json "
                              f"--out {args.cpuq_root}/es_run_combined > {args.cpuq_root}/es_run_combined/"
                              f"SWEEP_{sweep_id}.log 2>&1 < /dev/null & echo launched")
                    subprocess.run(["scp", "-q", str(out / f"MANIFEST_{sweep_id}.json"),
                                    f"cpuQ:{args.cpuq_root}/es_run_combined/MANIFEST_{sweep_id}.json"],
                                   check=True)
                    # The remote worker may hold the ssh session open after
                    # launch; the poll loop (ENV/CELL files) is the liveness
                    # source of truth, so a lingering ssh here is not an error.
                    try:
                        subprocess.run(["ssh", "-n", "cpuQ", remote], check=True,
                                       timeout=30, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                    except subprocess.TimeoutExpired:
                        pass
                    children.append((sweep_id, None))
            while True:
                time.sleep(args.poll)
                pull(f"{args.cpuq_root}/es_run_combined", out, "cpuQ")
                present = sum((out / cell_name(gen, c, fg["corpus"], fg["group"],
                                               fg["arm"], args.fitness_horizon)).is_file()
                              for c, fg in expected)
                if present == len(expected):
                    break
                if time.monotonic() - gen_began > args.gen_timeout:
                    raise TimeoutError(f"generation {gen} timeout ({present}/{len(expected)})")
                for sweep_id, proc in children:
                    if proc is not None and proc.poll() is not None and proc.returncode != 0:
                        raise RuntimeError(f"gpuA sweeper {sweep_id} rc={proc.returncode}")
                print(json.dumps(dict(gen=gen, cells=f"{present}/{len(expected)}",
                                      elapsed=round(time.monotonic() - gen_began, 1))), flush=True)
            results = {}
            for cand, params in cand_params.items():
                cells = [json.loads((out / cell_name(gen, cand, fg["corpus"], fg["group"],
                                                     fg["arm"], args.fitness_horizon)).read_text())
                         for fg in groups]
                fitness, reports = fitness_from_cells(cells, scales, anchor_scales)
                results[cand] = dict(fitness=fitness, params=params_doc(params), **reports)
            receipt = dict(gen=gen, seconds=round(time.monotonic() - gen_began, 1),
                           sigma=args.sigma, lr=args.lr, x_unit=[float(v) for v in x],
                           candidates={str(c): r for c, r in results.items()})
            gen_best = min(results.items(), key=lambda kv: kv[1]["fitness"])
            improved = best is None or gen_best[1]["fitness"] < best["fitness"]
            if improved:
                best = dict(gen=gen, cand=int(gen_best[0]), **gen_best[1])
                stale = 0
            else:
                stale += 1
            receipt.update(best_overall=best, improved=bool(improved), stale_generations=stale)
            with (out / f"GEN_{gen:03d}.json").open("x") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
            print(json.dumps(dict(gen=gen, best_fitness=gen_best[1]["fitness"],
                                  improved=improved, stale=stale,
                                  seconds=receipt["seconds"])), flush=True)
            if gen == 0:
                continue
            order = sorted(results)
            x = es_update(x, eps, [results[c]["fitness"] for c in order], args.lr, args.sigma)
            if stale >= args.stop_patience:
                break
    except Exception as exc:  # fail-closed: keep every artifact + blocker receipt
        import traceback
        with (out / "ES_ABORT.json").open("x") as stream:
            json.dump(dict(blocker=f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc(), completed_generation=gen),
                      stream, indent=2, allow_nan=False)
        raise
    with (out / "ES_FINAL.json").open("x") as stream:
        json.dump(dict(best=best, stopped_after_gen=gen,
                       elapsed_s=round(time.monotonic() - began, 1),
                       incumbents=params_doc(incumbent_values)), stream, indent=2,
                  allow_nan=False)
    return 0


def Path_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sweep", "drive"), required=True)
    args, rest = parser.parse_known_args(argv)
    if args.mode == "sweep":
        return sweep(rest)
    return drive(rest)


if __name__ == "__main__":
    raise SystemExit(main())
