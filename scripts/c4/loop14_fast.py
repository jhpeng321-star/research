"""Default-OFF MAIN14 scheduling extensions. No process is armed by importing this module."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

FLAGS = ("speed2_policy", "two_box", "overlap_policy", "spot_resume", "cpu_checks", "repoint", "base_three")


def validate(cfg, refused):
    f = cfg["fastloop"]
    for flag in FLAGS:
        if flag in f and not isinstance(f[flag], bool):
            raise refused(f"fastloop.{flag} must be boolean")
    if f.get("two_box"):
        if f.get("boxes") != ["cpuP", "cpuQ"]:
            raise refused("two_box requires ordered boxes cpuP,cpuQ")
        if not 1 <= int(f.get("workers_per_box", 0)) <= 14:
            raise refused("per-box workers must be 1..14")
        if any(box not in cfg["hosts"] for box in f["boxes"]):
            raise refused("missing collector host")
        if cfg["convert"].get("val_frac", 0.) != 0.:
            raise refused("two_box currently requires registered val_frac=0 (no per-shard split drift)")
    if f.get("spot_resume") and not f.get("two_box"):
        raise refused("spot_resume requires two_box")
    if f.get("overlap_policy") and not f.get("two_box"):
        raise refused("overlap requires CPU artifact pipeline")
    if f.get("cpu_checks") and not f.get("check_roots"):
        raise refused("cpu_checks needs explicit TRAIN check_roots on CPU boxes")
    if f.get("cpu_checks") and not f.get("two_box"):
        raise refused("CPU checks require durable two-box artifacts")
    if f.get("repoint") and "cpuP" not in cfg["hosts"]:
        raise refused("repoint needs hosts.cpuP")
    if f.get("base_three"):
        for key in ("verdict_candidates", "base_models"):
            if set(f.get(key, {})) != {"13e_b2", "14b", "16"}:
                raise refused(f"base_three needs explicit {key} for all three models")


def cases(cfg, seeds):
    """Expand BEFORE sharding: weighted-cycle indices must never restart on a box."""
    c = cfg["collect"]
    pens = c["penetrations"].split(",")
    weights = [int(v) for v in c["pen_weights"].split(",")]
    if len(weights) != len(pens) or min(weights) < 1:
        raise ValueError("invalid penetration weights")
    cycle = [p for p, n in zip(pens, weights) for _ in range(n)]
    demands = c["demands"].split(",")
    return [{"index": i, "seed": s, "demand": demands[(i // len(cycle)) % len(demands)],
             "penetration": cycle[i % len(cycle)], "box": ("cpuP", "cpuQ")[i % 2]}
            for i, s in enumerate(seeds)]


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact_valid(path):
    from tools.fastloop.world import verify
    try:
        verify(Path(path))
        return True
    except (OSError, ValueError, KeyError):
        return False


def _world(loop, cfg, state, ph, root, item, box, attempt):
    local = root / "worlds" / f"c{state['cycles'][-1]['cycle']:02d}" / f"w{item['seed']}.a{attempt}"
    local.mkdir(parents=True, exist_ok=False)
    one = copy.deepcopy(cfg)
    one["hosts"]["cpu"] = one["hosts"][box]
    one["collect"].update(worlds_per_cycle=1, workers=1, demands=item["demand"],
                          penetrations=item["penetration"], pen_weights="1",
                          work_root=cfg["collect"]["work_root"] + f"/w{item['seed']}")
    if loop.long_group_sim_s(cfg):
        c = one["collect"]
        point = (int(item["index"]) + int(state["cycles"][-1]["cycle"])) % int(c["branch_points"])
        c["sumocredit"] = {**c.get("sumocredit", {}), "long_group_horizon": 300,
                           "long_group_time": int(c["branch_first"]) + point * int(c["branch_cadence"])}
    receipt = loop.run_collect(one, copy.deepcopy(state), {**ph, "attempt": attempt}, local,
                               _seeds=[item["seed"]])
    h = loop.host(one, "cpu")
    env = {"PYTHONPATH": f"{h['code_root']}/src:{h['code_root']}", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
           "C4_ALLOWED_ROOTS": receipt["work"]}
    argv = [h["python"], str(Path(h["code_root"]) / "tools/fastloop/world.py"),
            "--work", receipt["work"], "--split-seed", str(cfg["convert"].get("split_seed", 0)),
            "--val-frac", str(cfg["convert"].get("val_frac", 0.))]
    if loop.run_cmd(h, argv, env=env, log=local / "convert.log", timeout_s=10800):
        raise RuntimeError("world conversion failed; raw world remains charged")
    dst = local / "artifact"
    loop.disk_guard(cfg, loop.host(cfg, "gpu"), cfg["loop_root"],
                    float(cfg["convert"]["mb_per_episode"]) * int(cfg["collect"]["branch_points"]) * 2 / 1024)
    loop.sync(one, "cpu", receipt["work"] + "/artifact/", "gpu", str(dst), local / "pull.log")
    if not artifact_valid(dst):
        raise RuntimeError("incomplete world artifact; refuse completion")
    return {"case": item, "box": box, "attempt": attempt, "artifact": str(dst), "collect": receipt,
            "hashes_sha256": _hash(dst / "HASHES.json")}


def collect(loop, cfg, state, ph, root):
    """Durable per-world journal. retry collect retains successes and replays lost worlds.

    Every dispatched attempt is conservatively charged in full BEFORE dispatch, including
    failed SSH/staging attempts; never pretend an interrupted SUMO cost zero.
    """
    k = state["cycles"][-1]["cycle"]
    collection_wm = state.get("policy_wm", state["accepted_wm"]) if loop.fast_enabled(cfg, "overlap_policy") else state["accepted_wm"]
    need = (int(cfg["collect"]["worlds_per_cycle"]) * int(cfg["collect"]["branch_points"]) * 2 *
            float(cfg["convert"]["mb_per_episode"]) * 2 / 1024)
    loop.disk_guard(cfg, loop.host(cfg, "gpu"), cfg["loop_root"], need)
    journal = root / f"worlds_c{k:02d}.json"
    if journal.exists():
        doc = loop.read_json(journal)
        if not loop.fast_enabled(cfg, "spot_resume"):
            raise loop.Refused("world journal exists; spot_resume required for retry")
    else:
        seeds = loop.allocate_seeds(cfg, state, loop.check_registration(cfg), root)
        doc = {"cases": cases(cfg, seeds), "worlds": {}, "attempts": {}, "charged": 0,
               "policy_sha256": state["policy"]["sha256"],
               "wm_sha256": collection_wm["sha256"]}
        loop.write_atomic(journal, doc)
        loop.append(root / "seed_ledger.jsonl", {"cycle": k, "seeds": seeds, "event": "allocate_worlds"})
        state["seeds_used"] += seeds
        loop.save_state(cfg, state)
    if (doc["policy_sha256"] != state["policy"]["sha256"] or
            doc["wm_sha256"] != collection_wm["sha256"]):
        raise loop.Refused("retry policy changed")
    # The append-only charge ledger is authoritative across crashes between atomic state/journal writes.
    ledger_path = root / "seed_ledger.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
    seeds = [item["seed"] for item in doc["cases"]]
    if not any(row.get("event") == "allocate_worlds" and row.get("cycle") == k for row in ledger):
        loop.append(ledger_path, {"cycle": k, "seeds": seeds, "event": "allocate_worlds"})
    allocated = [s for row in ledger for s in row.get("seeds", [])] + seeds
    state["seeds_used"] += [s for s in dict.fromkeys(allocated) if s not in state["seeds_used"]]
    attempts = [row for row in ledger if row.get("event") == "world_attempt"]
    # A live loop can adopt two-box scheduling after an already-charged single-box failure.
    # Include ALL original batch charges as well as world charges, even after a stale-state crash.
    charges = [row for row in ledger if "sim_s_charged" in row]
    doc["charged"] = sum(row["sim_s_charged"] for row in charges if row["cycle"] == k)
    state["sim_s_charged"] = max(state["sim_s_charged"], sum(row["sim_s_charged"] for row in charges))
    for row in attempts:
        if row["cycle"] == k:
            key = str(row["seed"])
            doc["attempts"][key] = max(doc["attempts"].get(key, 0), row["attempt"])
    per_world = loop.collect_sim_s(cfg) // int(cfg["collect"]["worlds_per_cycle"])
    pending = []
    reg = loop.check_registration(cfg)
    for item in doc["cases"]:
        key = str(item["seed"])
        previous = doc["worlds"].get(key)
        if previous and artifact_valid(previous["artifact"]):
            continue
        attempt = doc["attempts"].get(key, 0) + 1
        box = item["box"] if attempt % 2 else ("cpuQ" if item["box"] == "cpuP" else "cpuP")
        pending.append((item, box, attempt))
    charge = per_world * len(pending)
    long_per_world = loop.long_group_sim_s(cfg) // int(cfg["collect"]["worlds_per_cycle"])
    long_charged = sum(row.get("long_group_sim_s_charged", 0) for row in charges if row["cycle"] == k)
    if long_charged + long_per_world * len(pending) > 8000:
        raise loop.Refused("long GROUP attempts exceed 8,000 additional sim-s/cycle (no refunds)")
    if doc["charged"] + charge > reg["budget"]["sim_s_per_cycle"]:
        raise loop.Refused("world retries exceed registered per-cycle sim-s (failed attempts stay charged)")
    if state["sim_s_charged"] + charge > reg["budget"]["sim_s_total"]:
        raise loop.Refused("world retries exceed total sim-s")
    # Journal attempted worlds before starting any SSH/SUMO. A crash here is overcharged, never undercharged.
    for item, box, attempt in pending:
        doc["attempts"][str(item["seed"])] = attempt
        loop.append(root / "seed_ledger.jsonl", {"cycle": k, "seed": item["seed"], "host": box,
                    "attempt": attempt, "sim_s_charged": per_world, "event": "world_attempt",
                    **({"long_group_sim_s_charged": long_per_world} if long_per_world else {})})
    doc["charged"] += charge
    state["sim_s_charged"] += charge
    loop.save_state(cfg, state)
    loop.write_atomic(journal, doc)
    failures = []
    # One pool per box enforces a total worker cap even when failover shifts all worlds.
    cap = int(cfg["fastloop"]["workers_per_box"])
    with ThreadPoolExecutor(max_workers=cap) as p, ThreadPoolExecutor(max_workers=cap) as q:
        futures = {(p if box == "cpuP" else q).submit(_world, loop, cfg, state, ph, root, item, box, attempt): item
                   for item, box, attempt in pending}
        for future in as_completed(futures):
            item = futures[future]
            try:
                doc["worlds"][str(item["seed"])] = future.result()
                incidents = sum((w["collect"].get("branch_collisions") or 0) +
                                (w["collect"].get("branch_teleports") or 0) for w in doc["worlds"].values())
                if incidents > int(reg["stop"].get("max_branch_incidents_per_cycle", 0)):
                    state["stopped"] = "SUMO safety incident in completed world (including partial cycle)"
                loop.save_state(cfg, state)
            except Exception as error:
                failures.append(f"{item['seed']}: {type(error).__name__}: {error}")
            loop.write_atomic(journal, doc)
    if failures:
        raise RuntimeError("worlds incomplete; retry collect resumes on peer: " + "; ".join(failures))
    worlds = [doc["worlds"][str(item["seed"])] for item in doc["cases"]]
    return {"journal": str(journal), "worlds": worlds, "seeds": [c["seed"] for c in doc["cases"]],
            "sim_s_charged": doc["charged"], "branch_collisions": sum(w["collect"].get("branch_collisions") or 0 for w in worlds),
            "branch_teleports": sum(w["collect"].get("branch_teleports") or 0 for w in worlds)}


def convert(loop, cfg, state, ph, root):
    k = state["cycles"][-1]["cycle"]
    tag = f"c{k:02d}.a{ph['attempt']}"
    target = root / "data" / f"pairs_{tag}"
    target.mkdir(parents=True, exist_ok=False)
    worlds = state["cycles"][-1]["phases"]["collect"]["receipt"]["worlds"]
    episodes, manifest = [], None
    merged_work = root / "merged" / tag
    for w in worlds:
        artifact = Path(w["artifact"])
        if not artifact_valid(artifact):
            raise RuntimeError("durable artifact corrupt; refuse conversion")
        doc = loop.read_json(artifact / "pairs/C4_EPISODES_MANIFEST.json")
        manifest = manifest or doc
        prefix = f"w{w['case']['seed']}"
        shutil.copytree(artifact / "pairs/episodes", target / "episodes" / prefix)
        for entry in doc["episodes"]:
            row = dict(entry)
            row["dir"] = "episodes/" + prefix + "/" + Path(row["dir"]).name
            row["source"] = str(artifact / row["source"].removeprefix("@artifact/"))
            episodes.append(row)
        extra = merged_work / "collect" / prefix
        extra.mkdir(parents=True)
        for name in ("SUMOCREDIT_POINTS.jsonl", "GATE_CALIBRATION.jsonl"):
            shutil.copy2(artifact / name, extra / name)
    if len({e["k_slots"] for e in episodes}) != 1:
        raise RuntimeError("world artifacts disagree on slot count")
    manifest = {**manifest, "episodes": episodes, "n_episodes": len(episodes),
                "n_train": sum(e["split_role"] == "train" for e in episodes),
                "n_val": sum(e["split_role"] == "val" for e in episodes),
                "roots": [w["artifact"] for w in worlds]}
    loop.write_atomic(target / "C4_EPISODES_MANIFEST.json", manifest)
    returns = root / "returns" / f"returns_{tag}.npz"
    h = loop.host(cfg, "gpu")
    env = {"PYTHONPATH": f"{h['code_root']}/src:{h['code_root']}", "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "C4_ALLOWED_ROOTS": cfg["loop_root"]}
    log = root / "logs" / f"merge_{tag}.log"
    argv = [h["python"], "-m", "wms.control.loop14_returns", "--source", str(target), "--output", str(returns),
            "--policy-checkpoint", state["policy"]["sha256"], "--kind", "POLICY"]
    if loop.run_cmd(h, argv, env=env, log=log, timeout_s=3600):
        raise RuntimeError("merged real returns export failed")
    out = {"pairs": str(target), "returns": str(returns), "returns_sha256": loop.sha256(returns),
           "episodes": len(episodes), "pair_groups": len({e["pair_key"] for e in episodes}),
           "targets": loop.read_json(returns.with_suffix(".receipt.json"))["targets"],
           "durable_world_artifacts": [w["artifact"] for w in worlds]}
    if cfg["collect"].get("sumocredit") or loop.long_group_sim_s(cfg):
        local = copy.deepcopy(cfg)
        local["hosts"]["cpu"] = local["hosts"]["gpu"]
        out["sumocredit"] = loop.convert_sumocredit(local, state, str(merged_work), tag, env, log, root)
    return out


def overlap(loop, cfg, state, root):
    """Runs in an owned thread, never writes shared LOOP14_STATE. One-cycle stale inputs are explicit."""
    snapshot = copy.deepcopy(state)
    k = state["cycles"][-1]["cycle"] + 1
    snapshot["cycles"] = [loop.new_cycle(k)]
    snapshot["policy_wm"] = copy.deepcopy(snapshot["accepted_wm"])
    dependencies = {"wm": snapshot["accepted_wm"], "resume_policy": snapshot["policy"],
                    "real_returns": snapshot["real_returns"],
                    "real_returns_sha256": loop.sha256(snapshot["real_returns"]),
                    "real_credit": snapshot.get("real_credit"), "launched_after_cycle": k - 1,
                    "returns_lag": "latest completed cycle at launch; current collection unavailable"}
    credit = snapshot.get("real_credit")
    if credit:
        dependencies["real_credit_sha256"] = loop.sha256(credit)
    args = cfg["policy"].get("extra_args", [])
    for flag in ("--horizon-gate", "--gate-panel"):
        if flag in args:
            path = args[args.index(flag) + 1]
            dependencies[flag.removeprefix("--")] = {"path": path, "sha256": loop.sha256(path)}
    receipt = loop.run_policy(cfg, snapshot, {"attempt": 1}, root)
    doc = {"cycle": k, "dependencies": dependencies, "policy": snapshot["policy"], "receipt": receipt}
    loop.write_atomic(root / f"overlap_c{k:02d}.json", doc)
    return doc


def process_token(pid):
    """Read only an exact recorded coordinator PID; never signal or search processes."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def promote(loop, cfg, state, root):
    k = state["cycles"][-1]["cycle"]
    path = root / f"overlap_c{k:02d}.json"
    if not path.exists():
        return
    doc = loop.read_json(path)
    if loop.sha256(doc["policy"]["checkpoint"]) != doc["policy"]["sha256"]:
        raise loop.Refused("overlap policy hash changed")
    wm = doc["dependencies"]["wm"]
    if loop.sha256(wm["checkpoint"]) != wm["sha256"]:
        raise loop.Refused("overlap WM hash changed")
    state["policy"] = doc["policy"]
    state["policy_wm"] = wm
    state["cycles"][-1]["phases"]["policy"].update(status="done", attempt=1,
            receipt={**doc["receipt"], "overlap": str(path), "trained_on_wm": wm,
                     "accepted_wm_at_promotion": state["accepted_wm"], "dependencies": doc["dependencies"]})
    loop.append(root / "cycle_ledger.jsonl", {"event": "overlap_promoted", "cycle": k,
                "trained_on_wm_sha256": wm["sha256"], "accepted_wm_sha256": state["accepted_wm"]["sha256"]})


def drive(loop, waiter, cfg, arm_dir):
    """Single state writer; GPU policy k+1 joins before WM-FT k can use the GPU."""
    import time
    root = Path(cfg["loop_root"])
    future = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            if (arm_dir / "STOP_ARM").exists() or (root / "STOP_DRIVER").exists():
                if future is not None:
                    future.result()  # own foreground child reaches its existing boundary/timeout
                return
            state = loop.load_state(cfg)
            if state.get("stopped"):
                if future is not None:
                    future.result()
                return
            k = state["cycles"][-1]["cycle"]
            phase = loop._phase(state)["name"]
            if phase == "policy":
                waiter.prepare_horizon(cfg, state, arm_dir)
            if phase in ("collect", "convert") and future is None:
                reg = loop.check_registration(cfg)
                target = root / f"overlap_c{k+1:02d}.json"
                if k + 1 < reg["budget"]["max_cycles"] and not target.exists():
                    waiter.prepare_horizon(cfg, state, arm_dir)
                    frozen = copy.deepcopy(cfg)
                    # Pin the horizon file for checkpoint recomputation and provenance.
                    args = frozen["policy"].get("extra_args", [])
                    if "--horizon-gate" in args:
                        i = args.index("--horizon-gate") + 1
                        dst = root / f"overlap_horizon_c{k+1:02d}.json"
                        shutil.copy2(args[i], dst)
                        args[i] = str(dst)
                    future = pool.submit(overlap, loop, frozen, state, root)
            if phase == "wm_ft" and future is not None:
                future.result()  # failure closes the driver before WM training
                future = None
            code = loop.run_phase(cfg, phase)
            if code == 5:
                time.sleep(120)
                continue
            if code not in (0, 4):
                raise RuntimeError(f"fastloop {phase} exit {code}; use the loop's explicit retry command")
            if phase == "check":
                after = loop.load_state(cfg)
                completed = next(c for c in after["cycles"] if c["cycle"] == k)
                waiter.retire_pair_cache(cfg, completed, arm_dir)


def check(loop, cfg, state, ph, root):
    """Acceptance on one CPU peer; immutable inputs staged per retry, panel returned and pinned."""
    k = state["cycles"][-1]["cycle"]
    box = ("cpuP", "cpuQ")[(ph["attempt"] - 1) % 2]
    h = loop.host(cfg, box)
    work = f"{cfg['collect']['work_root']}/checks/c{k:02d}.a{ph['attempt']}"
    log = root / "logs" / f"check_c{k:02d}.a{ph['attempt']}.log"
    loop.disk_guard(cfg, h, work, cfg["check"].get("need_gb", 1.))
    ft = state["cycles"][-1]["phases"]["wm_ft"]["receipt"]
    conv = state["cycles"][-1]["phases"]["convert"]["receipt"]
    for name, src in (("candidate/wm_s.pt", ft["candidate"]),
                      ("reference/wm_s.pt", state["accepted_wm"]["checkpoint"])):
        loop.sync(cfg, "gpu", src, box, f"{work}/{name}", log)
        for sidecar in ("physics.json", "TRAIN_RECEIPT.json", "lane_calib.json"):
            path = Path(src).with_name(sidecar)
            if path.is_file():
                loop.sync(cfg, "gpu", str(path), box, f"{work}/{Path(name).parent}/{sidecar}", log)
    loop.sync(cfg, "gpu", conv["pairs"] + "/", box, f"{work}/pairs", log)
    # Manifest sources refer to observable summary copies, needed for paired-difference checks.
    staged_manifest = loop.read_json(Path(conv["pairs"]) / "C4_EPISODES_MANIFEST.json")
    for w in state["cycles"][-1]["phases"]["collect"]["receipt"]["worlds"]:
        artifact = Path(w["artifact"])
        remote_summaries = f"{work}/summaries/w{w['case']['seed']}"
        loop.sync(cfg, "gpu", str(artifact / "summaries") + "/", box, remote_summaries, log)
        for entry in staged_manifest["episodes"]:
            if Path(entry["source"]).is_relative_to(artifact / "summaries"):
                entry["source"] = remote_summaries + "/" + Path(entry["source"]).name
    staged = root / "checks" / f"manifest_c{k:02d}.a{ph['attempt']}.json"
    staged.parent.mkdir(parents=True, exist_ok=True)
    loop.write_atomic(staged, staged_manifest)
    loop.sync(cfg, "gpu", str(staged), box, f"{work}/pairs/C4_EPISODES_MANIFEST.json", log)
    panel = root / "panel.json"
    if panel.exists():
        loop.sync(cfg, "gpu", str(panel), box, f"{work}/panel.json", log)
    argv = [h["python"], "-B", f"{h['code_root']}/tools/loop14_checks/accept.py",
            "--candidate", f"{work}/candidate/wm_s.pt", "--reference", f"{work}/reference/wm_s.pt",
            "--config", f"{h['code_root']}/configs/{Path(cfg['wm']['config']).name}",
            "--roots", *cfg["fastloop"]["check_roots"], "--pairs", f"{work}/pairs",
            "--panel", f"{work}/panel.json", "--out", f"{work}/verdict",
            "--onestep-windows", str(cfg["check"]["onestep_windows"]),
            "--rollout-windows", str(cfg["check"]["rollout_windows"]),
            "--pair-groups", str(cfg["check"]["pair_groups"]), "--stride-eps", str(cfg["check"].get("stride_eps", 10)),
            "--workers", str(min(14, cfg["check"].get("workers", 4)))]
    env = {**cfg["fastloop"].get("check_env", {}), "PYTHONPATH": f"{h['code_root']}/src:{h['code_root']}",
           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "WMS_DATA_LOOP14_PAIRS": f"{work}/pairs",
           "C4_ALLOWED_ROOTS": ":".join([work, *cfg["fastloop"]["check_roots"]])}
    code = loop.run_cmd(h, argv, env=env, log=log, timeout_s=cfg["check"]["timeout_h"] * 3600)
    out = loop.out_dir(root / "checks", f"c{k:02d}", ph["attempt"])
    loop.sync(cfg, box, f"{work}/verdict/", "gpu", str(out), log)
    returned = out / "PANEL_RETURNED.json"
    loop.sync(cfg, box, f"{work}/panel.json", "gpu", str(returned), log)
    if panel.exists() and loop.sha256(panel) != loop.sha256(returned):
        raise RuntimeError("CPU acceptance panel changed")
    if not panel.exists():
        shutil.copy2(returned, panel)
    verdict = loop.read_json(out / "ACCEPT.json")
    return {**loop.apply_verdict(cfg, state, ft, verdict, code, out, k), "host": box}
