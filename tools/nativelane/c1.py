"""Secondary C1 PI u_same ranking: same inherited plant/seed/protocol, native lane flag only."""
from __future__ import annotations

import concurrent.futures as F
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path.home() / "nativelane"
REPO = ROOT / "repo"
SOURCE = Path.home() / "c1refresh"
DEADLINE = datetime.fromisoformat("2026-10-26T09:00:00+08:00").timestamp()
spec = importlib.util.spec_from_file_location("c1r_run", SOURCE / "code/tools/c1refresh/c1r_run.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
sys.path.insert(0, str(REPO / "src"))
original = C.make_closed_loop
ON = False


def make_controller(model, record, arm, ids, is_cav, n_steps, t0_s, plant="model"):
    inner, wall = original(model, record, arm, ids, is_cav, n_steps, t0_s, plant)
    if not ON or arm != "native":
        return inner, wall
    import torch
    from wms.model.cav import WM_LANE_TARGET_KEY, WM_MERGE_KEY

    def callback(state):
        out = inner(state).clone()
        mask = state["present"].bool() & state["is_cav"].bool() & (state["lane"] >= 0)
        out[..., 1] = torch.where(mask, state[WM_LANE_TARGET_KEY], out[..., 1]).to(out.dtype)
        # The persistent first-passage merge, not the C1 adapter's gap6 teacher.
        out[..., 1] = torch.where(mask & (state["lane"] == 2), state["lane"], out[..., 1])
        state[WM_MERGE_KEY] = mask & (state["lane"] == 2)
        return out

    callback.native_lane_mask = lambda state: state["present"].bool() & state["is_cav"].bool() & (state["lane"] >= 0)
    return callback, wall


C.make_closed_loop = make_controller


def init():
    C._init(str(SOURCE))


def work(task):
    global ON
    tag, u = task
    ON = tag == "on"
    if time.time() >= DEADLINE:
        return tag, u, {"error": "deadline"}
    return tag, u, C._run_unit(u)


def analyze():
    sys.path.insert(0, str(Path.home() / "mappodiag"))
    from analyze_b import boot
    res = {}
    for tag in ("off", "on"):
        docs = [json.loads(p.read_text()) for p in (ROOT / tag / "c1").glob("*.json")]
        bysid = {}
        for d in docs:
            if not d.get("error"):
                bysid.setdefault(d["sid"], {})[d["arm"]] = d
        res[tag] = {}
        for h in (60., 120., 300.):
            rows = []
            for sid, arms in bysid.items():
                if not all(a in arms for a in ("native", "pi")):
                    continue
                n, p = [arms[a]["traffic"].get(str(h)) for a in ("native", "pi")]
                if n is None or p is None:
                    continue
                pred = n["pred_all"]["network_delay"] - p["pred_all"]["network_delay"]
                actual = n["rec_all"]["network_delay"] - p["rec_all"]["network_delay"]
                entry = C_INDEX[sid]
                rows.append({"sid": sid, "world": entry["world"], "pred_gain": pred, "sumo_gain": actual,
                             "choose_pi": pred > 0, "correct": bool((pred > 0) == (actual > 0)),
                             "regret": abs(actual) if ((pred > 0) != (actual > 0)) else 0.})
            if rows:
                cl = [r["world"] for r in rows]
                res[tag][str(h)] = {"n": len(rows), "worlds": len(set(cl)),
                    "accuracy_ci95": boot([r["correct"] for r in rows], cl),
                    "regret_ci95": boot([r["regret"] for r in rows], cl), "rows": rows}
    p = ROOT / "C1_RANKING.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({tag: {h: {k: v for k, v in d.items() if k != "rows"} for h, d in x.items()}
                      for tag, x in res.items()}), flush=True)


index = json.loads((SOURCE / "states/INDEX.json").read_text())
C_INDEX = index["states"]


def main():
    tasks = []
    for sid, d in C_INDEX.items():
        if "pi_u_same" not in d["sets"]:
            continue
        import re
        match = re.search(r"[ws](\d{6})", d["world"])
        assert match is None or not 964000 <= int(match[1]) <= 964199
        for arm in ("native", "pi"):
            assert arm in d["arms"] and d["recorded_steps"][arm] >= 600
            u = ("wm13e", "cl", 300, sid, arm, "NATIVELANE")
            for tag in ("off", "on"):
                path = ROOT / tag / "c1" / f"{sid}__{arm}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists() or json.loads(path.read_text()).get("error"):
                    tasks.append((tag, u))
    print(f"C1 optional units remaining: {len(tasks)}", flush=True)
    with F.ProcessPoolExecutor(12, initializer=init) as pool:
        pending, todo = {}, iter(tasks)

        def submit():
            if time.time() >= DEADLINE or (ROOT / "STOP").exists():
                return
            t = next(todo, None)
            if t is not None:
                pending[pool.submit(work, t)] = t
        for _ in range(12):
            submit()
        while pending:
            done, _ = F.wait(pending, return_when=F.FIRST_COMPLETED)
            for f in done:
                pending.pop(f)
                tag, u, doc = f.result()
                path = ROOT / tag / "c1" / f"{u[3]}__{u[4]}.json"
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(doc))
                temp.replace(path)
                print(tag, u[3], u[4], doc.get("seconds"), doc.get("error"), flush=True)
                submit()
    analyze()


if __name__ == "__main__":
    main()
