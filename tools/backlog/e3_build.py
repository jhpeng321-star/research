"""S-BACKLOG step 3: S-E3DATA-1 calibration dataset with exact entrance-waiting cost clocks.

Runs on the collection host that holds the raw e3long tapes of its shard (cpuP: even seeds,
cpuQ: odd seeds).  Reads ONLY the registered 'calibration' and 'validation' worlds of
SPLIT.json (never 'fit') and the arms native / ours_c1 / coopoff_c1 (MAPPO has no loadable
CavPolicy, so it has no current-policy imagination row; it is excluded and reported).

Per episode: hard-links the already converted S-E3DATA-1 arrays (byte-identical), adds
``entry_backlog.npy`` computed from the raw tape's recorded request schedule
(``scenario/requests.rou.xml`` only; no other scenario file is opened) on the converted time
grid, and writes a per-tick real cost-curve sidecar (``cost_curves.npz``) under the E3-long
Measurement convention.  Both are reconciled to the AUTHORITATIVE cumulative_120/300 clocks of
the job (road_delay_vs, source_wait_vs; np.isclose rtol 1e-6 atol 1e-3, fail closed).

Usage: python e3_build.py --src ~/backlog/e3src --manifest e3_manifest.json --split e3_split.json \
           --parity 0 --out ~/backlog/stage/e3cal.cpuP
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wms.data.c4_episodes import exact_entry_backlog, read_requests  # noqa: E402

DT = 0.5
SPEED_LIMIT = 31.29
ARMS = ("native", "ours_c1", "coopoff_c1")
ROLES = ("calibration", "validation")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def e3_road_increments(present, vid, v_mps, t0_tick, n):
    """E3-long road delay per 0.5 s interval k -> k+1 (BranchTracker convention, reproduces the
    authoritative road_delay_vs): cohort = slots present at k; v_end = v[k+1] if the same vehicle
    continues, else v[k] (exit or slot reuse)."""
    p0 = present[t0_tick:t0_tick + n]
    p1 = present[t0_tick + 1:t0_tick + 1 + n]
    v0, v1 = v_mps[t0_tick:t0_tick + n], v_mps[t0_tick + 1:t0_tick + 1 + n]
    same = p1 & (vid[t0_tick + 1:t0_tick + 1 + n] == vid[t0_tick:t0_tick + n])
    v_eff = np.where(same, v1, v0)
    return DT * np.maximum(0.0, 1.0 - v_eff / SPEED_LIMIT) * p0           # [n, slots]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--parity", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, src = Path(a.out), Path(a.src)
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr)
        return 2
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))["worlds"]
    chosen = [e for e in man["episodes"]
              if split[str(e["seed"])] in ROLES and e["arm"] in ARMS and e["seed"] % 2 == a.parity]
    entries, curves, recon = [], {}, []
    for e in chosen:
        assert e["e3data_split"] == split[str(e["seed"])]
        job = Path(e["source"]["job"])
        tape = job / "deploy" / "episode"
        if not (tape / "episode.npz").is_file():
            print(f"REFUSED: raw tape missing {tape}", file=sys.stderr)
            return 3
        tape_sha = sha256(tape / "episode.npz")
        with np.load(tape / "episode.npz", allow_pickle=False) as d:
            ep = {k: d[k] for k in d.files}
        t_all = np.asarray(ep["time_s"], np.float64)
        present = np.asarray(ep["present"], bool)
        vid = np.asarray(ep["vid_index"], np.int32)
        v = np.where(present, ep["v_mps"], 0.0).astype(np.float64)
        names, depart, entrance = read_requests(tape / "scenario" / "requests.rou.xml")
        queue, qstats = exact_entry_backlog(t_all, vid, present, [str(x) for x in ep["vehicle_ids"]],
                                            names, depart, entrance, ep.get("entry_first_seen_s"))
        # converted episode grid = contiguous slice of the raw tape grid
        dst_dir = src / e["dir"]
        conv_t = np.load(dst_dir / "time_s.npy")
        i0 = int(np.searchsorted(t_all, conv_t[0] - 1e-6))
        if not np.array_equal(t_all[i0:i0 + len(conv_t)], conv_t):
            print(f"REFUSED: converted grid is not a slice of the raw tape for {e['episode']}", file=sys.stderr)
            return 4
        backlog = queue["entry_backlog"][i0:i0 + len(conv_t)].astype(np.float32)
        # E3-long window clocks from t0
        t0 = float(e["branch"]["t0_s"])
        tick = int(np.searchsorted(t_all, t0 - 1e-6))
        n = int(round(float(e["horizon_s"]) / DT))
        road = e3_road_increments(present, vid, v, tick, n).sum(1)               # [n]
        src_k = DT * queue["entry_backlog"][tick:tick + n].sum(1).astype(np.float64)
        rec = {"episode": e["episode"], "tape_episode_sha256": tape_sha,
               "manifest_episode_sha256": e["source"]["episode_sha256"]}
        for hz in (120, 300):
            cum = json.loads((job / f"cumulative_{hz}.json").read_text(encoding="utf-8"))
            m = int(hz / DT)
            r_ok = np.isclose(road[:m].sum(), cum["road_delay_vs"], rtol=1e-6, atol=1e-3)
            s_ok = np.isclose(src_k[:m].sum(), cum["source_wait_vs"], rtol=1e-6, atol=1e-3)
            rec[f"h{hz}"] = {"road": float(road[:m].sum()), "road_authoritative": cum["road_delay_vs"],
                             "source": float(src_k[:m].sum()), "source_authoritative": cum["source_wait_vs"],
                             "reconciled": bool(r_ok and s_ok),
                             "cumulative_sha256": sha256(job / f"cumulative_{hz}.json"),
                             "manifest_clock_sha256": e["source"]["clocks_sha256"].get(str(hz))}
            if not (r_ok and s_ok):
                print(f"REFUSED: clocks do not reconcile for {e['episode']} h={hz}: {rec[f'h{hz}']}",
                      file=sys.stderr)
                return 5
        recon.append(rec)
        dst = out / e["dir"]
        dst.mkdir(parents=True)
        for f in sorted(dst_dir.iterdir()):
            if f.name == "episode.json":
                continue
            os.link(f, dst / f.name)                      # byte-identical converted arrays
        np.save(dst / "entry_backlog.npy", backlog, allow_pickle=False)
        entry = {**e, "queue_arrays": {"source": "scenario/requests.rou.xml", **qstats,
                                       "e3_clock_reconciliation": {k: v for k, v in rec.items()
                                                                    if k.startswith("h")}},
                 "split_role": "train", "role_e3": split[str(e["seed"])]}
        (dst / "episode.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
        entries.append(entry)
        curves[e["episode"]] = (road + src_k).astype(np.float64)
    manifest = {**{k: v for k, v in man.items() if k != "episodes"}, "episodes": entries,
                "n_episodes": len(entries), "n_train": len(entries), "n_val": 0,
                "backlog_e3": {"task": "S-BACKLOG", "parity": a.parity, "roles": list(ROLES), "arms": list(ARMS),
                               "excluded_arm": "mappo (no CavPolicy checkpoint; no current-policy imagination)",
                               "worlds": len({e["seed"] for e in entries})}}
    (out / "C4_EPISODES_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    np.savez(out / "cost_curves.npz", **curves)
    (out / "E3_CLOCK_RECONCILIATION.json").write_text(json.dumps(recon, indent=1), encoding="utf-8")
    sums = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS.json"}
    (out / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "episodes": len(entries), "worlds": manifest["backlog_e3"]["worlds"],
                      "all_reconciled": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
