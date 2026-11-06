"""Synthetic wz3to2 sources for a MECHANICS smoke of the fine-tune runner (no SUMO, not evidence).

Uses WM-A's deterministic kinematic generator (tests/wz/synth.py, WZ3TO2 geometry) to write
``c4_sumo_episode/1`` episodes at the real slot count (K = 512), converts each arm into a C4
episode source with the unchanged converter (world-grouped split), so ``run.py prepare/train
--smoke-only`` can measure seconds per step and peak GPU memory before real data exists.

    python -m tools.wzwm.synth_smoke --out DIR [--worlds 6] [--k 512] [--total-s 230]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(out: Path, *, worlds: int = 6, k: int = 512, total_s: float = 230.0, warmup_s: float = 90.0,
          first_seed: int = 989101) -> dict:
    from tests.wz.synth import WZ3TO2, write_episode
    from wms.data.c4_episodes import convert

    import shutil

    out = Path(out)
    srcs, seeds, seed = {}, [], first_seed
    while len(seeds) < worlds:          # the kinematic generator rejects a few seeds (edge case)
        probe = out / "probe"
        try:
            write_episode(probe, WZ3TO2, seed=seed, demand=3200, penetration=0.2, warmup_s=warmup_s,
                          total_s=total_s, k=k)
            seeds.append(seed)
        except ValueError:
            pass
        shutil.rmtree(probe, ignore_errors=True)
        seed += 1
        if seed > first_seed + 90:
            raise RuntimeError("synthetic generator rejected too many seeds")
    for arm in ("native", "mappo_u0"):
        eroot = out / "episodes" / arm
        for s in seeds:
            write_episode(eroot, WZ3TO2, seed=s, demand=3200, penetration=0.2, policy=arm,
                          warmup_s=warmup_s, total_s=total_s, k=k,
                          extra_meta={"map": "wz3to2", "synthetic": True})
        doc = convert([eroot], out / f"src_{arm}", val_frac=0.34, split_seed=0)
        srcs[f"wz_{arm}"] = {"path": str(out / f"src_{arm}"), "n_train": doc["n_train"], "n_val": doc["n_val"]}
    (out / "SYNTH.json").write_text(json.dumps({"synthetic": True, "k": k, "worlds": worlds,
                                                "total_s": total_s, "sources": srcs}, indent=1))
    return srcs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--total-s", type=float, default=230.0)
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.out), worlds=a.worlds, k=a.k, total_s=a.total_s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
