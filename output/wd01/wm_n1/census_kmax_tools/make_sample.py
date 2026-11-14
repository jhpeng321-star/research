"""Draw the fixed 300-run stratified census sample (A5 rule note §1).

Population: the featurizable TRAIN + CAL runs of pack_v2c, read from
``D:/rd-staging/wmn1_feat/pack_v2c/INDEX.json`` (READ-ONLY).  Strata are
batch x loading x penetration; allocation is proportional with at least one
run per non-empty stratum; the draw uses ``random.Random(20261001)``.

Seal guards (refuse = abort, never skip silently): HOLDOUT-named worlds,
chunk-5 seed block 8707-8736, chunk-5 batch labels, and any overlap with
``inventory/holdout_names.json``.  The sample file is written on the E:
worktree ONLY, before any run is parsed.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WM_N1 = _HERE.parent
for _p in (str(_WM_N1), str(_WM_N1.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus import _util as U          # noqa: E402

INDEX_PATH = Path("D:/rd-staging/wmn1_feat/pack_v2c/INDEX.json")
HOLDOUT_NAMES = _WM_N1 / "inventory" / "holdout_names.json"
SAMPLE_PATH = _WM_N1 / "KCENSUS_SAMPLE.json"
DATA_ROOT = "E:/research-dense"
SAMPLE_N = 300
SEED = 20261001
KEEP_SPLITS = ("TRAIN", "CALIBRATION")
CHUNK5_SEED_LO, CHUNK5_SEED_HI = 8707, 8736
WORLD_RE = re.compile(r"^w\d+m(\d{4})p(\d{2})t\d+$")


class SampleRefusal(PermissionError):
    """A sealed/excluded run reached the sample draw."""


def world_seed(world: str):
    m = re.match(r"^[a-zA-Z]*(\d+)", str(world))
    return int(m.group(1)) if m else None


def load_population() -> list:
    index = U.read_json(INDEX_PATH)
    runs = [r for r in index.get("runs", [])
            if str(r.get("split", "")).upper() in KEEP_SPLITS]
    holdout = U.read_json(HOLDOUT_NAMES) if HOLDOUT_NAMES.is_file() \
        else {"runs": []}
    holdout_keys = {f"{h.get('batch')}/{h.get('world')}__{h.get('arm')}"
                    for h in holdout.get("runs", [])}
    for r in runs:
        world = str(r["world"])
        key = str(r["key"])
        if str(r.get("split", "")).upper() not in KEEP_SPLITS:
            raise SampleRefusal(f"{key}: split {r.get('split')!r} refused")
        if world.upper().startswith("HOLDOUT"):
            raise SampleRefusal(f"{key}: HOLDOUT world refused")
        if any(str(r.get("batch", "")).startswith(c)
               for c in ("chunk5", "chunk_5", "chunk-5")):
            raise SampleRefusal(f"{key}: chunk-5 batch refused")
        seed = world_seed(world)
        if seed is not None and CHUNK5_SEED_LO <= seed <= CHUNK5_SEED_HI:
            raise SampleRefusal(
                f"{key}: seed {seed} in sealed chunk-5 block refused")
        if key in holdout_keys:
            raise SampleRefusal(f"{key}: listed in holdout_names.json")
        if not WORLD_RE.match(world):
            raise SampleRefusal(f"{key}: unparseable world name {world!r}")
        if not r.get("root_rel") or not r.get("shard_dir"):
            raise SampleRefusal(f"{key}: missing run location fields")
    return runs


def stratum_of(run: dict) -> tuple:
    m = WORLD_RE.match(str(run["world"]))
    return (str(run["batch"]), f"m{m.group(1)}", f"p{m.group(2)}")


def allocate(sizes: dict) -> dict:
    """Proportional allocation, >= 1 per non-empty stratum, sum == SAMPLE_N."""
    total = sum(sizes.values())
    raw = {s: SAMPLE_N * n / total for s, n in sizes.items()}
    k = {s: min(n, max(1, int(raw[s]))) for s, n in sizes.items()}
    rem = SAMPLE_N - sum(k.values())
    if rem > 0:
        order = sorted(sizes, key=lambda s: (-(raw[s] - int(raw[s])), s))
        while rem > 0:
            progressed = False
            for s in order:
                if rem == 0:
                    break
                if k[s] < sizes[s]:
                    k[s] += 1
                    rem -= 1
                    progressed = True
            if not progressed:
                raise ValueError("cannot allocate: all strata exhausted")
    elif rem < 0:
        order = sorted(sizes, key=lambda s: (-k[s], s))
        while rem < 0:
            progressed = False
            for s in order:
                if rem == 0:
                    break
                if k[s] > 1:
                    k[s] -= 1
                    rem += 1
                    progressed = True
            if not progressed:
                raise ValueError("cannot allocate: every stratum at minimum")
    assert sum(k.values()) == SAMPLE_N
    return k


def main() -> int:
    runs = load_population()
    by_stratum: dict = {}
    for r in runs:
        by_stratum.setdefault(stratum_of(r), []).append(r)
    sizes = {s: len(v) for s, v in by_stratum.items()}
    k = allocate(sizes)

    rng = random.Random(SEED)
    sampled = []
    for s in sorted(by_stratum):
        members = sorted(r["key"] for r in by_stratum[s])
        chosen = rng.sample(members, k[s])
        sampled.extend(chosen)
    by_key = {r["key"]: r for r in runs}
    entries = []
    for key in sampled:
        r = by_key[key]
        st = stratum_of(r)
        entries.append(dict(
            key=key, batch=r["batch"], world=r["world"], arm=r["arm"],
            split=r["split"], loading=st[1], penetration=st[2],
            t0_s=float(r["t0_s"]),
            run_dir=f"{DATA_ROOT}/{r['root_rel']}/{r['world']}__{r['arm']}",
            shard_dir=r["shard_dir"]))
    assert len(entries) == SAMPLE_N == len({e["key"] for e in entries})

    doc = dict(
        schema="wm_n1_kcensus_sample/1",
        rule_note="REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md",
        rule_note_sha256=U.sha256_file(
            _WM_N1 / "REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md"),
        created_utc=U.utc_now(),
        seed=SEED,
        sample_n=SAMPLE_N,
        population=dict(
            inventory=str(INDEX_PATH),
            inventory_sha256=U.sha256_file(INDEX_PATH),
            n_runs=len(runs),
            splits=list(KEEP_SPLITS),
            n_strata=len(sizes),
        ),
        method=(
            "strata = batch x loading x penetration; proportional allocation "
            "k_s = max(1, floor(300*n_s/N)) with largest-remainder top-up "
            "(ties by stratum key) and >=1 per non-empty stratum; draw via "
            "random.Random(20261001).sample(sorted(keys), k_s) per stratum "
            "in sorted stratum-key order; written before any run was parsed"),
        strata=[dict(batch=s[0], loading=s[1], penetration=s[2],
                     n=sizes[s], k=k[s]) for s in sorted(sizes)],
        runs=entries,
    )
    U.write_json_atomic(SAMPLE_PATH, doc)
    print(f"wrote {SAMPLE_PATH}")
    print(f"  runs={len(entries)} strata={len(sizes)} seed={SEED}")
    print(f"  sha256={U.sha256_file(SAMPLE_PATH)}")
    print(f"  size={SAMPLE_PATH.stat().st_size} bytes")
    print(f"  created_utc={doc['created_utc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
