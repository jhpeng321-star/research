"""WD-01 VALIDATION LOOK #8 (single, final for the frozen p3o
candidate).  Dispatch brief: evaluate the FROZEN p3o candidate on the
4 native Part A validation roots under gate v2 exactly as registered
(VALIDATION_GATE_V2.md + AMENDMENT_1: primary H = 30, same thresholds;
(d) per-horizon binding; TTC coverage included).

HASH GATE FIRST: the candidate, the calibration and BOTH gate files
must match their registered sha256 (LF-normalised text, the part 3m
convention) BEFORE anything is rolled; any mismatch aborts the run.

What is rolled (the ONE validation evaluation of this candidate):
  results/raw/raw_val8_m{0..4}.jsonl  the 5 FROZEN candidate members
                                      (candidate_p3o_iter2.json, engine
                                      flags lc_scale_disc +
                                      merge_accept_prob) x 4 validation
                                      roots x t0 every 30 s in
                                      [120, 480] (52 windows) x H in
                                      {30, 60}, TYPED rows (LC by type
                                      on both sides);
  results/raw/raw_val8_L0.jsonl       L0 (deterministic IDM-normal)
                                      on the SAME 52 windows, computed
                                      in THIS run -- the (c) MAE and
                                      (d) per-horizon reference.

Per-member files are resumable: a complete file (104 parsable rows) is
skipped, an incomplete one is re-written with the SAME frozen settings
(no parameter change ever).  Raw rows are saved BEFORE any aggregation
(aggregate_val8.py reads them; it never re-rolls).

ZERO simulation: no SUMO/TraCI anywhere; the only dynamics is the
part-3 rollout engine (pure torch).  HOLDOUT BLOCKED: the holdout
roots are never passed to root_arrays (asserted below); this look
never touches them.  ONE look: this evaluation pass is not repeated
with changed settings regardless of the verdict.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
P3O = HERE.parent / "part3o"
sys.path.insert(0, str(P3O))
import p3o_lib as O  # noqa: E402
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402
import p3lib as P3  # noqa: E402

# Registered freeze 2026-09-27T05:09:19Z (dispatch brief; the gate-file
# hashes were given truncated there as 84b8a23d...a47f / 27aeddab...79d4
# -- these are the full values, prefix/suffix matching the brief).
REGISTERED = {
    "candidate_p3o_iter2.json":
        "458890d9c4b0ee6ee03de145be50498d9bd0873505a4e86f2bf8a1e224ebe45d",
    "calibration_p3o.json":
        "c7ffa7540768fa67e75e99881eec6f801201990dea4ace90af93c202f789c22a",
    "VALIDATION_GATE_V2.md":
        "84b8a23df1ff654f8d602e1fd75119517c4bd7cb458dfb2191d2d088452da47f",
    "VALIDATION_GATE_V2_AMENDMENT_1.md":
        "27aeddab51a899dd1e7a7e24f073fbfe4d29efbe6c2cd3db7e992663d39d79d4",
}
EXPECTED_VAL_ROOTS = ["cv20_7005m2000", "cv20_7006m2000",
                      "cv30_7305m2700", "seed_7535m2400"]

RAW_DIR = HERE / "results" / "raw"
STDOUT_DIR = HERE / "results" / "stdout"
VAL_T0S = tuple(range(120, 481, 30))       # 13 t0 x 4 roots = 52 windows
ROWS_PER_MEMBER = 2 * len(VAL_T0S) * 4     # (window, H) rows per member
T_START = time.time()


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def verify_registered():
    """HASH GATE: every registered file must exist and match its
    registered LF-normalised sha256; the two freeze sidecars must
    carry the same values.  Refuses to run otherwise."""
    ok = True
    for name, want in REGISTERED.items():
        p = (P3O / "results" / name) if name.endswith(".json") else \
            (HERE.parent / "part3f" / name)
        if not p.exists():
            print(f"HASH GATE FAIL: {p} MISSING"); ok = False; continue
        got = N.sha256_text(p.read_text(encoding="utf-8"))
        flag = "OK" if got == want else "MISMATCH"
        ok &= got == want
        print(f"HASH {flag}: {p}")
        print(f"        registered {want}")
        print(f"        observed  {got}")
    for stem in ("candidate_p3o_iter2.json", "calibration_p3o.json"):
        side = N.load_json(P3O / "results" / (stem + ".FROZEN.sha256"))
        agree = side.get("sha256") == REGISTERED[stem]
        ok &= agree
        print(f"SIDECAR {'OK' if agree else 'MISMATCH'}: {stem}.FROZEN."
              f"sha256 sha256={side.get('sha256')} "
              f"frozen_utc={side.get('frozen_utc')}")
    if not ok:
        raise SystemExit("HASH GATE FAILED -- the evaluation REFUSES to "
                         "run (frozen inputs modified or missing)")
    print("HASH GATE PASSED (candidate, calibration, gate v2, "
          "amendment 1).")


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STDOUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(STDOUT_DIR / "stdout_eval_val8.txt")

    print("priority below-normal:", D.below_normal_priority())
    print("=== WD-01 VALIDATION LOOK #8 (single, final for this "
          "candidate) ===")
    print("ZERO simulation; HOLDOUT not accessed (never passed to "
          "root_arrays); ONE evaluation pass, frozen settings only.")

    # ---- 1. hash gate FIRST (refuse to run on mismatch)
    verify_registered()

    # ---- 2. split asserts (holdout blocked; validation disjoint)
    sp = O.splits()
    val, holdout = list(sp["val"]), set(sp["holdout"])
    assert sorted(val) == EXPECTED_VAL_ROOTS, val
    assert not (set(val) & holdout)
    assert not (set(val) & set(sp["fit"]))
    assert not (set(val) & set(sp["calibration"]))
    print(f"validation roots {val}; t0 every 30 s in [120, 480] "
          f"({len(VAL_T0S)} t0 x {len(val)} roots = "
          f"{len(VAL_T0S)*len(val)} windows); H {O.HORIZONS}; M=5 "
          f"members; L0 rolled in this run")
    print(f"holdout {sorted(holdout)} BLOCKED (never passed anywhere)")

    # ---- 3. the FROZEN candidate (hash verified above)
    cand = N.load_json(P3O / "results" / "candidate_p3o_iter2.json")
    assert cand["frozen_candidate"] is True
    configs = [dict(m) for m in cand["members"]]
    assert len(configs) == 5
    print(f"frozen candidate: {cand['variant']}")
    print(f"engine flags: {cand['engine_flags']}")
    for mi, cfg in enumerate(configs):
        print(f"  member {mi}: {O.cfg_note(cfg)}")

    # ---- 4. the ONE validation evaluation (resumable raw rows)
    dev = P3.device()
    for mi, cfg in enumerate(configs):
        path = RAW_DIR / f"raw_val8_m{mi}.jsonl"
        if O.member_file_complete(path, ROWS_PER_MEMBER):
            log(f"member {mi}: {path.name} complete "
                f"({ROWS_PER_MEMBER} rows) -- resumed, skipped")
            continue
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for root in val:
                for t0 in VAL_T0S:
                    rows = O.roll_member_rows(
                        root, float(t0), cfg, mi, "L6", dev,
                        "val8 (frozen p3o candidate iter2): " +
                        O.cfg_note(cfg))
                    for row in rows:
                        f.write(json.dumps(row, default=str) + "\n")
                        n += 1
                    f.flush()
        log(f"member {mi}: wrote {n} rows -> {path.name}")

    l0_path = RAW_DIR / "raw_val8_L0.jsonl"
    if O.member_file_complete(l0_path, ROWS_PER_MEMBER):
        log(f"L0: {l0_path.name} complete -- resumed, skipped")
    else:
        n = 0
        with open(l0_path, "w", encoding="utf-8") as f:
            for root in val:
                for t0 in VAL_T0S:
                    r = O.roll_l0(root, float(t0), dev)
                    for row in N.window_raw_rows(
                            r, "L0", 0, root, t0, 1.0,
                            "val8: L0 idm_normal (deterministic; rolled "
                            "in this run for (c) MAE and (d) reference)"):
                        f.write(json.dumps(row, default=str) + "\n")
                        n += 1
                    f.flush()
        log(f"L0: wrote {n} rows -> {l0_path.name}")

    n_l0 = len(O.read_rows(l0_path))
    n_l6 = sum(len(O.read_rows(RAW_DIR / f"raw_val8_m{mi}.jsonl"))
               for mi in range(5))
    print(f"\nraw rows saved BEFORE aggregation: L6 {n_l6} "
          f"(5 members x {len(val)*len(VAL_T0S)} windows x "
          f"{len(O.HORIZONS)} H) + L0 {n_l0}")
    print("\nDONE the ONE validation evaluation (next: "
          "aggregate_val8.py -- no re-roll, no second look regardless "
          "of verdict).")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
