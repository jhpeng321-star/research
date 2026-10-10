"""Run the read-only K_max drop-location census (A5 rule note §1-§3).

Consumes the committed ``KCENSUS_SAMPLE.json`` (300 runs; NEVER re-drawn
here), replays the frozen featurizer slot-assignment loop at K = 128 (plus a
K = 192 residual replay) per run with <= 6 below-normal-priority workers and
a >= 6 GB free-virtual-memory gate, and writes ``KCENSUS_PER_RUN.json`` +
``KCENSUS_RESULT.json`` / ``KCENSUS_RESULT.md`` on the E: worktree only.

Per-run integrity: sha256 of the raw physics file must equal the pack INDEX
``input_sha256`` (the featurizer's exact source bytes), and the census
dropped vehicle-ticks must equal the pack_v2c meta ``n_dropped_slot_steps``
(SLOT_POOL_CENSUS.json).  Sampled runs whose raw source no longer exists are
recorded as explicit failures -- the fixed 300-run sample is never re-drawn,
shrunk or substituted.  Zero SUMO; no outcome, label or summary file is
opened (open-path audit per run).  Counts and positions only; the B / K=192
decision is NOT applied or stated -- §3 numbers are output as measured.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WM_N1 = _HERE.parent
for _p in (str(_WM_N1), str(_WM_N1.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus import _util as U              # noqa: E402
from corpus import memory as MEM           # noqa: E402
import census_lib as CL                    # noqa: E402

SAMPLE_PATH = _WM_N1 / "KCENSUS_SAMPLE.json"
PER_RUN_PATH = _WM_N1 / "KCENSUS_PER_RUN.json"
RESULT_JSON = _WM_N1 / "KCENSUS_RESULT.json"
RESULT_MD = _WM_N1 / "KCENSUS_RESULT.md"
STATUS_MD = _WM_N1 / "KCENSUS_STATUS.md"
CENSUS_JSON = _WM_N1 / "SLOT_POOL_CENSUS.json"

LOADINGS = ("m2000", "m2400", "m2600", "m2700")
MAX_WORKERS = 6
MIN_FREE_VIRTUAL_GB = 6.0
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 20261001


def load_jobs() -> list:
    doc = U.read_json(SAMPLE_PATH)
    index = U.read_json(Path("D:/rd-staging/wmn1_feat/pack_v2c/INDEX.json"))
    sha_by_key = {r["key"]: r.get("input_sha256") for r in index["runs"]}
    jobs = []
    for e in doc["runs"]:
        arm = str(e["arm"])
        t_origin = 0.0 if arm in ("base", "native_base") else float(e["t0_s"])
        jobs.append(dict(key=e["key"], run_dir=e["run_dir"],
                         t_origin=t_origin, loading=e["loading"],
                         penetration=e["penetration"], batch=e["batch"],
                         split=e["split"],
                         input_sha256=sha_by_key.get(e["key"])))
    return jobs


def load_expected() -> dict:
    doc = U.read_json(CENSUS_JSON)
    return {r["key"]: int(r["dropped_vehicle_ticks"]) for r in doc["runs"]}


def spot_check_shard_meta(jobs: list, expected: dict, n: int = 3) -> list:
    """Paste-able D: read-only check: shard meta == SLOT_POOL_CENSUS rows."""
    checks = []
    for job in jobs[:n]:
        shard_dir = None
        for e in U.read_json(SAMPLE_PATH)["runs"]:
            if e["key"] == job["key"]:
                shard_dir = Path(e["shard_dir"])
                break
        meta = U.read_json(shard_dir / "meta.json")
        meta_drops = int(meta["n_dropped_slot_steps"])
        ok = meta_drops == expected[job["key"]]
        checks.append(dict(key=job["key"], shard_meta=meta_drops,
                           slot_pool_census=expected[job["key"]], ok=ok))
        if not ok:
            raise AssertionError(
                f"shard meta mismatch for {job['key']}: {meta_drops} != "
                f"{expected[job['key']]}")
    return checks


def wait_for_memory(min_free_virtual_gb: float, poll_s: float = 10.0,
                    max_wait_s: float = 600.0) -> dict:
    deadline = time.monotonic() + max_wait_s
    while True:
        sample = MEM.free_memory()
        if sample.get("free_virtual_gb", 0.0) >= min_free_virtual_gb:
            return sample
        if time.monotonic() > deadline:
            raise MemoryError(
                f"free virtual memory {sample.get('free_virtual_gb'):.2f} GB "
                f"< {min_free_virtual_gb} GB for > {max_wait_s:.0f}s")
        print(f"memory gate: free virtual "
              f"{sample.get('free_virtual_gb'):.2f} GB < "
              f"{min_free_virtual_gb:.0f} GB, waiting {poll_s:.0f}s",
              flush=True)
        time.sleep(poll_s)


def percentile(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def dist_from_hist(hist: list) -> dict:
    """p50/p99/max over the pooled per-tick rostered-count histogram."""
    total = sum(hist)
    if total == 0:
        return dict(p50=None, p99=None, max=None, ticks=0)
    out = {}
    cum = 0
    targets = dict(p50=0.50 * total, p99=0.99 * total)
    for count, n in enumerate(hist):
        if n == 0:
            continue
        cum += n
        for name, t in targets.items():
            if out.get(name) is None and cum >= t:
                out[name] = count
    mx = max(c for c, n in enumerate(hist) if n)
    return dict(p50=out.get("p50"), p99=out.get("p99"), max=int(mx),
                ticks=total)


def pct_ci(values: list) -> dict:
    vals = sorted(v for v in values if v == v)          # drop NaN
    if not vals:
        return dict(lo=None, hi=None)
    return dict(lo=round(percentile(vals, 0.025), 6),
                hi=round(percentile(vals, 0.975), 6))


class Aggregator:
    """§3 rule quantities + descriptive extras from per-run rows."""

    def __init__(self, rows: list):
        self.rows = rows

    def out_share(self, rows=None) -> float:
        rows = self.rows if rows is None else rows
        dropped = sum(r["dropped_total"] for r in rows)
        out = sum(r["dropped_out"] for r in rows)
        return out / dropped if dropped else float("nan")

    def in_zone_share(self, loading: str, rows=None) -> float:
        rows = self.rows if rows is None else rows
        dropped = sum(r["dropped_in"] for r in rows
                      if r["loading"] == loading)
        rostered = sum(r["rostered_in"] for r in rows
                       if r["loading"] == loading)
        return dropped / rostered if rostered else float("nan")

    def residual_share(self, rows=None) -> float:
        rows = self.rows if rows is None else rows
        dropped = sum(r["n_dropped_192"] for r in rows)
        rostered = sum(r["rostered"] for r in rows)
        return dropped / rostered if rostered else float("nan")

    def roster_dist(self, loading: str) -> dict:
        if not self.rows:
            return dist_from_hist([])
        width = max(len(r["tick_hist"]) for r in self.rows)
        hist = [0] * width
        for r in self.rows:
            if r["loading"] != loading:
                continue
            for count, n in enumerate(r["tick_hist"]):
                hist[count] += n
        return dist_from_hist(hist)

    def bootstrap(self, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED):
        rng = random.Random(seed)
        n = len(self.rows)
        outs, ins = [], {l: [] for l in LOADINGS}
        for _ in range(b):
            sample = [self.rows[rng.randrange(n)] for _ in range(n)]
            outs.append(self.out_share(sample))
            for l in LOADINGS:
                ins[l].append(self.in_zone_share(l, sample))
        return dict(out_share=pct_ci(outs),
                    in_zone_drop_share={l: pct_ci(ins[l])
                                        for l in LOADINGS})


def write_result(rows: list, checks: list, sample_doc: dict,
                 started_utc: str, finished_utc: str, elapsed_s: float,
                 failures: list, missing_raw: list) -> None:
    agg = Aggregator(rows)
    sample_sha = U.sha256_file(SAMPLE_PATH)
    n_sampled = sample_doc["sample_n"]
    status = "complete" if not failures and not missing_raw else "partial"
    out_share = agg.out_share()
    shares = {l: agg.in_zone_share(l) for l in LOADINGS}
    cis = agg.bootstrap()
    roster = {l: agg.roster_dist(l) for l in LOADINGS}
    totals = dict(
        n_runs=len(rows),
        dropped_total=sum(r["dropped_total"] for r in rows),
        dropped_in=sum(r["dropped_in"] for r in rows),
        dropped_out=sum(r["dropped_out"] for r in rows),
        rostered=sum(r["rostered"] for r in rows),
        rostered_in=sum(r["rostered_in"] for r in rows),
        rostered_out=sum(r["rostered_out"] for r in rows),
        n_dropped_192=sum(r["n_dropped_192"] for r in rows),
        ticks=sum(r["n_ticks"] for r in rows),
    )
    reconcile_bad = [r["key"] for r in rows if not r.get("reconcile_ok")]
    doc = dict(
        schema="wm_n1_kcensus_result/1",
        status=status,
        rule_note="REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md §1-§3",
        note=("counts and positions only; the §4 B / K=192 decision is NOT "
              "applied or stated here -- numbers only (owner instruction)"),
        sample=dict(path=str(SAMPLE_PATH), sha256=sample_sha,
                    created_utc=sample_doc["created_utc"],
                    seed=sample_doc["seed"], n=sample_doc["sample_n"]),
        coverage=dict(
            n_sampled=n_sampled, n_censused=len(rows),
            n_missing_raw=len(missing_raw),
            missing_raw=missing_raw,
            note=("the fixed 300-run sample is NOT re-drawn or shrunk; "
                  "numbers cover the censused subset only")),
        method=dict(
            replay=("frozen features.featurizer.parse_run imported "
                    "unmodified; scoped read-only json shim records every "
                    "consumed row; dropped = presence - rostered"),
            zone="A5 §2: lane_phys==2 anywhere; lanes 0-1 s_down in "
                 "[-800, 225.44); junction-internal IN-zone",
            k_census=CL.K_CENSUS, k_residual=CL.K_RESIDUAL,
            roster_share_denominator="rostered vehicle-ticks at K=128",
            bootstrap="run-cluster bootstrap, 1000 resamples, "
                      "seed 20261001, percentile 2.5/97.5",
        ),
        started_utc=started_utc, finished_utc=finished_utc,
        elapsed_s=round(elapsed_s, 1),
        totals=totals,
        rule_quantities=dict(
            out_share=round(out_share, 6),
            in_zone_drop_share={l: round(shares[l], 6) for l in LOADINGS},
        ),
        bootstrap_ci=cis,
        descriptive=dict(
            roster_concurrent_per_loading=roster,
            residual_drop_share_k192=round(agg.residual_share(), 6),
            residual_note=("n_dropped_192 / rostered@K128 vehicle-ticks, "
                           "pooled over the sample"),
        ),
        integrity=dict(
            raw_sha_verified=("per-run sha256(raw physics) == pack INDEX "
                              "input_sha256 checked before parsing"),
            spot_check_shard_meta=checks,
            per_run_reconcile=dict(
                rule="census dropped_total == SLOT_POOL_CENSUS "
                     "dropped_vehicle_ticks (pack_v2c meta "
                     "n_dropped_slot_steps)",
                n_checked=len(rows), n_ok=len(rows) - len(reconcile_bad),
                mismatches=reconcile_bad),
            audit="per-run open-path audit clean (no outcome/label/summary "
                  "file opened)",
        ),
        failures=failures,
        per_run=rows,
    )
    U.write_json_atomic(RESULT_JSON, doc)

    lines = [
        "# KCENSUS_RESULT (A5 §1-§3, read-only drop-location census)",
        "",
        f"- sample: `KCENSUS_SAMPLE.json` sha `{sample_sha[:16]}…` "
        f"({sample_doc['created_utc']}, seed {sample_doc['seed']}, "
        f"{sample_doc['sample_n']} runs); census elapsed "
        f"{elapsed_s:.0f} s ({started_utc} → {finished_utc})",
        f"- totals: rostered {totals['rostered']:,} vehicle-ticks, dropped "
        f"{totals['dropped_total']:,} "
        f"(in-zone {totals['dropped_in']:,} / out "
        f"{totals['dropped_out']:,}); "
        f"still dropped at K=192: {totals['n_dropped_192']:,}",
        "",
        "## §3 rule quantities (numbers only; no decision applied)",
        "",
        f"- `out_share` = {out_share:.6f} "
        f"({totals['dropped_out']:,} / {totals['dropped_total']:,} pooled "
        f"dropped vehicle-ticks outside the zone)",
        "- `in_zone_drop_share(L)` = dropped in-zone / rostered in-zone:",
        "",
        "| loading | share | dropped_in | rostered_in | bootstrap 95% CI |",
        "|---|---|---|---|---|",
    ]
    for l in LOADINGS:
        ci = cis["in_zone_drop_share"][l]
        lines.append(
            f"| {l} | {shares[l]:.6f} | "
            f"{sum(r['dropped_in'] for r in rows if r['loading'] == l):,} | "
            f"{sum(r['rostered_in'] for r in rows if r['loading'] == l):,} | "
            f"[{ci['lo']:.6f}, {ci['hi']:.6f}] |")
    oc = cis["out_share"]
    lines += [
        "",
        f"- `out_share` bootstrap 95% CI: [{oc['lo']:.6f}, {oc['hi']:.6f}]",
        "",
        "## Descriptive extras (not rule inputs)",
        "",
        "| loading | roster p50 | roster p99 | roster max | residual drop "
        "share @K=192 |",
        "|---|---|---|---|---|",
    ]
    res = agg.residual_share()
    for l in LOADINGS:
        d = roster[l]
        lines.append(f"| {l} | {d['p50']} | {d['p99']} | {d['max']} | "
                     f"(pooled) |")
    lines += [
        "", f"- pooled residual drop share at K=192: {res:.6f} "
        f"({totals['n_dropped_192']:,} / {totals['rostered']:,})",
        "",
        "## Integrity / reconcile",
        "",
        "- raw source fidelity: sha256(physics_steps.jsonl) == pack INDEX "
        "`input_sha256` verified before parsing for all "
        f"{len(rows)} censused runs",
        "- per run: census `dropped_total` == `SLOT_POOL_CENSUS.json` "
        f"`dropped_vehicle_ticks` (pack_v2c meta `n_dropped_slot_steps`): "
        f"{len(rows) - len(reconcile_bad)}/{len(rows)} ok",
    ]
    if missing_raw:
        lines += [
            "",
            f"## Missing raw sources ({len(missing_raw)} of {n_sampled} "
            f"sampled runs; numbers cover the {len(rows)} censused runs)",
            "",
        ]
        for m in missing_raw:
            lines.append(f"- `{m['key']}`: {m['reason']}")
    for c in checks:
        lines.append(f"- spot check {c['key']}: shard meta "
                     f"{c['shard_meta']} == slot-pool census "
                     f"{c['slot_pool_census']} -> "
                     f"{'ok' if c['ok'] else 'MISMATCH'}")
    if failures:
        lines.append(f"- failures: {len(failures)} "
                     f"({', '.join(f['key'] for f in failures[:5])}…)")
    lines.append("")
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_status(rows: list, n_total: int, reason: str) -> None:
    STATUS_MD.write_text(
        "# KCENSUS_STATUS (in progress / stopped at time limit)\n\n"
        f"- utc: {U.utc_now()}\n- reason: {reason}\n"
        f"- runs completed: {len(rows)} / {n_total}\n"
        f"- per-run rows: `KCENSUS_PER_RUN.json` (resume skips completed)\n"
        f"- next action: re-run "
        f"`python census_kmax_tools/census_run.py` in "
        f"output/wd01/wm_n1 to finish the remaining runs; the sample is "
        f"NOT shrunk\n",
        encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--deadline-min", type=float, default=40.0)
    args = ap.parse_args(argv)
    workers = min(int(args.workers), MAX_WORKERS)
    if workers < 1:
        raise ValueError("workers must be >= 1")

    sample_doc = U.read_json(SAMPLE_PATH)
    jobs = load_jobs()
    expected = load_expected()
    for j in jobs:
        j["expected_dropped"] = expected[j["key"]]
    print(f"sample: {len(jobs)} runs "
          f"(sha {U.sha256_file(SAMPLE_PATH)[:16]}…), workers={workers}, "
          f"deadline={args.deadline_min:.0f} min", flush=True)

    checks = spot_check_shard_meta(jobs, expected)
    for c in checks:
        print(f"spot-check {c['key']}: shard_meta={c['shard_meta']} "
              f"slot_pool_census={c['slot_pool_census']} ok={c['ok']}",
              flush=True)

    done = {}
    meta_by_key = {j["key"]: j for j in jobs}
    if PER_RUN_PATH.is_file():
        for r in U.read_json(PER_RUN_PATH).get("runs", []):
            if r.get("ok"):
                j = meta_by_key.get(r["key"], {})
                r.setdefault("loading", j.get("loading"))
                r.setdefault("split", j.get("split"))
                done[r["key"]] = r
        print(f"resume: {len(done)} previously completed runs", flush=True)

    # raw-presence partition (stat only, no file opened): runs whose raw
    # physics the featurizer read no longer exist are recorded as explicit
    # failures -- the fixed sample is never re-drawn or substituted.
    missing_raw = []
    todo = []
    for j in jobs:
        if j["key"] in done:
            continue
        physics = Path(j["run_dir"]) / "run" / "physics_steps.jsonl"
        if not physics.is_file():
            want = str(j.get("input_sha256") or "")[:16]
            missing_raw.append(dict(
                key=j["key"], loading=j["loading"],
                reason=("raw physics_steps.jsonl absent at the shard "
                        "meta input_physics path; pack INDEX input_sha256 "
                        f"{want}…; relocated candidates (chunk1_fix27 "
                        "re-runs) sha-mismatched; source not recoverable "
                        "from this workspace")))
        else:
            todo.append(j)
    if missing_raw:
        print(f"missing raw sources: {len(missing_raw)} "
              f"(recorded as failures; sample NOT re-drawn)", flush=True)
        for m in missing_raw:
            print(f"  [missing] {m['key']}", flush=True)

    wait_for_memory(MIN_FREE_VIRTUAL_GB)
    started_utc = U.utc_now()
    t0 = time.monotonic()
    deadline = t0 + args.deadline_min * 60.0
    stopped_early = None
    rows = list(done.values())

    if todo:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            pending = {}
            it = iter(todo)
            while True:
                now = time.monotonic()
                if now > deadline and not pending:
                    stopped_early = "deadline reached with runs remaining"
                    break
                while len(pending) < workers \
                        and time.monotonic() <= deadline:
                    try:
                        job = next(it)
                    except StopIteration:
                        break
                    pending[ex.submit(CL.census_job, job)] = job
                if not pending:
                    break
                done_futs = [f for f in pending if f.done()]
                if not done_futs:
                    if time.monotonic() > deadline and pending:
                        stopped_early = ("deadline reached while runs were "
                                         "in flight; in-flight runs were "
                                         "finished and recorded")
                    time.sleep(1.0)
                    continue
                for f in done_futs:
                    job = pending.pop(f)
                    row = f.result()
                    rows.append(row)
                    done[row["key"]] = row
                    if row["ok"]:
                        print(f"[ok] {row['key']} dropped={row['dropped_total']} "
                              f"in={row['dropped_in']} out={row['dropped_out']} "
                              f"k192={row['n_dropped_192']} "
                              f"reconcile={row['reconcile_ok']} "
                              f"({row['duration_s']}s)", flush=True)
                    else:
                        print(f"[error] {row['key']} {row['error']}",
                              flush=True)
                U.write_json_atomic(PER_RUN_PATH, dict(
                    schema="wm_n1_kcensus_per_run/1", runs=rows))
                if len(rows) % 25 < workers:
                    wait_for_memory(MIN_FREE_VIRTUAL_GB)

    U.write_json_atomic(PER_RUN_PATH, dict(
        schema="wm_n1_kcensus_per_run/1", runs=rows))
    elapsed = time.monotonic() - t0
    failures = [dict(key=r["key"], error=r["error"]) for r in rows
                if not r.get("ok")]
    n_uncensused = (len(missing_raw)
                    + len([r for r in rows if not r.get("ok")])
                    + len([j for j in todo if j["key"] not in done]))
    if stopped_early:
        write_status(rows + [dict(key=m["key"], ok=False,
                                  error="missing raw") for m in missing_raw],
                     len(jobs), stopped_early)
        print(f"stopped early: {len(rows)}/{len(jobs)} recorded; "
              f"KCENSUS_STATUS.md written", flush=True)
        return 2

    ok_rows = [r for r in rows if r.get("ok")]
    write_result(ok_rows, checks, sample_doc, started_utc, U.utc_now(),
                 elapsed, failures, missing_raw)
    print(f"census finished: {len(ok_rows)} censused, "
          f"{len(failures)} failed, {len(missing_raw)} missing raw "
          f"({n_uncensused} of {len(jobs)} not censused) in {elapsed:.0f}s",
          flush=True)
    print(f"wrote {RESULT_JSON.name} and {RESULT_MD.name}", flush=True)
    return 0 if not failures and not missing_raw else 3


if __name__ == "__main__":
    raise SystemExit(main())
