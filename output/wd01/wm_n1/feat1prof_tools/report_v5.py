"""FEAT1 v5 close-out: FEAT1_REPORT.md + FEAT1_REPORT.json.

Reads the v5 evidence set (shards INDEX, consolidation/retries/assertion
receipts, quarantine manifests v2/v3/v4, LAUNCH/prelaunch, non-featurized
list, pack totals, upload manifest sha) and writes:

* ``<workspace>/output/wd01/wm_n1/FEAT1_REPORT.md``  (committed evidence)
* ``D:/rd-staging/wmn1_feat/FEAT1_REPORT.json``      (machine report)

Read-only on shards/pack.  Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):

  python -X utf8 feat1prof_tools/report_v5.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))
BASE = Path("D:/rd-staging/wmn1_feat")
LOGS5 = BASE / "logs_v5"


def _jload(p: Path):
    raw = Path(p).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    # keep a portable UTF-8 copy next to any UTF-16 Tee-Object original
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        u8 = Path(str(p) + ".utf8")
        if not u8.is_file():
            u8.write_text(text, encoding="utf-8")
    return json.loads(text)


def main() -> int:
    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override(str(BASE))

    index = _jload(BASE / "shards_v2" / "INDEX.json")["runs"]
    pack = _jload(BASE / "pack_v2" / "INDEX.json")
    manifest = _jload(BASE / "upload_manifest.json")
    cons = _jload(LOGS5 / "consolidation.json")
    assertion = _jload(LOGS5 / "assertion_v5.json")
    retries = _jload(LOGS5 / "retries_v5.json")["retries"]
    launch = _jload(LOGS5 / "LAUNCH.json")
    prelaunch = _jload(LOGS5 / "prelaunch_resources.json")
    nonfeat = _jload(WM_N1 / "inventory" / "non_featurized_traincal.json")
    q2 = _jload(BASE / "quarantine_v2" / "20260930T224632Z" / "QUARANTINE.json")
    q3a = _jload(BASE / "quarantine_v3" / "20261001T013855Z" / "QUARANTINE.json")
    q3b = _jload(BASE / "quarantine_v3" / "20261001T030409Z" / "QUARANTINE.json")
    q4 = _jload(BASE / "quarantine_v4" / "20261001T042854Z" / "QUARANTINE.json")

    by_batch = Counter()
    by_split = Counter()
    by_policy = Counter()
    ticks_by_batch: Counter = Counter()
    bytes_by_batch: Counter = Counter()
    for rec in index.values():
        b = rec.get("batch") or rec["chunk"]
        by_batch[b] += 1
        by_split[rec.get("split")] += 1
        by_policy[rec.get("policy")] += 1
        ticks_by_batch[b] += int(rec.get("n_ticks") or 0)
        bytes_by_batch[b] += int(rec.get("total_bytes") or 0)
    n_runs = len(index)
    n_ticks = sum(ticks_by_batch.values())
    shard_bytes = sum(bytes_by_batch.values())

    times = []
    for k in range(6):
        lp = LOGS5 / f"sub{k}.log"
        times.append((lp.stat().st_ctime, lp.stat().st_mtime))
    t0v5 = datetime.fromtimestamp(min(t[0] for t in times), tz=timezone.utc)
    t1v5 = datetime.fromtimestamp(max(t[1] for t in times), tz=timezone.utc)
    v5_wall_min = round((t1v5 - t0v5).total_seconds() / 60.0, 1)

    report = dict(
        schema="wm_n1_feat1_report/1",
        created_utc=U.utc_now(),
        branch="zcode/feat1run5 (== zcode/feat1run4 ae7550c + close-out)",
        workspace="E:/rd-wt/feat1run5",
        corpus=dict(
            n_runs=n_runs, n_ticks=n_ticks,
            shard_bytes=shard_bytes,
            pack_bytes=pack["totals"]["bytes"],
            n_decision=pack["totals"]["n_decision"],
            splits=dict(by_split), by_batch=dict(by_batch),
            by_policy=dict(by_policy),
            ticks_by_batch=dict(ticks_by_batch),
            k_max=pack["k_max"]),
        v5=dict(
            launched_utc=t0v5.strftime("%Y-%m-%dT%H:%M:%SZ"),
            finished_utc=t1v5.strftime("%Y-%m-%dT%H:%M:%SZ"),
            wall_minutes=v5_wall_min,
            n_subsets=6, n_remaining=launch["n_remaining"],
            subset_sizes=[s["n_runs"] for s in launch["subsets"]],
            workers_per_process=1, mode="sequential in-process "
                                       "(_run_batch workers<=1: no pool)",
            idea_fast=True, force_used=False,
            prelaunch=prelaunch,
            pids=[int((LOGS5 / f"sub{k}.pid").read_text().strip())
                  for k in range(6)],
            written=assertion["subset_end_lines"],
            excluded=len(retries)),
        assertion=assertion,
        exclusions=[dict(key=r["key"], error=r["retry_error"],
                         detail=r["retry_detail"]) for r in retries],
        non_featurized=nonfeat,
        j4_relabel=dict(
            runs_with_amendment2_flag=cons["j4_relabel_runs"],
            road_end_tokens_relabelled=cons["j4_relabelled_total"],
            source="IDEA_A_DONE.json receipts (consolidation.json)"),
        quarantine=dict(
            v2=dict(stamp=q2["utc"], n_quarantined=q2["n_quarantined"],
                    n_kept_complete=q2["n_kept_complete"],
                    n_kept_idea_a_pending=q2["n_kept_idea_a_pending"]),
            v3=[dict(stamp=q3a["created_utc"], scanned=q3a["n_scanned"],
                     ok=q3a["n_ok"], moved=q3a["n_moved"],
                     keys=[e["key"] for e in q3a["entries"]]),
                dict(stamp=q3b["created_utc"], scanned=q3b["n_scanned"],
                     ok=q3b["n_ok"], moved=q3b["n_moved"],
                     keys=[e["key"] for e in q3b["entries"]])],
            v4=dict(stamp=q4["created_utc"], scanned=q4["n_scanned"],
                    ok=q4["n_ok"], moved=q4["n_moved"],
                    keys=[e["key"] for e in q4["entries"]])),
        verify=dict(
            corpus_verify=U.sanitize(_jload(LOGS5 / "verify_pack_v2.json")),
            idea_a_verify=U.sanitize(_jload(LOGS5 / "verify_idea_a_v2.json")),
            manifest=dict(path=str(BASE / "upload_manifest.json"),
                          sha256=U.sha256_file(
                              BASE / "upload_manifest.json"),
                          n_runs=manifest["n_runs"],
                          splits={k: len(v) for k, v in
                                  manifest["splits"].items()},
                          refused=manifest["refused"]),
            pack_sums_sha256=U.sha256_file(BASE / "pack_v2" / "SHA256SUMS")),
        disclosure_chain=[
            "1. first attempt (policy-keyed shard dirs) ABORTED: "
            "replicated-arm + cross-batch key collisions (fixed1/fixed2, "
            "switching1..3; chunk2/3 vs truearms) -> shard key redefined to "
            "batch + run-dir arm suffix, policy demoted to metadata "
            "(commit bf5d81a); superseded shards v1 tree kept at "
            "D:/rd-staging/wmn1_feat/shards (untouched)",
            "2. inventory refresh df30b17 -> restart 6 workers, then "
            "owner-approved 6->12 restart (commit 4a7c343 + bcf1ddc, "
            "logs_v3, launcher 14980 / featurizer 32340)",
            "3. 09:28 crash #1: BrokenProcessPool after a worker raised "
            "wall_tokens_v04.WallTokenError the parent could not unpickle "
            "(ModuleNotFoundError: wall_tokens_v04 in the executor result "
            "reader); 1825/2463 done; parent sys.path fix + per-run "
            "[start]/[end]/[error] logging (commit 573a02a)",
            "4. ~10:53-11:01 crash #2, SAME signature: the first fix added "
            "only the wm_n1/wd01 roots but wall_tokens_v04 lives in "
            "output/wd01/partB_v0data; featurize_all parent now imports "
            "features.actions_adapter too + per-run exclusion-and-continue "
            "(commit ae7550c); 1-worker repro: 9/10 in-flight written, "
            "w8658m2600p10t360__w-lo-early reproducibly raises WallTokenError",
            "5. 11:15 v4b relaunch (10 workers, featurizer PID 1132): the "
            "pool STALLED (workers idle 67 min); the integrator stopped PID "
            "1132 + its children WITH MAIN'S AUTHORIZATION (no run was "
            "corrupted: shards are per-run dirs with receipt-gated writes)",
            "6. final finish WITHOUT a pool: quarantine v4 (1859 scanned / "
            "1850 verified / 9 moved), 613 remaining runs split into 6 "
            "DISJOINT subset parquets, 6 independent detached featurize-all "
            "processes --workers 1 (sequential in-process path, no pool), "
            "--idea-fast, BELOW_NORMAL, per-run [start]/[end]/[error] logs + "
            "PID files (logs_v5); 595 written, 18 errored, all 18 re-run "
            "ALONE once, all 18 errored again -> excluded; count assertion "
            "2476-13-18=2445 == verified 2445; pack_v2 + corpus verify "
            "ok:true + corpus.idea_a verify ok:true + upload manifest",
            "7. --idea-fast byte-identity evidence: corpus tests fast==slow "
            "(48 seeded scenes, commit b1d9598); v4-resume 3/3 TRAIN runs "
            "61 sha comparisons (logs_v4/byte_identity_resume_20261001.txt); "
            "v5 runs-after-first-1824: 3/3 v5-written TRAIN runs (incl. J4 "
            "wall arm w8706m2400p40t360__w-mid-late) re-derived through the "
            "SLOW reference path, 61 sha comparisons ALL equal "
            "(logs_v5/byte_identity_v5_slowcheck.txt)"],
        artifact_paths=dict(
            shards="D:/rd-staging/wmn1_feat/shards_v2",
            pack="D:/rd-staging/wmn1_feat/pack_v2",
            manifest="D:/rd-staging/wmn1_feat/upload_manifest.json",
            logs_v5="D:/rd-staging/wmn1_feat/logs_v5",
            quarantine_v2="D:/rd-staging/wmn1_feat/quarantine_v2/"
                          "20260930T224632Z",
            quarantine_v3="D:/rd-staging/wmn1_feat/quarantine_v3",
            quarantine_v4="D:/rd-staging/wmn1_feat/quarantine_v4/"
                          "20261001T042854Z"),
    )
    (BASE / "FEAT1_REPORT.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")

    md = []
    md.append("# FEAT1 v5 report — WM-N1 featurization complete\n")
    md.append(f"Generated {report['created_utc']} on branch "
              f"`zcode/feat1run5` (workspace `E:/rd-wt/feat1run5`), "
              f"main-authorized finish without a process pool.\n")
    md.append("## Corpus\n")
    md.append(f"- verified shards: **{n_runs}** (assertion "
              f"{assertion['arithmetic']})")
    md.append(f"- ticks: **{n_ticks:,}** | decision rows: "
              f"{pack['totals']['n_decision']:,} | k_max: {pack['k_max']}")
    md.append(f"- bytes: pack {pack['totals']['bytes']:,} "
              f"({pack['totals']['bytes']/1e9:.2f} GB, all 15 arrays + "
              f"outcomes) | shard v1 receipts {shard_bytes:,} (12 v1 arrays "
              f"+ json/meta/outcomes; the 5 Idea A arrays are receipted "
              f"separately: {cons['idea_a_arrays_bytes']:,} B)")
    md.append(f"- splits: {dict(by_split)}")
    md.append(f"- by batch: {dict(by_batch)}")
    md.append(f"- by policy: {dict(by_policy)}\n")
    md.append("## v5 final run (6 x sequential, no pool)\n")
    md.append(f"- launched {report['v5']['launched_utc']} -> finished "
              f"{report['v5']['finished_utc']} (wall {v5_wall_min} min; "
              f"subset sizes {report['v5']['subset_sizes']})")
    md.append(f"- PIDs {report['v5']['pids']}, --workers 1 each (in-process "
              f"sequential path `corpus/featurize_all.py:_run_batch`), "
              f"--idea-fast, BELOW_NORMAL, force never used, "
              f"no shard overwritten")
    md.append(f"- prelaunch gates: {prelaunch['gates']}\n")
    md.append("## J4 amendment-2 relabel\n")
    md.append(f"- {cons['j4_relabel_runs']} wall-arm runs flagged; "
              f"{cons['j4_relabelled_total']} road-end tokens relabelled "
              f"(IDEA_A_DONE receipts)\n")
    md.append("## Non-featurized runs (13, never readable)\n")
    for r in nonfeat:
        md.append(f"- `{r['batch']}/{r['run']}` ({r['split']}): "
                  f"{r['reasons']}" +
                  (", smoke" if r.get("is_smoke") else ""))
    md.append("")
    md.append("## Excluded runs (18, errored twice: subset run + alone retry)\n")
    for r in retries:
        md.append(f"- `{r['key']}`: {r['retry_error']}: {r['retry_detail']}")
    md.append("")
    md.append("## Quarantine ledger (MOVE only, never delete/overwrite)\n")
    md.append(f"- v2 `{q2['utc']}`: moved {q2['n_quarantined']}, kept "
              f"complete {q2['n_kept_complete']}, kept Idea-A-pending "
              f"{q2['n_kept_idea_a_pending']}")
    for q in (q3a, q3b):
        md.append(f"- v3 `{q['created_utc']}`: scanned {q['n_scanned']}, ok "
                  f"{q['n_ok']}, moved {q['n_moved']}: "
                  + ", ".join("`" + e["key"] + "`" for e in q["entries"]))
    md.append(f"- v4 `{q4['created_utc']}`: scanned {q4['n_scanned']}, ok "
              f"{q4['n_ok']}, moved {q4['n_moved']} (all partial in-flight "
              f"dirs of chunk4/w8660m2700p10t300 + walls_pilot w8660):")
    for e in q4["entries"]:
        md.append(f"  - `{e['key']}`")
    md.append("")
    md.append("## Verify + manifest\n")
    md.append("```json\n" +
              json.dumps(_jload(LOGS5 / "verify_pack_v2.json"), indent=1) +
              "\n```")
    md.append("```json\n" +
              json.dumps(_jload(LOGS5 / "verify_idea_a_v2.json"), indent=1) +
              "\n```")
    md.append(f"- manifest `D:/rd-staging/wmn1_feat/upload_manifest.json` "
              f"sha256 `{report['verify']['manifest']['sha256']}` "
              f"({manifest['n_runs']} runs, splits "
              f"{report['verify']['manifest']['splits']}, refused [])")
    md.append(f"- `pack_v2/SHA256SUMS` sha256 "
              f"`{report['verify']['pack_sums_sha256']}`\n")
    md.append("## Disclosure chain\n")
    for line in report["disclosure_chain"]:
        md.append(f"- {line}")
    md.append("")
    md.append("## Artifacts\n")
    for k, v in report["artifact_paths"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    (WM_N1 / "FEAT1_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"FEAT1_REPORT.md -> {WM_N1 / 'FEAT1_REPORT.md'}")
    print(f"FEAT1_REPORT.json -> {BASE / 'FEAT1_REPORT.json'}")
    print(json.dumps(dict(n_runs=n_runs, n_ticks=n_ticks, by_split=by_split,
                          by_batch=dict(by_batch), v5_wall_min=v5_wall_min),
                      indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
