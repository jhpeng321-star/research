"""Bounded local receipt collector; preserves foreign edits, commits only ES artifacts.

No rollouts or dispatches: safe to run while the registered ES/probes finish.
One exclusive PID receipt prevents duplicate collectors. Never pushes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

TREE = Path(__file__).resolve().parents[2]
RESULTS = TREE / "docs/paper_results"
RECEIPTS = RESULTS / "wmfix_es_receipts"
REPORT = RESULTS / "WMFIX_ES.md"
GPU = "/root/autodl-tmp/c4/wmfixmerge20261031/es_cpu_v3"
CPU = "/home/ubuntu/wmfix_es"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv, check=True):
    result = subprocess.run(argv, cwd=TREE, text=True, capture_output=True, timeout=30)
    if check and result.returncode:
        raise RuntimeError("command failed: " + repr(argv) + " " + result.stderr)
    return result


def pull():
    brought = []
    specs = [("cpuQ", CPU + "/probe_k8", "PROBE_f*.json", "cpuQ_"),
             ("gpuA", GPU + "/probe_k8", "PROBE_f*.json", "gpuA_"),
             ("gpuA", GPU + "/es_run_full",
              "GEN_*.json ES_FINAL.json FINAL.json ES_ABORT.json SCORING_READY.json VERDICT_PREFLIGHT_FINAL.json",
              "")]
    for host, root, patterns, prefix in specs:
        paths = " ".join(root + "/" + pattern for pattern in patterns.split())
        listing = command(["ssh", "-o", "ConnectTimeout=8", host, "ls " + paths + " 2>/dev/null"], False)
        if listing.returncode == 255:
            raise RuntimeError("SSH unavailable: " + host)
        for remote in listing.stdout.split():
            name = remote.rsplit("/", 1)[-1]
            if not re.fullmatch(r"(?:PROBE_f[0-7]|GEN_\d{3}|ES_FINAL|FINAL|ES_ABORT|SCORING_READY|VERDICT_PREFLIGHT_FINAL)\.json", name):
                raise ValueError("unexpected receipt filename")
            target = RECEIPTS / (prefix + name)
            if target.exists():
                continue
            command(["scp", "-q", host + ":" + remote, str(target)])
            json.loads(target.read_text())  # reject partial/non-JSON receipts
            brought.append(target.name)
    return brought


def load(name):
    path = RECEIPTS / name
    return json.loads(path.read_text()) if path.is_file() else None


def probe(host):
    docs = [load(f"{host}_PROBE_f{i}.json") for i in range(8)]
    if not all(docs):
        return None, sum(doc is not None for doc in docs)
    if {d["future_index"] for d in docs} != set(range(8)) or any(d["n_futures"] != 8 or d["threads"] != 1 for d in docs):
        raise ValueError("incomplete/non-K8 probe")
    return dict(cpu_seconds=sum(d["group_seconds"] for d in docs),
                parallel_wall=max(d["group_seconds"] + d["runtime_load_seconds"] for d in docs),
                rss_min=min(d["max_rss_mb"] for d in docs), rss_max=max(d["max_rss_mb"] for d in docs)), 8


def render(template):
    q, nq = probe("cpuQ")
    a, na = probe("gpuA")
    if q and a:
        tq = 12 * q["cpu_seconds"] / 30 / 60
        ta = 12 * a["cpu_seconds"] / 18 / 60
        combined = 12 / (30 / q["cpu_seconds"] + 18 / a["cpu_seconds"]) / 60
        throughput = ["## Throughput — genuine K8, entry path/mix ON, threads=1",
            "- Each host: 8 sticky-future processes, each two paired arms x {120,300} s.",
            f"- cpuQ: {q['cpu_seconds']:.1f} CPU-worker-s/K8 group; 8-way wall ~{q['parallel_wall']:.1f} s;",
            f"  peak RSS {q['rss_min']:.0f}-{q['rss_max']:.0f} MB/process (measured).",
            f"- gpuA CPUs: {a['cpu_seconds']:.1f} CPU-worker-s/group; wall ~{a['parallel_wall']:.1f} s;",
            f"  peak RSS {a['rss_min']:.0f}-{a['rss_max']:.0f} MB/process (measured).",
            f"- Extrapolated 12-FIT-group K8 panel: cpuQ N=30 {tq:.1f} min; gpuA N=18 {ta:.1f} min;",
            f"  combined {combined:.1f} min ({combined / 28.5:.2f}x requested GPU reference 28.5 min).",
            "  Linear scaling estimate, not a measured full parallel-panel run.",
            "- Prior probe was K1; its ~33 min/52-world estimate was invalid and withdrawn.",
            "  GPU pilot code is also K1; 28.5 min is not measured K8 GPU throughput.", ""]
    else:
        throughput = ["## Throughput — genuine K8 measurement in progress",
            f"- Completed futures: cpuQ {nq}/8; gpuA {na}/8. Eight sticky processes per host.",
            "- Each process: two paired arms x {120,300} s, entry path/mix ON, threads=1.",
            "- Prior K1 measurements: cpuQ 1766.6 s/1372 MB; gpuA 1920.0 s/1775 MB.",
            "  Prior K8/~33-min aggregate claims withdrawn; full K8 receipts pending.", ""]
    before, rest = template.split("## Throughput", 1)
    protocol = "## Full-corpus" + rest.split("## Full-corpus", 1)[1].split("## ES curve", 1)[0]
    gens = [json.loads(p.read_text()) for p in sorted(RECEIPTS.glob("GEN_*.json"))]
    curve = "; ".join(f"g{g['gen']}={min(g['candidates'].values(), key=lambda c: c['fitness'])['paired_J_MAE']:.2f}"
                      for g in gens) or "pending generation 0"
    source = "; ".join(f"g{g['gen']}={min(g['candidates'].values(), key=lambda c: c['fitness'])['source_contrast_MAE']:.2f}"
                       for g in gens) or "pending"
    final, abort, ready = load("ES_FINAL.json"), load("ES_ABORT.json"), load("SCORING_READY.json")
    status = "ABORTED: " + abort["blocker"] if abort else "COMPLETE" if final else "RUNNING (gpuA driver PID 195604)"
    preflight = load("VERDICT_PREFLIGHT_FINAL.json")
    blockers = "; ".join(preflight["blockers"]) if preflight else "accept panel, C1 cohort/runner, candidate quartet completion"
    tail = ["## ES curve / verdict / recommendation", f"- Calibration: {status}; `es_cpu_v3/es_run_full`, cpuQ `es_run_full`.",
        "- Curve (generation winner by objective, TRAIN paired J MAE veh-s): " + curve + ".",
        "- Source contrast MAE reported separately: " + source + ".",
        f"- Scoring-ready selected checkpoint: {'YES' if ready else 'pending'}; 20 CPUQ + 3 receipt checks PASS.",
        "- Verdict INCOMPLETE / NOT ADOPTABLE; unchanged A vs WM2 AND entry, B, C1/accept.",
        "- Missing registered inputs: " + blockers + ".",
        "- MAIN: bind C1/accept inputs and allocate A/B GPU slot after checkpoint readiness.",
        "- Recommendation: retain WM2 until registered independent A/B/C1/accept PASS;",
        "  120 s TRAIN calibration alone does not establish 300/600 s response quality."]
    best = final["best"] if final else gens[-1]["best_overall"] if gens else None
    if best:
        tail.insert(4, f"- Selected/best-so-far g{best['gen']}/c{best['cand']}: paired J MAE {best['paired_J_MAE']:.2f} veh-s.")
    lines = (before + "\n".join(throughput) + "\n" + protocol).rstrip().splitlines() + [""] + tail
    if len(lines) > 60:
        raise ValueError("WMFIX_ES.md would exceed 60 lines")
    return "\n".join(lines) + "\n"


def commit_receipts(owned):
    status = command(["git", "status", "--porcelain", "--untracked-files=all"]).stdout
    intended = []
    for line in status.splitlines():
        name = line[3:].replace("\\", "/")
        if name not in owned or sha(TREE / name) != owned[name]:
            print("Foreign changes present; receipts preserved, automatic commit deferred: " + name, flush=True)
            return
        intended.append(name)
    if not status.strip():
        return
    print(command(["git", "diff", "--", "docs/paper_results/WMFIX_ES.md"]).stdout, flush=True)
    print(command(["git", "log", "--oneline", "-10"]).stdout, flush=True)
    command(["git", "diff", "--check"])
    command(["git", "add", "--", *intended])
    print(command(["git", "diff", "--cached", "--stat"]).stdout, flush=True)
    print(command(["git", "commit", "-m", "S-WMFIX-ES: collect completed CPU calibration receipts"]).stdout, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=21600.)
    args = parser.parse_args()
    RECEIPTS.mkdir(exist_ok=True)
    template, expected = REPORT.read_text(encoding="utf-8"), sha(REPORT)
    owned = {REPORT.relative_to(TREE).as_posix(): expected}
    pid = TREE / "_out/es_collect_PID.json"
    with pid.open("x") as stream:
        json.dump(dict(pid=os.getpid(), started=time.time(), report_sha256=expected), stream)
    start = time.monotonic()
    while True:
        try:
            brought = pull()
            for name in brought:
                path = RECEIPTS / name
                owned[path.relative_to(TREE).as_posix()] = sha(path)
            if sha(REPORT) != expected:
                raise ValueError("WMFIX_ES.md changed externally; refusing overwrite")
            text = render(template)
            if text != REPORT.read_text(encoding="utf-8"):
                REPORT.write_text(text, encoding="utf-8")
                expected = sha(REPORT)
                owned[REPORT.relative_to(TREE).as_posix()] = expected
            if args.commit:
                commit_receipts(owned)
            print(json.dumps(dict(collected=brought, elapsed=round(time.monotonic() - start))), flush=True)
            probes_done = probe("cpuQ")[1] == probe("gpuA")[1] == 8
            finished = load("ES_ABORT.json") or (load("ES_FINAL.json") and load("SCORING_READY.json")
                                                and load("VERDICT_PREFLIGHT_FINAL.json"))
            if not args.watch or (probes_done and finished):
                return
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(str(exc), flush=True)
            if not args.watch:
                raise
        if time.monotonic() - start >= args.max_seconds:
            print("collector wall cap; receipts retained, no completion claim", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
