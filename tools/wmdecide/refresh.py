"""Bounded local CPU report refresh; reads existing producer output JSON only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

# These apply before NumPy/evaluator imports, including an unattended fresh process.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

from .__main__ import credibility
from .publish import publish
from .rbeval_adapter import safe_path, sha256
from .report import write_json
from .snapshot import ROOT, TASK, snapshot

PRODUCERS = (
    ("wm2", "gpuR", ROOT),
    ("full_v1", "gpuA", ROOT+"/int/WM-PHYS-FULL-v1"),
    ("full_v3", "gpuA", ROOT+"/int/WM-PHYS-FULL-v3-gpu"),
    ("a2", "gpuA", ROOT+"/runa/WM-PHYS-A2"),
    ("b2", "gpuR", ROOT+"/runb/WM-PHYS-B2"),
    ("c2", "gpuA", ROOT+"/runa/WM-PHYS-C2"),
)
BEGIN = "<!-- wmdecide-refresh:start -->"
END = "<!-- wmdecide-refresh:end -->"


def handoff_text(old, relative, receipt):
    """Replace only our bounded latest-publication block; retain all prior work."""
    if old.count(BEGIN) != old.count(END) or old.count(BEGIN) > 1:
        raise ValueError("Ambiguous handoff markers")
    lines = [BEGIN, "## Latest automated saved-output credibility refresh", "",
             f"- Collected/scored {receipt['completed_utc']}; start at `{relative}/INDEX.md`.",
             "- All scores use validated complete K8 worlds; per-candidate matched WM2",
             "  sections use exactly the candidate's available common worlds.",
             "- Read-only collection; CPU-only analysis; no inference/dispatch/process control.",
             "- Diagnostic only; EVAL/HDV60 adoption gate remains authoritative."]
    for name, result in receipt["models"].items():
        lines.append(f"- {name}: {result['status']}; "
                     f"{len(result.get('worlds', []))} scored worlds.")
    block = "\n".join(lines+[END, ""])
    if BEGIN in old:
        start, tail = old.split(BEGIN)
        _, tail = tail.split(END)
        return start+block+tail.lstrip("\n")
    title, rest = old.split("\n", 1)
    return title+"\n\n"+block+"\n"+rest.lstrip("\n")


def complete_signature(files):
    """Cache by identities and complete-K8 task bytes; partial tasks earn no score."""
    worlds = {}
    for name in files:
        task = Path(name).name
        if TASK.fullmatch(task):
            seed, future = task[:-5].split("_k")
            worlds.setdefault(seed, set()).add(int(future))
    complete = {seed for seed, futures in worlds.items() if futures == set(range(8))}
    selected = {name: digest for name, digest in files.items()
                if Path(name).name == "IDENTITY.json" or
                (TASK.fullmatch(Path(name).name) and Path(name).name.split("_k")[0] in complete)}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def refresh_once(eval_root, scratch, publication, previous=None):
    """Collect once, validate, score available worlds and matched candidate cohorts."""
    scratch, publication = safe_path(scratch), safe_path(publication)
    if scratch.exists() or publication.exists():
        raise ValueError("Refresh paths must both be NEW")
    if not scratch.parent.is_dir() or not publication.parent.is_dir():
        raise ValueError("Refresh parents must exist")
    scratch.mkdir()
    receipt = dict(schema="wmdecide-refresh/1", started_utc=datetime.now(timezone.utc).isoformat(),
                   models={}, refresh_sha256=sha256(Path(__file__)))
    reports, audits, inputs = {}, {}, {}
    for name, host, root in PRODUCERS:
        state = receipt["models"][name] = dict(host=host, root=root)
        dest = scratch / (name+"_snapshot")
        try:
            collected = snapshot(host, root, dest)
            state.update(snapshot=str(dest), snapshot_sha256=sha256(dest/"SNAPSHOT.json"),
                         tasks=sum(bool(TASK.fullmatch(Path(p).name)) for p in collected["files"]))
            inputs[name] = sorted(p for p in dest.iterdir() if p.is_dir())
            signature = complete_signature(collected["files"])
            cached = (previous or {}).get(name, {})
            if signature == cached.get("signature") and "report" in cached:
                report = safe_path(cached["report"])
                manifest = json.loads((report/"FILES.json").read_text(encoding="utf-8"))
                for member, digest in manifest.items():
                    if Path(member).name != member or sha256(report/member) != digest:
                        raise ValueError("Cached score output checksum changed")
                audit = json.loads((report/"audit.json").read_text(encoding="utf-8"))
                state["reused_report"] = "identical_complete_K8_bytes_and_identity"
            else:
                report = scratch / (name+"_report")
                audit = credibility(eval_root, inputs[name], report, allow_partial=True)
            state.update(status=audit["status"], worlds=audit["audit"]["scored_worlds"],
                         signature=signature, report=str(report),
                         manifest_sha256=sha256(report/"FILES.json"))
            if state["worlds"]:
                reports[name], audits[name] = report, audit
        except Exception as error:
            # Retain all collected bytes and incomplete outputs. A later poll is
            # a new read, never an inference retry or a changed scientific task.
            state.update(status="UNAVAILABLE_OR_INVALID", error=f"{type(error).__name__}: {error}")
    if not reports:
        write_json(scratch/"REFRESH_FAILED.json", receipt)
        raise ValueError("No validated complete K8 report available; preserve previous handoff")
    publication.mkdir()
    publish(list(reports.values()), publication/"available")
    lines = ["# Saved-output credibility refresh", "",
             "[Available-evidence sections](available/INDEX.md) have model-specific world sets.",
             "Use the matched sections below for model comparisons. All results are diagnostic;",
             "the original distributional/HDV60 adoption gate remains authoritative.",
             "Unchanged complete ensembles reuse hash-verified prior scores; their original",
             "report audits retain the original snapshot's partial-task counts.", ""]
    for name in reports:
        if name == "wm2" or "wm2" not in reports:
            continue
        worlds = sorted(set(audits["wm2"]["audit"]["scored_worlds"]) &
                        set(audits[name]["audit"]["scored_worlds"]))
        if not worlds:
            continue
        matched_sources = [receipt["models"][source]["signature"] for source in ("wm2", name)]
        cached = (previous or {}).get(name, {})
        if cached.get("matched_sources") == matched_sources and cached.get("matched_worlds") == worlds:
            pair = [safe_path(p) for p in cached["matched_reports"]]
        else:
            pair = []
            for source in ("wm2", name):
                if audits[source]["audit"]["scored_worlds"] == worlds:
                    pair.append(reports[source])
                else:
                    out = scratch / (name+"_matched_"+source)
                    credibility(eval_root, inputs[source], out, allow_partial=True, worlds=worlds)
                    pair.append(out)
        publish(pair, publication/("matched_"+name), matched=True)
        receipt["models"][name].update(matched_worlds=worlds, matched_sources=matched_sources,
                                      matched_reports=[str(p) for p in pair])
        lines.append(f"- [{name} vs WM2](matched_{name}/INDEX.md): exact worlds {worlds}.")
    for name, state in receipt["models"].items():
        lines.append(f"- {name}: {state['status']}; {len(state.get('worlds', []))} complete worlds.")
        if "error" in state:
            lines.append(f"  Collection/validation detail: {state['error']}")
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (publication/"INDEX.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(publication/"REFRESH.json", receipt)
    write_json(publication/"FILES.json", {str(p.relative_to(publication)).replace("\\", "/"): sha256(p)
                                        for p in sorted(publication.rglob("*")) if p.is_file()})
    return receipt


def git(repo, *args):
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                            encoding="utf-8", errors="backslashreplace", check=True)
    return result.stdout.strip()


def require_clean_branch(repo):
    if git(repo, "branch", "--show-current") != "c4/rb-decide":
        raise ValueError("Refresh commits belong only to c4/rb-decide")
    if git(repo, "status", "--porcelain"):
        raise ValueError("Unfamiliar worktree changes; preserve them and stop automatic publication")


def commit_publication(repo, relative):
    # Inspect all required context; stage only our new artifact and handoff.
    inspected = {"status": git(repo, "status", "--short"), "diff": git(repo, "diff"),
                 "log": git(repo, "log", "--oneline", "-10")}
    if git(repo, "diff", "--cached", "--name-only"):
        raise ValueError("Existing staged work; do not include it in refresh commit")
    if git(repo, "branch", "--show-current") != "c4/rb-decide":
        raise ValueError("Branch changed during refresh")
    if git(repo, "diff", "--name-only") != "HANDOFF.md":
        raise ValueError("Unfamiliar tracked changes during refresh")
    others = git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    if not others or any(not name.startswith(relative+"/") for name in others):
        raise ValueError("Unfamiliar untracked changes during refresh")
    git(repo, "diff", "--check")
    git(repo, "add", "--", "HANDOFF.md", relative)
    inspected["commit"] = git(repo, "commit", "-m", "S-RB-DECIDE: refresh saved K8 credibility "+
                              Path(relative).name)
    return inspected


def run(repo, eval_root, scratch, claim, until, interval=1200, previous_receipt=None):
    """One exclusive collector; no PID signalling, deletion or inference ownership."""
    import psutil

    repo, scratch, claim = safe_path(repo).resolve(), safe_path(scratch), safe_path(claim)
    if until.tzinfo is None or interval < 60:
        raise ValueError("Use an explicit timezone and an interval of at least60 seconds")
    if until.timestamp() <= time.time():
        raise ValueError("Deadline already passed")
    require_clean_branch(repo)
    if scratch.exists() or not scratch.parent.is_dir() or not claim.parent.is_dir():
        raise ValueError("New scratch and existing scratch/claim parents required")
    identity = dict(pid=os.getpid(), process_created=psutil.Process().create_time(),
                    started_utc=datetime.now(timezone.utc).isoformat(), until=until.isoformat(),
                    scratch=str(scratch), repo=str(repo), interval_seconds=interval,
                    code_sha256=sha256(Path(__file__)), inference=False, process_signalling=False)
    if previous_receipt is not None:
        previous_receipt = safe_path(previous_receipt)
        identity.update(previous_receipt=str(previous_receipt),
                        previous_sha256=sha256(previous_receipt))
    # Persistent exclusive claim survives crashes; never silently reclaim/relaunch.
    write_json(claim, identity)
    scratch.mkdir()
    write_json(scratch/"OWNER.json", identity)
    try:
        previous = (json.loads(previous_receipt.read_text(encoding="utf-8"))["models"]
                    if previous_receipt is not None else None)
        while time.time() < until.timestamp():
            now = datetime.now(timezone.utc)
            label = now.strftime("refresh_%Y%m%d_%H%M%SZ")
            cycle = scratch/label
            load = psutil.cpu_percent(interval=1)
            if load > 90:
                write_json(scratch/(label+"_DEFERRED.json"), dict(cpu_percent=load,
                           reason="less_than_one_tenth_local_CPU_idle"))
            else:
                require_clean_branch(repo)
                relative = "docs/paper_results/wmdecide/"+label
                handoff = repo/"HANDOFF.md"
                old = handoff.read_text(encoding="utf-8")
                receipt = refresh_once(eval_root, cycle, repo/relative, previous=previous)
                previous = receipt["models"]
                if handoff.read_text(encoding="utf-8") != old:
                    raise ValueError("Handoff changed during collection; preserve peer edits")
                handoff.write_text(handoff_text(old, relative, receipt), encoding="utf-8")
                write_json(cycle/"COMMIT.json", commit_publication(repo, relative))
                print(json.dumps(dict(publication=relative, completed_utc=receipt["completed_utc"])),
                      flush=True)
            remaining = until.timestamp()-time.time()
            if remaining > 0:
                time.sleep(min(interval, remaining))
        write_json(scratch/"COMPLETE.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                   reason="registered_0900_refresh_deadline", owner=identity))
    except Exception as error:
        write_json(scratch/"STOP.json", dict(error=f"{type(error).__name__}: {error}", owner=identity))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--until", required=True, help="ISO8601 deadline with explicit timezone")
    parser.add_argument("--interval", type=float, default=1200)
    parser.add_argument("--previous", help="Preserved completed refresh receipt; reuse verified scores")
    args = parser.parse_args()
    run(args.repo, args.eval_root, args.scratch, args.claim,
        datetime.fromisoformat(args.until), args.interval, args.previous)


if __name__ == "__main__":
    main()
