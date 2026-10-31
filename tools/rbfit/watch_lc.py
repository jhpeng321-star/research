"""Durable owner-authorized LC continuation until09:00; no duplicate dispatches.

Reconciles existing owned work, publishes/commits each verified version, and
refits only after committed LANE physics bytes change. No process signals,
training, GPU allocation, SUMO invocation or evaluation-world data access.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from freeze_lc_version import REPO,REMOTE_PYTHON,source_identity

BOUNDARIES = "/home/ubuntu/rbfit20261101_lc_boundaries"
REMOTE_HELPER = BOUNDARIES+"/continuation_remote.py"
RECEIPTS = REPO/"artifacts/rbfit/receipts"
STATE_PATH = RECEIPTS/"LC_WATCH_0128_STATE.json"
MARK_BEGIN,MARK_END = "<!-- LC-CONTINUATION:BEGIN -->","<!-- LC-CONTINUATION:END -->"
ARTIFACTS = ("lc2013_population.json","lc2013_posteriors.json","lc2013_shrinkage.json",
             "validation.json","provenance.json","lane_events.json","VERIFIED.json")


def command(argv,timeout=60):
    completed = subprocess.run(argv,cwd=REPO,capture_output=True,text=True,timeout=timeout)
    if completed.returncode:
        raise RuntimeError(json.dumps({"argv":argv,"returncode":completed.returncode,
            "stdout":completed.stdout,"stderr":completed.stderr}))
    return completed.stdout


def remote(text,timeout=60):
    return command(["ssh","cpuQ",text],timeout=timeout)


def inspect(job,action="inspect"):
    return json.loads(remote(f"{REMOTE_PYTHON} {REMOTE_HELPER} --root {job['root']} --name {job['name']} {action}",timeout=90))


def save_state(state):
    STATE_PATH.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")


def publish(job,state):
    name = job["publication"]
    local = REPO/"artifacts/rbfit"/name
    if local.exists():
        # An interrupted publication is preserved for manual reconciliation.
        raise RuntimeError(f"Existing publication directory requires reconciliation: {local}")
    if command(["git","diff","--cached","--name-only"]).strip():
        raise RuntimeError("Unfamiliar staged work; publication commit blocked")
    controlled = ["HANDOFF.md","docs/paper_results/RBFIT.md"]
    if command(["git","diff","--name-only","--",*controlled]).strip():
        raise RuntimeError("Uncommitted handoff/report edits; preserve them before publication")
    local.mkdir()
    sources = [f"cpuQ:{job['root']}/{job['name']}/{name}" for name in ARTIFACTS]
    command(["scp",*sources,str(local)+"/"],timeout=90)
    verification = json.loads((local/"VERIFIED.json").read_text())
    assert verification["status"]=="PASS"
    for name,expected in verification["artifact_sha256"].items():
        assert hashlib.sha256((local/name).read_bytes()).hexdigest()==expected,name
    # Mirror the owned dispatch too; data observations remain on cpuQ.
    command(["scp",f"cpuQ:{job['root']}/{job['name']}_process.json",str(local/"process.json")])
    population = json.loads((local/"lc2013_population.json").read_text())
    policy = json.loads((local/"lc2013_shrinkage.json").read_text())
    raw = verification["aggregate"]["raw_physics_shrunk_posterior"]
    likelihood = verification["aggregate"]["likelihood_shrunk_posterior"]
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = (f"Latest verified `{name}` ({now}); LC SHA{population['lc2013_source_sha256'][:12]}, "
        f"snapshot{population.get('snapshot_version','v1')};{verification['drivers']}driver posteriors.\n"
        f"Raw physical expected changes{raw['expected_event_count']:.1f} vs{raw['events']}recorded; "
        f"eventBrier{raw['event_brier']:.6f}. Likelihood-emission Brier{likelihood['event_brier']:.6f} "
        f"is a separate diagnostic. **NOT ADOPTED**; no free-running/action-effect qualification.\n"
        f"Boundary-aware LC sampler required; weights already conditioned. "
        f"Inverse temperature={policy.get('inverse_temperature','not applicable')}; "
        f"IDM preferred delivery remains `shrinkage_v1`. Live receipts:LC_WATCH_0128_STATE.json.\n")
    handoff = REPO/"HANDOFF.md"
    text = handoff.read_text(encoding="utf-8")
    managed = MARK_BEGIN+"\n"+summary+MARK_END
    if MARK_BEGIN in text:
        start,end = text.index(MARK_BEGIN),text.index(MARK_END)+len(MARK_END)
        text = text[:start]+managed+text[end:]
    else:
        first,rest = text.split("\n",1)
        text = first+"\n\n"+managed+"\n\n"+rest
    handoff.write_text(text,encoding="utf-8")
    paper = REPO/controlled[1]
    lines = paper.read_text(encoding="utf-8").splitlines()
    for i,line in enumerate(lines):
        if line.startswith(("- Boundary v4 running:","- LC continuation:")):
            lines[i] = (f"- LC continuation: `{name}/`, snapshot{population.get('snapshot_version','v1')}; "
                f"{verification['drivers']}posteriors independently verified. RAW physical event Brier "
                f"{raw['event_brier']:.6f}, expected{raw['expected_event_count']:.1f}vs{raw['events']}events. "
                "Population LS/categorical likelihood and causal temperature use motion only; no free-running adoption.")
    assert len(lines)<=40
    paper.write_text("\n".join(lines)+"\n",encoding="utf-8")
    receipt = local/"PUBLICATION.json"
    audit = {"status":"PASS","timestamp":now,"verified_sha256":hashlib.sha256((local/"VERIFIED.json").read_bytes()).hexdigest(),
        "git_status_before_stage":command(["git","status","--short"]),
        "git_diff_before_stage":command(["git","diff","--",*controlled]),
        "git_log_before_stage":command(["git","log","--oneline","-10"]),"paper_lines":len(lines)}
    receipt.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    command(["git","diff","--check"])
    stage = [str(local.relative_to(REPO)),*controlled]
    if job.get("key"):
        source_bundle = REPO/"artifacts/rbfit/source_versions"/job["key"]
        if source_bundle.is_dir():
            stage.append(str(source_bundle.relative_to(REPO)))
    command(["git","add","--",*stage])
    command(["git","commit","-m",f"S-RB-FIT: publish verified LC continuation {name}"])
    commit = command(["git","rev-parse","HEAD"]).strip()
    shared = REPO.parent/"rb-shared/fit"
    latest = {"timestamp":now,"commit":commit,"directory":str(local),"adopted":False,
        "source_sha256":population["lc2013_source_sha256"],"raw_physical_scores":raw,
        "idm_preferred_directory":str(REPO/"artifacts/rbfit/shrinkage_v1"),
        "boundary_reader_required":True,"handoff":str(handoff)}
    (shared/"LATEST_LC.json").write_text(json.dumps(latest,indent=2)+"\n",encoding="utf-8")
    job["published_commit"] = commit
    state["latest_publication"] = latest
    print(json.dumps({"event":"published","name":name,"commit":commit}),flush=True)


def dispatch_raw(job,state):
    if job["name"]=="lane_boundaries_v4":
        next_job = dict(root=job["root"],name="lane_raw_v5",publication="lane_raw_v5")
        action = f"{REMOTE_PYTHON} {job['root']}/dispatch_raw.py fit"
    else:
        next_job = dict(root=job["root"],name="lane_raw",publication="lane_raw_"+job["key"],key=job["key"])
        action = f"{REMOTE_PYTHON} {job['root']}/dispatch_lc_version.py --root {job['root']} raw"
    existing = inspect(next_job)
    if not existing["claimed"]:
        remote(action)
    if not any(j["root"]==next_job["root"] and j["name"]==next_job["name"] for j in state["jobs"]):
        state["jobs"].append(next_job)
    job["raw_dispatched"] = True


def prepare_new_source(commit,digest,state):
    key = digest[:12]
    # Claim in durable state BEFORE preparing anything remotely. Failure is
    # terminal for this identity until a human reconciles it; no automatic retry.
    state["seen_source_digests"].append(digest)
    save_state(state)
    output = command([sys.executable,"tools/rbfit/freeze_lc_version.py","--lane-commit",commit,"--prepare"],timeout=90)
    info = json.loads(output.strip().splitlines()[-1])
    root = info["remote_root"]
    test = (f"env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RBFIT_TOOLS={root} "
        f"RBFIT_MOTION_FIT={root}/motion_fit.py PYTHONPATH={root}/src {REMOTE_PYTHON} -m pytest "
        f"{root}/test_rbfit_directional.py -q -p no:cacheprovider")
    results = remote(test)
    (Path(info["bundle"])/"CPU_TESTS.txt").write_text(results,encoding="utf-8")
    job = dict(root=root,name="lane_fit",publication="lane_source_"+key,key=key,dispatch_pending=True)
    state["jobs"].append(job)
    save_state(state)
    remote(f"{REMOTE_PYTHON} {root}/dispatch_lc_version.py --root {root} fit")
    job["dispatch_pending"] = False
    print(json.dumps({"event":"new_source_dispatched","lane_commit":commit,"key":key}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until",required=True)
    parser.add_argument("--last-start",required=True)
    parser.add_argument("--poll-seconds",type=int,default=60)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    until,last_start = datetime.fromisoformat(args.until),datetime.fromisoformat(args.last_start)
    if args.poll_seconds <= 0 or not datetime.now().astimezone()<until or not last_start<until:
        raise ValueError("Invalid continuation window")
    if args.launch:
        if not RECEIPTS.is_dir():
            raise ValueError("Existing own receipt parent required")
        receipt = RECEIPTS/"LC_WATCH_0128_process.json"
        with receipt.open("x",encoding="utf-8") as stream:
            json.dump({"state":"claimed","until":args.until,"last_start":args.last_start},stream)
        argv = [sys.executable,"-u",str(Path(__file__)),"--until",args.until,"--last-start",args.last_start,
                "--poll-seconds",str(args.poll_seconds)]
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS) if os.name=="nt" else 0
        with (RECEIPTS/"LC_WATCH_0128.log").open("xb") as log:
            process = subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                creationflags=flags,start_new_session=os.name!="nt")
        data = {"state":"dispatched","pid":process.pid,"argv":argv,"until":args.until,
                "last_start":args.last_start,"timestamp":datetime.now().astimezone().isoformat(),"automatic_restart":False}
        receipt.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(data))
        return
    if STATE_PATH.exists():
        raise RuntimeError("Existing durable continuation state: inspect, do not duplicate watcher")
    _,current_digest,_ = source_identity("0a1e7e93")
    state = {"started":datetime.now().astimezone().isoformat(),"until":args.until,"last_start":args.last_start,
        "seen_source_digests":[current_digest],"jobs":[
            dict(root=BOUNDARIES,name="lane_boundaries_v4",publication="lane_boundaries_v4"),
            dict(root="/home/ubuntu/rbfit20261101_lc_source_ec0f2a74974b",name="lane_fit",
                 publication="lane_source_ec0f2a74974b",key="ec0f2a74974b")],"errors":[]}
    with STATE_PATH.open("x",encoding="utf-8") as stream:
        json.dump(state,stream,indent=2)
    while datetime.now().astimezone()<until:
        active = False
        for job in list(state["jobs"]):
            if job.get("blocked") or (job.get("published_commit") and (job["name"] in ("lane_raw","lane_raw_v5") or job.get("raw_dispatched"))):
                continue
            try:
                status = inspect(job)
                if job.get("dispatch_pending") and not status["claimed"]:
                    remote(f"{REMOTE_PYTHON} {job['root']}/dispatch_lc_version.py --root {job['root']} fit")
                    job["dispatch_pending"] = False
                    status = inspect(job)
                job["last_status"] = status
                active |= status["running"]
                if status["complete"]:
                    if not status["verified"]:
                        inspect(job,"verify")
                    if not job.get("published_commit"):
                        publish(job,state)
                    if job["name"] in ("lane_fit","lane_boundaries_v4") and not job.get("raw_dispatched"):
                        dispatch_raw(job,state)
                elif status["claimed"] and not status["running"]:
                    job["blocked"] = "Owned job exited without complete output; preserve logs, no retry"
            except Exception as exc:
                # Load admission can be retried before a receipt exists; all
                # other errors require manual reconciliation, never job restart.
                message = str(exc)
                if "free-load admission failed" not in message:
                    job["blocked"] = message
                state["errors"].append({"timestamp":datetime.now().astimezone().isoformat(),"job":job["publication"],"error":message})
                print(json.dumps({"event":"blocked","job":job["publication"],"error":message}),flush=True)
        try:
            commit,digest,_ = source_identity()
            if digest not in state["seen_source_digests"] and not active and datetime.now().astimezone()<last_start:
                prepare_new_source(commit,digest,state)
        except Exception as exc:
            state["errors"].append({"timestamp":datetime.now().astimezone().isoformat(),"source_check_error":str(exc)})
            print(json.dumps({"event":"source_check_error","error":str(exc)}),flush=True)
        state["heartbeat"] = datetime.now().astimezone().isoformat()
        save_state(state)
        time.sleep(args.poll_seconds)
    state["finished"] = datetime.now().astimezone().isoformat()
    save_state(state)
    print(json.dumps({"event":"owner_window_complete","finished":state["finished"]}),flush=True)


if __name__ == "__main__":
    main()
