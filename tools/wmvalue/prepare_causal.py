"""Export amended FIT features and reconciled real labels, no fitting or SUMO."""
import argparse
import json
import os
import time

import numpy as np

from .audit import ARMS, allowed_seed, digest, safe
from .core import features
from .reconstruct import OBS_KEYS, command_summaries, label_clock, observations


def prepare(root, corpus, out, limit=None):
    root, out = safe(root), safe(out)
    if out.exists() or out.with_suffix(".json").exists() or not out.parent.is_dir():
        raise ValueError("output must be new with an existing parent")
    seeds = range(969400, 969432) if corpus == "wmdata" else range(969550, 969600)
    arms = ARMS if corpus == "wmdata" else (*ARMS, "generic_random")
    records, x, y, actions, groups, row_arms, times = [], [], [], [], [], [], []
    begun = time.monotonic()
    for seed in seeds:
        allowed_seed(seed, "fit")
        for arm in arms:
            job = safe(root/"jobs"/f"s{seed}_{arm}")
            ep = safe(job/"deploy/episode/episode.npz")
            manifest_path = safe(job/"deploy/episode/manifest.json")
            manifest = json.loads(manifest_path.read_text())
            if (int(manifest["seed"]) != seed or manifest["status"] != "complete"
                    or manifest["warmup_s"] != 300 or manifest["horizon_s"] != 600):
                raise ValueError("episode identity/clock/status mismatch")
            with np.load(ep, allow_pickle=False) as archive:
                # Explicit whitelist. Never load hidden length/type/parameter members.
                tape = {k: archive[k] for k in (*OBS_KEYS, "vehicle_ids")}
            objectives = {str(h): json.loads(safe(job/f"objective_{h}.json").read_text())
                          for h in (120, 300, 600)}
            clock = label_clock(tape, seed, float(manifest["demand_vph"]), objectives)
            grid = list(range(300, 901, 30))
            log = safe(job/"deploy/episode/run/action_timeline.jsonl.gz")
            known = {t: {str(tape["vehicle_ids"][i]) for i in tape["vid_index"][t*2][
                         tape["present"][t*2] & tape["is_cav"][t*2]]} for t in grid}
            summaries = command_summaries(log, grid, known)
            for state in observations(tape, grid):
                t = int(state["time_s"])
                x.append(features(state))
                y.append(float(clock["remaining"][(t-300)*2]))
                actions.append(summaries[t])
                groups.append(seed)
                row_arms.append(arm)
                times.append(t)
            record = dict(seed=seed, arm=arm, episode_sha256=digest(ep),
                          command_sha256=digest(log), manifest_sha256=digest(manifest_path),
                          objective_sha256={str(h): digest(job/f"objective_{h}.json")
                                            for h in (120, 300, 600)},
                          clock_differences=clock["reconciled"], observed_states=len(grid))
            records.append(record)
            if len(records) % 40 == 0 or limit is not None:
                print(json.dumps(dict(done=len(records), seed=seed, arm=arm,
                                      seconds=round(time.monotonic()-begun, 2))), flush=True)
            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break
    expected = len(seeds)*len(arms)
    doc = dict(schema="wmvalue_causal_fit/1", corpus=corpus, role="fit", root=str(root),
               episodes=len(records), expected_episodes=expected, complete=len(records) == expected,
               real_data_fit=False, reserved_worlds_opened=False, sumo_seconds=0,
               feature_count=len(x[0]), states=len(x), records=records,
               source_features=False, causal_flow=True, nominal_length_only=True,
               seconds=time.monotonic()-begun,
               code_sha256={name: digest(safe(__file__).parent/name) for name in
                            ("prepare_causal.py", "reconstruct.py", "core.py", "audit.py")})
    with out.open("xb") as stream:
        np.savez_compressed(stream, x=np.stack(x), y=np.asarray(y), actions=np.stack(actions),
                            seed=np.asarray(groups), arm=np.asarray(row_arms), time_s=np.asarray(times))
    doc["arrays_sha256"] = digest(out)
    with out.with_suffix(".json").open("x") as stream:
        json.dump(doc, stream, indent=1, allow_nan=False)
    print(json.dumps({k: v for k, v in doc.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--corpus", required=True, choices=("wmdata", "wmdata2"))
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if hasattr(os, "nice"):
        os.nice(19)
    prepare(a.root, a.corpus, a.out, a.limit)
