"""Concrete corrected validation adapter; all four verdict stages share hash bindings."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

from tools.wmfixmerge.data import WM2_SHA, digest, safe


def _group_binding(plan, candidate, candidate_config):
    """Every per-group A_*.json file carries the exact candidate binding."""
    return dict(candidate_sha256=digest(candidate), candidate_config_sha256=digest(candidate_config),
                reference_sha256=digest(plan["reference"]))


def load_group_file(path, binding):
    """Fail-closed reuse of an existing per-group shard file (resume/shard-merge)."""
    doc = json.loads(safe(path).read_text())
    if doc.get("binding") != binding or not isinstance(doc.get("rows"), list) or not doc["rows"]:
        raise ValueError("existing panel-A group file is foreign or empty: " + path.name)
    return doc["rows"]


def save_group_file(path, rows, binding):
    with path.open("x") as stream:
        json.dump(dict(binding=binding, rows=rows), stream, allow_nan=False)


def parse_groups(spec, total):
    """'0-17' / '18,19' / '3' -> sorted unique index list within range."""
    indices = set()
    for token in str(spec).split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = sorted(int(x) for x in token.split("-", 1))
            indices.update(range(lo, hi + 1))
        else:
            indices.add(int(token))
    ordered = sorted(indices)
    if not ordered or ordered[0] < 0 or ordered[-1] >= total:
        raise ValueError("group shard out of range")
    return ordered


def validation(plan, candidate, candidate_config, out, device, groups=None):
    import torch
    from tools.wmcombined import e3_score as e3
    from wms.data.c4_pairs import PairedBranchSource
    from tools.wmrequal.grouped_feedback import feedback
    import wms.control.entrant as entrant
    own = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(own / "tools/wmrequal"))
    spec = importlib.util.spec_from_file_location("wmfix_corrected_score", own / "tools/wmrequal/score.py")
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    if digest(plan["reference"]) != WM2_SHA:
        raise ValueError("validation reference not registered WM2")
    real = json.loads(safe(plan["validation_real"]).read_text())["panels"]["validation"]
    root = safe(plan["validation_root"])
    manifest = json.loads(safe(root / "C4_EPISODES_MANIFEST.json").read_text())
    split = json.loads(safe(root / "SPLIT.json").read_text())
    entries = [e for e in manifest["episodes"] if split["worlds"][str(e["seed"])] == "validation"]
    if {e["pair_key"] for e in entries} != {r["pair_key"] for r in real}:
        raise ValueError("corrected validation real/source group mismatch")
    source = PairedBranchSource(root, horizon_steps=600, history_s=60., history_stride_s=1.,
        entries_v2=True, junction_fix="ffill", command_known=True, arm_kinds=("B", "POLICY"),
        role="train", role_of={e["world"]: "train" if e in entries else "val" for e in manifest["episodes"]})
    if len(source.groups) != 36 or source.horizon != 600:
        raise ValueError("incomplete corrected validation cohort")
    specs = json.loads(safe(plan["controllers"]).read_text())["controllers"]
    entrant.feedback_from_specs = feedback
    original_entries = score.Entries
    class ConservingEntries(original_entries):
        def __init__(self, process):
            process.cfg = replace(process.cfg, conserve_queue=True)
            super().__init__(process)
    binding = _group_binding(plan, candidate, candidate_config)
    shard = None if groups is None else parse_groups(groups, len(source.groups))
    samples = {}
    for name, config, ckpt in (("base", plan["base_config"], plan["reference"]),
                                ("entry", plan["base_config"], plan["reference"]),
                                ("candidate", candidate_config, candidate)):
        _, settings, model = e3.load_for_scoring(safe(config), safe(ckpt))
        if name == "candidate" and model.merge_response is None:
            raise ValueError("candidate head missing under verdict config")
        if name == "entry" and model.merge_response is not None:
            raise ValueError("entry arm must be the zero-head WM2 entry path")
        settings = replace(settings, seed=140014, entrant_control=True,
            entrant_controllers={a: specs[a] for a in ("mappo", "ours_c1", "coopoff_c1")},
            entry_queue_state=True, entry_pad_auto=True, entry_pad_max=1024, entry_insert_rule="sumo_like_v2")
        model = model.to(device).eval().requires_grad_(False)
        model.cav_exec_cap = "idm"
        entry_path = name in ("entry", "candidate")
        score.Entries = ConservingEntries if entry_path else original_entries
        todo = range(len(source.groups)) if shard is None else shard
        for i in todo:
            path = out / f"A_{name}_{i:03d}.json"
            if path.is_file():
                continue  # already computed by this or another shard process
            rows = score.score_batch(model, settings, e3, source, [i], real, torch.device(device),
                                      600, False, native_lanes=True, mix_modes=(True,),
                                      learned_admission=entry_path)
            save_group_file(path, rows, binding)
        del model
    score.Entries = original_entries
    missing = [f"{n}_{i:03d}" for n in ("base", "entry", "candidate")
               for i in range(len(source.groups)) if not (out / f"A_{n}_{i:03d}.json").is_file()]
    if missing:
        return dict(schema="wmfix_validation_shard/1", complete=False,
                    missing=len(missing), first_missing=missing[0], binding=binding)
    # Single source of truth: assembly always reads the immutable per-group files.
    for name in ("base", "entry", "candidate"):
        samples[name] = [row for i in range(len(source.groups))
                         for row in load_group_file(out / f"A_{name}_{i:03d}.json", binding)]
    lookup = {(r["pair_key"], r["arm"]): r for r in real}
    rows = []
    for key in sorted({r["pair_key"] for r in real}):
        ref = next(r for r in real if r["pair_key"] == key and r["native"])
        for arm in ("ours_c1", "coopoff_c1", "mappo"):
            observed = lookup[key, arm]
            for h in (60, 90, 120, 180, 300):
                row = dict(world=observed["world"], pair_key=key, arm=arm, horizon=h,
                    real=observed["values"][str(h)]["J"] - ref["values"][str(h)]["J"], components={})
                for name in ("base", "entry", "candidate"):
                    select = lambda a: sorted([r for r in samples[name] if r["pair_key"] == key and
                                              r["arm"] == a], key=lambda r: r["draw"])
                    aa, bb = select(arm), select("native")
                    if [r["draw"] for r in aa] != list(range(8)) or [r["draw"] for r in bb] != list(range(8)):
                        raise ValueError("incomplete K8/native-arm pair")
                    row[name + "_draws"] = [a["values"][str(h)]["J"] - b["values"][str(h)]["J"]
                                             for a, b in zip(aa, bb)]
                    row["components"][name] = {c: [a["values"][str(h)][c] - b["values"][str(h)][c]
                        for a, b in zip(aa, bb)] for c in ("source", "upstream", "downstream", "discharge")}
                row["components"]["real"] = {c: observed["values"][str(h)][c] - ref["values"][str(h)][c]
                    for c in ("source", "upstream", "downstream", "discharge")}
                rows.append(row)
    return dict(schema="wmfix_validation/2", K=8, mix="ON", native_lanes=True, rows=rows,
                validation_real_sha256=digest(plan["validation_real"]),
                scorer_sha256=digest(own / "tools/wmrequal/score.py"),
                ownership_sha256=digest(own / "tools/wmrequal/native_lanes.py"))


def safety(plan, candidate, config, out):
    """Unmodified canonical acceptance; step0 is a hash-bound zero-head WM2 wrapper."""
    from tools.loop14_checks.accept import main
    import torch
    from tools.wmfixmerge.data import WM2_SHA
    reference = safe(plan["zero_reference"])
    payload = torch.load(reference, map_location="cpu", weights_only=True)
    base = torch.load(safe(plan["reference"]), map_location="cpu", weights_only=True)
    if digest(plan["reference"]) != WM2_SHA or payload.get("wmfix", {}).get("step") != 0:
        raise ValueError("safety reference not WM2 step0")
    if any(not torch.equal(v, payload["model"].get(n)) for n, v in base["model"].items()):
        raise ValueError("safety reference legacy tensors differ from WM2")
    if payload["model"]["merge_response.net.2.weight"].count_nonzero() or \
            payload["model"]["merge_response.net.2.bias"].count_nonzero():
        raise ValueError("safety reference correction is not zero")
    args = ["--candidate", str(candidate), "--reference", str(reference), "--config", str(config),
            "--roots", *plan["accept_roots"], "--panel", plan["accept_panel"], "--out", str(out / "accept")]
    code = main(args)
    result = json.loads(safe(out / "accept/ACCEPT.json").read_text())
    if code != 0 or result.get("decision") != "ACCEPT":
        raise ValueError("canonical accept.py did not ACCEPT")
    return result
