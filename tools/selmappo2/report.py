"""Read-only collection and offline reporting for the immutable per-CAV run.

Only ``--action collect`` uses SSH (the shared remote.run transport). It executes
the stored BUNDLE's ops bootstrap, never the current trainer or a simulator.
Collections and curve assets are exclusive, append-only files. ``--action report``
reads one local STATUS or COLLECTION JSON; --output explicitly publishes Markdown.

Optional peer evaluation input must be byte-pinned with --evaluation-sha256 and
have schema selmappo2.evaluation/1, contract S-C1SWITCH2, status COMPLETE,
contract_sha256, checkpoint_sha256, worlds (the ordered 16-cell panel), and
results (one normal episode result per world, mode evaluation, iteration 169).
This tool neither runs evaluation nor selects a checkpoint by its performance.
"""
import argparse
import base64
import csv
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import re
import time
import uuid
import zlib

from . import ops


BUILD = "build_b42c257b09813c175fcb7261e341c7599d0bb5c4c971529e2856890d2fc8038c"
PID = 428649
SCHEMA = "selmappo2.collection/1"
FIRST, COUNT, UPDATES, BATCH = 18100000, 7774, 169, 46
SMOKES = (18107998, 18107999)
PANEL = tuple(range(969136, 969152))
CELLS = tuple((p, d) for p in (.05, .1, .2, .4) for d in (1900, 2400, 2700))
PANEL_CELLS = CELLS + ((.2, 2400), (.2, 2700), (.4, 2400), (.4, 2700))
SHA = re.compile(r"[0-9a-f]{64}\Z")
FIXED = ("DISPATCH.json", "INTEGRATION.json", "training/OWNER.json",
          "training/TRAIN_OWNER.json", "training/training.jsonl",
          "training/coordinator.jsonl", "training/FROZEN_FINAL.json", "CONTINUATION.json")
SAFETY = ("collisions", "teleports", "ttc_below_3_vs", "hard_brake_vs")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    require(isinstance(value, str) and SHA.fullmatch(value), "invalid SHA256")
    return value


def identity(row, build, pid):
    require(ops.BUILD.fullmatch(build), "invalid build identifier")
    binding = build.removeprefix("build_")
    require(type(pid) is int and pid > 0, "invalid PID")
    require(all(row.get(k) == v for k, v in dict(
        owner=ops.OWNER, host=ops.HOST, root=ops.PARENT + "/" + build,
        binding=binding, pid=pid).items()), "receipt build/root/PID ownership mismatch")


def jsonl(data):
    """Only newline-terminated records are durable; keep a partial suffix visible."""
    text = data.decode("utf-8")
    lines = text.splitlines(keepends=True)
    partial = bool(lines and not lines[-1].endswith("\n"))
    if partial:
        lines.pop()
    rows = [json.loads(line) for line in lines]
    require(all(isinstance(row, dict) for row in rows), "JSONL record is not an object")
    return rows, partial


def expected_world(seed, mode):
    require(type(seed) is int, "seed must be an integer")
    if mode == "smoke":
        require(seed in SMOKES, "unowned smoke seed")
        return dict(seed=seed, penetration=.4, demand=2700)
    require(mode == "training" and FIRST <= seed < FIRST + COUNT,
            "unowned seed or mode (evaluation belongs to the peer panel)")
    iteration, within = divmod(seed - FIRST, BATCH)
    cells = list(CELLS)
    random.Random(iteration).shuffle(cells)
    p, demand = cells[within % len(cells)]
    return dict(seed=seed, penetration=p, demand=demand)


def ledger(rows):
    """Replay all coordinator events, retaining failed and unresolved charges."""
    records = {}
    for row in rows:
        seed, event = row.get("seed"), row.get("event")
        require(type(seed) is int, "invalid ledger seed")
        if event == "START":
            mode = row.get("mode")
            world = expected_world(seed, mode)
            require(seed not in records and row.get("world") == world
                    and row.get("reserved_sim_s") == 900 and row.get("panel") == [],
                    "duplicate or invalid reservation")
            records[seed] = dict(start=row, actual=None, terminal=None)
        else:
            require(seed in records, "ledger event without reservation")
            record = records[seed]
            require(record["terminal"] is None, "duplicate terminal event")
            if event == "ACTUAL":
                actual = row.get("actual_sim_s")
                require(record["actual"] is None and number(actual) and 0 <= actual <= 900,
                        "invalid actual accounting")
                record["actual"] = actual
            elif event == "RESULT":
                require(record["actual"] == 900, "RESULT without 900 actual seconds")
                record["terminal"] = event
            elif event == "FAILED":
                require(row.get("actual_sim_s") == record["actual"], "failed actual mismatch")
                record["terminal"] = event
            else:
                raise ValueError("unknown coordinator event")
    return records


def _remote_collect(sources, binding, build, expected_pid):
    """Self-contained extension executed in the *stored ops* global namespace.

    No recursive data traversal: fixed receipts and START-derived result paths
    only. The final checkpoint is streamed through SHA256, never transported.
    """
    import contextlib
    import io

    # These names deliberately come from the frozen ops bootstrap, not imports
    # from this reporting module (which is absent from the staged inventory).
    api = globals()
    _remote_main, tagged = api["_remote_main"], api["tagged"]
    checked, read_json = api["checked"], api["read_json"]
    sha, canonical = api["sha"], api["canonical"]
    PARENT, OWNER, HOST = api["PARENT"], api["OWNER"], api["HOST"]
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        _remote_main("status", sources, binding, build)
    statuses = tagged(capture.getvalue(), "STATUS_RECEIPT")
    if len(statuses) != 1:
        raise ValueError("missing owned status")
    status = statuses[0]
    if not status.get("dispatched") or status.get("pid") != expected_pid:
        raise ValueError("dispatch PID not the explicitly expected PID")
    root = Path(PARENT) / build
    dispatch = read_json(root / "DISPATCH.json")
    if dispatch.get("out") != str(root / "training"):
        raise ValueError("dispatch output root mismatch")
    files, missing = {}, []

    def snapshot(name):
        path = root / name
        # checked() rejects symlinks in all components before opening contents.
        try:
            path = checked(path)
        except FileNotFoundError:
            missing.append(name)
            return b""
        with path.open("rb") as stream:
            data = stream.read(os.fstat(stream.fileno()).st_size)
        files[name] = dict(bytes=len(data), sha256=sha(data),
                           base64=base64.b64encode(data).decode("ascii"),
                           timestamp_ns=time.time_ns())
        return data

    # Read the final manifest before updates, and updates before the ledger.
    # A final freeze arriving later is intentionally deferred to the next collect.
    for name in ("DISPATCH.json", "INTEGRATION.json", "CONTINUATION.json", "training/OWNER.json",
                 "training/TRAIN_OWNER.json", "training/FROZEN_FINAL.json",
                 "training/training.jsonl"):
        snapshot(name)
    data = snapshot("training/coordinator.jsonl")
    starts = {}
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        row = json.loads(line)
        if row.get("event") != "START":
            continue
        seed, mode = row.get("seed"), row.get("mode")
        allowed = (type(seed) is int and
                   ((mode == "training" and 18100000 <= seed <= 18107773)
                    or (mode == "smoke" and seed in (18107998, 18107999))))
        if (not allowed or seed in starts or row.get("world", {}).get("seed") != seed
                or row.get("reserved_sim_s") != 900 or row.get("panel") != []):
            raise ValueError("unowned/duplicate ledger path")
        starts[seed] = row
    if starts and "training/OWNER.json" not in files:
        raise ValueError("ledger without validated training ownership")
    for seed in sorted(starts):
        snapshot(f"training/{seed}/result.jsonl")
    checkpoint = None
    if "training/FROZEN_FINAL.json" in files:
        path = checked(root / "training/checkpoint_169.pt")
        h, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
                size += len(chunk)
        checkpoint = dict(path="training/checkpoint_169.pt", bytes=size, sha256=h.hexdigest())
    result = dict(schema="selmappo2.collection/1", owner=OWNER, host=HOST,
                  root=str(root), binding=binding, pid=expected_pid,
                  timestamp_ns=time.time_ns(), status=status, files=files,
                  missing=missing, checkpoint=checkpoint, new_sumo_seconds=0)
    print("COLLECTION_RECEIPT", canonical(result), flush=True)


def collect(build, pid, output_dir):
    """Explicit CLI-only remote access using the immutable local bundle."""
    require(ops.BUILD.fullmatch(build), "invalid build identifier")
    require(type(pid) is int and pid > 0, "invalid expected PID")
    ops.checked(output_dir, directory=True)
    binding = build.removeprefix("build_")
    saved = ops.read_json(ops.EVIDENCE / f"BUNDLE_{binding}.json")
    sources = saved["sources"]
    files = {ops.relative_name(name): base64.b64decode(data, validate=True)
             for name, data in saved["files"].items()}
    require(saved["binding"] == binding and ops.binding_for(sources) == binding
            and {name: ops.sha(data) for name, data in files.items()} == sources,
            "local immutable bundle mismatch")
    text = files["tools/selmappo2/ops.py"].decode("utf-8")
    remote_file = ops.PARENT + "/" + build + "/pytools/tools/selmappo2/ops.py"
    script = (f"__name__ = '_selmappo2_remote'\n__file__ = {remote_file!r}\n" + text
              + "\n" + inspect.getsource(_remote_collect)
              + f"\n_remote_collect({sources!r}, {binding!r}, {build!r}, {pid!r})\n")
    packed = base64.b64encode(zlib.compress(script.encode("utf-8"))).decode("ascii")
    code = f"import base64,zlib;exec(zlib.decompress(base64.b64decode({packed!r})))"
    from .shared import load
    package = load(ops.SHARED)
    run = importlib.import_module(package.__name__ + ".remote").run
    # shared remote.run prints stdout; keep the large receipt off the terminal.
    # The exact receipt is saved below, rather than duplicating megabytes in chat.
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = run(ops.HOST, code, timeout=300)
    rows = ops.tagged(result.stdout, "COLLECTION_RECEIPT")
    require(len(rows) == 1, "missing/ambiguous collection receipt")
    payload = rows[0]
    analyze(payload, build, pid)  # Fail closed before preserving a purported collection.
    name = f"COLLECTION_{binding}_{time.time_ns()}_{uuid.uuid4().hex}.json"
    path = Path(output_dir) / name
    ops.exclusive_json(path, payload)
    return path


def unpack(payload):
    files = {}
    for name, record in payload["files"].items():
        ops.relative_name(name)
        require(name in FIXED or re.fullmatch(r"training/1810\d{4}/result\.jsonl", name),
                "unexpected collection file")
        data = base64.b64decode(record["base64"], validate=True)
        require(type(record["bytes"]) is int and record["bytes"] == len(data)
                and ops.sha(data) == record["sha256"], "collection file hash/size mismatch")
        files[name] = data
    return files


def episode(row, world, mode, iteration):
    require(row.get("world") == world and row.get("mode") == mode
            and type(row.get("iteration")) is int and row["iteration"] == iteration
            and row.get("actual_sim_s") == 900 and row.get("decisions") == 30,
            "episode contract mismatch")
    require(number(row.get("J600")) and row["J600"] >= 0
            and number(row.get("episode_return"))
            and row["J600"] == -row["episode_return"], "invalid episode return")
    components = row.get("components", {})
    require(all(number(components.get(k)) and components[k] >= 0 for k in
                (*SAFETY, "road_delay_vs", "source_wait_vs", "team_delay_vs")),
            "missing/nonfinite episode safety or delay components")
    require(math.isclose(components["team_delay_vs"], row["J600"], rel_tol=1e-9)
            and math.isclose(components["road_delay_vs"] + components["source_wait_vs"],
                             row["J600"], rel_tol=1e-9), "episode delay component mismatch")
    return row


def analyze(payload, build=BUILD, pid=PID, checkpoint_sha=None):
    """Pure local validation. STATUS tails are never promoted to full curves."""
    identity(payload, build, pid)
    if checkpoint_sha is not None:
        digest(checkpoint_sha)
    if payload.get("schema") != SCHEMA:
        require("schema" not in payload and "summaries" in payload
                and payload.get("dispatched") is True, "not a STATUS or COLLECTION receipt")
        observed = []
        summary = payload["summaries"].get("training/training.jsonl", {})
        for line in summary.get("tail", "").splitlines():
            try:
                row = json.loads(line)
                if type(row.get("iteration")) is int:
                    observed.append(row["iteration"])
            except (ValueError, AttributeError):
                pass
        return dict(state="PARTIAL", status=payload, points=[], records={}, episodes={},
                    safety={}, reserved_sim_s=None, actual_sim_s=None, files={},
                    issues=["Status-only tails: full ledger, receipts and updates are uncollected."],
                    observed_iteration=max(observed, default=None), checkpoint_sha256=None)
    status = payload["status"]
    identity(status, build, pid)
    require(status.get("dispatched") is True and payload.get("new_sumo_seconds") == 0,
            "unvalidated collection status")
    files = unpack(payload)
    issues = []
    partials = set()

    def rows(name):
        data = files.get(name, b"")
        result, partial = jsonl(data)
        if partial:
            partials.add(name)
            issues.append(f"Unterminated JSONL suffix excluded: {name}")
        return result

    def obj(name):
        return json.loads(files[name]) if name in files else None

    dispatch = obj("DISPATCH.json")
    require(dispatch is not None, "collection missing DISPATCH")
    identity(dispatch, build, pid)
    require(dispatch.get("out") == payload["root"] + "/training", "dispatch output mismatch")
    records = ledger(rows("training/coordinator.jsonl"))
    continuation = obj("CONTINUATION.json")
    historical = set()
    if continuation is not None:
        old = continuation.get("original", {})
        require(all(continuation.get(k) == v for k, v in dict(
                    owner=ops.OWNER, host=ops.HOST, root=payload["root"], binding=payload["binding"],
                    inherited_reserved_sim_s=900, inherited_actual_sim_s=300,
                    skipped_seed=ops.HISTORICAL_SEED, only_unused_smoke_seed=SMOKES[1],
                    production_episodes=COUNT, cap_sim_s=7012800,
                    evaluation_reserved_unstarted_sim_s=14400).items())
                and continuation.get("command") == dispatch.get("command")
                and isinstance(continuation.get("command"), list)
                and "--continuation" in continuation["command"]
                and continuation.get("authority", "").startswith("- 01:48 MAIN ACKNOWLEDGEMENTS / DECISIONS:")
                and "S-SELMAPPO2 same-cap continuation ACKNOWLEDGED (unused second smoke world only, then production)"
                    in continuation["authority"]
                and all(old.get(k) == v for k, v in dict(
                    root=ops.PARENT + "/build_" + ops.HISTORICAL_BINDING,
                    binding=ops.HISTORICAL_BINDING, pid=ops.HISTORICAL_PID,
                    ledger_sha256=ops.HISTORICAL_LEDGER_SHA, reserved_sim_s=900,
                    actual_sim_s=300, seed=ops.HISTORICAL_SEED, status="FAILED").items()),
                "unacknowledged or mismatched continuation")
        prefix = b"".join(files.get("training/coordinator.jsonl", b"").splitlines(keepends=True)[:3])
        # Before training ownership/ledger appears, collection is still safe and partial.
        if records:
            require(ops.sha(prefix) == ops.HISTORICAL_LEDGER_SHA,
                    "continuation historical failure hash mismatch")
            historical = {ops.HISTORICAL_SEED}
            issues.append("Acknowledged historical smoke failure retained: 900 reserved / 300 actual s.")
    owner, phase = obj("training/OWNER.json"), obj("training/TRAIN_OWNER.json")
    integration = obj("INTEGRATION.json")
    if owner is not None:
        require(integration is not None and owner.get("pid") == pid
                and owner.get("variant") == "per_cav" and owner.get("workers") == 12
                and owner.get("optimization_seed") == 1 and phase == dict(pid=pid)
                and owner.get("bindings") == dict(
                    design_sha256=ops.DESIGN_SHA,
                    integration_sha256=ops.sha(files["INTEGRATION.json"]))
                and integration.get("binding") == payload["binding"]
                and integration.get("owner") == ops.OWNER
                and integration.get("design_sha256") == ops.DESIGN_SHA
                and integration.get("schema") == "selmappo.integration/2"
                and integration.get("status") == "CPU_INTEGRATION_PASS"
                and integration.get("variant") == "per_cav"
                and integration.get("protocol") == "S-C1SWITCH2/v2"
                and integration.get("cadence_s") == 20
                and integration.get("library") == ["native", "ctrl0", "mappo", "pi", "circles", "wall"],
                "training source/integration ownership mismatch")
        sources = integration.get("files", {})
        for name, value in sources.items():
            ops.relative_name(name)
            digest(value)
        expected = {Path(name).stem: value for name, value in sources.items()
                    if name.startswith("tools/selmappo/")}
        expected.update({"selmappo2." + name: sources.get(f"tools/selmappo2/{name}.py")
                         for name in ("shared", "mixed", "training")})
        require(sources and ops.binding_for(sources) == payload["binding"]
                and owner.get("sources") == expected
                and owner.get("registration_sha256") == sources.get("REGISTRATION.md")
                and integration.get("registration_sha256") == sources.get("REGISTRATION.md"),
                 "training immutable source mismatch")
        if continuation is not None:
            require(continuation.get("registration_sha256") == sources.get("REGISTRATION.md"),
                    "continuation registration mismatch")
    else:
        require(not records and not files.get("training/training.jsonl")
                and "training/FROZEN_FINAL.json" not in files, "data without training OWNER")
        issues.append("Training OWNER not yet observed.")
    episodes = {}
    for seed, record in records.items():
        name = f"training/{seed}/result.jsonl"
        result = rows(name)
        require(len(result) <= 1, "duplicate episode result")
        if result:
            start = record["start"]
            iteration = 0 if start["mode"] == "smoke" else (seed - FIRST) // BATCH
            episodes[seed] = episode(result[0], start["world"], start["mode"], iteration)
        if record["terminal"] == "RESULT" and seed not in episodes:
            issues.append(f"Missing complete episode receipt: {seed}")
    require(all(name in FIXED or int(name.split("/")[1]) in records for name in files),
            "episode file without ledger reservation")
    updates = rows("training/training.jsonl")
    points = []
    for index, row in enumerate(updates, 1):
        require(type(row.get("iteration")) is int and row["iteration"] == index <= UPDATES,
                "duplicate/out-of-order update")
        digest(row.get("checkpoint_sha256"))
        for key in ("J600_mean", "J600_min", "J600_max", "J600_std",
                    "charged_sim_s", "actual_sim_s", "simulation_wall_s", "total_wall_s"):
            require(number(row.get(key)) and row[key] >= 0, "invalid complete-update metric")
        expected_s = (len(SMOKES) + index * BATCH) * 900
        require(row["charged_sim_s"] == expected_s
                and row["actual_sim_s"] == expected_s - (600 if historical else 0),
                "update actual/reserved accounting mismatch")
        seeds = [*(seed for seed in SMOKES if seed not in historical),
                 *range(FIRST, FIRST + index * BATCH)]
        if any(seed not in episodes or records[seed]["terminal"] != "RESULT" for seed in seeds):
            issues.append(f"Update {index} lacks a complete receipt prefix; curve stops here.")
            break
        cohort = [episodes[s]["J600"] for s in range(FIRST + (index - 1) * BATCH,
                                                    FIRST + index * BATCH)]
        mean = math.fsum(cohort) / BATCH
        stats = dict(J600_mean=mean, J600_min=min(cohort), J600_max=max(cohort),
                     J600_std=math.sqrt(math.fsum((v - mean) ** 2 for v in cohort) / BATCH))
        require(all(math.isclose(row[k], v, rel_tol=1e-9, abs_tol=1e-8)
                    for k, v in stats.items()), "update statistics differ from episode receipts")
        points.append(row)
    safety = {key: math.fsum(row["components"][key] for row in episodes.values())
              for key in SAFETY}
    successful = {seed for seed, r in records.items() if r["terminal"] == "RESULT"}
    failed = any(r["terminal"] == "FAILED" or
                 (r["actual"] is not None and r["actual"] != 900)
                 for seed, r in records.items() if seed not in historical)
    unsafe = safety["collisions"] != 0 or safety["teleports"] != 0
    if failed:
        issues.append("Failed/short episode accounting observed; reservations are not refunded.")
    if unsafe:
        issues.append("Nonzero collision or teleport receipts observed.")
    frozen = obj("training/FROZEN_FINAL.json")
    checkpoint = payload.get("checkpoint")
    final_sha = None
    if frozen is not None:
        final_sha = digest(frozen.get("checkpoint_sha256"))
        require(owner is not None and frozen.get("iteration") == UPDATES
                and frozen.get("sources") == owner["sources"]
                and frozen.get("bindings") == owner["bindings"], "frozen final ownership mismatch")
        require(checkpoint is not None and checkpoint.get("path") == "training/checkpoint_169.pt"
                and type(checkpoint.get("bytes")) is int and checkpoint["bytes"] > 0
                and checkpoint.get("sha256") == final_sha, "final checkpoint hash mismatch")
        require(len(updates) == UPDATES and updates[-1]["checkpoint_sha256"] == final_sha,
                "final169/update checkpoint mismatch")
        if checkpoint_sha is not None:
            require(final_sha == checkpoint_sha, "caller-pinned final checkpoint mismatch")
    expected_seeds = (set(range(FIRST, FIRST + COUNT)) | set(SMOKES)) - historical
    complete = (len(points) == UPDATES and successful == expected_seeds
                and set(episodes) == expected_seeds and frozen is not None
                and not partials and not failed and not unsafe)
    return dict(state="COMPLETE" if complete else "FAILED" if failed or unsafe else "PARTIAL",
                status=status, points=points, records=records, episodes=episodes, files=files,
                safety=safety, reserved_sim_s=len(records) * 900,
                actual_sim_s=math.fsum(r["actual"] for r in records.values()
                                      if r["actual"] is not None),
                issues=issues, observed_iteration=len(updates), checkpoint_sha256=final_sha)


def evaluation(path, pinned_sha, report):
    require(bool(path) == bool(pinned_sha), "evaluation requires both JSON and explicit SHA256")
    if path is None:
        return None
    digest(pinned_sha)
    raw = ops.checked(Path(path).absolute()).read_bytes()
    require(ops.sha(raw) == pinned_sha, "evaluation JSON hash mismatch")
    data = json.loads(raw)
    require(report["state"] == "COMPLETE", "evaluation requires verified complete final169")
    require(data.get("schema") == "selmappo2.evaluation/1"
            and data.get("contract") == "S-C1SWITCH2" and data.get("status") == "COMPLETE"
            and data.get("checkpoint_sha256") == report["checkpoint_sha256"],
            "evaluation contract/checkpoint mismatch")
    digest(data.get("contract_sha256"))
    worlds = [dict(seed=seed, penetration=p, demand=d)
              for seed, (p, d) in zip(PANEL, PANEL_CELLS)]
    require(data.get("worlds") == worlds, "evaluation differs from peer-owned ordered panel")
    results = data.get("results", [])
    require(len(results) == len(worlds), "evaluation panel incomplete")
    by_seed = {}
    for row in results:
        seed = row.get("world", {}).get("seed")
        require(type(seed) is int and seed in PANEL and seed not in by_seed,
                "duplicate/unowned evaluation seed")
        by_seed[seed] = row
    for world in worlds:
        episode(by_seed[world["seed"]], world, "evaluation", UPDATES)
    return dict(sha256=pinned_sha, contract_sha256=data["contract_sha256"],
                J600_mean=math.fsum(r["J600"] for r in results) / len(results),
                safety={k: math.fsum(r["components"][k] for r in results) for k in SAFETY})


def curve_svg(points):
    """Standard-library SVG, with actual cumulative sim-s including two smokes."""
    require(bool(points), "cannot draw a learning curve with zero complete updates")
    xs, ys = [r["actual_sim_s"] for r in points], [r["J600_mean"] for r in points]
    xmin, xmax = 0, max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max((ymax - ymin) * .1, abs(ymax) * .01, 1.)
    ymin, ymax = max(0, ymin - pad), ymax + pad
    def x(v):
        return 100 + 750 * (v - xmin) / (xmax - xmin)

    def y(v):
        return 350 - 290 * (v - ymin) / (ymax - ymin)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="440" '
           'viewBox="0 0 920 440" role="img" aria-labelledby="title desc">',
           '<title id="title">SELMAPPO2 training learning curve</title>',
           '<desc id="desc">Mean J600 per complete update of 46 fresh training episodes; '
           'lower is better. Cumulative actual simulation includes warmup and two smokes. '
           'This is training performance, not held-out evaluation.</desc>',
           '<rect width="920" height="440" fill="white"/>',
           '<g font-family="sans-serif" font-size="13" fill="#222">',
           '<text x="100" y="28">SELMAPPO2 — training J600 (lower is better)</text>']
    for i in range(5):
        xv, yv = xmax * i / 4, ymin + (ymax - ymin) * i / 4
        out.extend((f'<path d="M{x(xv):.2f},60 V350" stroke="#eee"/>',
                    f'<text x="{x(xv):.2f}" y="375" text-anchor="middle">{xv:,.0f}</text>',
                    f'<text x="90" y="{y(yv) + 4:.2f}" text-anchor="end">{yv:,.1f}</text>'))
    coords = " ".join(f"{x(a):.2f},{y(b):.2f}" for a, b in zip(xs, ys))
    out.append(f'<polyline points="{coords}" fill="none" stroke="#176b9b" stroke-width="2"/>')
    for a, b in zip(xs, ys):
        out.append(f'<circle cx="{x(a):.2f}" cy="{y(b):.2f}" r="2.5" fill="#176b9b"/>')
    out.extend(('<path d="M100,60 V350 H850" fill="none" stroke="#222"/>',
                '<text x="475" y="410" text-anchor="middle">Cumulative actual simulation (s)</text>',
                '<text transform="translate(22,205) rotate(-90)" text-anchor="middle">Mean J600</text>',
                '</g></svg>'))
    return "\n".join(out) + "\n"


def curve_csv(points):
    stream = io.StringIO(newline="")
    columns = ["iteration", "actual_sim_s", "charged_sim_s", "J600_mean", "J600_min",
               "J600_max", "J600_std", "checkpoint_sha256"]
    columns += sorted({k for row in points for k, v in row.items()
                       if k not in columns and (number(v) or isinstance(v, str))})
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(points)
    return stream.getvalue()


def markdown(report, provenance, peer=None, assets=None):
    status, records = report["status"], report["records"]
    alive = status.get("alive")
    lines = ["# SELMAPPO2 results", "", f"**Training status: {report['state']}**", "",
             f"Evidence: {provenance}", f"Remote root: `{status['root']}/training`.",
             f"Dispatch PID: `{status['pid']}`; alive at observation: `{alive}`; "
             f"observation timestamp (ns): `{status.get('timestamp_ns')}`.", "",
             "## Training progress", "",
             f"Verified complete updates: {len(report['points'])}/169 (46 fresh episodes/update).",
             f"Update records observed: {report['observed_iteration'] if report['observed_iteration'] is not None else 'unknown'}."]
    if report["actual_sim_s"] is None:
        lines.append("Full coordinator actual/reserved totals: unknown from status tails.")
    else:
        prod = sum(r["terminal"] == "RESULT" and r["start"]["mode"] == "training"
                   for r in records.values())
        smoke = sum(r["terminal"] == "RESULT" and r["start"]["mode"] == "smoke"
                    for r in records.values())
        pending = sum(r["terminal"] is None for r in records.values())
        failed = sum(r["terminal"] == "FAILED" for r in records.values())
        unknown = sum(r["actual"] is None for r in records.values())
        lines.extend((f"Coordinator successful episodes: {prod}/7,774 production + {smoke}/2 smoke.",
                      f"Reservations: {len(records)}; unresolved: {pending}; failed: {failed}; "
                      f"actual not yet receipted: {unknown}.",
                      f"Raw newly incurred training + smoke simulation: "
                      f"{report['actual_sim_s']:,.1f} actual s; "
                      f"{report['reserved_sim_s']:,} reserved s (no refunds)."))
    lines.extend(("", "Full planned production simulation: 6,996,600 s; smoke: 1,800 s; "
                  "total training + smoke: 6,998,400 s. The separate 16-world evaluation "
                  "would cost 14,400 s (combined allocation 7,012,800 s).",
                  "Inherited controller-library training and world-model (WM) costs: unknown; "
                  "excluded from raw new-run simulation, not assumed zero.",
                  "Reported simulation includes 300 s native warmup + 600 s control per episode.",
                  "Wall times in the curve CSV are measured per-update coordinator wall times; "
                  "they are not CPU-hours or inherited training costs.", "", "## Learning curve", ""))
    if report["points"]:
        lines.append("J600 is full-team delay over the 600 s control interval; lower is better. "
                     "Each point uses a verified complete 46-episode update. The x-axis is "
                     "cumulative actual simulation seconds, including two smoke episodes and warmup. "
                     "Fresh scheduled worlds vary between updates; this is training progress, "
                     "not an efficacy comparison or checkpoint-selection rule.")
        if assets:
            lines.extend(("", f"![Training learning curve]({assets[0]})", "",
                          f"[Complete-update metrics CSV]({assets[1]})"))
    else:
        lines.append("No verified complete updates are available; no learning curve is fabricated.")
    lines.extend(("", "## Safety receipts", ""))
    if report["episodes"]:
        lines.append(f"Collected complete episode result receipts: {len(report['episodes'])}. "
                     + "; ".join(f"{k}: {v:,.3f}" for k, v in report["safety"].items()) + ".")
        lines.append("These are recorded control-interval components only. Warmup safety and "
                     "dropped-vehicle counters are not separately recorded in result.jsonl; "
                     "unrecorded metrics are not asserted to be zero.")
    else:
        lines.append("Episode-level safety receipts have not yet been collected.")
    lines.extend(("", "## Frozen final169", ""))
    lines.append(f"Verified checkpoint SHA256: `{report['checkpoint_sha256']}`."
                 if report["checkpoint_sha256"] else "Final169 manifest/checkpoint hash not yet verified.")
    lines.extend(("", "## Evaluation", ""))
    if peer is None:
        lines.append("**PENDING** — peer-owned S-C1SWITCH2 panel seeds 969136..969151. "
                     "No byte-pinned complete evaluation JSON has been supplied. "
                     "No final efficacy claim can be made from training progress.")
    else:
        lines.extend((f"Complete byte-pinned S-C1SWITCH2 panel: 16/16, seeds 969136..969151; "
                      f"mean J600: {peer['J600_mean']:,.6f}.",
                      f"Evaluation JSON SHA256: `{peer['sha256']}`; "
                      f"peer contract SHA256: `{peer['contract_sha256']}`.",
                      "Evaluation safety: " + "; ".join(f"{k}: {v:,.3f}"
                          for k, v in peer["safety"].items()) + ".",
                      "This absolute panel result alone does not establish improvement over baselines."))
    if report["issues"]:
        lines.extend(("", "## Evidence limitations", ""))
        lines += [f"- {issue}" for issue in report["issues"]]
    return "\n\n".join(lines) + "\n"


def exclusive_text(path, text):
    ops.checked(path.parent, directory=True)
    require(not ops.FORBIDDEN.search(str(path)), "unsafe output path")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def publish(output, report, provenance, peer):
    output = Path(output).absolute()
    require(output.suffix == ".md" and not ops.FORBIDDEN.search(str(output)), "unsafe report path")
    ops.checked(output.parent, directory=True)
    if output.exists() or output.is_symlink():
        ops.checked(output)
    assets = None
    if report["points"]:
        # Exclusive assets ensure prior reports can always be reproduced.
        token = f"selmappo2_curve_{time.time_ns()}_{uuid.uuid4().hex}"
        svg, csv_path = output.parent / (token + ".svg"), output.parent / (token + ".csv")
        exclusive_text(svg, curve_svg(report["points"]))
        exclusive_text(csv_path, curve_csv(report["points"]))
        assets = (svg.name, csv_path.name)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown(report, provenance, peer, assets))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("collect", "report"))
    parser.add_argument("--build", default=BUILD)
    parser.add_argument("--pid", type=int, default=PID)
    parser.add_argument("--input", help="one local immutable STATUS or COLLECTION JSON")
    parser.add_argument("--output", help="explicit Markdown publication path; otherwise stdout")
    parser.add_argument("--collection-dir", type=Path, default=ops.EVIDENCE)
    parser.add_argument("--checkpoint-sha256", help="optional additional final169 hash pin")
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--evaluation-sha256", help="required byte pin for evaluation JSON")
    args = parser.parse_args(argv)
    if args.action == "collect":
        if any((args.input, args.output, args.checkpoint_sha256,
                args.evaluation_json, args.evaluation_sha256)):
            parser.error("collect accepts only --build, --pid and --collection-dir")
        path = collect(args.build, args.pid, args.collection_dir.absolute())
        print(path)
        return path
    if not args.input:
        parser.error("report requires --input (no implicit remote collection)")
    path = ops.checked(Path(args.input).absolute())
    raw = path.read_bytes()
    report = analyze(json.loads(raw), args.build, args.pid, args.checkpoint_sha256)
    peer = evaluation(args.evaluation_json, args.evaluation_sha256, report)
    provenance = f"`{path.name}`; SHA256 `{ops.sha(raw)}`."
    if args.output:
        result = publish(args.output, report, provenance, peer)
        print(result)
        return result
    print(markdown(report, provenance, peer), end="")
    return report


if __name__ == "__main__":
    main()
