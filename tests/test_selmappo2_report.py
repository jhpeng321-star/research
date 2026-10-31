"""Synthetic provenance/accounting tests: no SSH, trainer import or simulator."""
import base64
import copy
import csv
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zlib

import pytest

from tools.selmappo2 import report as r


def put(payload, name, data):
    if not isinstance(data, bytes):
        data = (r.ops.canonical(data) + "\n").encode()
    payload["files"][name] = dict(bytes=len(data), sha256=r.ops.sha(data),
                                 base64=base64.b64encode(data).decode())


def put_rows(payload, name, rows, suffix=b""):
    put(payload, name, b"".join((r.ops.canonical(row) + "\n").encode() for row in rows) + suffix)


def get_rows(payload, name):
    return r.jsonl(base64.b64decode(payload["files"][name]["base64"]))[0]


def episode(world, mode, iteration, j=600.):
    return dict(world=world, mode=mode, iteration=iteration, decisions=30,
                actual_sim_s=900., J600=j, episode_return=-j, elapsed_s=2.,
                components=dict(collisions=0., teleports=0., hard_brake_vs=4.,
                                ttc_below_3_vs=3., road_delay_vs=j - 100.,
                                source_wait_vs=100., team_delay_vs=j))


def collection(updates=1, frozen=False, ops_text=None):
    source_files = {name: name.encode() for name in (
        "REGISTRATION.md", "tools/selmappo/__init__.py", "tools/selmappo/protocol.py",
        "tools/selmappo2/shared.py", "tools/selmappo2/mixed.py", "tools/selmappo2/training.py")}
    if ops_text is not None:
        source_files["tools/selmappo2/ops.py"] = ops_text
    sources = {name: r.ops.sha(data) for name, data in source_files.items()}
    binding = r.ops.binding_for(sources)
    build = "build_" + binding
    own = dict(owner=r.ops.OWNER, host=r.ops.HOST, binding=binding,
               root=r.ops.PARENT + "/" + build, pid=r.PID)
    payload = dict(own, schema=r.SCHEMA, status=dict(own, alive=False, dispatched=True,
                                                  timestamp_ns=100),
                   timestamp_ns=101, files={}, checkpoint=None, new_sumo_seconds=0)
    put(payload, "DISPATCH.json", dict(own, out=own["root"] + "/training"))
    integration = dict(schema="selmappo.integration/2", status="CPU_INTEGRATION_PASS",
                       owner=r.ops.OWNER, binding=binding, files=sources,
                       registration_sha256=sources["REGISTRATION.md"], variant="per_cav",
                       protocol="S-C1SWITCH2/v2", cadence_s=20,
                       library=["native", "ctrl0", "mappo", "pi", "circles", "wall"],
                       design_sha256=r.ops.DESIGN_SHA)
    put(payload, "INTEGRATION.json", integration)
    bindings = dict(integration_sha256=payload["files"]["INTEGRATION.json"]["sha256"],
                    design_sha256=r.ops.DESIGN_SHA)
    trainer_sources = {Path(n).stem: v for n, v in sources.items()
                       if n.startswith("tools/selmappo/")}
    trainer_sources.update({"selmappo2." + n: sources[f"tools/selmappo2/{n}.py"]
                            for n in ("shared", "mixed", "training")})
    put(payload, "training/OWNER.json", dict(pid=r.PID, variant="per_cav", workers=12,
        optimization_seed=1, bindings=bindings, sources=trainer_sources,
        registration_sha256=sources["REGISTRATION.md"]))
    put(payload, "training/TRAIN_OWNER.json", dict(pid=r.PID))
    events = []
    for seed in [*r.SMOKES, *range(r.FIRST, r.FIRST + updates * r.BATCH)]:
        mode = "smoke" if seed in r.SMOKES else "training"
        iteration = 0 if mode == "smoke" else (seed - r.FIRST) // r.BATCH
        world = r.expected_world(seed, mode)
        events.extend((dict(event="START", seed=seed, mode=mode, world=world,
                            reserved_sim_s=900, panel=[]),
                       dict(event="ACTUAL", seed=seed, actual_sim_s=900),
                       dict(event="RESULT", seed=seed)))
        put_rows(payload, f"training/{seed}/result.jsonl", [episode(world, mode, iteration)])
    put_rows(payload, "training/coordinator.jsonl", events)
    points = [dict(iteration=i, actual_sim_s=(2 + i * r.BATCH) * 900,
                   charged_sim_s=(2 + i * r.BATCH) * 900, J600_mean=600.,
                   J600_min=600., J600_max=600., J600_std=0.,
                   simulation_wall_s=5., total_wall_s=6., policy_loss=-.02,
                   checkpoint_sha256=r.ops.sha(str(i).encode())) for i in range(1, updates + 1)]
    put_rows(payload, "training/training.jsonl", points)
    if frozen:
        sha = points[-1]["checkpoint_sha256"]
        put(payload, "training/FROZEN_FINAL.json", dict(iteration=169, checkpoint_sha256=sha,
            sources=trainer_sources, bindings=bindings))
        payload["checkpoint"] = dict(path="training/checkpoint_169.pt", bytes=300, sha256=sha)
    return payload, build


@pytest.fixture(scope="module")
def final():
    return collection(169, frozen=True)


def test_complete_requires_exact_full_allocation_and_matching_three_hashes(final):
    payload, build = final
    report = r.analyze(payload, build)
    assert report["state"] == "COMPLETE"
    assert len(report["points"]) == 169
    assert len(report["episodes"]) == 7776
    assert report["actual_sim_s"] == report["reserved_sim_s"] == 6_998_400
    assert report["safety"]["hard_brake_vs"] == 7776 * 4
    text = r.markdown(report, "synthetic")
    assert "PENDING" in text and "969136..969151" in text
    assert "Inherited controller-library" in text and "unknown" in text
    with pytest.raises(ValueError, match="caller-pinned"):
        r.analyze(payload, build, checkpoint_sha="0" * 64)


def amended_collection(payload):
    payload = copy.deepcopy(payload)
    seed = r.ops.HISTORICAL_SEED
    events = [dict(event="START", seed=seed, mode="smoke", world=r.expected_world(seed, "smoke"),
                   reserved_sim_s=900, panel=[]),
              dict(event="ACTUAL", seed=seed, actual_sim_s=300.),
              dict(event="FAILED", seed=seed, actual_sim_s=300., error=r.ops.HISTORICAL_ERROR)]
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in events).encode()
    assert r.ops.sha(raw) == r.ops.HISTORICAL_LEDGER_SHA
    remaining = get_rows(payload, "training/coordinator.jsonl")[3:]
    put(payload, "training/coordinator.jsonl", raw + b"".join(
        (r.ops.canonical(row) + "\n").encode() for row in remaining))
    payload["files"].pop(f"training/{seed}/result.jsonl")
    points = get_rows(payload, "training/training.jsonl")
    for row in points:
        row["actual_sim_s"] -= 600
    put_rows(payload, "training/training.jsonl", points)
    dispatch = json.loads(base64.b64decode(payload["files"]["DISPATCH.json"]["base64"]))
    dispatch["command"] = ["python", "-m", "tools.selmappo2.training", "--continuation"]
    put(payload, "DISPATCH.json", dispatch)
    owner = json.loads(base64.b64decode(payload["files"]["training/OWNER.json"]["base64"]))
    receipt = {k: payload[k] for k in ("owner", "host", "root", "binding")}
    receipt.update(original=dict(root=r.ops.PARENT + "/build_" + r.ops.HISTORICAL_BINDING,
        binding=r.ops.HISTORICAL_BINDING, pid=r.ops.HISTORICAL_PID,
        ledger_sha256=r.ops.HISTORICAL_LEDGER_SHA, reserved_sim_s=900,
        actual_sim_s=300, seed=seed, status="FAILED"), inherited_reserved_sim_s=900,
        inherited_actual_sim_s=300, skipped_seed=seed, only_unused_smoke_seed=r.SMOKES[1],
        production_episodes=r.COUNT, cap_sim_s=7012800, evaluation_reserved_unstarted_sim_s=14400,
        registration_sha256=owner["registration_sha256"], command=dispatch["command"],
        authority="- 01:48 MAIN ACKNOWLEDGEMENTS / DECISIONS: (2) S-SELMAPPO2 same-cap continuation "
                  "ACKNOWLEDGED (unused second smoke world only, then production).")
    put(payload, "CONTINUATION.json", receipt)
    return payload


def test_amended_final_accepts_only_hash_bound_historical_failure_and_retains_charge(final):
    payload, build = final
    amended = amended_collection(payload)
    report = r.analyze(amended, build)
    assert report["state"] == "COMPLETE" and len(report["points"]) == 169
    assert report["reserved_sim_s"] == 6_998_400
    assert report["actual_sim_s"] == 6_997_800
    assert len(report["episodes"]) == 7775
    assert any("historical smoke failure" in issue for issue in report["issues"])


def test_amended_collection_rejects_ledger_tampering_and_new_failure():
    payload, build = collection(1)
    amended = amended_collection(payload)
    bad = copy.deepcopy(amended)
    raw = base64.b64decode(bad["files"]["training/coordinator.jsonl"]["base64"])
    put(bad, "training/coordinator.jsonl", raw.replace(b"300.0", b"301.0"))
    with pytest.raises(ValueError, match="historical failure hash"):
        r.analyze(bad, build)
    bad = copy.deepcopy(amended)
    raw = base64.b64decode(bad["files"]["training/coordinator.jsonl"]["base64"])
    # Keep the pinned prefix byte-identical; a different failure still fails.
    lines = raw.splitlines(keepends=True)
    lines[5] = (r.ops.canonical(dict(event="FAILED", seed=r.SMOKES[1],
                                   actual_sim_s=900, error="new fault")) + "\n").encode()
    put(bad, "training/coordinator.jsonl", b"".join(lines))
    report = r.analyze(bad, build)
    assert report["state"] == "FAILED" and not report["points"]


@pytest.mark.parametrize("target", ["checkpoint", "update", "frozen_sources"])
def test_reject_final_provenance_mismatches(final, target):
    payload, build = final
    payload = copy.deepcopy(payload)
    if target == "checkpoint":
        payload["checkpoint"]["sha256"] = "0" * 64
    elif target == "update":
        rows = get_rows(payload, "training/training.jsonl")
        rows[-1]["checkpoint_sha256"] = "0" * 64
        put_rows(payload, "training/training.jsonl", rows)
    else:
        obj = json.loads(base64.b64decode(payload["files"]["training/FROZEN_FINAL.json"]["base64"]))
        obj["sources"] = {}
        put(payload, "training/FROZEN_FINAL.json", obj)
    with pytest.raises(ValueError, match="mismatch"):
        r.analyze(payload, build)


def test_missing_episode_suppresses_curve_and_final_completion(final):
    payload, build = final
    payload = copy.deepcopy(payload)
    del payload["files"][f"training/{r.FIRST}/result.jsonl"]
    report = r.analyze(payload, build)
    assert report["state"] == "PARTIAL" and not report["points"]
    assert "No verified complete updates" in r.markdown(report, "synthetic")
    with pytest.raises(ValueError, match="zero complete"):
        r.curve_svg(report["points"])


def test_all_updates_without_frozen_final_remain_partial(final):
    payload, build = final
    payload = copy.deepcopy(payload)
    del payload["files"]["training/FROZEN_FINAL.json"]
    payload["checkpoint"] = None
    report = r.analyze(payload, build)
    assert len(report["points"]) == 169
    assert report["state"] == "PARTIAL" and report["checkpoint_sha256"] is None


def test_failed_and_inflight_accounting_retains_reserved_and_unknown_actual():
    payload, build = collection(0)
    rows = get_rows(payload, "training/coordinator.jsonl")
    for seed in (r.FIRST, r.FIRST + 1):
        rows.append(dict(event="START", seed=seed, world=r.expected_world(seed, "training"),
                         mode="training", reserved_sim_s=900, panel=[]))
    rows.extend((dict(event="ACTUAL", seed=r.FIRST, actual_sim_s=317.5),
                 dict(event="FAILED", seed=r.FIRST, actual_sim_s=317.5, error="synthetic")))
    put_rows(payload, "training/coordinator.jsonl", rows)
    report = r.analyze(payload, build)
    assert report["state"] == "FAILED"
    assert report["reserved_sim_s"] == 3600
    assert report["actual_sim_s"] == 2117.5
    text = r.markdown(report, "synthetic")
    assert "actual not yet receipted: 1" in text and "no refunds" in text


@pytest.mark.parametrize("mutation", ["hash", "duplicate", "world", "statistics", "foreign_path"])
def test_reject_corruption_duplicates_world_drift_and_unowned_paths(mutation):
    payload, build = collection()
    if mutation == "hash":
        payload["files"]["training/training.jsonl"]["sha256"] = "0" * 64
    elif mutation == "duplicate":
        rows = get_rows(payload, "training/coordinator.jsonl")
        put_rows(payload, "training/coordinator.jsonl", rows + [rows[0]])
    elif mutation == "world":
        rows = get_rows(payload, "training/coordinator.jsonl")
        rows[0]["world"]["demand"] = 9999
        put_rows(payload, "training/coordinator.jsonl", rows)
    elif mutation == "statistics":
        rows = get_rows(payload, "training/training.jsonl")
        rows[0]["J600_mean"] = 123.
        put_rows(payload, "training/training.jsonl", rows)
    else:
        put(payload, "training/../../other/result.jsonl", b"secret")
    with pytest.raises(ValueError):
        r.analyze(payload, build)


def test_unterminated_append_is_visible_but_never_a_complete_update():
    payload, build = collection()
    rows = get_rows(payload, "training/training.jsonl")
    put_rows(payload, "training/training.jsonl", rows, b'{"iteration":2')
    report = r.analyze(payload, build)
    assert len(report["points"]) == 1 and report["state"] == "PARTIAL"
    assert any("Unterminated" in issue for issue in report["issues"])
    put_rows(payload, "training/training.jsonl", rows, b"not-json\n")
    with pytest.raises(ValueError):
        r.analyze(payload, build)


def test_status_tails_cannot_be_used_for_curves_or_complete_status():
    payload, build = collection()
    status = payload["status"]
    status["summaries"] = {"training/training.jsonl": dict(bytes=999999,
        tail='partial-line\n{"iteration":169,"J600_mean":1}\n')}
    report = r.analyze(status, build)
    assert report["observed_iteration"] == 169
    assert report["state"] == "PARTIAL" and report["points"] == []
    assert report["actual_sim_s"] is None
    status["pid"] += 1
    with pytest.raises(ValueError, match="ownership"):
        r.analyze(status, build)


def test_safety_nonzero_blocks_completion_and_missing_is_not_zero():
    payload, build = collection()
    name = f"training/{r.FIRST}/result.jsonl"
    rows = get_rows(payload, name)
    rows[0]["components"]["collisions"] = 1.
    put_rows(payload, name, rows)
    assert r.analyze(payload, build)["state"] == "FAILED"
    del rows[0]["components"]["collisions"]
    put_rows(payload, name, rows)
    with pytest.raises(ValueError, match="safety"):
        r.analyze(payload, build)


def test_offline_cli_publishes_svg_and_csv_without_network_or_input_changes(tmp_path, monkeypatch):
    from tools.selmappo2 import shared

    def forbidden(*args, **kwargs):
        pytest.fail("offline reporting attempted collection/shared imports")

    monkeypatch.setattr(shared, "load", forbidden)
    monkeypatch.setattr(r, "collect", forbidden)
    payload, build = collection()
    source, output = tmp_path / "collection.json", tmp_path / "SELMAPPO2.md"
    raw = (r.ops.canonical(payload) + "\n").encode()
    source.write_bytes(raw)
    argv = ["--action", "report", "--input", str(source), "--build", build,
            "--output", str(output)]
    r.main(argv)
    text = output.read_text(encoding="utf-8")
    assert "Training status: PARTIAL" in text and r.ops.sha(raw) in text
    svg = next(tmp_path.glob("*.svg"))
    ET.fromstring(svg.read_text(encoding="utf-8"))
    csv_path = next(tmp_path.glob("*.csv"))
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["actual_sim_s"] == "43200" and rows[0]["policy_loss"] == "-0.02"
    assert "including two smoke" in text and "No final efficacy claim" in text
    original_assets = {p: p.read_bytes() for p in (svg, csv_path)}
    r.main(argv)
    assert len(list(tmp_path.glob("*.svg"))) == 2
    assert source.read_bytes() == raw
    assert all(p.read_bytes() == data for p, data in original_assets.items())


def test_evaluation_requires_pin_exact_peer_panel_and_complete_final(tmp_path, final):
    payload, build = final
    report = r.analyze(payload, build)
    worlds = [dict(seed=s, penetration=p, demand=d)
              for s, (p, d) in zip(r.PANEL, r.PANEL_CELLS)]
    data = dict(schema="selmappo2.evaluation/1", contract="S-C1SWITCH2", status="COMPLETE",
                contract_sha256="a" * 64, checkpoint_sha256=report["checkpoint_sha256"],
                worlds=worlds, results=[episode(w, "evaluation", 169) for w in worlds])
    path = tmp_path / "evaluation.json"
    path.write_text(r.ops.canonical(data), encoding="utf-8")
    sha = r.ops.sha(path.read_bytes())
    with pytest.raises(ValueError, match="both"):
        r.evaluation(path, None, report)
    with pytest.raises(ValueError, match="hash mismatch"):
        r.evaluation(path, "0" * 64, report)
    peer = r.evaluation(path, sha, report)
    assert peer["J600_mean"] == 600.
    assert "16/16" in r.markdown(report, "synthetic", peer)
    with pytest.raises(ValueError, match="complete final169"):
        r.evaluation(path, sha, dict(report, state="PARTIAL"))
    data["results"][0] = data["results"][1]
    path.write_text(r.ops.canonical(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        r.evaluation(path, r.ops.sha(path.read_bytes()), report)


def test_encoded_collector_reads_only_fixed_and_ledger_paths_and_streams_final_hash(tmp_path):
    root = tmp_path / "build"
    root.mkdir()
    training = root / "training"
    training.mkdir()
    seed = r.SMOKES[0]
    episode_dir = training / str(seed)
    episode_dir.mkdir()
    (root / "DISPATCH.json").write_text(json.dumps(dict(out=str(training))))
    (training / "OWNER.json").write_text("{}")
    (training / "FROZEN_FINAL.json").write_text("{}")
    (training / "checkpoint_169.pt").write_bytes(b"stream-only, never transport these bytes")
    (training / "coordinator.jsonl").write_text(json.dumps(dict(event="START", seed=seed,
        mode="smoke", world=dict(seed=seed), reserved_sim_s=900, panel=[])) + "\n")
    (episode_dir / "result.jsonl").write_text("{}\n")
    opened, bootstrap = [], []

    def checked(path):
        opened.append(path.relative_to(root).as_posix())
        return r.ops.checked(path)

    def status(action, sources, binding, build):
        bootstrap.append(action)
        print("STATUS_RECEIPT", json.dumps(dict(dispatched=True, pid=r.PID)))

    namespace = dict(vars(r.ops), _remote_main=status, checked=checked, PARENT=str(tmp_path))
    exec(inspect.getsource(r._remote_collect), namespace)
    capture = io.StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(capture):
        namespace["_remote_collect"]({}, "binding", "build", r.PID)
    result = r.ops.tagged(capture.getvalue(), "COLLECTION_RECEIPT")[0]
    assert bootstrap == ["status"]
    assert set(opened) == set(r.FIXED) | {
        f"training/{seed}/result.jsonl", "training/checkpoint_169.pt"}
    assert "training/checkpoint_169.pt" not in result["files"]
    assert result["checkpoint"]["sha256"] == r.ops.sha((training / "checkpoint_169.pt").read_bytes())
    with pytest.raises(ValueError, match="PID"):
        namespace["_remote_collect"]({}, "binding", "build", r.PID + 1)
    (training / "coordinator.jsonl").write_text(json.dumps(dict(event="START", seed=969136,
        mode="evaluation", world=dict(seed=969136), reserved_sim_s=900, panel=[])) + "\n")
    with pytest.raises(ValueError, match="ledger path"):
        namespace["_remote_collect"]({}, "binding", "build", r.PID)


def test_explicit_collect_uses_stored_ops_shared_transport_and_append_only_receipts(tmp_path, monkeypatch):
    from tools.selmappo2 import shared
    ops_text = b"# frozen ops source, not current working tree\n"
    payload, build = collection(ops_text=ops_text)
    sources = json.loads(base64.b64decode(payload["files"]["INTEGRATION.json"]["base64"]))["files"]
    binding = payload["binding"]
    bundle = dict(binding=binding, sources=sources,
                  files={name: base64.b64encode(ops_text if name == "tools/selmappo2/ops.py"
                                               else name.encode()).decode() for name in sources})
    calls = []

    def run(host, code, **kwargs):
        packed = code.split("base64.b64decode(", 1)[1].split(")", 1)[0]
        import ast
        script = zlib.decompress(base64.b64decode(ast.literal_eval(packed))).decode()
        assert ops_text.decode() in script and "_remote_main(\"status\"" in script
        assert "_remote_dispatch(" not in script
        calls.append((host, kwargs))
        return SimpleNamespace(stdout="COLLECTION_RECEIPT " + json.dumps(payload) + "\n")

    monkeypatch.setattr(r.ops, "read_json", lambda path: bundle)
    monkeypatch.setattr(shared, "load", lambda root: SimpleNamespace(__name__="fake_shared"))
    original_import = r.importlib.import_module
    monkeypatch.setattr(r.importlib, "import_module", lambda name: SimpleNamespace(run=run)
                       if name == "fake_shared.remote" else original_import(name))
    paths = [r.main(["--action", "collect", "--build", "build_" + binding,
                     "--collection-dir", str(tmp_path)]) for _ in range(2)]
    assert paths[0] != paths[1] and all(path.exists() for path in paths)
    assert calls == [("cpuQ", dict(timeout=300))] * 2
    assert json.loads(paths[0].read_text())["schema"] == r.SCHEMA


def test_cli_requires_explicit_local_input_and_rejects_mixed_actions():
    with pytest.raises(SystemExit):
        r.main(["--action", "report"])
    with pytest.raises(SystemExit):
        r.main(["--action", "collect", "--input", "unused"])
