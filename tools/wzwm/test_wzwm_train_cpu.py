"""CPU tests of the wz3to2 WM fine-tune tooling (WM-E): tiny fakes, no data, no GPU, no SUMO.

Covers: world-seed rules and the split by world, re-initialisation correctness (tiny tensors and
the real WM-S model of the fine-tune config), resume (trimmed stages + a fake trainer that crashes
once), the truth-input guard, the phase overrides on the real configs and the checkpoint manifest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")

from tools.wzwm import estimate, guard, plan, reinit, run, worlds  # noqa: E402


# ------------------------------------------------------------------ worlds / split ---
def _manifest(entries, **kw):
    return {"schema": "wms_c4_episode_source/1", "split": "TRAIN", "map": "wz3to2",
            "episodes": [{"world": w, "split_role": r, "arm": a, "dir": f"episodes/{w}__{a}"}
                         for w, r, a in entries], **kw}


def test_world_seed_rules():
    assert worlds.world_seed("w989103m3200p60t300") == 989103
    assert worlds.world_seed(989150) == 989150
    with pytest.raises(worlds.WorldError):
        worlds.world_seed("w12m3200")
    assert worlds.problems(989150) == []                       # WMDATA range
    assert worlds.problems(989300) == []                       # TRAIN range
    for bad in (989200, 989216, 989351, 989401, 989580, 989599):   # WM-D reserved
        assert "reserved for WM-D qualification" in worlds.problems(bad)
    assert worlds.problems(989650)                              # held-out
    assert worlds.problems(964100) and worlds.problems(962500) and worlds.problems(977001)
    assert worlds.problems(988999)                              # outside the wz registry


def test_split_by_world_deterministic_and_disjoint():
    ws = [f"w{989100 + i}m3200p60t300" for i in range(50)]
    tr, va = worlds.split_worlds(ws, val_frac=0.2)
    assert len(va) == 10 and len(tr) == 40
    assert not set(tr) & set(va) and set(tr) | set(va) == set(ws)
    assert (tr, va) == worlds.split_worlds(list(reversed(ws)), val_frac=0.2)
    assert worlds.split_worlds(ws[:1], val_frac=0.2) == (ws[:1], [])


def test_worlds_manifest_val_wins_and_lists_every_world():
    a = _manifest([("w989101m3200p60t300", "train", "native"), ("w989102m3200p60t300", "val", "native"),
                   ("w989103m2400p20t300", "train", "native")])
    b = _manifest([("w989101m3200p60t300", "val", "mappo_u0"), ("w989103m2400p20t300", "train", "mappo_u0")])
    m = worlds.worlds_manifest({"wz_native": a, "wz_mappo_u0": b})
    assert m["val_worlds"] == ["w989101m3200p60t300", "w989102m3200p60t300"]   # val wins
    assert m["train_worlds"] == ["w989103m2400p20t300"]
    assert m["train_seeds"] == [989103] and m["val_seeds"] == [989101, 989102]
    assert m["role_overridden_worlds"] == ["w989101m3200p60t300"]
    assert m["sources"]["wz_native"]["train_episodes"] == 1 and m["sources"]["wz_mappo_u0"]["val_episodes"] == 1
    assert worlds.overlap(m) == [] and len(m["sha256"]) == 64


@pytest.mark.parametrize("world", ["w989205m3200p60t300", "w989590m3200p60t300", "w989650m3200p60t300",
                                   "w964010m2700p20t300"])
def test_worlds_manifest_refuses_reserved_heldout_forbidden(world):
    a = _manifest([("w989101m3200p60t300", "train", "native"), ("w989102m3200p60t300", "val", "native"),
                   (world, "train", "native")])
    with pytest.raises(worlds.WorldError):
        worlds.worlds_manifest({"wz_native": a})


def test_worlds_manifest_refuses_no_val_seed_split_and_other_map():
    with pytest.raises(worlds.WorldError, match="no validation"):
        worlds.worlds_manifest({"s": _manifest([("w989101m3200p60t300", "train", "native"),
                                                ("w989102m3200p60t300", "train", "native")])})
    with pytest.raises(worlds.WorldError, match="both split sides"):
        worlds.worlds_manifest({"s": _manifest([("w989101m3200p60t300", "train", "native"),
                                                ("w989101m2400p20t300", "val", "native"),
                                                ("w989102m3200p60t300", "train", "native")])})
    with pytest.raises(worlds.WorldError, match="map"):
        worlds.worlds_manifest({"s": _manifest([("w989101m3200p60t300", "train", "n"),
                                                ("w989102m3200p60t300", "val", "n")], map="merge_original")})


# ------------------------------------------------------------------ reinit ---
def _tiny(seed):
    g = torch.Generator().manual_seed(seed)
    return {"driver_encoder.head.weight": torch.randn(3, 2, generator=g),
            "merge_pivot.tau_raw": torch.randn((), generator=g),
            "merge_yield.log_distance": torch.randn((), generator=g),
            "residual_net.accel.0.weight": torch.randn(4, 4, generator=g)}


def test_reinit_state_tiny():
    src, fresh = _tiny(1), _tiny(2)
    state, rep = reinit.reinit_state(src, fresh, ("merge_pivot.", "merge_yield."))
    assert rep["n_reinit"] == 2 and rep["n_kept"] == 2 and rep["n_new_in_model"] == 0
    for k in ("merge_pivot.tau_raw", "merge_yield.log_distance"):
        assert torch.equal(state[k], fresh[k]) and not torch.equal(state[k], src[k])
        assert rep["reinit"][k]["changed"]
    for k in ("driver_encoder.head.weight", "residual_net.accel.0.weight"):
        assert torch.equal(state[k], src[k]) and rep["kept_sha"][k] == reinit.tensor_sha(src[k])
    state["merge_pivot.tau_raw"] += 1               # clones, not aliases
    state["residual_net.accel.0.weight"] += 1
    assert torch.equal(fresh["merge_pivot.tau_raw"], _tiny(2)["merge_pivot.tau_raw"])
    assert torch.equal(src["residual_net.accel.0.weight"], _tiny(1)["residual_net.accel.0.weight"])


def test_reinit_state_fails_closed():
    src, fresh = _tiny(1), _tiny(2)
    with pytest.raises(reinit.ReinitError, match="absent"):
        reinit.reinit_state({**src, "old.head": torch.zeros(1)}, fresh, ("merge_pivot.",))
    bad = dict(fresh, **{"residual_net.accel.0.weight": torch.zeros(4, 5)})
    with pytest.raises(reinit.ReinitError, match="shape"):
        reinit.reinit_state(src, bad, ("merge_pivot.",))
    with pytest.raises(reinit.ReinitError, match="match no"):
        reinit.reinit_state(src, fresh, ("merge_pivot.", "ramp_speed_adapt."))
    state, rep = reinit.reinit_state(src, {**fresh, "new.bias": torch.ones(2)}, ("merge_pivot.",))
    assert rep["new_in_model"] == ["new.bias"] and torch.equal(state["new.bias"], torch.ones(2))


def test_reinit_checkpoint_sha_payload_write_once(tmp_path):
    src = tmp_path / "wm2.pt"
    torch.save({"format": "wm_s/1", "model": _tiny(1), "config": {"a": 1}, "history": [1]}, src)
    good = reinit.sha256_file(src)
    with pytest.raises(reinit.ReinitError, match="sha256"):
        reinit.reinit_checkpoint(src, tmp_path / "x.pt", expected_sha256="0" * 64, fresh=_tiny(2),
                                 prefixes=("merge_pivot.",))
    rec = reinit.reinit_checkpoint(src, tmp_path / "init.pt", expected_sha256=good, fresh=lambda: _tiny(2),
                                   prefixes=("merge_pivot.", "merge_yield."), receipt_path=tmp_path / "r.json")
    out = torch.load(tmp_path / "init.pt", map_location="cpu", weights_only=True)
    assert out["format"] == "wm_s/1" and out["config"] == {"a": 1} and "history" not in out
    assert out["wz_reinit"]["source_sha256"] == good
    assert sorted(out["wz_reinit"]["reinit_keys"]) == ["merge_pivot.tau_raw", "merge_yield.log_distance"]
    assert rec["out_sha256"] == reinit.sha256_file(tmp_path / "init.pt")
    assert json.loads((tmp_path / "r.json").read_text())["n_kept"] == 2
    with pytest.raises(FileExistsError):
        reinit.reinit_checkpoint(src, tmp_path / "init.pt", expected_sha256=good, fresh=_tiny(2),
                                 prefixes=("merge_pivot.",))


def test_reinit_real_wms_model_touches_only_geometry(tmp_path):
    """The real fine-tune model: a stand-in 'WM2' (seed-1 model + noise) re-initialised from the
    fresh seed model changes exactly the merge-geometry tensors and loads strictly."""
    sets = plan.phase_overrides("A", init_from="x.pt", sources=[plan.Source("wz_native", "/a", 0.5),
                                plan.Source("wz_mappo_u0", "/b", 0.5)], base="wz_native")
    fresh = reinit.fresh_state_from_config(plan.CONFIGS["A"], sets, repo=REPO)
    again = reinit.fresh_state_from_config(plan.CONFIGS["A"], sets, repo=REPO)
    assert all(torch.equal(fresh[k], again[k]) for k in fresh)        # deterministic fresh init
    fake_wm2 = {k: (v + 0.01 if v.is_floating_point() else v.clone()) for k, v in fresh.items()}
    state, rep = reinit.reinit_state(fake_wm2, fresh, reinit.DEFAULT_PREFIXES)
    geo = [k for k in fresh if k.startswith(reinit.DEFAULT_PREFIXES)]
    assert rep["n_reinit"] == len(geo) == 35 and rep["n_kept"] == len(fresh) - 35
    assert {k.split(".")[0] for k in geo} == {"merge_pivot", "merge_core", "merge_yield", "ramp_speed_adapt"}
    for k in fresh:
        want = fresh[k] if k in geo else fake_wm2[k]
        assert torch.equal(state[k], want), k
    assert any(k.startswith("driver_encoder.") for k in rep["kept_sha"])   # driver settings kept (learned)
    from wms.cli.assemble import build_model, build_physics
    cfg = run.load_cfg(plan.CONFIGS["A"], sets)
    model = build_model(cfg, build_physics(cfg))
    model.load_state_dict(state, strict=True)


# ------------------------------------------------------------------ resume ---
CUR_B = {"unroll_s": [120.0, 300.0], "stage_steps": [40, 60], "truncation_steps_per_stage": [120, 60]}
OPT_B = {"lr_per_stage": [5e-5, 3e-5], "batch_size_per_stage": [4, 2]}


@pytest.mark.parametrize("done,expect", [
    (0, {"stage_steps": [40, 60], "unroll_s": [120.0, 300.0], "lr": [5e-5, 3e-5]}),
    (10, {"stage_steps": [30, 60], "unroll_s": [120.0, 300.0], "lr": [5e-5, 3e-5]}),
    (40, {"stage_steps": [60], "unroll_s": [300.0], "lr": [3e-5]}),
    (55, {"stage_steps": [45], "unroll_s": [300.0], "lr": [3e-5]}),
])
def test_remaining_lists(done, expect):
    r = plan.remaining_lists(CUR_B, OPT_B, done)
    assert r["curriculum.stage_steps"] == expect["stage_steps"]
    assert r["curriculum.unroll_s"] == expect["unroll_s"]
    assert r["optim.lr_per_stage"] == expect["lr"]
    assert len(r["curriculum.truncation_steps_per_stage"]) == len(expect["stage_steps"])
    assert sum(r["curriculum.stage_steps"]) == 100 - done


def test_remaining_lists_finished():
    assert plan.remaining_lists(CUR_B, OPT_B, 100) == {}


FAKE_TRAINER = r'''
import hashlib, json, os, sys
from pathlib import Path
import torch
args = sys.argv[1:]
assert args[0] == "train", args
out = Path(args[args.index("--out") + 1]); out.mkdir(parents=True)
sets = dict(a.split("=", 1) for a in [args[i + 1] for i, x in enumerate(args) if x == "--set"])
(out / "ARGS.json").write_text(json.dumps(sets))
init = json.loads(sets["model.init_from"])
h = hashlib.sha256(Path(init).read_bytes()).hexdigest()
(out / "INIT_FROM.json").write_text(json.dumps({"sha256": h, "loaded": ["w"], "new_in_model": [],
    "shape_mismatch": [], "unused_from_checkpoint": [], "expanded": []}))
steps = sum(json.loads(sets.get("curriculum.stage_steps", os.environ["FAKE_STEPS"])))
marker = Path(os.environ["FAKE_MARKER"])
for i in range(1, steps + 1):
    with open(out / "train_log.jsonl", "a") as fh:
        fh.write(json.dumps({"step": i, "loss": 1.0 / i, "seconds": 0.01, "cuda_peak_mb": 5.0}) + "\n")
    torch.save({"format": "wm_s/1", "model": {"w": torch.tensor(float(i))}, "step": i}, out / "wm_s_latest.pt")
    if not marker.exists() and i == int(os.environ["FAKE_CRASH_AT"]):
        marker.write_text("crashed"); sys.exit(3)
torch.save({"format": "wm_s/1", "model": {"w": torch.tensor(9.0)}}, out / "wm_s.pt")
for n in ("physics.json", "RUN_SUMMARY.json"):
    (out / n).write_text("{}")
'''


def _fake_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "wms").mkdir(parents=True)
    (repo / "src" / "wms" / "__init__.py").write_text("")
    (repo / "src" / "wms" / "__main__.py").write_text(FAKE_TRAINER)
    (repo / "configs").mkdir()
    (repo / "configs" / "fake.toml").write_text("")
    return repo


def test_run_phase_resumes_after_crash(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    init = root / "INIT_WZ.pt"
    torch.save({"format": "wm_s/1", "model": {"w": torch.tensor(0.0)}}, init)
    monkeypatch.setenv("FAKE_STEPS", "[5]")
    monkeypatch.setenv("FAKE_CRASH_AT", "3")
    monkeypatch.setenv("FAKE_MARKER", str(tmp_path / "marker"))
    p = {"repo": str(repo), "python": sys.executable, "env": {}, "max_retries": 2,
         "phases": {"A": {"config": "configs/fake.toml", "overrides": ['runtime.device="cpu"'],
                          "curriculum": {"unroll_s": [120.0], "stage_steps": [5],
                                         "truncation_steps_per_stage": [120]},
                          "optim": {"lr_per_stage": [5e-5], "batch_size_per_stage": [4]}}}}
    done = run.run_phase(root, p, "A", str(init), reinit.sha256_file(init))
    a1, a2 = root / "phases/A/a1", root / "phases/A/a2"
    assert json.loads((a1 / "RESULT.json").read_text())["exit"] == 3
    att2 = json.loads((a2 / "ATTEMPT.json").read_text())
    assert att2["done_before"] == 3 and att2["resumed"]
    assert att2["init"] == str(a1 / "run" / "wm_s_latest.pt")
    args2 = json.loads((a2 / "run" / "ARGS.json").read_text())
    assert json.loads(args2["curriculum.stage_steps"]) == [2]
    assert done["final"] == str(a2 / "run" / "wm_s.pt") and done["attempts"] == 2
    assert run.run_phase(root, p, "A", str(init), "x") == done        # DONE is reused, no retrain
    assert run.resume_point(root / "phases/A")[0] == 5


def test_run_phase_stops_without_progress(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    init = root / "INIT_WZ.pt"
    torch.save({"format": "wm_s/1", "model": {}}, init)
    (repo / "src" / "wms" / "__main__.py").write_text("import sys; sys.exit(5)\n")
    p = {"repo": str(repo), "python": sys.executable, "env": {}, "max_retries": 2,
         "phases": {"A": {"config": "configs/fake.toml", "overrides": [],
                          "curriculum": {"unroll_s": [120.0], "stage_steps": [5],
                                         "truncation_steps_per_stage": [120]}, "optim": {}}}}
    with pytest.raises(RuntimeError, match="failed"):
        run.run_phase(root, p, "A", str(init), "x")
    assert len(run._attempts(root / "phases/A")) == 1


def test_run_lock_is_exclusive(tmp_path):
    with run.RunLock(tmp_path):
        with pytest.raises(RuntimeError, match="alive"):
            with run.RunLock(tmp_path):
                pass
    assert not (tmp_path / "RUN.lock").exists()


# ------------------------------------------------------------------ guard ---
def test_guard_real_configs_pass_and_truth_inputs_refused():
    srcs = [plan.Source("wz_native", "/d/wz_native", 0.5), plan.Source("wz_mappo_u0", "/d/wz_mappo", 0.5)]
    for ph in plan.PHASES:
        cfg = run.load_cfg(plan.CONFIGS[ph], plan.phase_overrides(ph, init_from="/x/INIT_WZ.pt",
                                                                  sources=srcs, base="wz_native"))
        m = run.cfg_mapping(cfg)
        assert guard.config_problems(m) == [], ph
        assert m["model"]["driver_param_mode"] == "physical" and m["model"]["driver_param_init"] == "checkpoint"
        assert m["model"]["rollout_mode"] != "driver_params" and not m["model"]["physics_params"]
    bad = json.loads(json.dumps(m))
    bad["data"]["sidecars"]["driver_sidecar"] = "/d/hetdata_truth.csv"
    bad["model"]["physics_params"] = "/d/vtypes.json"
    probs = guard.config_problems(bad)
    assert any("driver_sidecar" in p for p in probs) and any("physics_params" in p for p in probs)
    srcbad = json.loads(json.dumps(m))
    srcbad["data"]["c4_sources"][0]["path"] = "/d/sealed/x"
    assert guard.config_problems(srcbad)
    with pytest.raises(guard.GuardError):
        guard.check_path("/home/x/holdout_worlds/a")
    with pytest.raises(guard.GuardError):
        guard.check_path("/home/x/cal_split_a4")
    guard.check_batch({"kin": 1, "present": 1, "is_cav": 1, "history": 1, "cav_teacher": 1,
                       "merge_pivot_scene": {"states": 1, "present": 1}})
    for key in ("driver_class", "vtype", "true_type", "driver_params"):
        with pytest.raises(guard.GuardError):
            guard.check_batch({"kin": 1, key: 1})
    with pytest.raises(guard.GuardError):
        guard.check_batch({"kin": 1, "scene": {"vtype_id": 1}})


# ------------------------------------------------------------------ plan / mix / estimate / manifest ---
def test_mix_sources_and_profiles():
    srcs, base = plan.mix_sources([("wz_native", "/a", None), ("wz_mappo_u0", "/b", None)],
                                  {"wz_native": 30, "wz_mappo_u0": 10})
    assert base == "wz_native" and [s.mix_ratio for s in srcs] == [0.75, 0.25]
    with pytest.raises(ValueError):
        plan.mix_sources([("wz_native", "/a", None)], {"wz_native": 3})
    assert plan.parse_source("a=/x/y@0.3") == ("a", "/x/y", 0.3)
    assert plan.parse_source("a=/x/y") == ("a", "/x/y", None)
    sets = plan.phase_overrides("B", init_from="/x", sources=srcs, base=base, profile="night")
    assert "curriculum.stage_steps=[40, 24]" in sets
    assert not any(s.startswith("loss.paired_diff") for s in sets)


def test_split_by_arm_derives_absolute_sources(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / run.MANIFEST_NAME).write_text(json.dumps(_manifest(
        [("w989101m3200p60t300", "train", "native"), ("w989101m3200p60t300", "train", "mappo_u0"),
         ("w989102m3200p60t300", "val", "native")])))
    out = run.split_by_arm(str(src), tmp_path / "derived")
    assert [n for n, _ in out] == ["wz_mappo_u0", "wz_native"]
    nat = run.read_manifest(dict(out)["wz_native"])
    assert nat["n_episodes"] == 2 and nat["derived_arm"] == "native"
    assert all(Path(e["dir"]).is_absolute() for e in nat["episodes"])


def test_estimate_and_manifest(tmp_path):
    p = {"phases": {"A": {"curriculum": {"unroll_s": [120.0], "stage_steps": [30]}, "overrides": []},
                    "B": {"curriculum": {"unroll_s": [120.0, 300.0], "stage_steps": [40, 24]},
                          "overrides": []}}}
    e = estimate.estimate(p)
    sc = e["scenarios"]
    assert sc["low"]["total_h"] < sc["central"]["total_h"] < sc["high"]["total_h"]
    m = estimate.estimate(p, measured_s_per_step_120=300.0, measured_peak_mb=4000.0)
    assert m["basis"] == "measured" and m["gpu_peak_mb"] == 4000.0
    root = tmp_path / "root"
    root.mkdir()
    wm = worlds.worlds_manifest({"s": _manifest([("w989101m3200p60t300", "train", "n"),
                                                 ("w989102m3200p60t300", "val", "n")])})
    run.write(root / "WORLDS.json", wm)
    run.write(root / "PLAN.json", {})
    plan_body = {"map": "wz3to2", "road": {}, "init": {"wm2_sha256": "a", "init_wz_sha256": "b",
                 "reinit_prefixes": list(reinit.DEFAULT_PREFIXES), "reinit_receipt_sha256": "c"},
                 "sources": [], "phases": {"A": {"config": "c", "config_sha256": "d", "overrides": []}},
                 "code_commit": "e"}
    body = run.manifest(root, plan_body, {"final": "a.pt", "final_sha256": "f"},
                        {"final": "b.pt", "final_sha256": "g", "best": None, "best_sha256": None})
    assert body["train_seeds"] == [989101] and body["val_seeds"] == [989102]
    assert body["reserved_qualification_overlap"] == []
    assert json.loads((root / "CHECKPOINT_MANIFEST.json").read_text())["val_worlds"] == ["w989102m3200p60t300"]


# ------------------------------------------------------------------ prepare end to end (fake sources) ---
def _fake_sources(tmp_path, extra_world=None):
    out = []
    for arm in ("native", "mappo_u0"):
        d = tmp_path / f"src_{arm}"
        d.mkdir()
        eps = [(f"w{989101 + i}m3200p60t300", "val" if i in (0, 5) else "train", arm) for i in range(8)]
        if extra_world:
            eps.append((extra_world, "train", arm))
        (d / run.MANIFEST_NAME).write_text(json.dumps(_manifest(eps)))
        out.append(f"wz_{arm}={d}")
    return out


def _fake_wm2(tmp_path):
    sets = plan.phase_overrides("A", init_from="x.pt", sources=[plan.Source("wz_native", "/a", 0.5),
                                plan.Source("wz_mappo_u0", "/b", 0.5)], base="wz_native")
    state = reinit.fresh_state_from_config(plan.CONFIGS["A"], sets, seed=7, repo=REPO)
    state = {k: (v + 0.05 if v.is_floating_point() else v) for k, v in state.items()}
    p = tmp_path / "wm2.pt"
    torch.save({"format": "wm_s/1", "model": state, "config": {}}, p)
    return p


def _prepare_args(root, wm2, sources, **kw):
    a = ["prepare", "--root", str(root), "--init", str(wm2), "--init-sha", reinit.sha256_file(wm2),
         "--allow-no-road", "--device", "cpu", "--python", sys.executable]
    for s in sources:
        a += ["--source", s]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return a, None


def test_prepare_end_to_end_fake_sources(tmp_path):
    wm2 = _fake_wm2(tmp_path)
    root = tmp_path / "root"
    argv, _ = _prepare_args(root, wm2, _fake_sources(tmp_path))
    assert run.main(argv) == 0
    p = json.loads((root / "PLAN.json").read_text())
    assert p["n_train_worlds"] == 6 and p["n_val_worlds"] == 2 and p["base_source"] == "wz_native"
    assert [s["mix_ratio"] for s in p["sources"]] == [0.5, 0.5]
    assert p["init"]["n_reinit"] == 35 and p["init"]["wm2_sha256"] == reinit.sha256_file(wm2)
    assert p["phases"]["A"]["steps"] == 30 and p["phases"]["B"]["curriculum"]["stage_steps"] == [40, 24]
    assert any(s.startswith("model.init_from=") and "INIT_WZ.pt" in s for s in p["phases"]["A"]["overrides"])
    wm = json.loads((root / "WORLDS.json").read_text())
    assert wm["val_seeds"] == [989101, 989106] and worlds.overlap(wm) == []
    with pytest.raises(FileExistsError):
        run.main(argv)                                              # PLAN.json is write-once


def test_prepare_refuses_reserved_world_before_any_init(tmp_path):
    wm2 = _fake_wm2(tmp_path)
    root = tmp_path / "root"
    argv, _ = _prepare_args(root, wm2, _fake_sources(tmp_path, extra_world="w989210m3200p60t300"))
    with pytest.raises(worlds.WorldError):
        run.main(argv)
    assert not (root / "INIT_WZ.pt").exists() and not (root / "PLAN.json").exists()


def test_prepare_refuses_wrong_init_sha(tmp_path):
    wm2 = _fake_wm2(tmp_path)
    argv, _ = _prepare_args(tmp_path / "root", wm2, _fake_sources(tmp_path))
    argv[argv.index("--init-sha") + 1] = "0" * 64
    with pytest.raises(reinit.ReinitError):
        run.main(argv)


def test_prepare_with_wz_road_binding(tmp_path, monkeypatch):
    """With a wz3to2 MAP_SPEC the road is bound (stub/N1 descriptor) and its through-lane exit must
    equal the configs' loss.agg_delay.road_end_m (1379.44)."""
    from wms.core import road as road_mod
    spec = tmp_path / "MAP_SPEC.json"
    spec.write_text(json.dumps({"name": "wz3to2"}))
    monkeypatch.setenv("WMS_MAP_SPEC", str(spec))
    try:
        wm2 = _fake_wm2(tmp_path)
        argv, _ = _prepare_args(tmp_path / "root", wm2, _fake_sources(tmp_path))
        argv = [a for a in argv if a != "--allow-no-road"] + ["--map-spec", str(spec)]
        assert run.main(argv) == 0
        p = json.loads((tmp_path / "root" / "PLAN.json").read_text())
        assert p["road"]["name"] == "wz3to2" and p["road"]["road_end_m"] == 1379.44
        assert p["env"]["WMS_MAP_SPEC"] == str(spec)
    finally:
        road_mod._reset_for_tests()
