"""R12 post-run check logic: BASE row classes / rules, the M8 rule (round 12 vs round 9), the seed-replicated
behaviour bootstrap, the default-off run_onestep / run_rollouts flags and run_all.sh hygiene.

The two slow tests (real one-step replay on one TRAIN window, ~1-3 min) run only with R12_SLOW=1."""
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tools" / "r12_checks", ROOT / "tools" / "r11_checks", ROOT / "tools" / "r10b_checks",
              ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import base_calibration as BC  # noqa: E402
import m8_r12 as M8  # noqa: E402
import run_onestep as OS  # noqa: E402
import run_rollouts as RR  # noqa: E402

SLOW = pytest.mark.skipif(os.environ.get("R12_SLOW") != "1", reason="slow real replay (set R12_SLOW=1)")
COLS = OS.COLS + OS.BASE_COLS


# ------------------------------------------------------------------------------------------------ synthetic one-step
def _rows(n_per=200, slow_bias=0.1, free_bias=0.05, far_bias=0.05, brake_bias=0.5, last3_bias=0.2, seed=0):
    """Synthetic onestep rows: one block per class, recorded 0 (braking: -2), model = recorded + class bias."""
    rng = np.random.default_rng(seed)
    blocks = []

    def block(**kw):
        d = {c: np.zeros(n_per) for c in COLS}
        d.update({"lane": np.ones(n_per), "has": np.zeros(n_per), "tm": np.full(n_per, np.nan),
                  "dA": np.full(n_per, np.nan), "dend": np.full(n_per, np.nan), "bgap": np.full(n_per, np.nan),
                  "gapL": np.full(n_per, np.inf), "gapF": np.full(n_per, np.nan), "s": np.full(n_per, -100.0),
                  "v": np.full(n_per, 10.0), "win": rng.integers(0, 40, n_per).astype(float),
                  "dacc": np.full(n_per, 1.2), "ddec": np.full(n_per, 2.0)})
        for k, v in kw.items():
            d[k] = np.broadcast_to(np.asarray(v, float), (n_per,)).copy()
        d["fin"] = d["rec"] + d.pop("bias")
        blocks.append(d)

    block(v=1.0, gapF=10.0, gapL=10.0, s=-100.0, bias=slow_bias)                    # slow start (near)
    block(v=1.0, gapF=12.0, gapL=12.0, s=-500.0, bias=far_bias)                     # far-upstream slow start
    block(v=12.0, bias=free_bias)                                                    # free road (no leader)
    block(v=3.0, has=1.0, tm=2.0, dA=4.0, dend=20.0, bgap=2.0, gapF=12.0, gapL=12.0, bias=last3_bias)
    block(v=8.0, rec=-2.0, gapF=30.0, gapL=30.0, bias=brake_bias)                    # recorded braking
    return {c: np.concatenate([b[c] for b in blocks]) for c in COLS}


def _write_os(path: Path, arms: dict, replay="filled"):
    path.mkdir(parents=True, exist_ok=True)
    for arm, d in arms.items():
        rows = np.stack([d[c] for c in COLS], -1).astype(np.float32)
        np.savez_compressed(path / f"onestep_{arm}.npz", rows=rows, cols=np.array(COLS))
        (path / f"onestep_{arm}_events.json").write_text("{}")
    (path / "ONESTEP_DONE.json").write_text(json.dumps({"windows": 40, "cols": COLS, "replay_present": replay}))


def test_row_classes_select_the_registered_rows():
    d = _rows(n_per=50)
    cls = BC.row_classes(d)
    n = {k: int(v.sum()) for k, v in cls.items()}
    assert n["slow_start"] == 100 and n["far_upstream_slow"] == 50          # far-upstream is a subset
    assert n["free_road"] == 50                                             # no leader, 5 < v < 20
    assert n["last3_target_follower"] == 50 and n["far_partner"] == 0 and n["alongside"] == 0
    assert n["recorded_braking"] == 50 and n["after_merge_zone"] == 0 and n["all_lane1"] == 250
    # gapF (replay presence) wins over gapL; a lane-0 row is never in a class
    d["gapF"][:50], d["gapL"][:50] = 40.0, 10.0
    d["lane"][50:100] = 0
    cls = BC.row_classes(d)
    assert int(cls["slow_start"].sum()) == 0 and int(cls["all_lane1"].sum()) == 200
    # without the base columns gapL is used
    d2 = {k: v for k, v in _rows(n_per=50).items() if k != "gapF"}
    assert int(BC.row_classes(d2)["slow_start"].sum()) == 100


def _stats(r12, r9, hz=None):
    out = {"r9": BC.class_stats(r9, 50), "r10b": BC.class_stats(r12, 50)}
    if hz is not None:
        out["headzero"] = BC.class_stats(hz, 50)
    return out


def test_base_rules():
    ok = BC.decide(_stats(_rows(slow_bias=0.1), _rows(slow_bias=1.5)))
    assert [ok[k][0] for k in ("BASE-1", "BASE-2", "BASE-3", "BASE-4")] == ["PASS", "PASS", "PASS", "READ"]
    assert BC.overall(ok).startswith("PASS")
    # BASE-1: > 0.3, or not smaller than round 9 (magnitudes)
    def both(x):   # slow_start pools the near and the far-upstream block
        return _rows(slow_bias=x, far_bias=x)
    assert BC.decide(_stats(both(0.35), both(1.5)))["BASE-1"][0] == "REJECT"
    assert BC.decide(_stats(both(-0.2), both(0.2)))["BASE-1"][0] == "REJECT"       # not smaller in magnitude
    assert BC.decide(_stats(both(-0.2), both(0.25)))["BASE-1"][0] == "PASS"
    # BASE-2: free road or far-upstream slow start > max(0.3, round 9's |bias| + 0.1)
    assert BC.decide(_stats(_rows(free_bias=-0.35), _rows()))["BASE-2"][0] == "REJECT"
    assert BC.decide(_stats(_rows(free_bias=-0.25), _rows()))["BASE-2"][0] == "PASS"
    assert BC.decide(_stats(_rows(far_bias=0.31, slow_bias=0.0), _rows(slow_bias=1.0)))["BASE-2"][0] == "REJECT"
    assert BC.decide(_stats(_rows(free_bias=0.45), _rows(free_bias=0.4)))["BASE-2"][0] == "PASS"   # within ref + 0.1
    assert BC.decide(_stats(_rows(free_bias=0.55), _rows(free_bias=0.4)))["BASE-2"][0] == "REJECT"
    # BASE-3: braking |bias| worse than round 9 by > 0.1
    assert BC.decide(_stats(_rows(brake_bias=0.65), _rows(brake_bias=0.5)))["BASE-3"][0] == "REJECT"
    assert BC.decide(_stats(_rows(brake_bias=0.55), _rows(brake_bias=0.5)))["BASE-3"][0] == "PASS"
    assert BC.decide(_stats(_rows(brake_bias=-0.65), _rows(brake_bias=0.5)))["BASE-3"][0] == "REJECT"
    # BASE-4 read lists all three arms
    r = BC.decide(_stats(_rows(last3_bias=0.2), _rows(last3_bias=1.3), _rows(last3_bias=0.8)))["BASE-4"]
    assert r[0] == "READ" and "+0.200" in r[1] and "+0.800" in r[1] and "within +-0.3: True" in r[1]
    # a missing class never passes silently
    few = _rows(n_per=2)
    und = BC.decide(_stats(few, few))
    assert und["BASE-1"][0] == "UNDECIDED" and BC.overall(und).startswith("UNDECIDED")
    assert BC.overall({**ok, "BASE-3": ("REJECT", "")}).startswith("REJECT ROUND 12")


def test_encoded_param_flag():
    d = _rows(n_per=50)
    e = BC.encoded_params(d)
    assert e["accel"]["available"] and not e["accel"]["flag"] and abs(e["accel"]["pct"]["p50"] - 1.2) < 1e-6
    d["dacc"][:60] = 0.31                                   # 60/250 = 24 % within 1 % of the range of a = 0.3
    assert BC.encoded_params(d)["accel"]["flag"]
    d["dacc"][:] = np.nan
    assert not BC.encoded_params(d)["accel"]["available"]


def test_base_calibration_dry_run(tmp_path):
    _write_os(tmp_path / "os", {"r9": _rows(slow_bias=1.5, last3_bias=1.3), "r10b": _rows(),
                                "headzero": _rows(last3_bias=0.8)})
    out, js = tmp_path / "BASE.md", tmp_path / "BASE.json"
    assert BC.main(["--os", str(tmp_path / "os"), "--out", str(out), "--json", str(js), "--boot", "50"]) == 0
    res = json.loads(js.read_text())
    assert res["verdict"].startswith("PASS") and res["replay_present"] == "filled"
    text = out.read_text()
    assert "BASE verdict: PASS" in text and "slow_start" in text and "round 12 head-zero" in text
    # missing inputs: still exit 0, verdict undecided
    assert BC.main(["--os", str(tmp_path / "none"), "--out", str(out), "--json", str(js)]) == 0
    assert "UNDECIDED" in json.loads(js.read_text())["verdict"]


# ------------------------------------------------------------------------------------------------ M8 / M5
BASE = {"merges_per_run": 10.0, "stuck_mean_ramp_in_zone": 1.0, "stuck_share_wait_gt60": 0.1,
        "ttc_lt_1p5_share": 0.01, "min_gap_ins_p5": 2.0, "post_ins_hard_brake_share": 0.1}


def test_m8_rule_needs_bias_not_below_round9_and_a_worse_behaviour():
    same = {M8.REF: dict(BASE), M8.R12: dict(BASE)}
    worse = {M8.REF: dict(BASE), M8.R12: {**BASE, "merges_per_run": 9.0}}
    # bias below round 9 (magnitude) -> pass even with a worse behaviour
    assert M8.decide({M8.REF: 1.3, M8.R12: 0.2}, {M8.REF: 1.3, M8.R12: -0.3}, worse, -0.3)["verdict"] == "PASS"
    # bias not below, behaviour equal -> pass
    assert M8.decide({M8.REF: 0.5, M8.R12: 0.6}, {}, same, -0.3)["verdict"] == "PASS"
    # both -> reject; one-step alone can trigger; magnitude: -1.4 is not below +1.3
    d = M8.decide({M8.REF: 1.3, M8.R12: 0.2}, {M8.REF: 1.3, M8.R12: -1.4}, worse, -0.3)
    assert d["verdict"] == "REJECT ROUND 12" and d["bias_not_below_round9"] == {"rollout": False, "one_step": True}
    assert M8.decide({M8.REF: 1.0, M8.R12: 1.0}, {}, worse, -0.3)["verdict"] == "REJECT ROUND 12"   # equal = not below
    for key, sign in M8.WORSE_SIGN.items():
        beh = {M8.REF: dict(BASE), M8.R12: {**BASE, key: BASE[key] + sign * 0.5}}
        assert M8.decide({M8.REF: 0.0, M8.R12: 0.5}, {}, beh, -0.3)["worse_than_round9"] == [key]
    assert M8.decide({}, {}, same, None)["verdict"] == "MISSING"
    assert M8.decide({M8.REF: 0.0, M8.R12: 0.0}, {}, same, -0.05)["under_yield_flag"]
    assert not M8.decide({M8.REF: 0.0, M8.R12: 0.0}, {}, same, -0.3)["under_yield_flag"]


def test_m5_and_seed_consistency():
    d = {"r10b-r9": {"ttc_lt_1p5_share": [0.01, 0.002, 0.02], "overlap_share": [0.0, -0.001, 0.001]}}
    assert M8.m5_decide(d) == {"verdict": "REJECT ROUND 12", "worse_ci_above_0": ["ttc_lt_1p5_share"]}
    d["r10b-r9"]["ttc_lt_1p5_share"] = [0.01, -0.002, 0.02]
    assert M8.m5_decide(d)["verdict"] == "PASS" and M8.m5_decide({})["verdict"] == "MISSING"
    assert M8.combine({"seed set 1": "PASS", "pooled": "PASS"}) == "PASS"
    c = M8.combine({"seed set 1": "PASS", "seed set 2": "REJECT ROUND 12", "pooled": "PASS"})
    assert c.startswith("PASS (seed-inconsistent")


def _roll_file(path: Path, state: int, k: int, ttc_rows: int, n_rows=100, n_ev=3, accel=0.5, gap=5.0):
    """One synthetic run: lane-1 HDV rows (analyze.C layout) and n_ev HDV merge events."""
    import analyze as A
    r = np.zeros((n_rows, len(A.C)))
    c = {name: i for i, name in enumerate(A.C)}
    r[:, c["ep"]], r[:, c["t"]], r[:, c["lane"]] = state, np.arange(n_rows) * 0.5, 1
    r[:, c["hasL"]], r[:, c["gapL"]], r[:, c["dvL"]] = 1, 50.0, 0.0
    r[:ttc_rows, c["gapL"]], r[:ttc_rows, c["dvL"]] = 1.0, -2.0            # TTC 0.5 s
    r[:, c["nzone"]], r[:, c["waitR"]] = 1.0, np.nan
    ev = np.full((n_ev, 65, 13), np.nan)
    ev[:, :, 2], ev[:, :, 3], ev[:, :, 11] = accel, 1, gap
    me = np.zeros((n_ev, 8))
    me[:, 0] = state
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f"s{state:03d}_k{k}.npz", rows=r.astype(np.float32), ev=ev.astype(np.float32), me=me,
                        head=np.zeros((0, 16), np.float32), jerk_switch=np.zeros(0, np.float32),
                        jerk_all=np.zeros(3, np.float32), n_jerk_all=0, misc=np.zeros(5))


def test_run_stats_and_seed_replicated_bootstrap(tmp_path):
    for s in range(12):
        _roll_file(tmp_path / "a" / "roll" / "sumo", s, 0, 0, accel=0.0)
        for k in range(2):
            _roll_file(tmp_path / "a" / "roll" / "r9", s, k, ttc_rows=1, accel=1.0)
            _roll_file(tmp_path / "a" / "roll" / "r10b", s, k, ttc_rows=5 + s % 3, accel=0.2)
            _roll_file(tmp_path / "b" / "roll" / "r10b", s, k, ttc_rows=5, accel=0.2)
            _roll_file(tmp_path / "b" / "roll" / "r9", s, k, ttc_rows=1, accel=1.0)
        _roll_file(tmp_path / "b" / "roll" / "sumo", s, 0, 0, accel=0.0)
    one = M8.load_set([str(tmp_path / "a" / "roll")], "r10b")
    assert sorted(one) == list(range(12)) and one[0]["runs"] == 2 and one[0]["nrows"] == 200
    pooled = {arm: M8.load_set([str(tmp_path / "a" / "roll"), str(tmp_path / "b" / "roll")], arm) for arm in ("r9", "r10b")}
    sumo = M8.load_set([str(tmp_path / "a" / "roll"), str(tmp_path / "b" / "roll")], "sumo")
    assert pooled["r10b"][0]["runs"] == 4
    b = M8.bootstrap(pooled, sumo, 100)
    assert b["n_states"] == 12
    assert abs(b["r9"]["ttc_lt_1p5_share"][0] - 0.01) < 1e-9 and abs(b["r9"]["merges_per_run"][0] - 3.0) < 1e-9
    assert abs(b["r9"]["abs_bias_rollout_m3_0"][0] - 1.0) < 1e-6 and abs(b["r10b"]["abs_bias_rollout_m3_0"][0] - 0.2) < 1e-6
    d = b["r10b-r9"]["ttc_lt_1p5_share"]
    assert d[0] > 0 and d[1] > 0 and d[1] <= d[0] <= d[2]                    # paired CI above 0 -> M5 reject
    assert M8.m5_decide(b)["verdict"] == "REJECT ROUND 12"
    # end to end (no one-step, M1 read may fail without states: still exit 0)
    out, js = tmp_path / "M8.md", tmp_path / "M8.json"
    assert M8.main(["--shm", str(tmp_path / "a"), "--shm2", str(tmp_path / "b"), "--out", str(out), "--json", str(js),
                    "--boot", "50"]) == 0
    res = json.loads(js.read_text())
    assert set(res["sets"]) == {"seed set 1", "seed set 2", "pooled"}
    assert res["M5_verdict"].startswith("REJECT ROUND 12")
    assert res["decision"]["pooled"]["bias_not_below_round9"] == {"rollout": False}   # |0.2| < |1.0|
    assert "pooled" in out.read_text()


# ------------------------------------------------------------------------------------------------ default-off flags
def test_run_onestep_flags_default_to_the_old_behaviour():
    a = OS.build_parser().parse_args(["--repo", "R", "--roots", "X", "--out", "O"])
    assert a.replay_present == "loss_valid" and a.base_cols is False and a.arm_set == []
    assert a.config == "c4_wm_v10round10b.toml"
    b = OS.build_parser().parse_args(["--repo", "R", "--roots", "X", "--out", "O", "--replay-present", "filled",
                                      "--base-cols", "--arm-set", "r9=model.driver_param_mode=legacy"])
    assert b.replay_present == "filled" and b.base_cols
    with pytest.raises(SystemExit):
        OS.build_parser().parse_args(["--repo", "R", "--roots", "X", "--out", "O", "--replay-present", "other"])
    assert OS.parse_arm_sets(["r9=model.a=1", "r9=model.b=2", "x=k=v"]) == {"r9": ("model.a=1", "model.b=2"),
                                                                            "x": ("k=v",)}
    with pytest.raises(SystemExit):
        OS.parse_arm_sets(["r9"])
    # the R11 column layout is a prefix of the R12 one (analyze.py reads columns by name)
    assert OS.COLS[-4:] == ["tm", "te", "s", "closing"] and OS.BASE_COLS == ["gapF", "vLF", "dacc", "ddec"]
    assert inspect.signature(OS._init).parameters["replay_present"].default == "loss_valid"
    assert inspect.signature(OS._init).parameters["base_cols"].default is False
    assert OS.stored_driver_param_mode({"config": {}}) == "legacy"
    assert OS.stored_driver_param_mode({"config": {"driver_param_mode": "physical"}}) == "physical"


def test_run_rollouts_seed_offset_is_default_off():
    assert inspect.signature(RR._init).parameters["seed_offset"].default is None
    assert RR.SEED_OFFSET is None
    assert RR.seed_of(3, 1000) == RR.seed_of(3, 1000) != RR.seed_of(3, 1001) != RR.seed_of(4, 1000)
    src = (ROOT / "tools" / "r10b_checks" / "run_rollouts.py").read_text()
    assert 'add_argument("--seed-offset", type=int, default=None' in src


def test_run_all_sh_is_lf_and_parses():
    p = ROOT / "tools" / "r12_checks" / "run_all.sh"
    raw = p.read_bytes()
    assert b"\r" not in raw
    text = raw.decode()
    for needle in ("--replay-present filled", "--base-cols", "--arm-set r9=model.driver_param_mode=legacy",
                   "m8_r12.py", "base_calibration.py", "m7_head_range.py", "VERDICT_R12_CHECKS_COMBINED_$MODE.md",
                   "VERDICT_R12_CHECKS.md", "--seed-offset 0", "--seed-offset $SEED2_OFFSET",
                   "c4_wm_v10_round9/wm_s_stage300.pt", "chk_r10b/states192.json.gz", "c4_wm_v10round12.toml"):
        assert needle in text, needle
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    r = subprocess.run([bash, "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------------------------------ slow: real replay
R9_CK = ROOT / "tmp_r12" / "ck" / "r9_stage300.pt"
R8_CK = Path("E:/agent-work/c4-v10r9fix/data/r9_train/wm_s_stage300.pt")
TRAIN_ROOT = Path("E:/agent-work/c4-v10r10b/tmp_r10b/roots/v5_native")   # TRAIN episodes of the r9_train corpus
LEGACY = ["--arm-set", "r9=model.driver_param_mode=legacy", "--arm-set", "r9=model.driver_param_init=checkpoint",
          "--arm-set", "r8=model.driver_param_mode=legacy", "--arm-set", "r8=model.driver_param_init=checkpoint"]


def _run_os(out, *extra, arms=(("r9", R9_CK), ("r8", R8_CK)), config="c4_wm_v10round12.toml"):
    args = [sys.executable, str(ROOT / "tools" / "r10b_checks" / "run_onestep.py"), "--repo", str(ROOT),
            "--config", config, *sum((["--var", f"{a}={c}"] for a, c in arms), []), "--roots", str(TRAIN_ROOT),
            "--workers", "1", "--limit", "1", "--out", str(out), *extra]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT), str(ROOT / "tests")])}
    return subprocess.run(args, capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=900)


@SLOW
def test_real_onestep_filled_replay_one_train_window(tmp_path):
    if not (R9_CK.exists() and R8_CK.exists() and TRAIN_ROOT.exists()):
        pytest.skip("local round-8/9 checkpoints or TRAIN roots missing")
    r = _run_os(tmp_path / "f", *LEGACY, "--replay-present", "filled", "--base-cols")
    assert r.returncode == 0, r.stderr[-2000:]
    done = json.loads((tmp_path / "f" / "ONESTEP_DONE.json").read_text())
    assert done["replay_present"] == "filled" and done["cols"] == COLS
    z = np.load(tmp_path / "f" / "onestep_r9.npz")
    d = {c: z["rows"][:, i] for i, c in enumerate(z["cols"])}
    assert len(d["fin"]) > 100 and np.isfinite(d["dacc"]).all() and np.isfinite(d["ddec"]).all()
    assert (d["dacc"] >= 0.3).all() and np.isfinite(d["gapF"]).any()
    # the BASE check runs on it
    assert BC.main(["--os", str(tmp_path / "f"), "--out", str(tmp_path / "b.md"), "--json", str(tmp_path / "b.json"),
                    "--boot", "20"]) == 0


@SLOW
def test_real_onestep_refuses_an_encoder_mode_mismatch(tmp_path):
    if not R9_CK.exists():
        pytest.skip("local round-9 checkpoint missing")
    r = _run_os(tmp_path / "x", arms=(("r9", R9_CK),))        # physical config, legacy checkpoint, no --arm-set
    assert r.returncode != 0 and "driver_param_mode" in (r.stderr + r.stdout)
