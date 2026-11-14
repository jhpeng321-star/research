"""WD-01-C4L1 access-control tests: the loader refuses the 9 CAL
worlds without the frozen registration (matching LF-sha256), refuses
HOLDOUT (split + excluded roots), and the freeze gate refuses on any
hash/commit mismatch.  Hermetic: synthetic plans and fake run files in
tmp_path; the real CAL truth is never opened here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

WD01 = Path(__file__).resolve().parents[2]
if str(WD01) not in sys.path:
    sys.path.insert(0, str(WD01))
HERE = WD01 / "partA_v2g"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_c4l1 as C  # noqa: E402
from partA_v2g import eval_a as EA  # noqa: E402

CAL = sorted(C.CAL_WORLD_NAMES)
EXCLUDED_HOLDOUT = "w8407m2400p02t360"   # protocols/holdout_exclusions.json
CLEAN_HOLDOUT = "w8451m2700p20t420"


def _write_registration(tmp_path, body="x\n" * 3) -> Path:
    p = tmp_path / "not_the_registration.md"
    p.write_text(body, encoding="utf-8", newline="\n")
    return p


def _fake_run(root: Path, world: str, t0: float = 240.0) -> None:
    d = root / "runs" / "campaign_stage1_chunk4" / f"{world}__native_base"
    (d / "run").mkdir(parents=True, exist_ok=True)
    dur = t0 + 600.0
    (d / "run" / "summary.json").write_text(
        json.dumps({"duration_s": dur}), encoding="utf-8")
    (d / "RUN_RECEIPT.json").write_text(
        json.dumps({"exit_status": "complete"}), encoding="utf-8")
    (d / "run" / "physics_steps.jsonl").write_text(
        json.dumps({"time_s": t0 + 0.5, "vehicles": {}}) + "\n" +
        json.dumps({"time_s": dur, "vehicles": {}}) + "\n",
        encoding="utf-8")


def _plan(worlds: list) -> dict:
    return {"schema": "x", "worlds": worlds}


def _cal_plan_entry(name: str, split: str = "CALIBRATION") -> dict:
    return dict(world=name, seed=1, loading_veh_h=2700, penetration=0.02,
                split=split, t0_s=240.0)


@pytest.fixture()
def cal_root(tmp_path):
    for w in CAL:
        _fake_run(tmp_path, w)
    plan = tmp_path / "output" / "campaign_stage1_v3" / \
        "results_plan_chunk4" / "WORLD_PLAN.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(
        _plan([_cal_plan_entry(w) for w in CAL])), encoding="utf-8")
    return tmp_path


def test_loader_refuses_cal_without_matching_registration(cal_root,
                                                          tmp_path):
    wrong = _write_registration(tmp_path)
    before = set(EA.OPENED_RUN_FILES)
    with pytest.raises(C.AccessRefused, match="LF-sha256 mismatch"):
        C.verify_registration(wrong)
    with pytest.raises(C.AccessRefused, match="LF-sha256 mismatch"):
        C.load_cal_worlds(cal_root, wrong,
                          plan_path=cal_root / "output" /
                          "campaign_stage1_v3" / "results_plan_chunk4" /
                          "WORLD_PLAN.json")
    # nothing was opened before the refusal
    assert set(EA.OPENED_RUN_FILES) == before


def test_loader_accepts_frozen_registration_copy(cal_root, tmp_path):
    good = tmp_path / "registration_copy.md"
    good.write_bytes((WD01 / "REGISTRATION_C4L1.md").read_bytes())
    opened: set = set()
    worlds = C.load_cal_worlds(
        cal_root, good,
        plan_path=cal_root / "output" / "campaign_stage1_v3" /
        "results_plan_chunk4" / "WORLD_PLAN.json",
        opened=opened)
    assert {w["world"] for w in worlds} == set(CAL)
    for w in worlds:
        assert w["native_run"].exists()
    assert any("WORLD_PLAN.json" in p for p in opened)


def test_loader_refuses_holdout_split(cal_root, tmp_path):
    good = tmp_path / "registration_copy.md"
    good.write_bytes((WD01 / "REGISTRATION_C4L1.md").read_bytes())
    # one of the 9 registered names re-labelled HOLDOUT in the plan:
    entries = [_cal_plan_entry(w) for w in CAL]
    entries[0]["split"] = "HOLDOUT"
    with pytest.raises(C.AccessRefused, match="HOLDOUT-split world"):
        C.select_cal_worlds(_plan(entries))
    # a plan whose CAL set itself contains an excluded root:
    bad_names = list(CAL)
    bad_names[0] = EXCLUDED_HOLDOUT
    plan_bad = _plan([_cal_plan_entry(w) for w in bad_names])
    with pytest.raises(C.AccessRefused, match="CALIBRATION set mismatch"):
        C.select_cal_worlds(plan_bad)


def test_select_refuses_excluded_root_with_data():
    names = list(CAL)
    names[2] = EXCLUDED_HOLDOUT
    plan = _plan([_cal_plan_entry(w) for w in names])
    # bypass the exact-set check to exercise the holdout guard itself:
    plan["worlds"][2]["world"] = CAL[2]  # set matches again
    data = {"excluded": [{"world": CAL[2]}], "kept_disclosed": []}
    hm = _FakeGuard()
    with pytest.raises(_FakeGuard.Error):
        C.select_cal_worlds(plan, holdout_data=data, holdout_mod=hm)


class _FakeGuard:
    class Error(ValueError):
        pass

    @staticmethod
    def assert_final_holdout_eligible(worlds, data=None):
        ex = {r["world"] for r in (data or {}).get("excluded", [])}
        if {w.split("__")[0] for w in worlds} & ex:
            raise _FakeGuard.Error("excluded root")


def test_freeze_gate_refuses_on_script_hash_mismatch(tmp_path):
    reg = tmp_path / "reg.md"
    reg.write_bytes((WD01 / "REGISTRATION_C4L1.md").read_bytes())
    (tmp_path / "params_nominal.json").write_bytes(
        (HERE / "params_nominal.json").read_bytes())
    fz = dict(
        registration_lf_sha256=C.REGISTRATION_LF_SHA,
        params_nominal_lf_sha256=C.lf_sha256(
            tmp_path / "params_nominal.json"),
        engine_commit="deadbeef",
        eval_scripts={"engine.py": "0" * 64})
    fp = tmp_path / "C4L1_FREEZE.json"
    fp.write_text(json.dumps(fz), encoding="utf-8")
    (tmp_path / "engine.py").write_text("print('real')\n", encoding="utf-8")
    with pytest.raises(C.AccessRefused, match="script sha mismatch"):
        C.verify_freeze(fp, reg, root=tmp_path, engine_commit_sha="deadbeef")
    # and on engine-commit mismatch with a correct script hash:
    fz["eval_scripts"] = {"engine.py": C.raw_sha256(tmp_path / "engine.py")}
    fp.write_text(json.dumps(fz), encoding="utf-8")
    with pytest.raises(C.AccessRefused, match="engine commit mismatch"):
        C.verify_freeze(fp, reg, root=tmp_path,
                        engine_commit_sha="cafebabe")


def test_freeze_gate_matches_the_real_files():
    """On the frozen line the gate passes.  On the integration line
    (zcode/wd01), later merges may legitimately change a frozen eval script
    (e.g. LC2 variant flags, wall law v2.1 in cavctrl); the gate must then
    REFUSE, and only for scripts that really changed since the freeze's
    engine commit (C4L1 itself is complete; this checks the gate, not a
    re-run)."""
    import subprocess
    try:
        fz = C.verify_freeze(C.FREEZE, C.REGISTRATION)
    except C.AccessRefused as exc:
        msg = str(exc)
        assert "freeze script sha mismatch for " in msg, msg
        name = msg.split("freeze script sha mismatch for ", 1)[1].split(":")[0]
        fz = json.loads(Path(C.FREEZE).read_text(encoding="utf-8"))
        changed = subprocess.run(
            ["git", "diff", "--quiet", fz["engine_commit"], "--",
             str(Path(C.HERE) / name)], cwd=str(C.HERE)).returncode
        assert changed == 1, f"gate refused {name} but it is unchanged"
        return
    assert fz["registration_lf_sha256"] == C.REGISTRATION_LF_SHA
