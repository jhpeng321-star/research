"""PL-NOISE1 analysis tests (ZERO SIMULATION: the fake-world closed loop +
hand-built/scripted noise only; the real PL-EXP1c logs do not exist yet).

Mapping to the registered acceptance criteria:

(a) test_a_zero_noise_replay_reproduces_logged_decisions /
    test_a_tampered_fixture_raises_the_named_stop_error
(b) test_b_retained_share_is_one_at_zero_noise_and_falls_with_noise +
    test_noise_sd_law_is_s_times_the_native_delay
(c) test_c_false_switch_is_counted_when_noise_makes_a_harmful_candidate_win
(d) test_d_unpriced_plan_fallback_keeps_the_current_plan_and_is_counted
(e) test_e_paired_rho_zero_leaves_delta_v_unchanged
(f) test_f_uncertainty_aware_margin_blocks_the_large_sd_switch
(g) test_g_* (code-sha refusal, TRAIN-only refusals, out-dir refusal)
(h) test_h_same_seed_gives_identical_records (+ the CLI-level rerun check
    in TestCliEndToEnd)

The fixtures reuse the fake-world closed loop (``Rig``) with the
REGISTERED planner constants (margin floor 60, agreement 0.80, rule C) so
the logged decisions and the replay agree by construction.
"""

import json
import shutil

import numpy as np
import pytest

from output.planner_v1.noise_study import analyze_noise as az
from output.planner_v1.noise_study import metrics, noise, replay, runs, study
from tests.planner.test_closed_loop import Rig, start_plan
from traffic_shepherd.planner import PlannerConfig
from traffic_shepherd.planner.candidate_replay import group_rows_by_decision
from traffic_shepherd.planner.interface import plan_key, with_actions

CONFIG = PlannerConfig()  # the registered constants (margin 60, agreement .80, rule C)
CAV = ("c0", "c1", "c2")


def rich_rig(tmp_path, name, *, cav=CAV, delays=None, couplings=(("c0", "c1"),),
             scripts=True, **over):
    """The fake-world closed loop with the REGISTERED planner constants and
    savings that clear the 60 s margin (damping -100, pi -80)."""
    rig = Rig(tmp_path / name, run_id=name, cav=cav, log_candidates=True, **over)
    rig.components.planner_config = CONFIG
    rig.model.action_delay = delays or {
        c: {"damping": -100.0, "pi-external-u": -80.0} for c in rig.cav
    }
    for pair in couplings:
        rig.model.traj_coupling_pos[pair] = {"damping": 10.0}
    if scripts:
        base = start_plan(rig.cav)
        bad_window = with_actions(base, {"c1": "damping", "c0": "native"})
        rig.model.safety_script[plan_key(bad_window)] = {"ttc_lt1p5_exposure_s": 5.0}
        bad_abs = with_actions(base, {"c2": "pi-external-u"})
        rig.model.safety_script[plan_key(bad_abs)] = {
            "collision": True, "min_ttc_s": 0.5
        }
    rig.run()
    return rig


def rows_of(rig):
    text = (rig.run_dir / "candidates.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def refs_of(rig):
    refs, skipped = study.prepare_decisions(rows_of(rig), CONFIG)
    assert refs, "no decidable decisions in the fixture"
    return refs


def add_noise_to_rows(rows, decision_index, per_cand, delta):
    """Deep-copied rows with +delta added to one decision's selected
    candidates ({cand_index}); a test-side tamper helper."""
    out = []
    for row in rows:
        row = json.loads(json.dumps(row))
        if row["decision_index"] == decision_index and row["cand_index"] in per_cand:
            arrays = row["arrays"]["delay_vs_native_s"]
            row["arrays"]["delay_vs_native_s"] = [
                [v + delta for v in cell] for cell in arrays
            ]
        out.append(row)
    return out


def run_scripted(refs, sd, draws, seed, config=CONFIG, model_config=None):
    """Accumulated outcomes of draws with N(0, sd) noise on EVERY candidate
    row (test-side scripted noise; sd is absolute veh·s)."""
    acc = metrics.LevelAccumulator()
    for ref in refs:
        acc.add_decision(ref)
    rng = np.random.default_rng(seed)
    cfg = config if model_config is None else model_config
    for _ in range(draws):
        for ref in refs:
            rows5 = study.expand_rows(ref.rows, 5)
            noise_map = {
                row["cand_index"]: rng.normal(0.0, sd, (5, 1)) for row in rows5
            }
            acc.add_draw(
                ref, replay.replay_draw(ref, cfg, replay.scripted_noise(noise_map), rows5)
            )
    return acc.finalize()


# ------------------------------------------------------------------- (a)
class TestPrerequisiteAndZeroNoise:
    def test_a_zero_noise_replay_reproduces_logged_decisions(self, tmp_path):
        rig = rich_rig(tmp_path, "zero")
        rows = rows_of(rig)
        refs, skipped = study.prepare_decisions(rows, CONFIG)
        assert len(refs) >= 3 and skipped == []
        decisions = {d["decision_index"]: d for d in rig.read_log("decisions.jsonl")}
        assert any(d["switched"] for d in decisions.values()), "fixture has real gains"
        logged = group_rows_by_decision(rows)
        for ref in refs:
            chosen = [r for r in logged[ref.decision_index] if r["chosen"]]
            assert len(chosen) == 1
            assert ref.oracle_key == chosen[0]["plan_key"]
            assert dict(ref.rows[ref.oracle_row]["actions"]) == decisions[
                ref.decision_index
            ]["actions"]
        # the seeded s=0 level reproduces the logged decisions exactly
        record = study.evaluate_level(
            refs, CONFIG, model_key="independent", level_pct=0.0,
            seed=noise.level_seed("independent", 0), draws=3, samples=5,
        )
        assert record["retained_share"] == 1.0
        assert record["false_switch_rate"] == 0.0
        assert record["missed_gain_rate"] == 0.0
        assert record["unpriced_fallback_rate"] == 0.0
        assert record["safety_veto_share"] == 0.0

    def test_a_tampered_fixture_raises_the_named_stop_error(self, tmp_path):
        rig = rich_rig(tmp_path, "tamper")
        rows = rows_of(rig)
        logged = group_rows_by_decision(rows)
        decision_index = next(
            index
            for index, group in sorted(logged.items())
            if any(
                role["stage"] == "individual"
                for row in group
                for role in row["roles"]
            )
            and any(
                row.get("chosen") and row["actions"].get("c2") == "damping"
                for row in group
            )
        )
        # tamper ONE individual candidate (c2->damping) of that decision so
        # the replay's pick flips -> a different plan is chosen
        victim = next(
            row["cand_index"]
            for row in logged[decision_index]
            if row["actions"] == {"c0": "native", "c1": "native", "c2": "damping"}
        )
        tampered = add_noise_to_rows(rows, decision_index, {victim}, 1000.0)
        with pytest.raises(replay.ReplayMismatchError, match="decision"):
            study.prepare_decisions(tampered, CONFIG)


# ------------------------------------------------------------------- (b)
class TestRetainedShare:
    def test_b_retained_share_is_one_at_zero_noise_and_falls_with_noise(
        self, tmp_path
    ):
        rig = rich_rig(tmp_path, "retain")
        refs = refs_of(rig)
        zero = run_scripted(refs, sd=0.0, draws=50, seed=11)
        moderate = run_scripted(refs, sd=40.0, draws=200, seed=11)
        large = run_scripted(refs, sd=120.0, draws=200, seed=11)
        assert zero["retained_share"] == 1.0
        assert zero["false_switch_rate"] == 0.0
        assert zero["missed_gain_rate"] == 0.0
        assert 0.0 < moderate["retained_share"] < 1.0
        assert large["retained_share"] < moderate["retained_share"]
        # the registered events actually occur on a fixture with real gains
        # (every candidate here has DeltaV <= 0, so false switches are
        # impossible by construction -- they are covered in (c) with a
        # harmful candidate)
        assert moderate["missed_gain_rate"] > 0.0 or large["missed_gain_rate"] > 0.0

    def test_noise_sd_law_is_s_times_the_native_delay(self):
        rng = noise.make_rng(noise.BASE_SEED)
        independent = noise.delay_noise_factory("independent", 10.0, 5)
        block = np.concatenate(
            [independent(rng, 200.0, (4, 5, 1)) for _ in range(2000)]
        )
        assert block.std() == pytest.approx(0.10 * 200.0, rel=0.05)
        paired = noise.delay_noise_factory("paired_rho0.5", 10.0, 5)
        block_p = np.concatenate(
            [paired(rng, 200.0, (4, 5, 1)) for _ in range(2000)]
        )
        assert block_p.std() == pytest.approx(0.5 * 0.10 * 200.0, rel=0.05)
        assert noise.level_seed("independent", 3) == noise.BASE_SEED + 3_000
        with pytest.raises(ValueError, match="native lookahead"):
            noise.level_sd(1.0, float("nan"))


# ------------------------------------------------------------------- (c)
class TestFalseSwitch:
    def test_c_false_switch_is_counted_when_noise_makes_a_harmful_candidate_win(
        self, tmp_path
    ):
        # c1's damping is only -10 so the flip {c0: pi, c1: damping} has
        # oracle DeltaV = +10 - 10 = 0 (harmful, never chosen noise-free);
        # scripted noise makes it the best logged GROUP candidate -> the
        # replay switches onto a plan the oracle prices at DeltaV >= 0.
        delays = {
            "c0": {"damping": -100.0, "pi-external-u": 10.0},
            "c1": {"damping": -10.0, "pi-external-u": 10.0},
        }
        rig = rich_rig(tmp_path, "harm", cav=("c0", "c1"), delays=delays, scripts=False)
        refs = refs_of(rig)
        ref0 = refs[0]
        assert ref0.oracle_dv < 0.0  # the oracle gain (the joint) is real
        rows5 = study.expand_rows(ref0.rows, 5)
        harmful_flip = {
            row["cand_index"]: np.full((5, 1), -500.0)
            for row in rows5
            if row["actions"] == {"c0": "pi-external-u", "c1": "damping"}
            and any(role["stage"] == "group" for role in row["roles"])
        }
        assert harmful_flip, "fixture lacks the harmful group-flip row"
        outcome = replay.replay_draw(ref0, CONFIG, replay.scripted_noise(harmful_flip), rows5)
        assert outcome.unpriced_fallback is False
        chosen_row = ref0.rows[outcome.chosen_row]
        assert chosen_row["actions"] == {"c0": "pi-external-u", "c1": "damping"}
        assert outcome.chosen_key != ref0.current_key
        assert ref0.dv_by_row[outcome.chosen_row] >= 0.0  # harmful in oracle terms
        acc = metrics.LevelAccumulator()
        acc.add_decision(ref0)
        acc.add_draw(ref0, outcome)
        summary = acc.finalize()
        assert summary["false_switch_rate"] == 1.0
        # zero noise keeps the oracle (no false switch)
        clean = replay.replay_draw(ref0, CONFIG, replay.scripted_noise({}), rows5)
        assert clean.chosen_key == ref0.oracle_key
        acc2 = metrics.LevelAccumulator()
        acc2.add_decision(ref0)
        acc2.add_draw(ref0, clean)
        assert acc2.finalize()["false_switch_rate"] == 0.0


# ------------------------------------------------------------------- (d)
class TestUnpricedFallback:
    def test_d_unpriced_plan_fallback_keeps_the_current_plan_and_is_counted(
        self, tmp_path
    ):
        # two disjoint groups; group B's joint fails the 60 s margin
        # noise-free (saving 40 < 60) so the all-four plan was NEVER priced;
        # scripted noise makes B pass too -> the combine needs the unpriced
        # plan -> UnpricedPlanError -> the pre-declared fallback
        delays = {
            "c0": {"damping": -100.0, "pi-external-u": -80.0},
            "c1": {"damping": -100.0, "pi-external-u": -80.0},
            "c2": {"damping": -20.0, "pi-external-u": 10.0},
            "c3": {"damping": -20.0, "pi-external-u": 10.0},
        }
        rig = rich_rig(
            tmp_path, "unpriced", cav=("c0", "c1", "c2", "c3"), delays=delays,
            couplings=(("c0", "c1"), ("c2", "c3")), scripts=False,
        )
        refs = refs_of(rig)
        ref0 = refs[0]
        logged = group_rows_by_decision(rows_of(rig))[ref0.decision_index]
        plan_keys = {row["plan_key"] for row in logged}
        joint_b = next(
            row
            for row in logged
            if row["actions"] == {
                "c0": "native", "c1": "native", "c2": "damping", "c3": "damping"
            }
        )
        all_four = {
            "c0": "damping", "c1": "damping", "c2": "damping", "c3": "damping"
        }
        assert json.dumps(
            plan_key(with_actions(start_plan(rig.cav), all_four)),
            sort_keys=True, default=str,
        ) not in plan_keys
        assert joint_b["switch"] is not None and not joint_b["switch"]["passes"]
        noise_map = {joint_b["cand_index"]: np.full((5, 1), -150.0)}
        outcome = replay.replay_draw(
            ref0, CONFIG, replay.scripted_noise(noise_map), study.expand_rows(ref0.rows, 5)
        )
        assert outcome.unpriced_fallback is True
        assert outcome.chosen_key == ref0.current_key
        acc = metrics.LevelAccumulator()
        acc.add_decision(ref0)
        acc.add_draw(ref0, outcome)
        summary = acc.finalize()
        assert summary["unpriced_fallback_rate"] == 1.0
        # the draw scored the CURRENT plan's oracle DeltaV (kept, not switched)
        assert summary["chosen_keys"] == [[ref0.decision_index, ref0.current_key]]


# ------------------------------------------------------------------- (e)
class TestPairedCancellation:
    def test_e_paired_rho_zero_leaves_delta_v_unchanged(self, tmp_path):
        rig = rich_rig(tmp_path, "paired0")
        refs = refs_of(rig)
        rng = noise.make_rng(7)
        zeros = noise.delay_noise_factory("paired_rho0", 15.0, 5)
        block = np.concatenate([zeros(rng, 999.0, (4, 5, 1)) for _ in range(500)])
        assert not block.any()
        for level in (0.5, 5.0, 15.0):
            record = study.evaluate_level(
                refs, CONFIG, model_key="paired_rho0", level_pct=level,
                seed=noise.level_seed("paired_rho0", 0), draws=5, samples=5,
            )
            assert record["retained_share"] == 1.0, level
            assert record["false_switch_rate"] == 0.0
            assert record["missed_gain_rate"] == 0.0
            assert record["unpriced_fallback_rate"] == 0.0
        # rho > 0 keeps a CONTRAST error (its sd law = rho*s*D_nat is proven
        # in test_noise_sd_law_is_s_times_the_native_delay); rho = 0 is the
        # exact full-cancellation limit asserted above.


# ------------------------------------------------------------------- (f)
class TestUncertaintyAwareMargin:
    def test_f_uncertainty_aware_margin_blocks_the_large_sd_switch(self, tmp_path):
        rig = rich_rig(
            tmp_path,
            "ua",
            cav=("c0",),
            couplings=(),
            scripts=False,
            delays={"c0": {"damping": -100.0, "pi-external-u": 10.0}},
        )
        ref0 = refs_of(rig)[0]
        # sample savings 100 - eps = [130, 115, 100, 70, 40]: median 100
        # (beats the floor), std ~32.3 -> base margin 92.3 passes, the UA
        # margin 60 + 1.64*32.3 = 113.0 blocks.
        eps = np.array([[-30.0], [-15.0], [0.0], [30.0], [60.0]])
        noise_map = {
            row["cand_index"]: eps
            for row in ref0.rows
            if row["actions"] == {"c0": "damping"}
        }
        assert noise_map
        rows5 = study.expand_rows(ref0.rows, 5)
        base = replay.replay_draw(ref0, CONFIG, replay.scripted_noise(noise_map), rows5)
        variant = replay.replay_draw(
            ref0, metrics.variant_config(CONFIG), replay.scripted_noise(noise_map), rows5
        )
        clean = replay.replay_draw(ref0, CONFIG, replay.scripted_noise({}), rows5)
        clean_variant = replay.replay_draw(
            ref0, metrics.variant_config(CONFIG), replay.scripted_noise({}), rows5
        )
        # zero noise: both rules switch (they reproduce the oracle)
        assert clean.chosen_key == clean_variant.chosen_key == ref0.oracle_key
        assert clean.chosen_key != ref0.current_key
        # large sd: the base rule still switches, the UA margin does not
        assert base.chosen_key == ref0.oracle_key
        assert variant.chosen_key == ref0.current_key
        # the variant evaluates over the same draws in the level machinery
        record = study.evaluate_level(
            [ref0], CONFIG, model_key="independent", level_pct=0.0,
            seed=1, draws=2, samples=5,
        )
        assert record["variant_retained_share"] == record["retained_share"] == 1.0


# ------------------------------------------------------------------- (g)
def _synthetic_runs_root(tmp_path, rig, *, sha_override=None, split=None):
    """The registered PL-EXP1c layout over a synthetic tmp runs root."""
    root = tmp_path / "runs" / "plexp1c_oracle_v1"
    dest = root / f"{rig.config.world}__planner_oracle"
    dest.mkdir(parents=True)
    shutil.copy(rig.run_dir / "candidates.jsonl", dest / "candidates.jsonl")
    receipt = json.loads((rig.run_dir / "RUN_RECEIPT.json").read_text(encoding="utf-8"))
    if sha_override is not None:
        receipt["planner_code_sha256"] = sha_override
    if split is not None:
        receipt["split"] = split
    (dest / "RUN_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
    worlds = tmp_path / "worlds.json"
    worlds.write_text(
        json.dumps({"worlds": [{"world": rig.config.world, "split": "TRAIN"}]}),
        encoding="utf-8",
    )
    return root, worlds, receipt


class TestGuards:
    def test_g_run_with_unlisted_code_sha_refused(self, tmp_path):
        rig = rich_rig(tmp_path, "sha")
        root, worlds, _ = _synthetic_runs_root(tmp_path, rig, sha_override="deadbeef")
        entries = runs.load_worlds(worlds)
        runs.check_train_only(entries)
        with pytest.raises(runs.PlannerCodeShaRefusedError, match="planner_code_sha256"):
            runs.discover_runs(root, entries, ["goodsha"])
        code = az.main(
            [
                "--runs-root", str(root),
                "--worlds", str(worlds),
                "--out", str(tmp_path / "out"),
                "--required-planner-code-sha", "goodsha",
                "--draws", "2",
                "--models", "independent",
                "--levels", "0",
            ]
        )
        assert code == 2

    def test_g_non_train_worlds_and_receipts_refused(self, tmp_path):
        with pytest.raises(runs.NonTrainWorldRefusedError, match="TRAIN"):
            runs.check_train_only([{"world": "w1", "split": "HOLDOUT"}])
        with pytest.raises(runs.NonTrainWorldRefusedError, match="HOLDOUT"):
            runs.check_train_only([{"world": "chunk2_HOLDOUT", "split": "TRAIN"}])
        with pytest.raises(runs.NonTrainWorldRefusedError, match="TEST"):
            runs.check_train_only([{"world": "w1", "split": "train"},
                                   {"world": "w2_TEST", "split": "TRAIN"}])
        rig = rich_rig(tmp_path, "trainrefuse")
        root, worlds, _ = _synthetic_runs_root(tmp_path, rig, split="HOLDOUT")
        entries = runs.load_worlds(worlds)
        runs.check_train_only(entries)
        with pytest.raises(runs.NonTrainWorldRefusedError, match="TRAIN only"):
            runs.discover_runs(root, entries, ["any-sha"])

    def test_g_out_dir_inside_runs_tree_or_on_d_refused(self, tmp_path):
        runs_root = tmp_path / "runs"
        with pytest.raises(runs.OutDirRefusedError, match="inside the runs root"):
            runs.check_out_dir(runs_root / "out", runs_root)
        with pytest.raises(runs.OutDirRefusedError, match="contains the runs root"):
            runs.check_out_dir(tmp_path, runs_root)
        doomed = "D" + ":/somewhere/out"  # built, never a literal (no_d_drive)
        with pytest.raises(runs.OutDirRefusedError, match="D:"):
            runs.check_out_dir(doomed, runs_root)


# ------------------------------------------------------------------- (h)
class TestDeterminism:
    def test_h_same_seed_gives_identical_records(self, tmp_path):
        rig = rich_rig(tmp_path, "det")
        refs = refs_of(rig)
        seed = noise.level_seed("independent", 3)
        kw = dict(model_key="independent", level_pct=5.0, seed=seed, draws=25, samples=5)
        first = study.evaluate_level(refs, CONFIG, **kw)
        second = study.evaluate_level(refs, CONFIG, **kw)
        assert first == second  # every field, incl. the chosen-key trace
        assert first["chosen_keys"] == second["chosen_keys"]
        # the seed is the ONLY randomness: a different seed yields a
        # different (but equally reproducible) noise stream
        shape = (len(refs), 5, 1)
        fn = noise.delay_noise_factory("independent", 5.0, 5)
        rng_a, rng_b = noise.make_rng(seed), noise.make_rng(seed + 1)
        block_a = fn(rng_a, refs[0].native_delay, shape)
        block_b = fn(rng_b, refs[0].native_delay, shape)
        block_a2 = fn(noise.make_rng(seed), refs[0].native_delay, shape)
        assert not np.array_equal(block_a, block_b)
        assert np.array_equal(block_a, block_a2)


# ---------------------------------------------------------------- metrics
class TestMetricHelpers:
    def test_sigma50_and_sigma_harm_interpolation(self):
        levels = [0.0, 5.0, 10.0, 15.0]
        retained = [1.0, 0.8, 0.4, 0.1]
        assert metrics.sigma50_pct(levels, retained) == pytest.approx(8.75)
        assert metrics.sigma50_pct([0.0, 5.0], [1.0, 0.9]) is None
        assert metrics.sigma50_pct([0.0], [None]) is None
        savings = [-100.0, -50.0, 1.0, 20.0]
        assert metrics.sigma_harm_pct(levels, savings) == pytest.approx(10.0)
        assert metrics.sigma_harm_pct(levels, [-5.0, -1.0]) is None

    def test_variant_config_uses_the_registered_z(self):
        variant = metrics.variant_config(CONFIG)
        assert variant.margin_k == noise.UNCERTAINTY_Z
        assert variant.wall_margin_k == pytest.approx(2.0 * noise.UNCERTAINTY_Z)
        assert variant.margin_floor_s == CONFIG.margin_floor_s

    def test_bias_rows_multiplies_plan_min_ttc_and_keeps_nan(self):
        rows = [
            {
                "cand_index": 0,
                "arrays": {"min_ttc_s": [[1.0], [None]]},
            }
        ]
        biased = noise.bias_min_ttc_rows(rows, 10.0)
        assert biased[0]["arrays"]["min_ttc_s"] == [[1.1], [None]]
        assert rows[0]["arrays"]["min_ttc_s"] == [[1.0], [None]]  # untouched
        down = noise.bias_min_ttc_rows(rows, -10.0)
        assert down[0]["arrays"]["min_ttc_s"] == [[pytest.approx(0.9)], [None]]


# ------------------------------------------------------------------- CLI
class TestCliEndToEnd:
    def _cli_args(self, root, worlds, sha, out):
        return [
            "--runs-root", str(root),
            "--worlds", str(worlds),
            "--out", str(out),
            "--required-planner-code-sha", sha,
            "--draws", "6",
            "--samples", "5",
            "--levels", "0,10",
            "--ttc-levels=-10,10",
        ]

    def test_cli_end_to_end_on_synthetic_runs_root(self, tmp_path):
        rig = rich_rig(tmp_path, "cli")
        root, worlds, receipt = _synthetic_runs_root(tmp_path, rig)
        out = tmp_path / "out"
        code = az.main(self._cli_args(root, worlds, receipt["planner_code_sha256"], out))
        assert code == 0
        expected_files = (
            "summary.json",
            "metrics_per_level.csv",
            "summary.md",
            "retained_vs_sigma.png",
            "runs_used.json",
        )
        for name in expected_files:
            path = out / name
            assert path.is_file() and path.stat().st_size > 0, name
        payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert payload["schema"] == "pl_noise_study_results/1"
        assert payload["prerequisite"]["passed"] is True
        assert payload["decisions_used"] >= 3
        assert payload["zero_simulation"] is True and payload["train_only"] is True
        assert set(payload["models"]) == {
            "independent", "paired_rho0.25", "paired_rho0.5", "ttc_bias"
        }
        assert payload["models"]["independent"]["levels"][0]["retained_share"] == 1.0
        for block in payload["models"].values():
            assert len(block["levels"]) == 2
            for record in block["levels"]:
                assert record["seed"] >= payload["base_seed"]
        csv_lines = (out / "metrics_per_level.csv").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        assert len(csv_lines) == 1 + 8  # header + 4 models x 2 levels
        assert "retained_share" in csv_lines[0]
        markdown = (out / "summary.md").read_text(encoding="utf-8")
        assert "sigma50" in markdown and "unpriced" in markdown.lower()
        assert (out / "retained_vs_sigma.png").stat().st_size > 1000
        # determinism at the CLI level: the same seed rewrites byte-identical
        # JSON/CSV (the plot embeds no timestamp either, but only the data
        # files are compared here)
        out2 = tmp_path / "out2"
        code = az.main(self._cli_args(root, worlds, receipt["planner_code_sha256"], out2))
        assert code == 0
        assert (out2 / "summary.json").read_bytes() == (out / "summary.json").read_bytes()
        assert (
            (out2 / "metrics_per_level.csv").read_bytes()
            == (out / "metrics_per_level.csv").read_bytes()
        )

    def test_cli_refuses_an_out_dir_inside_the_runs_root(self, tmp_path):
        rig = rich_rig(tmp_path, "clirefuse")
        root, worlds, receipt = _synthetic_runs_root(tmp_path, rig)
        code = az.main(
            self._cli_args(
                root, worlds, receipt["planner_code_sha256"], root / "nested_out"
            )
        )
        assert code == 2

    def test_cli_prerequisite_stop_on_a_tampered_log(self, tmp_path):
        rig = rich_rig(tmp_path, "clitamper")
        root, worlds, receipt = _synthetic_runs_root(tmp_path, rig)
        rows = rows_of(rig)
        logged = group_rows_by_decision(rows)
        decision_index = next(
            index
            for index, group in sorted(logged.items())
            if any(
                row.get("chosen") and row["actions"].get("c2") == "damping"
                for row in group
            )
        )
        victim = next(
            row["cand_index"]
            for row in logged[decision_index]
            if row["actions"] == {"c0": "native", "c1": "native", "c2": "damping"}
        )
        tampered = add_noise_to_rows(rows, decision_index, {victim}, 1000.0)
        (root / f"{rig.config.world}__planner_oracle" / "candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in tampered), encoding="utf-8"
        )
        code = az.main(
            self._cli_args(root, worlds, receipt["planner_code_sha256"], tmp_path / "o")
        )
        assert code == 2
