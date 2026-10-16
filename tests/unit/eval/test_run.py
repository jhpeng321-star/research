"""Scoring configured models over configured state sets (``wms score`` / ``wms eval`` driver)."""
import json

import pytest
import torch

from wms.config.eval import EvalConfig, EvalSetConfig
from wms.config.schema import from_mapping
from wms.eval import run as run_mod
from wms.eval.errors import EvalError
from wms.eval.subsets import select_states
from wms.model.checkpoint import save_checkpoint
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig


def veh(i, s, v, lane=0, cav=False):
    return {"id": f"v{i}", "s": s, "v": v, "lane": lane, "present": True, "is_cav": cav}


def truth(k):
    return [veh(1, 10.0 + 5.0 * k, 10.0), veh(2, 50.0 + 2.5 * k, 5.0), veh(3, 0.0, 10.0, cav=True)]


def stream(n=241):
    return [{"time_s": 0.5 * k, "vehicles": truth(k)} for k in range(n)]


def record(world, t0=0.0):
    return {"world": world, "t0_s": t0, "key": f"{world}@{t0}", "source_arm": "native",
            "paired": True, "initial_state": {"time_s": 0.0, "vehicles": truth(0)},
            "recorded": {"native": stream(), "wall": stream()}}


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpts")
    paths = []
    for index, seed in enumerate((1, 2)):
        torch.manual_seed(seed)
        path = root / f"m{index}.pt"
        save_checkpoint(path, WmSModel(ModelSpec(), physics=PhysicsConfig()), {})
        paths.append(path)
    return paths


@pytest.fixture()
def states_file(tmp_path):
    records = [record(f"w{i}", t0) for i in range(4) for t0 in (0.0, 30.0)]
    path = tmp_path / "wall_same.json"
    path.write_text(json.dumps({"windows": records}), encoding="utf-8")
    return path


def eval_config(states_file, models, subset=None, **extra):
    return from_mapping(EvalConfig, {
        "sets": ["wall_same"], "horizons_s": [10.0, 60.0],
        "models": models, "previous_best": "ref",
        "set_specs": {"wall_same": {"states": states_file.as_posix(), "subset": subset}}, **extra})


def test_subset_spec_is_validated():
    assert EvalSetConfig(subset="first_per_world:20").subset == "first_per_world:20"
    for bad in ("first", "first:0", "last:3", "first_per_world:x"):
        with pytest.raises(Exception, match="subset"):
            EvalSetConfig(subset=bad)


def test_select_states_applies_the_named_subset(states_file):
    assert len(list(select_states(states_file, None))) == 8
    assert [r["key"] for r in select_states(states_file, "first:3")] == \
        ["w0@0.0", "w0@30.0", "w1@0.0"]
    per_world = select_states(states_file, "first_per_world:2")
    assert [r["key"] for r in per_world] == ["w0@0.0", "w1@0.0"]
    with pytest.raises(EvalError, match="unknown state subset"):
        select_states(states_file, "bogus:2")


def test_build_result_scores_every_model_and_compares_with_the_reference(checkpoints, states_file):
    cand, ref = checkpoints
    cfg = eval_config(states_file, {"cand": cand.as_posix(), "ref": ref.as_posix()},
                      subset="first:4")
    result = run_mod.build_result(cfg.models, cfg.sets, config=cfg)
    assert result["schema"] == run_mod.RESULT_SCHEMA and result["previous_best"] == "ref"
    assert result["n_states"] == {"wall_same": 4}
    assert set(result["models"]) == {"cand", "ref"}
    assert result["models"]["cand"]["sha256"] != result["models"]["ref"]["sha256"]
    for block in result["models"].values():
        states = block["states_by_set"]["wall_same"]
        assert len(states) == 4 and all(a["error"] is None for s in states
                                        for a in s["arms"].values())
    comparison = result["comparisons"]["cand"]
    assert comparison["verdict"] in ("improved", "regressed", "inconclusive")
    assert set(comparison["sets"]["wall_same"]["horizons"]) == {"60"}
    assert "ref" not in result["comparisons"]


def test_a_model_compared_with_itself_is_inconclusive(checkpoints, states_file):
    ref = checkpoints[0].as_posix()
    cfg = eval_config(states_file, {"same": ref, "ref": ref}, subset="first_per_world:4")
    result = run_mod.build_result(cfg.models, cfg.sets, config=cfg)
    comparison = result["comparisons"]["same"]
    assert comparison["verdict"] == "inconclusive"
    assert comparison["sets"]["wall_same"]["horizons"]["60"]["paired_new_minus_previous"]["lo"] == 0.0


def test_missing_state_file_and_missing_horizon_are_refused(checkpoints, tmp_path):
    cfg = eval_config(tmp_path / "absent.json", {"ref": checkpoints[0].as_posix()})
    with pytest.raises(FileNotFoundError, match="not found"):
        run_mod.build_result(cfg.models, cfg.sets, config=cfg)


def test_eval_cli_writes_the_comparison(checkpoints, states_file, tmp_path, capsys):
    from wms.cli import __main__ as cli

    preset = tmp_path / "mini_eval.toml"
    preset.write_text("\n".join([
        "schema_version = 2", 'name = "mini"', 'sets = ["wall_same"]', "horizons_s = [10.0, 60.0]",
        'previous_best = "ref"', "", "[models]", f'ref = "{checkpoints[1].as_posix()}"', "",
        "[set_specs.wall_same]", f'states = "{states_file.as_posix()}"',
        'subset = "first:2"', ""]), encoding="utf-8")
    out = tmp_path / "out"
    code = cli.main(["eval", "--config", str(preset), "--checkpoint", checkpoints[0].as_posix(),
                     "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "Comparison:" in text
    assert (out / "candidate" / "COMPARISON.md").is_file() and (out / "result.json").is_file()
    assert json.loads((out / "candidate" / "COMPARISON.json").read_text())["schema"] \
        == "wms_eval_comparison/1"
    code = cli.main(["score", "--config", str(preset), "--out", str(tmp_path / "score_out")])
    assert code == 0 and (tmp_path / "score_out" / "result.json").is_file()
