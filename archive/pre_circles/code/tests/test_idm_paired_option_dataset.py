import json

import numpy as np
import pytest

from idm.paired_option_dataset import (
    PHYSICAL_OBSERVATION_DIM,
    RECORD_SCHEMA,
    PairedOptionRecord,
    load_dataset,
    write_dataset,
)
from idm.paired_option_env import PairedIDMOptionEnv


def _row(sample_id="row", seed=1, reward=0.25):
    return PairedOptionRecord(
        sample_id=sample_id,
        source="fixture",
        demand=2000,
        seed=seed,
        penetration=0.1,
        decision_index=1,
        decision_step=10,
        vehicle_id="cav",
        observation=tuple(float(index) for index in range(PHYSICAL_OBSERVATION_DIM)),
        activate_reward=reward,
        physical_utility=reward,
        constraints_pass=True,
        classification="helpful" if reward > 0 else "harmful",
        dose_steps=20,
    )


def test_record_rejects_future_or_mask_fields_in_actor_vector():
    with pytest.raises(ValueError, match="expected 134"):
        PairedOptionRecord.from_dict(
            {
                **_row().to_dict(),
                "schema": RECORD_SCHEMA,
                "observation": [0.0] * 136,
            }
        )


def test_one_step_environment_preserves_binary_reward():
    env = PairedIDMOptionEnv([_row(reward=-2.0)], seed=4)
    obs, info = env.reset(options={"index": 0})
    assert obs.shape == (PHYSICAL_OBSERVATION_DIM,)
    assert info["sample_id"] == "row"
    _, reward, terminated, truncated, result = env.step(1)
    assert reward == -2.0
    assert terminated and not truncated
    assert result["executed_action"] == 1
    env.reset(options={"index": 0})
    assert env.step(0)[1] == 0.0


def test_dataset_write_load_hash_and_grouped_split(tmp_path):
    rows = [_row("train", 1), _row("validation", 2, -0.1)]
    manifest = {"schema": "idm_paired_option_dataset/1", "dataset_sha256": "fixture"}
    written = write_dataset(
        tmp_path, rows, manifest, train_seeds=[1], validation_seeds=[2]
    )
    assert written["split_counts"] == {"train": 1, "validation": 1}
    train, loaded = load_dataset(tmp_path, "train")
    assert [row.sample_id for row in train] == ["train"]
    assert loaded["train_seeds"] == [1]
    path = tmp_path / "records.jsonl"
    payload = json.loads(path.read_text().splitlines()[0])
    payload["observation"][0] = float(np.pi)
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(tmp_path, "train")
