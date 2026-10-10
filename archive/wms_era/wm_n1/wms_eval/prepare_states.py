"""Build the 20-state TRAIN development list from the frozen reference."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CHUNK = ROOT / "_ext" / "campaign_chunk4"


def _actual_base(name: str) -> tuple[str, Path]:
    expected = CHUNK / f"{name}__base"
    if expected.is_dir():
        return name, expected
    prefix = name.split("m", 1)[0]
    candidates = sorted(CHUNK.glob(f"{prefix}m*__base"))
    if len(candidates) != 1:
        raise RuntimeError(f"cannot resolve TRAIN state {name!r}: {candidates}")
    actual = candidates[0].name.removesuffix("__base")
    return actual, candidates[0]


def build(out_path: Path | None = None) -> Path:
    ref = json.loads((ROOT / "_ref" / "HORIZON_ERROR.json").read_text(encoding="utf-8"))
    records = []
    for listed_name in ref["meta"]["states"]:
        world, base = _actual_base(listed_name)
        manifest = json.loads(
            (base / "saved_state" / "STATE_MANIFEST.json").read_text(encoding="utf-8")
        )
        row = manifest["physics_row_at_t0"]
        cav_ids = list(row.get("cav_ids", ()))
        records.append({
            "world": world,
            "listed_reference_state": listed_name,
            "split": "TRAIN",
            "t0_s": float(manifest["branch_point_t0_s"]),
            "horizon_s": 300.0,
            "dt_s": 0.5,
            "seed": int(manifest.get("seed", 0)),
            "v0_mps": 27.78,
            "base_run_dir": f"_ext/campaign_chunk4/{world}__base",
            "recorded_paths": {
                "native": f"_ext/campaign_chunk4/{world}__native/run/physics_steps.jsonl",
                "pi": f"_ext/campaign_chunk4/{world}__pi/run/physics_steps.jsonl",
            },
            "cav_ids": cav_ids,
            "state_row": row,
            "cav_controllers": {"arm": "native"},
        })
    out_path = out_path or (Path(__file__).resolve().parent / "states_dev.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    print(build())
