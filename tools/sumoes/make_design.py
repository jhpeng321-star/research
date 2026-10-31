"""Write the frozen S-SUMOES-RUN variant (b) design (seed list + identities) as JSON."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tools.sumoes.config import SumoEsConfig


def build(config: SumoEsConfig) -> dict:
    worlds = []
    for update in range(config.max_updates):
        for index, seed in enumerate(config.gradient_seeds(update)):
            p, d = config.cell_for(seed)
            worlds.append(dict(seed=seed, role="gradient", update=update, index=index,
                               penetration=p, demand=d))
    for index, seed in enumerate(config.panel_seeds()):
        p, d = config.cell_for(seed)
        worlds.append(dict(seed=seed, role="panel", index=index, penetration=p, demand=d))
    return {
        "schema": "sumoes_design/1",
        "experiment": "S-SUMOES-RUN",
        "variant": "b (owner 2026-10-31 14:25)",
        "owner_approval": "REGISTRATION.md 2026-10-31 14:25 OWNER APPROVED TRACK 3",
        "config": config.to_dict() | {"enabled": True},
        "planned_episodes": config.planned_episodes(),
        "planned_sim_s": config.planned_sim_s(),
        "hard_cap_sim_s": config.cap_sim_s,
        "episode": "300 s native warmup + 600 s control; charge 900 sim-s per attempt before dispatch",
        "fitness": "E3 team delay J600 = result.metrics cumulative_600 team (road deficit + source waiting, veh-s; lower better)",
        "pairing": "antithetic pair (+/-) and CTRL0/ours on the same world seed share the native warmup; initial_state_sha256 asserted equal within every pair and vs the panel baseline",
        "acceptance": "accept iff sum(panel J600 after) <= sum(panel J600 before) over the 8 panel worlds; else bitwise rollback of head+Adam (update consumed)",
        "adaptation": "if > 2/3 of updates 0-9 reject: halt, halve sigma target once, resume (single allowed adaptation)",
        "safety_stop": "any collision/teleport/dropped vehicle step, lane-audit call, actual != 900, or exception -> STOP dispatch; no retries",
        "policy": "candidate-B cap-relative residual on the frozen E3 C1 executor (policy_latest.pt sha 64f70af4...): residual = 1.2*tanh(residual_head(f)) with the 137 head scalars = theta; executed every physics tick, never released: clip(native + r, [min(native, -3), max(native, min(1.2, getAccel))]); theta = 0 is bitwise the CTRL0 forced-native path (zero.py 24095a10)",
        "runtime": "cpuQ ~/e3final manifest b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193 (543 files); wrappers zero.py 24095a10, observer.py b22af462, baselines.py c63b14bf (byte-identical S-WMDATA)",
        "host": {"name": "cpuQ", "workers_max": 24, "nice": 19, "threads": 1, "cuda_visible_devices": ""},
        "seed_block": [config.seed_block_first, config.seed_block_last],
        "seed_block_basis": "C4_LOOP_TRAIN (966000-969999) fresh sub-block 969440-969543 directly after S-WMDATA 969400-969439",
        "worlds": worlds,
    }


def main(argv: list[str]) -> int:
    config = SumoEsConfig.load(argv[0])
    payload = build(config)
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    out = Path(argv[1])
    out.write_text(text, encoding="utf-8", newline="\n")
    print(hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
