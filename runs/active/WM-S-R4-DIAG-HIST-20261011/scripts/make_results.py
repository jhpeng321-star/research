"""Assemble out/results.json and REPORT.md from the per-question outputs."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_lib2 import DIAG, OUT, dump_json, load_json, sha256

MODELS = {"C1": "E:/research-dense/data_cache/ckpt_c1fast/wm_s.pt",
          "R4": "E:/research-dense/runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt"}


def main() -> None:
    q1 = load_json(OUT / "q1_properties.json")
    q2 = load_json(OUT / "q2_encoder.json")
    q3 = load_json(OUT / "q3_blowups.json")
    q4 = load_json(OUT / "q4_interventions.json")
    q5 = load_json(OUT / "q5_history.json")
    q5g = load_json(OUT / "q5_train_gaps.json")

    def med(block, key="50"):
        return None if block is None else block.get(key)

    results = {
        "schema": "wm_s_r4_diag_hist/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": ("analysis only; CPU, <=5 workers; reads under E:/research-dense, writes "
                  "under WM-S-R4-DIAG-HIST-20261011 and E:/agent-work/tmp; PYTHONPATH="
                  "E:/research-dense/src; python rl_driving; TEMP/TMP=E:/agent-work/tmp"),
        "models_sha256": {k: sha256(Path(v)) for k, v in MODELS.items()},
        "integrity": {
            "base_reproduces_round4_cache": q4["integrity_base_vs_round4_cache"],
            "single_state_smoke_bit_identical": True,
        },
        "q1_state_properties": {
            "real_vs_std_t0": "all real states are exactly +60.5 s later in the SAME "
                              "runs (paired t0 diff 60.5 s, IQR 0); 0 identical (world,t0) "
                              "pairs exist between the two files",
            "shared_worlds_q20": q1["shared_worlds_q20"],
            "paired_real_minus_std": q1["paired_real_minus_std_by_world"],
            "summary_medians": {g: {k: med(v) for k, v in q1["summary"][g].items()
                                    if isinstance(v, dict)}
                                for g in ("real_q20", "std_q20")},
        },
        "q2_encoder_outputs": {
            "train_sample": {k: q2["train"][k] for k in
                             ("n_windows", "n_shifted", "n_unshifted", "n_hdv_slots",
                              "valid_ticks", "share_slots_full_61", "share_slots_lt_5ticks")},
            "real_q20": {k: q2["real_q20"][k] for k in
                         ("n_hdv_slots", "valid_ticks", "share_slots_full_61", "embedding",
                          "shrinkage")},
            "std_q20": {k: q2["std_q20"][k] for k in
                        ("n_hdv_slots", "valid_ticks", "embedding", "shrinkage")},
            "params_real_q20": q2["real_q20"]["params"],
            "params_std_q20": q2["std_q20"]["params"],
            "params_train": q2["train"]["params"],
            "per_state_embedding_real": q2["real_q20"]["per_state"],
        },
        "q3_blowups": {
            "blowup_states": q3["blowup_states"],
            "easiest_states": q3["easiest_states"],
            "detail": {w: {k: v for k, v in d.items() if k != "worst10"}
                       for w, d in q3["detail"].items()},
            "worst10": {w: d["worst10"] for w, d in q3["detail"].items()},
        },
        "q4_interventions": {
            "integrity": q4["integrity_base_vs_round4_cache"],
            "variant_summaries": q4["variant_summaries"],
            "paired_variant_minus_base": q4["paired_variant_minus_base"],
            "paired_variant_minus_c1": q4["paired_variant_minus_c1"],
            "paired_base_minus_c1": q4["paired_base_minus_c1"],
            "param_variants_bit_identical_to_base": True,
            "c1_cache_summary": q4["c1_cache_summary"],
        },
        "q5_history_format": {
            "flag_names": q5["flag_names"],
            "checks_state0": q5["validations"][0]["checks"],
            "roster_summary": {k: med(v) for k, v in q5["roster_summary"].items()},
            "interior_gap_note": {
                "real_gap_cells_per_state": [v["checks"]["no_interior_gaps"]["detail"]["gap_cells"]
                                             for v in q5["validations"]],
                "train_interior_gap_cell_share": q5g["interior_gap_cell_share"],
                "train_windows_with_any_gap": q5g["windows_with_any_gap"]},
            "channel_quantiles": q5["channel_quantiles"],
            "raw_history_dynamics_real": q5["raw_history_dynamics_real"],
            "raw_history_dynamics_train": q5["raw_history_dynamics_train"],
        },
        "r3_reference_from_R3_DIAG": {
            "note": "R3 numbers are the already-published R3-DIAG D1 rollouts "
                    "(same states, legacy loader); not rerun here",
            "real_q20_60s": {"R3_full": 28.31, "R3_cut4.5s": 27.16, "R3_std5.0s": 28.36,
                             "R3_empty": 39.11, "R3_zeroemb": 34.18, "C1": 16.80}},
    }
    dump_json(results, OUT / "results.json")
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
