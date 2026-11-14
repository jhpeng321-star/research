"""Read-only trajectory checks and reporting; no simulation or model selection."""

import json
from pathlib import Path

import numpy as np

from experiments.idm.check_published_idm_fit import ROOT, digest, inference, load_model
from experiments.idm.train_native_damping import atomic_json


def analyze():
    model = load_model()
    records = []
    for path in sorted((ROOT / "results").glob("*.json")):
        if path.name.endswith(".started.json"):
            continue
        record = json.loads(path.read_text())
        assert record["arrays_sha256"] == digest(path.with_suffix(".npz"))
        arrays = np.load(path.with_suffix(".npz"))
        intervals = arrays["intervals"]
        assert intervals.shape == (2400, 7) and np.isfinite(intervals).all()
        totals = intervals.sum(axis=0)
        expected = [
            record["braking_vehicle_seconds"],
            record["acceleration_effort"],
            record["system_seconds_per_request"] * record["requested"],
            record["source_seconds_per_request"] * record["requested"],
            record["ttc_below3_vehicle_seconds"],
            record["collision_involvements"],
            record["teleports"],
        ]
        assert np.allclose(totals, expected, atol=1e-8, rtol=0)
        audit = {"key": record["key"], "arrays_hash_and_totals_pass": True}
        if record["job"][3] == "native":
            seed = record["job"][0]
            source = ROOT.parents[1] / "rllib" / "native_damping_campaign_v1" / "references"
            old = json.loads((source / f"development_{seed}.json").read_text())
            reference = np.asarray(
                [
                    [
                        r["braking_vehicle_seconds"],
                        r["acceleration_energy"],
                        r["requested_not_arrived_vehicle_seconds"],
                    ]
                    for r in old["intervals"]
                ]
            )
            errors = np.abs(intervals[:, :3] - reference).max(axis=0)
            assert max(errors) < 1e-8
            audit["native_reference_max_interval_error"] = errors.tolist()
        else:
            observations, actions = arrays["observations"], arrays["actions"]
            assert len(observations) == len(actions)
            indices = np.unique(
                np.linspace(0, len(actions) - 1, min(1000, len(actions))).astype(int)
            )
            replay = inference(model, observations[indices, 1:])
            assert np.array_equal(replay, actions[indices, 0], equal_nan=True)
            c = record["counts"]
            audit.update(
                replayed_actions=len(indices),
                singleton_inference_exact=True,
                command_fraction=c.get("commands", 0) / c["cav_decisions"],
                modified_command_fraction=c.get("output_modified", 0)
                / max(1, c.get("commands", 0)),
                native_fraction=1 - c.get("commands", 0) / c["cav_decisions"],
                invalid_outputs=c.get("invalid_output", 0),
            )
        records.append(audit)
    atomic_json(
        ROOT / "verification.json", {"completed_verified": len(records), "records": records}
    )
    print(json.dumps(records, indent=2))

    if (ROOT / "summary.json").exists():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summary = json.loads((ROOT / "summary.json").read_text())["comparisons"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        metrics = [
            ("braking_change_pct", "Severe braking change (%)"),
            ("effort_change_pct", "Acceleration effort change (%)"),
            ("system_time_delta", "System time change (s / request)"),
            ("ttc_exposure_delta", "TTC < 3 s exposure change (vehicle-s)"),
        ]
        for ax, (metric, label) in zip(axes.flat, metrics):
            ax.axhline(0, color="#64748b", lw=1)
            for seed, demand, color in [(911001, 2000, "#2563eb"), (911003, 2200, "#e87924")]:
                rows = [r for r in summary if r["job"][0] == seed and r["job"][3] == "workspace"]
                direct = next(r for r in summary if r["job"][0] == seed and r["job"][3] == "direct")
                ax.plot(
                    [r["job"][2] * 100 for r in rows],
                    [r[metric] for r in rows],
                    marker="o",
                    color=color,
                    label=f"{demand}: constrained",
                )
                ax.scatter(
                    [20],
                    [direct[metric]],
                    s=100,
                    marker="D",
                    facecolors="white",
                    edgecolors=color,
                    linewidth=2,
                    zorder=4,
                    label=f"{demand}: direct",
                )
            ax.set_title(label, fontsize=11)
            ax.set_xticks([5, 10, 20, 30])
            ax.set_xlabel("Nominal CAV penetration (%)")
            ax.grid(alpha=0.15)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            "External RL controller in our IDM merge\n"
            "One exposed seed per demand; lower costs are better; direct output tested only at 20%",
            fontsize=13,
        )
        path = Path("output/published_idm_fit_2026-09-06.png")
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    analyze()
