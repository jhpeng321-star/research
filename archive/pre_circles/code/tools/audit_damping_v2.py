"""Read-only trajectory/reward audit and scientific figures; no simulations."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.idm.train_native_damping import atomic_json, load_rollout
from idm.damping_selection import case_metrics
from idm.damping_v2_actor import FixedExpert, SelectableActor
from idm.damping_v2_reward import prepare_batch, reward_series

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "runs/probe/native_selectable_damping_v2_check"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replay(actor, rollout):
    count = 0
    for time in np.unique(rollout["actor_times"]):
        idx = np.flatnonzero(rollout["actor_times"] == time)
        with torch.no_grad():
            prediction = actor(torch.tensor(rollout["actor_observations"][idx])).numpy()
        if not np.array_equal(prediction, rollout["actions"][idx]):
            raise ValueError("saved deterministic actions do not replay exactly")
        count += len(idx)
    return count


def plot_short():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col", constrained_layout=True)
    for col, profile in enumerate(("pulse", "sharp_brake")):
        for arm, color in (("native", "#526879"), ("learned", "#b34331")):
            data = read(CHECK / f"{arm}_{profile}.json")
            t = np.array([r["time"] for r in data["trace"]])
            trace = np.array([r["vehicles"] for r in data["trace"]])
            axes[0, col].plot(t, trace[:, 1, 1], label=arm.capitalize(), color=color, lw=1.8)
            axes[1, col].plot(
                t, np.cumsum(0.5 * (trace[:, 1:, 2] ** 2).sum(1)), color=color, lw=1.8
            )
            if arm == "native":
                axes[0, col].plot(
                    t, trace[:, 0, 1], color="#9b9b9b", ls="--", label="Scripted leader"
                )
        axes[0, col].set_title(
            "Two braking pulses" if profile == "pulse" else "Sharp braking and recovery"
        )
        axes[1, col].set_xlabel("Time (s)")
        axes[0, col].set_ylabel("CAV speed (m/s)")
        axes[1, col].set_ylabel("CAV + follower cumulative effort (m²/s³)")
        for ax in axes[:, col]:
            ax.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Native release works; the trained active response still adds recovery effort", fontsize=14
    )
    output = ROOT / "output/native_selectable_damping_v2"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "mechanism_traces.png", dpi=180)
    fig.savefig(output / "mechanism_traces.pdf")
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    short = read(CHECK / "summary.json")
    manifest = read(CHECK / "manifest.json")
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    actor = SelectableActor()
    payload = torch.load(CHECK / "checkpoint_012.pt", weights_only=True, map_location="cpu")
    actor.load_state_dict(payload["actor"])
    comparisons = {}
    samples = 0
    short_audit = {}
    for profile in ("steady", "pulse", "catchup", "sharp_brake"):
        for arm, model in (("expert", FixedExpert()), ("learned", actor)):
            r = load_rollout(CHECK / f"{arm}_{profile}.json")
            samples += replay(model, r)
            trace = np.array([q["vehicles"] for q in r["trace"]])
            effort = 0.5 * (trace[:, 1:, 2] ** 2).sum(0)
            assert np.isclose(effort.sum(), r["summary"]["totals"]["acceleration_energy"])
            targets = [
                c["target_acceleration"]
                for c in r["commands"]
                if c["target_acceleration"] is not None
            ]
            assert all(-1 - 1e-9 <= a <= 0.5 + 1e-9 for a in targets)
            short_audit[f"{arm}_{profile}"] = dict(
                per_vehicle_effort=effort.tolist(),
                minimum_actual_acceleration=trace[:, 1:, 2].min(0).tolist(),
                residual_range=[float(r["actions"][:, 1].min()), float(r["actions"][:, 1].max())],
                all_issued_accelerations_within_bounds=True,
            )
    plot_short()
    corridor = CHECK / "corridor"
    if not (corridor / "startup_recovery_01/native_911003.json").exists():
        atomic_json(
            CHECK / "short_audit.json",
            dict(short_audit=short_audit, exact_action_replay_samples=samples),
        )
        print("Short audit/figures complete; corridor recovery not yet complete.")
        return
    registration = read(corridor / "manifest.json")
    for name, expected in registration["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for path, expected in registration["cache"].items():
        path = Path(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["json_sha256"]
        assert (
            hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest()
            == expected["arrays_sha256"]
        )
    reward_audits = {}
    for seed in (911001, 911003):
        native_path = corridor / f"native_{seed}.json"
        if seed == 911003:
            native_path = corridor / "startup_recovery_01/native_911003.json"
        native = load_rollout(native_path)
        old_native = load_rollout(
            ROOT / f"runs/rllib/native_damping_campaign_v1/references/development_{seed}.json"
        )
        repeat = case_metrics(native["summary"], old_native["summary"])
        assert all(value == 0 for key, value in repeat.items() if key != "native_ttc")
        assert native["intervals"] == old_native["intervals"]
        rows = {}
        for arm in ("analytical", "expert", "learned"):
            path = (
                ROOT / f"runs/rllib/native_damping_campaign_v1/analytical/development_{seed}.json"
                if arm == "analytical"
                else corridor / f"{arm}_{seed}.json"
            )
            own = load_rollout(path)
            for key, total in own["summary"]["totals"].items():
                assert np.isclose(
                    sum(r[key] for r in own["intervals"]), total, rtol=1e-10, atol=1e-7
                )
            rows[arm] = dict(
                paired=case_metrics(own["summary"], native["summary"]), absolute=own["summary"]
            )
            if arm != "analytical":
                samples += replay(actor if arm == "learned" else FixedExpert(), own)
                rewards, audit = reward_series(own, native)
                batch = prepare_batch(own, native)
                exact = np.cumsum(rewards[::-1])[::-1][own["actor_times"]]
                np.testing.assert_allclose(batch["actor_advantages"], exact, rtol=1e-6, atol=1e-6)
                # Since deterministic eval stores V=0, this also checks every
                # early actor receives terminal cost after its vehicle has left.
                terminal = audit["terminal_penalty"]
                without_terminal = rewards.copy()
                without_terminal[-1] += terminal
                removed = np.cumsum(without_terminal[::-1])[::-1][own["actor_times"]]
                np.testing.assert_allclose(removed - exact, terminal, rtol=1e-7, atol=1e-7)
                duty = np.asarray(own["intervention_fractions"])
                preference = 0.02 * duty / 2400
                assert audit["intervention_cost"] == float(-preference.sum())
                no_cost = np.cumsum((rewards + preference)[::-1])[::-1][own["actor_times"]]
                expected_preference = np.cumsum(preference[::-1])[::-1][own["actor_times"]]
                np.testing.assert_allclose(no_cost - exact, expected_preference, atol=1e-7)
                audit.update(
                    full_clock_downstream_credit_verified=True,
                    terminal_penalty_preserved_for_every_actor=True,
                    intervention_cost_preserved_for_every_actor=True,
                    mean_requested_fraction=float(duty.mean()),
                    mean_executed_fraction=float(np.mean(own["executed_fractions"])),
                )
                reward_audits[f"{arm}_{seed}"] = audit
        comparisons[str(seed)] = dict(native=native["summary"], native_repeat_exact=True, arms=rows)
    metrics = [comparisons[str(seed)]["arms"]["learned"]["paired"] for seed in (911001, 911003)]
    constraints = all(
        m["system_time"] <= 2
        and m["source_wait"] <= 2
        and m["arrived"] >= 0
        and m["unfinished"] <= 0
        and m["uninserted"] <= 0
        and m["ttc"] <= max(5, 0.05 * m["native_ttc"])
        and m["collisions"] <= 0
        and m["teleports"] <= 0
        for m in metrics
    )
    benefits = np.mean([m["composite"] for m in metrics]) >= 0.05 and all(
        np.mean([m[k] for m in metrics]) >= 0 for k in ("braking", "energy")
    )
    result = dict(
        version="2.0",
        mechanics_pass=True,
        short_mechanism_pass=short["mechanism_gate_pass"],
        corridor_constraints_pass=bool(constraints),
        corridor_benefit_pass=bool(benefits),
        comparisons=comparisons,
        reward_audits=reward_audits,
        short_audit=short_audit,
        exact_deterministic_action_replay_samples=samples,
        successful_global_seconds=18000,
        charged_global_seconds=20400,
        registered_maximum=22800,
        accepted_checkpoint=False,
        expensive_campaign_authorized=False,
        decision="NO_GO_mechanics_corrected_but_damping_efficacy_not_established",
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    atomic_json(CHECK / "completed_physical_audit.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "mechanics_pass",
                    "short_mechanism_pass",
                    "corridor_constraints_pass",
                    "corridor_benefit_pass",
                    "exact_deterministic_action_replay_samples",
                    "decision",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
