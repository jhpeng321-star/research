"""S-SUMOES driver skeleton: update loop, budget ledger, receipts. Default OFF.

Per update: draw D directions -> WM screen keeps P -> SUMO-evaluate the P
antithetic pairs on W fresh TRAIN worlds (CRN = identical world seed within a
pair) -> screened ES gradient -> Adam step -> held-in panel acceptance with
actor+Adam rollback on rejection.  The real SUMO evaluator is the registered
cpuQ job protocol reusing the S-WMDATA/S-SPEEDHEAD runtime wrappers and is NOT
implemented in this design task; everything here runs on the CPU stub.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from .config import SumoEsConfig
from .es import EsState, HeadAdam, estimate, panel_improved


class BudgetExceeded(RuntimeError):
    pass


class MechanicalFailure(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    cap_sim_s: int
    episode_charge_sim_s: int
    charged_sim_s: int = 0
    actual_sim_s: int = 0
    jobs: int = 0

    def authorize(self, job_id: str) -> int:
        if self.remaining_sim_s() < self.episode_charge_sim_s:
            raise BudgetExceeded(
                f"{job_id}: remaining {self.remaining_sim_s()} < charge "
                f"{self.episode_charge_sim_s}; cap {self.cap_sim_s}")
        self.charged_sim_s += self.episode_charge_sim_s
        self.jobs += 1
        return self.episode_charge_sim_s

    def record_actual(self, job_id: str, actual: float) -> None:
        actual = min(max(int(actual), 0), self.episode_charge_sim_s)
        self.actual_sim_s += actual

    def remaining_sim_s(self) -> int:
        return self.cap_sim_s - self.charged_sim_s

    def to_dict(self) -> dict:
        return dict(cap_sim_s=self.cap_sim_s, charged_sim_s=self.charged_sim_s,
                    actual_sim_s=self.actual_sim_s, jobs=self.jobs)


class Evaluator(Protocol):
    def evaluate(self, theta: torch.Tensor, world_seed: int, update: int,
                 tag: str) -> dict:
        ...


class Screener(Protocol):
    def screen(self, theta: torch.Tensor, sigma: float, directions: torch.Tensor,
               update: int) -> dict:
        ...


class AcceptAllScreener:
    """Test/default screener: keeps the first ``keep`` directions unconditionally."""

    def __init__(self, keep: int) -> None:
        self.keep = keep

    def screen(self, theta, sigma, directions, update):
        kept = list(range(min(self.keep, directions.shape[0])))
        return dict(kept=kept, predicted_gain=[None] * len(kept), rejected=[])


class SumoRuntimeEvaluator:
    """Refuses: the real arms need the registered cpuQ runtime + interactive ACK."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "S-SUMOES-DESIGN ships the skeleton only: real SUMO episodes require the "
            "registered cpuQ E3 runtime (b3c031ad) with the byte-identical S-WMDATA/"
            "S-SPEEDHEAD wrappers and an interactive registration ACK; NO SUMO in this task")


@dataclass
class SumoEsDriver:
    config: SumoEsConfig
    evaluator: Evaluator
    screener: Screener
    sensitivity: float = 20.0
    receipts: list[dict] = field(default_factory=list)
    ledger: BudgetLedger | None = None

    def initial_state(self, dimension: int = 137) -> EsState:
        return EsState(theta=torch.zeros(dimension, dtype=torch.float64),
                       adam=HeadAdam(lr=self.config.adam_lr, beta1=self.config.adam_beta1,
                                     beta2=self.config.adam_beta2, eps=self.config.adam_eps),
                       sigma_target_residual_rms_mps2=self.config.sigma_target_residual_rms_mps2,
                       sigma_min_mps2=self.config.sigma_min_mps2,
                       sigma_halve_every_accepted=self.config.sigma_halve_every_accepted,
                       direction_seed=self.config.direction_seed)

    def _episode(self, state: EsState, theta: torch.Tensor, seed: int, update: int,
                 tag: str) -> dict:
        if self.ledger is None:
            self.ledger = BudgetLedger(self.config.cap_sim_s, self.config.episode_charge_sim_s)
        charged = self.ledger.authorize(f"s{seed}_{tag}")
        result = self.evaluator.evaluate(theta, seed, update, tag)
        self.ledger.record_actual(result["job"], result.get("actual_sim_s", charged))
        if not result.get("safety_clean", False):
            raise MechanicalFailure(
                f"{result['job']}: safety gate failed; stop dispatch, no retry authority")
        return result

    def panel_scores(self, state: EsState, theta: torch.Tensor, update: int,
                     tag: str) -> list[float]:
        return [self._episode(state, theta, seed, update, f"{tag}_s{seed}")["J600"]
                for seed in self.config.panel_seeds()]

    def baseline(self, state: EsState) -> dict:
        scores = self.panel_scores(state, state.theta, 0, "panel_ctrl0")
        state.panel_j = list(scores)
        canary = f"s{self.config.panel_seeds()[0]}_panel_ctrl0"
        return dict(update=-1, kind="ctrl0_panel_baseline", canary=canary
                    + " (canary; reused as baseline[0])",
                    panel_seeds=list(self.config.panel_seeds()), panel_J=scores,
                    budget=self.ledger.to_dict())

    def run_update(self, state: EsState) -> dict:
        update = state.update_count
        seeds = list(self.config.gradient_seeds(update))
        sigma = state.sigma(self.sensitivity)
        directions = state.directions(update, self.config.wm_screen_directions)
        screen = self.screener.screen(state.theta, sigma, directions, update)
        kept = [i for i in screen["kept"] if i < directions.shape[0]][:self.config.pairs_per_update]
        screen_receipt = dict(kept=kept, rejected=screen["rejected"],
                              predicted_gain=screen["predicted_gain"])
        if not kept:
            receipt = dict(update=update, seeds=seeds, sigma=sigma,
                           sigma_predicted_command_rms=sigma * self.sensitivity,
                           screen=screen_receipt, pair_deltas=[], pair_J=[], grad_norm=0.0,
                           panel_J_before=list(state.panel_j), panel_J_after=list(state.panel_j),
                           accepted=None, no_step=True, budget=self.ledger.to_dict())
            state.update_count += 1
            self.receipts.append(receipt)
            return receipt
        deltas, pair_j = [], []
        for index in kept:
            per_side = {}
            for sign, label in ((1, "plus"), (-1, "minus")):
                theta_side = state.theta + sign * sigma * directions[index]
                per_side[label] = [self._episode(state, theta_side, seed, update,
                                                 f"u{update}_d{index}_{label}_s{seed}")["J600"]
                                   for seed in seeds]
            deltas.append(sum(p - m for p, m in zip(per_side["plus"], per_side["minus"]))
                          / len(seeds))
            pair_j.append(dict(plus=per_side["plus"], minus=per_side["minus"]))
        gradient = estimate(directions[kept], torch.tensor(deltas, dtype=torch.float64),
                            sigma)
        adam_snapshot = state.adam.snapshot()
        theta_candidate = state.adam.step(state.theta, gradient)
        candidate_panel = self.panel_scores(state, theta_candidate, update, "panel_after")
        accepted = panel_improved(state.panel_j, candidate_panel)
        receipt = dict(update=update, seeds=seeds, sigma=sigma,
                       sigma_predicted_command_rms=sigma * self.sensitivity,
                       screen=screen_receipt,
                       pair_deltas=deltas, pair_J=pair_j, grad_norm=float(gradient.norm()),
                       panel_J_before=list(state.panel_j), panel_J_after=candidate_panel,
                       accepted=accepted, budget=self.ledger.to_dict())
        if accepted:
            state.theta = theta_candidate
            state.panel_j = list(candidate_panel)
            state.accepted_count += 1
        else:
            state.adam.restore(adam_snapshot)
        state.update_count += 1
        self.receipts.append(receipt)
        return receipt

    def run(self, state: EsState | None = None, updates: int | None = None) -> dict:
        if not self.config.enabled:
            raise PermissionError(
                "S-SUMOES is design-frozen and default OFF: enable only inside the "
                "registered run task after the exposure audit and ACK")
        if self.ledger is None:
            self.ledger = BudgetLedger(self.config.cap_sim_s, self.config.episode_charge_sim_s)
        if state is None:
            state = self.initial_state()
            self.receipts.append(self.baseline(state))
        for _ in range(self.config.max_updates if updates is None else updates):
            if state.update_count >= self.config.max_updates:
                break
            self.run_update(state)
        return dict(final_state=state.to_dict(), receipts=self.receipts,
                    budget=self.ledger.to_dict())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S-SUMOES driver skeleton (default OFF)")
    parser.add_argument("--config", default="configs/c4_sumoes_design.toml")
    parser.add_argument("--evaluator", choices=("stub", "sumo"), default="stub")
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=None)
    args = parser.parse_args(argv)
    config = SumoEsConfig.load(args.config)
    if args.evaluator == "sumo":
        SumoRuntimeEvaluator()
    from .stub import StubSimulator
    evaluator = StubSimulator()
    screener = AcceptAllScreener(config.pairs_per_update)
    driver = SumoEsDriver(config=config, evaluator=evaluator, screener=screener)
    summary = driver.run(updates=args.updates)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(config=config.to_dict(), budget=summary["budget"],
                   final_state=summary["final_state"], receipts=summary["receipts"])
    (out / "SUMOES_RECEIPTS.json").write_text(
        json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(budget=summary["budget"],
                          updates=len(summary["receipts"]) - 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
