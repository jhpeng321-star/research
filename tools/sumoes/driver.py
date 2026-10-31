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


def _sensitivity_from(results: list[dict]) -> float | None:
    rows = [r["head_stats"] for r in results
            if r.get("head_stats") and r["head_stats"].get("rows")]
    if not rows:
        return None
    total = sum(h["rows"] for h in rows)
    return (sum(h["rows"] * h["sensitivity_rms"] ** 2 for h in rows) / total) ** 0.5


@dataclass
class SumoEsDriver:
    config: SumoEsConfig
    evaluator: Evaluator
    screener: Screener
    sensitivity: float = 20.0
    receipts: list[dict] = field(default_factory=list)
    ledger: BudgetLedger | None = None
    world_hash: dict = field(default_factory=dict)
    state_path: Path | None = None
    canary_arm_check: bool = False

    def initial_state(self, dimension: int = 137) -> EsState:
        return EsState(theta=torch.zeros(dimension, dtype=torch.float64),
                       adam=HeadAdam(lr=self.config.adam_lr, beta1=self.config.adam_beta1,
                                     beta2=self.config.adam_beta2, eps=self.config.adam_eps),
                       sigma_target_residual_rms_mps2=self.config.sigma_target_residual_rms_mps2,
                       sigma_min_mps2=self.config.sigma_min_mps2,
                       sigma_halve_every_accepted=self.config.sigma_halve_every_accepted,
                       direction_seed=self.config.direction_seed)

    def _batch(self, requests: list[dict]) -> list[dict]:
        """Charge every request BEFORE dispatch, run them (in parallel when the evaluator
        supports it), enforce the safety gate and CRN world identity per seed."""
        if self.ledger is None:
            self.ledger = BudgetLedger(self.config.cap_sim_s, self.config.episode_charge_sim_s)
        need = len(requests) * self.config.episode_charge_sim_s
        if self.ledger.remaining_sim_s() < need:
            raise BudgetExceeded(f"batch of {len(requests)} needs {need} > remaining "
                                 f"{self.ledger.remaining_sim_s()}; cap {self.config.cap_sim_s}")
        for request in requests:
            self.ledger.authorize(request["tag"])
        if hasattr(self.evaluator, "evaluate_many"):
            results = self.evaluator.evaluate_many(requests)
        else:
            results = [self.evaluator.evaluate(r["theta"], r["seed"], r["update"], r["tag"])
                       for r in requests]
        for request, result in zip(requests, results):
            self.ledger.record_actual(result["job"], result.get(
                "actual_sim_s", self.config.episode_charge_sim_s))
            if not result.get("safety_clean", False):
                raise MechanicalFailure(
                    f"{result['job']}: safety gate failed; stop dispatch, no retry authority")
            digest = result.get("initial_state_sha256")
            known = self.world_hash.setdefault(request["seed"], digest)
            if digest is None or known != digest:
                raise MechanicalFailure(f"{result['job']}: initial_state_sha256 {digest} != "
                                        f"{known} for world {request['seed']} (CRN broken)")
        return results

    def _episode(self, state: EsState, theta: torch.Tensor, seed: int, update: int,
                 tag: str) -> dict:
        return self._batch([dict(theta=theta, seed=seed, update=update, tag=tag)])[0]

    def panel_results(self, theta: torch.Tensor, update: int, tag: str) -> list[dict]:
        return self._batch([dict(theta=theta, seed=seed, update=update, tag=f"{tag}_s{seed}")
                            for seed in self.config.panel_seeds()])

    def panel_scores(self, state: EsState, theta: torch.Tensor, update: int,
                     tag: str) -> list[float]:
        return [r["J600"] for r in self.panel_results(theta, update, tag)]

    def _calibrate(self, results: list[dict]) -> None:
        measured = _sensitivity_from(results)
        if measured is not None:
            self.sensitivity = measured

    def baseline(self, state: EsState) -> dict:
        seeds = self.config.panel_seeds()
        canary = None
        if self.canary_arm_check:
            # Registered canary: the ours path at theta=0 must reproduce the byte-identical
            # CTRL0 adapter on the first panel world (same J600, same world hash) first.
            pair = self._batch([dict(theta=state.theta, seed=seeds[0], update=-1, arm="ctrl0",
                                     tag=f"base_s{seeds[0]}_ctrl0"),
                                dict(theta=state.theta, seed=seeds[0], update=-1,
                                     tag=f"base_s{seeds[0]}_ours0")])
            if pair[0]["J600"] != pair[1]["J600"]:
                raise MechanicalFailure(f"canary: ours@0 J600 {pair[1]['J600']} != CTRL0 "
                                        f"{pair[0]['J600']} (zero identity broken)")
            canary = dict(ctrl0=pair[0], ours0=pair[1], identical=True)
            self.save(state, receipts=self.receipts + [dict(update=-1, kind="canary",
                                                            canary=canary)])
            rest = self._batch([dict(theta=state.theta, seed=s, update=-1,
                                     tag=f"base_s{s}_ours0") for s in seeds[1:]])
            results = [pair[1]] + rest
        else:
            results = self.panel_results(state.theta, -1, "base")
        self._calibrate(results)
        scores = [r["J600"] for r in results]
        state.panel_j = list(scores)
        return dict(update=-1, kind="ctrl0_panel_baseline", canary=canary,
                    panel_seeds=list(seeds), panel_J=scores,
                    panel_mean=sum(scores) / len(scores), sensitivity=self.sensitivity,
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
                           sensitivity=self.sensitivity,
                           sigma_predicted_command_rms=sigma * self.sensitivity,
                           screen=screen_receipt, pair_deltas=[], pair_J=[], grad_norm=0.0,
                           panel_J_before=list(state.panel_j), panel_J_after=list(state.panel_j),
                           accepted=None, no_step=True, budget=self.ledger.to_dict())
            state.update_count += 1
            self.receipts.append(receipt)
            self.save(state)
            return receipt
        requests = []
        for index in kept:
            for sign, label in ((1, "plus"), (-1, "minus")):
                theta_side = state.theta + sign * sigma * directions[index]
                requests += [dict(theta=theta_side, seed=seed, update=update,
                                  tag=f"u{update:02d}_d{index}_{label}_s{seed}",
                                  index=index, side=label) for seed in seeds]
        results = self._batch(requests)
        deltas, pair_j = [], []
        for index in kept:
            per_side = {label: [res["J600"] for req, res in zip(requests, results)
                                if req["index"] == index and req["side"] == label]
                        for label in ("plus", "minus")}
            deltas.append(sum(p - m for p, m in zip(per_side["plus"], per_side["minus"]))
                          / len(seeds))
            pair_j.append(dict(plus=per_side["plus"], minus=per_side["minus"]))
        gradient = estimate(directions[kept], torch.tensor(deltas, dtype=torch.float64),
                            sigma)
        adam_snapshot = state.adam.snapshot()
        theta_candidate = state.adam.step(state.theta, gradient)
        panel_results = self.panel_results(theta_candidate, update, f"u{update:02d}_panel")
        candidate_panel = [r["J600"] for r in panel_results]
        accepted = panel_improved(state.panel_j, candidate_panel)
        receipt = dict(update=update, seeds=seeds, sigma=sigma, sensitivity=self.sensitivity,
                       sigma_predicted_command_rms=sigma * self.sensitivity,
                       screen=screen_receipt,
                       pair_deltas=deltas, pair_J=pair_j, grad_norm=float(gradient.norm()),
                       panel_J_before=list(state.panel_j), panel_J_after=candidate_panel,
                       panel_mean_before=sum(state.panel_j) / len(state.panel_j),
                       panel_mean_after=sum(candidate_panel) / len(candidate_panel),
                       accepted=accepted, budget=self.ledger.to_dict())
        if accepted:
            state.theta = theta_candidate
            state.panel_j = list(candidate_panel)
            state.accepted_count += 1
            self._calibrate(panel_results)
        else:
            state.adam.restore(adam_snapshot)
        state.update_count += 1
        self.receipts.append(receipt)
        self.save(state)
        return receipt

    def save(self, state: EsState, receipts: list[dict] | None = None) -> None:
        """Durable resumable snapshot (atomic replace) + the accepted head per update."""
        if self.state_path is None:
            return
        import hashlib
        import os
        path = Path(self.state_path)
        payload = dict(state=state.to_dict(), receipts=self.receipts if receipts is None
                       else receipts, budget=None if self.ledger is None else self.ledger.to_dict(),
                       sensitivity=self.sensitivity,
                       world_hash={str(k): v for k, v in self.world_hash.items()})
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        values = [float(v) for v in state.theta.tolist()]
        (path.parent / f"theta_after_u{state.update_count:02d}.json").write_text(
            json.dumps(dict(updates_done=state.update_count, theta=values,
                            sha256=hashlib.sha256(json.dumps(values).encode()).hexdigest()))
            + "\n", encoding="utf-8")

    @classmethod
    def resume(cls, path: Path, **kwargs) -> tuple["SumoEsDriver", EsState]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        driver = cls(state_path=Path(path), **kwargs)
        driver.receipts = list(payload["receipts"])
        budget = payload["budget"]
        driver.ledger = BudgetLedger(budget["cap_sim_s"], driver.config.episode_charge_sim_s,
                                     budget["charged_sim_s"], budget["actual_sim_s"],
                                     budget["jobs"])
        driver.sensitivity = payload["sensitivity"]
        driver.world_hash = {int(k): v for k, v in payload["world_hash"].items()}
        return driver, EsState.from_dict(payload["state"])

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
            self.save(state)
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
        raise SystemExit("real SUMO runs only via tools/sumoes/remote/run.py on cpuQ "
                         "(registered S-SUMOES-RUN, design 05b84910)")
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
