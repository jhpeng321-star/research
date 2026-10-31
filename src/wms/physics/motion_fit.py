"""Opt-in readers/draws for observable-motion fitted physical parameter JSON.

Nothing imports this from legacy physics. Draw ONCE at rollout initialization
and cache by identity in the caller; never redraw physical settings each tick.
SciPy is required only by offline fitters, not by this runtime reader.
"""
from __future__ import annotations

import json
import gzip
from pathlib import Path

import torch

IDM_ORDER = ("v0", "time_headway", "accel", "decel", "min_gap")
LC_ORDER = ("lcStrategic", "lcCooperative", "lcSpeedGain", "lcKeepRight", "lcAssertive")


class MotionFit:
    """Load explicitly selected artifacts; population fallback for unknown VID.

    Present-driver posteriors are causal ONLY at their conditioning_end_s (300
    in the delivered artifact). For other start times use population draws or
    refit a posterior from the corresponding causal observable history.
    """

    def __init__(self, directory: str | Path):
        root = Path(directory)
        self.population = json.loads((root / "idm_population.json").read_text())
        raw_posterior = root / "idm_posteriors.json"
        if raw_posterior.exists():
            self.posterior = json.loads(raw_posterior.read_text())
        else:
            with gzip.open(root / "idm_posteriors.json.gz", "rt", encoding="utf-8") as stream:
                self.posterior = json.load(stream)
        if (self.population["schema"] != "wm_phys.idm_population.v1"
                or self.posterior["schema"] != "wm_phys.idm_posteriors.v1"
                or tuple(self.population["parameter_order"]) != IDM_ORDER
                or tuple(self.posterior["parameter_order"]) != IDM_ORDER):
            raise ValueError("Unsupported physical-fit IDM schema/order")

    def idm(self, world: int, vehicle_ids: torch.Tensor, *, t0_s: float,
            sample: bool = False, use_posterior: bool = True,
            generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Return settings with the same shape/device as the observable IDs.

        Repeated IDs within a call share one draw. World and numeric VID are
        observed identity, never a latent driver class or a vType label.
        Unknown IDs and explicitly population-only calls use learned population.
        """
        if use_posterior and t0_s != self.posterior["conditioning_end_s"]:
            raise ValueError("Posterior conditioning time mismatch; use population or causal refit")
        flat = vehicle_ids.detach().cpu().flatten().tolist()
        draws = {}
        for vid in dict.fromkeys(flat):
            if int(vid) != vid or vid < 0:
                raise ValueError("Only present, nonnegative observable vehicle IDs are accepted")
            distribution = (self.posterior["drivers"].get(f"{int(world)}:{int(vid)}", self.population)
                            if use_posterior else self.population)
            mu = torch.tensor(distribution["log_mean"], dtype=torch.float64, device=vehicle_ids.device)
            cov = torch.tensor(distribution["log_cov"], dtype=torch.float64, device=vehicle_ids.device)
            if mu.shape != (5,) or cov.shape != (5,5) or not torch.isfinite(mu).all() or not torch.isfinite(cov).all():
                raise ValueError("Invalid physical-fit moments")
            if sample:
                if self.population["distribution"] == "finite_support_mixture":
                    support = torch.tensor(self.population["support"],dtype=mu.dtype,device=mu.device)
                    indices = distribution.get("support_indices", list(range(len(support))))
                    weights = torch.tensor(distribution.get("weights",self.population["weights"]),
                                           dtype=mu.dtype,device=mu.device)
                    index = int(torch.multinomial(weights,1,generator=generator))
                    mu = support[indices[index]].log()
                else:
                    # Gauss-Newton covariance is SPD; no behavior clipping/floors.
                    root = torch.linalg.cholesky((cov + cov.T) / 2)
                    mu = mu + root @ torch.randn(5, dtype=mu.dtype, device=mu.device, generator=generator)
            elif "modal_parameters" in distribution:
                mu = torch.tensor([distribution["modal_parameters"][key] for key in IDM_ORDER],
                                   dtype=mu.dtype,device=mu.device).log()
            draws[vid] = mu.exp()
        values = (torch.stack([draws[vid] for vid in flat]).reshape(*vehicle_ids.shape, 5)
                  if flat else torch.empty((*vehicle_ids.shape, 5), dtype=torch.float64,
                                           device=vehicle_ids.device))
        result = {name: values[..., i] for i, name in enumerate(IDM_ORDER)}
        result["delta"] = torch.full_like(values[..., 0], float(self.population["delta"]))
        return result


def draw_lane_parameters(population_path: str | Path, posterior_path: str | Path,
                         world: int, observable_vid: int, *, t0_s: float,
                         generator: torch.Generator | None = None,
                         use_posterior: bool = True) -> dict[str, float]:
    """One persistent LC2013 vector; emission discrepancy is NOT runtime noise."""
    pop = json.loads(Path(population_path).read_text())
    post = json.loads(Path(posterior_path).read_text())
    if (pop["schema"] != "wm_phys.lc2013_population.v1"
            or post["schema"] != "wm_phys.lc2013_posteriors.v1"
            or tuple(pop["parameter_order"]) != LC_ORDER):
        raise ValueError("Unsupported LC2013 fit schema/order")
    if use_posterior and t0_s != post["conditioning_end_s"]:
        raise ValueError("LC posterior conditioning time mismatch")
    candidate = post["drivers"].get(f"{world}:{observable_vid}", {}) if use_posterior else {}
    probabilities = torch.tensor(candidate.get("weights", pop["weights"]), dtype=torch.float64)
    support = torch.tensor(pop["support"], dtype=torch.float64)
    if (support.shape != (len(probabilities), len(LC_ORDER))
            or not torch.isfinite(support).all() or not (support > 0).all()
            or not torch.isfinite(probabilities).all() or not (probabilities >= 0).all()
            or not probabilities.sum() > 0):
        raise ValueError("Invalid LC physical-fit distribution")
    index = int(torch.multinomial(probabilities, 1, generator=generator))
    return dict(zip(LC_ORDER, support[index].tolist()))
