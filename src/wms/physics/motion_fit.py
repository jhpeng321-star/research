"""Opt-in readers/draws for observable-motion fitted physical parameter JSON.

Nothing imports this from legacy physics. Draw ONCE at rollout initialization
and cache by identity in the caller; never redraw physical settings each tick.
SciPy is required only by offline fitters, not by this runtime reader.
"""
from __future__ import annotations

import json
import gzip
import hashlib
import math
from pathlib import Path

import torch

IDM_ORDER = ("v0", "time_headway", "accel", "decel", "min_gap")
LC_ORDER = ("lcStrategic", "lcCooperative", "lcSpeedGain", "lcKeepRight", "lcAssertive")
ENCODER_ORDER = ("v0", "time_headway", "min_gap", "accel", "decel", "politeness",
                 "mobil_threshold", "merge_gap", "merge_urgency")


class MotionFit:
    """Load explicitly selected artifacts; population fallback for unknown VID.

    Present-driver posteriors are causal ONLY at their conditioning_end_s (300
    in the delivered artifact). For other start times use population draws or
    refit a posterior from the corresponding causal observable history.
    """

    def __init__(self, directory: str | Path, *, shrinkage_path: str | Path | None = None):
        root = Path(directory)
        population_bytes = (root / "idm_population.json").read_bytes()
        self.population = json.loads(population_bytes)
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
        self.shrinkage = None
        if shrinkage_path is not None:
            if self.posterior.get('shrinkage_applied'):
                raise ValueError('Shrinkage already applied in exported posterior')
            policy = json.loads(Path(shrinkage_path).read_text())
            if (policy.get("schema") != "wm_phys.idm_shrinkage.v1"
                    or policy.get("idm_population_sha256") != hashlib.sha256(population_bytes).hexdigest()
                    or policy.get("conditioning_end_s") != self.posterior["conditioning_end_s"]
                    or self.population["distribution"] != "finite_support_mixture"):
                raise ValueError("Shrinkage policy/population binding mismatch")
            strength = policy.get("prior_count")
            if not policy.get("population_only") and (not isinstance(strength, (float, int))
                                                       or not math.isfinite(strength) or strength < 0):
                raise ValueError("Invalid learned shrinkage prior count")
            self.shrinkage = policy

    def _finite_weights(self, distribution, device):
        support = torch.tensor(self.population["support"], dtype=torch.float64, device=device)
        prior = torch.tensor(self.population["weights"], dtype=torch.float64, device=device)
        indices = torch.tensor(distribution.get("support_indices", list(range(len(support)))),
                               dtype=torch.long, device=device)
        weights = torch.tensor(distribution.get("weights", self.population["weights"]),
                               dtype=torch.float64, device=device)
        if (support.shape != (len(prior), 5) or not torch.isfinite(support).all()
                or not (support > 0).all() or indices.ndim != 1 or weights.shape != indices.shape
                or (indices < 0).any() or (indices >= len(support)).any()
                or len(indices.unique()) != len(indices) or not torch.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0
                or not torch.isfinite(prior).all() or (prior < 0).any() or prior.sum() <= 0):
            raise ValueError("Invalid finite physical-fit support/weights")
        dense = torch.zeros_like(prior)
        dense[indices] = weights / weights.sum()
        if self.shrinkage is not None and distribution is not self.population:
            n = distribution.get("n_observations", 0)
            if not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
                raise ValueError("Invalid posterior observation count")
            strength = self.shrinkage["prior_count"]
            r = 0. if self.shrinkage["population_only"] or n == 0 else n / (n+strength)
            dense = r*dense + (1-r)*prior/prior.sum()
        return support, dense

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
            if self.population["distribution"] == "finite_support_mixture":
                support, weights = self._finite_weights(distribution, mu.device)
                index = int(torch.multinomial(weights, 1, generator=generator)) if sample else int(weights.argmax())
                mu = support[index].log()
            elif sample:
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

    def unc_posterior_samples(self, worlds, vehicle_ids: torch.Tensor, present: torch.Tensor,
                              is_cav: torch.Tensor, point_params: dict[str, torch.Tensor], *,
                              n_samples: int, t0_s: float, seed: int = 0,
                              use_posterior: bool = True) -> torch.Tensor:
        """UNC encoded['posterior_samples'] in EXACT encoder [P,B,K,9] order.

        Replace five IDM coordinates for present HDVs only. Keep all CAV/padding
        and four unrelated encoder coordinates unchanged. Hash by world/VID/draw
        so packed-slot order cannot change latent settings. Supplied samples are
        experimental unless the caller has qualified their predictive coverage.
        """
        if (vehicle_ids.ndim != 2 or present.shape != vehicle_ids.shape or is_cav.shape != vehicle_ids.shape
                or len(worlds) != vehicle_ids.shape[0] or isinstance(n_samples, bool)
                or not isinstance(n_samples, int) or n_samples < 1
                or set(point_params) != set(ENCODER_ORDER)
                or any(p.shape != vehicle_ids.shape for p in point_params.values())):
            raise ValueError("Invalid UNC sample dimensions/parameter keys")
        if use_posterior and t0_s != self.posterior["conditioning_end_s"]:
            raise ValueError("Posterior conditioning time mismatch")
        point = torch.stack([point_params[name] for name in ENCODER_ORDER], -1)
        if not torch.isfinite(point).all():
            raise ValueError("Nonfinite UNC point settings")
        result = point.unsqueeze(0).expand(n_samples, *point.shape).clone()
        selected = present.bool() & ~is_cav.bool()
        for sample in range(n_samples):
            cache = {}
            for row, slot in selected.nonzero().tolist():
                world, vid = worlds[row], vehicle_ids[row, slot].item()
                if int(world) != world or int(vid) != vid or vid < 0:
                    raise ValueError("World and present VID must be observable nonnegative integers")
                key = (int(world), int(vid))
                if key not in cache:
                    digest = hashlib.sha256(f'rbfit:{seed}:{sample}:{key[0]}:{key[1]}'.encode()).digest()
                    rng = torch.Generator(device=vehicle_ids.device).manual_seed(
                        int.from_bytes(digest[:8], 'big') % (2**63-1))
                    cache[key] = self.idm(key[0], vehicle_ids[row, slot], t0_s=t0_s, sample=True,
                                          use_posterior=use_posterior, generator=rng)
                for name in IDM_ORDER:
                    result[sample, row, slot, ENCODER_ORDER.index(name)] = cache[key][name].to(point)
        return result


class MotionFitForUNC:
    """Read-only encoder proxy accepted directly by plan_hidden_state_futures.

    The rollout still uses the original model. No model attribute is mutated.
    Pass authentic IDs in padded slot order, and use the SAME proxy/seed for
    alternative controllers. Encoder embeddings and point settings are retained.
    """

    def __init__(self, model, fit: MotionFit, worlds, vehicle_ids: torch.Tensor, *,
                 t0_s: float, n_particles: int = 32, seed: int = 0, use_posterior: bool = True):
        self.model, self.fit = model, fit
        self.worlds, self.vehicle_ids = list(worlds), vehicle_ids.detach().clone()
        self.t0_s, self.n_particles, self.seed = t0_s, n_particles, seed
        self.use_posterior = use_posterior

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode_drivers(self, state, **kwargs):
        encoded, embedding, params = self.model.encode_drivers(state, **kwargs)
        samples = self.fit.unc_posterior_samples(self.worlds, self.vehicle_ids,
            state['present'], state['is_cav'], params, n_samples=self.n_particles,
            t0_s=self.t0_s, seed=self.seed, use_posterior=self.use_posterior)
        return dict(encoded or {}, posterior_samples=samples), embedding, params


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
