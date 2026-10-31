"""Explicit LC-boundary reader for the existing UNC persistent future sampler.

No legacy imports or model switches are changed. Call only when deliberately
testing a fitted LC candidate. IDM validation remains the original UNC reader's
strict-positive contract; LC's documented zero behavioral weights are legal.
"""
from __future__ import annotations

from pathlib import Path


def load_boundary_fitted_state(idm_directory: str | Path, *, lane_directory: str | Path):
    """Return the real UNC reader with an LC-only physical-domain extension."""
    import torch
    from wms.control.fitted_hidden import FittedHiddenState, LC_ORDER

    class BoundaryFittedHiddenState(FittedHiddenState):
        @staticmethod
        def _distribution(pop, distribution, order):
            if tuple(order) != LC_ORDER:
                return FittedHiddenState._distribution(pop,distribution,order)
            support = torch.as_tensor(pop["support"],dtype=torch.float64)
            if (support.ndim!=2 or support.shape[1]!=5 or not torch.isfinite(support).all()
                    or (support[:,:4]<0).any() or (support[:,1]>1).any() or (support[:,4]<=0).any()):
                raise ValueError("Invalid source-domain fitted LC support")
            indices = torch.as_tensor(distribution.get("support_indices",list(range(len(support)))))
            weights = torch.as_tensor(distribution["weights"],dtype=torch.float64)
            if (indices.ndim!=1 or indices.dtype not in (torch.int32,torch.int64)
                    or (indices<0).any() or (indices>=len(support)).any()
                    or weights.shape!=indices.shape or not torch.isfinite(weights).all()
                    or (weights<0).any() or weights.sum()<=0):
                raise ValueError("Invalid fitted LC support indices/weights")
            return support[indices],weights

    reader = BoundaryFittedHiddenState.from_directory(idm_directory,lane_directory=lane_directory)
    reader.provenance["lc_source_domain_reader"] = "rbfit_explicit_boundary_adapter.v1"
    return reader
