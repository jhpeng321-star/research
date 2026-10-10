"""Shared fixtures for the partB_v0 tests.

Canonical dataset (READ-ONLY) is used for the data-contract tests; the
resume / CLI / eval tests run on a tiny SYNTHETIC dataset built in a tmp
dir (same schema, ~5 worlds), so they are fast and never touch the
canonical tree.  WD_DEVICE is pinned to cpu; ZERO simulation, and no
CPU training beyond tiny smoke steps (owner rule).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
PART = HERE.parent                      # output/wd01/partB_v0
WD01 = PART.parent                      # output/wd01
if str(WD01) not in sys.path:
    sys.path.insert(0, str(WD01))
os.environ.setdefault("WD_DEVICE", "cpu")

# The box is often busy with an unrelated SUMO campaign: cap the torch
# thread pool for the test process (WD_TEST_THREADS overrides).  The
# bitwise-resume tests compare runs inside ONE process, so a fixed count
# is also determinism-friendly.
import torch  # noqa: E402

torch.set_num_threads(max(1, int(os.environ.get("WD_TEST_THREADS", "4"))))

import wdpaths  # noqa: E402

sys.path.insert(0, str(PART))           # `import partB_v0` as a package

import partB_v0  # noqa: E402
from partB_v0.data import SCHEMA, load_manifest  # noqa: E402

_env_data = os.environ.get("WD01_V0_DATA", "").strip()
CANONICAL = (Path(_env_data) if _env_data else
             Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" /
             "wd01_partB_v0data")

# DATA_CARD_V0.md counts (label rows), hardcoded from the card:
CARD_ROWS = {
    ("chunk1", "TRAIN", "base"): 16253,
    ("chunk1", "TRAIN", "native"): 9024,
    ("chunk1", "CALIBRATION", "base"): 2129,
    ("chunk1", "CALIBRATION", "native"): 1344,
    ("chunk2", "TRAIN", "base"): 15863,
    ("chunk2", "TRAIN", "native"): 9408,
    ("chunk2", "CALIBRATION", "base"): 2489,
    ("chunk2", "CALIBRATION", "native"): 1344,
}
CARD_TD = {
    ("chunk1", "TRAIN"): 34205,
    ("chunk1", "CALIBRATION"): 4775,
    ("chunk2", "TRAIN"): 34385,
    ("chunk2", "CALIBRATION"): 5135,
}
CARD_SOURCES = {
    ("chunk1", "TRAIN"): 96, ("chunk1", "CALIBRATION"): 14,
    ("chunk2", "TRAIN"): 98, ("chunk2", "CALIBRATION"): 14,
}
CARD_NATIVE_T570_TRAIN = {"chunk1": 141, "chunk2": 147}
CARD_MANIFEST_SHA = ("f7fd64116b7c3f0024c5168fe829eb58988a31816cb28e14e"
                     "bca30f366cc5dfc")

TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)


@pytest.fixture(scope="session")
def canonical_available() -> bool:
    return (CANONICAL / "MANIFEST.json").exists()


@pytest.fixture(scope="session")
def canonical_manifest():
    if not (CANONICAL / "MANIFEST.json").exists():
        pytest.skip("canonical V0 dataset not available")
    return load_manifest(CANONICAL, verify_files=True, log=lambda *_: None)


@pytest.fixture(scope="session")
def canonical_ds(canonical_manifest):
    from partB_v0.data import V0Dataset
    return V0Dataset(canonical_manifest, log=lambda *_: None)


# ------------------------------------------------- synthetic tiny data --

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_tiny_dataset(root: Path, *, n_train=3, n_cal=2, n_dec=14,
                      seed=7, y_fn=None, cav_raw=17, wall_raw=6) -> dict:
    """Write a schema-valid tiny V0 dataset (chunk1 only) and return the
    MANIFEST sha256.  Worlds have t0 = 240; base decision times over
    [120, t0+570], native over (t0, t0+570]; data_end = t0 + 600, so the
    tau=570 labels exist at t0 (base) and t0+10 (native) -- the CRN-pair
    layout the policy-contrast evaluation needs.

    cav_raw/wall_raw: token widths of the written root.  Defaults 17/6 =
    the PRE-v0.4 layout every existing dataset carries; the v0.4 action/
    wall inputs tests write 20/11 roots (nets are data-driven via
    model_dims).

    y_fn(tau, t, source) -> float overrides the delay label generator
    (default 30*tau + 5*t + native bump); used by the tau-conditioning
    acceptance test (y depends only on tau).
    """
    from partB_v0.data import POLICY_OF_SOURCE  # late import (path shim)

    rng = np.random.default_rng(seed)
    root = Path(root)
    t0 = 240.0
    data_end = t0 + 600.0
    worlds = {}
    for i in range(n_train + n_cal):
        split = "TRAIN" if i < n_train else "CALIBRATION"
        wname = f"w{i:04d}m2700p02t{int(t0)}"
        for source in ("base", "native"):
            lo = 120.0 if source == "base" else t0 + 10.0
            ts = np.arange(lo, t0 + 570.0 + 1e-9, 10.0)[:n_dec]
            tau_avail = data_end - ts
            n = len(ts)
            key = f"chunk1/{wname}__{source}"
            wdir = root / "worlds" / "chunk1"
            wdir.mkdir(parents=True, exist_ok=True)
            files = {}
            # features
            feats = dict(
                t=ts.astype(np.float64),
                tau_available=tau_avail.astype(np.float64),
                lane_bins=rng.normal(0, 0.1, (n, 3, 320, 3))
                .astype(np.float32),
                lane_mask=np.ones((n, 3, 320), np.float32),
                lane_geom=np.ones((3, 320), np.float32),
                cav_tokens=rng.normal(0, 0.1, (n, 64, cav_raw))
                .astype(np.float32),
                cav_mask=np.ones((n, 64), np.float32),
                wall_tokens=np.zeros((n, 4, wall_raw), np.float32),
                wall_mask=np.zeros((n, 4), np.float32),
                demand=rng.random((n, 21)).astype(np.float32),
                context=(rng.random((n, 8)) * 0.1).astype(np.float32),
                q_main=rng.random(n).astype(np.float32),
                q_ramp=rng.random(n).astype(np.float32),
                world=np.str_(wname), source=np.str_(source),
                world_t0=np.float64(t0), data_end=np.float64(data_end),
                chunk_id=np.str_("chunk1"),
                pre_fix=np.bool_(True), p40=np.bool_(False),
            )
            f_npz = wdir / f"{wname}__{source}_feat.npz"
            np.savez_compressed(f_npz, **feats)
            files[f"worlds/chunk1/{f_npz.name}"] = _sha(f_npz)
            # labels
            srng = np.random.default_rng(seed + hash(key) % 1000)
            lab_rows = []
            for ti, t in enumerate(ts):
                for tau in TAUS:
                    if tau > tau_avail[ti] + 1e-9:
                        continue
                    base_delay = 30.0 * tau + 5.0 * t
                    if y_fn is not None:
                        base_delay = float(y_fn(tau, t, source))
                    bump = 500.0 if source == "native" else 0.0
                    if y_fn is not None:
                        bump = 0.0
                    lab_rows.append(dict(
                        world=wname, source=source, chunk_id="chunk1",
                        split=split, demand_veh_h=2700, penetration=0.02,
                        p40=False, pre_fix=True, t0_s=t0,
                        policy=POLICY_OF_SOURCE[source], t=float(t),
                        tau=float(tau), tau_available=float(tau_avail[ti]),
                        cav_speed_mode=31,
                        total_delay_ref_s=base_delay + bump
                        + srng.normal(0, 10),
                        ttc_lt3_veh_s=abs(srng.normal(50, 5)),
                        hard_brake_s=abs(srng.normal(20, 2)),
                        collision_any=int(srng.random() < 0.05)))
            lab = pd.DataFrame(lab_rows)
            f_lab = wdir / f"{wname}__{source}_labels.parquet"
            lab.to_parquet(f_lab, index=False)
            files[f"worlds/chunk1/{f_lab.name}"] = _sha(f_lab)
            # TD tuples (t -> t+30 at commit 30 s)
            td_rows = []
            tset = {round(float(t), 6) for t in ts}
            lab_map = {(round(float(r.t), 6), float(r.tau)):
                       float(r.total_delay_ref_s) for r in lab.itertuples()}
            for t in ts:
                tn = round(float(t) + 30.0, 6)
                if tn not in tset:
                    continue
                d30 = lab_map.get((round(float(t), 6), 30.0), 100.0)
                for tau in TAUS:
                    if tau < 30.0 - 1e-9:
                        continue
                    complete = tau <= (data_end - t) + 1e-9
                    td_rows.append(dict(
                        world=wname, source=source, chunk_id="chunk1",
                        split=split, demand_veh_h=2700, penetration=0.02,
                        p40=False, pre_fix=True, t0_s=t0,
                        t=float(t), t_next=float(t) + 30.0, tau=float(tau),
                        tau_available=float(data_end - t), d30_s=d30,
                        label_mc_s=lab_map.get((round(float(t), 6), tau),
                                               np.nan)
                        if complete else np.nan,
                        label_mc_complete=int(complete),
                        cav_speed_mode=31))
            td = pd.DataFrame(td_rows)
            f_td = wdir / f"{wname}__{source}_td.parquet"
            td.to_parquet(f_td, index=False)
            files[f"worlds/chunk1/{f_td.name}"] = _sha(f_td)
            worlds[key] = dict(
                world=wname, source=source, chunk_id="chunk1",
                split=split, demand_veh_h=2700, penetration=0.02,
                p40=False, pre_fix=True, t0_s=t0,
                policy=POLICY_OF_SOURCE[source], cav_speed_mode=31,
                files=files,
                rows=dict(n_decision_times=int(n),
                          n_label_rows=int(len(lab)),
                          n_td_rows=int(len(td))))
    man = dict(schema=SCHEMA, zero_simulation=True, tau_s=list(TAUS),
               commit_window_s=30.0, chunks={"chunk1": {"chunk_id":
                                                        "chunk1"}},
               worlds=worlds)
    (root / "MANIFEST.json").write_text(json.dumps(man, indent=1),
                                        encoding="utf-8")
    return _sha(root / "MANIFEST.json")


@pytest.fixture()
def tiny_dataset(tmp_path):
    """(root, manifest_sha256) of a fresh synthetic dataset."""
    sha = make_tiny_dataset(tmp_path / "data")
    return tmp_path / "data", sha
