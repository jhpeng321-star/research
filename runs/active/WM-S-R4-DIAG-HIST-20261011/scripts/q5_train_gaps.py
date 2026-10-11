"""Supplement: interior-gap rate in TRAIN pack histories (same rule as q5 check)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_lib2 import DATA, OUT, dump_json, ensure_env, q, train_window_iter

ensure_env()
import numpy as np  # noqa: E402

rows = []
for win in train_window_iter(150, seed=2):
    m = win["feat_mask"]  # [K,61] post-featurisation validity
    # raw presence gaps inside each column's observed span, from the pack arrays
    rows.append({"world": win["world"]})
    if len(rows) >= 150:
        break

from wms.data.corpus import PackCorpus
corpus = PackCorpus.open(DATA / "pack_damping",
                         extra_pack_dirs=[DATA / "pack_dspeed", DATA / "pack_v3"])
vm = np.asarray
rates = []
gap_states = 0
n_states = 0
import torch
torch.manual_seed(2)
# reuse the same tick sampling as train_window_iter for gap accounting
gen = train_window_iter(150, seed=2)
for win in gen:
    pass  # consume; we need raw veh_mask slices -> re-open corpus arrays directly
veh_mask = corpus.array("veh_mask")
torch.manual_seed(2)
gen2 = train_window_iter(150, seed=2)
ticks = []
for win in gen2:
    ticks.append(win["tick"])
for tick in ticks:
    m = np.asarray(veh_mask[max(0, tick - 120):tick + 1], bool)
    n_states += 1
    gaps = 0
    for j in range(m.shape[1]):
        idx = np.flatnonzero(m[:, j])
        if idx.size < 2:
            continue
        gaps += int((~m[idx[0]:idx[-1] + 1, j]).sum())
    rates.append(gaps / (m.shape[0] * m.shape[1]))
dump_json({"n_windows": n_states, "interior_gap_cell_share": q(rates),
           "windows_with_any_gap": float(np.mean([r > 0 for r in rates]))},
          OUT / "q5_train_gaps.json")
print("train interior-gap cell share:", q(rates), "any-gap windows:",
      float(np.mean([r > 0 for r in rates])))
