"""TRAIN-scoped validation holdout for WM-S2 held-out monitoring.

The hetpack VAL/TEST window lists are scoped to ``pack_hetdata_val`` /
``pack_hetdata_test`` and the trainer is TRAIN-only, so they cannot drive
``--val-window-list`` on the TRAIN corpus (``lpo.filter_positions`` fails closed
on unmatched keys).  This module instead derives a held-out validation subset
FROM the TRAIN window list: ~10% of TRAIN worlds are held out of training and
monitored by ``--val-window-list``.

Both output lists keep the original ``pack``/``batch`` fields, so their
structured keys match the TRAIN corpus.  The split is a deterministic function
of ``(world order, seed)`` and reads metadata only; it never opens an array,
outcome or sealed source.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

WINDOW_SCHEMA = "wm_s_lpo_windows/1"
DEFAULT_FRAC = 0.1


def holdout_worlds(doc: dict, *, frac: float = DEFAULT_FRAC, seed: int = 0) -> list:
    """Deterministic held-out world list (~``frac`` of the document's worlds)."""
    if not 0.0 < float(frac) < 1.0:
        raise ValueError("frac must be in (0, 1)")
    worlds = sorted({str(w["world"]) for w in doc.get("windows", [])})
    if len(worlds) < 2:
        raise ValueError("need at least two worlds to hold one out")
    rng = random.Random(int(seed))
    shuffled = list(worlds)
    rng.shuffle(shuffled)
    n_hold = max(1, int(round(float(frac) * len(worlds))))
    n_hold = min(n_hold, len(worlds) - 1)
    return sorted(shuffled[:n_hold])


def split_document(doc: dict, *, frac: float = DEFAULT_FRAC, seed: int = 0
                   ) -> tuple[dict, dict]:
    """Split a window document into ``(train_doc, val_doc)`` by world."""
    if str(doc.get("schema", WINDOW_SCHEMA)) != WINDOW_SCHEMA:
        raise ValueError(f"unexpected window-list schema {doc.get('schema')!r}")
    windows = doc.get("windows")
    if not isinstance(windows, list) or "n_windows" not in doc:
        raise ValueError("window document has no windows/n_windows")
    if int(doc["n_windows"]) != len(windows):
        raise ValueError("window document n_windows does not match its windows")
    held = set(holdout_worlds(doc, frac=frac, seed=seed))
    train = [w for w in windows if str(w["world"]) not in held]
    val = [w for w in windows if str(w["world"]) in held]
    if not train:
        raise ValueError("holding out these worlds would leave no TRAIN windows")
    if not val:
        raise ValueError("held-out worlds contributed no windows")

    def _doc(subset, split):
        out = {k: v for k, v in doc.items() if k != "windows"}
        out["windows"] = subset
        out["n_windows"] = len(subset)
        out["split"] = split
        out["holdout_frac"] = float(frac)
        out["holdout_seed"] = int(seed)
        out["holdout_worlds"] = sorted(held)
        return out

    return _doc(train, "TRAIN_HOLDOUT_TRAIN"), _doc(val, "TRAIN_HOLDOUT_VAL")


def materialise_holdout(window_list: str | Path, out_dir: str | Path, *,
                        frac: float = DEFAULT_FRAC, seed: int = 0
                        ) -> tuple[str, str]:
    """Write the train/val holdout lists beside a run's output directory.

    Returns ``(train_list_path, val_list_path)`` (strings) so the CLI can pass
    them straight to the dataset loader.
    """
    src = Path(window_list)
    doc = json.loads(src.read_text(encoding="utf-8"))
    train_doc, val_doc = split_document(doc, frac=frac, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "windows_train_holdout.json"
    val_path = out_dir / "windows_val_holdout.json"
    train_path.write_text(json.dumps(train_doc, indent=1, sort_keys=True),
                          encoding="utf-8")
    val_path.write_text(json.dumps(val_doc, indent=1, sort_keys=True),
                        encoding="utf-8")
    return str(train_path), str(val_path)
