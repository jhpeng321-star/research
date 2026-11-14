"""Verify the ``pack_v3`` subset with the real loader (CPU, metadata-only).

Runs :meth:`wms.train.TrajectoryDataset.from_real_packs` against
(a) the subset and (b) the full pack, using the pack_v3-only window lists
derived from the ``lpwo_v3`` train/val lists, and asserts the loader keeps the
same pack_v3 windows (294 train / 47 val).  The identity sidecar for the subset
is the remapped, loader-compatible copy built by ``build_subset.py`` because
the loader keys identities by pack-local manifest position.

The subset is exposed to the loader under the canonical basename ``pack_v3``
through a directory junction (``<work>/alias/pack_v3``) so
``receipt["kept_by_pack"]`` is keyed ``pack_v3`` exactly like the full pack.

Usage (PYTHONPATH=output/wd01/wm_n1, cwd repo root):

    python _out/r3_packv3_subset/verify_subset.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WM_N1 = REPO / "output" / "wd01" / "wm_n1"
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

DEFAULT_SUBSET = r"D:\rd-staging\pack_v3_r3_subset"
DEFAULT_FULL = r"D:\rd-staging\wmn1_feat\pack_v3"
DEFAULT_WORK = str(Path(__file__).resolve().parent)
DEFAULT_IDENTITIES = str(REPO / "_out" / "sidecars_packv3_v2" / "identities")
DEFAULT_TRAIN = r"E:\rd-wt\ws-r3-jam\_out\lpwo_v3\windows_train_noPI_noWall.json"
DEFAULT_VAL = r"E:\rd-wt\ws-r3-jam\_out\lpwo_v3\windows_val_noPI_noWall.json"
EXPECT_TRAIN = 294
EXPECT_VAL = 47


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pack_v3_keys(list_path):
    data = load_json(list_path)
    return {str(w["key"]) for w in data["windows"] if w["pack"] == "pack_v3"}


def make_canonical_alias(target: Path, work: Path) -> Path:
    """Expose ``target`` under a path whose basename is ``pack_v3``."""
    if target.name == "pack_v3":
        return target
    alias = work / "alias" / "pack_v3"
    if not os.path.lexists(alias):
        alias.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True, text=True, check=False)
        if rc.returncode != 0:
            print(f"[warn] junction creation failed: {rc.stderr.strip()}")
            return target
    return alias


def loader_receipt(pack: Path, identities: Path, window_list: Path):
    os.environ.pop("WM_N1_TRAIN_IDENTITIES", None)
    os.environ.pop("WM_N1_TRAIN_OUTCOMES", None)
    from wms.train import TrajectoryDataset

    ds = TrajectoryDataset.from_real_packs(
        [str(pack)], identity_dirs=[str(identities)],
        horizon_steps=240, window_list=str(window_list))
    return ds.receipt


def check(label: str, pack: Path, identities: Path, window_list: Path,
          expected: int, expected_keys: set) -> bool:
    receipt = loader_receipt(pack, identities, window_list)
    kept = receipt.get("kept_by_pack") or {}
    total = sum(int(v) for v in kept.values())
    canonical = int(kept.get("pack_v3", total))
    wf = receipt.get("window_filter") or {}
    print(f"[{label}] kept_by_pack={json.dumps(kept, sort_keys=True)} "
          f"total={total} pack_v3={canonical}")
    print(f"[{label}] window_filter requested={wf.get('requested')} "
          f"matched={wf.get('matched')} kept={wf.get('kept')}")
    ok = canonical == expected and total == expected
    if wf and wf.get("requested") is not None:
        ok = ok and int(wf["requested"]) == expected \
            and int(wf["matched"]) == expected
    return ok


def subset_key_set(index_path: Path) -> set:
    index = load_json(index_path)
    return {str(r["key"]) for r in index["runs"]}


def strict_identity_ok(pack: Path, identities: Path) -> bool:
    """The subset's own sidecar must pass the strict identity loader too."""
    from train.identity_sidecar import load_identity_sidecar

    try:
        side = load_identity_sidecar(str(pack), str(identities))
    except Exception as exc:  # noqa: BLE001
        print(f"[strict-identity] FAILED: {exc}")
        return False
    print(f"[strict-identity] ok rows={len(side.vehicle_map or [])} "
          f"receipt={side.receipt_sha256[:12]}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--full-pack", default=DEFAULT_FULL)
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--identities-real", default=DEFAULT_IDENTITIES)
    ap.add_argument("--train-list", default=DEFAULT_TRAIN)
    ap.add_argument("--val-list", default=DEFAULT_VAL)
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args(argv)

    work = Path(args.work)
    subset = Path(args.subset)
    train_list = work / "windows_train_pack_v3.json"
    val_list = work / "windows_val_pack_v3.json"
    if not train_list.is_file():
        raise SystemExit(f"missing filtered train list: {train_list}")
    if not val_list.is_file():
        raise SystemExit(f"missing filtered val list: {val_list}")
    train_keys = pack_v3_keys(args.train_list)
    val_keys = pack_v3_keys(args.val_list)

    alias = make_canonical_alias(subset, work)
    print(f"subset alias: {alias}")

    # exact key-set invariant (subset INDEX run keys are a superset)
    sk = subset_key_set(subset / "INDEX.json")
    assert train_keys <= sk, "subset INDEX lacks train keys"
    assert val_keys <= sk, "subset INDEX lacks val keys"

    strict_ok = strict_identity_ok(subset, subset / "identities")

    ok = True
    ok &= strict_ok
    ok &= check("subset/train", alias, subset / "identities", train_list,
                EXPECT_TRAIN, train_keys)
    ok &= check("subset/val", alias, subset / "identities", val_list,
                EXPECT_VAL, val_keys)

    if not args.skip_full:
        full = Path(args.full_pack)
        identities = Path(args.identities_real)
        ok &= check("full/train", full, identities, train_list,
                    EXPECT_TRAIN, train_keys)
        ok &= check("full/val", full, identities, val_list,
                    EXPECT_VAL, val_keys)

    if not ok:
        print("VERIFY FAILED")
        return 1
    print("VERIFY OK: subset and loader keep identical pack_v3 windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
