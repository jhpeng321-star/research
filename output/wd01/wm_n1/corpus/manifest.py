"""Upload manifest with split refusals.

Builds the file list for the GPU-box upload and REFUSES any run whose split is
HOLDOUT, whose world is in ``protocols/holdout_exclusions.json``, or that
belongs to the sealed chunk-5 TEST split (seed block 8707-8736, any chunk
``chunk5``, or a world listed in ``protocols/chunk5_test_seal.json`` when
present).

The split of EVERY world is decided from its chunk ``WORLD_PLAN.json`` BEFORE
any packed data file is opened, so a sealed refusal never touches the payload.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from . import _util as U
from .featurize_all import sealed_reason
from .pack import INDEX_NAME, MANIFEST_NAME, SUMS_NAME

MANIFEST_SCHEMA = "wm_n1_upload_manifest/1"
KEEP_SPLITS = ("TRAIN", "CALIBRATION")


class ManifestRefusal(PermissionError):
    """A sealed/excluded/chunk-5 path reached the upload manifest."""


def _default_exclusions_path():
    try:
        import wdpaths

        rd = Path(wdpaths.RESEARCH_DENSE_ROOT)
    except Exception:  # noqa: BLE001
        rd = Path("E:/research-dense")
    for cand in (rd / "protocols" / "holdout_exclusions.json",
                 U.WORKSPACE_ROOT / "protocols" / "holdout_exclusions.json"):
        if cand.is_file():
            return cand
    return None


def _load_exclusions(exclusions_path, tools_dir):
    path = Path(exclusions_path) if exclusions_path \
        else _default_exclusions_path()
    data = {}
    if path is not None and Path(path).is_file():
        if tools_dir:
            guard = Path(tools_dir) / "holdout_guard.py"
            if guard.is_file():
                try:
                    spec = importlib.util.spec_from_file_location(
                        "corpus_holdout_guard", str(guard))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    data = mod.load_holdout_exclusions(str(path))
                except Exception:  # noqa: BLE001 - fall back to raw JSON
                    data = {}
        if not data:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
    excluded = {str(row.get("world")) for row in data.get("excluded", [])
                if row.get("world") is not None}
    return data, excluded


def _decide(index: dict, excluded, seal, plan_paths):
    from features import splits as S

    refusals = []
    splits: dict = {}
    seen_plans: dict = {}
    for r in index.get("runs", []):
        key = r.get("key")
        chunk = str(r.get("chunk"))
        world = str(r.get("world"))
        reason = sealed_reason(chunk, world, seal)
        if reason:
            refusals.append(f"{key}: {reason}")
            continue
        if world in excluded:
            refusals.append(
                f"{key}: world {world!r} in protocols/holdout_exclusions.json")
            continue
        plan_path = r.get("plan_path") or (plan_paths or {}).get(chunk)
        plan_path = Path(plan_path) if plan_path else None
        if plan_path is None or not plan_path.is_file():
            refusals.append(f"{key}: no WORLD_PLAN.json to decide the split")
            continue
        try:
            if plan_path in seen_plans:
                plan = seen_plans[plan_path]
                row = next(w for w in plan["worlds"] if w["world"] == world)
                split = row["split"]
                if split == "HOLDOUT" or split not in KEEP_SPLITS:
                    raise S.SplitRefusal(
                        f"split {split!r} is not {KEEP_SPLITS}")
            else:
                plan = S.load_plan(chunk, plan_path)
                seen_plans[plan_path] = plan
                split = S.decide_split(
                    chunk, world, plan_path=plan_path,
                    exclusions={"excluded": [{"world": w} for w in excluded]})
        except S.SplitRefusal as e:
            refusals.append(f"{key}: {e}")
            continue
        splits.setdefault(split, []).append(key)
    return refusals, splits


def _manifest_items(pack_dir: Path) -> list:
    p = pack_dir / MANIFEST_NAME
    if not p.is_file():
        return []
    return U.read_json(p).get("items", [])


def build_upload_manifest(pack_dir, *, exclusions_path=None, tools_dir=None,
                          plan_paths=None) -> dict:
    """Return ``{"files", "sha256", "refused", "splits", ...}``.

    Raises :class:`ManifestRefusal` if any run is HOLDOUT, excluded or chunk-5;
    the manifest never contains such a run, and no packed data file is opened
    before every split decision is complete.
    """
    pack_dir = Path(pack_dir)
    index_p = pack_dir / INDEX_NAME
    if not index_p.is_file():
        raise FileNotFoundError(f"no pack at {index_p}")
    index = U.read_json(index_p)

    _, excluded = _load_exclusions(exclusions_path, tools_dir)
    seal = _load_chunk5_seal()

    U.OpenedAudit.reset()
    U.OpenedAudit.start()
    try:
        refusals, splits = _decide(index, excluded, seal, plan_paths)
        # The train manifest is the payload contract; check its items too,
        # still before any array bytes are hashed.
        for it in _manifest_items(pack_dir):
            world = str(it.get("world"))
            reason = sealed_reason(str(it.get("chunk")), world, seal)
            if reason:
                refusals.append(f"item {world}/{it.get('arm')}: {reason}")
            if world in excluded:
                refusals.append(f"item {world}/{it.get('arm')}: excluded world")
            if str(it.get("split", "")).upper() not in KEEP_SPLITS:
                refusals.append(
                    f"item {world}/{it.get('arm')}: split "
                    f"{it.get('split')!r} not in {KEEP_SPLITS}")
        if refusals:
            raise ManifestRefusal(
                "upload manifest refused sealed/excluded runs: "
                + "; ".join(refusals))

        files = sorted(set(index.get("files", {}))
                       | {INDEX_NAME, MANIFEST_NAME})
        if (pack_dir / SUMS_NAME).is_file():
            files = sorted(set(files) | {SUMS_NAME})
        sha256 = {rel: U.sha256_file(pack_dir / rel) for rel in files}
    finally:
        U.OpenedAudit.stop()

    U.OpenedAudit.assert_clean()

    return dict(
        schema=MANIFEST_SCHEMA, pack_schema=index.get("schema"),
        pack_dir=str(pack_dir.resolve()), created_utc=U.utc_now(),
        n_runs=int(index.get("n_runs", len(index.get("runs", [])))),
        files=files, sha256=sha256, refused=[],
        splits={k: sorted(v) for k, v in sorted(splits.items())},
        opened_paths=len(U.OpenedAudit.paths),
    )


def _load_chunk5_seal():
    from .featurize_all import _load_chunk5_seal as _load

    return _load()


def write_manifest(manifest: dict, path) -> str:
    """Write the manifest JSON and return its sha256 (upload receipt)."""
    path = Path(path)
    body = json.dumps(manifest, indent=1, default=U._json_default)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return U.sha256_bytes(body.encode("utf-8"))
