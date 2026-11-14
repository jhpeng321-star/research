"""Split guard: decide a world's split from its chunk WORLD_PLAN.json ONLY.

The split is read from the plan before any run file is opened.  REFUSES:

* HOLDOUT worlds (plan ``split == "HOLDOUT"``);
* worlds listed in ``protocols/holdout_exclusions.json`` (via
  ``tools/holdout_guard.py`` when importable, otherwise the JSON directly);
* any split that is neither TRAIN nor CALIBRATION (for chunk-5 this is the
  sealed TEST split and needs both ``--final-test`` and the LF-normalised
  sha256 of ``REGISTRATION_CHUNK5_TEST.md``).

TRAIN and CALIBRATION are allowed (WM-N1 featurization is CAL-for-WM-N1).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

_WD01 = Path(__file__).resolve().parents[2]
if str(_WD01) not in sys.path:
    sys.path.insert(0, str(_WD01))
try:
    import wdpaths  # noqa: E402
    RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
except Exception:                                    # pragma: no cover
    RD = Path(os.environ.get("RESEARCH_DENSE_ROOT", "E:/research-dense"))

KEEP_SPLITS = ("TRAIN", "CALIBRATION")

CHUNKS = {
    "chunk1": {
        "plan": RD / "output/campaign_stage1_chunk1/results_plan_wm44/WORLD_PLAN.json",
        "runs": RD / "runs/campaign_stage1_chunk1"},
    "chunk1_fix27": {
        "plan": RD / "output/campaign_stage1_v3/results_plan_fix27/WORLD_PLAN.json",
        "runs": RD / "runs/campaign_stage1_chunk1_fix27"},
    "chunk2": {
        "plan": RD / "output/campaign_stage1_v2/results_plan_chunk2/WORLD_PLAN.json",
        "runs": RD / "runs/campaign_stage1_chunk2"},
    "chunk3": {
        "plan": RD / "output/campaign_stage1_v3/results_plan_chunk3/WORLD_PLAN.json",
        "runs": RD / "runs/campaign_stage1_chunk3"},
    "chunk4": {
        "plan": RD / "output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json",
        "runs": RD / "runs/campaign_stage1_chunk4"},
    "chunk5": {"plan": None, "runs": None, "sealed": True},
}

REGISTRATION_WM_N1 = (Path(__file__).resolve().parents[1]
                      / "REGISTRATION_WM_N1.md")
# The chunk-5 gate binds the SEALED chunk-5 registration, not the WM-N1 one
# (REGISTRATION_WM_N1.md section 3 records the chunk-5 LF-sha before TEST).
REGISTRATION_CHUNK5 = (Path(__file__).resolve().parents[1]
                       / "REGISTRATION_CHUNK5_TEST.md")
# LF-normalised sha256 of REGISTRATION_CHUNK5_TEST.md, frozen in
# C5TEST_FREEZE.json (registration_lf_sha256).  Hashes are line-ending
# independent so a CRLF checkout and the LF-canonical text agree.
CHUNK5_REGISTRATION_LF_SHA256 = (
    "f8c7d760baedeb359c501cd90a3e2a599fb0ddfef390ea555cf304c38214b31e")


class SplitRefusal(PermissionError):
    """Access to a sealed/excluded world was requested."""


def sha256_file(path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def lf_sha256_file(path) -> str:
    """sha256 of the file's bytes with CRLF/CR normalised to LF (CRLF-safe)."""
    data = Path(path).read_bytes()
    return hashlib.sha256(
        data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def registration_sha(path: Path | str | None = None) -> str:
    """LF-normalised sha256 of the chunk-5 registration (the sealed TEST gate)."""
    return lf_sha256_file(path or REGISTRATION_CHUNK5)


def wm_n1_registration_sha(path: Path | str | None = None) -> str:
    """LF-normalised sha256 of REGISTRATION_WM_N1.md (freeze-sidecar value)."""
    return lf_sha256_file(path or REGISTRATION_WM_N1)


def load_plan(chunk: str, plan_path: Path | str | None = None) -> dict:
    path = Path(plan_path) if plan_path else CHUNKS[chunk]["plan"]
    if path is None:
        raise SplitRefusal(
            f"chunk {chunk!r} has no WORLD_PLAN.json; pass an explicit plan")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_exclusions():
    """Owner exclusions via the canonical tools/holdout_guard.py when
    importable, else the raw JSON."""
    tools = RD / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        import holdout_guard as HG
        return HG.load_holdout_exclusions()
    except Exception:
        path = RD / "protocols" / "holdout_exclusions.json"
        return json.loads(path.read_text(encoding="utf-8"))


def excluded_worlds(exclusions=None) -> set:
    data = exclusions if exclusions is not None else load_exclusions()
    return {row["world"] for row in data.get("excluded", [])}


def world_split(chunk: str, world: str, plan_path: Path | str | None = None):
    plan = load_plan(chunk, plan_path)
    for w in plan["worlds"]:
        if w["world"] == world:
            return w
    raise SplitRefusal(f"world {world!r} is not in the {chunk} WORLD_PLAN.json")


def decide_split(chunk: str, world: str, *, plan_path=None, exclusions=None,
                 final_test: bool = False, registration_sha256=None,
                 expected_registration_sha=None) -> str:
    """Resolve the split for one world, refusing sealed/excluded access.

    Order (before any run file is opened): chunk-5 gate, plan lookup, HOLDOUT
    refusal, exclusion refusal.
    """
    if CHUNKS.get(chunk, {}).get("sealed"):
        if not final_test:
            raise SplitRefusal(
                f"chunk {chunk!r} is the sealed TEST split; pass --final-test "
                "to proceed")
        if registration_sha256 is None:
            raise SplitRefusal(
                "chunk-5 access needs --registration-sha (LF-normalised "
                "sha256 of REGISTRATION_CHUNK5_TEST.md)")
        want = (expected_registration_sha or CHUNK5_REGISTRATION_LF_SHA256)
        if registration_sha256 != want:
            raise SplitRefusal(
                f"chunk-5 registration sha mismatch: got "
                f"{registration_sha256!r}, expected {want!r}")
        if expected_registration_sha is None and REGISTRATION_CHUNK5.is_file():
            drift = registration_sha()
            if drift != CHUNK5_REGISTRATION_LF_SHA256:
                raise SplitRefusal(
                    "REGISTRATION_CHUNK5_TEST.md drifted from the frozen "
                    f"LF-sha256: got {drift!r}, frozen "
                    f"{CHUNK5_REGISTRATION_LF_SHA256!r}")
        return "TEST"

    row = world_split(chunk, world, plan_path)
    split = row["split"]
    if split == "HOLDOUT":
        raise SplitRefusal(
            f"world {world!r} is HOLDOUT (chunk {chunk}); refused before any "
            "run file was opened")
    if split not in KEEP_SPLITS:
        raise SplitRefusal(
            f"world {world!r} split {split!r} is not {KEEP_SPLITS}")
    if world in excluded_worlds(exclusions):
        raise SplitRefusal(
            f"world {world!r} is in protocols/holdout_exclusions.json; "
            "refused (owner ruling)")
    return split


def world_t0(chunk: str, world: str, plan_path=None) -> float:
    return float(world_split(chunk, world, plan_path)["t0_s"])


def run_dir(chunk: str, world: str, arm: str, runs_root=None) -> Path:
    root = Path(runs_root) if runs_root else CHUNKS[chunk]["runs"]
    if root is None:
        raise SplitRefusal(f"chunk {chunk!r} has no runs root; pass one")
    return root / f"{world}__{arm}"


def chunk_worlds(chunk: str, plan_path=None):
    """(kept TRAIN/CAL worlds, HOLDOUT worlds) -- plan only, no run files."""
    plan = load_plan(chunk, plan_path)
    keep = sorted((w["world"] for w in plan["worlds"]
                   if w["split"] in KEEP_SPLITS))
    hold = sorted(w["world"] for w in plan["worlds"]
                  if w["split"] not in KEEP_SPLITS)
    return keep, hold


class OpenAudit:
    """sys.addaudithook recorder of every opened path (like v04a_sources)."""

    _installed = False
    paths: list = []
    active = False

    @classmethod
    def start(cls):
        if not cls._installed:
            def hook(event, args):
                if cls.active and event == "open":
                    try:
                        cls.paths.append(str(args[0]))
                    except Exception:            # noqa: BLE001
                        pass
            sys.addaudithook(hook)
            cls._installed = True
        cls.active = True

    @classmethod
    def stop(cls):
        cls.active = False

    @classmethod
    def reset(cls):
        cls.paths = []

    @classmethod
    def opened_names(cls):
        return [Path(p).name for p in cls.paths]
