"""INC1 fresh-seed audit (READ-ONLY; registration section 2).

Checks the first-choice block 9601-9620 (and, mechanically, the next free
block of 20 at or above 9601) against:

* every integer/range constant exported by ``traffic_shepherd.seeds``
  (range membership, not just text);
* literal ``<seed>`` / ``w<seed>`` tokens in ``traffic_shepherd/seeds.py``;
* the workspace ``protocols/experiment_history.json``;
* the LIVE read-only registries ``E:/research-dense/protocols/
  {holdout_exclusions,chunk5_test_seal,experiment_history}.json``;
* every ``WORLD_PLAN.json`` and ``REGISTRATION*.md`` under the workspace
  ``output/`` and under the live ``E:/research-dense/output`` (text search
  only; no data file is opened).

A *collision* is a seed-identity use.  Two classes are deliberately NOT
collisions:

* decimal-float coincidences (e.g. a service delta ``13.9618``) are
  reported separately as ``incidental_numeric``;
* a 4-digit run inside a long hex hash (``[0-9a-fA-F]{32,}``, e.g. a
  ``sha256`` substring) is masked before matching -- a hash substring is
  never a world/seed identity use.

The INC1 registration and the INC1 ``WORLD_PLAN.json`` (this audit's own
downstream artifact) are excluded from the surfaces: counting the document
that NAMES the block, or the plan built FROM this audit, would make the
audit vacuous.  Writes ``INC1_SEED_AUDIT.json``; never reads an outcome
column and never opens HOLDOUT/CAL/chunk-5 data.

CLI (cwd ``output/wd01/inc1`` or repo root):

    python -m inc1.seeds_audit            # needs the output/wd01 on sys.path
    python output/wd01/inc1/seeds_audit.py
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from . import constants as C
except ImportError:  # script execution
    import constants as C  # type: ignore

#: the live read-only research-dense tree the brief binds (overridable for
#: a relocated checkout; default is the owner's path).
EXTERNAL_ROOT = Path(os.environ.get(
    "INC1_EXTERNAL_RESEARCH_ROOT", "E:/research-dense"))
EXTERNAL_PROTOCOLS = EXTERNAL_ROOT / "protocols"
EXTERNAL_OUTPUT = EXTERNAL_ROOT / "output"
#: a run of >=32 hex chars is a hash, not a seed identity token.
HEX_RUN_PATTERN = re.compile(r"[0-9a-fA-F]{32,}")


# ------------------------------------------------------------- seed membership

def seed_module_membership(seeds_module=None) -> dict:
    """{constant_name: container} for int-bearing constants in seeds.py."""
    if seeds_module is None:
        seeds_module = importlib.import_module("traffic_shepherd.seeds")
    out = {}
    for name in dir(seeds_module):
        if name.startswith("_"):
            continue
        value = getattr(seeds_module, name)
        if isinstance(value, (range, tuple, set, frozenset)):
            try:
                if value and all(isinstance(x, int) and not isinstance(x, bool)
                                 for x in value):
                    out[name] = value
            except TypeError:
                continue
    return out


def membership_hits(seed: int, containers: dict) -> list[str]:
    return sorted(name for name, value in containers.items() if seed in value)


# ------------------------------------------------------------- surface scan

def default_surfaces() -> list[dict]:
    """(label, path) for every audit surface that exists on disk.

    The INC1 registration and the INC1 ``WORLD_PLAN.json`` are excluded: the
    registration NAMES the first-choice block (``The first choice is
    9601-9620``) and the plan is built FROM this audit, so counting either
    would make the audit vacuous.  Every OTHER ``REGISTRATION*.md`` /
    ``WORLD_PLAN.json`` under the workspace ``output/`` and under the live
    ``E:/research-dense/output`` is scanned (text only).
    """
    inc1_registration = C.REGISTRATION_PATH.resolve()
    inc1_plan = C.PLAN_PATH.resolve()
    live_holdout = EXTERNAL_PROTOCOLS / "holdout_exclusions.json"
    surfaces = [
        {"label": "seeds.py", "path": C.WORKSPACE_ROOT / "traffic_shepherd" / "seeds.py"},
        {"label": "experiment_history.json",
         "path": C.WORKSPACE_ROOT / "protocols" / "experiment_history.json"},
        {"label": "holdout_exclusions.json",
         "path": (live_holdout if live_holdout.is_file()
                  else C.WORKSPACE_ROOT / "protocols" / "holdout_exclusions.json")},
        {"label": "chunk5_test_seal.json",
         "path": EXTERNAL_PROTOCOLS / "chunk5_test_seal.json"},
        {"label": "experiment_history.json(research-dense live)",
         "path": EXTERNAL_PROTOCOLS / "experiment_history.json"},
    ]
    for root, prefix in ((C.OUTPUT_ROOT, ""), (EXTERNAL_OUTPUT, "EXTERNAL_")):
        for path in sorted(root.rglob("WORLD_PLAN.json")):
            if path.resolve() == inc1_plan:
                continue
            surfaces.append({"label": f"{prefix}WORLD_PLAN:{path.as_posix()}",
                             "path": path})
        for path in sorted(root.rglob("REGISTRATION*.md")):
            if path.resolve() == inc1_registration:
                continue
            surfaces.append({"label": f"{prefix}REGISTRATION:{path.as_posix()}",
                             "path": path})
    return surfaces


def scan_surfaces(seeds, surfaces) -> tuple[dict, dict, int]:
    """Per-seed token hits across every surface (read-only text search).

    Returns ``(strict_hits, incidental_hits, masked_hex_runs)``.  ``strict``
    is the seed-identity token (``<seed>``/``w<seed>`` not inside a decimal
    or a long hex hash); ``incidental`` records loose integer matches that
    are part of a decimal literal.  Long hex runs (hashes) are masked before
    matching and counted in ``masked_hex_runs``.
    """
    strict_patterns = {seed: C.seed_token_pattern(seed) for seed in seeds}
    loose_patterns = {seed: re.compile(r"(?<![0-9])%d(?![0-9])" % seed)
                      for seed in seeds}
    strict: dict = {seed: [] for seed in seeds}
    incidental: dict = {seed: [] for seed in seeds}
    masked_hex_runs = 0
    for surface in surfaces:
        path = Path(surface["path"])
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text, count = HEX_RUN_PATTERN.subn(
            lambda match: "X" * len(match.group(0)), text)
        masked_hex_runs += count
        for seed in seeds:
            strict_spans = set()
            for match in strict_patterns[seed].finditer(text):
                strict_spans.add((match.start(), match.end()))
                context = text[max(0, match.start() - 24):match.end() + 24]
                strict[seed].append({
                    "surface": surface["label"],
                    "token": match.group(0),
                    "context": context.replace("\n", " ").strip(),
                })
            for match in loose_patterns[seed].finditer(text):
                if (match.start(), match.end()) in strict_spans:
                    continue
                context = text[max(0, match.start() - 24):match.end() + 24]
                incidental[seed].append({
                    "surface": surface["label"],
                    "token": match.group(0),
                    "context": context.replace("\n", " ").strip(),
                })
    return strict, incidental, masked_hex_runs


def classify(seed: int, containers: dict, token_hits: list[dict],
             incidental_hits: list[dict] | None = None) -> dict:
    member = membership_hits(seed, containers)
    collisions = bool(member) or bool(token_hits)
    return {
        "seed": seed,
        "membership_constants": member,
        "token_hits": token_hits,
        "incidental_numeric": incidental_hits or [],
        "collides": collisions,
    }


def audit_block(*, start: int | None = None, count: int | None = None,
                seeds_module=None, surfaces=None) -> dict:
    """The audit record for one 20-seed block (pure; no write)."""
    seeds = C.seed_block(start, count)
    containers = seed_module_membership(seeds_module)
    surfaces = default_surfaces() if surfaces is None else surfaces
    token_hits, incidental_hits, masked_hex_runs = scan_surfaces(seeds, surfaces)
    rows = [classify(seed, containers, token_hits[seed], incidental_hits[seed])
            for seed in seeds]
    collisions = [row["seed"] for row in rows if row["collides"]]
    return {
        "requested_start": seeds[0],
        "seeds": seeds,
        "rows": rows,
        "collisions": collisions,
        "free": not collisions,
        "surfaces": [s["label"] for s in surfaces],
        "missing_surfaces": [s["label"] for s in surfaces
                             if not Path(s["path"]).is_file()],
        "masked_hex_runs": masked_hex_runs,
    }


def choose_block(*, start: int | None = None, count: int | None = None,
                 seeds_module=None, surfaces=None, max_probe: int = 200) -> dict:
    """Return the first fully free block at or above ``start`` (mechanical).

    Each probe advances by the block size, exactly as registration section 2
    prescribes ("take the next fully free block of 20 at or above 9601").
    """
    start = C.SEED_BLOCK_START if start is None else int(start)
    count = C.SEED_BLOCK_SIZE if count is None else int(count)
    probed = []
    candidate = start
    for _ in range(max_probe):
        record = audit_block(start=candidate, count=count,
                             seeds_module=seeds_module, surfaces=surfaces)
        probed.append({"start": candidate, "free": record["free"],
                       "collisions": record["collisions"]})
        if record["free"]:
            record["probed"] = probed
            return record
        candidate += count
    raise RuntimeError(f"no free block of {count} found within {max_probe} probes")


def build_audit(*, start: int | None = None, count: int | None = None,
                seeds_module=None, surfaces=None) -> dict:
    """The committed audit payload: chosen block + its full row evidence."""
    chosen = choose_block(start=start, count=count, seeds_module=seeds_module,
                          surfaces=surfaces)
    registration_sha = (C.registration_lf_sha()
                        if C.REGISTRATION_PATH.is_file() else None)
    return {
        "schema": C.SEED_AUDIT_SCHEMA,
        "registration_path": str(C.REGISTRATION_PATH),
        "registration_lf_sha256": registration_sha,
        "registration_lf_sha256_expected": C.REGISTRATION_LF_SHA,
        "registration_matches": registration_sha == C.REGISTRATION_LF_SHA,
        "first_choice_block": C.seed_block(start, count),
        "chosen_block": chosen["seeds"],
        "chosen_start": chosen["seeds"][0],
        "chosen_free": chosen["free"],
        "collisions_in_first_choice": chosen["probed"][0]["collisions"],
        "probed_blocks": chosen["probed"],
        "rows": chosen["rows"],
        "surfaces": chosen["surfaces"],
        "missing_surfaces": chosen["missing_surfaces"],
        "masked_hex_runs": chosen["masked_hex_runs"],
        "read_only": True,
        "external_root": str(EXTERNAL_ROOT),
        "selection_rule": (
            "first fully free block of 20 at or above 9601; a collision is a "
            "seed-identity use (seeds.py range membership or a standalone "
            "<seed>/w<seed> token outside a decimal or a long hex hash); "
            "decimal-float coincidences and hash substrings are reported as "
            "incidental_numeric and do not collide. The INC1 registration "
            "and the INC1 WORLD_PLAN.json (this audit's downstream artifact) "
            "are excluded from the surfaces."),
    }


def write_audit(payload: dict, path: Path | None = None) -> Path:
    path = C.SEED_AUDIT_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="INC1 fresh-seed audit")
    parser.add_argument("--start", type=int, default=C.SEED_BLOCK_START)
    parser.add_argument("--count", type=int, default=C.SEED_BLOCK_SIZE)
    parser.add_argument("--out", type=Path, default=C.SEED_AUDIT_PATH)
    parser.add_argument("--no-write", action="store_true",
                        help="print the audit without writing the file")
    args = parser.parse_args(argv)
    payload = build_audit(start=args.start, count=args.count)
    if not args.no_write:
        path = write_audit(payload, args.out)
        print(f"wrote {path}")
    print(json.dumps({
        "chosen_block": [payload["chosen_start"],
                         payload["chosen_start"] + args.count - 1],
        "free": payload["chosen_free"],
        "collisions_in_first_choice": payload["collisions_in_first_choice"],
        "missing_surfaces": payload["missing_surfaces"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
