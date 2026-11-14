"""Saved-state exactness guard for the perfect simulator (PL-10).

VENDORED THIN ADAPTER of the real WM-41/WM-44 saved-state machinery,
<wm34 worktree>/output/campaign_pilot_v1/saved_state.py (read-only sibling
workspace, root from paths.rd_wt_root(); the exactness guard landed there in
fix e983f45e). The brief for
PL-10 sanctions exactly this: read-only import OR a vendored thin adapter --
never a re-implementation. :func:`validate_state_file_for_exactness` is the
real guard verbatim (SUMO's default saveState rounds positions to 2 decimals
and omits the RNG state; a restore from such a file is NOT exact with
vType sigma > 0). :func:`validate_manifest_for_restore` mirrors the restore
half of the real ``load_branch_state`` (schema, branch time, SUMO version
drift, state-file digest, scenario digests) so the planner's
``StateSnapshotter`` refuses low-precision / no-RNG snapshots on BOTH the
save and the restore path, loudly, with no silent fallback.

A branch state is only usable when the saving SUMO ran with
``--save-state.precision 17 --save-state.rng`` (:data:`REQUIRED_SAVE_STATE_CLI`;
SUMO >= 1.15.0 carries the simulator RNG state in the state file; the pinned
interpreter runs 1.27.1).

SCENARIO EXACTNESS (PL-10c): :func:`check_scenario_exactness` refuses any
scenario whose route files create vehicles with RANDOM post-t0 draws --
``<flow>`` elements, ``<vTypeDistribution>`` elements, a ``<vType>`` with
``speedDev`` absent or != 0, or a vType/vehicle-level ``speedFactor`` given
as a distribution string (e.g. ``norm(1, 0.1)``). A ``loadState`` restores
the snapshot's RNG, but a flow-created vehicle AFTER the restore (e.g.
mainline.12 in the PL-10 smoke) re-draws type/speedFactor per branch --
bit-exactness is gone. Campaign-format scenarios (explicit ``<vehicle>``
list, fixed ``type=``, ``speedDev="0"``) carry no such draw and pass. The
one exception is INSTALLED explicit arrival draws (option A): then the
post-t0 departures are data, not draws.

Zero simulation: no SUMO/traci/libsumo import anywhere in this module; the
file checks are pure text/digest reads.
"""

import gzip
import hashlib
import importlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .paths import rd_wt_root

#: Default root of the read-only wm34 sibling tree (env RD_WT_ROOT, E:
#: default; see paths.py; resolved at import).
WM34_OUTPUT_ROOT = str(rd_wt_root() / "wm34" / "output")

__all__ = [
    "HARNESS_COMPONENTS",
    "MANIFEST_FILE",
    "PINNED_SUMO_VERSION",
    "REQUIRED_SAVE_STATE_CLI",
    "SCHEMA",
    "STATE_DIR_NAME",
    "SUMO_RNG_STATE_MIN_VERSION",
    "SUMO_STATE_FILE",
    "ProcessReuseError",
    "ScenarioNotExactError",
    "StateSnapshotError",
    "check_scenario_exactness",
    "parse_sumo_version",
    "require_exact_state_file",
    "require_sumo_rng_state",
    "sha256_file",
    "sha256_text",
    "validate_manifest_for_restore",
    "validate_state_file_for_exactness",
]

#: wm41_saved_state/1 -- the real protocol schema (saved_state.py).
SCHEMA = "wm41_saved_state/1"
STATE_DIR_NAME = "saved_state"
SUMO_STATE_FILE = "state.xml"
MANIFEST_FILE = "STATE_MANIFEST.json"

#: SUMO state files carry the RNG state from 1.15.0; the pinned interpreter
#: runs 1.27.1 (both asserted by the real save/load paths).
SUMO_RNG_STATE_MIN_VERSION = (1, 15, 0)
PINNED_SUMO_VERSION = (1, 27, 1)

#: Declared cross-tick harness components (saved_state.py HARNESS_COMPONENTS).
HARNESS_COMPONENTS = (
    "ownership",
    "circles_controller",
    "pi_session",
    "loop_state",
    "request_ledger",
    "random_state",
)

#: The exactness contract for every saving SUMO (manifest state_options).
REQUIRED_SAVE_STATE_CLI = ("--save-state.precision", "17", "--save-state.rng", "true")

#: Real traci.getVersion() returns (api_version, "SUMO <maj>.<min>.<patch>
#: [build...]") -- the dotted triple is matched wherever it sits (the real
#: r2 launch-crash fix, kept verbatim).
_SUMO_VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")


class StateSnapshotError(RuntimeError):
    """A save/restore exactness-contract violation; never silently worked
    around (the real ``SavedStateError`` semantics, planner-side name)."""


class ProcessReuseError(StateSnapshotError):
    """A SUMO process that already restored+ran ONE branch was offered
    another restore (PL-10d). The re-smoke measured WHY one loadState per
    process is the contract: the first restore into a fresh process is
    bit-exact, while a native branch restored into the SAME process after a
    controlled (damping) branch diverged at t+5 s with deltas identical to
    the plan branch's -- simulator state carries over across loadState.
    Refused BEFORE any second loadState is issued; a fresh process per
    restored branch (native included) is the only path."""


def parse_sumo_version(version_info) -> "tuple[int, int, int]":
    """traci.getVersion() -> (22, 'SUMO 1.27.1') -> (1, 27, 1).

    Accepts the REAL two-element TraCI return, a bare dotted-version string,
    or an already-parsed triple of ints (verbatim from saved_state.py)."""
    if (
        isinstance(version_info, (tuple, list))
        and len(version_info) == 3
        and all(isinstance(p, int) and not isinstance(p, bool) for p in version_info)
    ):
        return tuple(version_info)
    if isinstance(version_info, str):
        match = _SUMO_VERSION_RE.search(version_info)
        if match:
            return tuple(int(part) for part in match.groups())
    if isinstance(version_info, (tuple, list)) and len(version_info) >= 2:
        match = _SUMO_VERSION_RE.search(str(version_info[1]))
        if match:
            return tuple(int(part) for part in match.groups())
    raise StateSnapshotError(f"unparseable SUMO version: {version_info!r}")


def require_sumo_rng_state(version) -> None:
    """Refuse SUMO versions that predate RNG-in-state support (verbatim)."""
    if tuple(version) < SUMO_RNG_STATE_MIN_VERSION:
        raise StateSnapshotError(
            f"SUMO {tuple(version)} predates RNG-in-state support "
            f"(>= {SUMO_RNG_STATE_MIN_VERSION}); the exactness guarantee "
            "cannot be made - upgrade SUMO or fall back to deterministic "
            "prefix re-runs (costed separately)"
        )


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_state_file_for_exactness(state_path) -> dict:
    """Refuse state files that cannot restore bit-exactly (WM-44 v8 guard).

    Vendored verbatim from the real saved_state.py: requires an <rngState>
    element and at least one pos= value with more than 2 decimal digits
    (proving a higher write precision than SUMO's default 2)."""
    text = Path(state_path).read_text(encoding="utf-8", errors="replace")
    problems = []
    has_rng = "<rngState" in text
    if not has_rng:
        problems.append(
            "state file carries no <rngState> element "
            "(--save-state.rng was not enabled on the saving run)"
        )
    max_decimals = 0
    for match in re.finditer(r'pos="([^"]*)"', text):
        for token in match.group(1).split():
            if "." in token:
                max_decimals = max(max_decimals, len(token.split(".", 1)[1]))
    if max_decimals <= 2:
        problems.append(
            f"state positions are written with at most {max_decimals} "
            "decimal digits (--save-state.precision too low for a "
            "bit-exact restore)"
        )
    return {
        "has_rng": has_rng,
        "max_pos_decimals": max_decimals,
        "problems": problems,
    }


def require_exact_state_file(state_path) -> dict:
    """The guard, raising: a snapshot that lacks full precision or RNG state
    is refused loudly (PERFECT_SIM_DESIGN.md 'Snapshot exactness
    requirements'; never a silent fallback)."""
    check = validate_state_file_for_exactness(state_path)
    if check["problems"]:
        raise StateSnapshotError(
            "refusing to use an unusable SUMO state file: " + "; ".join(check["problems"])
        )
    return check


def validate_manifest_for_restore(
    manifest: dict,
    *,
    expected_t0_s: float,
    sumo_version,
    scenario_digests=None,
    state_path=None,
) -> dict:
    """Restore-half of the real ``load_branch_state`` (vendored thin).

    Checks the manifest schema, the branch time, SUMO version drift, the
    state-file digest and the scenario digests, and REQUIRES the recorded
    state options to prove a full-precision RNG-carrying snapshot. ``state_path``
    (when given) is re-guarded with :func:`require_exact_state_file` so a
    doctored manifest cannot smuggle a rounded state file through. Returns
    the exactness check for the audit trail."""
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise StateSnapshotError(
            f"unexpected saved-state schema: "
            f"{manifest.get('schema') if isinstance(manifest, dict) else type(manifest)!r}"
        )
    if abs(float(manifest["branch_point_t0_s"]) - float(expected_t0_s)) > 1e-9:
        raise StateSnapshotError(
            "saved state is for a different branch time: "
            f"{manifest['branch_point_t0_s']} != {expected_t0_s}"
        )
    require_sumo_rng_state(parse_sumo_version(sumo_version))
    if tuple(manifest.get("sumo_version", ())) != tuple(parse_sumo_version(sumo_version)):
        raise StateSnapshotError(
            "SUMO version drift save-vs-restore: "
            f"{manifest.get('sumo_version')} vs {tuple(parse_sumo_version(sumo_version))}"
        )
    options = manifest.get("state_options", {})
    if not options.get("rng_state_in_file"):
        raise StateSnapshotError(
            "saved-state manifest records rng_state_in_file=False; the "
            "snapshot cannot restore bit-exactly -- refusing (the saving "
            f"run must pass {list(REQUIRED_SAVE_STATE_CLI)})"
        )
    if int(options.get("save_state_precision_decimals", 0)) <= 2:
        raise StateSnapshotError(
            "saved-state manifest records save_state_precision_decimals<=2; "
            "the snapshot is position-rounded and cannot restore bit-exactly "
            "-- refusing (the saving run must pass "
            f"{list(REQUIRED_SAVE_STATE_CLI)})"
        )
    if tuple(options.get("required_cli", ())) != tuple(REQUIRED_SAVE_STATE_CLI):
        raise StateSnapshotError(
            "saved-state manifest required_cli does not match the exactness "
            f"contract {list(REQUIRED_SAVE_STATE_CLI)}: {options.get('required_cli')!r}"
        )
    if state_path is not None:
        check = require_exact_state_file(state_path)
        if sha256_file(state_path) != manifest.get("sumo_state_sha256"):
            raise StateSnapshotError("saved SUMO state file fails its digest")
    else:
        check = {
            "has_rng": bool(options.get("rng_state_in_file")),
            "max_pos_decimals": int(options.get("save_state_precision_decimals", 0)),
            "problems": [],
        }
    for path, digest in (scenario_digests or {}).items():
        if manifest.get("scenario_digests", {}).get(path) != digest:
            raise StateSnapshotError(
                f"scenario digest drift for {path}: the branch process must "
                "start from the SAME scenario the base run used"
            )
    return check


def optional_real_saved_state(wm34_output_root: str = WM34_OUTPUT_ROOT):
    """The REAL saved_state module (read-only import), or None.

    Importing ``campaign_pilot_v1.saved_state`` as a namespace package keeps
    the wm34 tree read-only and optional: production snapshots prefer the
    real ``save_branch_state`` / ``load_branch_state`` byte-for-byte and the
    vendored adapter above is the standalone fallback (declared as such in
    the manifest). Tests never call this (fakes only)."""
    try:
        root = str(Path(wm34_output_root))
        if root not in __import__("sys").path:
            __import__("sys").path.insert(0, root)
        return importlib.import_module("campaign_pilot_v1.saved_state")
    except Exception:  # noqa: BLE001 - absence is an expected, reported state
        return None


def write_manifest(state_dir: Path, manifest: dict) -> Path:
    """Write STATE_MANIFEST.json once (x-mode, like the real save path)."""
    path = state_dir / MANIFEST_FILE
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False, sort_keys=True) + "\n")
    return path


# ------------------------------------------------- scenario exactness guard
class ScenarioNotExactError(StateSnapshotError):
    """The scenario itself would re-draw random vehicle attributes after the
    t0 restore (flows / vType distributions / non-zero speedDev /
    distribution speedFactor) and no explicit arrival draws are installed;
    branch bit-exactness cannot hold. Refused loudly, never silently worked
    around (PL-10c; the PL-10 smoke's mainline.12 divergence diagnosis)."""


def _is_plain_number(text: str) -> bool:
    """False exactly for SUMO distribution strings ("norm(1, 0.1)", ...)."""
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _xml_root(path: Path):
    """Root element of a route/config XML, plain or .gz (pure read).

    FAIL-CLOSED: a missing or unparseable file is a ScenarioNotExactError,
    never a FileNotFoundError/ParseError that could bypass the guard."""
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rb") as stream:
                return ET.parse(stream).getroot()
        return ET.parse(str(path)).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ScenarioNotExactError(
            f"scenario file is missing or unparseable: {path} ({exc!r})"
        ) from exc


def _scenario_offenses(path: Path, root) -> "list[str]":
    """Every element of one route file that draws randomness after t0."""
    offenses: "list[str]" = []
    where = str(path)
    for flow in root.iter("flow"):
        offenses.append(
            f"{where}: <flow id={flow.get('id')!r}> creates vehicles after "
            "t0 whose type/speedFactor draws depend on the RNG stream "
            "position (a restored branch re-draws them differently)"
        )
    for dist in root.iter("vTypeDistribution"):
        offenses.append(
            f"{where}: <vTypeDistribution id={dist.get('id')!r}> draws a "
            "random member vType per vehicle (per-branch divergence)"
        )
    for vtype in root.iter("vType"):
        vid = vtype.get("id")
        speed_dev = vtype.get("speedDev")
        if speed_dev is None:
            offenses.append(
                f"{where}: <vType id={vid!r}> has no speedDev attribute "
                "(SUMO default 0.1 draws a random per-vehicle speed deviation)"
            )
        else:
            try:
                deviates = float(speed_dev) != 0.0
            except ValueError:
                deviates = True  # unparseable: treat as random, never guess
            if deviates:
                offenses.append(
                    f"{where}: <vType id={vid!r}> speedDev={speed_dev!r} != 0 "
                    "draws a random per-vehicle speed deviation"
                )
        factor = vtype.get("speedFactor")
        if factor is not None and not _is_plain_number(factor):
            offenses.append(
                f"{where}: <vType id={vid!r}> speedFactor={factor!r} is a "
                "distribution string (random per-vehicle factor)"
            )
    for vehicle in root.iter("vehicle"):
        factor = vehicle.get("speedFactor")
        if factor is not None and not _is_plain_number(factor):
            offenses.append(
                f"{where}: <vehicle id={vehicle.get('id')!r}> "
                f"speedFactor={factor!r} is a distribution string (random "
                "per-vehicle factor)"
            )
    return offenses


def _cfg_input_option(root, name: str):
    """The value of the SUMO cfg input option ``name`` (e.g. route-files),
    under the ``<input>`` section of ANY root element (<configuration>,
    <sumoConfiguration>, ...), in either the attribute form
    (``<input route-files="...">``) or the element form
    (``<route-files value="..."/>``). None when the option is absent."""
    input_el = root.find("input")
    if input_el is None:
        return None
    if input_el.get(name) is not None:
        return input_el.get(name)
    element = input_el.find(name)
    if element is not None:
        return element.get("value")
    return None


_ROUTE_CONTENT_TAGS = {"route", "vehicle", "vtype", "vtypedistribution", "flow", "trip"}


def _carries_route_content(root) -> bool:
    """True when the parsed XML actually carries routes/vTypes/flows (an
    additional-files entry that is really an edges/TL logic file is NOT a
    route file and must not count towards the resolved set)."""
    if root.tag.lower() == "routes":
        return True
    return any(elem.tag.lower() in _ROUTE_CONTENT_TAGS for elem in root.iter())


def _expand_scenario_entry(entry) -> "list[tuple[Path, object]]":
    """One scenario entry -> [(route file, parsed root), ...].

    A .sumocfg (root element of any name) resolves its comma-separated
    ``<input>`` route-files, relative to the cfg directory, ALWAYS; each
    ``additional-files`` entry is parsed too but only counts when the file
    actually carries routes/vTypes/flows. A non-cfg entry IS a route file.
    FAIL-CLOSED: a cfg that resolves to ZERO route files (option absent or
    empty), a missing file or an unparseable file raises
    ScenarioNotExactError -- the guard never passes a scenario vacuously."""
    path = Path(entry)
    if path.suffix != ".sumocfg":
        return [(path, _xml_root(path))]
    cfg_dir = path.parent
    candidates: "list[tuple[Path, bool]]" = []  # (file, is route-files option)
    cfg_root = _xml_root(path)
    for option in ("route-files", "additional-files"):
        raw = _cfg_input_option(cfg_root, option)
        if raw is None:
            continue
        for name in raw.split(","):
            name = name.strip()
            if name:
                candidates.append((cfg_dir / name, option == "route-files"))
    if not candidates:
        raise ScenarioNotExactError(
            f"{path}: no route files resolved from the .sumocfg <input> "
            "(route-files absent or empty); the guard never passes a "
            "scenario vacuously -- refusing"
        )
    expanded: "list[tuple[Path, object]]" = []
    for candidate, from_route_files in candidates:
        root = _xml_root(candidate)
        if not from_route_files and not _carries_route_content(root):
            continue  # an additional file that is not route content
        expanded.append((candidate, root))
    return expanded


def check_scenario_exactness(route_files, *, explicit_arrivals_installed: bool) -> dict:
    """Refuse scenarios that would re-draw random vehicle attributes after
    the t0 restore (PL-10c).

    Parses every SUMO route XML in ``route_files`` (a path or sequence of
    paths; plain or ``.gz``; a ``.sumocfg`` expands to the route files its
    ``<input>`` names -- attribute or element form, any root element, with
    route-carrying ``additional-files`` honoured -- relative to the cfg
    directory) and collects every offending element: ``<flow>``,
    ``<vTypeDistribution>``, a ``<vType>`` whose ``speedDev`` is absent or
    != 0, and a vType/vehicle-level ``speedFactor`` given as a distribution
    string.

    FAIL-CLOSED: zero resolved route files, a missing file or an
    unparseable file raise :class:`ScenarioNotExactError` -- the guard
    never returns exact=True vacuously. It likewise raises listing EVERY
    offending element unless ``explicit_arrivals_installed`` is True -- the
    option-A exception: with a non-empty explicit departure list installed
    the post-t0 departures are data, not draws. Campaign-format scenarios
    (explicit ``<vehicle>`` list, fixed types, ``speedDev="0"``) pass
    outright.

    Returns the audit record {"route_files", "offenses", "exact",
    "explicit_arrivals_installed"}; pure file reads, zero simulation."""
    entries = [route_files] if isinstance(route_files, (str, Path)) else list(route_files)
    if not entries:
        raise ScenarioNotExactError("check_scenario_exactness: no route files given")
    checked: "list[str]" = []
    offenses: "list[str]" = []
    for entry in entries:
        for path, root in _expand_scenario_entry(entry):
            checked.append(str(path))
            offenses.extend(_scenario_offenses(path, root))
    if not checked:
        raise ScenarioNotExactError(
            "check_scenario_exactness resolved ZERO route files; the guard "
            "never passes a scenario vacuously -- refusing"
        )
    if offenses and not explicit_arrivals_installed:
        raise ScenarioNotExactError(
            "refusing a scenario that cannot branch bit-exactly: post-t0 "
            "departures would re-draw random vType/speed attributes per "
            "branch, and NO explicit arrival draws are installed "
            f"({len(offenses)} offending element(s)):\n  "
            + "\n  ".join(offenses)
            + "\nthe ONLY exception is installed option-A arrival draws; "
            "otherwise use a campaign-format scenario (explicit <vehicle> "
            'list, fixed types, speedDev="0")'
        )
    return {
        "route_files": checked,
        "offenses": offenses,
        "exact": not offenses,
        "explicit_arrivals_installed": bool(explicit_arrivals_installed),
    }
