"""INC1 launcher gates: registration, model lock, authorization, budget.

Registration sections 6-7: the runner refuses to start without an owner
``AUTHORIZATION_INC1.json`` that binds the registration's LF-sha256, and an
existing ``INC1_MODEL_LOCK.json`` whose shas are recorded; the declared budget
is 36,120 sim-s, complete runs are never retried, and a crashed run restarts
once with disclosure.  Every refusal is loud; no gate is silently relaxed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from . import constants as C
except ImportError:  # script execution
    import constants as C  # type: ignore


class GateError(PermissionError):
    """A launcher gate refusal; the run does not start."""


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise GateError(f"required file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"{path} is not valid JSON: {exc}") from exc


def check_registration(path: Path | None = None) -> str:
    """Return the registration LF-sha; refuse on a fingerprint mismatch."""
    path = C.REGISTRATION_PATH if path is None else Path(path)
    if not path.is_file():
        raise GateError(f"registration is missing: {path}")
    sha = C.lf_sha256(path)
    if sha != C.REGISTRATION_LF_SHA:
        raise GateError(
            f"registration LF-sha {sha} != registered {C.REGISTRATION_LF_SHA}; "
            "the registration must not be edited")
    return sha


def check_model_lock(path: Path | None = None, *,
                     registration_lf_sha: str | None = None) -> dict:
    """The model lock must exist and record the registered shas."""
    path = C.MODEL_LOCK_PATH if path is None else Path(path)
    lock = load_json(path)
    if lock.get("schema") != C.MODEL_LOCK_SCHEMA:
        raise GateError(f"{path}: schema {lock.get('schema')!r} != "
                        f"{C.MODEL_LOCK_SCHEMA!r}")
    if lock.get("template") is True:
        raise GateError(f"{path}: a template lock never authorizes a run")
    registration_lf_sha = (registration_lf_sha
                           if registration_lf_sha is not None
                           else C.REGISTRATION_LF_SHA)
    if lock.get("registration_lf_sha256") != registration_lf_sha:
        raise GateError(f"{path}: registration_lf_sha256 does not bind the "
                        "registration")
    required = ("member_1_sha256", "pack_manifest_sha256", "code_commit",
                "partA_v2g_param_sha256")
    for key in required:
        value = lock.get(key)
        if not isinstance(value, str) or not value.strip():
            raise GateError(f"{path}: missing or empty {key!r}")
    zero_shot = lock.get("zero_shot")
    if not isinstance(zero_shot, dict):
        raise GateError(f"{path}: missing zero_shot assertion")
    for key in ("event_tokens_nonempty", "obstacle_insertions"):
        if int(zero_shot.get(key, -1)) != 0:
            raise GateError(
                f"{path}: zero-shot assertion failed ({key}="
                f"{zero_shot.get(key)!r}); the incident type was seen in "
                "training -- escalate to the owner")
    lock["_path"] = str(path)
    # the lock is JSON text, so its gate sha is LF-normalised; a raw hash
    # would diverge between a Linux (LF) and a Windows (CRLF) checkout.
    lock["_sha256"] = C.lf_sha256(path)
    return lock


def check_authorization(path: Path | None = None, *,
                        registration_lf_sha: str | None = None,
                        model_lock: dict | None = None) -> dict:
    """The owner authorization must bind the registration and the lock."""
    path = C.AUTHORIZATION_PATH if path is None else Path(path)
    auth = load_json(path)
    if auth.get("schema") != C.AUTHORIZATION_SCHEMA:
        raise GateError(f"{path}: schema {auth.get('schema')!r} != "
                        f"{C.AUTHORIZATION_SCHEMA!r}")
    if auth.get("template") is True:
        raise GateError(f"{path}: a template authorization never authorizes "
                        "a run")
    registration_lf_sha = (registration_lf_sha
                           if registration_lf_sha is not None
                           else C.REGISTRATION_LF_SHA)
    if auth.get("registration_lf_sha256") != registration_lf_sha:
        raise GateError(f"{path}: registration_lf_sha256 does not bind the "
                        "registration")
    if not auth.get("model_lock_path"):
        raise GateError(f"{path}: missing model_lock_path")
    if model_lock is not None:
        declared = auth.get("model_lock_sha256")
        if declared != model_lock.get("_sha256"):
            raise GateError(
                f"{path}: model_lock_sha256 {declared!r} != the existing "
                f"{model_lock.get('_sha256')!r}")
    declared_budget = float(auth.get("budget_total_sim_s", -1.0))
    if declared_budget != float(C.BUDGET_TOTAL_S):
        raise GateError(f"{path}: budget_total_sim_s {declared_budget} != "
                        f"{C.BUDGET_TOTAL_S}")
    auth["_path"] = str(path)
    return auth


def check_generators_manifest(path: Path | None = None) -> dict:
    """A2.6 gate: the live R5/R6 generator package must match the recorded
    LF shas in ``INC1_GENERATORS.json`` (the launcher refuses otherwise)."""
    if str(C.WD01_ROOT) not in sys.path:
        sys.path.insert(0, str(C.WD01_ROOT))
    from inc1_gen import manifest as GM  # noqa: PLC0415

    manifest_path = (C.GENERATORS_MANIFEST_PATH if path is None
                     else Path(path))
    if not manifest_path.is_file():
        raise GateError(
            f"INC1_GENERATORS.json is missing: {manifest_path} (A2.6 gate)")
    verdict = GM.verify_manifest(manifest_path)
    if not verdict.get("ok"):
        raise GateError(
            "INC1_GENERATORS.json refused the live R5/R6 generator package "
            f"(A2.6): {verdict.get('detail')}")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"generators_manifest": str(manifest_path),
            "generator_module_sha256": recorded.get("generator_module_sha256"),
            "detail": verdict.get("detail")}


def check_launcher(*, authorization_path: Path | None = None,
                   model_lock_path: Path | None = None,
                   registration_path: Path | None = None,
                   generators_manifest_path: Path | None = None) -> dict:
    """Run every pre-simulation gate; return the bound identities."""
    registration_lf_sha = check_registration(registration_path)
    generators_manifest = check_generators_manifest(generators_manifest_path)
    model_lock = check_model_lock(model_lock_path,
                                  registration_lf_sha=registration_lf_sha)
    authorization = check_authorization(
        authorization_path, registration_lf_sha=registration_lf_sha,
        model_lock=model_lock)
    return {
        "registration_lf_sha256": registration_lf_sha,
        "generators_manifest": generators_manifest,
        "model_lock": model_lock,
        "authorization": authorization,
    }


def check_inc1_release_gates(*, verdict_path: Path | None = None) -> dict:
    """A1.6 release gates: inc1_hook sha + freeze addendum + PASS verdict.

    The INC1 launcher additionally refuses unless
    (1) the patched INC1 driver copy matches the ``inc1_hook`` sha in
    ``PATCH_SHAS.json`` (and the diff matches),
    (2) ``WALL_V22_FREEZE_ADDENDUM_INC1.json`` exists, and
    (3) the harness-equivalence verdict is PASS.
    """
    from . import harness_equivalence as HE
    from . import wall_v22_freeze as W

    try:
        hook = W.verify_inc1_hook_patch()
    except W.FreezeError as exc:
        raise GateError(str(exc)) from exc
    path = (Path(verdict_path) if verdict_path is not None
            else HE.verdict_path(C.INC1_ROOT))
    if not HE.verdict_is_pass(path):
        raise GateError(
            "harness-equivalence verdict is not PASS (A1.6): "
            f"{path} missing or FAIL; run the A1.5 smoke first")
    return {"inc1_hook": hook, "harness_equivalence_verdict": str(path),
            "harness_equivalence_pass": True}


# -------------------------------------------------------------------- budget
def expected_sim_seconds(plan: dict) -> dict:
    """The plan's declared costs (warm-up + all arms)."""
    worlds = len(plan["worlds"])
    arms = sum(1 for w in plan["worlds"] for b in w["branches"]
               if b["arm"] != "warmup")
    warmup = worlds * float(C.T0_S)
    arm_s = arms * float(C.ARM_HORIZON_S)
    return {
        "warmup_sim_s": warmup,
        "arms_sim_s": arm_s,
        "subtotal_sim_s": warmup + arm_s,
    }


class BudgetCounter:
    """Append-only sim-second ledger; refuses to exceed the registered total."""

    def __init__(self, log_path: Path | None = None,
                 limit_sim_s: float = C.BUDGET_TOTAL_S):
        self.log_path = C.BUDGET_LOG if log_path is None else Path(log_path)
        self.limit_sim_s = float(limit_sim_s)

    def entries(self) -> list[dict]:
        if not self.log_path.is_file():
            return []
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def committed_sim_s(self) -> float:
        return sum(float(e.get("sim_s", 0.0)) for e in self.entries())

    def check(self, requested_sim_s: float) -> dict:
        committed = self.committed_sim_s()
        remaining = self.limit_sim_s - committed
        if requested_sim_s > remaining + 1e-9:
            raise GateError(
                f"budget refusal: requested {requested_sim_s:.1f} sim-s exceeds "
                f"remaining {remaining:.1f} of {self.limit_sim_s:.0f} sim-s "
                f"(committed {committed:.1f})")
        return {"committed_sim_s": committed, "remaining_sim_s": remaining,
                "requested_sim_s": float(requested_sim_s)}

    def charge(self, *, item: str, sim_s: float, run: str | None = None,
               detail: str | None = None) -> dict:
        check = self.check(float(sim_s))
        entry = {
            "schema": C.BUDGET_SCHEMA,
            "item": str(item),
            "run": run,
            "sim_s": float(sim_s),
            "committed_before": check["committed_sim_s"],
            "committed_after": check["committed_sim_s"] + float(sim_s),
            "detail": detail,
            "utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        return entry


def complete_receipt(run_dir: Path) -> Path:
    return Path(run_dir) / "RUN_RECEIPT.json"


def refuse_retry_of_complete(run_dir: Path) -> None:
    """A complete run is never retried (registration section 7)."""
    receipt = complete_receipt(run_dir)
    if receipt.is_file():
        raise GateError(
            f"refusing to re-run a completed run: {receipt} exists "
            "(complete runs are never retried)")


def crashed_restart_allowed(run_dir: Path) -> bool:
    """A crashed run restarts at most once; a second crash stays failed."""
    marker = Path(run_dir) / "CRASH_RESTART.json"
    if not marker.is_file():
        return True
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(record.get("restarts", 0)) < C.MAX_CRASHED_RESTARTS


def mark_crashed_restart(run_dir: Path, *, detail: str) -> dict:
    """Record the single disclosed restart of a crashed run."""
    marker = Path(run_dir) / "CRASH_RESTART.json"
    if marker.is_file():
        record = json.loads(marker.read_text(encoding="utf-8"))
    else:
        record = {"schema": "wm_n1_inc1_crash_restart/1", "restarts": 0,
                  "disclosures": []}
    record["restarts"] = int(record.get("restarts", 0)) + 1
    record["disclosures"].append({
        "utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "detail": detail,
    })
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def check_disk_preflight(path: Path | None = None, *,
                         min_free_gb: float = C.DISK_PREFLIGHT_MIN_FREE_GB,
                         disk_free_fn=None) -> dict:
    """Refuse unless the drive ``path`` resolves to has enough free space."""
    path = C.RUNS_ROOT if path is None else Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if disk_free_fn is None:
        import shutil as _shutil

        def disk_free_fn(p):
            return _shutil.disk_usage(str(p)).free

    free = float(disk_free_fn(path))
    free_gb = free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise GateError(
            f"disk preflight: {free_gb:.1f} GB free on {path} < "
            f"{min_free_gb:.0f} GB")
    return {"path": str(path), "free_gb": free_gb,
            "min_free_gb": float(min_free_gb)}
