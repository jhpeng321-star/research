"""R14: compare lane_change_secure_gap-off execution with the real, isolated 13E launch sources.

Run with ``python tools/r14_checks/bit_identity.py`` (Git checkout required).
Cases: the four R13E identity cases (13C helpers) plus 13E-like lane_learnable /
courtesy-mode models (emergency braking + secure-gap veto + learned yield on) whose
lane-1 -> lane-0 changes are open, so the R14 code path is exercised with the flag off.
Only selected package sources and the two synthetic test helpers are exported
from the local Git object database. No worktree switch, dataset, or checkpoint
from a training run is needed. All generated files stay below workspace/runs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]
BASE = "65cc714e0e73c532583c61f1bbf0ea7d1d6f6b98"   # S-R13E launch code
HELPERS = ("tests/test_r10b_courtesy.py", "tests/test_r11_courtesy.py")
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "truth", "vtype")


def workspace_path(path: str | Path) -> Path:
    path = Path(path)
    candidate = path if path.is_absolute() else ROOT / path
    for value in (str(candidate), str(candidate.resolve())):
        if any(token in value.lower() for token in FORBIDDEN):
            raise ValueError(f"restricted path: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"path must stay inside workspace: {candidate}")
    return resolved


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def export_base(destination: Path) -> list[str]:
    """Export a source-only allowlist, checking names before reading their bytes."""
    destination = workspace_path(destination)
    if destination.exists():
        raise ValueError("base export must be a new directory (no stale package files)")
    names = _git("ls-tree", "-r", "--name-only", BASE, "--", "src/wms", *HELPERS)
    selected = []
    for name in names.decode("utf-8").splitlines():
        path = PurePosixPath(name)
        allowed = (name.startswith("src/wms/") and path.suffix in (".py", ".json"))
        if not (allowed or name in HELPERS):
            continue
        if any(token in name.lower() for token in FORBIDDEN):
            continue
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe Git path: {name}")
        selected.append(name)
    if not set(HELPERS).issubset(selected) or "src/wms/__init__.py" not in selected:
        raise RuntimeError("base commit is missing the synthetic helpers or package")
    # Archive only the already-validated filenames, never the repository or data tree.
    archive = _git("archive", "--format=tar", BASE, "--", *selected)
    destination.mkdir(parents=True)
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        for member in stream:
            if member.isdir():
                continue
            if member.name not in selected or not member.isfile():
                raise ValueError(f"unexpected archive entry: {member.name}")
            target = workspace_path(destination / member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise RuntimeError(f"missing archive bytes: {member.name}")
            with source:
                target.write_bytes(source.read())
            seen.add(member.name)
    if seen != set(selected):
        raise RuntimeError("incomplete base source export")
    return selected


def bundled_base() -> tuple[Path, list[str]]:
    """Validate the pinned source export shipped in a Git-less deployment archive."""
    manifest = workspace_path(ROOT / "R13E_SOURCE.json")
    doc = json.loads(manifest.read_text(encoding="utf-8")).get("baseline", {})
    if doc.get("commit") != BASE or doc.get("directory") != ".r13e_baseline":
        raise ValueError("missing pinned round-13C source export")
    if (ROOT / ".r13e_baseline").is_symlink():
        raise ValueError("symlink baseline export")
    base = workspace_path(ROOT / ".r13e_baseline")
    files = doc.get("files", {})
    if not isinstance(files, dict) or not set((*HELPERS, "src/wms/__init__.py")) <= set(files):
        raise ValueError("incomplete round-13C source manifest")
    for name, digest in files.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts
                or not ((name.startswith("src/wms/") and path.suffix in (".py", ".json"))
                        or name in HELPERS)):
            raise ValueError(f"invalid baseline source name: {name}")
        if (base / name).is_symlink() or not isinstance(digest, str):
            raise ValueError(f"invalid baseline source: {name}")
        target = workspace_path(base / name)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"baseline source drift: {name}")
    actual = set()
    for parent, directories, names in os.walk(base, followlinks=False):
        for name in (*directories, *names):
            candidate = Path(parent) / name
            if candidate.is_symlink():
                raise ValueError("symlink in baseline export")
            workspace_path(candidate)
        actual.update((Path(parent) / name).relative_to(base).as_posix() for name in names)
    if actual != set(files):
        raise ValueError("baseline source file set changed")
    return base, sorted(files)


def snapshot(source: Path, output: Path, *, explicit_off: bool = False) -> None:
    """Executed in a fresh interpreter with exactly one chosen package on PYTHONPATH."""
    import torch
    import wms
    from dataclasses import replace
    from test_r10b_courtesy import sequence, state_at_step
    from test_r11_courtesy import model_v

    source, output = workspace_path(source), workspace_path(output)
    if not Path(wms.__file__).resolve().is_relative_to(source / "src/wms"):
        raise RuntimeError(f"wrong wms import: {wms.__file__}")
    for name in ("test_r10b_courtesy", "test_r11_courtesy"):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(source / "tests"):
            raise RuntimeError(f"wrong synthetic helper import: {name}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def freeze(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: freeze(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(freeze(item) for item in value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported snapshot output: {type(value)}")

    cases = (
        ("residual", 0, "legacy", None),
        ("driver_params", 0, "physical", None),
        ("hybrid", 2, "physical", None),
        ("hybrid", 2, "legacy", 9.0),
    )
    result = {}
    with torch.no_grad():
        for mode, version, parameter_mode, old_emergency in cases:
            key = f"{mode}/courtesy{version}/{parameter_mode}/old{old_emergency}"
            model = model_v(version=version, randomize=True, rollout_mode=mode,
                            driver_param_mode=parameter_mode)
            changes = {"emergency_decel_mps2": old_emergency}
            if explicit_off:
                changes["emergency_braking"] = False
                changes["secure_gap_veto"] = False
                changes["lane_change_secure_gap"] = False
            model.physics = replace(model.physics, **changes)
            batch = sequence(n=9, cav=True)
            # Exercise actual leaders, an absent slot, a ramp obstacle, and CAV execution.
            batch["states"][:, :, 1, 0] = 26.0
            batch["states"][:, :, 0, 4] = 18.0
            batch["states"][:, :, 1, 4] = 9.0
            batch["present"][:, 1, 3] = False
            outputs = {"state_dict": freeze(model.state_dict())}
            for step in range(3):
                state = state_at_step(batch, step)
                torch.manual_seed(1700 + step)
                outputs[f"forward/{step}"] = freeze(model(state))
                torch.manual_seed(2700 + step)
                outputs[f"step/{step}"] = freeze(model.step(
                    state_at_step(batch, step), cav_teacher=model.hold_teacher(state),
                    projection=None))
            torch.manual_seed(3700)
            outputs["rollout"] = freeze(model.rollout(state_at_step(batch, 0), n_steps=6,
                                                       cav_controller=model.hold_teacher,
                                                       projection=None))
            result[key] = outputs
        result.update(_lane_cases(explicit_off))
    torch.save(result, output)


def _lane_cases(explicit_off: bool) -> dict:
    """13E-like lane_learnable models with open mainline changes (R14 D1 path, flag off)."""
    import torch
    import torch.nn.functional as F
    from dataclasses import replace
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.physics.config import PhysicsConfig
    from wms.train.inputs import state_from_kin

    def freeze(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: freeze(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(freeze(item) for item in value)
        return value

    out = {}
    for mode in ("driver_params", "hybrid"):
        physics = PhysicsConfig(lane_learnable=True, lane_rule="learned_discretionary", lane_gap_rule="speed",
                                mainline_lane_mode="courtesy", stop_time_integration=True,
                                ramp_lane_end_stop=True, ramp_risk_retain=True,
                                emergency_braking=True, secure_gap_veto=True)
        if explicit_off:
            physics = replace(physics, lane_change_secure_gap=False)
        torch.manual_seed(1414)
        model = WmSModel(ModelSpec(rollout_mode=mode, lane_model="hazard_v5", lane_decision="sample",
                                   merge_decision="first_passage", merge_passage_law="hybrid",
                                   mainline_merge_yield_learnable=True), physics=physics).eval()
        pos = torch.tensor([-150., -230., -40., 30., 60., -130., -60., 10.])
        spd = torch.tensor([3., 25., 22., 0.5, 20., 2., 24., 4.])
        lanes = torch.tensor([1, 0, 0, 2, 1, 1, 0, 2])
        kin = torch.zeros(1, 8, 7)
        kin[..., 0], kin[..., 4], kin[..., 6] = pos, spd, 5.
        kin[..., 5] = torch.tensor([0., .5, -.3, 0., .2, -1., 0., .1])
        kin[..., 1:4] = F.one_hot(lanes, 3).float()
        present = torch.ones(1, 8, dtype=torch.bool)
        present[0, 7] = False
        cav = torch.zeros(1, 8, dtype=torch.bool)
        cav[0, 4] = True
        state = {**state_from_kin(kin, present), "is_cav": cav}
        res = {"state_dict": freeze(model.state_dict())}
        torch.manual_seed(4100)
        res["forward"] = freeze(model(state))
        torch.manual_seed(4200)
        res["step_draw"] = freeze(model.step(state, cav_teacher=model.hold_teacher(state),
                                             lane_uniform=torch.full((1, 8), 1e-9)))
        torch.manual_seed(4300)
        res["rollout"] = freeze(model.rollout(state, n_steps=6, cav_controller=model.hold_teacher,
                                              projection=None))
        out[f"lane_learnable/{mode}"] = res
    return out


def compare_snapshots(expected, actual, path: str = "root") -> int:
    """Check every tensor's dtype, shape and bytes (including signed zero / NaN bits)."""
    import torch

    if type(expected) is not type(actual):
        raise AssertionError(f"{path}: type mismatch {type(expected)} != {type(actual)}")
    if isinstance(expected, torch.Tensor):
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise AssertionError(f"{path}: tensor dtype/shape mismatch")
        left = expected.contiguous().reshape(-1).view(torch.uint8)
        right = actual.contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(left, right):
            raise AssertionError(f"{path}: tensor bytes differ")
        return 1
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"{path}: dictionary keys differ")
        return sum(compare_snapshots(value, actual[key], f"{path}/{key}")
                   for key, value in expected.items())
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: sequence lengths differ")
        return sum(compare_snapshots(left, right, f"{path}/{i}")
                   for i, (left, right) in enumerate(zip(expected, actual)))
    if expected != actual:
        raise AssertionError(f"{path}: scalar values differ: {expected!r} != {actual!r}")
    return 0


def run_check(out_dir: str | Path = "runs/r14_bit_identity") -> dict:
    import torch

    out_dir = workspace_path(out_dir)
    if not out_dir.is_relative_to(ROOT / "runs"):
        raise ValueError("identity artifacts must stay below workspace/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="check-", dir=out_dir))
    if (ROOT / ".git").exists():
        base = run / "base"
        exported = export_base(base)
    else:
        raise RuntimeError("R14 identity check needs the Git object database")
    snapshots = []
    for label, source, explicit in (("base", base, False), ("default", ROOT, False),
                                    ("explicit_off", ROOT, True)):
        output = run / f"{label}.pt"
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join((str(source / "src"), str(source / "tests"))),
                   PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
                   PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--snapshot", str(source),
                   "--output", str(output)]
        if explicit:
            command.append("--explicit-off")
        subprocess.run(command, cwd=run, env=env, check=True, capture_output=True, text=True,
                       timeout=180)
        snapshots.append(torch.load(output, weights_only=True, map_location="cpu"))
    counts = {label: compare_snapshots(snapshots[0], snapshot_data)
              for label, snapshot_data in zip(("default", "explicit_off"), snapshots[1:])}
    if not all(count > 0 for count in counts.values()):
        raise AssertionError("identity comparison checked no tensors")
    report = {"status": "PASS", "base_commit": BASE, "cases": len(snapshots[0]),
              "rollout_steps": 6, "tensor_comparisons": counts,
              "exported_source_files": len(exported), "artifacts": str(run.relative_to(ROOT))}
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/r14_bit_identity")
    parser.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--explicit-off", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.snapshot is not None:
        if args.output is None:
            parser.error("--snapshot requires --output")
        snapshot(args.snapshot, args.output, explicit_off=args.explicit_off)
    else:
        try:
            report = run_check(args.out_dir)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"identity subprocess failed:\n{exc.stderr}") from exc
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
