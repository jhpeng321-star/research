"""The predict package must be pure numpy + stdlib: importable in a fresh
interpreter without pulling torch / SUMO / TraCI (zero-simulation constraint)."""
import subprocess
import sys
from pathlib import Path

PKG_ROOT = str(Path(__file__).resolve().parents[2])

CODE = (
    "import sys\n"
    f"sys.path.insert(0, r'{PKG_ROOT}')\n"
    "import predict\n"
    "from predict import (\n"
    "    run_conformance, run_conformance_both_versions, broken_predictors,\n"
    "    safety_veto, ConstSpeedPredictor, validate_plan,\n"
    "    validate_state_samples, validate_prediction,\n"
    "    native_reference, native_unlocked_reference, SUPPORTED_CONTRACT_VERSIONS,\n"
    "    Predictor,\n"
    ")\n"
    "for banned in ('torch', 'sumolib', 'traci', 'ray'):\n"
    "    assert banned not in sys.modules, f'predict pulled in {banned}'\n"
    "print('no-torch OK')\n"
)


def test_package_imports_without_torch():
    proc = subprocess.run(
        [sys.executable, "-c", CODE],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(PKG_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "no-torch OK" in proc.stdout
