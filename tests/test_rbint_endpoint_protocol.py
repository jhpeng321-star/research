from pathlib import Path
import subprocess
import sys

from tools.rbeval.run_wm import arguments


def test_endpoint_flag_is_default_off_and_has_separate_model_identity(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_wm', '--out', 'synthetic'])
    normal = arguments()
    assert normal.model_name == 'wm2' and not normal.endpoint_only
    monkeypatch.setattr(sys, 'argv', ['run_wm', '--out', 'synthetic', '--endpoint-only',
                                    '--model-name', 'wm2_hdv60'])
    short = arguments()
    assert short.endpoint_only and short.model_name == 'wm2_hdv60'
    assert short.wm_config == normal.wm_config and short.runtime_root == normal.runtime_root


def test_baseline_wrapper_does_not_preimport_the_integrated_model_source():
    wrapper = Path(__file__).resolve().parents[1]/'tools/rbint/evaluate_WM2_60.py'
    code = f"import runpy,sys;runpy.run_path({str(wrapper)!r},run_name='import_only');assert not any(x=='wms' or x.startswith('wms.') for x in sys.modules)"
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
