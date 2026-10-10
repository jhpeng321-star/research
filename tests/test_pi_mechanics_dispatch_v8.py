"""PI mechanics stays opt-in through the supported entry point."""
import sys
from types import SimpleNamespace

import pytest

from traffic_shepherd import mechanics


def test_pi_dispatch_passes_registered_inputs(monkeypatch, tmp_path):
    calls = []
    module = SimpleNamespace(run_registered=lambda *args: calls.append(args) or {'status': 'complete'})
    monkeypatch.setitem(sys.modules, 'output.controller_verification_v8.pi_mechanics', module)
    assert mechanics.main(['--candidate', 'pi-external-u', '--registration', 'frozen.json',
                           '--output', str(tmp_path), '--sumo-binary', 'sumo.exe']) == 0
    assert calls == [('frozen.json', str(tmp_path), 'sumo.exe')]


def test_pi_requires_registration_before_import(tmp_path):
    with pytest.raises(SystemExit) as error:
        mechanics.main(['--candidate', 'pi-external-u', '--output', str(tmp_path)])
    assert error.value.code == 2
