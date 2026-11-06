"""clean5a blockers 1+2 (MAIN 2026-11-06): bound frozen-trunk pin for training workers; A = frozen base."""
import hashlib
import types

import pytest

from tools.coopmwm import frozen_trunk as ft, track2_backend as tb


def _fake_backend(default_sha):
    """A stand-in leverprobe backend whose setup pins MAPPO_SHA through its module globals."""
    module = types.ModuleType('fake_leverprobe_backend')
    module.MAPPO_SHA = default_sha
    module.calls = []
    code = ('def setup(runtime, checkpoint):\n'
            '    import hashlib\n'
            '    if hashlib.sha256(open(checkpoint, "rb").read()).hexdigest() != MAPPO_SHA:\n'
            '        raise ValueError("frozen source/checkpoint hash mismatch")\n'
            '    calls.append((runtime, checkpoint, MAPPO_SHA))\n'
            '    return "ok"\n')
    exec(code, module.__dict__)
    return module


def _files(tmp_path):
    old, new = tmp_path/'policy_00169.pt', tmp_path/'policy_00006.pt'
    old.write_bytes(b'u0 trunk'); new.write_bytes(b'mappo wz snapshot')
    return old, new, hashlib.sha256(b'u0 trunk').hexdigest(), hashlib.sha256(b'mappo wz snapshot').hexdigest()


def test_unbound_is_the_original_call(tmp_path):
    old, new, old_sha, _ = _files(tmp_path)
    backend = _fake_backend(old_sha)
    assert ft.setup('/rt', str(old), backend=backend, environ={}) == 'ok'
    assert backend.calls == [('/rt', str(old), old_sha)] and backend.MAPPO_SHA == old_sha
    with pytest.raises(ValueError):                       # default pin still refuses another trunk
        ft.setup('/rt', str(new), backend=backend, environ={})
    assert ft.DEFAULT_SHA256.startswith('d9042cc5')


def test_bound_new_sha_loads_only_that_file(tmp_path):
    old, new, old_sha, new_sha = _files(tmp_path)
    backend = _fake_backend(old_sha)
    env = {ft.ENV: f'{new}|{new_sha}'}
    assert ft.setup('/rt', str(new), backend=backend, environ=env) == 'ok'
    assert backend.calls[-1] == ('/rt', str(new), new_sha)
    assert backend.MAPPO_SHA == old_sha                   # module pin never mutated
    with pytest.raises(ValueError):                       # a different path than the bound base
        ft.setup('/rt', str(old), backend=backend, environ=env)
    new.write_bytes(b'tampered')
    with pytest.raises(ValueError):                       # bytes verified at every load
        ft.setup('/rt', str(new), backend=backend, environ=env)


def test_bound_default_sha_uses_original(tmp_path):
    old, _, old_sha, _ = _files(tmp_path)
    backend = _fake_backend(old_sha)
    assert ft.setup('/rt', str(old), backend=backend, environ={ft.ENV: f'{old}|{old_sha}'}) == 'ok'


@pytest.mark.parametrize('raw', ['relative.pt|'+'a'*64, '/abs.pt|ABC', '/abs.pt', '/a|b|'+'a'*64])
def test_malformed_binding_raises(raw):
    with pytest.raises(ValueError):
        ft.binding({ft.ENV: raw})


def test_config_record_and_process_check(tmp_path, monkeypatch):
    _, new, _, new_sha = _files(tmp_path)
    record = ft.config_record(str(new), new_sha, 'MAIN GO')
    monkeypatch.delenv(ft.ENV, raising=False)
    assert ft.check_process({}) is None
    with pytest.raises(ValueError):
        ft.check_process(dict(frozen_trunk=record, checkpoint=str(new)))
    monkeypatch.setenv(ft.ENV, f'{new}|{new_sha}')
    assert ft.check_process(dict(frozen_trunk=record, checkpoint=str(new)))['sha256'] == new_sha
    with pytest.raises(ValueError):
        ft.check_process({})
    with pytest.raises(ValueError):
        ft.config_record(str(new), '0'*64, 'x')


def test_frozen_control_arm_rule():
    assert not tb.frozen_control_arm({})
    assert tb.frozen_control_arm(dict(frozen_control_arm=tb.FROZEN_A_RULE))
    with pytest.raises(ValueError):
        tb.frozen_control_arm(dict(frozen_control_arm='A frozen'))


def test_frozen_a_train_makes_no_update(monkeypatch):
    stub = types.SimpleNamespace(cfg=dict(frozen_control_arm=tb.FROZEN_A_RULE), arm='A', update=3,
                                 base=object(), optimizer=object(), generator=object())
    import tools.coopmwm.sparse_synergy_credit as ssc
    monkeypatch.setattr(ssc, 'trigger_terms', lambda *a, **k: (None, None))
    import tools.coopmwm.accel_learner as al
    monkeypatch.setattr(al, 'update', lambda *a, **k: pytest.fail('frozen A must not run PPO'))
    out = tb.ArmRuntime.train(stub, [dict(arm='A', id='e', steps=[])])
    assert out['ppo']['status'] == 'frozen_base_no_update' and out['bank'] == dict(status='A_no_bank')
