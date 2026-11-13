"""Amendment 77 (S-MAPPO20C) training bundle: zero-SUMO, zero-training checks of the staged code tree.

Run from anywhere:  A77_CODE=<code root> python -m pytest -q ops/a77/train/test_a77_train.py
(default code root /home/ubuntu/a77/code = the training box). Checks: the A77 config = the A65 config with exactly the
three registered deltas; the A77 map adapter accepts exactly the fresh TRAIN block 987200-987599 and refuses the old
block, HELD-OUT and forbidden seeds; the sanity panel; the PLAN the launcher writes on the box (predicted bytes, sha
pinned in AMENDMENT 77); speed-only init (u0 export without a lane head) when the init file is staged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

CODE = Path(os.environ.get('A77_CODE', '/home/ubuntu/a77/code')).resolve()
BOX = '/home/ubuntu/a77'
A65_CFG = 'configs/c4_mappo_speed_wz3to2_d2800p60.toml'
A77_CFG = 'configs/c4_mappo_speed_wz3to2_d2800p20_a77.toml'
A65_CFG_SHA256_LF = '191be85cf34f275de6b396551a539e3420e42ec158d4fa2770a1ab3f0afd3e96'
A77_CFG_SHA256_LF = '76dc251b9d12a6372b31a283c4b21b29773855d93bcf2e849ab32432f1b4283f'
EXPECTED_PLAN_SHA256 = '57f51364e2d56f89539d483fb445704c2bbb6e7fce8a824626082e25eb3d9f5d'
SANITY_PANEL_SHA256 = '439d0c2dc55f2b03079de220cc276a075bcf1c8309ea1980003d504aeb5ff92d'
A65_SANITY_PANEL_SHA256 = 'cb64b674d80c91d5ca2698ae8ef5dd4a382cda74d992fd40e4c811fcace6b71b'
U0_SHA256 = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
PLAN_ARGS = dict(option='ft84_u0', host='cpuV', workers=None, wall_cap_h=30.0, phase_cap_sim_s=3_560_000.0,
                 other_charges_sim_s=54_000.0)

if not (CODE/A77_CFG).is_file():
    pytest.skip(f'no staged A77 code tree at {CODE} (set A77_CODE)', allow_module_level=True)
for _p in (str(CODE/'src'), str(CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from wms.core.toml import loads as toml_loads  # noqa: E402  (the trainer's own TOML reader)


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def flat(d, pre=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre+(k,)))
        else:
            out[pre+(k,)] = v
    return out


def test_config_is_a65_plus_exactly_the_registered_deltas():
    assert sha_lf(CODE/A65_CFG) == A65_CFG_SHA256_LF and sha_lf(CODE/A77_CFG) == A77_CFG_SHA256_LF
    a65 = flat(toml_loads((CODE/A65_CFG).read_text(encoding='utf-8')))
    a77 = flat(toml_loads((CODE/A77_CFG).read_text(encoding='utf-8')))
    assert set(a65) == set(a77)
    diff = {k: (a65[k], a77[k]) for k in a65 if a65[k] != a77[k]}
    assert diff == {('map', 'api'): ('tools.nm_mappo.n1_adapter:descriptor', 'tools.nm_mappo.n1_adapter_a77:descriptor'),
                    ('rollout', 'penetrations'): ([0.6], [0.2]),
                    ('rollout', 'seeds'): ('989200-989599', '987200-987599')}
    assert a77[('policy', 'lane_control')] == 'native' and a77[('rollout', 'demands_vph')] == [2800]
    assert a77[('rollout', 'episodes_per_iteration')] == 46 and a77[('rollout', 'keep_every')] == 6


def test_adapter_binds_only_the_fresh_block():
    from wms.control.mappo.mapbind import apply_map, MapBindingError
    from tools.coopmwm import map_scenarios as ms
    cfg = toml_loads((CODE/A77_CFG).read_text(encoding='utf-8'))
    resolved, b = apply_map(cfg)
    rec = b.record()
    assert rec['train_seeds'] == '987200-987599' and rec['heldout_seeds'] == '989600-989699'
    assert rec['env_name'] == 'tools.nm_mappo.n1_adapter_a77.make_env' and not rec['is_stub']
    assert rec['spec_sha256'].startswith('6fc6be65') and rec['extra']['network_file_sha256'].startswith('b6277898')
    assert resolved['rollout']['penetrations'] == [0.2] and resolved['rollout']['demands_vph'] == [2800]
    assert ms.world(987599, 2800, 0.2)['cell_key'] == 'D2800_p20'
    for bad in (989200, 989599, 989600, 964000, 962500, 977000, 987199, 987600):
        if bad != 989600:                                  # HELD-OUT exists as a world; mapbind refuses it below
            with pytest.raises(ValueError):
                ms.check_case(bad, 2800)
        with pytest.raises(MapBindingError):
            b.check_seeds([bad])


def test_sanity_panel_is_the_last_12_seeds_of_the_new_block():
    spec = importlib.util.spec_from_file_location('a77_sanity_eval', CODE/'tools/nm_mappo/sanity_eval.py')
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    _, binding, cells = S.resolve(S.load_cfg(CODE/A77_CFG), fake=False)
    panel = S.make_panel(binding, cells)
    assert [w['seed'] for w in panel] == list(range(987588, 987600))
    assert {(w['demand_vph'], w['penetration']) for w in panel} == {(2800, 0.2)}
    assert S.panel_sha256(panel) == SANITY_PANEL_SHA256


def _plan_doc():
    spec = importlib.util.spec_from_file_location('a77_launch', CODE/'tools/nm_mappo/launch.py')
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    plan = L.make_plan(run_dir=Path(f'{BOX}/run84'), init=f'{BOX}/inputs/mappo_u0_policy_00169.pt',
                       python=f'{BOX}/venv/bin/python', sumo_home=f'{BOX}/venv/lib/python3.12/site-packages/sumo',
                       config=CODE/A77_CFG, ledger=f'{BOX}/ledger/CHARGES.jsonl', allow_stub=False, **PLAN_ARGS)
    if str(CODE) == f'{BOX}/code':
        return plan
    lap, bs = str(CODE), chr(92)

    def fix(o):                                           # laptop copy: the box's code root in every path string
        if isinstance(o, str):
            if o.startswith(lap):
                return (f'{BOX}/code'+o[len(lap):]).replace(bs, '/')
            return o.replace(bs, '/') if o.replace(bs, '/').startswith(BOX) else o
        if isinstance(o, list):
            return [fix(x) for x in o]
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        return o
    return dict(fix(plan), source_commit=None)


def test_plan_bytes_are_the_registered_plan():
    doc = _plan_doc()
    assert doc['seeds'] == '987200-987599' and doc['iterations'] == 84 and doc['workers'] == 46
    assert doc['budget_sim_s']['planned'] == 3_477_600 and doc['max_sumo_seconds_controlled'] == 2_318_400
    assert doc['snapshots_preregistered'] == [24, 42, 84] and doc['init']['sha256'] == U0_SHA256
    assert doc['argv']['production'][-1] == '2318400' and '--update-threads' in doc['argv']['production']
    assert hashlib.sha256(json.dumps(doc, indent=1).encode()).hexdigest() == EXPECTED_PLAN_SHA256


def test_staged_u0_is_a_speed_only_export():
    init = Path(os.environ.get('A77_INIT', f'{BOX}/inputs/mappo_u0_policy_00169.pt'))
    if not init.is_file():
        pytest.skip('u0 init not staged here')
    assert hashlib.sha256(init.read_bytes()).hexdigest() == U0_SHA256
    import torch
    payload = torch.load(init, map_location='cpu', weights_only=False)
    assert not any(str(k).startswith('lane_head.') for k in payload['policy'])
