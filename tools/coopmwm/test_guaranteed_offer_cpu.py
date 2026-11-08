"""c4/guaranteed-offer (MAIN 2026-11-09): verified-only slate; every gated bank template is proposed (B = R), own
row + own sensing gate only; run-config registration via clean5b_restart.with_slate."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import clean5b_restart as c5b, fullpol_templates as ft
from .test_lc_families_cpu import wz  # noqa: F401
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401

DOC = Path(__file__).resolve().parents[2]/'clean5c'/'c4sit'/'GUIDANCE_C4SIT_priv.json'


def _state(monkeypatch, entries, guaranteed, rows):
    fp_cls = type('FP', (), {'from_dict': staticmethod(lambda d: d)})
    monkeypatch.setattr(ft, '_lc', lambda: type('LC', (), {'FreeProgram': fp_cls}))
    monkeypatch.setattr(ft, '_view', lambda frame: None)
    slot = dict(classes=['verified'], structured=[], templates=dict(B=entries, R=deepcopy(entries)),
                **({'guaranteed_offer': True} if guaranteed else {}))
    return dict(frame=None, rows=rows, fullpol=slot, now=500, ledger={}, arm='B')


def _r(v, s, ln, cav=True):
    return dict(vid=v, s_m=float(s), lane=ln, v_mps=20., is_cav=cav)


def _entries():
    cond = dict(trigger='lanes', own_wm_lanes=[0], zone_s=[-1800., -400.], spread_m=300.,
                roles=[dict(wm_lane=0), dict(wm_lane=1)], gate=None, start_gate=None, slots=1, cooldown_s=120,
                situation=dict(rule='x', mode='privileged', bins=[]))   # privileged: never a slate condition
    far = dict(cond, zone_s=[-200., 0.])
    return [dict(template={'source': dict(variant=f)}, offer_condition=c)
            for f, c in (('m_b', cond), ('m_a', cond), ('m_far', far))]


def test_guaranteed_offer_proposes_every_gated_template_without_a_draw(monkeypatch):
    own = _r('e', -1000, 0)
    rows = [own, _r('p', -1100, 1)]
    rng = np.random.default_rng(0)
    plain = ft.propose(_state(monkeypatch, _entries(), False, rows), own, rng)
    assert len(plain) == 1 and plain[0]['source']['variant'] in ('m_a', 'm_b')            # one random draw
    rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
    got = ft.propose(_state(monkeypatch, _entries(), True, rows), own, rng1)
    assert [t['source']['variant'] for t in got] == ['m_a', 'm_b']                         # all gated, sorted
    rng2.permutation(1)                                                                    # only the class order
    assert rng1.bit_generator.state == rng2.bit_generator.state                            # no template draw
    assert ft.propose(_state(monkeypatch, _entries(), True, [own]), own, np.random.default_rng(0)) == []


def test_with_slate_registers_verified_only_guaranteed_b_equals_r(cfg, wz):
    from .lc_families import CONFIG_KEY
    doc = json.loads(DOC.read_bytes())
    plain = c5b.keys_c(desc=wz, guidance_doc=doc, provenance='t')
    out = c5b.with_slate(c5b.keys_c)(desc=wz, guidance_doc=doc, provenance='t')
    args = out[CONFIG_KEY]['generator_args']
    assert args['classes'] == ['verified'] and args['structured'] == [] and args['guaranteed_offer'] is True
    assert args['templates'] == plain[CONFIG_KEY]['generator_args']['templates'] and args['templates']['B'] == args['templates']['R']
    assert {k: v for k, v in out.items() if k != CONFIG_KEY} == {k: v for k, v in plain.items() if k != CONFIG_KEY}
