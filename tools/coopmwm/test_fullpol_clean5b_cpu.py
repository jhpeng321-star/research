"""clean5b CPU E2E: FULLPOL templates + descriptor on LCEXEC's per-CAV free_program protocol.

Real EpisodePolicy / TriggerPolicy / feasible slates / v2v_runtime controller / lc_families protocol /
lc_program executors on the REAL N1 wz3to2 descriptor (LCEXEC fixtures). A CAV group chooses a FULLPOL
wall_fill program: the initiator changes lane INTO the closing lane and slot-tracks two joiners that hold
15 m/s; the decision rows then receive BC (initiate/join contexts), PPO and the (offline) EV labels.
No SUMO: the env writer is the LCEXEC Capture stand-in.
"""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import json

import numpy as np
import pytest
import torch
from torch import nn

from . import fullpol_head as fh, fullpol_learn as fl, lc_families as lf, lc_program as lp, track2_backend as tb
from .test_lc_families_cpu import Capture, controller, e2e_rows, make, step, wz  # noqa: F401
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)
from .trigger_policy import ppo_update
from tools.coopnative.policy import encode

PROV = 'FULLPOL clean5b CPU fixture (LCEXEC contract 2026-11-06)'
GEN = 'tools.coopmwm.fullpol_programs:propose'
DESC = 'tools.coopmwm.fullpol_head:program_descriptor'


def b_cfg(cfg, *, classes=('structured',), structured=('wall_fill',), cap=3):
    out = dict(cfg, trigger_partner_slots=4, trigger_candidates=4, group_lane_moves=True)
    out.update(lf.config_keys(provenance=PROV, cap=cap, generator=GEN,
                              generator_args=dict(classes=list(classes), structured=list(structured)),
                              descriptor=DESC, descriptor_width=fh.PROGRAM_DESCRIPTOR_WIDTH))
    return out


class PreferPrograms(nn.Module):
    """Test-only head on FULLPOL's descriptor block: initiate > join > anything else."""

    def __init__(self, trigger):
        super().__init__()
        hidden = trigger.base_policy.actor[-1].in_features
        base = len(trigger.families)+trigger.partner_slots+3
        start = hidden+base+trigger.extension.descriptor_width-fh.PROGRAM_DESCRIPTOR_WIDTH
        group = start+fh.TEMPLATE_CAP*len(fh.TEMPLATE_SLOT_FEATURES)
        self.initiate, self.join = group+0, group+1

    def forward(self, x):
        return (50.*x[..., self.initiate]+40.*x[..., self.join])[..., None]


def run(p, c, times):
    samples, requests = {}, {}
    for t in times:
        env = Capture()
        s, _ = step(p, c, e2e_rows(t), t, env)
        samples[t], requests[t] = s, (env.ticks[-1] if env.ticks else None)
    return samples, requests


def test_settings_bind_fullpol_generator_and_descriptor(cfg, wz):
    s = lf.settings(b_cfg(cfg))
    assert s['propose'].__name__ == 'propose' and s['descriptor'].__name__ == 'program_descriptor'
    assert s['descriptor_width'] == fh.PROGRAM_DESCRIPTOR_WIDTH
    p = make(b_cfg(cfg))
    assert p.trigger.option_families[-2:] == lf.PROGRAM_FAMILIES       # LCEXEC is the only registrant
    hidden = p.trigger.base_policy.actor[-1].in_features
    w = p.trigger.trigger_head[0].weight
    block = w[:, hidden+len(p.trigger.families)+p.trigger.partner_slots+3:]
    assert torch.equal(block, torch.zeros_like(block))                # zero-initialised descriptor columns


def test_e2e_group_chooses_lane_change_into_closing_lane_and_learns(cfg, wz):
    p = make(b_cfg(cfg))
    p.trigger.trigger_head = PreferPrograms(p.trigger)
    c = controller(p)
    samples, requests = run(p, c, range(300, 306))
    s = samples[300]
    trig = s['track2_trigger']
    i = trig['context']['members'].index('c0')
    chosen = trig['tables'][i][trig['actions'][i]]
    assert chosen['family'] == 'free_program_initiate' and trig['accepted']['c0']
    template = lp.FreeProgram.from_dict(chosen['template'])
    assert template.source['family'] == 'wall_fill' and fl.option_labels(chosen, b_cfg(cfg)) == ('wall_fill',)
    filler = template.programs[0]
    assert filler.lane.target == 2 and filler.lon.kind == 'track' and filler.lon.refs == ('@1', '@2')
    assert [r['lane'] for r in template.join_rules[1:]] == [1, 0]
    # joiners receive the t-1 intent and join at their rule slots; the filler requests the closing lane
    offers = samples[301]['track2']['lc_families']['offers']
    assert ['free_program:c0@300', 2] in offers['c1']                  # c1 (lane 0) matches slot 2's rule
    assert samples[301]['track2_trigger']['accepted']['c1']
    session = p.lc_families.sessions.get('free_program:c0@300') or {}
    joined = {e['vid'] for t in (301, 302) for e in samples[t]['track2']['lc_families'].get('joins', [])}
    assert 'c1' in joined or 'c1' in session.get('members', {})
    lanes = [r.get('lanes', {}) for r in requests.values() if r]
    assert any(ln.get('c0') == [2, 2.0] for ln in lanes)                # lane change INTO the closing lane
    accels = [r.get('accel', {}) for r in requests.values() if r]
    assert any('c1' in a or 'c2' in a for a in accels)                 # speed changes of the joiners
    for r in requests.values():
        for knob in (r or {}).values():
            assert not {'h1', 'h2'} & set(knob)                        # HDVs never commanded
    # PPO on the recorded program decision (exact replay of the recorded slate)
    record = json.loads(json.dumps(trig))
    batch = tb._batch(record)
    logp, _, _ = p.trigger.evaluate(batch)
    assert logp.tolist() == pytest.approx(record['logp'], abs=1e-5)
    p2 = make(b_cfg(cfg))                                              # a trainable (MLP) head
    batch2 = tb._batch(record)
    p2_logits = p2.trigger.distribution(batch2.context, batch2.descriptors, batch2.mask)[0]
    batch2 = tb.TriggerBatch(batch2.context, batch2.tables, batch2.descriptors, batch2.mask, batch2.actions,
                             torch.distributions.Categorical(logits=p2_logits).log_prob(batch2.actions).detach(),
                             batch2.values)
    before = {k: v.clone() for k, v in p2.trigger.state_dict().items()}
    m = ppo_update(p2.trigger, p2.trigger_optimizer, batch2, [1.]*len(batch2.actions), [0.]*len(batch2.actions),
                   **cfg['trigger_ppo'])
    assert np.isfinite(m['loss']) and any(not torch.equal(before[k], v) for k, v in p2.trigger.state_dict().items())


def test_bank_bc_initiate_and_join_contexts_from_a_verified_program(cfg, wz):
    """A verified concrete program (lc_convert shape) teaches c0 to INITIATE at t0 and c2 / c1 to
    JOIN at their true slots when their own receive step offers the reconstructed session."""
    p = make(b_cfg(cfg))
    p.trigger.trigger_head = PreferPrograms(p.trigger)
    c = controller(p)
    samples, _ = run(p, c, range(300, 302))
    template = lp.FreeProgram.from_dict(samples[300]['track2_trigger']['tables'][0][
        samples[300]['track2_trigger']['actions'][0]]['template'])
    concrete = lf.resolve(template.programs[0], {0: 'c0', 1: 'c2', 2: 'c1'})
    progs = tuple(lf.resolve(pp, {0: 'c0', 1: 'c2', 2: 'c1'}) for pp in template.programs)
    verified = lp.FreeProgram(('c0', 'c2', 'c1'), progs, template.map, (0, 0, 0), template.duration_s,
                              dict(family='wall_fill', verification_id='fp_wall_fill_v1'))
    assert concrete == progs[0]
    frames = {t: dict(vehicles=e2e_rows(t)) for t in range(299, 306)}
    pairs = []
    for t in range(300, 306):
        cav_rows = [r for r in e2e_rows(t)]
        context = encode(cav_rows)
        context['v2v'] = {v: [0.]*48 for v in context['members']}
        pairs.append((dict(context=context, intent_link=dict(t=t)), 1.))
    learner = make(b_cfg(cfg))
    plan = dict(free_program=verified.to_dict(), members=list(verified.members), duration_s=verified.duration_s)
    got = list(fl.program_protocol_contexts(learner, plan, pairs, frames, t0=300))
    labels = [(o.family, o.members[0], o.slot, t) for _, o, _, _, t in got]
    assert labels[0] == ('free_program_initiate', 'c0', 0, 300)
    assert sorted(labels[1:]) == [('free_program_join', 'c1', 2, 301), ('free_program_join', 'c2', 1, 301)]
    ctx, option, reader, _, _ = got[0]
    batch, same = fl.supervised_batch(learner, cfg|dict(training_seed=123), ctx, option, reader)
    assert int(batch.actions[0]) == 1 and bool(same[0, 1])
    rec = dict(batch=batch, weight=1., equivalent=same, meta=dict(kind='program', seed=989220))
    out = fl.bc_steps(learner, [rec], epochs=5, kl_budget=.045, max_grad_norm=.5)
    assert out['final_ce'] < out['initial_ce']
