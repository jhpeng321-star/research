"""c4/imit-kl-option: the registered imitation_kl_limit option (default .045 = branch point).

Default OFF is byte-identical: the GOLDEN reports below were captured from the pre-change
module (c4/v2v-resume branch point, same interpreter/BLAS/CPU); a default run must equal its
golden plus ONLY the per-update safety-note keys. run_separate reads the runtime binding
installed by v2v_live.hooks (explicit arguments win; registered alternatives .045/.06/.09
only). The configured limit is never exceeded (acceptance, halving backtracking and the
kl_targeted growth threshold = limit/2 all read it). v2v_launch3 pins it per runtime
(B == R enforced, A refused, unregistered values refused; a pre-option runtime without the
key keeps .045). learningwatch checks every receipt against the RUNTIME limit pinned in the
SHA-pinned V2V_RUNTIME.json (passed into inspect_update by watch): a receipt kl_limit that
differs from the runtime value FAILS ('imitation_kl_limit_mismatch'), the KL-breach bound is
the runtime limit (never the receipt's own claim), and an unregistered limit fails closed.
Safety note: the report adds total_kl_vs_pre_update = exact joint KL of the published policy
vs the frozen PRE-UPDATE policy (PPO + imitation combined), report-only; when the runtime
limit exceeds .045 it also carries the boolean total_kl_above_ppo_stop_plus_limit
(total_kl_vs_pre_update > PPO stop + limit), report-only and surfaced by the monitor.
"""
import copy
import json
import math

import pytest
import torch

from tools.coopmwm import bank_imitation as bi
from tools.coopmwm import bank_statistics as bs
from tools.coopmwm import v2v_launch3 as launch
from tools.coopmwm import v2v_learning_watch as lw
from tools.coopmwm import v2v_runner
from tools.coopmwm.test_bank_imitation_cpu import Policy, pairs

SAFETY_NOTE_KEYS = ('total_kl_vs_pre_update', 'total_kl_reference', 'safety_note')

# Branch-point default-mode reports (pre-change module, identical seeds/banks/interpreter).
GOLDEN = {
 'fixed_default': '{"action_error_after_mps2": 0.0950036272406578, "action_error_before_mps2": 0.10000000149011612, "action_error_before_update_mps2": 0.10000000149011612, "backtracks": 0, "critic_steps": 0, "epochs": [{"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.1659650057554245, "awr_loss_before": 0.1672041118144989, "epoch": 1, "kl": 1.0159342309634667e-05, "l1_action_error_after": 0.0989999994635582, "l1_action_error_before": 0.10000000149011612, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 1.0, "step_scale": 1.0}, {"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.164727583527565, "awr_loss_before": 0.1659650057554245, "epoch": 2, "kl": 4.04028469347395e-05, "l1_action_error_after": 0.09800021350383759, "l1_action_error_before": 0.0989999994635582, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 1.0, "step_scale": 1.0}, {"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.16349183022975922, "awr_loss_before": 0.164727583527565, "epoch": 3, "kl": 9.12342220544815e-05, "l1_action_error_after": 0.09700079262256622, "l1_action_error_before": 0.09800021350383759, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 1.0, "step_scale": 1.0}, {"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.16225793957710266, "awr_loss_before": 0.16349183022975922, "epoch": 4, "kl": 0.0001624343713046983, "l1_action_error_after": 0.09600187838077545, "l1_action_error_before": 0.09700079262256622, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 1.0, "step_scale": 1.0}, {"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.16102582216262817, "awr_loss_before": 0.16225793957710266, "epoch": 5, "kl": 0.0002541303401812911, "l1_action_error_after": 0.0950036272406578, "l1_action_error_before": 0.09600187838077545, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 1.0, "step_scale": 1.0}], "epochs_completed": 5, "epochs_started": 5, "imitation_failed": false, "kl": 0.0002541303401812911, "kl_limit": 0.045, "kl_reference": "frozen pre-imitation (post-PPO) policy", "last_gradient_norm": 1.008791446685791, "last_loss": 0.16225793957710266, "last_step_scale": 1.0, "member_actions_seen": 400, "min_additional_pass_step_scale": 0.015625, "optimizer_steps": 5, "own_policy_steps": 0, "pairs_filtered_unexecuted": 0, "pairs_seen": 200, "pairs_total": 40, "pass1_effective": true, "pass1_loss_drop": 0.0012391060590744019, "pass1_step_scale": 1.0, "plan_ratios_computed": 0, "rejected_optimizer_steps": 0, "requested_epochs": 5, "rule": "full-bank transactional passes; separate imitation KL; AWR weights unchanged", "stop_reason": "all registered imitation epochs complete"}',
 'targeted_default': '{"action_error_after_mps2": 0.07162124663591385, "action_error_before_mps2": 0.10000000149011612, "action_error_before_update_mps2": 0.10000000149011612, "backtracks": 0, "critic_steps": 0, "epochs": [{"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.16683219373226166, "awr_loss_before": 0.1672041416168213, "epoch": 1, "kl": 9.06028446934215e-07, "l1_action_error_after": 0.09970001131296158, "l1_action_error_before": 0.10000000149011612, "optimizer_steps": 3, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 1.0, "step_scale": 1.0, "step_scale_mult": 1.0, "steps_mult": 1}, {"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.16559356451034546, "awr_loss_before": 0.16683219373226166, "epoch": 2, "kl": 1.71147366927471e-05, "l1_action_error_after": 0.0987003743648529, "l1_action_error_before": 0.09970001131296158, "optimizer_steps": 5, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 2.0, "step_scale": 2.0, "step_scale_mult": 2.0, "steps_mult": 2}, {"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.16065815091133118, "awr_loss_before": 0.16559356451034546, "epoch": 3, "kl": 0.00028475839644670486, "l1_action_error_after": 0.09471412748098373, "l1_action_error_before": 0.0987003743648529, "optimizer_steps": 10, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 4.0, "step_scale": 4.0, "step_scale_mult": 4.0, "steps_mult": 4}, {"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.15088701248168945, "awr_loss_before": 0.16065815091133118, "epoch": 4, "kl": 0.0017963413847610354, "l1_action_error_after": 0.08683886379003525, "l1_action_error_before": 0.09471412748098373, "optimizer_steps": 10, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 8.0, "step_scale": 8.0, "step_scale_mult": 8.0, "steps_mult": 4}, {"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 0.1317652016878128, "awr_loss_before": 0.15088701248168945, "epoch": 5, "kl": 0.008666282519698143, "l1_action_error_after": 0.07162124663591385, "l1_action_error_before": 0.08683886379003525, "optimizer_steps": 10, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 16.0, "step_scale": 16.0, "step_scale_mult": 16.0, "steps_mult": 4}], "epochs_completed": 5, "epochs_started": 5, "imitation_failed": false, "kl": 0.008666282519698143, "kl_limit": 0.045, "kl_reference": "frozen pre-imitation (post-PPO) policy", "last_gradient_norm": 1.0062614679336548, "last_loss": 0.1336510181427002, "last_step_scale": 16.0, "member_actions_seen": 3000, "min_additional_pass_step_scale": 0.015625, "optimizer_steps": 38, "own_policy_steps": 0, "pairs_filtered_unexecuted": 0, "pairs_seen": 1500, "pairs_total": 300, "pass1_effective": true, "pass1_loss_change": {"bootstrap": 10000, "ci95": [-0.0003719329833984375, -0.0003719329833984375], "mean": -0.0003719329833984375, "n": 300, "rule": "paired per-pair pass-1 AWR loss change after-before; percentile bootstrap over imitation pairs", "seed": 20261101, "significantly_better": true, "significantly_worse": false}, "pass1_loss_drop": 0.00037194788455963135, "pass1_step_scale": 1.0, "plan_ratios_computed": 0, "rejected_optimizer_steps": 0, "requested_epochs": 5, "rule": "full-bank transactional passes; separate imitation KL; AWR weights unchanged", "step_mode": "kl_targeted", "step_scale_cap": 16, "steps_cap": 4, "stop_reason": "all registered imitation epochs complete"}',
 'backtrack_default': '{"action_error_after_mps2": 0.8296633362770081, "action_error_before_mps2": 0.8999999761581421, "action_error_before_update_mps2": 0.8999999761581421, "backtracks": 16, "critic_steps": 0, "epochs": [{"accepted": true, "attempt_status": ["accepted"], "attempts": 1, "awr_loss_after": 1.0755308866500854, "awr_loss_before": 1.2338706254959106, "epoch": 1, "kl": 0.02313472144305706, "l1_action_error_after": 0.8499999642372131, "l1_action_error_before": 0.8999999761581421, "rejected_optimizer_steps": 0, "smallest_step_scale_tried": 1.0, "step_scale": 1.0}, {"accepted": true, "attempt_status": ["kl_exceeded", "kl_exceeded", "accepted"], "attempts": 3, "awr_loss_after": 1.0417782068252563, "awr_loss_before": 1.0755308866500854, "epoch": 2, "kl": 0.035360053181648254, "l1_action_error_after": 0.8374887704849243, "l1_action_error_before": 0.8499999642372131, "rejected_optimizer_steps": 2, "smallest_step_scale_tried": 0.25, "step_scale": 0.25}, {"accepted": true, "attempt_status": ["kl_exceeded", "kl_exceeded", "kl_exceeded", "accepted"], "attempts": 4, "awr_loss_after": 1.0257114171981812, "awr_loss_before": 1.0417782068252563, "epoch": 3, "kl": 0.04231691360473633, "l1_action_error_after": 0.831229031085968, "l1_action_error_before": 0.8374887704849243, "rejected_optimizer_steps": 3, "smallest_step_scale_tried": 0.125, "step_scale": 0.125}, {"accepted": true, "attempt_status": ["kl_exceeded", "kl_exceeded", "kl_exceeded", "kl_exceeded", "kl_exceeded", "accepted"], "attempts": 6, "awr_loss_after": 1.0217782258987427, "awr_loss_before": 1.0257114171981812, "epoch": 4, "kl": 0.044140275567770004, "l1_action_error_after": 0.8296633362770081, "l1_action_error_before": 0.831229031085968, "rejected_optimizer_steps": 5, "smallest_step_scale_tried": 0.03125, "step_scale": 0.03125}, {"accepted": false, "attempt_status": ["kl_exceeded", "kl_exceeded", "kl_exceeded", "kl_exceeded", "kl_exceeded", "kl_exceeded", "kl_exceeded"], "attempts": 7, "awr_loss_after": 1.0217782258987427, "awr_loss_before": 1.0217782258987427, "epoch": 5, "kl": 0.04506436362862587, "l1_action_error_after": 0.8296633362770081, "l1_action_error_before": 0.8296633362770081, "rejected_optimizer_steps": 7, "smallest_step_scale_tried": 0.015625, "step_scale": null}], "epochs_completed": 4, "epochs_started": 5, "imitation_failed": false, "kl": 0.044140275567770004, "kl_limit": 0.045, "kl_reference": "frozen pre-imitation (post-PPO) policy", "last_gradient_norm": 2.026031494140625, "last_loss": 1.0257114171981812, "last_step_scale": 0.03125, "member_actions_seen": 160, "min_additional_pass_step_scale": 0.015625, "optimizer_steps": 4, "own_policy_steps": 0, "pairs_filtered_unexecuted": 0, "pairs_seen": 80, "pairs_total": 20, "pass1_effective": true, "pass1_loss_drop": 0.1583397388458252, "pass1_step_scale": 1.0, "plan_ratios_computed": 0, "rejected_optimizer_steps": 17, "requested_epochs": 5, "rule": "full-bank transactional passes; separate imitation KL; AWR weights unchanged", "stop_reason": "pass 5 not accepted at any step scale >= 0.015625 of base (statuses [\'kl_exceeded\', \'kl_exceeded\', \'kl_exceeded\', \'kl_exceeded\', \'kl_exceeded\', \'kl_exceeded\', \'kl_exceeded\']); additional passes stopped"}',
}


def run_case(name):
    """One golden case through the REAL default path (no explicit kl_limit)."""
    hot = name == 'backtrack_default'
    p = Policy()
    if hot:
        with torch.no_grad():p.mean.fill_(1.)
    kw = dict(step_mode='kl_targeted') if name == 'targeted_default' else {}
    lr, bank = (1e-4, pairs(300)) if name == 'targeted_default' else \
               ((.05, pairs(20)) if hot else (1e-3, pairs(40)))
    return bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=lr), bank,
                           torch.Generator().manual_seed(7), copy.deepcopy(p), passes=5, **kw)


# ------------------------------------------------------- default OFF is byte-identical

@pytest.mark.parametrize('name', sorted(GOLDEN))
def test_default_report_is_the_branch_point_plus_only_the_safety_note(name):
    out = run_case(name)
    assert out['kl_limit'] == .045
    stripped = {k: v for k, v in out.items() if k not in SAFETY_NOTE_KEYS}
    assert json.dumps(stripped, sort_keys=True) == GOLDEN[name]
    if name == 'targeted_default':   # kl_targeted carries the per-update safety note
        assert math.isfinite(out['total_kl_vs_pre_update'])
        assert out['safety_note'].startswith('total_kl_vs_pre_update is the exact joint KL')
    else:                            # fixed/.045 default: the report equals the branch point EXACTLY
        assert json.dumps(out, sort_keys=True) == GOLDEN[name]


def test_default_matches_explicit_registered_limit_byte_for_byte():
    a, b = Policy(), Policy()
    first = bi.run_separate(a, torch.optim.Adam(a.parameters(), lr=1e-3), pairs(40),
                            torch.Generator().manual_seed(7), copy.deepcopy(a), passes=5)
    second = bi.run_separate(b, torch.optim.Adam(b.parameters(), lr=1e-3), pairs(40),
                             torch.Generator().manual_seed(7), copy.deepcopy(b), passes=5,
                             kl_limit=.045)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_safety_note_reports_total_kl_vs_the_frozen_pre_update_policy():
    # the safety note is carried whenever an imitation option deviates from its default
    bi.install(step_mode='kl_targeted')
    old = Policy();p = copy.deepcopy(old)
    out = bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=1e-3), pairs(20),
                          torch.Generator().manual_seed(7), old, passes=5)
    samples = [s for s, _ in pairs(20)]
    assert out['total_kl_vs_pre_update'] == bi.joint_kl(p, old, samples)
    assert out['total_kl_vs_pre_update'] >= out['kl'] >= 0.
    assert out['safety_note'].startswith('total_kl_vs_pre_update is the exact joint KL')
    empty = bi.run_separate(Policy(), torch.optim.Adam(Policy().parameters()), [],
                            torch.Generator().manual_seed(7), copy.deepcopy(Policy()), passes=5)
    assert empty['total_kl_vs_pre_update'] == 0.
    plain = Policy()                                     # explicit fixed/.045: no note, golden bytes
    plain = bi.run_separate(plain, torch.optim.Adam(plain.parameters(), lr=1e-3), pairs(40),
                            torch.Generator().manual_seed(7), copy.deepcopy(plain), passes=5,
                            step_mode='fixed')
    assert 'total_kl_vs_pre_update' not in plain and 'safety_note' not in plain


def test_total_kl_flag_above_ppo_stop_plus_limit_when_the_limit_exceeds_default():
    # c4/imit-kl-option gate: with a limit above .045 the report carries the report-only
    # boolean flag vs the PPO stopping bound (PPO_STOP_KL + limit); fixed/.045 never carries it.
    pre_update = Policy()                       # the caller's frozen PRE-UPDATE policy
    hot = Policy()
    with torch.no_grad():hot.mean.fill_(1.)
    out = bi.run_separate(hot, torch.optim.Adam(hot.parameters(), lr=.05), pairs(20),
                          torch.Generator().manual_seed(7), copy.deepcopy(pre_update), passes=5,
                          kl_limit=.06)
    assert out['total_kl_vs_pre_update'] > bi.PPO_STOP_KL + .06   # PPO-sized drift + imitation
    assert out['total_kl_above_ppo_stop_plus_limit'] is True
    plain = bi.run_separate(Policy(), torch.optim.Adam(Policy().parameters(), lr=1e-3), pairs(40),
                            torch.Generator().manual_seed(7), copy.deepcopy(Policy()), passes=5,
                            step_mode='fixed')
    assert 'total_kl_vs_pre_update' not in plain and 'total_kl_above_ppo_stop_plus_limit' not in plain
    empty = bi.run_separate(Policy(), torch.optim.Adam(Policy().parameters()), [],
                            torch.Generator().manual_seed(7), copy.deepcopy(Policy()), passes=5,
                            kl_limit=.09)
    assert empty['total_kl_above_ppo_stop_plus_limit'] is False
    assert empty['total_kl_vs_pre_update'] == 0.


# ------------------------------------------------------- the limit is read from the runtime

def test_limit_is_read_from_the_runtime_binding_and_explicit_argument_wins():
    bi.install(step_mode='fixed', intent_bank_term=True, kl_limit=.06)
    hot = Policy()
    with torch.no_grad():hot.mean.fill_(1.)
    bound = bi.run_separate(hot, torch.optim.Adam(hot.parameters(), lr=.05), pairs(20),
                            torch.Generator().manual_seed(7), copy.deepcopy(hot), passes=5)
    assert bound['kl_limit'] == .06
    explicit = bi.run_separate(hot, torch.optim.Adam(hot.parameters(), lr=.05), pairs(20),
                               torch.Generator().manual_seed(7), copy.deepcopy(hot), passes=5,
                               kl_limit=.045)
    assert explicit['kl_limit'] == .045


def test_runtime_limit_gives_real_extra_room_and_is_never_exceeded():
    hot = Policy()
    with torch.no_grad():hot.mean.fill_(1.)
    out = bi.run_separate(hot, torch.optim.Adam(hot.parameters(), lr=.05), pairs(20),
                          torch.Generator().manual_seed(7), copy.deepcopy(hot), passes=5,
                          kl_limit=.06)
    assert out['kl_limit'] == .06
    assert all(r['kl'] <= .06 for r in out['epochs'] if r['accepted'])
    assert out['kl'] <= .06
    # the extra room was genuinely used: fitting continued past the default .045 bound, and the
    # pass whose smallest attempt still exceeds .06 is refused exactly like .045 refuses its own
    assert out['kl'] > .045 and out['epochs'][3]['kl'] > .045
    assert not out['epochs'][4]['accepted'] and out['epochs'][4]['kl'] > .06
    assert out['backtracks'] < run_case('backtrack_default')['backtracks']


def test_unregistered_limits_refuse_everywhere():
    with pytest.raises(ValueError, match='imitation_kl_limit'):
        bi.install(step_mode='fixed', intent_bank_term=True, kl_limit=.05)
    with pytest.raises(ValueError, match='imitation_kl_limit'):
        bi.install(step_mode='fixed', intent_bank_term=True, kl_limit=True)
    with pytest.raises(ValueError, match='imitation_kl_limit'):
        bi.install(step_mode='fixed', intent_bank_term=True, kl_limit='0.06')
    assert (bi.RUNTIME_STEP_MODE, bi.RUNTIME_INTENT_BANK_TERM, bi.RUNTIME_KL_LIMIT) == ('fixed', True, .045)
    p = Policy()
    for bad in (.05, .07, math.nan, math.inf, '0.06', True):
        with pytest.raises(ValueError, match='imitation_kl_limit'):
            bi.run_separate(p, torch.optim.Adam(p.parameters()), pairs(5),
                            torch.Generator().manual_seed(7), copy.deepcopy(p), passes=5, kl_limit=bad)
    ok = bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=1e-4), pairs(5),
                         torch.Generator().manual_seed(7), copy.deepcopy(p), passes=5, kl_limit=.09)
    assert ok['kl_limit'] == .09


def test_kl_targeted_growth_threshold_reads_the_runtime_limit(monkeypatch):
    # one joint_kl per attempt: pass 1 ends at .026 (inside BOTH bounds, but below .06/2 only)
    calls = []
    def kl(*a):
        calls.append(1)
        return .026 if len(calls) == 1 else .01
    monkeypatch.setattr(bi, 'joint_kl', kl)
    rows = []
    for limit in (.045, .06):
        calls.clear()
        bi.install(step_mode='kl_targeted', intent_bank_term=True, kl_limit=limit)
        p = Policy()
        out = bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=1e-3), pairs(20),
                              torch.Generator().manual_seed(1), copy.deepcopy(p), passes=5)
        rows.append([r['step_scale_mult'] for r in out['epochs']])
    # identical accepted pass-1 KL (.026): under .045 it is above limit/2 (no growth); under .06
    # it is below limit/2, so the next pass grows. Growth then resumes once later passes end low.
    assert rows == [[1., 1., 2., 4., 8.], [1., 2., 4., 8., 16.]]


# ------------------------------------------------------- launcher / runtime / hooks (B == R)

def _cfg(tmp_path, **kw):
    cfg = v2v_runner.prepare_intent(dict(budget={}), root=tmp_path/'v2v', legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256='a'*64, **kw)
    live = dict(cfg, launch_authorized=True, phase_funding_authorized=True,
                budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))
    return cfg, live


def test_registered_defaults_and_alternatives(tmp_path):
    assert launch.BANK_OPTIONS['imitation_kl_limit'] == .045
    assert launch.IMITATION_KL_LIMITS == (.045, .06, .09)
    cfg, live = _cfg(tmp_path)
    assert cfg['imitation_kl_limit'] == .045
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v', source_root=tmp_path/'src') is live
    for limit in (.045, .06, .09):
        _, alt = _cfg(tmp_path, imitation_kl_limit=limit)
        assert launch.bind_runtime(dict(config=alt), root=tmp_path/'v2v', source_root=tmp_path/'src') is alt
        assert (launch.imitation_kl_limit_for(alt, 'B'), launch.imitation_kl_limit_for(alt, 'R')) == (limit, limit)


def test_launch3_refuses_unregistered_values_and_b_r_mismatch(tmp_path):
    _, live = _cfg(tmp_path)
    for bad in (.05, .07, None, True, '0.06', 0):
        with pytest.raises(launch.Refused, match='imitation_kl_limit'):
            launch.bind_runtime(dict(config=dict(live, imitation_kl_limit=bad)),
                                root=tmp_path/'v2v', source_root=tmp_path/'src')
        with pytest.raises(launch.Refused, match='imitation_kl_limit'):
            launch.imitation_kl_limit_for(dict(live, imitation_kl_limit=bad), 'B')
    for by_arm in (dict(B=.06, R=.045), dict(B=.045, R=.045, but=1),
                   dict(B=.06, R=.09), dict(B='0.06', R=.06)):
        with pytest.raises(launch.Refused, match='identical for B and R'):
            launch.imitation_kl_limit_for(dict(live, imitation_kl_limit=.06,
                                               imitation_kl_limit_by_arm=by_arm), 'B')
    ok = dict(live, imitation_kl_limit=.06, imitation_kl_limit_by_arm=dict(B=.06, R=.06))
    assert launch.bind_runtime(dict(config=ok), root=tmp_path/'v2v', source_root=tmp_path/'src') is ok
    for arm in ('A',):
        with pytest.raises(launch.Refused, match='only arms B and R'):
            launch.imitation_kl_limit_for(live, arm)


def test_pre_option_runtime_without_the_key_keeps_the_default(tmp_path):
    _, live = _cfg(tmp_path)
    legacy = dict(live);legacy.pop('imitation_kl_limit')
    assert launch.bind_runtime(dict(config=legacy), root=tmp_path/'v2v', source_root=tmp_path/'src') is legacy
    assert launch.imitation_kl_limit_for(legacy, 'B') == .045


def test_stale_imitation_kl_cap_label_is_gone_and_refused_when_inconsistent(tmp_path):
    # fix round: the pinned BANK_OPTIONS label is the runtime's imitation_kl_limit itself; the
    # old imitation_kl_cap=.045 label left BANK_OPTIONS (nothing reads it) and bind_runtime
    # refuses any runtime whose stale cap claim disagrees with its own imitation_kl_limit.
    assert 'imitation_kl_cap' not in launch.BANK_OPTIONS
    cfg, live = _cfg(tmp_path)
    assert 'imitation_kl_cap' not in cfg and 'imitation_kl_cap' not in live
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v', source_root=tmp_path/'src') is live
    for cap, limit in ((.045, .06), (.06, .045), (.045, .09), (.09, .06), (True, .045)):
        with pytest.raises(launch.Refused, match='imitation_kl_cap'):
            launch.bind_runtime(dict(config=dict(live, imitation_kl_cap=cap, imitation_kl_limit=limit)),
                                root=tmp_path/'v2v', source_root=tmp_path/'src')
    # a pre-option runtime that still carries the legacy label WITH the matching default binds
    old = dict(live);old.update(imitation_kl_cap=.045);old.pop('imitation_kl_limit')
    assert launch.bind_runtime(dict(config=old), root=tmp_path/'v2v', source_root=tmp_path/'src') is old
    agree = dict(live, imitation_kl_cap=.06, imitation_kl_limit=.06)
    assert launch.bind_runtime(dict(config=agree), root=tmp_path/'v2v', source_root=tmp_path/'src') is agree


def test_producer_refuses_non_registered_limits(tmp_path):
    with pytest.raises(ValueError, match='unregistered imitation_kl_limit'):
        _cfg(tmp_path, imitation_kl_limit=.05)
    with pytest.raises(ValueError, match='unregistered imitation_kl_limit'):
        _cfg(tmp_path, imitation_kl_limit='0.06')
    with pytest.raises(ValueError, match='unregistered imitation_kl_limit'):
        _cfg(tmp_path, imitation_kl_limit=True)
    assert _cfg(tmp_path, imitation_kl_limit=None)[0]['imitation_kl_limit'] == .045   # None = default pin


def test_hooks_bind_the_runtime_kl_limit_for_b_and_r(tmp_path, monkeypatch):
    from tools.coopmwm import v2v_live
    installed = []
    monkeypatch.setattr(bi, 'install',
                        lambda *, step_mode, intent_bank_term, kl_limit:
                        installed.append((step_mode, intent_bank_term, kl_limit)))
    monkeypatch.setattr(bs, 'install', lambda rb_, bank_rule='boot_v2', sources=None: None)
    cfg, _ = _cfg(tmp_path, imitation_kl_limit=.06)
    v2v_live.hooks(tmp_path, cfg)
    assert installed == [('kl_targeted', True, .06)]
    installed.clear()
    with pytest.raises(launch.Refused):
        v2v_live.hooks(tmp_path, dict(cfg, imitation_kl_limit_by_arm=dict(B=.06, R=.045)))
    assert installed == []


# ------------------------------------------------------- learningwatch reads the runtime limit

def receipt(**bank_extra):
    bank = dict(pairs_total=226, epochs_completed=5, requested_epochs=5, pass1_effective=True,
                pass1_loss_drop=.0085, imitation_failed=False, kl=.055, kl_limit=.06,
                total_kl_vs_pre_update=.061, v2v_intent=dict(before=dict(intent_loss=2.4),
                                                             after=dict(intent_loss=2.535)))
    bank.update(bank_extra)
    return dict(arm='B', update=2, bank_imitation=bank,
                exact_imitation=dict(imitation_filter=dict(empty_bank=False, imitated_cooperative_examples=8)),
                ppo=[dict(own_policy_steps=128, kl=0., loss=8., gradient_norm=7.)])


def test_monitor_checks_against_the_pinned_runtime_limit():
    report = lw.inspect_update(receipt(), .06)                # receipt agrees with the runtime pin
    assert report['learning_mechanics_ok'] and report['kl_limit'] == .06
    assert report['runtime_kl_limit'] == .06
    assert report['total_kl_vs_pre_update'] == .061
    mismatch = lw.inspect_update(receipt(kl=.0014), .045)     # receipt claims .06, runtime pins .045
    assert 'imitation_kl_limit_mismatch' in mismatch['failures'] and not mismatch['learning_mechanics_ok']
    over = lw.inspect_update(receipt(kl=.061), .06)           # breach of the RUNTIME limit fails
    assert 'imitation_kl_out_of_limit' in over['failures'] and not over['learning_mechanics_ok']
    both = lw.inspect_update(receipt(), .045)                 # .055 > .045: the breach dominates
    assert 'imitation_kl_out_of_limit' in both['failures'] and not both['learning_mechanics_ok']
    missing = lw.inspect_update(receipt(kl_limit=None), .06)  # no receipt limit: cannot agree
    assert 'imitation_kl_out_of_limit' in missing['failures']
    unregistered = lw.inspect_update(receipt(kl=.05, kl_limit=.05), .05)  # unregistered runtime value
    assert 'unregistered_imitation_kl_limit' in unregistered['failures']
    legacy = lw.inspect_update(receipt(kl=.0014, kl_limit=.045, total_kl_vs_pre_update=.002), .045)
    assert legacy['learning_mechanics_ok'] and legacy['kl_limit'] == .045
    # a receipt can never relax its own bound: kl .055 inside a self-claimed 1.0 is still a
    # breach of the runtime .045 pin (and the claim itself is a mismatch when it survives)
    relaxed = lw.inspect_update(receipt(kl=.055, kl_limit=1.0), .045)
    assert 'imitation_kl_out_of_limit' in relaxed['failures']


def test_watch_reader_extracts_and_validates_the_pinned_runtime_limit():
    assert lw.runtime_kl_limit(dict(config=dict(imitation_kl_limit=.06))) == .06
    assert lw.runtime_kl_limit(dict(config=dict())) == .045          # pre-option runtime default
    with pytest.raises(ValueError, match='imitation_kl_limit'):
        lw.runtime_kl_limit(dict(config=dict(imitation_kl_limit=.05)))
    with pytest.raises(ValueError, match='imitation_kl_limit'):
        lw.runtime_kl_limit(dict())
    with pytest.raises(ValueError, match='imitation_kl_limit'):
        lw.runtime_kl_limit('not a runtime value')


def test_monitor_derives_and_surfaces_the_total_kl_flag_report_only():
    # bank_imitation sets the flag when the runtime limit exceeds .045; the monitor surfaces it
    # and derives it for a receipt that predates the flag. Report-only: never in failures.
    derived = lw.inspect_update(receipt(total_kl_vs_pre_update=.2), .06)   # .2 > .045 + .06
    assert derived['total_kl_above_ppo_stop_plus_limit'] is True
    assert derived['reported_only'] == ['total_kl_above_ppo_stop_plus_limit']
    assert derived['learning_mechanics_ok']
    below = lw.inspect_update(receipt(total_kl_vs_pre_update=.1), .06)     # .1 <= .045 + .06
    assert below['total_kl_above_ppo_stop_plus_limit'] is False
    assert below['reported_only'] == [] and below['learning_mechanics_ok']
    explicit = lw.inspect_update(receipt(total_kl_vs_pre_update=.061,
                                         total_kl_above_ppo_stop_plus_limit=False), .06)
    assert explicit['total_kl_above_ppo_stop_plus_limit'] is False
    none = lw.inspect_update(receipt(total_kl_vs_pre_update=None), .06)   # no total in the receipt
    assert none['total_kl_above_ppo_stop_plus_limit'] is None and none['learning_mechanics_ok']


# ------------------------------------------------------- receipt path carries the safety note

def test_exact_update_receipt_embeds_the_safety_note(tmp_path, monkeypatch):
    from tools.coopmwm import exact_grade as xg
    bi.install(step_mode='kl_targeted')                      # the deployed V2V runtime (owner 08:52)
    monkeypatch.setattr(bi, 'run', bi.run_separate)          # the deployed V2V binding (hooks)
    step = dict(sample=dict(context={'members': ['a', 'b']}, value=0., behaviour=dict(executed=True)), reward=0.)
    episodes = [dict(steps=[copy.deepcopy(step) for _ in range(3)])]
    p = Policy();opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    metrics = xg.exact_update(p, opt, episodes, torch.Generator().manual_seed(3), pairs(10),
                              imitation_epochs=5)
    report = metrics[-1]['bank_imitation']
    assert report['kl_reference'].startswith('frozen pre-imitation')
    assert math.isfinite(report['total_kl_vs_pre_update'])
    assert report['total_kl_reference'] == 'frozen pre-update policy (PPO + imitation combined)'
    json.dumps(metrics)
