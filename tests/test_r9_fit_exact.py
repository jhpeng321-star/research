"""Full-head fitter integration, causal batch replay and survivor conditioning."""
import copy
import json
import math

import numpy as np
import pytest
import torch

from tools import merge_fit_v2 as legacy
from tools import r9_exact as exact
from tools import r9_fit_exact as fit
from wms.model.controllers import DIRECT_ID
from wms.train.losses.merge_core_seqtf import build_recorded_scene


@pytest.fixture(scope='module')
def frozen():
    return exact.load_frozen()


def episode(n=8):
    kin = torch.zeros(n+1, 3, 7)
    kin[..., 0] = torch.tensor([-2., 15., -16.]) + torch.arange(n+1).unsqueeze(1) * 3
    kin[:, 0, 3], kin[:, 1:, 2] = 1, 1
    kin[..., 4] = torch.tensor([6., 5.5, 6.5])
    kin[..., 5] = torch.tensor([.2, -.1, .3])
    kin[..., 6] = torch.tensor([5., 4.6, 5.4])
    mask = torch.ones(n+1, 3, dtype=torch.bool)
    cav = torch.zeros_like(mask)
    vid = torch.arange(3).expand(n+1, -1).clone()
    return dict(kin=kin, mask=mask, cav=cav, vid=vid)


def prepare(frozen, ep=None, **kwargs):
    return fit.prepare_episode(frozen, **(episode() if ep is None else ep),
                               allow_unverified_physics=True, **kwargs)


def sequence(cache, *, entry=1, stop=7, event=True, score_start=0):
    s = dict(slot=0, vehicle=0, entry_row=entry, event_row=stop if event else None,
             event=event, unknown_age=False, terminal_kind='event' if event else 'horizon_censor',
             times=[.5*(i+1) for i in range(stop-entry)], score_start=score_start)
    return fit.attach_episode([s], cache)[0]


@pytest.mark.parametrize('chunk_size', [1, 3, 16])
def test_batched_replay_matches_reference_including_birth_history_wait_and_actions(frozen, chunk_size):
    ep = episode()
    ep['mask'][3:5, 2] = False  # history carries as ORIGINAL runtime does
    ep['vid'][5:, 2] = 88
    ep['cav'][:, 1] = True
    actions = {'action_id': torch.full((8, 3), DIRECT_ID),
               'action_command': torch.tensor([0., 30.]).expand(8, 3, 2).clone()}
    actions['action_command'][:, 1, 0] = -.4
    ref = prepare(frozen, ep, vectorized=False, actions=actions)
    fast = prepare(frozen, ep, chunk_size=chunk_size, actions=actions)
    assert fast['metadata']['replay_engine'] == 'batched_causal_scan'
    for key in ('proposals', 'embedding', 'birth_embedding', 'intent'):
        torch.testing.assert_close(fast[key], ref[key], atol=1e-5, rtol=1e-5)
    for mode in ('original_runtime', 'corrected', 'seq_tf'):
        torch.testing.assert_close(fast[mode]['logits'], ref[mode]['logits'], atol=1e-5, rtol=1e-5)
        assert torch.equal(fast[mode]['eligible'], ref[mode]['eligible'])
        for key in exact.FEATURES:
            torch.testing.assert_close(fast[mode]['features'][key], ref[mode]['features'][key],
                                       atol=1e-5, rtol=1e-5)


def test_only_bounded_batched_forward_and_no_future_teacher(frozen, monkeypatch):
    calls = []
    forward = frozen.model.forward

    def counted(state, **kwargs):
        calls.append(state['veh_kin'].shape[0])
        return forward(state, **kwargs)

    monkeypatch.setattr(frozen.model, 'forward', counted)
    ep = episode()
    baseline = prepare(frozen, ep, chunk_size=3)
    assert calls == [3, 3, 2]
    changed = copy.deepcopy(ep)
    changed['kin'][5:, :, 0] += 200
    other = prepare(frozen, changed, chunk_size=3)
    for mode in ('original_runtime', 'corrected', 'seq_tf'):
        torch.testing.assert_close(baseline[mode]['logits'][..., :5],
                                   other[mode]['logits'][..., :5], atol=1e-5, rtol=1e-5)
    assert torch.equal(baseline['proposals'][..., 1:4], ep['kin'][:-1, :, 1:4])


def test_missing_physics_refuses_or_marks_explicit_diagnostic(frozen):
    assert not frozen.metadata['physics_confirmed']
    with pytest.raises(ValueError, match='authoritative run physics'):
        fit.prepare_episode(frozen, **episode())
    cache = prepare(frozen)
    assert not cache['metadata']['physics_confirmed']
    assert 'UNVERIFIED' in cache['metadata']['physics_source']
    assert cache['metadata']['joint_conflict_likelihood'] is False


def test_identity_repair_uses_reference_semantics(frozen):
    ep = episode()
    ep['vid'][4:, 0] = 99
    cache = prepare(frozen, ep, identity_policy='identity')
    assert cache['metadata']['replay_engine'] == 'reference_serial'
    assert cache['metadata']['identity_policy'] == 'identity'


def test_non_direct_controls_are_not_silently_approximated(frozen):
    ep = episode()
    ep['cav'][:, 1] = True
    with pytest.raises(ValueError, match='controller callback'):
        prepare(frozen, ep, actions={'action_id': torch.zeros(8, 3, dtype=torch.long)})


@pytest.mark.parametrize('kind', fit.MODELS)
@pytest.mark.parametrize('n_quad', [1, 8, 16])
def test_interface_distribution_nll_and_quadrature_refinement(frozen, kind, n_quad):
    cache = prepare(frozen)
    s = sequence(cache)
    model = fit.ExactCandidate(kind, frozen)
    batch = fit.exact_pack([s])
    dist = model.distribution(batch, n_quad)
    data = s['exact'][fit.MODE[kind]]
    if kind == 'M0-seqTF':
        scene = {key: value.view(1, -1, 1) for key, value in data['features'].items()}
        value = exact.seq_tf_logits(scene, frozen.core, n_quad=n_quad)
        logits = value['logits'][0, 0]
    else:
        value = exact.head_logits(frozen.core, data['features'], s['exact']['embedding'],
                                  birth_embedding=s['exact']['birth_embedding'],
                                  intent=s['exact']['intent'], n_quad=n_quad)
        logits = value['logits']
    oracle = exact.hazard_distribution(logits, value['weights'], data['eligible'])
    for key in dist:
        torch.testing.assert_close(dist[key][0], oracle[key], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.nll(batch, n_quad)[0], -oracle['log_mass'][-1])
    torch.testing.assert_close(dist['log_mass'].exp().sum()+dist['log_survival'][0, -1].exp(),
                               torch.tensor(1.), atol=1e-5, rtol=1e-5)
    assert model.median_margin(batch).shape == (1, len(s['times']))
    via_sequences = model.nll({**legacy.pack([s]), 'sequences': [s]}, n_quad)
    torch.testing.assert_close(via_sequences, model.nll(batch, n_quad))
    assert set(('s0', 'T', 'b', 'tau_a', 'temperature', 'sigma')) <= model.params().keys()


def test_actual_seqtf_omits_nonzero_heads_and_is_reported_separately(frozen):
    local = copy.deepcopy(frozen)
    with torch.no_grad():
        local.core.residual_head[-1].bias.fill_(.5)
        local.core.residual_head[-1].weight.fill_(.03)
    ep = episode()
    cache = prepare(local, ep)
    targets = type('Targets', (), dict(states=ep['kin'].unsqueeze(0), n_steps=8,
                                      loss_valid=ep['mask'].unsqueeze(0),
                                      is_cav=ep['cav'].unsqueeze(0), dt_s=.5))()
    scene = build_recorded_scene(targets, merge_lane=2, zone_s0_m=0, zone_s1_m=75.44, k=3)
    actual = exact.seq_tf_logits(scene, local.core)
    torch.testing.assert_close(cache['seq_tf']['logits'], actual['logits'][0])
    seq = sequence(cache)
    batch = fit.exact_pack([seq])
    runtime = fit.ExactCandidate('M0', local)
    seqtf = fit.ExactCandidate('M0-seqTF', local)
    assert not torch.allclose(runtime.nll(batch), seqtf.nll(batch))
    refit = fit.ExactCandidate('M0-refit', local)
    torch.testing.assert_close(runtime.nll(batch), refit.nll(batch))


def test_quadrature_64_positive_tail_weights_without_float32_floor(frozen):
    seq = sequence(prepare(frozen))
    model = fit.ExactCandidate('Soft-hazard', frozen)
    batch = fit.exact_pack([seq])
    law = model.distribution(batch, 64)
    assert law['log_mass'].dtype == torch.float64
    assert torch.isfinite(model.nll(batch, 64)).all()
    torch.testing.assert_close(law['log_mass'].exp().sum()+law['log_survival'][0, -1].exp(),
                               torch.tensor(1., dtype=torch.float64), atol=1e-6, rtol=0)


@pytest.mark.parametrize('kind', ['M0', 'M0-refit', 'Soft-hazard', 'M0-seqTF'])
def test_prefix_is_decision_count_and_scores_conditional_survival_not_prior_reset(frozen, kind):
    cache = prepare(frozen)
    seq = sequence(cache, score_start=3)
    model = fit.ExactCandidate(kind, frozen)
    conditional = model.distribution(fit.exact_pack([seq]))
    unconditioned = copy.deepcopy(seq)
    unconditioned['score_start'] = 0
    whole = model.distribution(fit.exact_pack([unconditioned]))
    log_denom = whole['log_survival'][0, 2]
    for key in ('log_mass', 'log_survival'):
        torch.testing.assert_close(conditional[key][0, 3:], whole[key][0, 3:]-log_denom,
                                   atol=1e-5, rtol=1e-5)
    assert torch.isneginf(conditional['log_mass'][0, :3]).all()
    assert torch.equal(conditional['cdf'][0, :3], torch.zeros(3))
    # Prefix censor + suffix event telescopes to full event, without a redraw.
    torch.testing.assert_close(-log_denom + model.nll(fit.exact_pack([seq]))[0],
                               -whole['log_mass'][0, -1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(conditional['log_mass'].exp().sum() +
                               conditional['log_survival'][0, -1].exp(), torch.tensor(1.),
                               atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('kind', fit.MODELS)
def test_unknown_history_and_zero_prefix_denominator_are_unsupported(frozen, kind):
    cache = prepare(frozen)
    model = fit.ExactCandidate(kind, frozen)
    seq = sequence(cache)
    seq['unknown_age'] = True
    batch = fit.exact_pack([seq])
    assert torch.isnan(model.distribution(batch)['cdf']).all()
    assert torch.isposinf(model.nll(batch)).all()
    seq['unknown_age'] = False
    seq['score_start'] = 1
    # Exercise exact deterministic atoms directly in the frozen cached path.
    if kind not in fit.REFITS:
        seq['exact'][fit.MODE[kind]]['logits'][:, 0] = torch.inf
        seq['exact'][fit.MODE[kind]]['eligible'][0] = True
        assert torch.isposinf(model.nll(fit.exact_pack([seq]))).all()


def test_veto_padding_censor_and_competing_exit_are_distinct(frozen):
    cache = prepare(frozen)
    short = sequence(cache, stop=4, event=False)
    long = sequence(cache, stop=7)
    long['exact']['corrected']['eligible'][-1] = False
    model = fit.ExactCandidate('Soft-hazard', frozen)
    batch = fit.exact_pack([short, long])
    loss = model.nll(batch)
    assert torch.isfinite(loss[0]) and torch.isposinf(loss[1])
    dist = model.distribution(batch)
    torch.testing.assert_close(dist['log_survival'][0, 2:],
                               dist['log_survival'][0, 2].expand(4))
    short['terminal_kind'] = 'junction_exit'
    assert torch.isposinf(model.nll(fit.exact_pack([short]))[0])
    assert torch.isfinite(model.distribution(fit.exact_pack([short]))['cdf']).all()


@pytest.mark.parametrize('kind', fit.REFITS)
def test_complete_refit_gradients_frozen_WM_and_exact_save_reload(frozen, kind):
    snapshot = {key: value.clone() for key, value in frozen.model.state_dict().items()}
    cache = prepare(frozen)
    seq = sequence(cache, score_start=2)
    batch = fit.exact_pack([seq])
    model = fit.ExactCandidate(kind, frozen)
    assert set(model.core.state_dict()) == set(frozen.core.state_dict())
    loss = model.nll(batch).sum()
    assert torch.isfinite(loss)
    loss.backward()
    for name in ('s0_raw', 't_raw', 'b_raw', 'tau_raw', 'temperature_raw',
                 'alpha_bias', 'alpha_log_sigma_raw'):
        grad = getattr(model.core, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert model.core.alpha_net[-1].weight.grad.abs().sum() > 0
    assert model.core.residual_head[-1].weight.grad.abs().sum() > 0
    assert not model.core.yield_scale_raw.requires_grad
    assert all(p.grad is None for p in frozen.model.parameters())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    clone = fit.ExactCandidate(kind, frozen)
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone.nll(batch), model.nll(batch), atol=0, rtol=0)
    for key, value in frozen.model.state_dict().items():
        assert torch.equal(value, snapshot[key]), key


def test_build_dataset_augments_same_existing_cohort_and_alignment(frozen, tmp_path):
    ep = episode()
    ep['kin'][7:, 0, 3], ep['kin'][7:, 0, 2] = 0, 1
    for key, name in (('kin', 'veh_kin'), ('mask', 'veh_mask'), ('cav', 'is_cav'), ('vid', 'vid')):
        np.save(tmp_path / (name+'.npy'), ep[key].numpy())
    np.save(tmp_path/'time_s.npy', np.arange(9)*.5)
    meta = dict(seed=963001, episode='synthetic_train', world='synthetic_world',
                source_name='synthetic', manifest_role='train', warmup_s=2.)
    (tmp_path/'episode.json').write_text(json.dumps(meta), encoding='utf-8')
    sequences, report = fit.build_dataset([{**meta, 'path': tmp_path}], frozen,
                                           allow_unverified_physics=True, chunk_size=3)
    assert len(sequences) == 1 and sequences[0]['event_row'] == 7
    seq = sequences[0]
    assert seq['entry_row'] == 1 and len(seq['times']) == 6
    assert seq['score_start'] == 2  # output times 1.0 and 1.5, not absolute row 2
    assert not seq['unknown_age']
    assert seq['exact']['embedding'].shape == (6, frozen.core.input_dim)
    assert seq['exact']['original_runtime']['logits'].shape == (8, 6)
    assert report['physics_diagnostic'] and report['timing'][0]['engine'] == 'batched_causal_scan'
    assert 'omits' in report['seqTF_caveat']
    assert math.isfinite(fit.ExactCandidate('M0', frozen).nll(fit.exact_pack(sequences))[0])


@pytest.mark.parametrize('path', ['../outside', 'data/hetdata_truth/x', 'data/sealed/x',
                                  'data/holdout/x', 'data/world964000/x', 'data/world964199/x'])
def test_data_boundary_refuses_before_open(frozen, path):
    with pytest.raises(ValueError):
        fit.build_dataset([{'path': path}], frozen)


@pytest.mark.parametrize('start', [-1, 1.5, True, 7])
def test_bad_prefix_decision_counts_rejected(frozen, start):
    seq = sequence(prepare(frozen))
    seq['score_start'] = start
    with pytest.raises(ValueError, match='score_start'):
        fit.exact_pack([seq])


def test_real_TRAIN_full_scene_vectorization_parity(frozen):
    loaded = exact.load_episode('data/r9_train/v5_native/episodes/w963004m1900p40t300__native')
    ep = {new: loaded[old][600:609] for new, old in
          (('kin', 'kin'), ('mask', 'present'), ('cav', 'is_cav'), ('vid', 'identity'))}
    fast, serial = prepare(frozen, ep, chunk_size=4), prepare(frozen, ep, vectorized=False)
    torch.testing.assert_close(fast['proposals'], serial['proposals'], atol=1e-5, rtol=1e-5)
    for mode in ('original_runtime', 'corrected', 'seq_tf'):
        torch.testing.assert_close(fast[mode]['logits'], serial[mode]['logits'], atol=1e-5, rtol=1e-5)
        assert torch.equal(fast[mode]['eligible'], serial[mode]['eligible'])
