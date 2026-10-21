"""Exact round-8 heads on the existing recorded-path fitter interface (CPU).

Usage from the workspace::

    from tools import r9_exact, r9_fit_exact as fit
    frozen = r9_exact.load_frozen(physics=authoritative_physics)
    sequences, report = fit.build_dataset(episodes, frozen, n_quad=8)
    batch = fit.exact_pack(sequences[:128])
    model = fit.ExactCandidate('M0-refit', frozen)
    loss = model.nll(batch, 8).mean()
    loss.backward()  # optimize every requires_grad parameter of model

Alternatively augment existing sequences with ``attach_episode`` and pass
``{**merge_fit_v2.pack(subset), 'sequences': subset}``. No monkeypatching needed.
``prepare_episode(frozen, kin, mask, cav, vid, n_quad, actions=None)`` caches
one causal proposal per input row. It never consumes the recorded next motion.

Laws (all share the same cohort, timestamps, prefixes and frozen encodings):
M0 = intact frozen deployed head, ORIGINAL geometry/envelope;
M0-seqTF = actual legacy zero-location/no-residual seqTF conditional logits;
M0-corrected = intact frozen head on corrected geometry;
M0-refit = complete decision-head refit on fixed ORIGINAL geometry;
Soft-hazard = complete decision-head refit on fixed corrected geometry, fallback.
The yield head is frozen because it affects the cached longitudinal path.
Neither a refit nor corrected bookkeeping is labeled untouched runtime parity.
SeqTF slot selection and its unsupported-event omission are intentionally NOT
copied into full-cohort scoring; use r9_exact.seq_tf_score to diagnose those.

``score_start`` counts PREFIX DECISIONS from entry (not seconds or an absolute
row). Prefix survival conditions the persistent latent, is not scored again,
and is differentiable during refit. Unknown age/history returns NaN curves and
infinite NLL; it never gets a new population prior. Competing exits have a
survival curve but infinite full-outcome NLL without a declared exit model.

Default replay uses r9_exact's encode-once/no-EMA, k=0, runtime identity policy.
Full-scene inputs are batched over time, NOT interpreted as an unroll. Rolling
histories, wait, lane age, lane-v5 accumulators and birth encodings stay causal.
Other identity protocols or stateful models use the reference serial replay.
No observational driver-EMA or non-direct controller is invented.

The copied stage300 checkpoint lacks authoritative run physics. By default
corrected replay refuses it. ``allow_unverified_physics=True`` permits only an
explicit DIAGNOSTIC using flat switches plus PhysicsConfig defaults; this is
not verified SUMO motion/substep/projection or joint conflict likelihood.

Timing-only CLI (does not fit or inspect audit scores)::

    python tools/r9_fit_exact.py --source native=data/r9_train/v5_native \
        --max-worlds 8 --allow-unverified-physics --out results/r9_exact_timing.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from tools import merge_fit_v2 as legacy  # noqa: E402
from tools import r9_exact as exact  # noqa: E402
from wms.model import lane_v5 as lv5  # noqa: E402
from wms.model.merge_frailty import gauss_hermite  # noqa: E402

MODELS = ('M0', 'M0-seqTF', 'M0-corrected', 'M0-refit', 'Soft-hazard')
MODE = {'M0': 'original_runtime', 'M0-seqTF': 'seq_tf',
        'M0-corrected': 'corrected', 'M0-refit': 'original_runtime',
        'Soft-hazard': 'corrected'}
REFITS = ('M0-refit', 'Soft-hazard')


class _SparsePairAttention(nn.Module):
    """Evaluate the frozen pair-bias MLP only on its exact attention support.

    Same learned layers, dense q/k/v product, neighborhood, softmax and output
    as residual_net._KinAttention. Only values which are subsequently masked
    to -inf are omitted. No neighbor/radius rule or attention edge is changed.
    A replay-local shallow module tree leaves the checkpoint model untouched.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x, mask, geo, courtesy=None):
        base = self.base
        b, k, _ = x.shape
        q = base.q(x).view(b, k, base.h, base.dh).transpose(1, 2)
        kk = base.k(x).view(b, k, base.h, base.dh).transpose(1, 2)
        vv = base.v(x).view(b, k, base.h, base.dh).transpose(1, 2)
        logits = (q @ kk.transpose(-1, -2)) / math.sqrt(base.dh)
        neigh = base._neighbourhood(geo, mask)
        batch, ego, other = neigh.nonzero(as_tuple=True)
        s, lane, v = geo['s'], geo['lane'], geo['v']
        ds = s[batch, ego] - s[batch, other]
        feats = torch.stack((ds / base.radius,
                             (lane[batch, ego] - lane[batch, other]).to(s.dtype) / 2.,
                             (v[batch, ego] - v[batch, other]) / 20.,
                             ds.abs() / base.radius,
                             ((lane[batch, ego] == lane[batch, other]) &
                              (lane[batch, ego] >= 0)).to(s.dtype),
                             (ego == other).to(s.dtype)), -1)
        bias = base.bias(feats)
        if base.courtesy_bias is not None and courtesy is not None:
            nb = courtesy[batch, other]
            bias = bias + base.courtesy_bias(nb[:, [0, 1, 2, 4, 6]])
        logits = logits.permute(0, 2, 3, 1)
        logits[batch, ego, other] += bias
        logits = logits.permute(0, 3, 1, 2).masked_fill(~neigh.unsqueeze(1), -torch.inf)
        attn = torch.nan_to_num(torch.softmax(logits, -1), nan=0.)
        out = (attn @ vv).transpose(1, 2).reshape(b, k, -1)
        return base.o(out) * mask.unsqueeze(-1).to(out.dtype), attn


def _replay_model(model):
    """Copy module containers, sharing every frozen layer/tensor read-only."""
    replay = copy.copy(model)
    replay._modules = model._modules.copy()
    net = copy.copy(model.residual_net)
    net._modules = model.residual_net._modules.copy()
    blocks = []
    for block in model.residual_net.blocks:
        clone = copy.copy(block)
        clone._modules = block._modules.copy()
        clone.attn = _SparsePairAttention(block.attn)
        blocks.append(clone)
    net.blocks = nn.ModuleList(blocks)
    replay.residual_net = net
    return replay


def _arrays(kin, mask, cav, vid):
    kin = torch.as_tensor(kin, dtype=torch.float32, device='cpu')
    mask, cav = (torch.as_tensor(v, dtype=torch.bool, device='cpu') for v in (mask, cav))
    vid = torch.as_tensor(vid, device='cpu')
    if (kin.ndim != 3 or kin.shape[-1] != 7 or kin.shape[0] < 2 or kin.shape[1] < 1
            or any(v.shape != kin.shape[:2] for v in (mask, cav, vid))):
        raise ValueError('need N+1 full-scene rows and matching mask/CAV/identity arrays')
    if not bool(torch.isfinite(kin[mask]).all()):
        raise ValueError('missing present kinematics cannot be imputed')
    return torch.where(mask.unsqueeze(-1), kin, 0), mask, cav, vid


def _batchable(model, identity_policy):
    # Refuse a fast path for unmodeled recurrent state. The reference helper is
    # the semantic fallback, rather than flattening a stateful forward over time.
    return (identity_policy == 'runtime' and model.lane_v5 is not None
            and model.merge_negotiation is None and model.merge_pivot is None
            and model.temporal_residual_mode == 'direct'
            and model.diag_temporal_tau_s is None
            and getattr(model.merge_core, 'context_mode', 'off') == 'off')


@torch.no_grad()
def prepare_episode(frozen, kin, mask, cav, vid, n_quad=8, actions=None, *,
                    chunk_size=16, allow_unverified_physics=False,
                    identity_policy='runtime', vectorized=True):
    """Cache causal proposals/features and full/seqTF logits, using shared code.

    actions is r9_exact's runtime vocabulary: action_id [N,K] and optional
    action_command [N,K,2]. Only direct commands can be replayed without a
    causal controller callback. None is the explicitly reported WM CAV fallback.
    Memory for pairwise scene features is bounded by chunk_size, independently
    of episode length. The returned cache can be released after attach_episode.
    """
    if chunk_size < 1 or not 1 <= n_quad <= 64:
        raise ValueError('positive chunk_size and n_quad in 1..64 required')
    if not frozen.metadata['physics_confirmed'] and not allow_unverified_physics:
        raise ValueError('authoritative run physics required; use '
                         'allow_unverified_physics=True only for a DIAGNOSTIC replay')
    kin, mask, cav, vid = _arrays(kin, mask, cav, vid)
    started = perf_counter()
    model = frozen.model
    if not vectorized or not _batchable(model, identity_policy):
        cache = exact.precompute_episode(
            frozen, kin=kin, present=mask, is_cav=cav, identity=vid, n_quad=n_quad,
            actions=actions, identity_policy=identity_policy,
            allow_unverified_physics=allow_unverified_physics)
        cache['metadata']['replay_engine'] = 'reference_serial'
        cache['metadata']['replay_seconds'] = perf_counter() - started
        return cache

    model = _replay_model(model)

    n, k = kin.shape[0] - 1, kin.shape[1]
    initial = exact._state(kin[:1], mask[:1], cav[:1])
    gap, _ = exact.lead_neighbour(kin[:1], mask[:1], model.physics)
    history, history_mask = exact.initial_history(
        exact.history_step_features(kin[:1], gap), mask[:1], model.history_len)
    dh = dm = None
    if model.long_history:
        dh, dm = exact.long_history_features(
            kin[:1], mask[:1], mask[:0], mask[0], history_s=model.history_s,
            history_stride_s=model.history_stride_s, dt_s=model.physics.dt_s)
        dh, dm = dh.unsqueeze(0), dm.unsqueeze(0)
    _, embedding, params = model.encode_drivers(
        initial, driver_history=dh, driver_history_mask=dm,
        history=history, history_mask=history_mask)
    if embedding is None:
        raise ValueError('secure-gap replay requires driver embeddings')
    wait = model.initial_merge_wait(mask[:1], kin[:1], dh, dm)
    age = model._initial_lane_age(initial, mask[:1], kin[:1], dh, dm)
    acc = model.initial_lane_v5_acc(mask[:1], kin[:1], dh, dm)
    has_birth = torch.zeros(k, dtype=torch.bool)
    birth = torch.zeros_like(embedding[0])
    collected = {mode: [] for mode in ('original_runtime', 'corrected', 'seq_tf')}
    proposals, intents, births, geometry_flags = [], [], [], []

    for lo in range(0, n, chunk_size):
        hi = min(n, lo + chunk_size)
        current, present, is_cav = kin[lo:hi], mask[lo:hi], cav[lo:hi]
        gap, _ = exact.lead_neighbour(current, present, model.physics)
        step_features = exact.history_step_features(current, gap)
        hs, hms, ages, waits, birth_rows = [], [], [], [], []
        for row in range(lo, hi):
            if row:
                history, history_mask = exact.push_history(
                    history, history_mask, step_features[row-lo:row-lo+1], mask[row:row+1])
                changed = kin[row:row+1, :, 1:4].argmax(-1) != kin[row-1:row, :, 1:4].argmax(-1)
                age = torch.where(changed, 0.0, age + 1)
                wait = exact.update_merge_wait(wait, kin[row:row+1], mask[row:row+1], model.physics)
            new = mask[row] & (torch.ones(k, dtype=torch.bool) if row == 0 else ~mask[row-1])
            has_birth &= mask[row] & ~new
            ramp = mask[row] & (kin[row, :, 1:4].argmax(-1) == model.physics.merge_lane)
            need = ramp & ~has_birth
            birth = torch.where(need.unsqueeze(-1), embedding[0], birth)
            has_birth |= need
            hs.append(history)
            hms.append(history_mask)
            ages.append(age)
            waits.append(wait)
            birth_rows.append(birth)
        ages, waits = torch.cat(ages), torch.cat(waits)
        # Shared instantaneous lane geometry is batched; only its inexpensive
        # causal accumulator scan remains chronological. forward receives the
        # PREVIOUS accumulator, so it still performs exactly one state update.
        instant = lv5.instant_features_chunked(current, present, model.physics, ages,
                                               extra=model.lane_v5.extra_features)
        previous_acc = []
        for row in range(hi-lo):
            previous_acc.append(acc)
            acc = lv5.update_acc(acc, {key: value[row:row+1] for key, value in instant.items()},
                                 model.physics, extra=model.lane_v5.extra_features)
        state = exact._state(current, present, is_cav)
        state['merge_wait_s'] = waits
        enc = embedding.expand(hi-lo, -1, -1)
        kwargs = dict(k_steps=0, history=torch.cat(hs), history_mask=torch.cat(hms),
                      driver_params={key: value.expand(hi-lo, -1) for key, value in params.items()},
                      driver_embedding=enc, lane_age=ages, lane_v5_acc=torch.cat(previous_acc))
        out = model.forward(state, **kwargs)
        action = {} if actions is None else {
            key: torch.as_tensor(value[lo:hi], device='cpu') for key, value in actions.items()}
        moved, _ = exact.longitudinal_proposal(
            frozen, state, out=out, forward_kwargs=kwargs,
            allow_unverified_physics=allow_unverified_physics, **action)
        proposals.append(moved)
        original = {key: current[..., 4 if key == 'v' else 5] if key in ('v', 'a')
                    else out['merge_core_' + key] for key in exact.FEATURES}
        ramp = present & (state['lane'] == model.physics.merge_lane)
        in_zone = ((current[..., 0] >= model.physics.merge_zone_s0_m)
                   & (current[..., 0] <= model.physics.merge_zone_s1_m))
        original_eligible = ramp & in_zone & out['lane_v5_acts'].bool() & ~is_cav
        corrected = exact._corrected_features(moved, present, is_cav, model.physics)
        geometry_flags.append({key: corrected[key].T for key in
                               ('overlap', 'physical_eligible', 'reservation_veto', 'ramp')})
        local, birth_enc = enc.transpose(0, 1), torch.stack(birth_rows, 1)
        intent = out['merge_core_intent'].T
        births.append(birth_enc)
        intents.append(intent)
        for name, features, eligible in (
                ('original_runtime', original, original_eligible),
                ('corrected', corrected, corrected['eligible'])):
            features = {key: features[key].T for key in exact.FEATURES}
            result = exact.head_logits(frozen.core, features, local,
                                       birth_embedding=birth_enc, intent=intent, n_quad=n_quad)
            collected[name].append({**result, 'features': features, 'eligible': eligible.T})
        seq = exact.seq_tf_logits({key: value.unsqueeze(0) for key, value in original.items()},
                                  frozen.core, n_quad=n_quad)
        collected['seq_tf'].append({'logits': seq['logits'][0],
                                    'features': collected['original_runtime'][-1]['features'],
                                    'eligible': (ramp & in_zone & ~is_cav).T})

    cache = {'kin': kin, 'present': mask, 'is_cav': cav, 'identity': vid,
             'embedding': embedding[0].unsqueeze(1).expand(-1, n, -1),
             'birth_embedding': torch.cat(births, 1), 'intent': torch.cat(intents, 1),
             'proposals': torch.cat(proposals),
             'weights': collected['original_runtime'][0]['weights'],
             'geometry_flags': {key: torch.cat([v[key] for v in geometry_flags], -1)
                                for key in geometry_flags[0]},
             'metadata': {**frozen.metadata, 'replay_engine': 'batched_causal_scan',
                          'attention': 'shared_frozen_layers_exact_neighborhood_sparse_pair_bias',
                          'chunk_size': chunk_size, 'identity_policy': identity_policy,
                          'driver_protocol': 'encode_once_no_EMA', 'k_steps': 0,
                          'corrected': True, 'CAV_fallback': 'WM_physics_no_future_teacher',
                          'geometry': 'one_update_pre_change_lanes_full_scene_reservations',
                          'decision_likelihood_scope': 'conditional_proposal_before_commit_arbitration',
                          'lane_conflict_resolve': model.lane_conflict_resolve,
                          'joint_conflict_likelihood': False,
                          'runtime_inputs_overridden': False}}
    for mode, values in collected.items():
        cache[mode] = {'logits': torch.cat([v['logits'] for v in values], -1),
                       'eligible': torch.cat([v['eligible'] for v in values], -1),
                       'features': {key: torch.cat([v['features'][key] for v in values], -1)
                                    for key in exact.FEATURES}}
    cache['metadata']['replay_seconds'] = perf_counter() - started
    return cache


def attach_episode(sequences, cache):
    """Augment existing fitter sequences by input entry/terminal, not output row.

    Copies only the per-car exposure, avoiding views retaining an entire scene
    cache after an episode is released. Retains unknown-age and exit metadata.
    Mutates and returns the supplied sequences for convenient integration.
    """
    for seq in sequences:
        slot, entry = int(seq['slot']), int(seq['entry_row'])
        stop = entry + len(seq['times'])
        if not 0 <= entry < stop <= cache['kin'].shape[0]-1:
            raise ValueError('sequence entry/terminal outside cached decision rows')
        if (not bool(cache['present'][entry:stop, slot].all()) or
                not bool((cache['identity'][entry:stop, slot] == seq['vehicle']).all())):
            raise ValueError('sequence identity does not match episode cache')
        if seq['event'] and seq['event_row'] != stop:
            raise ValueError('event output row must equal last decision row + 1')
        data = {'n_quad': len(cache['weights'])}
        for name in ('embedding', 'birth_embedding', 'intent'):
            data[name] = cache[name][slot, entry:stop].clone()
        for mode in ('original_runtime', 'corrected', 'seq_tf'):
            values = cache[mode]
            data[mode] = {'logits': values['logits'][slot, :, entry:stop].clone(),
                          'eligible': values['eligible'][slot, entry:stop].clone(),
                          'features': {key: value[slot, entry:stop].clone()
                                       for key, value in values['features'].items()}}
        seq['exact'] = data
        seq.setdefault('score_start', 0)
        # Every updated candidate in the fitter sees this same corrected path
        # and eligibility, including the deterministic opportunity reservation.
        for mode, target in (('original_runtime', 'legacy'), ('corrected', 'updated')):
            seq[target] = {key: value.tolist() for key, value in data[mode]['features'].items()}
        seq['legacy_support'] = data['original_runtime']['eligible'].tolist()
        seq['physical'] = data['corrected']['eligible'].tolist()
        flags = cache.get('geometry_flags')
        reasons = []
        for row, ok in enumerate(seq['physical'], entry):
            reason = ('ok' if ok else 'outside_merge_zone_or_ineligible')
            if not ok and flags is not None:
                if flags['overlap'][slot, row]:
                    reason = 'target_interval_intersection'
                elif flags['reservation_veto'][slot, row]:
                    reason = 'simultaneous_reservation_veto'
            reasons.append(reason)
        seq['physical_reasons'] = reasons
    return sequences


def build_dataset(episodes, frozen, *, n_quad=8, chunk_size=16,
                  allow_unverified_physics=False, actions_by_episode=None):
    """Use the existing cohort builder, augment once per episode; no audit fit.

    Controls must be supplied in runtime vocabulary via actions_by_episode(ep).
    Otherwise CAV motion is explicitly WM physics fallback, never recorded future
    motion or silently converted native/PI/damping commands. All paths and seed
    checks run before legacy code opens data (including its manifest/metadata).
    """
    sequences, exclusions, diagnostics, timing = [], [], [], []
    started = perf_counter()
    for ep in episodes:
        path = exact.guard(ep['path'])
        for key in ('world', 'source_name', 'episode'):
            exact.guard(ROOT / str(ep[key]))
        legacy.check_seed(ep.get('seed', -1))
        loaded = exact.load_episode(path)
        for name in ('control_time_s', 'controller', 'issued_acceleration_mps2',
                     'issued_target_speed_mps'):
            exact.guard(path / (name + '.npy'))
        actions = None if actions_by_episode is None else actions_by_episode(ep)
        cache = prepare_episode(frozen, loaded['kin'], loaded['present'], loaded['is_cav'],
                                loaded['identity'], n_quad, actions, chunk_size=chunk_size,
                                allow_unverified_physics=allow_unverified_physics)
        index = 0

        def proposal(**kwargs):
            nonlocal index
            value = cache['proposals'][index].numpy()
            index += 1
            return value

        subset, report = legacy.build_dataset([{**ep, 'path': path}], adapter=proposal)
        attach_episode(subset, cache)
        for seq in subset:
            entry, length = seq['entry_row'], len(seq['times'])
            output_times = loaded['time_s'][entry+1:entry+length+1]
            seq['score_start'] = int(np.sum(output_times < float(ep.get('warmup_s', 0))))
        sequences.extend(subset)
        exclusions.extend(report['exclusions'])
        diagnostics.extend(report['episodes'])
        timing.append({'world': ep['world'], 'episode': ep['episode'], 'cars': len(subset),
                       'decisions': cache['proposals'].shape[0],
                       'replay_seconds': cache['metadata']['replay_seconds'],
                       'engine': cache['metadata']['replay_engine']})
        print(f"exact replay {ep['episode']}: {timing[-1]['replay_seconds']:.3f}s", flush=True)
        cache = None  # release full-scene storage before preparing the next episode
    if not sequences:
        raise ValueError('no recorded ramp sequences')
    return sequences, {'exclusions': exclusions, 'episodes': diagnostics, 'timing': timing,
                       'build_seconds': perf_counter()-started, 'checkpoint': frozen.metadata,
                       'CAV_controls': 'WM_physics_fallback' if actions_by_episode is None else 'supplied',
                       'physics_diagnostic': not frozen.metadata['physics_confirmed'],
                       'conditioning': 'full available risk prefix; score_start counts decisions',
                       'seqTF_caveat': 'actual loss omits location/residual; no deployed parity claim',
                       'exit_likelihood': 'unidentified competing outcomes are unsupported',
                       'likelihood_scope': 'fixed recorded path, proposal before commit arbitration',
                       'joint_conflict_likelihood': False,
                       'original_LCSTATE_event_denominator': 2915}


def exact_pack(sequences):
    """Padded full-head data plus legacy.pack fields for existing consumers.

    The exact payload is float32, matching the checkpoint. Padding is ineligible;
    quadrature can be refined later because features/encodings remain available.
    """
    if not sequences:
        raise ValueError('cannot pack an empty sequence batch')
    batch = legacy.pack(sequences)
    n, length = batch['valid'].shape
    data = {'score_start': torch.tensor([s.get('score_start', 0) for s in sequences], dtype=torch.long),
            'unknown': torch.tensor([bool(s.get('unknown_age', False)
                                         or s.get('unknown_history', False)) for s in sequences]),
            'exit_unsupported': torch.tensor([s.get('terminal_kind', 'event' if s['event'] else
                                                   'horizon_censor') not in
                                              ('event', 'horizon_censor', 'administrative_end')
                                              for s in sequences])}
    for i, seq in enumerate(sequences):
        start = seq.get('score_start', 0)
        if isinstance(start, bool) or int(start) != start or not 0 <= start <= len(seq['times']):
            raise ValueError('score_start must count prefix decisions within the sequence')
        if seq['event'] and start == len(seq['times']):
            raise ValueError('event decision cannot lie inside conditioning prefix')
        if 'exact' not in seq:
            raise ValueError('attach_episode must supply full-head exact data first')
    for key in ('embedding', 'birth_embedding', 'intent'):
        tail = sequences[0]['exact'][key].shape[1:]
        array = torch.zeros(n, length, *tail, dtype=torch.float32)
        for i, seq in enumerate(sequences):
            array[i, :len(seq['times'])] = seq['exact'][key]
        data[key] = array
    for mode in ('original_runtime', 'corrected', 'seq_tf'):
        counts = {s['exact']['n_quad'] for s in sequences}
        if len(counts) != 1:
            raise ValueError('cached quadrature counts must agree within a batch')
        q = counts.pop()
        eligible = torch.zeros(n, length, dtype=torch.bool)
        logits = torch.zeros(n, q, length, dtype=torch.float32)
        features = {key: torch.zeros(n, length, dtype=torch.bool if key.startswith('has_')
                                     else torch.float32) for key in exact.FEATURES}
        for i, seq in enumerate(sequences):
            stop = len(seq['times'])
            source = seq['exact'][mode]
            eligible[i, :stop], logits[i, :, :stop] = source['eligible'], source['logits']
            for key in features:
                features[key][i, :stop] = source['features'][key]
        data[mode] = {'features': features, 'eligible': eligible, 'logits': logits}
    batch['exact_pack'] = data
    batch['sequences'] = sequences
    return batch


class ExactCandidate(nn.Module):
    """Fitter-compatible full-head law; dynamics/encoders are never trainable."""

    def __init__(self, kind, frozen):
        super().__init__()
        if kind not in MODELS:
            raise ValueError(f'exact kind must be one of {MODELS}')
        self.kind = kind
        self.core = (exact.matched_refit_core(frozen) if kind in REFITS
                     else copy.deepcopy(frozen.core).requires_grad_(False))
        self.core.eval()

    def params(self):
        return {**self.core.params(), 'sigma': self.core.alpha_scale(),
                'alpha_bias': self.core.alpha_bias,
                'residual_scale': torch.nn.functional.softplus(self.core.residual_scale_raw) + 1e-6}

    def _batch(self, batch):
        if 'exact_pack' in batch:
            return batch, batch['exact_pack']
        if 'sequences' not in batch:
            raise ValueError("use exact_pack(sequences) or batch['sequences'] with attached exact data")
        packed = exact_pack(batch['sequences'])
        return packed, packed['exact_pack']

    def _logits(self, data, n_quad):
        mode = data[MODE[self.kind]]
        if self.kind not in REFITS and mode['logits'].shape[-2] == n_quad:
            _, weights = gauss_hermite(n_quad, dtype=mode['logits'].dtype)
            return mode['logits'], weights
        if self.kind == 'M0-seqTF':
            # Convert [cars,T] features to actual seqTF's [B,N,K] convention.
            scene = {key: value.T.unsqueeze(0) for key, value in mode['features'].items()}
            result = exact.seq_tf_logits(scene, self.core, n_quad=n_quad)
            return result['logits'][0], result['weights']
        result = exact.head_logits(self.core, mode['features'], data['embedding'],
                                   birth_embedding=data['birth_embedding'], intent=data['intent'],
                                   n_quad=n_quad)
        return result['logits'], result['weights']

    def distribution(self, batch, n_quad=8):
        if not 1 <= n_quad <= 64:
            raise ValueError('n_quad must be in 1..64')
        batch, data = self._batch(batch)
        logits, weights = self._logits(data, n_quad)
        eligible = data[MODE[self.kind]]['eligible'] & batch['valid']
        whole = exact.hazard_distribution(logits, weights, eligible)
        start = data['score_start']
        ix = (start-1).clamp_min(0).unsqueeze(-1)
        denominator = torch.where(start > 0, whole['log_survival'].gather(1, ix).squeeze(1),
                                  torch.zeros_like(logits[:, 0, 0]))
        unsupported = data['unknown'] | ~torch.isfinite(denominator)
        safe = torch.where(unsupported, torch.zeros_like(denominator), denominator)
        prefix = torch.arange(logits.shape[-1]).unsqueeze(0) < start.unsqueeze(1)
        mass = torch.where(prefix, -torch.inf, whole['log_mass'] - safe.unsqueeze(1))
        survival = torch.where(prefix, 0.0, whole['log_survival'] - safe.unsqueeze(1))
        # Unknown history is not a population-prior reset, and a zero prefix
        # denominator is not repaired by a floor or a fresh latent.
        mass = torch.where(unsupported.unsqueeze(1), torch.nan, mass)
        survival = torch.where(unsupported.unsqueeze(1), torch.nan, survival)
        return {'log_mass': mass, 'log_survival': survival, 'cdf': -torch.expm1(survival)}

    def nll(self, batch, n_quad=8):
        batch, data = self._batch(batch)
        dist = self.distribution(batch, n_quad)
        ix = batch['last'].unsqueeze(1)
        event = dist['log_mass'].gather(1, ix).squeeze(1)
        censor = dist['log_survival'].gather(1, ix).squeeze(1)
        score = -torch.where(batch['event'], event, censor)
        return torch.where(torch.isnan(score) | data['exit_unsupported'], torch.inf, score)

    def median_margin(self, batch):
        batch, data = self._batch(batch)
        # Q=1 Gauss-Hermite is exactly the zero normal node, preserving BOTH
        # birth location and the per-step local-median residual for full heads.
        logits, _ = self._logits(data, 1)
        margin = logits[:, 0] * self.core.params()['temperature'].clamp_min(1e-3)
        return torch.where(batch['valid'] & ~data['unknown'].unsqueeze(1), margin, torch.nan)


def main(argv=None):
    parser = argparse.ArgumentParser(description='CPU timing/cohort preparation, no audit fitting')
    parser.add_argument('--source', action='append', required=True, help='NAME=TRAIN_DIR')
    parser.add_argument('--checkpoint', default=str(exact.DEFAULT_CHECKPOINT))
    parser.add_argument('--physics', help='authoritative workspace-local PhysicsConfig JSON')
    parser.add_argument('--allow-unverified-physics', action='store_true')
    parser.add_argument('--max-worlds', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--n-quad', type=int, default=8)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.max_worlds < 1:
        parser.error('--max-worlds must be positive')
    out = exact.guard(args.out)
    # Guard before legacy discovery opens any path, including the stronger
    # truth-table and resolved evaluation-seed restrictions in r9_exact.guard.
    for source in args.source:
        name, sep, directory = source.partition('=')
        if not sep:
            parser.error('--source expects NAME=DIR')
        exact.guard(name)
        exact.guard(directory)
        manifest = exact.guard(Path(directory) / 'C4_EPISODES_MANIFEST.json')
        doc = json.loads(manifest.read_text(encoding='utf-8-sig'))
        entries = doc.get('episodes', [])
        for entry in entries.values() if isinstance(entries, dict) else entries:
            episode_path = exact.guard(Path(directory) / entry['dir'])
            exact.guard(episode_path / 'episode.json')
    physics = (None if args.physics is None else
               json.loads(exact.guard(args.physics).read_text(encoding='utf-8-sig')))
    frozen = exact.load_frozen(args.checkpoint, physics=physics)
    episodes, sources = legacy.discover(args.source, args.max_worlds)
    sequences, report = build_dataset(episodes, frozen, n_quad=args.n_quad,
                                     chunk_size=args.chunk_size,
                                     allow_unverified_physics=args.allow_unverified_physics)
    report.update(sources=sources, worlds=len({ep['world'] for ep in episodes}),
                  cars=len(sequences), events=sum(s['event'] for s in sequences),
                  unknown_age=sum(s['unknown_age'] for s in sequences),
                  prefix_conditioned=sum(s['score_start'] > 0 for s in sequences),
                  status='preparation_timing_only_no_audit_or_launch_claim')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(legacy.clean_json(report), indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in
                      ('status', 'worlds', 'cars', 'events', 'build_seconds', 'physics_diagnostic')}))


if __name__ == '__main__':
    main()
