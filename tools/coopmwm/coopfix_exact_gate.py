"""COOPFIX EXACT GATE: one fixed update from MAPPO u0 (v2v, +intent when present), then the
isolated, CRN-paired policy-synergy check on the certified cooperative bank states.

OFFLINE (CPU only): load MAPPO u0 exactly like the live V2V runner
(v2v_runtime.load_checkpoint(path, config={'v2v': True, ...}); its forward widens recorded
contexts; with the intent option present the live intent config keys are used), build the
bank under the bank rule per_seed_v1 (owner 2026-11-03 21:45; bank_statistics.install(rb,
bank_rule='per_seed_v1', sources=...) - the V2V reader, restored afterwards only when this call
installed it) and run ONE fixed update =
one transactional full-pass imitation run (bank_imitation.run_separate, the live V2V path).
Offline gate fields: run_separate pass-1 effectiveness/loss drop/failure, epochs, backtracks,
unexecuted-filtered pairs, final KL, cooperative-pair error change (paired per-pair cluster
bootstrap 95% CI on the unique cooperative slice of the bank layout), intent-pair error
change, and movement of the zero-initialised partner (intent, when present) input columns.

EXACT (planned here, executed by the live agent on cpuQ, consumed here). Owner 2026-11-03 21:45:
plans are seed-specific - no cross-seed transfer requirement anywhere. A state is certified on
the seed(s) where its per_seed_v1 example is admitted AND cooperative within that seed
(mandatory 986066, 986048, 986003 must each be certified on >= 1 seed). On EACH certified seed
only - with that verified example's OWN CRN (seed 1 = the decision snapshot; seed 2 =
replay_bank.replicate(snapshot, verification_id), the exact salt of the confirmation
replication) - run group-only NEW ('all'), one SOLO NEW per member, background-only NEW and a
u3 reference (policy_synergy.specs_for, isolated=True; captured t0 action retained,
intervention t0+1). The seed-1 reference is adopted from the first-seed receipt only when the
decision snapshot policy IS u3; otherwise a fresh u3 reference branch is planned. Synergy per
seed = best-solo branch minus group branch on that seed's tapes; the per_seed_v1 paired
bootstrap (bank_statistics._stats on ONE seed: per-unit paired differences of the RESULT.json
per-second team_delay, circular 30 s blocks, B=2000, RNG 20261103) gives that seed's one-sided
95% lower bound. Verdict PASS iff the offline checks pass AND 986066 is complete on every planned
seed AND its synergy lower bound > 0 on at least one of its certified seeds (both reported).

Registered budget caps (isolated diagnostic registration MAIN2026-11-03 12:26, imported as
constants - never behavioural): per state n<=6 group members, window end <= 1200 s, isolated
cap 9600 sim-s per state. The certified-seed set is quoted in full; any overage over the per-state
cap is printed and recorded, never silently trimmed. Execution admits, precharges the ledger
(replay_bank._confirm_charge) and checks warm_health.assert_running like run_update.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import bank_imitation, bank_statistics, bank_v2v, policy_synergy as ps, sumo_verify as sv
from . import replay_bank as rb, v2v_obs, v2v_runtime
from .accel_protocol import SEED
from .exact_grade import _seal, _write_new, verify_sealed
from .protocol import digest, sha
from .v2v_model import BASE_INPUT, FEATURES, _first_linear

GATE = 'coopfix_exact'
GATE_UPDATE = 0
MANDATORY_STATES = (986066,)
GATE_STATE = 986066                    # the 6-CAV state the verdict is bound to
MAX_GROUP = 6                          # registered isolated diagnostic caps (MAIN 12:26)
MAX_WINDOW_END_S = 1200
ISOLATED_STATE_CAP_SIM_S = 9600
CAP_SOURCE = 'isolated diagnostic registration MAIN2026-11-03 12:26'
BOOTSTRAP_SEED = 20261101              # replay_bank._boot_ci estimator family seed
N_BOOT = 10000
SEEDS = (1, 2)
PHASE = 'bank_verification_ext'        # the registered isolated policy-synergy phase
PLAN_FILE = 'COOPFIX_EXACT_PLAN.json'
GATE_FILE = 'COOPFIX_GATE.json'
GATE_RULE = 'per_seed_v1'               # owner 2026-11-03 21:45: per-seed certification and testing


class CertificationFailed(ValueError):
    """Mandatory cooperative states are not certified by per_seed_v1 on any seed (gate FAIL before physics)."""

    def __init__(self, message, statuses):
        super().__init__(message)
        self.statuses = statuses


# ---------------------------------------------------------------- v2v / intent layout

def intent_module():
    """tools.coopmwm.v2v_intent when this source carries the intent option, else None."""
    name = __package__+'.v2v_intent'
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


RUNTIME_FILE = 'V2V_RUNTIME.json'
RUNTIME_KEYS = ('v2v', 'v2v_dim', 'v2v_intent', 'v2v_intent_horizon', 'v2v_intent_loss_weight',
                'v2v_intent_detach_trunk')


class GateIdentityError(ValueError):
    """The gate cannot bind itself to the deployed runtime/commit (audit B2/B3): refuse."""


def load_runtime(root):
    """(sha256 of <root>/V2V_RUNTIME.json bytes, its config) - the SAME file the launcher hashes
    into binding['runtime_sha256'] (audit B2/B3). Missing or malformed -> GateIdentityError."""
    path = Path(root)/RUNTIME_FILE
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as error:
        raise GateIdentityError(f'deployed runtime {path} unavailable: {error!r}') from error
    cfg = value.get('config') if isinstance(value, dict) else None
    if not isinstance(cfg, dict):
        raise GateIdentityError(f'{path} has no config object')
    return hashlib.sha256(raw).hexdigest(), cfg


def runtime_gate_config(runtime_cfg, v2v=True, intent='auto'):
    """Audit B3: the gate's model config IS the deployed runtime's v2v/intent keys; any mismatch
    with the requested gate options, the launcher's pinned options or the gate bank rule refuses."""
    from .v2v_launch3 import BANK_OPTIONS, INTENT_OPTIONS
    if intent not in ('auto', 'on', 'off'):
        raise ValueError(f'intent must be auto/on/off, not {intent!r}')
    cfg = {k: runtime_cfg[k] for k in RUNTIME_KEYS if k in runtime_cfg}
    on = bool(cfg.get('v2v_intent'))
    problems = []
    if not v2v and cfg.get('v2v'):
        problems.append('gate --no-v2v but the deployed runtime has v2v=True')
    if (intent == 'on' and not on) or (intent == 'off' and on):
        problems.append(f'gate intent={intent} but the deployed runtime has v2v_intent={on}')
    if on:
        for key, want in INTENT_OPTIONS.items():
            have = cfg.get(key)
            if have != want or isinstance(have, bool) != isinstance(want, bool):
                problems.append(f'deployed runtime {key}={have!r} differs from launcher {want!r}')
        module = intent_module()
        if module is None:
            problems.append('deployed runtime is intent-ON but this source has no v2v_intent module')
        elif not problems:
            module.settings(cfg)                 # validates v2v_dim against the intent block width
    rule = runtime_cfg.get('bank_statistics')
    if rule != GATE_RULE:
        problems.append(f'deployed runtime bank_statistics={rule!r}, gate rule is {GATE_RULE!r}')
    if rule != BANK_OPTIONS.get('bank_statistics'):
        problems.append(f'deployed runtime bank_statistics={rule!r} differs from the launcher')
    if problems:
        raise GateIdentityError('gate config refused: '+'; '.join(problems))
    return cfg if cfg.get('v2v') else None


def source_identity(source_root=None, source_manifest=None):
    """Audit B2: the commit the gate ran from - the qualified source manifest's commit for a
    packaged release (with that manifest's SHA256), else the exact git HEAD of the source root,
    refusing tracked modifications (the launcher compares this with its deployed commit)."""
    source = Path(source_root) if source_root else Path(__file__).resolve().parents[2]
    if source_manifest:
        raw = Path(source_manifest).read_bytes()
        manifest = json.loads(raw)
        commit = manifest.get('commit')
        if (not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit)
                or manifest.get('complete_source_inventory') is not True):
            raise GateIdentityError('qualified complete source manifest with a 40-hex commit required')
        return dict(commit=commit, source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                    method='qualified_source_manifest', source_root=str(source))

    def git(*args):
        try:
            return subprocess.run(['git', '-C', str(source), *args], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as error:
            raise GateIdentityError(f'git identity of {source} unavailable: {error!r}') from error
    head = git('rev-parse', '--verify', 'HEAD')
    if head.returncode:
        raise GateIdentityError(f'git HEAD of {source} unavailable; pass a source manifest')
    dirty = git('diff', '--quiet', 'HEAD', '--', '*.py')
    if dirty.returncode:
        raise GateIdentityError(f'tracked python sources under {source} differ from HEAD')
    return dict(commit=head.stdout.decode().strip(), source_manifest_sha256=None,
                method='exact_git_HEAD', source_root=str(source))


def gate_identity(root, v2v=True, intent='auto', source_root=None, source_manifest=None):
    """(model config, identity record, runtime config) bound into the plan and the gate report."""
    runtime_sha256, runtime_cfg = load_runtime(root)
    cfg = runtime_gate_config(runtime_cfg, v2v, intent)
    ident = dict(source_identity(source_root, source_manifest), runtime_sha256=runtime_sha256,
                 runtime_path=str(Path(root)/RUNTIME_FILE), runtime_config=cfg,
                 bank_statistics=runtime_cfg.get('bank_statistics'))
    return cfg, ident, runtime_cfg


IDENTITY_FIELDS = ('commit', 'runtime_sha256', 'source_manifest_sha256')


def gate_config(v2v=True, intent='auto'):
    """The live arm config keys the gate loads/builds with: None (no v2v), v2v-only
    ({'v2v': True, 'v2v_dim': V2V_DIM}) or the intent module's own config_keys()."""
    if intent not in ('auto', 'on', 'off'):
        raise ValueError(f'intent must be auto/on/off, not {intent!r}')
    if not v2v:
        if intent == 'on':
            raise ValueError('intent requires v2v')
        return None
    module = intent_module()
    use = module is not None if intent == 'auto' else intent == 'on'
    if use and module is None:
        raise ValueError('intent requested but tools/coopmwm/v2v_intent.py is absent in this source')
    if use:
        return dict(module.config_keys())
    return dict(v2v=True, v2v_dim=int(v2v_obs.V2V_DIM))


def block_layout(width):
    """Per-ego v2v block layout for a block width, derived from v2v_obs (feature-detected):
    the base partner-major layout (V2V_DIM), the intent layout K*(6+H+1) when v2v_obs exposes
    v2v_intent_dim/intent_columns, or a named V2V_INTENT_PARTNER_FEATURES layout."""
    width, k = int(width), int(v2v_obs.V2V_K)
    names = tuple(v2v_obs.V2V_PARTNER_FEATURES)
    present = names.index('present')
    if width == int(v2v_obs.V2V_DIM):
        return dict(width=width, k=k, stride=len(names), present=present, intent=(),
                    intent_valid=(), horizon=0, kind='v2v')
    dim, columns = getattr(v2v_obs, 'v2v_intent_dim', None), getattr(v2v_obs, 'intent_columns', None)
    if dim is not None and columns is not None and width % k == 0:
        horizon = width//k-len(names)-1
        if horizon >= 1 and int(dim(k, horizon)) == width:
            stride = width//k
            return dict(width=width, k=k, stride=stride, present=present,
                        intent=tuple(int(c) for c in columns(k, horizon)),
                        intent_valid=tuple(s*stride+len(names)+horizon for s in range(k)),
                        horizon=horizon, kind='v2v_intent')
    named = getattr(v2v_obs, 'V2V_INTENT_PARTNER_FEATURES', None)
    if named and width == k*len(named):
        named, stride = tuple(named), len(named)
        return dict(width=width, k=k, stride=stride, present=named.index('present'),
                    intent=tuple(s*stride+i for s in range(k) for i, n in enumerate(named) if 'intent' in n),
                    intent_valid=tuple(s*stride+i for s in range(k) for i, n in enumerate(named)
                                       if n == 'intent_valid'),
                    horizon=sum('intent' in n and n != 'intent_valid' for n in named), kind='v2v_intent_named')
    raise ValueError(f'v2v block width {width} matches no v2v_obs layout')


def _present_bits(sample, field='present'):
    """True when any partner presence bit (field='present') or intent-valid bit
    (field='intent_valid') is set in the sample's v2v blocks (layout from v2v_obs)."""
    blocks = (sample.get('context') or {}).get('v2v') or {}
    for row in blocks.values():
        if not len(row):
            continue
        layout = block_layout(len(row))
        if field == 'present':
            idx = [slot*layout['stride']+layout['present'] for slot in range(layout['k'])]
        else:
            idx = list(layout['intent_valid'])
        if any(row[i] for i in idx):
            return True
    return False


def intent_pairs(pairs):
    """Pairs whose context actually carries partner intent (an intent_valid bit) when the
    block layout has intent features; otherwise pairs with a present partner."""
    out = []
    for s, w in pairs:
        rows = list(((s.get('context') or {}).get('v2v') or {}).values())
        intent = bool(rows) and bool(block_layout(len(rows[0]))['intent'])
        if _present_bits(s, 'intent_valid' if intent else 'present'):
            out.append((s, w))
    return out


# ---------------------------------------------------------------- offline fixed update

def coop_pairs(pairs, summary=None):
    """(unique cooperative pairs, record). bank_pairs lays pairs out as
    other + coop*dup (+ coop[:rem]), so the unique cooperative slice is
    pairs[n_non_coop : n_non_coop + raw] (summary.exposure). Without a bank summary the
    structural rule (multi-CAV plan) is used, deduplicated by content digest - disclosed."""
    exposure = (summary or {}).get('exposure') or {}
    if exposure.get('cooperative_samples_raw') is not None \
            and exposure.get('non_cooperative_samples') is not None:
        start, raw = int(exposure['non_cooperative_samples']), int(exposure['cooperative_samples_raw'])
        if start+raw > len(pairs):
            raise ValueError(f'bank exposure ({start}+{raw}) exceeds the {len(pairs)} pairs')
        return list(pairs[start:start+raw]), dict(source='bank_exposure_slice', n_non_coop=start,
                                                  raw=raw, slice=[start, start+raw])
    seen, out = set(), []
    for s, w in pairs:
        if len(s.get('behaviour', {}).get('proposal_members', [])) <= 1:
            continue
        key = digest(dict(sample=s, weight=float(w)))
        if key not in seen:
            seen.add(key)
            out.append((s, w))
    return out, dict(source='structural_fallback_no_summary', n_non_coop=None, raw=None, slice=None)


@torch.no_grad()
def per_pair_errors(policy, pairs):
    """Mean member-step L1 action error (m/s^2) of each pair (the bank_imitation estimator)."""
    out = []
    for sample, _w in pairs:
        js = bank_imitation.indices(sample)
        if not js:
            continue
        mean, _ = policy(sample['context'])
        target = torch.as_tensor(sample['action'], dtype=torch.float32).reshape(-1, 1)
        out.append(float((mean[js] - target[js]).abs().mean()))
    return out


def paired_fall_ci(before, after, n_boot=N_BOOT, seed=BOOTSTRAP_SEED):
    """Paired cluster bootstrap 95% CI of the mean per-pair error change (after - before);
    negative = the fixed update moved actions toward the plan. None CI under 2 pairs."""
    deltas = [float(a) - float(b) for b, a in zip(before, after)]
    if not deltas:
        return None, None, 0
    mean = float(np.mean(deltas))
    if len(deltas) < 2:
        return mean, None, len(deltas)
    values = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), (int(n_boot), len(values)))].mean(1)
    return mean, [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], len(deltas)


def _columns(block):
    if block.shape[1] == 0:
        return dict(columns=0, nonzero=False, l2=0., max_abs=0.)
    return dict(columns=int(block.shape[1]), nonzero=bool((block != 0).any()),
                l2=float(block.norm()), max_abs=float(block.abs().max()))


def weight_report(policy):
    """Movement of the zero-initialised widened input columns of the first actor/critic
    layers: partner (v2v) columns and, when the layout has them, intent columns reported
    separately. 'moved' requires the intent columns when present, else the v2v columns."""
    actor = _first_linear(policy.actor)
    extra = int(actor.weight.shape[1])-BASE_INPUT['actor']
    layout = block_layout(extra) if extra > 0 else None
    intent = set(layout['intent']) if layout else set()
    v2v_idx = [FEATURES+o for o in range(max(extra, 0)) if o not in intent]
    intent_idx = [FEATURES+o for o in sorted(intent)]
    heads = {}
    for name in ('actor', 'critic'):
        w = _first_linear(getattr(policy, name)).weight.detach()
        heads[name] = dict(v2v=_columns(w[:, v2v_idx]), intent=_columns(w[:, intent_idx]))
    v2v_moved = any(h['v2v']['nonzero'] for h in heads.values())
    intent_moved = any(h['intent']['nonzero'] for h in heads.values())
    out = dict(extra_dim=extra, layout=(dict(layout, intent=list(layout['intent']),
                                             intent_valid=list(layout['intent_valid'])) if layout else None),
               heads=heads, v2v_columns_moved=v2v_moved, intent_columns_moved=intent_moved,
               required='intent columns' if intent_idx else 'v2v columns',
               moved=bool(intent_moved if intent_idx else (extra > 0 and v2v_moved)))
    head = getattr(policy, 'intent_head', None)
    if head is not None:
        out['intent_head'] = dict(weight_l2=float(head.weight.detach().norm()),
                                  bias_l2=float(head.bias.detach().norm()))
    return out


IMITATION_FIELDS = ('pass1_effective', 'pass1_loss_drop', 'pass1_step_scale', 'imitation_failed',
                    'failure_reason', 'epochs', 'backtracks', 'pairs_filtered_unexecuted', 'pairs_total',
                    'rejected_optimizer_steps', 'action_error_before_mps2', 'action_error_after_mps2')


def fixed_update(checkpoint, pairs, summary=None, passes=5, coefficient=1., v2v=True, intent='auto',
                 config=None):
    """ONE fixed update: the live V2V loader + one bank_imitation.run_separate run on the
    bank pairs; returns (post-update policy state dict, offline gate fields)."""
    cfg = config if config is not None else gate_config(v2v, intent)
    if cfg and cfg.get('v2v'):
        lacking = sum(1 for s, _ in pairs if not (s.get('context') or {}).get('v2v'))
        if lacking:
            raise ValueError(f'{lacking} bank pairs lack recorded v2v partner blocks; the live V2V '
                             'policy refuses them (bank_pairs(v2v=True) excludes such pairs)')
    policy, optimizer, generator, payload = v2v_runtime.load_checkpoint(checkpoint, config=cfg)
    reference = copy.deepcopy(policy)
    refresh = None
    if getattr(policy, 'intent_horizon', 0):
        # Audit B1: exactly like the live exact_update, the bank partner-intent inputs are the
        # CURRENT head's publications on the partners' own t-1 contexts (never recorded commands);
        # without this refresh the intent columns see only zeros and can never move.
        from .v2v_intent import refresh_bank_intents
        refresh = refresh_bank_intents(policy, pairs)
    try:
        report = bank_imitation.run_separate(policy, optimizer, pairs, generator, reference,
                                             coefficient=coefficient, passes=passes)
        error = None
    except RuntimeError as e:          # no complete pass fits the KL bound; policy restored
        report, error = dict(imitation_failed=True, failure_reason=str(e), epochs=[], kl=None,
                             kl_limit=.045, requested_epochs=int(passes), epochs_completed=0,
                             optimizer_steps=0, pairs_seen=0, stop_reason='imitation error'), repr(e)
    coop, coop_record = coop_pairs(pairs, summary)
    intent_set = intent_pairs(pairs)
    coop_before, coop_after = per_pair_errors(reference, coop), per_pair_errors(policy, coop)
    intent_before, intent_after = per_pair_errors(reference, intent_set), per_pair_errors(policy, intent_set)
    fall_mean, fall_ci, fall_clusters = paired_fall_ci(coop_before, coop_after)
    intent_mean, intent_ci, intent_clusters = paired_fall_ci(intent_before, intent_after)
    offline = dict(
        gate=GATE, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        update=payload.get('update'), v2v=bool(cfg and cfg.get('v2v')), config=cfg,
        v2v_dim=max(0, int(_first_linear(policy.actor).weight.shape[1])-BASE_INPUT['actor']),
        imitation_runner='bank_imitation.run_separate',
        imitation=dict({k: report.get(k) for k in IMITATION_FIELDS}, imitation_error=error),
        passes=dict(requested=report.get('requested_epochs'), completed=report.get('epochs_completed'),
                    optimizer_steps=report.get('optimizer_steps'), pairs_seen=report.get('pairs_seen'),
                    stop_reason=report.get('stop_reason')),
        kl=report.get('kl'), kl_limit=report.get('kl_limit'), kl_reference=report.get('kl_reference'),
        rule=report.get('rule'),
        cooperative_pairs=dict(count=len(coop), **coop_record,
                               error_before_mps2=float(np.mean(coop_before)) if coop_before else None,
                               error_after_mps2=float(np.mean(coop_after)) if coop_after else None,
                               change_mean_mps2=fall_mean, change_ci95=fall_ci,
                               clusters=fall_clusters),
        intent_pairs=dict(count=len(intent_set),
                          error_before_mps2=float(np.mean(intent_before)) if intent_before else None,
                          error_after_mps2=float(np.mean(intent_after)) if intent_after else None,
                          change_mean_mps2=intent_mean, change_ci95=intent_ci,
                          clusters=intent_clusters),
        partner_intent_weights=weight_report(policy), partner_intents_refreshed=refresh)
    offline['checks'] = offline_checks(offline)
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}, offline


def offline_checks(offline):
    """Offline verdict checks: >=1 accepted full pass inside the separate KL bound, pass 1
    effective with a positive AWR loss drop, no imitation failure, the cooperative-pair error
    falls with its CI entirely below 0, and the partner (intent) input columns moved."""
    imitation, passes = offline.get('imitation') or {}, offline['passes']
    coop, weights = offline['cooperative_pairs'], offline['partner_intent_weights']
    drop, kl = imitation.get('pass1_loss_drop'), offline.get('kl')
    return dict(
        full_pass=(passes.get('completed') or 0) >= 1,
        kl_within_limit=kl is not None and offline.get('kl_limit') is not None
        and kl <= offline['kl_limit'],
        pass1_effective=imitation.get('pass1_effective') is True and drop is not None and drop > 0,
        imitation_ok=not imitation.get('imitation_failed'),
        cooperative_error_falls=bool(coop.get('change_ci95')) and coop['change_ci95'][1] < 0.,
        partner_intent_weights_moved=bool(weights['moved']))


# ---------------------------------------------------------------- bank rule per_seed_v1

def _arm_sources(sources, arm):
    return tuple((sources or {}).get(arm, ()))


@contextlib.contextmanager
def bank_rule(sources=None, rule=GATE_RULE):
    """bank_statistics.install(rb, bank_rule=rule, sources=...) for the block; the previous
    reader is restored ONLY when this call installed it (a different installed rule is
    temporarily unwrapped to the base reader, never stacked)."""
    original = rb.confirmations
    installed = getattr(original, 'bank_rule', None) != rule
    if installed:
        base = original
        while hasattr(base, 'original'):
            base = base.original
        rb.confirmations = base
        bank_statistics.install(rb, bank_rule=rule, sources=sources)
    try:
        yield
    finally:
        if installed:
            rb.confirmations = original


def statistical_bank_pairs(root, arm, training_seed, sources=None, config=None, **kw):
    """bank_pairs(v2v=True) under per_seed_v1 with the per-arm sources; intent configs are
    passed as the live v2v_intent binding when this source's bank_pairs supports it."""
    kw = dict(kw, v2v=True, sources=_arm_sources(sources, arm))
    if config and config.get('v2v_intent'):
        if 'v2v_intent' not in inspect.signature(rb.bank_pairs).parameters:
            raise ValueError('intent config but replay_bank.bank_pairs has no v2v_intent option')
        kw['v2v_intent'] = config
    with bank_rule(sources):
        return rb.bank_pairs(root, arm, int(training_seed), **kw)


def certified_seeds(c):
    """Seeds whose per_seed_v1 example is admitted AND cooperative within that seed."""
    return [e['seed_index'] for e in ((c or {}).get('per_seed_v1') or {}).get('examples', [])
            if e.get('admitted') is True and e.get('cooperative') is True]


def certified_plans(r, c):
    """{seed: plan} of the certified cooperative example per seed. Legacy per_seed_v1 views
    carry no per-example plan (the first-seed chosen plan on both seeds); c4/verify-perseed all8
    records carry each example's OWN plan - the lowest candidate index wins on a seed with several
    cooperative examples (fixed rule, never outcome-selected)."""
    out = {}
    for e in sorted(((c or {}).get('per_seed_v1') or {}).get('examples', []),
                    key=lambda e: (e['seed_index'], e.get('candidate_index') or 0)):
        if e.get('admitted') is True and e.get('cooperative') is True and e['seed_index'] not in out:
            out[e['seed_index']] = copy.deepcopy(e.get('plan') or r['chosen_plan'])
    return out


def certified_candidates(root, arm, sources=None):
    """(candidates, statuses): admitted positive decisions certified cooperative on >= 1 seed
    under per_seed_v1 (no replication requirement), oldest first, as (receipt, confirmation,
    certified seeds); statuses discloses every positive decision (data_missing / no_solos / ...)."""
    with bank_rule(sources):
        decisions = rb.bank_decisions(root, arm, sources=_arm_sources(sources, arm))
        try:
            confirmed = rb.confirmations(root, arm)
        except bank_statistics.BankDataMissing as e:
            statuses = {r['verification_id']: dict(seed=r['episode']['world']['seed'], admitted=False,
                                                   cooperative=None, cooperative_status='data_missing',
                                                   certified_seeds=[], reason=str(e))
                        for r in decisions if r['status'] == 'accepted' and (r['grade_veh_s'] or 0) > 0}
            return [], statuses
    out, statuses = [], {}
    for r in decisions:
        c = confirmed.get(r['verification_id'])
        if (r['status'] != 'accepted' or not (r['grade_veh_s'] or 0) > 0) and not (
                c is not None and c.get('verify_mode') == 'all8_both_seeds'):
            continue                    # all8 records: seed-2-only winners of first-seed losers count
        co = (c or {}).get('cooperation') or {}
        status = 'not_confirmed' if c is None else co.get('cooperative_status', 'no_cooperation_record')
        seeds = certified_seeds(c)
        examples = ((c or {}).get('per_seed_v1') or {}).get('examples', [])
        statuses[r['verification_id']] = dict(
            seed=r['episode']['world']['seed'], admitted=bool(c and c.get('admitted')),
            cooperative=co.get('cooperative'), cooperative_status=status, certified_seeds=seeds,
            reason=((c or {}).get('per_seed_v1') or {}).get('reason') if c else 'no confirmation',
            per_seed=[dict(seed_index=e['seed_index'], admitted=e.get('admitted'),
                           cooperative_status=e.get('cooperative_status'),
                           lower95_one_sided_veh_s=e.get('lower95_one_sided_veh_s'),
                           synergy_lower95_veh_s=(e.get('synergy') or {}).get('lower95_one_sided_veh_s'))
                      for e in examples],
            synergy_reason=co.get('statistical_reason'))
        if c is not None and c.get('admitted') is True and seeds:
            out.append((r, c, seeds))
    return out, statuses


def _quote(vid, specs, phase=PHASE):
    return dict(verification_id=vid, branch_ids=[s['verification_id'] for s in specs],
                sim_s=sum(s['sim_s'] for s in specs), phase=phase)


# ---------------------------------------------------------------- job planning

def _policy_sha(snapshot):
    if snapshot.get('weights') is not None:
        return sv._state_sha(snapshot['weights'])
    return (snapshot.get('policy') or {}).get('state_dict_sha256')


def _reference(snapshot, receipt, update, arm, old_state):
    """Fresh all-u3 reference on this seed's snapshot (v2v_live.isolated rule)."""
    spec, snap = ps.specs_for(snapshot, receipt, update, arm, old_state, old_state)[0]
    spec = dict(spec)
    spec.pop('branch_sha256')
    spec.update(verification_id=spec['verification_id']+'_u3_reference', branch='reference')
    return _seal(spec, 'branch_sha256'), snap


def seed_snapshot(r, snap, seed):
    """(receipt, snapshot) of the VERIFIED example on ``seed``: seed 1 = the decision snapshot;
    seed 2 = replay_bank.replicate(snapshot, verification_id), the exact CRN salt the
    confirmation replication used (replay_bank.confirm) - never a fresh seed."""
    if seed == 1:
        return r, snap
    if seed != 2:
        raise ValueError(f'unknown seed index {seed!r}')
    return dict(r, verification_id=r['verification_id']+'_seed2'), rb.replicate(snap, r['verification_id'])


def state_jobs(r, snap, update, arm, new_state, old_state, seeds=SEEDS, plans=None):
    """Per certified seed: all / solo_i / background NEW (+ fresh u3 reference) on that
    example's own CRN snapshot and with that seed's certified plan group (plans={seed: plan};
    default the first-seed chosen plan). Seed 1 adopts the receipt's policy branch as the u3
    reference only when the decision snapshot policy is u3 AND seed 1 tests the chosen plan."""
    u3_snapshot = _policy_sha(snap) == sv._state_sha(old_state)
    out, adopted = {}, None
    for seed in sorted(seeds):
        receipt, sn = seed_snapshot(r, snap, seed)
        if plans and seed in plans:
            receipt = dict(receipt, chosen_plan=plans[seed])
        specs = ps.specs_for(sn, receipt, update, arm, new_state, old_state, isolated=True)
        if seed == 2 or not u3_snapshot:
            specs.append(_reference(sn, receipt, update, arm, old_state))
        out[seed] = specs
    if u3_snapshot and 1 in out and (not plans or plans.get(1, r['chosen_plan']) == r['chosen_plan']):
        branch = r['branches'][0]
        adopted = dict(seed=1, verification_id=r['verification_id'], branches_dir=str(r['_branches']),
                       branch=dict(branch), J_veh_s=float(branch['J_veh_s']),
                       rule='decision snapshot policy is u3: first-seed receipt policy branch adopted')
    return out, adopted


def plan(root, arm, control, new_state=None, old_state=None, offline=None, training_seed=SEED,
         pairs_source=None, update=GATE_UPDATE, sources=None, v2v=True, intent='auto', passes=5,
         coefficient=1., u0=None, u3=None, phase=PHASE, source_root=None, source_manifest=None):
    """Plan every SUMO job of the gate (sealed existing-format specs + snapshots + sim-s
    quotes) after the offline fixed update; writes COOPFIX_EXACT_PLAN.json atomically.
    Offline FAIL or uncertified mandatory states write a FAIL gate report (no physics).
    The model config comes from <root>/V2V_RUNTIME.json (audit B3) and the plan and every gate
    report seal commit / runtime_sha256 / source_manifest_sha256 (audit B2)."""
    root, control = Path(root), Path(control)
    if (control/PLAN_FILE).exists():
        return load_plan(control)
    if (control/GATE_FILE).exists():
        return json.loads((control/GATE_FILE).read_text())
    cfg, identity, runtime_cfg = gate_identity(root, v2v, intent, source_root, source_manifest)
    if sources is None and runtime_cfg.get('replay_sources'):
        sources = {k: list(v) for k, v in runtime_cfg['replay_sources'].items()}
    if offline is None or new_state is None:
        try:
            pairs, summary = (statistical_bank_pairs(root, arm, training_seed, sources, cfg)
                              if pairs_source is None else _pairs_file(pairs_source))
        except bank_statistics.BankDataMissing as e:     # per_seed_v1 cannot evaluate the bank
            return _fail(control, arm, update, None, 'bank', [repr(e)], None, identity)
        _new, offline = fixed_update(u0 or u0_path(root, arm), pairs, summary, passes=passes,
                                     coefficient=coefficient, config=cfg)
        new_state = _new                                     # the fixed update's own weights
    if old_state is None:
        old_state = torch.load(u3 or root/f'{arm}_checkpoint_003.pt', map_location='cpu',
                               weights_only=False)['policy']
    if not all(offline['checks'].values()):
        return _fail(control, arm, update, offline, 'offline', [], None, identity)
    try:
        candidates, statuses = certified_candidates(root, arm, sources)
        seeds = {r['episode']['world']['seed']: (r, c, cs) for r, c, cs in candidates}
        missing = [s for s in MANDATORY_STATES if s not in seeds]
        if missing:
            seen = {v['seed']: v['cooperative_status'] for v in statuses.values()}
            raise CertificationFailed(
                f'per_seed_v1 has not certified mandatory cooperative states {missing} on any seed '
                f'(statuses {dict((s, seen.get(s, "absent")) for s in missing)}); '
                f'certified: {sorted(seeds)}', statuses)
    except CertificationFailed as e:
        return _fail(control, arm, update, offline, 'certification', [str(e)], e.statuses, identity)
    links = [json.loads(p.read_text()) for p in (rb.bank_root(root, arm)/'decisions').glob('*.json')]
    by_vid = {link['verification_id']: link for link in links}
    states, snapshots, overages = [], {}, []
    for seed in sorted(seeds):
        r, c, crn_seeds = seeds[seed]
        plans = certified_plans(r, c)
        groups = {str(k): list(plans.get(k, r['chosen_plan'])['members']) for k in crn_seeds}
        group = groups[str(min(crn_seeds))]
        end_s = int(r['window']['end_s'])
        for g in groups.values():
            if not 2 <= len(g) <= MAX_GROUP:
                raise ValueError(f'state {seed}: group size {len(g)} outside the registered '
                                 f'isolated diagnostic range 2..{MAX_GROUP}')
        if end_s > MAX_WINDOW_END_S:
            raise ValueError(f'state {seed}: window end {end_s} > registered {MAX_WINDOW_END_S}')
        snap = rb._decision_snapshot(root, arm, r, by_vid)
        per_seed, adopted = state_jobs(r, snap, update, arm, new_state, old_state, crn_seeds, plans)
        jobs = []
        for s, specs in per_seed.items():
            for spec, sn in specs:
                snapshots[spec['verification_id']] = sn
                jobs.append(_job(spec, s, sn))
        quotes = {f'seed{s}': _quote(r['verification_id']+('' if s == 1 else '_seed2'),
                                     [x for x, _ in specs], phase) for s, specs in per_seed.items()}
        total = sum(q['sim_s'] for q in quotes.values())
        isolated = {f'seed{s}': sum(x['sim_s'] for x, _ in specs if x['branch'] != 'reference')
                    for s, specs in per_seed.items()}
        overage = max(0, total-ISOLATED_STATE_CAP_SIM_S)
        if overage:
            overages.append(f'state {seed}: certified-seed CRN set (seeds {sorted(per_seed)}) {total} sim-s '
                            f'exceeds the registered isolated cap {ISOLATED_STATE_CAP_SIM_S} sim-s by '
                            f'{overage} sim-s (isolated n+2 sets {isolated}); not trimmed')
        co = c.get('cooperation') or {}
        examples = {e['seed_index']: e for e in (c.get('per_seed_v1') or {}).get('examples', [])}
        states.append(dict(
            seed=int(seed), verification_id=r['verification_id'], group=group, groups=groups,
            plan_ids={str(k): plans.get(k, r['chosen_plan'])['id'] for k in crn_seeds}, seeds=sorted(per_seed),
            window=dict(r['window']), grade_veh_s=float(r['grade_veh_s']),
            plan_cooperation_gain_veh_s=co.get('cooperation_gain_veh_s'),
            certification=dict(rule=GATE_RULE, certified_seeds=sorted(per_seed),
                               replicated=bool((c.get('per_seed_v1') or {}).get('replicated')),
                               cooperative_status=co.get('cooperative_status'),
                               statistical_reason=co.get('statistical_reason'),
                               synergy_lower95_veh_s_per_seed={
                                   str(k): (e.get('synergy') or {}).get('lower95_one_sided_veh_s')
                                   for k, e in examples.items()}),
            reference_adopted=adopted, quotes=quotes,
            cost=dict(total_sim_s=total, isolated_n_plus_2_sim_s=isolated,
                      cap_sim_s=ISOLATED_STATE_CAP_SIM_S, overage_sim_s=overage),
            jobs=jobs))
    for line in overages:
        print('COOPFIX GATE CAP NOTE: '+line, file=sys.stderr)
    plan_record = dict(gate=GATE, arm=arm, update=update, unix=time.time(), root=str(root), phase=phase,
                       identity=identity, **{k: identity[k] for k in IDENTITY_FIELDS},
                       sources={k: list(v) for k, v in (sources or {}).items()},
                       new_state_sha256=sv._state_sha(new_state),
                       old_state_sha256=sv._state_sha(old_state),
                       rule='per CERTIFIED seed only, on the verified example\'s own CRN (seed 2 = '
                            'replay_bank.replicate(snapshot, verification_id)): group-only NEW, solo NEW '
                            'per member, background-only NEW, u3 reference; synergy = best-solo branch - '
                            'group branch on that seed\'s tapes; per_seed_v1 single-seed paired bootstrap '
                            '(bank_statistics._stats); PASS iff offline checks pass AND state '
                            f'{GATE_STATE} is complete on every planned seed AND its one-sided 95% lower '
                            'bound > 0 on at least one certified seed (both reported); no cross-seed '
                            'transfer requirement',
                       caps=dict(max_group=MAX_GROUP, window_end_s=MAX_WINDOW_END_S,
                                 isolated_state_cap_sim_s=ISOLATED_STATE_CAP_SIM_S, source=CAP_SOURCE,
                                 overages=overages),
                       certification=dict(rule='per_seed_v1: a state is certified on each seed whose '
                                               'example is admitted AND cooperative within that seed',
                                          statuses=statuses),
                       mandatory_states=list(MANDATORY_STATES), offline=offline, states=states,
                       totals=dict(states=len(states), jobs=sum(len(s['jobs']) for s in states),
                                   total_sim_s=sum(j['sim_s'] for s in states for j in s['jobs']),
                                   per_state_sim_s={str(s['seed']): s['cost']['total_sim_s'] for s in states},
                                   overage_sim_s=sum(s['cost']['overage_sim_s'] for s in states)))
    (control/'jobs').mkdir(parents=True, exist_ok=True)
    for vid, snap in snapshots.items():                       # executor inputs (cpuQ runs these)
        with (control/'jobs'/f'{vid}.SNAPSHOT.pt').open('xb') as f:
            torch.save(snap, f)
    for state in states:
        for job in state['jobs']:
            spec = job.pop('spec')
            _write_new(control/'jobs'/f'{job["verification_id"]}.{job["branch"]}.spec.json', spec)
    _write_new(control/PLAN_FILE, _seal(plan_record, 'gate_plan_sha256'))
    return load_plan(control)


def _fail(control, arm, update, offline, stage, reasons, statuses, identity=None):
    """FAIL gate report written before any physics (offline or certification failure)."""
    control.mkdir(parents=True, exist_ok=True)
    identity = identity or {}
    checks = dict(offline_checks=bool(offline) and all(offline['checks'].values()),
                  mandatory_states_certified=False if stage == 'certification' else None)
    report = dict(gate=GATE, arm=arm, update=update, unix=time.time(), stage=stage, offline=offline,
                  offline_checks=(offline or {}).get('checks'), certification_statuses=statuses, states=[],
                  gate_checks=checks, reasons=[k for k, ok in checks.items() if ok is False]+list(reasons),
                  gate_state_synergy_lower_bound=None, sim_s_planned=0, verdict='FAIL',
                  identity=identity or None, **{k: identity.get(k) for k in IDENTITY_FIELDS})
    _write_new(control/GATE_FILE, _seal(report, 'gate_report_sha256'))
    return report


def u0_path(root, arm):
    path = Path(root)/f'{arm}_checkpoint_000.pt'
    if not path.exists():
        raise ValueError(f'MAPPO u0 checkpoint missing: {path}')
    return path


def _job(spec, seed, snapshot):
    return dict(branch=spec['branch'], verification_id=spec['verification_id'], seed=int(seed),
                sim_s=spec['sim_s'], snapshot_sha256=sv.snapshot_sha(snapshot),
                generator_state_sha256=(snapshot.get('rng') or {}).get('generator_state_sha256'),
                spec_sha256=spec['branch_sha256'], spec=spec)


def load_plan(control):
    path = Path(control)/PLAN_FILE
    if not path.exists():
        raise ValueError(f'no {PLAN_FILE} under {control}; run the plan stage first')
    plan_record = json.loads(path.read_text())
    verify_sealed(plan_record, 'gate_plan_sha256')
    return plan_record


def _pairs_file(path):
    payload = json.loads(Path(path).read_text())
    return [(sample, float(w)) for sample, w in payload['pairs']], payload.get('summary')


# ---------------------------------------------------------------- execution (cpuQ; never on this laptop)

def _load_job(control, job):
    spec = json.loads((control/'jobs'/f'{job["verification_id"]}.{job["branch"]}.spec.json').read_text())
    verify_sealed(spec, 'branch_sha256')
    if spec['branch_sha256'] != job['spec_sha256']:
        raise ValueError('planned spec differs from its sealed job record')
    snap = torch.load(control/'jobs'/f'{job["verification_id"]}.SNAPSHOT.pt', weights_only=False)
    if sv.snapshot_sha(snap) != job['snapshot_sha256']:
        raise ValueError('planned snapshot differs from its sealed job record')
    return spec, snap


def execute(control, root=None):
    """Run every planned branch through the EXISTING policy_synergy.BranchRunner behind
    bank_v2v.enabled(v2v), one branch at a time: per state a durable claim, runner.admit,
    ONE atomic ledger precharge of the complete certified-seed set (replay_bank._confirm_charge,
    registered phase) and warm_health.assert_running before every branch - as run_update /
    v2v_live.isolated. No retry: a failed/stopped branch is recorded and never imputed."""
    from .warm_health import assert_running
    plan_record = load_plan(control)
    control = Path(control)
    root = Path(root or plan_record['root'])
    arm, update = plan_record['arm'], plan_record['update']
    kind = f'{GATE}_u{update:03}_{arm}'
    runner = ps.BranchRunner(control/'branches', workers=1)
    quotes = []
    for state in plan_record['states']:
        pending = [(job,)+_load_job(control, job) for job in state['jobs']]
        specs = [spec for _, spec, _ in pending]
        claim = control/'claims'/f'{state["verification_id"]}.CLAIM.json'
        if claim.exists():                  # interrupted or done: never replayed
            quotes.append(dict(verification_id=state['verification_id'],
                               status='claimed_previously_not_replayed'))
            continue
        blocked = runner.admit(specs)
        if blocked:
            raise RuntimeError(blocked)
        claim.parent.mkdir(parents=True, exist_ok=True)
        _write_new(claim, dict(unix=time.time(), state=state['seed'], kind=kind,
                               phase=plan_record['phase'], sim_s=sum(s['sim_s'] for s in specs)))
        # Complete certified-seed set precharged atomically before the first branch; ledger ids are
        # seed-qualified (branch labels repeat across the two CRN seeds).
        charge_rows = [dict(branch=f's{job["seed"]}_{spec["branch"]}', sim_s=spec['sim_s'])
                       for job, spec, _ in pending]
        rb._confirm_charge(root, plan_record['phase'], kind)(state['verification_id'], charge_rows)
        for job, spec, snap in pending:
            assert_running(root, arm)
            runner.prepare([spec], snap)
            with bank_v2v.enabled(plan_record['offline']['v2v']):
                receipt = sv.run_branch(spec, snap, runner)
            _write_new(control/'branches'/spec['verification_id']/f'{spec["branch"]}.graded.json', receipt)
        quotes.append(_quote(state['verification_id'], specs, plan_record['phase']))
    return quotes


# ---------------------------------------------------------------- consumption and verdict

class StoredRunner:
    """Regrades a stored raw branch result (BranchRunner-layout .result.json) through
    sumo_verify.run_branch without any SUMO."""

    def __init__(self, control):
        self.control = Path(control)

    def run(self, spec, snapshot):
        path = self.control/'branches'/spec['verification_id']/f'{spec["branch"]}.result.json'
        return json.loads(path.read_text())


def _receipt(control, job):
    """The graded receipt of one planned job: the executor's .graded.json, or a raw
    .result.json regraded through sumo_verify.run_branch (prefix check included)."""
    base = Path(control)/'branches'/job['verification_id']
    graded = base/f'{job["branch"]}.graded.json'
    if graded.exists():
        return json.loads(graded.read_text())
    spec, snap = _load_job(Path(control), job)
    return sv.run_branch(spec, snap, StoredRunner(control))


def state_receipts(control, state):
    """{(seed, branch): (receipt, tape loader)}; the adopted seed-1 reference reads the
    first-seed receipt's own policy-branch tape."""
    out = {}
    for job in state['jobs']:
        receipt = _receipt(control, job)
        out[(job['seed'], job['branch'])] = (receipt, lambda job=job, receipt=receipt: bank_statistics._read_result(
            Path(control)/'branches', job['verification_id'],
            dict(branch=job['branch'], result_sha256=receipt.get('result_sha256'))))
    adopted = state.get('reference_adopted')
    if adopted:
        receipt = dict(adopted['branch'], status=adopted['branch'].get('status', 'graded'),
                       J_veh_s=adopted['J_veh_s'], adopted=True)
        out[(1, 'reference')] = (receipt, lambda: bank_statistics._read_result(
            adopted['branches_dir'], adopted['verification_id'], adopted['branch']))
    return out


def state_metrics(state, receipts):
    """Per state and planned (certified) CRN seed: group-only gain, best solo, synergy (= group-only
    gain - best solo = J_best_solo - J_all, same seed), background-only delta, all vs the same
    seed's u3 reference (positive = NEW improved), each with its OWN single-seed per_seed_v1
    paired bootstrap (bank_statistics._stats on that seed's RESULT.json per-second team_delay
    tapes). Never pooled across seeds."""
    members = list(state['group'])
    window = state['window']
    seeds = [int(s) for s in state.get('seeds', SEEDS)]
    groups = {s: list((state.get('groups') or {}).get(str(s), members)) for s in seeds}
    names = {s: ['all', 'background', 'reference']+[f'solo{i}' for i in range(len(groups[s]))] for s in seeds}
    out = dict(seed=state['seed'], verification_id=state['verification_id'], group=members, seeds=seeds,
               window=window, quotes=state.get('quotes'), cost=state.get('cost'),
               reference_adopted=bool(state.get('reference_adopted')))
    status = {(s, b): (receipts.get((s, b)) or ({'status': 'missing'},))[0]['status']
              for s in seeds for b in names[s]}
    out['ungraded_branches'] = sorted(f'seed{s}/{b}' for (s, b), v in status.items() if v != 'graded')

    def J(s, b):
        return float(receipts[(s, b)][0]['J_veh_s'])
    per_seed = {}
    for s in seeds:
        if any(status[(s, b)] != 'graded' for b in names[s]):
            continue
        ref, g = J(s, 'reference'), groups[s]
        solos = {m: ref-J(s, f'solo{i}') for i, m in enumerate(g)}
        best = max(g, key=lambda m: solos[m])
        per_seed[s] = dict(group=g, J_reference_veh_s=ref, group_only_gain_veh_s=ref-J(s, 'all'),
                           solo_gain_veh_s=solos, best_solo_member=best, best_solo_gain_veh_s=solos[best],
                           synergy_veh_s=J(s, f'solo{g.index(best)}')-J(s, 'all'),
                           background_only_delta_veh_s=ref-J(s, 'background'))
    out['per_seed'] = {str(s): v for s, v in per_seed.items()}
    if out['ungraded_branches']:
        out.update(status='incomplete', complete=False, synergy_lower95_veh_s=None,
                   reason='ungraded branches; never imputed')
        return out

    def tape(s, b):
        return receipts[(s, b)][1]()

    def stats(s, alt, ref='reference'):
        return bank_statistics._stats([(tape(s, ref), tape(s, alt), window)])
    try:
        for s in seeds:
            best = f'solo{groups[s].index(per_seed[s]["best_solo_member"])}'
            synergy = dict(stats(s, 'all', ref=best), best_solo_branch=best)
            per_seed[s].update(synergy_lower95_veh_s=synergy['lower95_one_sided_veh_s'], synergy_v2=synergy,
                               group_only_v2=stats(s, 'all'), background_only_v2=stats(s, 'background'),
                               per_vehicle={m: stats(s, f'solo{i}') for i, m in enumerate(groups[s])})
    except (bank_statistics._Missing, ValueError) as e:
        out.update(status='data_missing', complete=False, synergy_lower95_veh_s=None,
                   reason='per-unit tape data missing or inconsistent: '+repr(e))
        return out
    lowers = {s: per_seed[s]['synergy_lower95_veh_s'] for s in seeds}
    out.update(status='graded', complete=True,
               synergy_lower95_veh_s_per_seed={str(s): v for s, v in lowers.items()},
               synergy_lower95_veh_s=max(lowers.values()),        # best certified seed (rule: >= 1 seed)
               seeds_synergy_lower95_gt_0=[s for s in seeds if lowers[s] > 0],
               group_only_gain_veh_s=float(np.mean([per_seed[s]['group_only_gain_veh_s'] for s in seeds])),
               synergy_veh_s=float(np.mean([per_seed[s]['synergy_veh_s'] for s in seeds])),
               background_only_delta_veh_s=float(np.mean([per_seed[s]['background_only_delta_veh_s']
                                                           for s in seeds])))
    return out


def collect(control, plan_record=None):
    """Consume every planned job output, compute the per-state CRN-paired synergy metrics and
    write COOPFIX_GATE.json (offline fields from the sealed plan; verdict per the gate rule)."""
    control = Path(control)
    plan_record = plan_record or load_plan(control)
    identity = plan_record.get('identity')
    if not identity or any(k not in plan_record for k in IDENTITY_FIELDS):
        raise GateIdentityError('gate plan lacks the sealed commit/runtime identity (audit B2)')
    runtime = Path(identity['runtime_path'])
    if runtime.exists() and hashlib.sha256(runtime.read_bytes()).hexdigest() != plan_record['runtime_sha256']:
        raise GateIdentityError(f'{runtime} changed after the gate was planned')
    states = [state_metrics(state, state_receipts(control, state)) for state in plan_record['states']]
    report = dict(gate=GATE, arm=plan_record['arm'], update=plan_record['update'],
                  identity=identity, **{k: plan_record[k] for k in IDENTITY_FIELDS},
                  unix=time.time(), plan_sha256=digest(plan_record),
                  offline=plan_record['offline'], offline_checks=plan_record['offline']['checks'],
                  states=states, plan_totals=plan_record['totals'], caps=plan_record['caps'])
    report['verdict'] = verdict(report)
    _write_new(control/GATE_FILE, _seal(report, 'gate_report_sha256'))
    return report


def verdict(report):
    """PASS iff the offline checks pass AND the 6-CAV state 986066 is complete on every planned
    (certified) seed AND its single-seed synergy one-sided 95% lower bound > 0 on at least one
    of those seeds (each seed's bound reported; no cross-seed transfer requirement)."""
    states = {s['seed']: s for s in report['states']}
    gate_state = states.get(GATE_STATE)
    lower = (gate_state or {}).get('synergy_lower95_veh_s')
    checks = dict(offline_checks=all(report['offline_checks'].values()),
                  gate_state_present=gate_state is not None,
                  gate_state_complete=bool(gate_state and gate_state['complete']),
                  gate_state_synergy_lower95_gt_0=bool(lower is not None and lower > 0.))
    report['gate_state_synergy_lower_bound'] = lower
    report['gate_state_synergy_lower_bounds_per_seed'] = (gate_state or {}).get('synergy_lower95_veh_s_per_seed')
    report['gate_state_seeds_tested'] = (gate_state or {}).get('seeds')
    report['gate_checks'] = checks
    report['reasons'] = [name for name, ok in checks.items() if ok is not True]
    return 'PASS' if not report['reasons'] else 'FAIL'


# ---------------------------------------------------------------- CLI

def parse_sources(values):
    """--sources ARM=PATH[,PATH...] (repeatable) or a JSON file holding {arm: [paths]} (or a
    V2V_RUNTIME.json whose config carries replay_sources)."""
    out = {}
    for value in values or ():
        path = Path(value)
        if '=' not in value and path.suffix == '.json' and path.exists():
            v = json.loads(path.read_text())
            v = (v.get('config') or {}).get('replay_sources') or v.get('replay_sources') or v
            for arm, paths in v.items():
                out.setdefault(arm, []).extend(str(p) for p in paths)
            continue
        arm, _, paths = value.partition('=')
        if not arm or not paths:
            raise ValueError(f'--sources expects ARM=PATH[,PATH], got {value!r}')
        out.setdefault(arm, []).extend(p for p in paths.split(',') if p)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('plan', 'execute', 'collect', 'status'))
    p.add_argument('--root', help='live run root (bank, decisions, u0/u3 checkpoints, ledger)')
    p.add_argument('--arm', default='B', choices=('B', 'R'))
    p.add_argument('--control', required=True, help='gate control directory (plan, jobs, receipts)')
    p.add_argument('--sources', action='append', default=[],
                   help="per-arm replay sources (live cfg['replay_sources']): ARM=PATH[,PATH] "
                        'repeatable, or a JSON file')
    p.add_argument('--u0', type=Path, help='MAPPO u0 checkpoint (default {root}/{arm}_checkpoint_000.pt)')
    p.add_argument('--u3', type=Path, help='frozen reference checkpoint (default {root}/{arm}_checkpoint_003.pt)')
    p.add_argument('--pairs-file', type=Path, help='offline bank pairs json (default: the live bank)')
    p.add_argument('--training-seed', type=int, default=SEED)
    p.add_argument('--passes', type=int, default=5)
    p.add_argument('--coefficient', type=float, default=1.)
    p.add_argument('--no-v2v', action='store_true')
    p.add_argument('--intent', default='auto', choices=('auto', 'on', 'off'))
    p.add_argument('--phase', default=PHASE)
    p.add_argument('--source-root', type=Path,
                   help='deployed source root whose git HEAD is sealed as the gate commit (default: this checkout)')
    p.add_argument('--source-manifest', type=Path,
                   help='qualified source manifest of a packaged release (commit + sha256 sealed instead of git HEAD)')
    a = p.parse_args(argv)
    if a.cmd == 'status':
        control = Path(a.control)
        out = {name: (json.loads((control/name).read_text())
                      if (control/name).exists() else None) for name in (PLAN_FILE, GATE_FILE)}
        out['planned_jobs'] = out[PLAN_FILE].get('totals', {}).get('jobs') if out[PLAN_FILE] else None
        out['graded_jobs'] = len(list((control/'branches').glob('*/*.graded.json'))) \
            if (control/'branches').exists() else 0
        return out
    if a.cmd == 'execute':
        return execute(a.control, a.root)
    if a.cmd == 'collect':
        return collect(a.control)
    if not a.root:
        p.error('plan requires --root')
    return plan(a.root, a.arm, a.control, training_seed=a.training_seed, pairs_source=a.pairs_file,
                sources=parse_sources(a.sources) or None, v2v=not a.no_v2v, intent=a.intent, passes=a.passes,
                coefficient=a.coefficient, u0=a.u0, u3=a.u3, phase=a.phase,
                source_root=a.source_root, source_manifest=a.source_manifest)


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=str))
