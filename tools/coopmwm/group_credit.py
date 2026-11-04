"""T2E counterfactual GROUP credit for the cooperation layer (CPU only).

credit = WM-estimated (group outcome WITH move) - (group outcome WITHOUT move, base
MAPPO), decomposed "together vs alone" exactly like the isolated-synergy diagnostic:
joint vs each member solo vs all-base, giving D_i (alone contributions), D_G (group
effect) and C_G = D_G - sum(D_i) (complementarity). The layer's advantage signal
replaces the per-CAV reward FOR THE LAYER ONLY. Everything is computed from belief
state / observables (fail-closed guard refuses truth-shaped keys); a registered
fraction of estimates is emitted as requests for the existing all8/isolated SUMO
verification machinery - this module never runs SUMO itself.
"""
import math
import numpy as np

REGISTRATION = dict(
    metric='team_delay', direction='lower', wm_samples=4,
    sumo_verify_fraction=0.25, verify_mode='all8_both_seeds', isolated=True,
    c_g_share=0.5, truth_patterns=('truth', 'hidden', 'oracle', 'global_state',
                                   'vtype', 'driver'),
)
CONFIG_KEYS = frozenset(REGISTRATION) - {'truth_patterns'}


def credit_config(cfg=None):
    cfg = dict(cfg or {})
    unknown = sorted(set(cfg) - CONFIG_KEYS)
    if unknown:
        raise ValueError('unregistered group_credit keys: %r' % (unknown,))
    out = {k: REGISTRATION[k] for k in CONFIG_KEYS}
    for k, v in cfg.items():
        out[k] = v
    if out['metric'] != 'team_delay' or out['direction'] != 'lower':
        raise ValueError("registered metric team_delay, direction 'lower' only")
    for k in ('wm_samples',):
        if type(out[k]) is not int or out[k] < 1:
            raise ValueError('%s positive int required' % k)
    for k in ('sumo_verify_fraction', 'c_g_share'):
        v = out[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) \
                or not math.isfinite(v) or not 0. <= float(v) <= 1.:
            raise ValueError('%s in [0,1] required' % k)
    if out['verify_mode'] != 'all8_both_seeds' or out['isolated'] is not True:
        raise ValueError("verify_mode 'all8_both_seeds' with isolated=True required")
    return out


def assert_observable(payload, patterns=None):
    """Fail-closed: refuse payloads carrying truth-shaped keys, at any depth."""
    pats = REGISTRATION['truth_patterns'] if patterns is None else patterns
    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                kl = str(k).lower()
                if any(p in kl for p in pats):
                    raise ValueError('non-observable key %r at %s' % (k, path))
                walk(v, path + '/' + str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, path + '[%d]' % i)
    walk(payload, '')


def _mean(rows):
    for x in rows:
        if not math.isfinite(float(x)):
            raise ValueError('non-finite WM outcome')
    return float(np.mean(rows))


def _child(rng):
    return np.random.default_rng(int(rng.integers(0, 2 ** 31 - 1)))


def wm_estimate(scorer, belief, members, move, rng, cfg=None, draw_id=None):
    """Multiple WM samples over belief-only scenarios: base / joint / each member solo.

    scorer(belief, intervention, rng_child) -> outcome (team_delay, lower better).
    Interventions: {'kind':'base'} | {'kind':'joint','move':move}
    | {'kind':'alone','member':m,'move':move}. The move is opaque here.
    """
    cfg = credit_config(cfg)
    assert_observable(belief)
    members = list(members)
    if not members or len(set(map(str, members))) != len(members):
        raise ValueError('non-empty unique member list required')
    base, joint, alone = [], [], [[] for _ in members]
    for _ in range(cfg['wm_samples']):
        child = _child(rng)
        base.append(scorer(belief, {'kind': 'base'}, child))
        joint.append(scorer(belief, {'kind': 'joint', 'move': move}, child))
        for i, m in enumerate(members):
            child = _child(rng)
            alone[i].append(scorer(belief, {'kind': 'alone', 'member': m,
                                            'move': move}, child))
    b, j = _mean(base), _mean(joint)
    a = [_mean(x) for x in alone]
    # Same decomposition as the isolated-synergy diagnostic (credit.algebra):
    # D_i[i] = solo_i - joint (member i's alone contribution, delay units),
    # D_G = base - joint (group effect), C_G = D_G - sum(D_i) (together-vs-alone
    # complementarity). Positive = delay reduction.
    d_i = [x - j for x in a]
    d_g = b - j
    c_g = d_g - sum(d_i)
    return dict(draw_id=draw_id, members=members, move=move,
                outcome=dict(base=b, joint=j, alone=a),
                credit=dict(group_credit=j - b, improvement=b - j,
                            D_i=d_i, D_G=d_g, C_G=c_g))


def layer_advantage(estimate, cfg=None):
    """Per-member advantage replacing the per-CAV reward for the LAYER ONLY.

    adv_i = D_i[i] + c_g_share * C_G / n_members (own alone contribution + equal
    share of the together-only complementarity); team scalar follows the registered
    credit.bonus shape mean(D_i) + c_g_share * C_G.
    """
    cfg = credit_config(cfg)
    credit = estimate['credit']
    n = len(estimate['members'])
    share = cfg['c_g_share'] * credit['C_G']
    per = [d + share / n for d in credit['D_i']]
    team = float(np.mean(credit['D_i'])) + cfg['c_g_share'] * credit['C_G']
    return dict(members=list(estimate['members']), per_member=per, team=team)


def verify_requests(estimates, rng, cfg=None):
    """Bernoulli(sumo_verify_fraction) selection -> requests for the existing
    all8/isolated SUMO verification machinery (consumed later; never run here)."""
    cfg = credit_config(cfg)
    out = []
    for e in estimates:
        if not bool(rng.random() < cfg['sumo_verify_fraction']):
            continue
        out.append(dict(kind='group_credit_verification',
                        verify_mode=cfg['verify_mode'], isolated=cfg['isolated'],
                        draw_id=e['draw_id'], members=list(e['members']),
                        move=e['move'], wm=dict(e['credit'])))
    return out


def verified_stats(records):
    """WM vs SUMO-exact credit monitoring (CPU, offline only - never a PPO input,
    same rule as credit.calibration). Records need wm.credit and exact.credit."""
    if not records:
        raise ValueError('verification records required')
    pairs, diffs = [], []
    for r in records:
        x, y = r['wm']['D_G'], r['exact']['D_G']
        if x != 0 and y != 0:
            pairs.append((x, y))
        diffs.append(y - x)
    hits = sum((x > 0) == (y > 0) for x, y in pairs)
    return dict(n=len(records), paired=len(pairs), ties=len(records) - len(pairs),
                sign_agreement=hits / len(pairs) if pairs else None,
                mae_D_G=float(np.mean(np.abs(diffs))))
