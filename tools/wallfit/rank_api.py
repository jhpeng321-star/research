"""Probe/mining-compatible ranking call; the W3 head replaces the WM-only ranking ONLY when enabled.

DEFAULT OFF: with no head argument and the environment variable WALLFIT_HEAD unset/empty,
rank_groups() returns exactly the coop_probe ranking (tools/mining/probe_wm.py at c4/coop-probe
5e6eb9e4: paired-per-draw synergy = min solo team delay - group team delay; sorted by
(-wm_mean_synergy, -wm_saving_vs_policy)), and sort_key() reproduces the consumers' sort
(probe_plan.screen/auto). Nothing heavier than json is imported on the OFF path (no torch).

ENABLED (WALLFIT_HEAD=/path/HEAD.pt or head=<bundle|path>): every WM row keeps all its keys and
values and gains head_score / head_synergy_veh_s / head_admit_prob / head_latent_used / ranker;
rows are sorted by sort_key (head score first, then the WM keys).

Integration (two lines, no other change):
    probe_wm.main:   ranked = rank_api.rank_groups(sc, d['groups'], query=query, actions=actions, H=H, rt=rt)
    probe_plan:      ranked.sort(key=rank_api.sort_key)     # instead of the inline WM key
Mining lane rows (wm_lane_worker2.rank output) can be annotated with annotate_rows(...).
"""
import os

ENV = 'WALLFIT_HEAD'
_CACHE = {}


def head_path(head=None):
    if head is not None:
        return head
    return os.environ.get(ENV) or None


def enabled(head=None):
    return head_path(head) is not None


def wm_rank(sc, groups):
    """The current coop_probe WM ranking, unchanged."""
    def mean(xs):
        return sum(x['team_delay'] for x in xs)/len(xs)
    ranked = []
    for gid, g in groups.items():
        joint = sc[gid]
        paired = [min(sc[s][k]['team_delay'] for s in g['solos'])-joint[k]['team_delay']
                  for k in range(len(joint))]
        ranked.append(dict(group=gid, plan_id=g['plan_id'], wm_mean_synergy=sum(paired)/len(paired),
                           wm_synergy_draws=paired, wm_saving_vs_policy=mean(sc['policy'])-mean(joint),
                           wm_best_solo_saving=max(mean(sc['policy'])-mean(sc[s]) for s in g['solos'])))
    ranked.sort(key=lambda r: (-r['wm_mean_synergy'], -r['wm_saving_vs_policy']))
    return ranked


def sort_key(row):
    """Consumer sort key: identical to the WM key for rows without a head score."""
    if row.get('head_score') is not None:
        return (-row['head_score'], -row['wm_mean_synergy'], -row['wm_saving_vs_policy'])
    return (-row['wm_mean_synergy'], -row['wm_saving_vs_policy'])


def _bundle(head):
    if isinstance(head, dict):
        return head
    key = str(head)
    if key not in _CACHE:
        from . import guard, head as hd
        _CACHE[key] = hd.load(guard.check_path(key))
    return _CACHE[key]


def _latent(query, rt, latent):
    if latent is not None or rt is None or query is None:
        return latent
    from .latent import extract
    try:
        return {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in extract(rt, query).items()}
    except Exception:
        return None       # recorded per row as head_latent_used=False; never imputed


def score_rows(rows, moves, query, H, *, rt=None, latent=None, head=None):
    """Head predictions for WM rows (same order). rows carry wm_mean_synergy etc.; moves are the
    observable move dicts (features.normalize_move)."""
    from . import head as hd
    bundle = _bundle(head_path(head))
    lat = _latent(query, rt, latent)
    latents = {'q': lat} if lat is not None else None
    sit_rows = [{k: r.get(k) for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')}
                for r in (query or {}).get('record', {}).get('rows', [])]
    pairs = []
    for r, mv in zip(rows, moves):
        pairs.append(dict(world_seed=None, move=mv,
                          situation=dict(source='query' if lat is not None else 'context', query_id='q', rows=sit_rows),
                          wm=dict(mean_synergy=r['wm_mean_synergy'], saving_vs_policy=r.get('wm_saving_vs_policy'),
                                  best_solo_saving=r.get('wm_best_solo_saving'), draws=r.get('wm_synergy_draws'),
                                  horizon_s=H)))
    return hd.predict(bundle, pairs, latents)


def _annotate(row, pred, path):
    return dict(row, head_score=pred['score'], head_synergy_veh_s=pred['synergy_veh_s'],
                head_admit_prob=pred['admit_prob'], head_latent_used=pred['latent_used'],
                ranker='wallfit_head', head_path=str(path) if not isinstance(path, dict) else 'in-memory')


def rank_groups(sc, groups, *, query=None, actions=None, H=None, rt=None, latent=None, head=None):
    """Drop-in for the coop_probe ranking block. OFF (default) == wm_rank(sc, groups)."""
    ranked = wm_rank(sc, groups)
    if not enabled(head):
        return ranked
    from .features import move_from_action, normalize_move, parse_plan_id
    acts = {a['id']: a for a in actions or []}
    moves = []
    for r in ranked:
        a = acts.get(r['group'])
        moves.append(move_from_action(a, r['plan_id']) if a else normalize_move(parse_plan_id(r['plan_id'])))
    H = int(H) if H is not None else 120
    preds = score_rows(ranked, moves, query, H, rt=rt, latent=latent, head=head)
    out = [_annotate(r, p, head_path(head)) for r, p in zip(ranked, preds)]
    out.sort(key=sort_key)
    return out


def annotate_rows(rows, query, *, H=300, rt=None, latent=None, head=None):
    """Mining lane rank rows (family/members/onset/hold/depth + wm_*): add head fields and re-sort.
    OFF (default) returns the rows unchanged (same object)."""
    if not enabled(head):
        return rows
    from .features import normalize_move
    moves = [normalize_move(dict(family=r.get('family'), members=r.get('members'), onset=r.get('onset'),
                                 hold=r.get('hold'), depth=r.get('depth'))) for r in rows]
    preds = score_rows(rows, moves, query, H, rt=rt, latent=latent, head=head)
    out = [_annotate(r, p, head_path(head)) for r, p in zip(rows, preds)]
    out.sort(key=lambda r: sort_key(r) + (r.get('wm_action_id', ''),))
    return out
