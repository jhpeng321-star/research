"""LEARNED SCREEN: an amortised WM screen for the MPPI plan search (arm B; owner-gated).

Registration: docs/paper_results/LEARNED_SCREEN_PREREG.md (committed before any held-out result).

The network predicts a candidate plan's within-search standardised WM screen saving
vs 'policy continues' (K4/H300). Its inputs are the observable member features
(``plan_proposer.member_features``) and the plan (canonical member x 10 perturbation,
knot mask, D/300). At search time ``mppi_search.search(..., screen=screen_fn(...),
config=dict(top_k=8))`` samples exactly as the plain search does, scores the
candidates with the network instead of WM rollouts, and sends the top 16 to the
unchanged K8/H_final WM stage. CPU only; deterministic; SHA256 checkpoints.
"""
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import learned_search_diversity as ld
from . import learned_search_eval as le
from . import mppi_search as ms
from . import plan_proposer as pp

N_KNOTS = ms.canonical_knots()
TOKEN = pp.FEATURES+2*N_KNOTS+1
REGISTERED = dict(d_model=64, layers=2, heads=4, steps=2000, lr=1e-3, weight_decay=1e-4, seed=20261102,
                  batch_searches=16, batch_candidates=48, gold_weight=2., listnet_weight=1., eval_every=100,
                  inner_frac=.2, inner_min_worlds=5, threads=1, top_k=8, min_pool=16,
                  min_heldout_worlds=8, min_heldout_searches=40, bootstraps=2000, collapse_ratio=.75,
                  random_subsets=200)
SEARCH_WALL_S = {2: 559.567, 3: 794.766, 4: 1029.964, 5: 964.120, 6: 898.275}   # COOPMWM.md measured


def plan_tokens(x, z, D):
    """m x TOKEN: [state features | own perturbation row (Z units) | knot mask | D/300]."""
    z = np.asarray(z, float)/pp.Z_SCALE; m = z.shape[0]
    km = np.tile(pp.knot_mask(D).astype(float), (m, 1))
    return np.concatenate([np.asarray(x, float), z, km, np.full((m, 1), D/300.)], 1).astype(np.float32)


# --------------------------------------------------------------------------- data

def search_unit(receipt, cfg=REGISTERED):
    """Receipt -> {x, items: [(z m/s^2, D, y)], y standardised within search, gold lists}; None if unusable."""
    try:
        members = list(receipt['members']); m = len(members)
        if not 2 <= m <= 6:
            return None
        x = pp.member_features(receipt['rows'], members, receipt.get('regime') or 'unknown')
    except (KeyError, ValueError, TypeError):
        return None
    seen, items = set(), []
    for c in receipt.get('candidates') or []:
        if c.get('stage', 'screen') != 'screen' or c.get('id') in seen or c.get('saving_vs_policy') is None:
            continue
        z = np.asarray(c.get('perturbation_canonical'), float)
        if z.shape != (m, N_KNOTS) or not np.isfinite(z).all() or not math.isfinite(float(c['saving_vs_policy'])):
            continue
        seen.add(c['id']); items.append((c['id'], z, int(c['duration_s']), float(c['saving_vs_policy']), c.get('origin')))
    if len(items) < cfg['min_pool']:
        return None
    s = np.array([i[3] for i in items]); sd = float(s.std()) or 1.
    unit = dict(id=str(receipt.get('search_id')), world=str(receipt.get('world_id')), m=m, x=x,
                z=np.stack([i[1] for i in items]), D=np.array([i[2] for i in items]), saving=s,
                y=(s-s.mean())/sd, ids=[i[0] for i in items], origin=[i[4] for i in items], gold=None)
    sv = receipt.get('sumo_verification') or {}
    if sv.get('J_candidates_veh_s') is not None:
        byid = {c.get('id'): c for c in receipt.get('candidates') or []}
        gz, gD, gg = [np.zeros((m, N_KNOTS))], [300], [0.]           # 'policy continues' = zero perturbation
        for cid, j in zip(sv['candidate_ids'], sv['J_candidates_veh_s']):
            c = byid.get(cid)
            if c is not None and np.shape(c.get('perturbation_canonical')) == (m, N_KNOTS):
                gz.append(np.asarray(c['perturbation_canonical'], float)); gD.append(int(c['duration_s']))
                gg.append(float(sv['J_policy_veh_s'])-float(j))
        if len(gg) > 2:
            unit['gold'] = dict(z=np.stack(gz), D=np.array(gD), saving=np.array(gg))
    return unit


def build_units(receipts, cfg=REGISTERED):
    out = [search_unit(r, cfg) for r in receipts]
    return [u for u in out if u is not None]


# --------------------------------------------------------------------------- model

class ScreenNet(nn.Module):
    """Permutation-equivariant set network: candidate plan + state -> scalar score."""

    def __init__(self, cfg=REGISTERED):
        super().__init__()
        self.cfg = dict(cfg); d = cfg['d_model']
        self.embed = nn.Sequential(nn.Linear(TOKEN, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, cfg['heads'], 2*d, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, tok, mask):
        h = self.encoder(self.embed(tok), src_key_padding_mask=~mask)
        mf = mask.unsqueeze(-1).float()
        mean = (h*mf).sum(1)/mf.sum(1).clamp_min(1.)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


def _tensor(x, Z, D):
    m = x.shape[0]; P = ms.REGISTERED['max_size']
    tok = np.zeros((len(Z), P, TOKEN), np.float32); mask = np.zeros((len(Z), P), bool)
    for i, (z, d) in enumerate(zip(Z, D)):
        tok[i, :m] = plan_tokens(x, z, int(d)); mask[i, :m] = True
    return torch.from_numpy(tok), torch.from_numpy(mask)


def predict(model, x, Z, D, chunk=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), chunk):
            out.append(model(*_tensor(x, Z[i:i+chunk], D[i:i+chunk])).double().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _listnet(pred, target):
    p = torch.softmax(torch.as_tensor(target, dtype=pred.dtype), 0)
    return -(p*torch.log_softmax(pred, 0)).sum()


def _loss(model, units, gen, cfg):
    total, n = 0., 0
    for u in units:
        k = min(cfg['batch_candidates'], len(u['y']))
        idx = torch.randperm(len(u['y']), generator=gen)[:k].numpy()
        pred = model(*_tensor(u['x'], u['z'][idx], u['D'][idx]))
        y = torch.as_tensor(u['y'][idx], dtype=pred.dtype)
        total = total+((pred-y)**2).mean()+cfg['listnet_weight']*_listnet(pred, u['y'][idx]); n += 1
        g = u.get('gold')
        if g is not None:
            gp = model(*_tensor(u['x'], g['z'], g['D'])); gs = g['saving']
            gstd = (gs-gs.mean())/(gs.std() or 1.)
            total = total+cfg['gold_weight']*_listnet(gp, gstd)
    return total/max(n, 1)


def train(units, cfg=None):
    cfg = dict(REGISTERED, **(cfg or {}))
    if not units:
        raise ValueError('no training search with a full screened pool')
    fit, inner = units, []
    if len({u['world'] for u in units}) >= cfg['inner_min_worlds']:
        fit, inner, _ = pp.world_split(units, cfg['inner_frac'], cfg['seed']+1)
    torch.manual_seed(cfg['seed'])
    prev = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.use_deterministic_algorithms(True); torch.set_num_threads(cfg['threads'])
    try:
        model = ScreenNet(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        gen = torch.Generator().manual_seed(cfg['seed']); egen_seed = cfg['seed']+7
        best, log = (math.inf, -1, None), []
        for step in range(cfg['steps']):
            model.train()
            pick = torch.randint(len(fit), (min(cfg['batch_searches'], len(fit)),), generator=gen).tolist()
            loss = _loss(model, [fit[i] for i in pick], gen, cfg)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            if (step+1) % cfg['eval_every'] == 0 or step == cfg['steps']-1:
                row = dict(step=step+1, loss=float(loss.detach()))
                if inner:
                    model.eval()
                    with torch.no_grad():
                        row['inner'] = float(_loss(model, inner, torch.Generator().manual_seed(egen_seed), cfg))
                    if row['inner'] < best[0]:
                        best = (row['inner'], step+1, {k: v.clone() for k, v in model.state_dict().items()})
                log.append(row)
        if best[2] is not None:
            model.load_state_dict(best[2])
        model.eval()
        return model, dict(train_searches=len(units), fit=len(fit), inner=len(inner),
                           selected_step=best[1] if best[2] is not None else cfg['steps'], log=log)
    finally:
        torch.set_num_threads(prev[0]); torch.use_deterministic_algorithms(prev[1])


def save(model, meta, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); raw = buf.getvalue()
    pt = path.with_suffix('.pt'); pt.write_bytes(raw)
    body = dict(meta, schema='coopmwm.learned_screen.checkpoint/1', cfg=model.cfg, token=TOKEN,
                pt_file=pt.name, pt_sha256=pp._sha_bytes(raw))
    body['json_sha256'] = pp._digest(body)
    path.with_suffix('.json').write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
    return body


def load(path):
    path = Path(path); meta = json.loads(path.with_suffix('.json').read_text())
    if pp._digest({k: v for k, v in meta.items() if k != 'json_sha256'}) != meta['json_sha256']:
        raise ValueError('learned-screen JSON SHA mismatch')
    raw = (path.parent/meta['pt_file']).read_bytes()
    if pp._sha_bytes(raw) != meta['pt_sha256']:
        raise ValueError('learned-screen .pt SHA mismatch')
    model = ScreenNet(meta['cfg']); model.load_state_dict(torch.load(io.BytesIO(raw), weights_only=True)); model.eval()
    return model, meta


def screen_fn(model, record, group, regime):
    """``mppi_search`` screen callable: predicted COST relative to the policy (lower = better)."""
    x = pp.member_features(record.get('rows') or group['rows'], group['members'], regime or 'unknown')
    timer = dict(s=0., n=0)

    def screen(rec, grp, cands):
        t = time.perf_counter()
        out = -predict(model, x, [np.asarray(c['perturbation_canonical'], float) for c in cands],
                       [int(c['duration_s']) for c in cands])
        timer['s'] += time.perf_counter()-t; timer['n'] += len(cands)
        return out
    screen.source = 'learned_screen'; screen.timer = timer
    return screen


def learned_screen_kwargs(model, record, group, regime):
    """Keyword arguments for the deployed learned-screen search (top-16 to the WM final)."""
    return dict(screen=screen_fn(model, record, group, regime), config=dict(top_k=REGISTERED['top_k']))


# --------------------------------------------------------------------------- offline check (registered)

def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else 0.


def replay(receipt, sur, mode, model=None, cfg=REGISTERED):
    """Real ``mppi_search`` on the surrogate. Returns screened (vals, zs, Ds), final-stage values,
    WM futures used and network seconds."""
    group = le._group(receipt); m = len(group['members'])
    policy = np.zeros((m, ms.REGISTERED['max_duration_s'])); regime = receipt.get('regime') or 'unknown'
    record = dict(state_id=str(receipt.get('search_id')), rows=receipt['rows'],
                  policy_residuals={v: 0. for v in group['members']})
    calls = []

    def evaluate(rec, actions, K, H):
        cands = [a for a in actions if 'perturbation_canonical' in a]
        y = sur([a['perturbation_canonical'] for a in cands], [a['duration_s'] for a in cands]) if cands else []
        val = {a['id']: float(v) for a, v in zip(cands, y)}
        calls.append([val[a['id']] for a in cands])
        return {a['id']: [-val.get(a['id'], 0.)]*K for a in actions}
    screened = []

    def surrogate_screen(rec, grp, cands):     # plain modes: surrogate as the (noise-free) WM screen
        v = sur([c['perturbation_canonical'] for c in cands], [c['duration_s'] for c in cands])
        screened.extend(zip(v, [np.asarray(c['perturbation_canonical'], float) for c in cands],
                            [int(c['duration_s']) for c in cands]))
        return -np.asarray(v)
    kw, seed_tag = {}, []
    if mode == 'ls':
        net = screen_fn(model, record, group, regime)

        def ls_screen(rec, grp, cands):
            v = sur([c['perturbation_canonical'] for c in cands], [c['duration_s'] for c in cands])
            screened.extend(zip(v, [np.asarray(c['perturbation_canonical'], float) for c in cands],
                                [int(c['duration_s']) for c in cands]))
            return net(rec, grp, cands)
        kw = dict(screen=ls_screen, config=dict(top_k=cfg['top_k'])); seed_tag = ['ls']
    else:
        kw = dict(screen=surrogate_screen); seed_tag = ['alt'] if mode == 'plain_alt' else []
    seed = int(ms._digest([cfg['seed'], receipt.get('search_id')]+seed_tag)[:15], 16)
    res = ms.search(record, group, policy, le.WIDE, evaluate, regime, seed=seed, **kw)
    vals = np.array([v for v, _, _ in screened]); zs = [z for _, z, _ in screened]; Ds = [d for _, _, d in screened]
    nfin = len(calls[-1]) if calls else 0
    futures = (nfin+2)*ms.REGISTERED['final_K']
    if mode != 'ls':     # plain: WM screen futures + final (the surrogate stood in for the K4 screen)
        sc = ms.scaling(m, N_KNOTS)
        futures += (sc['population']+2)*ms.REGISTERED['screen_K']*ms.REGISTERED['iterations']
    return dict(vals=vals, zs=zs, Ds=Ds, final=np.array(calls[-1] if calls else []), futures=futures,
                net_s=net.timer['s'] if mode == 'ls' else 0., finalists=nfin,
                res=res)


def offline_check(model, held, cfg=None):
    """The pre-registered check (LEARNED_SCREEN_PREREG.md)."""
    cfg = dict(REGISTERED, **(cfg or {}))
    rows = []
    for r in held:
        u = search_unit(r, cfg); sur = le.surrogate_for(r)
        if u is None or sur is None:
            continue
        s, _ = pp.build_search(r)
        rng = np.random.default_rng(int(pp._digest([cfg['seed'], 'ls', u['id']])[:12], 16))
        # ---- S1: real WM labels on the recorded pool
        pred = predict(model, u['x'], list(u['z']), list(u['D']))
        act = u['saving']
        centre = next((a for a, o in zip(act, u['origin']) if o == 'centre'), 0.)
        top = act.max()
        if top <= centre:
            continue
        q16 = (act[np.argsort(-pred, kind='stable')[:16]].max()-centre)/(top-centre)
        rnd = float(np.mean([(act[rng.choice(len(act), min(16, len(act)), replace=False)].max()-centre)/(top-centre)
                             for _ in range(cfg['random_subsets'])]))
        # ---- P: surrogate replays
        rp = {mode: replay(r, sur, mode, model, cfg) for mode in ('plain', 'plain_alt', 'ls')}
        base = float(sur([np.zeros((u['m'], N_KNOTS))], [300])[0])
        scale = rp['plain']['final'].max()-base
        if not scale > 0:
            continue
        q = {mode: float((v['final'].max()-base)/scale) for mode, v in rp.items()}
        # ---- S2 / diversity: elite clusters among each mode's screened candidates (by surrogate value)
        spacing = ld.nn_spacing(rp['plain']['zs'])
        E = ms.scaling(u['m'], N_KNOTS)['elites']
        div = {}
        for mode, v in rp.items():
            order = np.argsort(-v['vals'], kind='stable')
            te, t8 = order[:E], order[:8]
            div[mode] = dict(elite_clusters=ld.clusters([v['zs'][i] for i in te], spacing),
                             top8_clusters=ld.clusters([v['zs'][i] for i in t8], spacing),
                             top8_patterns=len({ld.member_pattern(v['zs'][i], v['Ds'][i]) for i in t8}),
                             top1_pattern=ld.member_pattern(v['zs'][order[0]], v['Ds'][order[0]]),
                             spread=ld.mean_pairwise(v['zs']))
        # ---- coverage of recorded (real WM) elites
        elites = [e['z']*pp.Z_SCALE for e in (s['elites'] if s else [])]

        def nearest(Z):
            Z = np.stack(Z)
            return [float(np.sqrt(((Z-e)**2).mean((1, 2))).min()) for e in elites]
        cov = {}
        if elites:
            dp = nearest(rp['plain']['zs'])
            for mode in ('plain_alt', 'ls'):
                cov[mode] = float(np.mean([a <= b+1e-12 for a, b in zip(nearest(rp[mode]['zs']), dp)]))
        wall_plain = SEARCH_WALL_S[u['m']]
        wall_ls = wall_plain*rp['ls']['futures']/rp['plain']['futures']+rp['ls']['net_s']
        rows.append(dict(id=u['id'], world=u['world'], m=u['m'], rho=_spearman(pred, act), q16=float(q16), q16_random=rnd,
                         q=q, div=div, coverage=cov, futures=dict(plain=rp['plain']['futures'], ls=rp['ls']['futures']),
                         net_s=rp['ls']['net_s'], wall_plain_s=wall_plain, wall_ls_s=wall_ls))
    if not rows:
        return dict(ready=False, enough_data=False, searches=0)
    worlds = [x['world'] for x in rows]
    ci = lambda v, k: dict(zip(('mean', 'ci_lo', 'ci_hi'), pp._bootstrap_ci(v, worlds, cfg['bootstraps'], cfg['seed']+50+k)))
    delta = float(np.mean([abs(x['q']['plain']-x['q']['plain_alt']) for x in rows]))
    P = ci([x['q']['ls']-x['q']['plain_alt'] for x in rows], 0)
    rho = ci([x['rho'] for x in rows], 1); q16 = ci([x['q16'] for x in rows], 2)
    q16r = float(np.mean([x['q16_random'] for x in rows]))
    ec = {mode: float(np.mean([x['div'][mode]['elite_clusters'] for x in rows])) for mode in ('plain', 'plain_alt', 'ls')}
    nw = len(set(worlds))
    enough = nw >= cfg['min_heldout_worlds'] and len(rows) >= cfg['min_heldout_searches']
    pass_p = P['ci_lo'] > -delta
    pass_s1 = rho['ci_lo'] > 0 and q16['ci_lo'] > q16r
    s2 = ec['ls'] >= cfg['collapse_ratio']*ec['plain_alt']
    from collections import Counter
    top1 = {mode: Counter(x['div'][mode]['top1_pattern'] for x in rows) for mode in ('plain', 'plain_alt', 'ls')}
    return dict(
        ready=bool(enough and pass_p and pass_s1 and s2), enough_data=enough, searches=len(rows), worlds=nw,
        P=dict(passed=bool(pass_p), q_ls_minus_plain_alt=P, delta=delta,
               q_mean={mode: float(np.mean([x['q'][mode] for x in rows])) for mode in ('plain', 'plain_alt', 'ls')}),
        S1=dict(passed=bool(pass_s1), spearman=rho, q16=q16, q16_random_mean=q16r),
        S2=dict(passed=bool(s2), elite_clusters=ec, ratio_ls_over_plain_alt=ec['ls']/ec['plain_alt'] if ec['plain_alt'] else None),
        coverage={mode: float(np.mean([x['coverage'][mode] for x in rows if mode in x['coverage']]))
                  for mode in ('plain_alt', 'ls') if any(mode in x['coverage'] for x in rows)},
        diversity={mode: dict(top8_clusters=float(np.mean([x['div'][mode]['top8_clusters'] for x in rows])),
                              top8_patterns=float(np.mean([x['div'][mode]['top8_patterns'] for x in rows])),
                              spread_ratio_vs_plain=float(np.mean([x['div'][mode]['spread']/x['div']['plain']['spread']
                                                                   for x in rows])),
                              top1_most_common_share=top1[mode].most_common(1)[0][1]/len(rows),
                              top1_distinct=len(top1[mode]))
                   for mode in ('plain', 'plain_alt', 'ls')},
        wall=dict(mean_plain_s=float(np.mean([x['wall_plain_s'] for x in rows])),
                  mean_ls_s=float(np.mean([x['wall_ls_s'] for x in rows])),
                  mean_net_cpu_s=float(np.mean([x['net_s'] for x in rows])),
                  futures_plain=float(np.mean([x['futures']['plain'] for x in rows])),
                  futures_ls=float(np.mean([x['futures']['ls'] for x in rows])),
                  basis='COOPMWM.md measured wall per query x WM-futures ratio + network CPU time; GPU-bound'),
        per_search=rows)
