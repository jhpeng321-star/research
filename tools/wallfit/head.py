"""SynergyHead: a small correction head on top of the FROZEN world model (W3 option b).

Per (situation, joint move) the head predicts, in slog units (features.slog):
    reg[0] synergy  = J(best solo) - J(group)      (ranking score)
    reg[1] group advantage vs policy
    reg[2] best single-member solo advantage vs policy
    admit_logit     P(per-seed synergy LB95 > 0)
Each regression output is a RESIDUAL on the WM's own prediction of the same quantity:
    reg_k = gain_k * slog(wm_k) * wm_mask + mlp_k(observable features, pooled WM latent)
with the MLP's last layer zero-initialised and gain = 1, so an UNTRAINED head ranks exactly like the
current WM (where the WM has a prediction). The WM itself is never modified or retrained.

Loss: Huber on slog synergy (+ aux_weight x Huber on the two advantages) + rank_weight x pairwise
logistic ranking loss on synergy + admit_weight x BCE on the admitted rate. Seed-ensemble of
`ensemble` small MLPs; full-batch Adam; fixed epochs (no early stopping on test data).
"""
from contextlib import contextmanager
import time

import numpy as np
import torch
from torch import nn

from . import features as ft

SCHEMA = 'wallfit-head-v1'
TARGETS = ('synergy', 'group_adv', 'best_solo_adv')
DEFAULT_CONFIG = dict(hidden=32, bottleneck=8, dropout=0.1, epochs=300, lr=1e-2, weight_decay=1e-3, ensemble=5,
                      aux_weight=0.25, rank_weight=1.0, admit_weight=0.5, rank_margin=0.05, max_rank_pairs=200000,
                      seed=20261106, device='cpu', cpu_threads=1)


class SynergyHead(nn.Module):
    def __init__(self, n_obs, n_lat, hidden=32, bottleneck=8, dropout=0.1):
        super().__init__()
        self.n_obs, self.n_lat = int(n_obs), int(n_lat)
        self.lat = nn.Linear(self.n_lat, bottleneck) if self.n_lat else None
        width = self.n_obs + 1 + ((bottleneck + 1) if self.n_lat else 0)
        self.trunk = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 4))
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)
        self.wm_gain = nn.Parameter(torch.ones(3))

    def forward(self, obs, lat, lat_mask, wm, wm_mask):
        parts = [obs, wm_mask[:, None]]
        if self.lat is not None:
            parts += [torch.tanh(self.lat(lat)) * lat_mask[:, None], lat_mask[:, None]]
        out = self.trunk(torch.cat(parts, -1))
        reg = self.wm_gain * wm * wm_mask[:, None] + out[:, :3]
        return reg, out[:, 3]


# ------------------------------------------------------------------ arrays

def latent_for(pair, latents):
    sit = pair.get('situation') or {}
    return (latents or {}).get(sit.get('query_id')) if sit.get('source') == 'query' else None


def latent_dim_of(latents):
    for v in (latents or {}).values():
        return ft.latent_row_dim(v)
    return 0


def inputs(pairs, latents=None, latent_dim=0):
    rows = [ft.example_inputs(p, latent_for(p, latents), latent_dim) for p in pairs]
    return dict(obs=np.stack([r['obs'] for r in rows]), wm=np.stack([r['wm'] for r in rows]),
                wm_mask=np.asarray([r['wm_mask'] for r in rows], np.float32),
                lat=np.stack([r['lat'] for r in rows]) if latent_dim else np.zeros((len(rows), 0), np.float32),
                lat_mask=np.asarray([r['lat_mask'] for r in rows], np.float32))


def targets(pairs):
    y = np.zeros((len(pairs), 3), np.float32)
    ym = np.zeros((len(pairs), 3), np.float32)
    w = np.zeros(len(pairs), np.float32)
    adm = np.zeros(len(pairs), np.float32)
    adm_m = np.zeros(len(pairs), np.float32)
    for i, p in enumerate(pairs):
        lab = p['labels']
        for j, key in enumerate(TARGETS):
            if lab.get(key) is not None:
                y[i, j], ym[i, j] = ft.slog(lab[key]), 1.
        w[i] = np.sqrt(max(1, lab.get('n_measured') or 0))
        if lab.get('admitted_rate') is not None:
            adm[i], adm_m[i] = lab['admitted_rate'], 1.
    return dict(y=y, y_mask=ym, w=w, admit=adm, admit_mask=adm_m)


def _stats(x, mask=None):
    if x.shape[1] == 0:
        return np.zeros(0, np.float32), np.ones(0, np.float32)
    sel = x if mask is None or mask.sum() == 0 else x[mask > 0]
    mu = sel.mean(0)
    sd = sel.std(0)
    sd = np.where(sd < 1e-3, 1., sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def _t(x, device):
    return torch.as_tensor(np.asarray(x, np.float32), device=device)


# ------------------------------------------------------------------ training

def _loss(model, X, T, cfg):
    reg, logit = model(X['obs'], X['lat'], X['lat_mask'], X['wm'], X['wm_mask'])
    hub = nn.functional.smooth_l1_loss(reg, T['y'], reduction='none') * T['y_mask']
    w = T['w'][:, None]
    syn = (hub[:, 0] * w[:, 0]).sum() / (T['y_mask'][:, 0] * w[:, 0]).sum().clamp_min(1.)
    aux = (hub[:, 1:] * w).sum() / (T['y_mask'][:, 1:] * w).sum().clamp_min(1.)
    loss = syn + cfg['aux_weight'] * aux
    if cfg['rank_weight'] and T['pairs'] is not None:
        i, j = T['pairs']
        s = reg[:, 0]
        loss = loss + cfg['rank_weight'] * nn.functional.softplus(-(s[i] - s[j])).mean()
    if cfg['admit_weight'] and T['admit_mask'].sum() > 0:
        bce = nn.functional.binary_cross_entropy_with_logits(logit, T['admit'], reduction='none')
        loss = loss + cfg['admit_weight'] * (bce * T['admit_mask']).sum() / T['admit_mask'].sum()
    return loss


def _rank_pairs(y, mask, cfg, gen):
    idx = np.flatnonzero(mask > 0)
    if len(idx) < 2:
        return None
    a, b = np.meshgrid(idx, idx, indexing='ij')
    a, b = a.ravel(), b.ravel()
    keep = (y[a] - y[b]) > cfg['rank_margin']
    a, b = a[keep], b[keep]
    if not len(a):
        return None
    if len(a) > cfg['max_rank_pairs']:
        sel = gen.choice(len(a), cfg['max_rank_pairs'], replace=False)
        a, b = a[sel], b[sel]
    return a, b


@contextmanager
def _threads(cfg):
    """Tiny full-batch models: intra-op threading only adds overhead on CPU (restored after)."""
    old = torch.get_num_threads()
    if str(cfg.get('device', 'cpu')) == 'cpu' and cfg.get('cpu_threads'):
        torch.set_num_threads(int(cfg['cpu_threads']))
    try:
        yield
    finally:
        torch.set_num_threads(old)


def fit(pairs, latents=None, config=None):
    """Train the seed ensemble on `pairs` (TRAIN-split pairs only; the caller owns the split)."""
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    if not pairs:
        raise ValueError('no training pairs')
    with _threads(cfg):
        return _fit(pairs, latents, cfg)


def _fit(pairs, latents, cfg):
    begin = time.time()
    device = torch.device(cfg['device'])
    latent_dim = latent_dim_of(latents)
    raw = inputs(pairs, latents, latent_dim)
    tg = targets(pairs)
    obs_mu, obs_sd = _stats(raw['obs'])
    lat_mu, lat_sd = _stats(raw['lat'], raw['lat_mask'])
    gen = np.random.default_rng(cfg['seed'])
    rp = _rank_pairs(tg['y'][:, 0], tg['y_mask'][:, 0], cfg, gen)
    X = dict(obs=_t((raw['obs'] - obs_mu) / obs_sd, device), wm=_t(raw['wm'], device),
             wm_mask=_t(raw['wm_mask'], device), lat=_t((raw['lat'] - lat_mu) / lat_sd if latent_dim else raw['lat'], device),
             lat_mask=_t(raw['lat_mask'], device))
    T = dict(y=_t(tg['y'], device), y_mask=_t(tg['y_mask'], device), w=_t(tg['w'], device),
             admit=_t(tg['admit'], device), admit_mask=_t(tg['admit_mask'], device),
             pairs=None if rp is None else (torch.as_tensor(rp[0], device=device), torch.as_tensor(rp[1], device=device)))
    states, losses = [], []
    for m in range(int(cfg['ensemble'])):
        torch.manual_seed(int(cfg['seed']) + 1009 * m)
        model = SynergyHead(raw['obs'].shape[1], raw['lat'].shape[1], cfg['hidden'], cfg['bottleneck'],
                            cfg['dropout']).to(device)
        decay = [p for n, p in model.named_parameters() if n != 'wm_gain']
        opt = torch.optim.Adam([dict(params=decay, weight_decay=cfg['weight_decay']),
                                dict(params=[model.wm_gain], weight_decay=0.)], lr=cfg['lr'])
        model.train()
        for _ in range(int(cfg['epochs'])):
            opt.zero_grad()
            loss = _loss(model, X, T, cfg)
            loss.backward()
            opt.step()
        losses.append(float(loss.detach()))
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    return dict(schema=SCHEMA, config=cfg, n_obs=int(raw['obs'].shape[1]), latent_dim=int(latent_dim),
                n_lat=int(raw['lat'].shape[1]),
                input_names=list(ft.INPUT_NAMES), obs_mean=torch.as_tensor(obs_mu), obs_std=torch.as_tensor(obs_sd),
                lat_mean=torch.as_tensor(lat_mu), lat_std=torch.as_tensor(lat_sd), states=states,
                final_losses=losses, n_train_pairs=len(pairs),
                train_seeds=sorted({int(p['world_seed']) for p in pairs}), fit_wall_s=time.time() - begin)


def untrained(n_obs=len(ft.INPUT_NAMES), latent_dim=0, config=None):
    """A head at initialisation (== the WM's own ranking where the WM has a prediction)."""
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    torch.manual_seed(int(cfg['seed']))
    model = SynergyHead(n_obs, 2 * latent_dim, cfg['hidden'], cfg['bottleneck'], cfg['dropout'])
    return dict(schema=SCHEMA, config=cfg, n_obs=n_obs, latent_dim=latent_dim, n_lat=2 * latent_dim,
                input_names=list(ft.INPUT_NAMES), obs_mean=torch.zeros(n_obs), obs_std=torch.ones(n_obs),
                lat_mean=torch.zeros(2 * latent_dim), lat_std=torch.ones(2 * latent_dim),
                states=[model.state_dict()], final_losses=[], n_train_pairs=0,
                train_seeds=[], fit_wall_s=0.)


def predict(bundle, pairs, latents=None):
    """Ensemble-mean predictions: score (slog synergy), synergy_veh_s, group/solo advantages, admit_prob."""
    if not pairs:
        return []
    if list(bundle['input_names']) != list(ft.INPUT_NAMES):
        raise ValueError('head feature schema differs from this code (retrain or pin the matching source)')
    cfg = bundle['config']
    raw = inputs(pairs, latents, bundle['latent_dim'])
    obs = (raw['obs'] - bundle['obs_mean'].numpy()) / bundle['obs_std'].numpy()
    lat = (raw['lat'] - bundle['lat_mean'].numpy()) / bundle['lat_std'].numpy() if bundle['latent_dim'] else raw['lat']
    X = [_t(obs, 'cpu'), _t(lat, 'cpu'), _t(raw['lat_mask'], 'cpu'), _t(raw['wm'], 'cpu'), _t(raw['wm_mask'], 'cpu')]
    regs, probs = [], []
    with torch.no_grad():
        for state in bundle['states']:
            model = SynergyHead(bundle['n_obs'], bundle['n_lat'], cfg['hidden'], cfg['bottleneck'], cfg['dropout'])
            model.load_state_dict(state)
            model.eval()
            reg, logit = model(*X)
            regs.append(reg.numpy())
            probs.append(torch.sigmoid(logit).numpy())
    reg = np.mean(regs, 0)
    prob = np.mean(probs, 0)
    return [dict(score=float(r[0]), synergy_veh_s=ft.inv_slog(r[0]), group_adv_veh_s=ft.inv_slog(r[1]),
                 best_solo_adv_veh_s=ft.inv_slog(r[2]), admit_prob=float(q), latent_used=bool(lm), wm_used=bool(wm))
            for r, q, lm, wm in zip(reg, prob, raw['lat_mask'], raw['wm_mask'])]


def save(bundle, path):
    torch.save(bundle, str(path))


def load(path):
    bundle = torch.load(str(path), map_location='cpu', weights_only=True)
    if bundle.get('schema') != SCHEMA:
        raise ValueError(f'not a wallfit head: {path}')
    return bundle
