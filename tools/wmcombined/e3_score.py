"""S-WMCOMBINED E3 paired scorer: WM-vs-real paired dJ on S-E3DATA-1 worlds.

Loads the (arm, native) CRN pairs of one S-E3DATA-1 world role (calibration or
validation, per its SPLIT.json - never 'fit') with the production
``PairedBranchSource`` (the exact loader of the longwm paired loss), rolls every arm
and its native reference 300 s with identity-keyed coupled randomness (lane uniforms
from ``PairedRandom``; no state noise; the v2 entry process and the recorded CAV
command timeline of the training convention), and reports the t0-HDV paired dJ
(``wm_eval_common.delay_proxy``, the registered horizon-gate objective) of WM vs the
recorded tapes at 10/30/60/90/120/180/300 s.

The output is a ``coopw_horizon_gate/1`` superset (rows/dJ/horizons/threshold), so the
S-LONGLOSSTEST verdict tooling (``paired_statistics.compare`` / ``amended.evaluate``)
applies unchanged; rows additionally carry arm identities for per-arm (metering)
slopes: MAPPO-vs-native and ours-vs-native separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HORIZONS = (10., 30., 60., 90., 120., 180., 300.)
DT_S = 0.5
STEPS = 600


def build_source(root, role):
    """PairedBranchSource restricted to one S-E3DATA-1 split role (never 'fit')."""
    from wms.data.c4_pairs import PairedBranchSource
    manifest = json.loads((Path(root)/'C4_EPISODES_MANIFEST.json').read_text())
    split = json.loads((Path(root)/'SPLIT.json').read_text())
    role_of_world = {}
    for e in manifest['episodes']:
        role_of_world[e['world']] = split['worlds'][str(e['seed'])]
    assert set(role_of_world.values()) <= {'fit', 'calibration', 'validation'}
    role_of = {w: ('train' if r == role else 'val') for w, r in role_of_world.items()}
    source = PairedBranchSource(
        root, horizon_steps=STEPS, history_s=60.0, history_stride_s=1.0,
        entries_v2=True, junction_fix='ffill', command_known=True,
        arm_kinds=('B', 'POLICY'), role='train', role_of=role_of,
        name=f'e3_{role}')
    r = source.receipt
    assert r['n_groups'] == 36 and r['n_windows'] == 144, r
    assert r['groups_with_native_reference'] == 36, r
    assert r['dropped_groups_t0_mismatch'] == 0 and r['dropped_groups_single_arm'] == 0, r
    return source


def _arm_name(source, row):
    ep = source.dataset.positions[int(row)][0]
    entry = source.dataset._loader.entries[ep]
    return str(entry.get('arm', '?'))


def group_rows(source, group_index):
    """(pair_key, reference row, [(arm_name, row), ...]) with a fixed sorted arm order."""
    g = source.groups[group_index]
    ref, arms = g['rows'][0], g['rows'][1:]
    named = sorted((_arm_name(source, r), r) for r in arms)
    return g['pair_key'], ref, named


def rollout_dj(model, settings, source, indexes, device, groups_per_batch=12,
                steps=STEPS, horizons=HORIZONS, entrant_controller_factory=None, samples=1):
    """Batched coupled rollouts -> {pair_key: {'arm_names': [...], 'dj': {h: rows}}}.

    ``dj[h]`` rows follow ``[reference, *arms]``: ``[wm_delay, real_delay]`` per row.
    ``steps``/``horizons`` can be reduced for plumbing smoke tests (preflight).
    """
    import torch
    from wms.data.collate import collate
    from tools.wm_eval_common import delay_proxy
    from wms.train.inputs import prepare_inputs

    if samples < 1:
        raise ValueError('samples must be positive')
    if samples > 1:
        from dataclasses import replace
        from wms.train.scoring_random import sample_seed
        draws = [rollout_dj(model, replace(settings, seed=sample_seed(settings.seed, k)),
                    source, indexes, device, groups_per_batch, steps, horizons,
                    entrant_controller_factory, samples=1) for k in range(samples)]
        return {key: {'arm_names': draws[0][key]['arm_names'],
                      'dj': {h: np.asarray([d[key]['dj'][h] for d in draws]).mean(0).tolist()
                             for h in draws[0][key]['dj']},
                      'draw_dj': [d[key]['dj'] for d in draws]} for key in draws[0]}

    was_training = model.training
    model.eval()
    out = {}
    try:
        for c0 in range(0, len(indexes), groups_per_batch):
            chunk = indexes[c0:c0 + groups_per_batch]
            records, layout = [], []
            for gi in chunk:
                key, ref, arms = group_rows(source, gi)
                base = len(records)
                records.extend(source.group_records(
                    gi, [ref] + [r for _, r in arms], base_row=base))
                layout.append((key, base, [a for a, _ in arms]))
            batch = collate(records)
            from wms.train.scoring_random import scoring_rng, bind_merge_crn
            feedback = (None if entrant_controller_factory is None else
                        entrant_controller_factory(records, model, device))
            with scoring_rng(settings.seed, device):
                inp = prepare_inputs(batch, model, settings, device, steps,
                                     entrant_controller=feedback)
            bind_merge_crn(inp)
            pr = inp.paired_random
            assert pr is not None, 'coupled randomness off (config must set it)'
            kwargs = {}
            if getattr(model, '_samples_lanes', False):
                kwargs['lane_uniform'] = pr.uniform('lane', range(steps))
            with torch.no_grad(), scoring_rng(settings.seed, device):
                roll = model.rollout(
                    inp.initial, n_steps=steps, k0=0,
                    cav_teacher=None if inp.teacher is None else inp.teacher[:, :steps + 1],
                    present_seq=inp.present_seq[:, :steps + 1],
                    is_cav_seq=inp.is_cav[:, :steps + 1],
                    history=inp.hist, history_mask=inp.hist_mask,
                    controller_id=inp.controller_id, action_timeline=inp.action_timeline,
                    cav_controller=inp.cav_controller, entries=inp.entries,
                    driver_history=inp.observed_hist,
                    driver_history_mask=inp.observed_hist_mask, **kwargs)
            v0 = inp.initial['v_mps'].detach().to('cpu').numpy()
            m0 = inp.present[:, 0].detach().to('cpu').numpy().astype(bool)
            cav0 = inp.is_cav[:, 0].detach().to('cpu').numpy().astype(bool)
            speed = np.concatenate([v0[:, None],
                                    roll['speeds'].detach().to('cpu').numpy()], 1)
            mask = np.concatenate([m0[:, None],
                                   roll['masks'].detach().to('cpu').numpy().astype(bool)], 1)
            rec_speed = inp.states[:, :steps + 1, :, 4].detach().to('cpu').numpy()
            rec_mask = inp.present_seq[:, :steps + 1].detach().to('cpu').numpy().astype(bool)
            sel = m0 & ~cav0
            for key, base, arm_names in layout:
                dj = {str(int(h)): [] for h in horizons}
                for b in range(base, base + 1 + len(arm_names)):
                    for h in horizons:
                        dj[str(int(h))].append([
                            delay_proxy(speed[b], mask[b], sel[b], h),
                            delay_proxy(rec_speed[b], rec_mask[b], sel[b], h)])
                out[key] = {'arm_names': arm_names, 'dj': dj}
    finally:
        model.train(was_training)
    return out


def evaluate(model, settings, root, role, device, wm_sha=None, threshold=None,
              indexes=None, steps=STEPS, horizons=HORIZONS, samples=1):
    source = build_source(root, role)
    indexes = list(range(len(source.groups))) if indexes is None else list(indexes)
    horizons = tuple(h for h in horizons if h <= steps*DT_S + 1e-9)
    rolled = rollout_dj(model, settings, source, indexes, device,
                        steps=steps, horizons=horizons, samples=samples)
    rows = []
    for gi in indexes:
        key, _, arms = group_rows(source, gi)
        r = rolled[key]
        assert r['arm_names'] == [a for a, _ in arms]
        # gate rows: per-arm paired DIFFERENCES (arm - native reference), [wm, real]
        diff = {h: [[arm[0] - r['dj'][h][0][0], arm[1] - r['dj'][h][0][1]]
                    for arm in r['dj'][h][1:]]
                for h in r['dj']}
        rows.append({'pair_key': key, 'arms': len(arms), 'arm_names': r['arm_names'],
                     'dJ': diff})
    doc = _summarize(rows, role, root, wm_sha, threshold)
    doc['source_receipt'] = source.receipt
    doc['samples'] = samples
    doc['seed'] = settings.seed
    doc['entrant_control'] = settings.entrant_control
    if samples > 1:
        doc['sample_delays'] = {k: v['draw_dj'] for k, v in rolled.items()}
    return doc


def _sign(pairs):
    kept = [(w, s) for w, s in pairs if s != 0]
    return (float(np.mean([np.sign(w) == np.sign(s) for w, s in kept]))
            if kept else None)


def _slope(pairs):
    a = np.asarray(pairs, float)
    if len(a) < 3 or a[:, 1].std() == 0:
        return None
    return float(np.polyfit(a[:, 1], a[:, 0], 1)[0])


def _boot_groups(groups, stat, draws=2000, seed=0):
    """World-cluster bootstrap: resample whole groups (worlds), 2000 draws."""
    rng = np.random.default_rng(seed)
    point = stat([v for g in groups for v in g])
    out = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        value = stat([v for i in pick for v in groups[i]])
        if value is not None and np.isfinite(value):
            out.append(value)
    lo, hi = (np.quantile(out, [.025, .975]).tolist() if out else [None, None])
    return point, [lo, hi]


def _stable_seed(name, h):
    return int(hashlib.sha256(f'e3arm:{name}:{h}'.encode()).hexdigest()[:8], 16)


def _summarize(rows, role, root, wm_sha, threshold):
    horizons = {}
    for h in sorted({h for r in rows for h in r['dJ']}, key=int):
        per_group = [r['dJ'][h] for r in rows]
        sign, sign_ci = _boot_groups(per_group, _sign, seed=int(h))
        arr = np.asarray([v for g in per_group for v in g], float)
        e = arr[:, 0] - arr[:, 1]
        horizons[h] = {
            'sign_agreement': sign, 'sign_ci95': sign_ci,
            'mean_signed_error': float(e.mean()), 'error_sd': float(e.std()),
            'mae': float(abs(e).mean()),
            'n_pairs': len(arr), 'n_groups': len(per_group)}
    doc = {'schema': 'wmcombined_e3_gate/1', 'role': role, 'pairs_root': str(root),
           'wm_sha256': wm_sha, 'groups': len(rows),
           'rule': 'paired t0-HDV dJ (arm - native) of a 300 s WM rollout with coupled '
                   'randomness vs the recorded tapes; world-cluster bootstrap 2000 seed 0',
           'threshold': threshold,
           'horizons': horizons, 'rows': rows, 'arm_analysis': arm_analysis(rows)}
    return doc


def arm_analysis(rows):
    """Per-arm paired dJ statistics at 120/300 s (the METERVIS metering question)."""
    out = {}
    for name in sorted({n for r in rows for n in r['arm_names']}):
        arms = {}
        for h in ('120', '300'):
            if not any(h in r['dJ'] for r in rows):
                continue
            per_group = []
            for r in rows:
                if name in r['arm_names']:
                    # one (world, pair) group each: whole-world bootstrap resamples it
                    per_group.append([r['dJ'][h][r['arm_names'].index(name)]])
            flat = [g[0] for g in per_group]
            arr = np.asarray(flat, float)
            e = arr[:, 0] - arr[:, 1]
            slope, slope_ci = _boot_groups(per_group, _slope,
                                           seed=_stable_seed(name, h))
            arms[h] = {'mean_signed_error': float(e.mean()), 'error_sd': float(e.std()),
                       'mae': float(abs(e).mean()), 'n_pairs': len(e),
                       'sign_agreement': _sign(flat),
                       'pooled_slope': slope, 'pooled_slope_ci95': slope_ci}
        out[name] = arms
    return out


def load_for_scoring(config, ckpt):
    """accept.load_model with the training entry/coupling conventions restored."""
    from tools.r10_attribution import load_model
    cfg, settings, model, report = load_model(
        str(config), str(ckpt),
        ('data.entries_v2=true', 'loss.paired_diff.couple_randomness=true'))
    assert not report['unused'], f'{ckpt}: unused tensors {report["unused"][:5]}'
    return cfg, settings, model


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--role', choices=('calibration', 'validation'), required=True)
    ap.add_argument('--wm-ckpt', required=True)
    ap.add_argument('--wm-config', required=True)
    ap.add_argument('--pairs', default='/root/autodl-tmp/c4/e3data')
    ap.add_argument('--threshold', default='/root/autodl-tmp/c4/longloss20261027/'
                                           'runtime/configs/coopw_horizon_gate_13E_b2.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--seed', type=int, default=140014)
    ap.add_argument('--entrant-control', action='store_true', default=None)
    ap.add_argument('--entrant-controllers', help='JSON arm -> hash-bound feedback specification')
    ap.add_argument('--ramp-speed-adapt', choices=('on', 'off'))
    args = ap.parse_args(argv)
    import torch
    cfg, settings, model = load_for_scoring(args.wm_config, args.wm_ckpt)
    from dataclasses import replace
    updates = {'seed': args.seed}
    if args.entrant_control is not None:
        updates['entrant_control'] = args.entrant_control
    if args.entrant_controllers:
        updates['entrant_controllers'] = json.loads(Path(args.entrant_controllers).read_text())
    settings = replace(settings, **updates)
    if args.ramp_speed_adapt is not None:
        model.ramp_speed_adapt_learnable = args.ramp_speed_adapt == 'on'
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.85)
        model = model.to(device)
    else:
        torch.set_num_threads(4)
    threshold = None
    if args.threshold:
        threshold = json.loads(Path(args.threshold).read_text())['threshold']
    doc = evaluate(model, settings, args.pairs, args.role, device,
                   wm_sha=sha256(args.wm_ckpt), threshold=threshold, samples=args.samples)
    Path(args.out).write_text(json.dumps(doc, indent=1))
    slim = {k: v for k, v in doc.items() if k != 'rows'}
    print(json.dumps(slim, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
