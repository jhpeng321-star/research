"""clean5b-F fairness (owner REGISTRATION 2026-11-07 04:0x; DEFAULT OFF, byte-identical when off).

Owner rule (registered): the firm 15 m/s solo2 hold (wall_h_solo2) is banned for fairness; a strategy is an
EV-BC target in a cell only if EVERY group - origin SUMO lane 0/1/2 and HDV vs CAV - has a mean per-vehicle
delay increase <= THRESHOLD_S (10 s) vs MAPPO u0, AND lb95 > 0 on the total gain. The training cost adds,
per update and world, paired with the frozen A arm on the SAME world, the excess harm
    sum_g n_g * max(0, mean_delay_g(B) - mean_delay_g(A) - THRESHOLD_S)      [veh-s]
one-for-one to the episode cost, for B and R identically. Per-vehicle group delays come from an EVAL-ONLY
logger (N6 ops/nm_finaleval.vehicle_log_report format) and are never a policy input.

This module: (1) MULTICRIT -> fairness-flagged guidance rows (gain sign, registered J CI), (2) group summaries
from the N6 vehicle-log report, (3) the excess-harm penalty and its per-row distribution.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

THRESHOLD_S = 10.0
RULE = 'fullpol-clean5b-F-fairness-v1'
PROVENANCE = ('owner REGISTRATION 2026-11-07 04:0x (MAIN): wall_h_solo2 banned; EV-BC eligibility = every group '
              '(origin SUMO lane 0/1/2, HDV, CAV) mean per-vehicle delay increase <= 10 s vs u0 AND lb95 > 0 on '
              'total gain; reward term = sum_g n_g max(0, dB_g - dA_g - 10 s) veh-s, paired with frozen A on the '
              'same world, B = R')
GROUP_KEYS = ('L0_mean', 'L1_mean', 'L2_mean', 'hdv_mean', 'cav_mean')   # MULTICRIT metrics (SUMO-lane origin)
BANNED = ('wall_h_solo2',)


# ---------------------------------------------------------------- (1) fairness-flagged guidance rows

def multicrit_rows(analysis, *, arm='B', baseline='mappo', threshold_s=THRESHOLD_S, source=None):
    """MULTICRIT analysis JSON (cells[cell].arms[name].metrics, cells[cell].u0) -> guidance rows.

    gain = -(J_arm - J_u0) (registered paired bootstrap of the MULTICRIT analysis; lb95 = -J.hi);
    group_increase_s = metrics[g] - u0[g] per group (seconds per vehicle); fair = all <= threshold."""
    out = []
    for cell, c in sorted(analysis['cells'].items()):
        u0 = c['u0']
        for name, a in sorted(c['arms'].items()):
            m = a['metrics']
            j = m['J']
            # paired per-seed mean difference (seconds per vehicle) of each group's mean delay vs u0
            inc = {g: float(m[g]['diff']) if isinstance(m[g], dict) else float(m[g])-float(u0[g])
                   for g in GROUP_KEYS if g in m and (isinstance(m[g], dict) or g in u0)}
            if len(inc) != len(GROUP_KEYS):
                raise ValueError(f'MULTICRIT {cell}/{name}: group means {GROUP_KEYS} required')
            out.append(dict(arm=arm, regime=cell, family=name, baseline=baseline, n=int(a['n']),
                            mean_veh_s=-float(j['diff']), lb95_veh_s=-float(j['hi']), ub95_veh_s=-float(j['lo']),
                            group_increase_s=inc, worst_group_increase_s=max(inc.values()),
                            fair=bool(max(inc.values()) <= threshold_s) and name not in BANNED,
                            banned=name in BANNED, source=source))
    return out


def combine(rows):
    """One row per (arm, regime, family) across sources: eligible only if fair in EVERY source; the most
    conservative estimate (smallest lb95) is kept; the worst group increase is the max over sources."""
    merged = {}
    for r in rows:
        key = (r['arm'], r['regime'], r['family'])
        if key not in merged:
            merged[key] = dict(r, sources=[r.get('source')])
            continue
        m = merged[key]
        keep = r if r['lb95_veh_s'] < m['lb95_veh_s'] else m
        merged[key] = dict(keep, fair=m['fair'] and r['fair'], sources=m['sources']+[r.get('source')],
                           worst_group_increase_s=max(m['worst_group_increase_s'], r['worst_group_increase_s']))
    return [merged[k] for k in sorted(merged)]


def build_guidance(paths, *, provenance, arm='B', threshold_s=THRESHOLD_S):
    """MULTICRIT JSON files -> ``regime_guidance`` (schema fullpol-regime-guidance-v1) with fairness flags."""
    import hashlib
    from .fullpol_learn import guidance_doc
    rows, sources = [], []
    for path in paths:
        raw = Path(path).read_bytes()
        rows += multicrit_rows(json.loads(raw.decode('utf-8')), arm=arm, threshold_s=threshold_s,
                               source=Path(path).name)
        sources.append(dict(name=Path(path).name, sha256=hashlib.sha256(raw).hexdigest(), baseline='mappo'))
    doc = guidance_doc(combine(rows), sources=sources, provenance=provenance)
    doc['fairness'] = dict(rule=RULE, threshold_s=threshold_s, banned=list(BANNED), provenance=PROVENANCE)
    return doc


# ---------------------------------------------------------------- (2) group summaries (eval-only logger)

def summary_from_report(report):
    """N6 ``vehicle_log_report`` dict -> {group: {n, mean_s}} (class:cav / class:hdv / origin:<o>)."""
    out = {}
    cls = report.get('delay_by_class') or {}
    for name, n_key, m_key in (('class:cav', 'cav', 'cav_mean_s'), ('class:hdv', 'hdv', 'hdv_mean_s')):
        if cls.get(m_key) is not None and report.get(n_key):
            out[name] = dict(n=int(report[n_key]), mean_s=float(cls[m_key]))
    counts = report.get('origins') or {}
    for o, mean in (report.get('delay_by_origin_s') or {}).items():
        if mean is not None and counts.get(o):
            out[f'origin:{o}'] = dict(n=int(counts[o]), mean_s=float(mean))
    return out


# ---------------------------------------------------------------- (3) excess harm and its distribution

def excess_harm(groups_b, groups_a, threshold_s=THRESHOLD_S):
    """sum_g n_g(B) * max(0, mean_B - mean_A - threshold) over groups present in both (veh-s)."""
    total, parts = 0., {}
    for g in sorted(set(groups_b) & set(groups_a)):
        b, a = groups_b[g], groups_a[g]
        part = float(b['n'])*max(0., float(b['mean_s'])-float(a['mean_s'])-threshold_s)
        parts[g] = part
        total += part
    if not math.isfinite(total):
        raise ValueError('nonfinite fairness excess harm')
    return total, parts


def settings(cfg):
    """None when OFF; else the validated fair slot of clean5_ev (registered threshold only)."""
    from .fullpol_ev import settings as ev_settings
    ev = ev_settings(cfg)
    fair = None if ev is None else ev.get('fair')
    if not fair:
        return None
    if (not isinstance(fair, dict) or fair.get('rule') != RULE or float(fair.get('threshold_s', -1)) != THRESHOLD_S
            or set(fair)-{'rule', 'threshold_s', 'solo_variants', 'duo12', 'gate0', 'ev_filter', 'reward'}):
        raise ValueError('clean5_ev.fair needs the registered rule and threshold_s = 10 s')
    return deepcopy(fair)


def reference_path(cfg, update):
    return Path(cfg['run_root'])/'fair'/f'A_GROUPS_u{int(update):03}.json'


def reference(cfg, update):
    """Frozen A arm's per-world group summaries of this update ({str(world seed): {group: {n, mean_s}}})."""
    path = reference_path(cfg, update)
    if not path.exists():
        raise ValueError(f'fairness reward needs the paired A group summaries {path} (fail closed)')
    return json.loads(path.read_text(encoding='utf-8'))


def penalties(cfg, arm, episodes, ref, threshold_s=THRESHOLD_S):
    """({episode_id: harm veh-s}, receipt) for ordinary episodes paired with A on the same world."""
    out, rows, skipped = {}, [], []
    for e in episodes:
        if e.get('practice'):
            skipped.append(dict(episode_id=e['id'], reason='practice start (no same-world A episode)'))
            continue
        seed = str(e['world']['seed'])
        if seed not in ref or 'vehicle_groups' not in e:
            raise ValueError(f'fairness reward: missing paired A / own group summary for {e["id"]} (fail closed)')
        harm, parts = excess_harm(e['vehicle_groups'], ref[seed], threshold_s)
        out[e['id']] = harm
        rows.append(dict(episode_id=e['id'], world_seed=int(seed), harm_veh_s=harm, groups=parts))
    return out, dict(rule=RULE, threshold_s=threshold_s, arm=arm, episodes=rows, skipped=skipped,
                     total_harm_veh_s=sum(out.values()))


def apply(targets, harm_by_episode):
    """Distribute each episode's harm uniformly over its (tick, CAV) reward rows (sum = harm, veh-s),
    i.e. the episode cost rises one-for-one; rewards are -cost (veh-s). Returns a new targets dict."""
    out = {k: list(v) for k, v in targets.items()}
    for eid, harm in harm_by_episode.items():
        if harm <= 0:
            continue
        keys = [k for k in out if k[0] == eid]
        n = sum(len(out[k]) for k in keys)
        if not n:
            continue
        for k in keys:
            out[k] = [x-harm/n for x in out[k]]
    return out


def main(argv=None):
    """python -m tools.coopmwm.fullpol_fair guidance --multicrit A.json [B.json ...] --provenance ... --out G.json"""
    import argparse
    ap = argparse.ArgumentParser(description='clean5b-F fairness-flagged EV guidance (CPU, no physics)')
    ap.add_argument('command', choices=['guidance'])
    ap.add_argument('--multicrit', nargs='+', required=True)
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit('refusing to overwrite '+str(out))
    doc = build_guidance(a.multicrit, provenance=a.provenance)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding='utf-8')
    print(json.dumps(dict(out=str(out), rows=len(doc['rows']), sha256=doc['source']['sha256'],
                          fair=sorted({r['family'] for r in doc['rows'] if r['fair']}))))


if __name__ == '__main__':
    main()
