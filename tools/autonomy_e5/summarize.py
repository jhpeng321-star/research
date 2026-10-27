"""Fixed S-E5DONE paired, cell-stratified world bootstrap and paper artifacts."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'tools/autonomy_e5/receipts_done'
PAPER = Path('E:/research-dense/docs/paper_results')
ARMS = ('coopON', 'coopOFF', 'native', 'mappo')
PAIRS = (('coopON', 'coopOFF'), ('coopON', 'native'), ('coopOFF', 'native'),
         ('coopON', 'mappo'), ('coopOFF', 'mappo'), ('mappo', 'native'))
METRICS = {
    'team_delay_vs': 'veh-s', 'road_delay_vs': 'veh-s', 'source_wait_vs': 'veh-s',
    'mainline_delay_vs': 'veh-s', 'ramp_delay_vs': 'veh-s', 'throughput_vph': 'veh/h',
    'arrivals': 'vehicles', 'merge_crossings': 'vehicles', 'requested': 'vehicles',
    'inserted': 'vehicles', 'inserted_total': 'vehicles', 'worst_group_delay_s': 's/request',
    'hard_braking_vs': 'veh-s', 'ttc_below_3_vs': 'veh-s', 'collisions': 'count',
    'teleports': 'count', 'dropped_vehicle_steps': 'count', 'acceleration_energy': '(m/s2)^2 veh-s',
    'cost_ttc': 'integrated TTC cost', 'smooth': 'integrated effort',
    'fairness_cvar90_s': 's/vehicle',
}
BEHAVIOURS = ('multi_slow', 'side_by_side', 'metering', 'vacate', 'vacate_ramp_present')


def draw_plan():
    """One shared plan across arms, clocks, metrics and scopes: 19 worlds/cell."""
    seeds = np.arange(973000, 973228)
    groups = [np.flatnonzero((seeds-973000) % 12 == c) for c in range(12)]
    rng = np.random.default_rng(20261027)
    draws = [g[rng.integers(0, 19, (5000, 19))] for g in groups]
    assert all(len(g) == 19 for g in groups)
    return seeds, groups, draws


def physical(values, indices, draws):
    selected = np.concatenate(indices)
    boot = np.mean([values[d].mean(1) for d in draws], axis=0)
    return float(values[selected].mean()), boot


def pooled_rate(counts, exposures, indices, draws):
    selected = np.concatenate(indices)
    boot_counts = sum(counts[d].sum(1) for d in draws)
    boot_exposure = sum(exposures[d].sum(1) for d in draws)
    assert np.all(boot_exposure > 0) and exposures[selected].sum() > 0
    return float(counts[selected].sum()/exposures[selected].sum()), boot_counts/boot_exposure


def result(kind, metric, arm, reference, horizon, share, demand, n, value, boot, unit, **extra):
    lo, hi = map(float, np.quantile(boot, [.025, .975]))
    assert np.isfinite([value, lo, hi]).all()
    return dict(kind=kind, metric=metric, treatment=arm, reference=reference, horizon_s=horizon,
                share=share, demand=demand, n_worlds=n, value=value, ci_low=lo, ci_high=hi,
                unit=unit, convention='treatment-minus-reference' if reference else 'absolute', **extra)


def summarize(data):
    seeds, groups, samples = draw_plan()
    records, behaviour = {}, {}
    for d in data:
        assert d['proof']['status'] == 'PASS' and d['proof']['worlds'] == 114
        for r in d['rows']:
            key = r['seed'], r['arm']
            assert key not in records
            records[key] = r
        for r in d['behaviour']:
            key = r['seed'], r['arm']
            assert key not in behaviour
            behaviour[key] = r
    expected = {(int(s), a) for s in seeds for a in ARMS}
    assert set(records) == set(behaviour) == expected
    assert all(len({records[s, a]['initial_state_sha256'] for a in ARMS}) == 1 for s in seeds)
    stats, worlds = [], []
    for h in (120, 300, 600):
        m = {a: {k: np.array([records[s, a]['windows'][str(h)][k] for s in seeds]) for k in METRICS} for a in ARMS}
        exp = {a: np.array([behaviour[s, a]['windows'][str(h)]['exposure_cav_h'] for s in seeds]) for a in ARMS}
        counts = {a: {b: np.array([behaviour[s, a]['windows'][str(h)]['counts'][b] for s in seeds]) for b in BEHAVIOURS} for a in ARMS}
        for s in seeds:
            for a in ARMS:
                w = dict(seed=int(s), arm=a, horizon_s=h, share=records[s, a]['share'], demand=records[s, a]['demand'],
                         cell=records[s, a]['cell'], initial_state_sha256=records[s, a]['initial_state_sha256'],
                         trajectory_sha256=behaviour[s, a]['trajectory_sha256'],
                         **{k: records[s, a]['windows'][str(h)][k] for k in METRICS})
                w['exposure_cav_h'] = behaviour[s, a]['windows'][str(h)]['exposure_cav_h']
                w.update({b+'_count': behaviour[s, a]['windows'][str(h)]['counts'][b] for b in BEHAVIOURS})
                worlds.append(w)
        for share in ('All', .05, .1, .2, .4):
            for demand in ('All', 1900, 2400, 2700):
                cells = [c for c in range(12) if (share == 'All' or (.05, .1, .2, .4)[c//3] == share)
                         and (demand == 'All' or (1900, 2400, 2700)[c % 3] == demand)]
                ix, draw = [groups[c] for c in cells], [samples[c] for c in cells]
                selected = np.concatenate(ix)
                n = len(selected)
                for metric, unit in METRICS.items():
                    estimates = {a: physical(m[a][metric], ix, draw) for a in ARMS}
                    for a, (v, boot) in estimates.items():
                        stats.append(result('physical_absolute', metric, a, '', h, share, demand, n, v, boot, unit))
                    for a, b in PAIRS:
                        # Paired differences, sharing the identical world draw plan.
                        v, boot = physical(m[a][metric]-m[b][metric], ix, draw)
                        stats.append(result('physical_contrast', metric, a, b, h, share, demand, n, v, boot, unit))
                        if metric == 'team_delay_vs':
                            reverse = result('saved_delay', metric, a, b, h, share, demand, n, -v, -boot, unit)
                            reverse['convention'] = 'reference-minus-treatment; positive better for treatment'
                            stats.append(reverse)
                for b in BEHAVIOURS:
                    rates = {a: pooled_rate(counts[a][b], exp[a], ix, draw) for a in ARMS}
                    for a, (v, boot) in rates.items():
                        stats.append(result('behaviour_rate', b, a, '', h, share, demand, n, v, boot,
                                            'events/CAV-hour', count=int(counts[a][b][selected].sum()),
                                            exposure_cav_h=float(exp[a][selected].sum())))
                    for a, ref in PAIRS:
                        av, ab = rates[a]
                        bv, bb = rates[ref]
                        stats.append(result('behaviour_contrast', b, a, ref, h, share, demand, n,
                                            av-bv, ab-bb, 'events/CAV-hour'))
    return stats, worlds


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def report(stats, data):
    def get(metric, a='coopON', b='coopOFF', h=600, share='All', demand='All', kind='physical_contrast'):
        return next(r for r in stats if (r['metric'], r['treatment'], r['reference'], r['horizon_s'],
                    r['share'], r['demand'], r['kind']) == (metric, a, b, h, share, demand, kind))

    def fmt(r, decimals=1):
        return f"{r['value']:+,.{decimals}f} [{r['ci_low']:+,.{decimals}f}, {r['ci_high']:+,.{decimals}f}]"

    primary = get('team_delay_vs')
    identical = all(r['value'] == r['ci_low'] == r['ci_high'] == 0 for r in stats
                    if r['kind'] == 'physical_contrast' and r['treatment'] == 'coopON' and r['reference'] == 'coopOFF')
    headline = ('the two policies have **exactly equal recorded outcomes in every world**' if identical else
                f"ON − OFF **{fmt(primary)} veh-s/world** (pointwise 95% CI)")
    lines = ['# E5 — matched cycle-1 cooperation-loss ablation', '',
        f"**Complete: 228/228 paired TRAIN/dev worlds;** {headline}. "
        f"ON − OFF total delay at 120/300/600 s: **0 [0, 0] veh-s/world at each clock**." if identical else
        f"**Complete: 228/228 paired TRAIN/dev worlds;** {headline}.", '',
        '## Panel, provenance and accounting', '',
        '- Exact worlds 973000–973227; 19 in each of 12 CAV-share×demand cells: shares 5/10/20/40%; demands 1900/2400/2700 veh/h. '
        'All four arms share each world’s observable native initial-state SHA256. Native warmup 300 s; intervention clocks 120/300/600 s.',
        '- coopON cycle-1 checkpoint `64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c`; '
        'coopOFF cycle-1 `e1fa4881bbc5c89d05298e979d98b23284308331ad1bbb12f9608fa67e105675`; '
        'MAPPO `aecd7d70`; both original native-relative gates pass. Frozen scientific source manifest `b3c031ad` (543 files) verified on both hosts.',
        '- MAIN completion amendment was already registered; exactly six worlds 973217/219/221/223/225/227 completed once on cpuQ. '
        'All receipt-bound simulation predecessors exited. This resumed analysis dispatched zero simulations.',
        '- E5 coopOFF: **210,600 charged / 205,200 actual sim-s** (cpuP 102,600/102,600; cpuQ 108,000/102,600). '
        'Six historical pre-SUMO failures and their 5,400 charges remain; each has only its previously authorized repair. '
        'Completion consumed exactly +5,400 charged/actual. Policy phase 415,800 charged / 410,400 actual of 615,600; 199,800 unused. '
        'Baseline+policy production 826,200 charged / 820,800 actual of 1,026,000.',
        f"- Complete result files, unique attempt charges, immutable input/source hashes, all standalone cumulative clocks, full-population/source/group conservation verified; "
        f"maximum conservation residual {max(d['proof']['max_conservation_residual_vs'] for d in data):.9g} veh-s. No complete-case selection.", '',
        '## Estimands and uncertainty', '',
        '**Cost convention:** treatment − reference, negative = less delay. Total delay = on-road speed-deficit vehicle-seconds + entrance/source waiting vehicle-seconds. '
        '**Saved-delay convention:** reference − treatment, positive = better for treatment. These are exact sign reversals, with CI endpoints reversed. '
        'Means weight the registered worlds equally. All intervals use the same fixed 5,000 whole-world bootstrap draws, resampling 19 worlds within each original cell, '
        'seed 20261027; arms, clocks and metrics remain paired. Cell/secondary intervals are pointwise, without multiplicity adjustment. '
        'The matched world panel is not replicated policy training: it compares these two cycle-1 checkpoints, rather than estimating a general training-seed ablation effect.', '',
        '## Exact-tie provenance check', '',
        'All **228 ON/OFF observable trajectory NPZ files are byte-identical**, and their complete measurement dictionaries are exactly equal. '
        'This is an observed deterministic tie, not merely a nonsignificant mean or an equivalence test. '
        'A zero-SUMO audit checked every deployment summary: the correct, distinct checkpoint paths were used, each for 600 policy ticks. '
        'Both stored checkpoints are update180 native-residual policies; their residual-head weights/bias differ (maximum tensor difference 0.0017030201), '
        'while all 26 group-encoder tensors match. Independent CPU loading reproduced each checkpoint’s stored policy tensors exactly, with deterministic execution, no hold and no added exploration noise. '
        'Observable requested and issued acceleration ledgers differ in **all 228 worlds**, with maximum ON/OFF difference **0.004012585 m/s²**; lane-command ledgers match. '
        'Thus the stored commands differ but recorded realised trajectories do not. The precise executor/plant mechanism producing the tie has not been isolated by this analysis. '
        'Do not infer that the cooperation objective is generally ineffective, or attribute either policy’s native-relative gain to cooperation. '
        'Per-world evidence: `IDENTITY_AUDIT.json` and `TELEMETRY_AUDIT.json` on both hosts.', '',
        '## Total delay, all worlds', '']
    for h in (120, 300, 600):
        cost, saved = get('team_delay_vs', h=h), get('team_delay_vs', h=h, kind='saved_delay')
        lines += [f"- **{h} s:** ON − OFF {fmt(cost)} veh-s; OFF − ON saved {fmt(saved)} veh-s. "
                  f"Road contribution {fmt(get('road_delay_vs', h=h))}; entrance contribution {fmt(get('source_wait_vs', h=h))} veh-s."]
    lines += ['', 'Comparisons against native and MAPPO (cost convention; veh-s/world):', '']
    for h in (120, 300, 600):
        lines.append(f"- **{h} s:** ON − native {fmt(get('team_delay_vs', b='native', h=h))}; "
                     f"OFF − native {fmt(get('team_delay_vs', a='coopOFF', b='native', h=h))}; "
                     f"ON − MAPPO {fmt(get('team_delay_vs', b='mappo', h=h))}; "
                     f"OFF − MAPPO {fmt(get('team_delay_vs', a='coopOFF', b='mappo', h=h))}.")
    lines += ['', '## CAV share × demand at 600 s', '',
              'ON − OFF total-delay cost, veh-s/world (n=19 each); all clocks, all comparator contrasts, marginal strata and reverse saved-delay signs are in the CSV.', '']
    for share in (.05, .1, .2, .4):
        lines.append(f"- **{share:.0%} CAV:** " + '; '.join(f"{d} veh/h: {fmt(get('team_delay_vs', share=share, demand=d))}" for d in (1900, 2400, 2700)) + '.')
    lines += ['', '## Throughput and safety', '',
              'Absolute all-world means [95% CI] at 600 s; vph is arrivals × 3600 / intervention duration. Hard braking and TTC<3 use vehicle-seconds, not event counts.', '']
    for a in ARMS:
        def absolute(metric):
            return fmt(get(metric, a=a, b='', kind='physical_absolute'))
        lines.append(f"- **{a}:** throughput {absolute('throughput_vph')} veh/h; hard braking {absolute('hard_braking_vs')} veh-s; "
                     f"TTC<3 {absolute('ttc_below_3_vs')} veh-s.")
    for h in (120, 300, 600):
        lines.append(f"- **ON − OFF, {h} s:** throughput {fmt(get('throughput_vph', h=h))} veh/h (positive = more discharge); "
                     f"hard braking {fmt(get('hard_braking_vs', h=h))} veh-s; TTC<3 {fmt(get('ttc_below_3_vs', h=h))} veh-s (negative = less exposure).")
    incident_totals = {k: {a: sum(r['windows']['600'][k] for d in data for r in d['rows'] if r['arm'] == a) for a in ARMS}
                       for k in ('collisions', 'teleports', 'dropped_vehicle_steps')}
    lines += [f"- Across 912 arm-worlds at the full clock: collisions {incident_totals['collisions']}; teleports {incident_totals['teleports']}; "
              f"dropped vehicle steps {incident_totals['dropped_vehicle_steps']}.", '', '## Frozen COOPMEAS observable behaviours', '',
        'Detector SHA256 `452d44a78a1c1280aa0fd6f3d9f58a7498d35432ac4158706ba40a922a04c9f4`; thresholds '
        '`9119cd77d27889c9546cfaacd91af5a3f7f89beda3a5b6616e988aa67860701f`, unchanged. '
        'Rates pool event counts / observed modelled-lane CAV-hours, rather than averaging per-world rates. '
        'Paired bootstrap resamples each world’s counts and arm-specific exposure together. Side-by-side is a subset of multi-CAV slowing; '
        'lane-vacate counts all registered 1→0 moves at −400..0 m, with the ramp-present courtesy subset additionally reported. '
        'Event starts and exposure are restricted to [300,300+h); each detector receives only that prefix. '
        'Saved tapes begin at 300.5 s (first 1 Hz frame 301 s), so detector rolling features warm up inside the intervention: '
        'pre-intervention behaviour history and events already underway at the boundary are unobserved. This common left-censoring is retained, not filled in.', '',
        'Full 600 s pooled count / CAV-hour and rate [95% CI]:', '']
    for b, label in (('multi_slow', 'Multi-CAV slowing'), ('side_by_side', 'Side-by-side'), ('metering', 'Upstream metering'),
                     ('vacate', 'Lane vacate (all)'), ('vacate_ramp_present', 'Lane vacate (ramp-present courtesy subset)')):
        parts = []
        for a in ARMS:
            r = get(b, a=a, b='', kind='behaviour_rate')
            parts.append(f"{a} {r['count']}/{r['exposure_cav_h']:.3f} h = {fmt(r, 2)}")
        contrast = get(b, kind='behaviour_contrast')
        lines.append(f"- **{label}:** " + '; '.join(parts) + f" events/CAV-hour. ON − OFF {fmt(contrast, 2)} events/CAV-hour.")
    lines += ['', 'These are whole-policy trajectory associations. They do not isolate causal cooperation, nor establish that any detected behaviour caused the delay contrast.', '',
        '## Reproduction and evidence', '',
        '- `tools/autonomy_e5/analyze_shard.py`: receipt-bound zero-SUMO validation and frozen detector, one nice19/single-thread detached process per CPU host; '
        '`analysis_ops.py` stages/polls/fetches without dispatching simulations.',
        '- `python -B -m tools.autonomy_e5.summarize` regenerates estimates from `tools/autonomy_e5/receipts_done/{cpuP,cpuQ}/ANALYSIS_DATA.json`. '
        'Source trajectories remain at their recorded CPU-host paths; every accessed input is hashed in these receipts.',
        '- `E5_COOPOFF_C1.csv`: full absolute physical/rate estimates, paired contrasts and saved-delay signs at all clocks and strata. '
        '`E5_COOPOFF_C1_worlds.csv`: all 2,736 arm×world×clock rows, physical metrics, observable detector counts/exposures and pairing hashes. '
        '`RESULTS.json` and `MANIFEST.json` preserve exact estimates and artifact hashes. Report/CSV copies are committed under `tools/autonomy_e5/receipts_done/`.']
    return '\n'.join(lines)+'\n'


def main():
    data = [json.loads((EVIDENCE/box/'ANALYSIS_DATA.json').read_text()) for box in ('cpuP', 'cpuQ')]
    stats, worlds = summarize(data)
    assert PAPER.is_dir()
    csv_write(EVIDENCE/'E5_COOPOFF_C1.csv', stats)
    csv_write(EVIDENCE/'E5_COOPOFF_C1_worlds.csv', worlds)
    (EVIDENCE/'E5_COOPOFF_C1.md').write_text(report(stats, data), encoding='utf-8')
    (EVIDENCE/'RESULTS.json').write_text(json.dumps(dict(bootstrap_draws=5000, seed=20261027,
        resampling='whole worlds within original 12 cells; shared paired plan', estimates=stats), indent=2), encoding='utf-8')
    for name in ('E5_COOPOFF_C1.csv', 'E5_COOPOFF_C1_worlds.csv', 'E5_COOPOFF_C1.md'):
        (PAPER/name).write_bytes((EVIDENCE/name).read_bytes())
    names = ('E5_COOPOFF_C1.csv', 'E5_COOPOFF_C1_worlds.csv', 'E5_COOPOFF_C1.md', 'RESULTS.json')
    audit_names = [(Path(box)/n).as_posix() for box in ('cpuP', 'cpuQ')
                   for n in ('ANALYSIS_DATA.json', 'ANALYSIS_STATUS.json', 'ANALYSIS_VALIDATION.json',
                             'IDENTITY_AUDIT.json', 'TELEMETRY_AUDIT.json')]
    manifest = dict(artifacts={n: hashlib.sha256((EVIDENCE/n).read_bytes()).hexdigest() for n in names},
                    evidence={n: hashlib.sha256((EVIDENCE/n).read_bytes()).hexdigest() for n in audit_names},
                    n_worlds=228, n_world_rows=len(worlds), n_estimates=len(stats),
                    scripts={n: hashlib.sha256((ROOT/'tools/autonomy_e5'/n).read_bytes()).hexdigest()
                             for n in ('analyze_shard.py', 'analysis_ops.py', 'summarize.py',
                                       'identity_audit.py', 'telemetry_audit.py', 'verify_report.py')})
    (EVIDENCE/'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    for r in stats:
        if r['kind'] == 'physical_contrast' and r['metric'] == 'team_delay_vs' and r['treatment'] == 'coopON' and r['reference'] == 'coopOFF' and r['share'] == r['demand'] == 'All':
            print(json.dumps(r))


if __name__ == '__main__':
    main()
