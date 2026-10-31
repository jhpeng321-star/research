"""Complete-panel descriptive distributions; frozen detector imported as one AST node."""
import ast
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper_results/ensprobe'
EVIDENCE=ROOT/'tools/ensprobe/evidence'
DETECTOR=Path('E:/agent-work/c4-entrant/tools/lrdiag/replay.py')
CLOCKS=(120,300,600)
ARMS=('native','mappo')
METRICS=('road','source','J')


def load_detector():
    text=DETECTOR.read_text(encoding='utf-8')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='breakdown')
    body=ast.get_source_segment(text,node)
    namespace={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(DETECTOR),'exec'),namespace)
    return namespace['breakdown'],dict(path=str(DETECTOR),file_sha256=hashlib.sha256(DETECTOR.read_bytes()).hexdigest(),
                                      function_sha256=hashlib.sha256(body.encode()).hexdigest(),source=body)


def spread(x):
    a=np.asarray(x,dtype=float)
    assert a.size and np.isfinite(a).all()
    return dict(n=int(a.size),mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else None,
                minimum=float(a.min()),maximum=float(a.max()))


def variance_parts(matrix):
    a=np.asarray(matrix,dtype=float)
    assert a.ndim==2 and min(a.shape)>1 and np.isfinite(a).all()
    within=float(a.var(axis=1,ddof=1).mean())
    means_var=float(a.mean(axis=1).var(ddof=1))
    between=means_var-within/a.shape[1]
    return dict(states=int(a.shape[0]),seeds=int(a.shape[1]),within_seed_variance=within,
                state_mean_variance=means_var,between_state_variance_untruncated=between,
                between_state_variance=max(0.,between),within_seed_sd=within**.5,
                between_state_sd=max(0.,between)**.5)


def csv_write(name,rows):
    assert rows
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def verify(data):
    results=data['results'];keys={(r['world'],r['arm'],r['rng']) for r in results}
    expected={(w,a,k) for w in range(969040,969045) for a in ARMS for k in range(969086,969094)}
    assert keys==expected and len(results)==80
    assert len(data['trunks'])==5
    for r in results:
        assert r['gate']['pass'] and r['gate']['rng_saved']=='false' and int(r['gate']['sumoseed'])==r['rng']
        assert r['actual_sim_s']==r['charged_sim_s']==600
        assert set(r['clocks'])=={'120','300','600'} and not any(r['safety'].values())
        for c in CLOCKS:
            row=r['clocks'][str(c)]
            assert abs(row['J']-row['road']-row['source'])<1e-7
        assert all(len(x)==66 for x in r['speed_bins'].values())
        assert all(len(x)==66 for x in r['bin_counts'].values())
    charges=data['accounting']
    assert len({r['id'] for r in charges})==len(charges)==87
    assert sum(r['sim_s'] for r in charges)==49869
    # Original300 prefix + repeated300 prefix + other4 prefixes +80 branches.
    assert sum(t['actual_sim_s'] for t in data['trunks'])+sum(r['actual_sim_s'] for r in results)+300+69==49869
    assert data['receipts']['COMPLETE.json']['actual_sim_s']==49869
    assert data['hashes']['w969040/w969040_native_r969086_RESULT.json']== \
        data['receipts']['BINDING.json']['recovery']['retained_result_sha256']
    return {k:r for k,r in zip([(r['world'],r['arm'],r['rng']) for r in results],results)}


def main():
    data=json.loads((EVIDENCE/'collection.json').read_text())
    lookup=verify(data)
    detector,binding=load_detector()
    OUT.mkdir(parents=True,exist_ok=True)
    seed_rows=[];events=[];summaries=[];pairs=[];variances=[];breakdowns=[];restore=[]
    for w in range(969040,969045):
        for arm in ARMS:
            panel=[lookup[w,arm,k] for k in range(969086,969094)]
            for r in panel:
                restore.append(dict(world=w,arm=arm,rng=r['rng'],**r['gate']))
                for clock in CLOCKS:
                    values=r['clocks'][str(clock)]
                    seed_rows.append(dict(world=w,penetration=r['penetration'],demand_vph=r['demand'],arm=arm,
                                          rng=r['rng'],clock_s=clock,road_delay_veh_s=values['road'],
                                          source_delay_veh_s=values['source'],team_delay_veh_s=values['J'],
                                          physical_tape_sha256=r['tape_sha256']))
                    for region,bins in r['speed_bins'].items():
                        events.append(dict(world=w,arm=arm,rng=r['rng'],clock_s=clock,region=region,
                                           **detector(bins[:6+clock//10])))
            for clock in CLOCKS:
                for metric in METRICS:
                    summaries.append(dict(world=w,penetration=panel[0]['penetration'],demand_vph=panel[0]['demand'],
                                          arm=arm,clock_s=clock,metric=metric,units='veh-s',
                                          **spread([r['clocks'][str(clock)][metric] for r in panel])))
                for region in panel[0]['speed_bins']:
                    er=[r for r in events if r['world']==w and r['arm']==arm and r['clock_s']==clock and r['region']==region]
                    valid=[r for r in er if r['event'] is not None]
                    onsets=[r['onset_s'] for r in valid if r['event']]
                    breakdowns.append(dict(world=w,arm=arm,clock_s=clock,region=region,n_seeds=len(er),n_valid=len(valid),
                                           n_breakdown=sum(bool(r['event']) for r in valid),
                                           fraction_breakdown=sum(bool(r['event']) for r in valid)/len(valid) if valid else None,
                                           n_preexisting=sum(bool(r.get('preexisting')) for r in valid),
                                           onset_min_s=min(onsets) if onsets else None,onset_max_s=max(onsets) if onsets else None,
                                           onset_mean_s=float(np.mean(onsets)) if onsets else None))
        for clock in CLOCKS:
            for metric in METRICS:
                delta=[lookup[w,'mappo',k]['clocks'][str(clock)][metric]-lookup[w,'native',k]['clocks'][str(clock)][metric]
                       for k in range(969086,969094)]
                pairs.append(dict(world=w,clock_s=clock,metric=metric,units='veh-s',contrast='MAPPO-minus-native',**spread(delta)))
    for arm in ARMS:
        for clock in CLOCKS:
            for metric in METRICS:
                matrix=[[lookup[w,arm,k]['clocks'][str(clock)][metric] for k in range(969086,969094)] for w in range(969040,969045)]
                variances.append(dict(arm=arm,clock_s=clock,metric=metric,variance_units='(veh-s)^2',
                                      sd_units='veh-s',**variance_parts(matrix)))
    csv_write('per_seed_delay.csv',seed_rows);csv_write('state_spread.csv',summaries)
    csv_write('breakdown_per_seed.csv',events);csv_write('breakdown_fraction.csv',breakdowns)
    csv_write('paired_arm_spread.csv',pairs);csv_write('variance_components.csv',variances)
    csv_write('physical_restore.csv',restore)
    primary=[r for r in breakdowns if r['clock_s']==600 and r['region']=='upstream']
    mixed=sum(0<r['fraction_breakdown']<1 for r in primary if r['fraction_breakdown'] is not None)
    team600=[r for r in summaries if r['clock_s']==600 and r['metric']=='J']
    effects=[r for r in pairs if r['clock_s']==600 and r['metric']=='J']
    unique_tapes=[dict(world=w,arm=a,distinct_seed_tapes=len({lookup[w,a,k]['tape_sha256'] for k in range(969086,969094)}))
                  for w in range(969040,969045) for a in ARMS]
    summary=dict(states=5,state_arm_cells=10,branches=80,charged_sim_s=49869,actual_sim_s=49869,
                 max600_seed_sd_veh_s=max(r['sd'] for r in team600),max600_seed_range_veh_s=max(r['maximum']-r['minimum'] for r in team600),
                 mixed_upstream_state_arm_cells=mixed,upstream600=primary,paired600=effects,
                 variance600=[r for r in variances if r['clock_s']==600 and r['metric']=='J'],
                 tape_diversity=unique_tapes,detector=binding,
                 max_position_error_m=max(r['max_position_m'] for r in restore),max_speed_error_mps=max(r['max_speed_mps'] for r in restore),
                 transient_pending_cache_mismatches=sum(not r['pending_api_equal'] for r in restore),
                 collection_sha256=hashlib.sha256((EVIDENCE/'collection.json').read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    figure(summaries,pairs,summary)
    report(summary,summaries)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('detector','tape_diversity','upstream600','paired600')},indent=2))


def figure(summaries,pairs,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.4,1]})
    labels=['5% / 2400','10% / 2700','20% / 2400','40% / 2700','20% / 2700']
    for arm,offset,color in [('native',-.12,'#555555'),('mappo',.12,'#176B87')]:
        rows=[r for r in summaries if r['arm']==arm and r['clock_s']==600 and r['metric']=='J']
        means=np.array([r['mean'] for r in rows]);y=np.arange(5)+offset
        errors=np.array([[r['mean']-r['minimum'] for r in rows],[r['maximum']-r['mean'] for r in rows]])
        axes[0].errorbar(means,y,xerr=errors,fmt='o',capsize=4,color=color,label=arm.upper() if arm=='mappo' else 'Native')
    axes[0].set_yticks(np.arange(5),labels);axes[0].set_xlabel('600 s fleet + source delay (veh-s)')
    axes[0].set_ylabel('Fresh physical state: CAV share / demand (veh/h)');axes[0].invert_yaxis()
    axes[0].set_title('K8 mean and full seed range');axes[0].legend(frameon=False)
    delta=[r for r in pairs if r['clock_s']==600 and r['metric']=='J']
    for i,row in enumerate(delta):
        axes[1].errorbar(row['mean'],i,xerr=[[row['mean']-row['minimum']],[row['maximum']-row['mean']]],
                         fmt='o',color='#176B87',capsize=4)
    axes[1].axvline(0,color='#999999',lw=1)
    axes[1].set_yticks(np.arange(5),labels);axes[1].invert_yaxis()
    axes[1].set_xlabel('MAPPO − native delay at 600 s (veh-s)');axes[1].set_ylabel('Paired physical state')
    axes[1].set_title('Controller effect: K8 mean and full range')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.suptitle(f"SUMO continuation RNG ensemble: maximum within-state SD {summary['max600_seed_sd_veh_s']:.3g} veh-s",fontsize=13)
    fig.text(.01,.01,'Source: S-ENSPROBE fresh TRAIN969040–969044; SUMO1.27.1, t300→900; fixed requests/drivers, 8 new SUMO seeds per arm.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'ENSPROBE.png',dpi=180);plt.close(fig)


def report(summary,summaries):
    v={r['arm']:r for r in summary['variance600']}
    effects=summary['paired600'];ds=[r['mean'] for r in effects]
    lines=[
        '# S-ENSPROBE — conditional SUMO RNG ensemble',
        '',
        '## What was run',
        '- Five fresh TRAIN native-warmup states at t=300 s; paired native/speed-only MAPPO169, K8 seeds each, 600 s: 80 branches / 10 state-arm cells.',
        '- Worlds969040–969044; cells(.05,2400),(.10,2700),(.20,2400),(.40,2700),(.20,2700); continuation RNG969086–969093. Two ordinary-share and three dense/high-share states; same physical starts across arms.',
        '- Exact original plan49,500 sim-s; cache-gate recovery adds300 s native prefix; own scheduling stop retains69partial sim-s. **49,869 charged = 49,869 actual sim-s**; unused131 never allocated.',
        '- Frozen cpuQ e3final manifestb3c031ad, SUMO1.27.1, textbook population; both guards and registry/filename exposure audit passed. Requests/routes/driver population stay fixed; only SUMO RNG changes.',
        '',
        '## Physical restart validity',
        f"- 80/80 starts pass: max lane-position error {summary['max_position_error_m']:.3g} m, max speed error {summary['max_speed_error_mps']:.3g} m/s; xy/lanes/routes/route indices/due waiting objects/source queues/lane queues verified.",
        f"- Residual: getPendingVehicles() insertion-attempt cache differs at {summary['transient_pending_cache_mismatches']}/80 loads. First attempt stopped at0 ticks; zero-step diagnosis found36/36 due vehicles still loaded/off-road/not departed with correct routes. Amended direct-object queue gate is pre-outcome and published.",
        '- Repeated first warmup reproduces original physical trunk exactly. save-state.rng=false and each requested new --seed verified. CF/LC internal memory need not reproduce the un-restored trunk continuation: this is a physical-state-conditional restored ensemble.',
        '- A second0-tick stop exposed deepcopy memo reuse. Fresh memos fix it; first-state bookkeeping was reconstructed at0ticks from the same public t0 state/due waiters, with exact retained prehistory and native600 result. MAPPO uses the frozen current-frame state API and fresh controller memory; four later worlds retain full native Python histories. No completed physics rerun or result-based selection.',
        '- Spare-CPU admission was corrected after an owned cooperative scheduling stop; only its incomplete69s prefix reran and all ticks remain charged. One nice19 thread, load>20/own-YIELD pause; priority peers untouched.',
        '',
        '## Distribution results',
        f"- Maximum within-state/arm600 s team-delay SD = **{summary['max600_seed_sd_veh_s']:.6g} veh-s**; maximum min–max range = **{summary['max600_seed_range_veh_s']:.6g} veh-s**.",
        f"- Same-state sometimes-breaks/sometimes-not: **{summary['mixed_upstream_state_arm_cells']}/10 state-arm cells** under the unchanged LRDIAG BIC detector (600 s upstream); onset and preexisting flags per seed in CSV. No speed cutoff or behavioral threshold fitted.",
        f"- At600 s, pooled within-seed variance native/MAPPO = {v['native']['within_seed_variance']:.6g}/{v['mappo']['within_seed_variance']:.6g} (veh-s)^2; corrected between-state variance = {v['native']['between_state_variance']:.6g}/{v['mappo']['between_state_variance']:.6g} (veh-s)^2 (five states only, descriptive).",
        f"- Paired MAPPO-minus-native600 s state-mean effects span {min(ds):+.3f} to {max(ds):+.3f} veh-s; mean across five states {np.mean(ds):+.3f} veh-s. Largest within-state SD of that paired effect = {max(r['sd'] for r in effects):.6g} veh-s. Positive means MAPPO worse.",
        '- All120/300/600 s fleet/source/team means, sample SDs, minima/maxima, paired spreads and variance components are in the CSVs. Event timing uses full on-road mapped lanes,10 s half-open vehicle-time bins and60 s prehistory; missing/preexisting distinct.',
        '',
        '## Plain-language answer and scope',
        '- This ensemble changes simulator RNG, not the future demand/driver realization. It cannot by itself measure the real-world metastable distribution conditional on an observable traffic state.',
        '- The code audit identifies randomness in Python-generated departures, CAV identities and heterogeneous IDM/LC parameters, fixed before the continuation. IDM has no dawdling noise; sigma=0, speedDev=0, lcSigma defaults0; insertion best/max and fixed routes add no identified process noise.',
        '- A degenerate seed ensemble therefore indicates insufficient configured continuation stochasticity, not proof that real traffic is deterministic or that a WM should have no uncertainty. Observable-state uncertainty and fresh future arrivals/drivers require a separately registered ensemble design.',
        '- Complete audit and version-pinned documentation: [ENSPROBE_RANDOMNESS.md](ENSPROBE_RANDOMNESS.md). No assignment/truth/forbidden paths inspected; realized type-specific quantities remain undetermined.',
        '',
        '## Artifacts',
        '- [Figure: K8 delay distributions and controller effects](ensprobe/ENSPROBE.png).',
        '- CSVs: ensprobe/{per_seed_delay,state_spread,breakdown_per_seed,breakdown_fraction,paired_arm_spread,variance_components,physical_restore}.csv.',
        '- Provenance: tools/ensprobe/evidence/{collection,v4_stage,v4_cpu,queue_diagnosis,waiting_diagnosis}.json; ensprobe/summary.json includes exact detector source/hash. Old STOPs and all original attempts retained on cpuQ.',
    ]
    assert len(lines)<=50
    (ROOT/'docs/paper_results/ENSPROBE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.verify_only:
        panel=json.loads((EVIDENCE/'collection.json').read_text());verify(panel)
        print(json.dumps(dict(verified_branches=80,physical_gates=80,charged_sim_s=49869,actual_sim_s=49869,
                              retained_result_sha256_matches=True)))
    else:main()
