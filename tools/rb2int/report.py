"""Plain-language final delivery assembled only from completed registered scores."""
import argparse
import csv
import json
from pathlib import Path

from tools.rb2int.audit import ROOT
from tools.rbeval.core import load, digest, save_new

MODELS=('WM2','FULL-v4','FULL-v5','FULL-v5-LP')
HS=(120,300,600)


def rows(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def mean(records,field):
    values=[float(r[field]) for r in records if r.get(field) not in (None,'','None')]
    return sum(values)/len(values) if values else None


def number(value,digits=0):
    return 'NA' if value is None else f'{value:,.{digits}f}'


def interval(row,scale=1.,digits=0):
    point=number(row.get('mean')*scale if row.get('mean') is not None else None,digits)
    return point+' ['+number(row['low']*scale if row.get('low') is not None else None,digits)+', '+number(row['high']*scale if row.get('high') is not None else None,digits)+']'


def render(score_root,comparison):
    root=Path(score_root);comp=load(comparison)
    if not comp['complete_registered_scope'] or comp['scope']!='primary54' or len(comp['matched_complete_worlds'])!=54:
        raise ValueError('a complete registered primary comparison is required for final delivery')
    candidates=('FULL-v5','FULL-v5-LP')
    statuses={m:comp['switching'][m]['switching_eligibility'] for m in candidates}
    gates={m:load(root/m/'gate/gate.json') for m in MODELS}
    lines=['# WM-PHYS v5 — final registered comparison',
        '**The fixed candidates have now been evaluated; neither can establish full12-cell distributional qualification on this recorded panel.**',
        'Switching eligibility: '+ '; '.join(m+' '+statuses[m] for m in candidates)+'.',
        'The headline comparisons below use all54 predeclared replicated-cell worlds, allfivearms and K8 at120/300/600s. Four singleton worlds remain descriptive.',
        '', '## What was fixed and frozen',
        '- Sourceb2df7458; SUMO IDM/LC2013/merge/insertion, causal physical driver/LC posteriors and wishes, physical latent-spread fitting, lane-speed and CAV-setting consistency.',
        '- LP differs only by the learned physical departure mixture; v5/LP share hardware for each world. All learned corrections OFF; no neural/black-box retraining or SUMO.',
        '- Original WM2/FULL-v4 code, weights and controllers were hash verified; dev kernel replays matched exactly.128completed reserved-baseline futures were reused.',
        '- All-four-source135CPU tests and both two-world/all-five-arm600s canaries PASS. LC attachment now preserves CAL\'s entrant-mixture wrapper.',
        '', '## Distributional comparison (negative contrast is better)']
    absolute={}
    primary=set(comp['matched_complete_worlds'])
    for m in MODELS:
        table=[r for r in rows(root/m/'gate/distribution_worlds.csv') if int(r['seed']) in primary]
        delay=[r for r in table if r['metric']=='team_delay_vs']
        absolute[m]={h:dict(crps=mean([r for r in delay if int(r['horizon_s'])==h],'wm_crps'),
                           bias=mean([r for r in delay if int(r['horizon_s'])==h],'error')) for h in HS}
        lines.append('- '+m+' delay CRPS120/300/600 (veh-s): '+ '/'.join(number(absolute[m][h]['crps']) for h in HS)+
                     ';600s mean delay bias='+number(absolute[m][600]['bias'])+'veh-s.')
    lines.insert(4,'At600s the mean delay bias is '+number(absolute['FULL-v5'][600]['bias'])+'veh-s for v5 and '+
        number(absolute['FULL-v5-LP'][600]['bias'])+' for LP, versus '+number(absolute['FULL-v4'][600]['bias'])+
        ' for v4 and '+number(absolute['WM2'][600]['bias'])+' for WM2 (same registered54 worlds).')
    for a,b in (('FULL-v5','WM2'),('FULL-v5','FULL-v4'),('FULL-v5-LP','WM2'),('FULL-v5-LP','FULL-v4'),('FULL-v5-LP','FULL-v5')):
        selected=[r for r in comp['delay_crps'] if (r['candidate'],r['baseline'])==(a,b)]
        lines.append('- '+a+'−'+b+' delay CRPS120/300/600; simultaneous95% CIs: '+' / '.join(interval(r) for r in selected)+'veh-s.')
    lines += ['', '## Breakdown, timing, traffic fields and gate']
    for m in MODELS:
        event=rows(root/m/'gate/events_worlds.csv')
        upstream=[r for r in event if r['region']=='upstream' and int(r['seed']) in primary]
        predicted=[mean([r for r in upstream if int(r['horizon_s'])==h],'wm_event_probability') for h in HS]
        actual=[]
        for h in HS:
            vals=[r['real_event'] for r in upstream if int(r['horizon_s'])==h and r['real_event'] in ('True','False')]
            actual.append(sum(v=='True' for v in vals)/len(vals) if vals else None)
        lines.append('- '+m+' pooled upstream rates120/300/600: forecast '+ '/'.join(number(v*100 if v is not None else None,1) for v in predicted)+
            '%; recorded '+ '/'.join(number(v*100 if v is not None else None,1) for v in actual)+'%.')
        lines.append('- '+m+' gate NOT_QUALIFIED: '+'; '.join(gates[m]['reasons'])+'.')
    lines += ['- Full12-cell qualification remains unavailable: four singleton cells and insufficient simultaneous rate precision. This is not cured by favorable pooled means.',
        '- Cell/arm timing CDF/IBS, regional speed CRPS/support, source/road components, expected native/CTRL0 shifts, arm-order and PIT/range diagnostics are retained per model in the gate evidence.',
        '- BIC detects a dominant speed-loss step, not proof of a metastable phase transition. Pre-existing and no-onset mass remain separate.',
        '', '## Switching eligibility and uncertainty']
    for m in candidates:
        rate=next(r for r in comp['upstream_rate_distance'] if (r['candidate'],r['baseline'],r['horizon_s'])==(m,'WM2',300))
        endpoint=next(r for r in comp['hdv60'] if (r['candidate'],r['baseline'])==(m,'WM2'))
        status=comp['switching'][m]
        lines.append('- '+m+'−WM2:300s breakdown-distance '+interval(rate,100,2)+'pp; HDV60 mean-world-median error '+interval(endpoint,1,2)+'m; common endpoints '+str(endpoint['counts'].get('common_hdv',0))+'.')
        lines.append('- '+m+' point rule '+status['point_status']+', simultaneous support '+status['simultaneous_support']+': '+
            '; '.join(status['point_failures']+status['missing_evidence'] or ['all point conditions satisfied'])+'.')
    lines += ['- 10,000fixed demand×penetration paired-world resamples, seed20261101. Every contributing scored stratum replicated; no arm/future pseudoreplication or unscored-peer variation.',
        '- Delay/HDV intervals condition on stored K8; rate means additionally include paired future resampling. Absolute rate-distance CIs conservatively propagate joint signed-bias bands through the zero-bias kink.',
        '- Intervals are finite-sample bootstrap approximations; two-world cells do not establish precision. Missing/zero-estimable support is NA. Source/attrition manifests are retained.',
        '', '## Decision credibility (fixed0.8rule; no transplanted v4 map)']
    for m in MODELS:
        credibility=load(root/m/'credibility/CREDIBILITY.json')
        summary=credibility['summary']
        calls=[next(r for r in summary if (r['target'],r['metric'],r['horizon_s'])==('selected_preference','outcome',h)) for h in HS]
        forecast=[next(r for r in summary if (r['target'],r['metric'],r['horizon_s'])==('selected_preference','probability',h)) for h in HS]
        ranges=[next(r for r in summary if r['target']=='delay' and r['metric']=='covered' and r['interval']=='range' and r['horizon_s']==h) for h in HS]
        lines.append('- '+m+' selected calls120/300/600='+ '/'.join(str(r['count']) for r in calls)+
            '; observed wins '+ '/'.join(number(r['mean']*100 if r['mean'] is not None else None,1) for r in calls)+
            '% vs forecast '+ '/'.join(number(r['mean']*100 if r['mean'] is not None else None,1) for r in forecast)+
            '%; absolute ensemble-range coverage '+ '/'.join(number(r['mean']*100 if r['mean'] is not None else None,1) for r in ranges)+'%.')
    lines += ['- Strict paired P(beats incumbent)>=0.8 AND lower expected delay; ties are not wins. Central50/80/90% intervals and range are separate; K8 continuous exchangeable range reference is7/9, not90%.',
        '- Credibility confidence bands resample complete worlds, retaining zero-selected opportunities; empty/single-world selected bins cannot certify reliability.',
        '', '## Scope and delivery',
        '- Fitting/selection only969400–969431 and declared development evidence. Additional recorded worlds were exposure-audited before use; prior aggregate exposure is disclosed, not historically untouched data.',
        '- Reserved8 results and full58/singleton sensitivities remain separate. Primary54 is the frozen recorded mixture of eight replicated cells, not a uniform12-cell population.',
        '- Remaining engineering limits: serial LC/internal-junction negotiation and some cold/critical-leader state are incomplete. Conditional LC precision and narrow LP birth scores are not full-model qualification.',
        '- S-WMTEACH received immutable FULL-v5/LP engineering readiness; roster-only inputs need explicit prior/cold-memory adapters. No automatic controller/policy adoption follows.',
        '- Immutable release/source/recipe/data/ownership, matched task attrition, all six model contrasts and per-model gate/credibility evidence accompany this report.']
    return '\n'.join(lines)+'\n',dict(absolute=absolute,switching=statuses,worlds=comp['matched_complete_worlds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',required=True)
    parser.add_argument('--comparison',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    text,summary=render(args.scores,args.comparison)
    out=Path(args.out)
    if not out.parent.is_dir(): raise ValueError('existing report parent required')
    out.write_text(text,encoding='utf-8')
    receipt=out.with_suffix('.delivery.json')
    save_new(receipt,dict(report_sha256=digest(out),comparison_sha256=digest(args.comparison),
        source_sha256=digest(__file__),summary=summary,final_inference_repeated=False))
    print('FINAL REPORT',out,len(text.splitlines()),'lines')


if __name__=='__main__':
    main()
