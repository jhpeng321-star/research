"""Render the <=60-line plain-language report from verified scalar CSV outputs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/paper_results/bdrank'


def rows(name):
    with (OUT/(name+'.csv')).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def one(data, **where):
    selected = [r for r in data if all(r.get(k) == str(v) for k, v in where.items())]
    assert len(selected) == 1, (where, len(selected))
    return selected[0]


def main():
    costs = rows('cost_by_status')
    fits = rows('cost_variance_decomposition')
    ranks = rows('controller_concordance')
    ds = rows('controller_decision_summary')
    summary = json.loads((OUT/'summary.json').read_text())
    minutes = summary['option_missing']['estimated_repeat_worker_gpu_minutes']
    rb2 = summary['round2']
    controller_minutes = json.loads((OUT/'controller_cache_manifest.json').read_text())['total_recorded_worker_gpu_minutes']
    def cost(split, status, h):
        return float(one(costs, split=split, region='upstream', horizon_s=h, level='pooled', status=status)['mean_J'])
    def costs_text(split):
        return '; '.join(f"H{h} {cost(split,'event',h):,.0f}/{cost(split,'no_event',h):,.0f}" for h in (120,300,600))
    def rnk(split, signal, target='mean_delay', adjusted=True):
        return '/'.join(f"{100*float(one(ranks,split=split,region='upstream',horizon_s=h,signal=signal,target=target,level='pooled')['tie_adjusted_concordance' if adjusted else 'concordance']):.1f}" for h in (120,300,600))
    def decomp(split):
        r=one(fits, split=split, region='upstream', horizon_s=600, level='pooled')
        return '/'.join(f"{100*float(r[k]):.1f}%" for k in ('shapley_occurrence','shapley_timing','shapley_severity','residual_fraction'))
    def decision(split,h):
        text=[]
        for rule in ('probability','onset','severity','delay','breakdown_then_delay'):
            saving=float(one(ds,split=split,horizon_s=h,rule=rule,metric='saving',level='pooled')['mean'])
            regret=float(one(ds,split=split,horizon_s=h,rule=rule,metric='regret',level='pooled')['mean'])
            text.append(f'{saving:+,.0f}/{regret:,.0f}')
        return '; '.join(text)
    lines = [
        '# S-BDRANK — rank actions, not individual traffic futures',
        '**Answer: predicted mean delay ranks useful actions better than predicted breakdown percentage on development; probability saturation largely removes the latter’s ranking information.**',
        '**Do not switch to breakdown-first on this evidence:** all development-selected rules lose to MAPPO on the separate reserved panel; the caches do not identify a large causal cost of avoiding this detector’s event.',
        '',
        '## Contract and scope',
        '- Recorded cost: WMDATA969400–969431 development32, separate969432–969439 evaluation8; WMDATA2 replication969550–969599 (50 worlds, six controllers). 500 episodes total.',
        '- Controller forecasts: complete FULL-v4/22cc32f6, 20 worlds ×5 controllers ×K8; development12 (969420–969431) and reserved8 separate. WMTEACH: fixed12 diagnostic states, v4/v5 each96 cached futures, H120/H300.',
        '- Exact registered LRDIAG BIC detector SHA `6de4df03…`, 10s full-traffic region speeds/60s prehistory; upstream primary, merge/ramp CSV diagnostics. No speed thresholds. Pre-existing steps separate; horizon-specific dominant steps are not nested first-breakdown events.',
        '- J includes road delay **and source waiting**, in veh-s; positive saving means MAPPO minus chosen J. Each horizon uses the same-horizon recorded outcome. A detected speed-loss step is not proof of a phase transition.',
        '- 95% CIs use10,000 fixed-cell whole-world bootstrap draws, seed20261101. A singleton contributing demand×share stratum gives **NA**, including pooled controller/option claims here; replicated-cell CSV intervals remain scoped diagnostics.',
        '',
        '## Q1 — how expensive is recorded breakdown?',
        '- Upstream development event/no-new-event mean J: '+costs_text('dev32')+'; pre-existing cases excluded from these two statuses.',
        '- Replication event/no-new-event mean J: '+costs_text('replication50')+'.',
        '- Evaluation8 separately: '+costs_text('reserved8')+'; at600s only ONE no-event outcome, versus38 event/one pre-existing.',
        '- These pooled differences are confounded: cell×controller-standardized event-minus-no-event J300/J600 is −3,373/−5,457 in development, +2,959/+15,414 in replication. Only52/20 and193/96 episodes respectively support those contrasts; no causal avoidance-cost claim.',
        '- Within-world ALL-controller-pair regression, excluding pre-existing pairs: occurrence / timing / depth / unexplained shares of squared J600 differences are development **'+decomp('dev32')+'**.',
        '- Replication same decomposition: **'+decomp('replication50')+'**; evaluation8 separately: **'+decomp('reserved8')+'**. These are order-independent Shapley allocations of in-sample R², not causal mediation.',
        '- Development combined R² at120/300/600 is7.5/15.1/21.7%; leave-world-out R² −2.9/7.6/16.9%. Replication600s:19.9% fitted/17.9% leave-world-out. Most controller variation remains unexplained.',
        '- Conditional paired600s depth slopes: +1,264 development/+1,191 replication veh-s per m/s deeper fitted step. Timing and severity matter more than occurrence in this linear association; depth is not queue length.',
        '- Confirm WMBREAKDOWN narrowly: development600s upstream MAPPO/PI/CIRCLES shifts vs native or CTRL0 are +3.125/−3.125/+3.125pp, despite MAPPO/PI J gains486/1,498 vs CTRL0.',
        '- The ±3pp statement is not universal: replication300s CIRCLES is +16pp vs native/+18pp vs CTRL0; evaluation600s all three are +12.5pp. All cells, references and horizons are exported.',
        '',
        '## Q2 — which ranking signal is right?',
        '- Below: tie-adjusted concordance against **real delay ordering**, %, H120/300/600; predicted ties earn½, factual ties are excluded. Untied concordance and all three tie classes are exported separately.',
        '- Development: probability '+rnk('development12','occurrence')+'; later onset '+rnk('development12','onset_rmst')+'; lower depth '+rnk('development12','depth_expected')+'; **mean delay '+rnk('development12','mean_delay')+'**.',
        '- Reserved8 separately: probability '+rnk('reserved8','occurrence')+'; onset '+rnk('reserved8','onset_rmst')+'; depth '+rnk('reserved8','depth_expected')+'; mean delay '+rnk('reserved8','mean_delay')+'.',
        '- Matching-signal untied fidelity, development: event '+rnk('development12','occurrence','occurrence',False)+'; onset '+rnk('development12','onset_rmst','onset_rmst',False)+'; depth '+rnk('development12','depth_expected','depth_expected',False)+'%. These score the matching factual signal, not delay.',
        '- Matching-signal reserved fidelity: event '+rnk('reserved8','occurrence','occurrence',False)+'; onset '+rnk('reserved8','onset_rmst','onset_rmst',False)+'; depth '+rnk('reserved8','depth_expected','depth_expected',False)+'%. Event600s has only1 development/3 reserved doubly-untied pairs: 0% is extremely sparse.',
        '- Saturation: probability ties on112/120 development and68/80 reserved600s controller pairs; against real delay its untied score62.5%/50% hides93.3%/85% tied predictions. Mean delay has no predicted ties and gets82/120 and52/80 right.',
        '- WMTEACH available O0–O4 delay-pair concordance: v4 **65.5/75.9%**, v5 **63.8/51.7%** at120/300s (58 pairs/horizon/version, zero prediction ties). Unavailable options are excluded by the inherited availability mask.',
        '- Exact-choice agreement v4 50.0/66.7%, v5 16.7/33.3%. These are roster-only/source-prior/cold-memory diagnostics, one state per cell; neither a live-policy test nor student fitting data.',
        '',
        '## Q3 — fixed choice rules, development first',
        '- Rule order (a–e): lowest P(event); latest restricted onset; lowest expected detected depth; lowest mean delay; lexicographic P(event)-then-delay. Numbers below are **saving/regret** per world, veh-s.',
        '- Onset score integrates detector-onset survival to H: new onset t, no-event censored at H, pre-existing0; it does not impute an exact unseen onset. Depth=E[event × fitted depth], zero without a new event; conditional onset/depth are separately exported.',
        '- Development120: '+decision('development12',120)+'.',
        '- Development300: '+decision('development12',300)+'.',
        '- Development600: '+decision('development12',600)+'.',
        '- Reserved120: '+decision('reserved8',120)+'.',
        '- Reserved300: '+decision('reserved8',300)+'.',
        '- Reserved600: '+decision('reserved8',600)+'.',
        '- Development selects probability at120, delay at300/600; frozen before constructing reserved choices. Confirmation savings **−14/−393/−480**, regrets187/1,207/2,533. No reserved rule is selected.',
        '- Exact prediction ties prefer MAPPO, then native/CTRL0/PI/CIRCLES. Probability ties at the minimum in9/10/10 of12 development worlds; its modest savings depend heavily on fallback. CSVs include full tie sets and best/worst tie-choice ranges.',
        '- This is hypothetical selection among the five recorded controllers, not a validated online policy; no P(win) gate is added. Delay ranks best on development at every horizon, but ranking accuracy alone does not guarantee positive selected-action saving.',
        '- Option delay-only chosen saving/regret: v4 H120 +19.5/40.2, H300 +680.3/593.0; v5 −28.2/87.9 and +172.7/1,100.6. O0=MAPPO; diagnostic exact labels never select controller rules.',
        '',
        '## Stopped items and evidence',
        f'- **Duration and queue extent STOP:** absent from the registered definition/output; no threshold or time-to-horizon substitute. Real reductions/defining a duration metric need0 GPU-min. If no separate spatial archive exists, controller queue extent needs160 five-controller H600 forecasts saving position/speed/lane/presence: approximately{controller_minutes:.1f} occupied GPU-min plus instrumentation, after a definition is registered; not run.',
        '- **Option breakdown/onset/severity and rules(a,b,c,e) STOP:** read-only audit of all192 cached v4/v5 futures finds scalar delay and `breakdown=None`, no regional speed/prehistory; supplied exact option labels likewise contain delay effects only.',
        f"- Needed, not run:96 paired H300 option futures/version saving registered10s speed bins + valid60s observable prehistory. Observed task totals estimate **{minutes['FULL-v4']:.1f} GPU-min v4 +{minutes['FULL-v5']:.1f} v5 ={sum(minutes.values()):.1f}**, excluding instrumentation overhead. Exact detector labels additionally require archived exact trajectories; if absent, new SUMO would be required and is outside this task.",
        f"- Round2 controller-v5 panel remains pending in frozen16:16:43 receipt:{rb2['complete_world_jobs']}/{rb2['expected_world_jobs']} complete world jobs across models. Only completion metadata read; no partial-panel efficacy, redispatch or peer-job interaction.",
        '- [CSV/evidence index](bdrank/README.md) · [analysis definitions/reproduction](../../tools/bdrank/README.md) · [input/task SHA manifests](bdrank/input_manifest.json) · [verification](bdrank/verification.json).',
        '- Analysis rechecks11,700 registered detector outputs and all source-inclusive clock sums; independent verification checks source hashes, pair/tie arithmetic, decision/regret identities and paired-regression equivalence to within-world fixed effects. No SUMO, training, new inference or process signals.',
        '- Retrospective, sparse, historically exposed panels: association and ranking diagnostics only. A lower predicted breakdown percentage is not presently a demonstrated better decision rule.',
    ]
    assert len(lines) <= 60, len(lines)
    (ROOT/'docs/paper_results/BDRANK.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('BDRANK report lines:', len(lines))


if __name__ == '__main__':
    main()
