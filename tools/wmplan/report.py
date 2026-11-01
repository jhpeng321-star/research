"""Independent full-panel accounting, forecast/action QA and paired inference."""
import argparse
from collections import Counter
from copy import deepcopy
import json

import numpy as np

from tools.coopbandit.protocol import canonical, digest, sha
from tools.coopplan.planner import TIMES, candidates
from tools.coopplan.predictor import Prediction, Outcome, ObservableState
from tools.wmplan.design import ARMS, CREDIT, H_CREDIT, panel, budget, decision, admit
from tools.wmplan.ops import LOCAL, OUT, save


def ci(values):
    x = np.asarray(values,float)
    if len(x) != 24 or not np.isfinite(x).all():
        raise ValueError('all24 finite paired worlds required')
    draws = np.random.default_rng(20261101).integers(0,24,(10000,24))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[draws].mean(1),[.025,.975]).tolist(),worlds=24)


def credit_ci(world_credits):
    sums = np.array([sum(c['C_G'] for c in rows) for rows in world_credits],float)
    counts = np.array([len(rows) for rows in world_credits])
    if not counts.sum():
        return None
    draws = np.random.default_rng(20261101).integers(0,24,(10000,24))
    den = counts[draws].sum(1)
    good = den > 0
    boot = sums[draws].sum(1)[good]/den[good]
    dg = np.array([sum(c['D_G'] for c in rows) for rows in world_credits],float)
    solo = np.array([sum(sum(c['member_alone_D']) for c in rows) for rows in world_credits],float)
    return dict(mean=float(sums.sum()/counts.sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                joint_saving_mean=float(dg.sum()/counts.sum()),
                sum_member_alone_saving_mean=float(solo.sum()/counts.sum()),
                events=int(counts.sum()),positive=sum(c['C_G']>0 for rows in world_credits for c in rows),
                beneficial_superadditive=sum(c['C_G']>0 and c['D_G']>0 for rows in world_credits for c in rows),
                active_worlds=int((counts>0).sum()),zero_event_draws=int((~good).sum()),
                resampling='whole24world clusters including zero-action worlds; event-weighted mean')


def build(r, primary=False):
    arms = ('expected',) if primary else ARMS
    complete = r.get('primary_complete',False) if primary else r['complete']
    if not complete or r['stopped'] or r['incomplete'] or r['episodes'] != 24*len(arms):
        raise ValueError('full completed panel required')
    if r['actual'] != r['charged'] or r['charged'] > budget()['total']:
        raise ValueError('charge/actual cap')
    restart = r['binding'].get('restart')
    if restart and (r['charged'] > restart['new_run_upper_sim_s'] or
                    r['charged']+restart['prior_charged_sim_s'] > restart['combined_charged_upper_sim_s'] or
                    r['actual']+restart['prior_actual_sim_s'] > restart['combined_actual_upper_sim_s']):
        raise ValueError('combined restart cap including retained prior charges')
    charges = {v['id']:v for v in r['charges']}
    closes = {v['id']:v for v in r['closed']}
    if len(charges) != len(r['charges']) or charges.keys() != closes.keys():
        raise ValueError('unique closed attempts')
    for k,c in charges.items():
        if c['sim_s'] != closes[k]['actual_sim_s'] or c['sim_s'] != closes[k]['charged_sim_s']:
            raise ValueError('attempt accounting')
    predictions = {}
    for identity,f in r['forecasts'].items():
        p = Prediction({a:tuple(Outcome(**v) for v in fs) for a,fs in f['futures'].items()},
                       False,'WM-PHYS-'+f['backend'])
        p.validate([dict(id=a,members=[]) for a in p.actions],8)
        key = digest({a:[vars(v) for v in fs] for a,fs in p.actions.items()})
        predictions[key] = (p,r['observations'][identity])
    summaries, worlds, credits, expected_jobs = {},[],[],set()
    for arm in arms:
        safety = {a:{k:0. for k in ('collisions','teleports','ttc_below_3_vs','hard_brake')}
                  for a in ('mappo',arm)}
        actions, proposals = Counter(),Counter()
        pruned = requests = overrides = zero = 0
        arm_rows, arm_credits = [],[]
        for w in panel():
            seed = w['seed'];ident = f's{seed}_{arm}'
            s = r['results'][ident];b = r['baselines'][str(seed)]
            expected_jobs.add(ident)
            if s['world'] != w or b['world'] != w or s['arm'] != arm or not s['warmup_pass']:
                raise ValueError('world/arm/prefix binding')
            if s['actual_sim_s'] != 1200 or b['actual_sim_s'] != 1200:
                raise ValueError('full episodes')
            if [e['time_s'] for e in s['history']] != list(TIMES):
                raise ValueError('cadence')
            for name,data in (('mappo',b),(arm,s)):
                for k in safety[name]:
                    safety[name][k] += data['clocks']['900'][k]
            bytime = {c['time_s']:c for c in s['credits']}
            if len(bytime) != len(s['credits']):
                raise ValueError('duplicate group credits')
            spent = 0
            for e in s['history']:
                t = e['time_s'];key = f's{seed}_{arm}_t{t}'
                p,observation = predictions[e['prediction_id']]
                if observation['world_id'] != seed or observation['time_s'] != t:
                    raise ValueError('forecast observable world/clock binding')
                available = candidates(ObservableState(t,tuple(observation['rows']),key))
                if canonical(available) != canonical(e['candidates']):
                    raise ValueError('observable-only candidate reconstruction')
                p.validate(available,8)
                d = decision(p,arm)
                if canonical(d) != canonical(e['decision']):
                    raise ValueError('frozen forecast/decision mismatch')
                chosen = next(a for a in e['candidates'] if a['id']==d['selected'])
                v = dict(time_s=t,executed_action=deepcopy(chosen),group=chosen['members'])
                cost = admit(v,spent)
                for field in ('proposed_action','executed_action','budget_pruned'):
                    if canonical(v[field]) != canonical(e[field]):
                        raise ValueError('action/resource quota mismatch')
                name = e['executed_action']['id']
                actions[name]+=1;proposals[e['proposed_action']['id']]+=1
                pruned += e['budget_pruned']
                if name == 'O0':
                    if t in bytime:
                        raise ValueError('credit on non-executed action')
                    continue
                c = bytime[t]
                ids = [key+'_O0',key+'_'+name,key+'_'+name+'_solo0',key+'_'+name+'_solo1']
                branches = [r['results'][i] for i in ids]
                for i,branch in zip(ids,branches):
                    if not branch['prefix_pass'] or branch['actual_sim_s'] != t+H_CREDIT or charges[i]['kind'] != 'credit':
                        raise ValueError('exact quartet prefix/charge')
                js = [v['metrics']['team_delay_vs'] for v in branches]
                dg = js[0]-js[1];ds = [js[0]-j for j in js[2:]]
                if (c['action'] != chosen or c['horizon_s'] != H_CREDIT or c['D_G'] != dg or
                        c['member_alone_D'] != ds or c['C_G'] != dg-sum(ds)):
                    raise ValueError('complete exact C_G arithmetic/action')
                if r['reservations'][key] != dict(cost=cost,world=w,time_s=t,action=name):
                    raise ValueError('complete reservation binding')
                expected_jobs.update(ids)
                credits.append(dict(world_id=seed,arm=arm,**c));spent += cost
            if spent != s['credit_sim_s'] or spent > CREDIT:
                raise ValueError('per-world-arm quota')
            if not s['credits']:
                if not s['all_O0_bitwise_pass'] or b['final_prefix'] != s['final_prefix'] or b['clocks'] != s['clocks']:
                    raise ValueError('allO0 baseline bitwise/metrics mismatch')
                zero += 1
            row = dict(world_id=seed,arm=arm,cell=w['cell'],credit_sim_s=spent)
            for clock in ('600','900'):
                for metric in ('team_delay_vs','road_delay_vs','source_wait_vs'):
                    row[f'saving_J{clock}_{metric}'] = b['clocks'][clock][metric]-s['clocks'][clock][metric]
            arm_rows.append(row);worlds.append(row);arm_credits.append(s['credits'])
            requests+=s['requests'];overrides+=s['overrides']
        summaries[arm] = dict(savings={clock:{metric:ci([v[f'saving_J{clock}_{metric}'] for v in arm_rows])
                    for metric in ('team_delay_vs','road_delay_vs','source_wait_vs')} for clock in ('600','900')},
                    actions=dict(actions),proposals=dict(proposals),resource_refusals=pruned,
                    C_G=credit_ci(arm_credits),safety=safety,requests=requests,overrides=overrides,
                    all_O0_bitwise_worlds=zero)
    if set(r['results']) != expected_jobs or set(charges) != expected_jobs:
        raise ValueError('unexpected/missing job or branch')
    if len(r['reservations']) != len(credits):
        raise ValueError('outstanding reservations')
    accounting = dict(restart=restart,combined_actual=r['actual']+restart['prior_actual_sim_s'],
                      combined_charged=r['charged']+restart['prior_charged_sim_s']) if restart else {}
    return (dict(arms=summaries,worlds=24,actual=r['actual'],charged=r['charged'],upper=budget()['total'],**accounting,
                model=r['binding']['model'],decisions_per_arm=288,executed_groups=len(credits),
                exact_credit_prefixes=4*len(credits),bootstrap=dict(resamples=10000,seed=20261101,unit='whole_world')),
            worlds,credits)


def run(verify=False, primary=False):
    prefix = 'PRIMARY_' if primary else ''
    r = json.loads((OUT/(prefix+'RECEIPTS.json')).read_text())
    if r['binding'] != json.loads((OUT/'STAGED.json').read_text())['binding']:
        raise ValueError('staging identity')
    summary,worlds,credits = build(r,primary)
    distill = json.loads((LOCAL.parent/'c4-distill-1101/docs/paper_results/distill_live/WORLD_METRICS.json').read_text())
    byseed = {v['world_id']:v for v in distill}
    if set(byseed) != {w['seed'] for w in panel()}:
        raise ValueError('student comparison panel mismatch')
    summary['paired_vs_SUMO_teacher_student'] = {
        arm:{clock:ci([v[f'saving_J{clock}_team_delay_vs']-byseed[v['world_id']][f'saving_J{clock}']
                       for v in worlds if v['arm']==arm]) for clock in ('600','900')} for arm in summary['arms']}
    summary['paired_vs_WM_teacher_student'] = None
    if r.get('wm_student'):
        pairs = r['wm_student']['pairs']
        if [p['world'] for p in pairs] != panel():
            raise ValueError('WM student exact same panel required')
        byseed = {}
        for p in pairs:
            seed = str(p['world']['seed'])
            if (p['baseline']['final_prefix'] != r['baselines'][seed]['final_prefix'] or
                    p['baseline']['clocks'] != r['baselines'][seed]['clocks']):
                raise ValueError('WM student baseline bitwise comparability')
            byseed[int(seed)] = {clock:p['baseline']['clocks'][clock]['team_delay_vs']-
                                p['student']['clocks'][clock]['team_delay_vs'] for clock in ('600','900')}
        summary['paired_vs_WM_teacher_student'] = {
            arm:{clock:ci([v[f'saving_J{clock}_team_delay_vs']-byseed[v['world_id']][clock]
                           for v in worlds if v['arm']==arm]) for clock in ('600','900')} for arm in summary['arms']}
    qa = dict(pass_=True,all24pairs_each_arm=True,all_executed_groups_have_twins=True,
               forecast_action_accounting_recomputed=True,receipt_sha256=sha(OUT/(prefix+'RECEIPTS.json')),
               analysis_sources={str(p.relative_to(LOCAL)):sha(p) for p in
                                 (LOCAL/'tools/wmplan/report.py',LOCAL/'tools/wmplan/design.py')})
    for name,value in [('SUMMARY.json',summary),('WORLD_METRICS.json',worlds),('CREDITS.json',credits),('QA.json',qa)]:
        if verify:
            if json.loads((OUT/(prefix+name)).read_text()) != value:
                raise ValueError('report recomputation mismatch '+name)
        else:
            save(prefix+name,value)
    print(json.dumps(summary))


def publish(primary=False):
    from tools.wmplan.ops import REG
    prefix = 'PRIMARY_' if primary else ''
    s = json.loads((OUT/(prefix+'SUMMARY.json')).read_text())
    model = json.loads((OUT/'MODEL_COMPARISON.json').read_text())
    def interval(v):
        return f'{v["mean"]:+.3f} [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]'
    expected = s['arms']['expected']
    restart = s.get('restart')
    accounting_text = (f'New run actual=charged {s["actual"]:g}; retained prior actual900/charged3600; '
                       f'combined actual **{s["combined_actual"]:g}**, charged **{s["combined_charged"]:g} sim-s** '
                       '(no refund), within approved combined199440charged/196740actual upper and owner200000cap.'
                       if restart else f'Actual=charged **{s["actual"]:g} sim-s**, below registered199200upper/owner200000cap.')
    supported = all(expected['savings'][clock]['team_delay_vs']['ci95'][0] > 0 for clock in ('600','900'))
    conclusion = ('The expected-saving planner improved both registered delay scores on this '
                  'resource-limited development panel.' if supported else
                  'This test does not establish that the expected-saving planner beats MAPPO on both registered delay scores.')
    lines = ['# S-WMPLAN — online world-model-guided cooperative planner','',
             '## Plain-language result',conclusion,
             'The world model was used to look ahead at decision time; this is an online planner, not a trained student.',
              'Every executed group received complete exact member-alone twins. The simulation budget restricted each world-arm '
             'to at most one early group; these results do not measure an unrestricted planner.','',
              'Positive saving below means less source-inclusive full-team delay than MAPPO. Units are veh-s/world; '
             'brackets are 95% paired whole-world bootstrap intervals (10,000 resamples, RNG 20261101).','',
              '## Paired efficacy and behaviour']
    if restart:
        lines[6:6] = ['MAIN15:55 approved an engineering restart after an initial_rows executor-ABI startup fault; '
                      'the failed v1 start had no efficacy endpoint. All900actual/3600charged sim-s remain accounted; '
                      'the successful first primary episode was the smoke and was counted once.','']
    if primary:
        lines += ['**Primary arm complete; secondary strict arm pending.** The strict arm starts after this report is saved.','']
    for arm in s['arms']:
        a = s['arms'][arm]
        lines += [f'### {arm}'+(' — PRIMARY' if arm=='expected' else ' — SECONDARY'),
                  f'- J600 saving: **{interval(a["savings"]["600"]["team_delay_vs"])}**.',
                  f'- J900 saving: **{interval(a["savings"]["900"]["team_delay_vs"])}**.',
                  f'- Executed options: {json.dumps(a["actions"],sort_keys=True)}; raw proposals: '
                  f'{json.dumps(a["proposals"],sort_keys=True)}; resource-forced O0: {a["resource_refusals"]}.',
                  f'- All-O0 worlds bitwise equal to reused MAPPO: {a["all_O0_bitwise_worlds"]}/24.']
        for clock in ('600','900'):
            lines += [f'- J{clock} decomposition: on-road {interval(a["savings"][clock]["road_delay_vs"])}; '
                      f'source-wait {interval(a["savings"][clock]["source_wait_vs"])}.']
        cg = a['C_G']
        if cg:
            lines += [f'- Exact H120 C_G: {interval(cg)} veh-s/group; {cg["positive"]}/{cg["events"]} positive, '
                      f'{cg["beneficial_superadditive"]} with BOTH D_G>0 and C_G>0. '
                      'CI resamples all24 world clusters, including zero-action worlds.',
                      f'- C_G decomposition: mean joint saving D_G {cg["joint_saving_mean"]:+.3f}; '
                      f'mean sum of member-alone savings {cg["sum_member_alone_saving_mean"]:+.3f} veh-s/group.']
        else:
            lines += ['- No executed group in this arm; no C_G estimate is defined.']
        lines += [f'- Safety MAPPO/planner: '+', '.join(f'{k} {a["safety"]["mappo"][k]:g}/{a["safety"][arm][k]:g}'
                  for k in ('collisions','teleports','ttc_below_3_vs','hard_brake'))+'.',
                  f'- Safety-clamped requests {a["overrides"]}/{a["requests"]}; TTC and hard braking above are exposure veh-s.']
    lines += ['', 'Option types: O0 = MAPPO; O1/L2 = each member slows relative to its initial speed; '
              'O2/L3 = cross-lane pair shares a fixed slower speed target; O3/L4 = ramp-conditioned gap/headway target; '
              'O4/L1 = gap/headway target. All retain inherited executors and safety.',
              '', '## Model choice made before live outcomes',
              f'- Selected **{model["selected"]}** by the preregistered two-condition rule; reused existing forecasts, new quality SUMO0.',
              f'- FULL-v4: helpful argmax {model["v4_helpful"]["exact_argmax_matches"]}/'
              f'{model["v4_helpful"]["helpful_states"]}, all12 exact chosen saving '
              f'{model["v4_all"]["exact_chosen_saving"]:+.6f}; helpful chosen saving '
              f'{model["v4_helpful"]["helpful_exact_chosen_saving"]:+.6f}.',
              f'- FULL-v5: helpful argmax {model["v5_helpful"]["exact_argmax_matches"]}/'
              f'{model["v5_helpful"]["helpful_states"]}, all12 exact chosen saving '
              f'{model["v5_all"]["exact_chosen_saving"]:+.6f}; helpful chosen saving '
              f'{model["v5_helpful"]["helpful_exact_chosen_saving"]:+.6f}.',
              '- Same12 exact states/K8/H300; complete comparison and identities in `wmplan/MODEL_COMPARISON.json`.','',
              '## Comparisons',
              '- S-DISTILL uses these identical24 worlds and MAPPO baseline bytes. Its published savings were '
              'J600 -96.482 [-575.232,+288.473], J900 -54.565 [-294.222,+188.886].']
    for arm in s['arms']:
        d = s['paired_vs_SUMO_teacher_student'][arm]
        lines += [f'- {arm} minus SUMO-teacher-student paired saving: J600 {interval(d["600"])}; '
                  f'J900 {interval(d["900"])}. Positive favours the online planner.']
    if s['paired_vs_WM_teacher_student']:
        for arm in s['arms']:
            d = s['paired_vs_WM_teacher_student'][arm]
            lines += [f'- {arm} minus WM-teacher-student paired saving: J600 {interval(d["600"])}; J900 {interval(d["900"])}.']
    else:
        lines += ['- No completed same-panel WM-teacher-student result was available at this report freeze.']
    lines += ['- The SUMO-oracle S-COOPPLAN H300 confirmation used18 DIFFERENT dev worlds: '
              'J600 +767.979 [157.470,1606.905], J900 +821.068 [-944.174,2576.533]. '
              'This is descriptive context only, not a paired oracle/WM difference. No same-panel oracle rerun was authorized.','',
              '## Integrity, resources and limitations',
              f'- All{24*len(s["arms"])} episodes/{288*len(s["arms"])} decisions and {s["executed_groups"]} executed-group quartets complete; '
              f'{s["exact_credit_prefixes"]} exact branch prefix gates pass. Independent QA recomputes forecasts, actions, quotas and C_G.',
              '- '+accounting_text+' No new MAPPO or mechanics episode; every replay step is charged. '
              'One attempt per episode in the approved restart; no further retries/refunds/transfers.',
              '- Fixed2950 credit allowance per world-arm; full-prefix H120 quartet4*(t+120). '
              'At most one group can fit, none aftert600. Refusals above quantify this material restriction.',
              '- Same inherited O0..O4/executors/safety; K8 paired futures/H300/cadence60 at t300..960; '
              'MAPPO-only tail to1200. No live-result-based tuning or model change.',
              '- Exact validated S-WMTEACH roster-only path: sampled frozen FIT driver/source priors, '
              'cold motion/controller memory, no observed queue/history claim, no hidden truth inputs. '
              'K8 empirical strict probabilities are not asserted calibrated.',
              '- Fresh DEV984000..984023/two per12cells, panelSHA72776ff1; these are development results, not final generalization evidence.',
              '- Dual-GPU inference: gpuA plus gpuR `/root/wmplan20261101_v1`; immutable gpuA '
              + ('`/root/autodl-tmp/wmplan20261101_v2`' if restart else '`/root/autodl-tmp/wmplan20261101_v1`')
              + ' ledger; model/baseline/source/design hashes, charges, closed jobs, forecasts, complete twins and QA in '
              + ('`wmplan/restart_v2/`.' if restart else '`wmplan/`.'),
              '- Reproduce: `py -3.13 -B -m '+('tools.wmplan.continuation' if restart else 'tools.wmplan.report')
              +' --verify'+(' --primary' if primary else '')+'`. MAIN owns master-ledger/route integration.']
    report = LOCAL/'docs/paper_results/WMPLAN.md'
    with report.open('w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    secondary = ('SECONDARYstrict pending; ' if primary else
                 f'SECONDARYstrict J600{interval(s["arms"]["strict"]["savings"]["600"]["team_delay_vs"])} '
                 f'J900{interval(s["arms"]["strict"]["savings"]["900"]["team_delay_vs"])}; ')
    line = (f'- S-WMPLAN {"PRIMARY" if primary else "LIVE"} COMPLETE: {model["selected"]}/same24DEV984000..984023/reusedMAPPO0newbaseline; '
            f'PRIMARYexpected J600{interval(expected["savings"]["600"]["team_delay_vs"])} '
            f'J900{interval(expected["savings"]["900"]["team_delay_vs"])}veh-s/world; '
            +secondary+
            f'all{24*len(s["arms"])}episodes/{288*len(s["arms"])}decisions/{s["executed_groups"]}completeCGquartets/'
            f'{s["exact_credit_prefixes"]}BITWISEprefixes; '+accounting_text+' '
            'Budget-limited2950/world-arm<=1earlygroup/noaftert600; proposals/refusals/safety/studentcomparisons/'
            'modelnoninferiority documented. No unlimited-planner or trained-policy headline implied. '
            f'Report E:/agent-work/c4-wmplan/docs/paper_results/WMPLAN.md; QAsha{sha(OUT/(prefix+"QA.json"))}; '
            'MAIN master-ledger integration pending.\n')
    text = REG.read_text(encoding='utf-8')
    if line not in text:
        with REG.open('a',encoding='utf-8') as f:
            f.write(line);f.flush()
    save(prefix+'RESULT_REGISTRATION.json',dict(line=line,report_sha256=sha(report)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--verify',action='store_true')
    p.add_argument('--publish',action='store_true')
    p.add_argument('--primary',action='store_true')
    a = p.parse_args()
    if a.publish:
        publish(primary=a.primary)
    else:
        run(a.verify,primary=a.primary)
