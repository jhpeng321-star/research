import pytest


def test_world_cluster_bootstrap_reselects_fixed_and_keeps_world_weights():
    pytest.importorskip('numpy')
    from tools.c1switch.analyze import primary,rank
    # Controller0/1 trade wins across worlds; correct bootstrap reselects which
    # fixed controller is best, instead of leaking the full-panel identity.
    result = primary([[0,100,200,200,200],[100,0,200,200,200]],
                     [[2,1],[4,1]],[[0,10,20,20,20],[10,0,20,20,20]])
    assert result['wm_minus_bb_regret60']['mean'] == 2
    assert result['wm_minus_best_fixed_regret60']['mean'] == 5
    assert result['wm_minus_best_fixed_regret60']['ci95'] == [0,5]
    assert rank([5,1,1,10]) == [3.,1.5,1.5,4.]


def test_recovery_full_panel_accounts_abandoned_and_random_subset(tmp_path):
    from tools.c1switch.analyze import summarize
    from tools.c1switch.design import CONTROLLERS, measurement_indices, worlds
    from tools.c1switch.episode import ARMS
    from tools.c1switch.storage import write
    import json
    budget=dict(wm=178380,bb=184440,random=5400,**dict.fromkeys(CONTROLLERS,14400))
    write(tmp_path/'BINDING.json',dict(total_sim_s=499680,authorization='MAIN23:25',
        retained_superseded_sim_s=12450,retained_abandoned_sim_s=17820,per_arm=budget))
    ledger=[dict(id='retained_step0',sim_s=47010),dict(id='superseded_wm',sim_s=12450)]
    ledger += [dict(id=a+'_all',sim_s=b) for a,b in budget.items()]
    (tmp_path/'accounting.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger))
    for a in ARMS:
        for w in worlds('main'):
            if a=='random' and w['seed']>=969118:continue
            decisions=[]
            for j in range(10):
                prefix=dict(seed=w['seed'],j=j)
                d=dict(j=j,prefix=prefix,pick='pi' if a=='wm' else 'native',
                    predicted_costs={c:10-i for i,c in enumerate(CONTROLLERS)})
                if a in ('wm','bb') and j in measurement_indices(w['seed']):
                    d['branches']={c:dict(prefix=prefix,**{'60':dict(team_delay_vs=10-i),'90':dict(team_delay_vs=20-i)}) for i,c in enumerate(CONTROLLERS)}
                    for c in CONTROLLERS:
                        write(tmp_path/'jobs'/f'{a}_s{w["seed"]}_j{j}_{c}'/'CLOSE.json',dict(actual_sim_s=390+60*j,charged_sim_s=390+60*j))
                decisions.append(d)
            value={'wm':90,'bb':95,'random':110}.get(a,100+100*CONTROLLERS.index(a) if a in CONTROLLERS else 0)
            p=tmp_path/'jobs'/f'{a}_s{w["seed"]}_episode'
            write(p/'RESULT.json',dict(world=w,decisions=decisions,switching_frequency=1,
                J600=dict(team_delay_vs=value,collisions=0,teleports=0,ttc_below_3_vs=0,hard_brake=0)))
            write(p/'CLOSE.json',dict(actual_sim_s=900,charged_sim_s=900))
    result=summarize(tmp_path)
    assert result['episode_n']['random']==6 and result['episode_n']['wm']==16
    assert result['primary']['wm_minus_bb_regret60']['mean']==-3
    assert result['primary']['wm_minus_best_fixed_regret60']['mean']==-3
    assert result['secondary_episode_paired']['wm']['random']['worlds']==6
    assert result['secondary_episode_paired']['wm']['random']['mean']==-20
    assert all(c['J600']['random'] is None for c in result['cells'] if c['penetration']>=.2)
    # A correctly charged but incomplete scientific panel must never publish results.
    p=tmp_path/'jobs/wm_s969127_episode/RESULT.json'
    p.rename(p.with_name('INCOMPLETE.json'))
    with pytest.raises(FileNotFoundError):summarize(tmp_path)


def test_sparse_cell_intervals_preserve_absent_random_and_disclose_n1():
    from tools.c1switch.analyze import cell_interval
    assert cell_interval([float('nan')]) is None
    assert cell_interval([10])==[10,10]
    assert cell_interval([0,100])==[0,100]
