import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pytest

from tools.rb2int import interim, interim_watch, resume_ci
from tools.rbint.local_resources import child_incarnation
from tools.rb2int.uncertainty import PairedWorldBootstrap
from tools.rb2int.credibility import ratio_bands
from tools.rb2int.gate_bootstrap import GateBootstrap
from tools.rbeval.core import ARMS, HS
from tools.rbeval.core import digest


def test_only_scored_replicated_cells_or_complete_reserved_are_ready():
    registration = dict(replicated_primary_worlds=[1,2,3,4,5],reserved8=[6,7])
    meta = {s:dict(demand=10 if s<3 else 20,penetration=.2) for s in range(1,8)}
    assert interim.scopes([1,3,6],registration,meta) == {}
    assert interim.scopes([1,2,3,6],registration,meta) == {'available_replicated_cells':[1,2]}
    assert interim.scopes([1,2,3,4,6],registration,meta) == {'available_replicated_cells':[1,2,3,4]}
    ready = interim.scopes(range(1,8),registration,meta)
    assert ready == dict(reserved8=[6,7],available_replicated_cells=[1,2,3,4,5])


def test_constant_nonbinary_bootstrap_support_never_prints_precision():
    metadata = [dict(seed=s,demand=1900,penetration=.05) for s in (1,2)]
    boot = PairedWorldBootstrap(metadata)
    # Multiple columns trigger NumPy's axis-reduction accumulated-mean error;
    # a single column often does not reproduce this failure.
    rows = boot.bands(np.full((2,6),.4))
    assert all(r['low'] is None and r['unavailable_reason']=='zero_estimable_variance' for r in rows)
    features = [([dict(seed=s,value=.4) for s in (1,2)],'value')]*6
    assert all(r['low'] is None for r in ratio_bands(boot,features))
    gate = GateBootstrap(metadata)
    assert all(r['low'] is None for r in gate.bands([[(1,.4),(2,.4)]]*6))


def test_constant_large_cost_support_is_guarded_relative_to_numerical_scale():
    metadata = [dict(seed=s,demand=1900,penetration=.05) for s in (1,2)]
    boot = PairedWorldBootstrap(metadata)
    assert boot.bands(np.full((2,1),1e8+.4))[0]['low'] is None


@pytest.mark.skipif(os.name!='nt',reason='Windows creation-time receipt binding')
def test_recovery_checks_owned_pid_creation_time_and_exit_without_signals():
    child = subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(1)'])
    record = dict(pid=child.pid,**child_incarnation(child))
    assert resume_ci.alive(record)
    assert not resume_ci.alive(dict(record,create_filetime=record['create_filetime']+1))
    child.wait(timeout=10)
    assert not resume_ci.alive(record)


def test_short_prefix_is_exact_cumulative_reported_cost_not_horizon_interpolation():
    bins = np.arange(100,dtype=float).reshape(5,20)
    costs = {'time_bins_30s':{'real_J':bins.tolist()}}
    costs.update({str(h):{'real_J':bins[:,:h//30].sum(1).tolist()} for h in HS})
    task = dict(costs=costs)
    assert np.array_equal(interim.prefix_cost(task,30),bins[:,0])
    assert np.array_equal(interim.prefix_cost(task,60),bins[:,:2].sum(1))
    task['costs']['120']['real_J'][0] += 1
    with pytest.raises(ValueError,match='differ'):
        interim.prefix_cost(task,60)


def test_unknown_source_wait_never_becomes_an_exact_team_sign():
    assert interim.bounded_direction(-1,-5,-2) == 1
    assert interim.bounded_direction(1,-5,-2) == 0
    assert interim.bounded_direction(1,-5,2) is None
    assert interim.bounded_direction(0,0,0) == 1
    assert interim.bounded_direction(1,0,0) == 0
    with pytest.raises(ValueError):
        interim.bounded_direction(1,2,-2)


def test_short_decision_quality_scores_saved_total_and_road_costs_separately(tmp_path):
    worlds = [969432,969433]
    targets, audits = [], {m:dict(inputs=[]) for m in interim.MODELS}
    meta = {s:dict(seed=s,demand=2400,penetration=.2) for s in worlds}
    for seed in worlds:
        for j,arm in enumerate(ARMS):
            road = {str(h):float(j*h/30) for h in (30,60,90)}
            targets.append(dict(seed=seed,arm=arm,road_delay_vs=road,
                team_delay_bounds_vs={h:[v,v] for h,v in road.items()}))
        for model in interim.MODELS:
            for k in range(8):
                bins = np.repeat(np.arange(5,dtype=float)[:,None],20,axis=1)
                if model=='FULL-v4':
                    bins = bins[::-1]
                costs = {'time_bins_30s':{key:bins.tolist() for key in ('real_J','real_road')}}
                costs.update({str(h):{key:bins[:,:h//30].sum(1).tolist() for key in ('real_J','real_road')} for h in HS})
                path = tmp_path/f'{model}_{seed}_{k}.json'
                path.write_text(json.dumps(dict(seed=seed,future=k,costs=costs)))
                audits[model]['inputs'].append(dict(task=str(path),sha256=digest(path)))
    result = interim.short_quality(audits,dict(episodes=targets),worlds,meta)
    for row in result['summary']:
        if row['metric']=='resolved':
            assert row['mean']==1
        else:
            assert row['mean']==float(row['model']=='FULL-v5')
    assert len(result['per_pair']) == 2*3*2*10


def test_report_block_replaces_only_its_owned_section():
    text = 'prior owner text\n'
    once = interim_watch.replace_block(text,'first\n')
    twice = interim_watch.replace_block(once,'second\n')
    assert twice.startswith(text)
    assert 'first' not in twice and 'second' in twice
    assert twice.count(interim_watch.BEGIN) == 1


def test_hold_handover_requires_exact_stop_and_prior_exit(tmp_path, monkeypatch):
    old = tmp_path/'runs/rb2int_interim_v3'
    old.mkdir(parents=True)
    (old/'PROCESS.json').write_text(json.dumps(dict(pid=44988,create_filetime=134379973214244749)))
    monkeypatch.setattr(interim_watch,'ROOT',tmp_path)
    monkeypatch.setattr(resume_ci,'alive',lambda record:True)
    assert not interim_watch.hold_ready()
    (old/'STOP.json').write_text(json.dumps(dict(error="ValueError('interim source/target changed; explicit recovery required')")))
    assert not interim_watch.hold_ready()
    monkeypatch.setattr(resume_ci,'alive',lambda record:False)
    assert interim_watch.hold_ready()
    (old/'STOP.json').write_text(json.dumps(dict(error='unrelated')))
    with pytest.raises(ValueError,match='unexpected'):
        interim_watch.hold_ready()


def test_hold_drain_waits_for_every_original_child_in_verified_cache(tmp_path, monkeypatch):
    evidence = tmp_path/'docs/rb2int'
    evidence.mkdir(parents=True)
    for tag, job in (('GPUA','v5_1'),('GPUR','wm2_2')):
        (evidence/f'OWNER_1755_{tag}_HOLD.json').write_text(json.dumps(dict(
            new_gpu_admissions=0,automatic_resume=False,evaluate_children=[dict(job=job)])))
    cache = tmp_path/'cache'
    monkeypatch.setattr(interim_watch,'ROOT',tmp_path)
    monkeypatch.setattr(interim_watch,'CACHE',cache)
    assert not interim_watch.hold_state()
    for host, job in (('gpuA','v5_1'),('gpuR','wm2_2')):
        folder = cache/host/job
        folder.mkdir(parents=True)
        (folder/'COLLECTED.json').write_text('{}')
    assert interim_watch.hold_state()


def test_hold_intro_removes_stale_two_world_claim_but_preserves_details():
    result = dict(worlds=[1,2,3,4],scope='replicated',
                  delay_crps_difference=[dict(mean=v) for v in (1,2,3)])
    text = 'stale two-world prose\n## What exists\nmodel bindings\n'+interim_watch.BEGIN+'\nevidence'
    changed = interim_watch.hold_intro(text,result,'now',25,False)
    assert 'stale' not in changed
    assert '4 matched complete K8' in changed
    assert 'ZERO new evaluation admissions' in changed
    assert '## What exists\nmodel bindings\n'+interim_watch.BEGIN+'\nevidence' in changed


def test_publication_waits_for_known_collector_staging_to_clear(tmp_path,monkeypatch):
    from tools.rb2int import finish
    monkeypatch.setattr(interim_watch,'COLLECTOR',tmp_path)
    stamp = datetime.fromtimestamp(interim_watch.time.time()-60).astimezone().strftime('%Y%m%dT%H%M%S%z')
    (tmp_path/'watch.log').write_text('COLLECT '+stamp+' 12 /216 complete world jobs\n')
    staged = iter(['docs/rb2int/COLLECT_owned.json\n',''])
    monkeypatch.setattr(finish,'command',lambda args:next(staged))
    waits = []
    monkeypatch.setattr(interim_watch.time,'sleep',lambda seconds:waits.append(seconds))
    monkeypatch.setattr(interim_watch,'git_commit',lambda paths,message:'committed-once')
    assert interim_watch.commit_when_quiet([], 'test') == 'committed-once'
    assert waits == [2]


def test_publication_never_waits_through_foreign_source_staging(tmp_path,monkeypatch):
    from tools.rb2int import finish
    monkeypatch.setattr(interim_watch,'COLLECTOR',tmp_path)
    monkeypatch.setattr(finish,'command',lambda args:'src/user_change.py\n')
    with pytest.raises(ValueError,match='foreign non-collector'):
        interim_watch.commit_when_quiet([], 'test')


def test_takeover_requires_exact_expected_stop_and_unchanged_original_sources(tmp_path,monkeypatch):
    monkeypatch.setattr(interim_watch,'COLLECTOR',tmp_path)
    assert not interim_watch.final_guard_ready()
    (tmp_path/'STOP.json').write_text(json.dumps(dict(error="ValueError('unexpected failure')")))
    with pytest.raises(ValueError,match='unexpected'):
        interim_watch.final_guard_ready()
    (tmp_path/'STOP.json').write_text(json.dumps(dict(error="ValueError('progress report changed during collection; preserve edits and assemble explicitly')")))
    (tmp_path/'START_ONCE.json').write_text(json.dumps(dict(source_hashes={})))
    assert interim_watch.final_guard_ready()
    source = tmp_path/'original.py';source.write_text('changed')
    (tmp_path/'START_ONCE.json').write_text(json.dumps(dict(source_hashes={str(source):'incorrect'})))
    with pytest.raises(ValueError,match='sources changed'):
        interim_watch.final_guard_ready()


def test_two_model_analysis_keeps_forecasts_and_factual_worlds_paired(tmp_path,monkeypatch):
    worlds = [969432,969433]
    real, pred = [], {m:{} for m in interim.MODELS}
    for i,seed in enumerate(worlds):
        for j,arm in enumerate(ARMS):
            target = dict(seed=seed,arm=arm,demand=2400,penetration=.2,
                clocks={str(h):dict(team_delay_vs=float(j+i)) for h in HS},
                bins=dict(upstream=dict(events={str(h):dict(event=bool(i)) for h in HS})))
            real.append(target)
            for m in interim.MODELS:
                pred[m][seed,arm] = [dict(clocks={str(h):dict(team_delay_vs=float(j+i+(2+i if m=='FULL-v4' else 1))) for h in HS},
                    bins=dict(upstream=dict(events={str(h):dict(event=bool(i)) for h in HS}))) for k in range(8)]
    real_path = tmp_path/'real.json';real_path.write_text(json.dumps(dict(episodes=real)))
    factual = tmp_path/'factual.json';factual.write_text(json.dumps(dict(episodes=[])))
    short = tmp_path/'short.json';short.write_text('{}')
    runs = tmp_path/'runs.json';runs.write_text('{}')
    monkeypatch.setattr(interim,'read_model',lambda runs,model,real,worlds:(pred[model],dict(inputs=[])))
    monkeypatch.setattr(interim,'world_errors',lambda *args:[dict(seed=s,median_difference_m=-i-1) for i,s in enumerate(worlds)])
    monkeypatch.setattr(interim,'observations',lambda *args:{})
    monkeypatch.setattr(interim,'summarize',lambda *args:([],[]))
    monkeypatch.setattr(interim,'short_quality',lambda *args:{})
    result = interim.analyze(runs,real_path,factual,short,worlds,'test',tmp_path/'result.json')
    assert result['replicated']
    assert [r['mean'] for r in result['delay_crps_difference']] == [-1.5]*3
    assert result['hdv60_difference']['mean'] == -1.5
    assert all(r['mean']==0 for r in result['breakdown_rate_distance_difference'])
    assert all(r['mean']==1 for r in result['direction_accuracy'])
    assert result['switching_eligibility'].startswith('NOT_ESTABLISHED')
    assert len(result['controller_shifts']) == 3*3*8
