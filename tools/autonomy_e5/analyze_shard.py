"""S-E5DONE read-only scientific validation and frozen observable-tape detection.

Run detached on each CPU host. No simulator, training, truth or assignment imports.
Only ANALYSIS_* receipts and per-job detector caches under operations_e5 are written.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import traceback

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')

R = Path('/home/ubuntu/e3long')
OP = R/'operations_e5'
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)
SOURCE = 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'
ON = '64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'
OFF = 'e1fa4881bbc5c89d05298e979d98b23284308331ad1bbb12f9608fa67e105675'
TH = '9119cd77d27889c9546cfaacd91af5a3f7f89beda3a5b6616e988aa67860701f'
FAILED = {973001, 973003, 973005, 973007, 973009, 973011}
COMPLETION = {973217, 973219, 973221, 973223, 973225, 973227}
INPUT_HASHES = {}


def safe(path):
    p = Path(path)
    assert not BAD.search(str(p)) and not BAD.search(str(p.resolve())), p
    assert p.resolve().is_relative_to(R), p
    return p


def read(path):
    p = safe(path)
    data = p.read_bytes()
    INPUT_HASHES[str(p.relative_to(R))] = hashlib.sha256(data).hexdigest()
    return data


def doc(path):
    return json.loads(read(path))


def digest(path):
    return hashlib.sha256(read(path)).hexdigest()


def write(path, value):
    p = safe(path)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(p)


def exited(name):
    receipt = doc(OP/name)
    proc = Path('/proc')/str(receipt['pid'])
    if proc.exists():
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        assert stat[19] != receipt['start_ticks'] or stat[0] == 'Z', (name, 'still running')
    return receipt


def validate(shard):
    predecessors = [exited('MAIN_PID.json')]
    if shard:
        predecessors += [exited('FIX_REMAINDER_PID.json'), exited('DONE_PID.json')]
        assert doc(OP/'DONE_STATUS.json')['status'] == 'COMPLETE'
        authority = doc(OP/'DONE_AUTHORITY.json')
        assert set(authority['seeds']) == COMPLETION and authority['additional_sim_s'] == 5400
        assert authority['cpuQ_cap_sim_s'] == 108000 and authority['E5_cap_sim_s'] == 210600
        assert digest(OP/'E5DONE_AMENDMENT.md') == authority['amendment_sha256']
        assert digest(OP/'done_runner.py') == authority['done_runner_sha256']
    manifest_path = R/'runtime/SOURCE_MANIFEST.json'
    assert digest(manifest_path) == SOURCE
    manifest = doc(manifest_path)
    files = manifest.get('files', manifest)
    assert len(files) == 543
    for name, sha in files.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert digest(R/'runtime'/name) == sha, name
    detector = R/'runtime/traffic_shepherd/c4/coopbehav.py'
    spec = importlib.util.spec_from_file_location('e5_frozen_coopbehav', detector)
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    thresholds = doc(R/'inputs/thresholds.json')
    assert digest(R/'inputs/thresholds.json') == TH
    for folder, policy in (('cycle1', ON), ('coopoff_cycle1', OFF)):
        inp = doc(R/'inputs'/folder/'INPUTS.json')
        assert digest(inp['policy']) == policy
        gate = doc(R/'inputs'/folder/'native_gate_result.json')
        assert gate['passed'] and gate['checkpoint_sha256'] == policy
    results, charges, rows = {}, {}, []
    residual = 0.0
    for phase, labels in (('baseline', {'native': 'native', 'mappo': 'mappo'}),
                          ('policy', {'ours': 'coopON'}), ('coopoff', {'ours': 'coopOFF'})):
        out = R/'main'/('e3long_'+phase)/('shard'+str(shard))
        plan = doc(out/'PLAN.json')
        assert plan['source_manifest_sha256'] == SOURCE
        if phase != 'baseline':
            assert plan['input_hashes']['policy'] == (ON if phase == 'policy' else OFF)
            assert plan['input_hashes']['thresholds'] == TH
            inp = doc(R/'inputs'/('cycle1' if phase == 'policy' else 'coopoff_cycle1')/'INPUTS.json')
            for key, sha in plan['input_hashes'].items():
                p = R/'inputs'/('cycle1' if phase == 'policy' else 'coopoff_cycle1')/'native_gate_result.json' if key == 'native_gate' else inp[key]
                assert digest(p) == sha, key
        else:
            assert digest(R/'inputs/mappo_v5_final.pt') == plan['input_hashes']['mappo']
        ledger = [json.loads(s) for s in read(out/'budget.jsonl').decode().splitlines()]
        assert len(ledger) == len({x['id'] for x in ledger})
        assert all(x['sim_s'] == 900 for x in ledger)
        cap = 205200 if phase == 'baseline' else 108000 if phase == 'coopoff' and shard else 102600
        assert sum(x['sim_s'] for x in ledger) == cap
        charges[phase] = dict(charged_sim_s=cap, n_attempts=len(ledger), actual_sim_s=0)
        all_rows = doc(out/'RESULTS.json')
        assert len(all_rows) == 114*len(labels)
        expected = {(s, a) for s in range(973000+shard, 973228, 2) for a in labels}
        assert {(x['job']['seed'], x['job']['arm']) for x in all_rows} == expected
        for row in all_rows:
            job = row['job']
            seed, arm = job['seed'], labels[job['arm']]
            assert row['ok'] and row['sim_s'] == 900
            assert job['cell'] == (seed-973000) % 12 and job['repeat'] == (seed-973000)//12
            cell = job['cell']
            assert job['penetration'] == (.05, .1, .2, .4)[cell//3]
            assert job['demand'] == (1900, 2400, 2700)[cell % 3]
            assert job['shard'] == shard and job['warmup'] == 300 and job['horizon'] == 600
            retry = phase == 'coopoff' and seed in FAILED
            aid = job['id']+'.e5fix1' if retry else job['id']
            assert sum(x['id'] == aid for x in ledger) == 1
            path = out/('retry_jobs' if retry else 'jobs')/job['id']
            assert doc(path/'result.json') == row
            if retry:
                failure = doc(out/'jobs'/job['id']/'failure.json')
                assert 'SUMO binary not found on PATH' in failure['traceback']
                assert sum(x['id'] == job['id'] for x in ledger) == 1
            if phase == 'coopoff' and seed in COMPLETION:
                assert next(x for x in ledger if x['id'] == aid)['task'] == 'S-E5DONE'
            m = row['metrics']
            assert not m['dropped_vehicle_steps'] and set(m['windows']) == {'120', '300', '600'}
            for h in (120, 300, 600):
                w = m['windows'][str(h)]
                assert w['t0_s'] == 300 and w['t_end_s'] == 300+h
                assert w['initial_state_sha256'] == m['initial_state_sha256']
                assert not w['dropped_vehicle_steps']
                assert doc(path/f'cumulative_{h}.json') == w
                for terms in (('road_delay_vs', 'source_wait_vs'), ('mainline_delay_vs', 'ramp_delay_vs')):
                    err = abs(w['team_delay_vs']-sum(w[k] for k in terms))
                    residual = max(residual, err)
                    assert err < 1e-3
                assert abs(w['source_wait_vs']-w['source_wait_per_vehicle_sum_vs']) < 1e-5
                assert w['requested'] == sum(g['requested'] for g in w['groups'].values())
                assert abs(w['throughput_vph']-w['arrivals']*3600/h) < 1e-7
            charges[phase]['actual_sim_s'] += row['sim_s']
            assert (seed, arm) not in results
            results[seed, arm] = m
            rows.append(dict(seed=seed, arm=arm, cell=cell, share=job['penetration'], demand=job['demand'],
                             initial_state_sha256=m['initial_state_sha256'], windows=m['windows'], path=str(path)))
    for seed in range(973000+shard, 973228, 2):
        assert len({results[seed, a]['initial_state_sha256'] for a in ('coopON', 'coopOFF', 'native', 'mappo')}) == 1
    return cb, thresholds, rows, dict(status='PASS', shard=shard, worlds=114, arm_worlds=456,
        source_files_verified=543, source_manifest_sha256=SOURCE, detector_sha256=digest(detector),
        thresholds_sha256=TH, charges=charges, max_conservation_residual_vs=residual,
        predecessors_exited=predecessors, no_new_simulation=True)


def run(shard):
    os.nice(19)
    cb, thresholds, rows, proof = validate(shard)
    write(OP/'ANALYSIS_VALIDATION.json', proof)
    cache = safe(OP/'analysis_detector')
    cache.mkdir(exist_ok=True)
    behaviour = []
    for i, row in enumerate(sorted(rows, key=lambda r: (r['seed'], r['arm']))):
        # All data are observable kinematics and the controller's own CAV identities.
        path = Path(row.pop('path'))/'traj_base.npz'
        sha = digest(path)
        output = cache/f"s{row['seed']}_{row['arm']}.json"
        if output.exists():
            saved = doc(output)
            assert saved['trajectory_sha256'] == sha and saved['detector_sha256'] == proof['detector_sha256']
            assert saved['thresholds_sha256'] == TH
        else:
            frame = cb.load_traj(safe(path))
            assert frame.t.min() == 301 and frame.t.max() == 900
            assert not frame.duplicated(['t', 'vid']).any()
            f = cb.features(frame)
            windows = {}
            for h in (120, 300, 600):
                # Prefix before detection: later samples cannot extend/classify a prefix event.
                det = cb.detect(f[f.t < 300+h], thresholds, (300, 300+h))
                counts = {name: sum(e['type'] == name for e in det['events'])
                          for name in ('multi_slow', 'metering', 'vacate')}
                counts['side_by_side'] = sum(e['type'] == 'multi_slow' and e['subtype'] == 'side_by_side' for e in det['events'])
                counts['vacate_ramp_present'] = sum(e['type'] == 'vacate' and e['subtype'] == 'ramp_present' for e in det['events'])
                windows[str(h)] = dict(counts=counts, exposure_cav_h=det['exposure_cav_s']/3600)
            saved = dict(seed=row['seed'], arm=row['arm'], cell=row['cell'], share=row['share'],
                         demand=row['demand'], trajectory_sha256=sha, detector_sha256=proof['detector_sha256'],
                         thresholds_sha256=TH, windows=windows)
            write(output, saved)
        behaviour.append(saved)
        write(OP/'ANALYSIS_STATUS.json', dict(status='RUNNING', completed=i+1, total=456, t=time.time()))
        if (i+1) % 12 == 0:
            print(json.dumps(dict(completed=i+1, total=456)), flush=True)
    result = dict(proof=proof, rows=rows, behaviour=behaviour, input_sha256=INPUT_HASHES,
                  analysis_script_sha256=digest(Path(__file__)))
    write(OP/'ANALYSIS_DATA.json', result)
    write(OP/'ANALYSIS_STATUS.json', dict(status='COMPLETE', completed=456, total=456, t=time.time(),
                                        data_sha256=digest(OP/'ANALYSIS_DATA.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=int, choices=(0, 1), required=True)
    args = p.parse_args()
    try:
        run(args.shard)
    except BaseException:
        write(OP/'ANALYSIS_STATUS.json', dict(status='FAILED', traceback=traceback.format_exc(), t=time.time()))
        raise
