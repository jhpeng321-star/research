"""Read-only episode reconciliation and exclusive final export; never runs SUMO."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OWN = Path('/home/ubuntu/sumoes20261031')
sys.path.insert(0, str(OWN / 'pytools'))

from tools.sumoes.remote.run import stop_reason  # noqa: E402
from tools.sumoes.sumo_eval import theta_sha  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(path, payload):
    with path.open('x') as handle:
        json.dump(payload, handle, indent=1, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def main():
    payload = json.loads((OWN / 'state.json').read_text())
    state, receipts = payload['state'], payload['receipts']
    reason = stop_reason(receipts)
    assert reason and state['update_count'] == 4 and state['accepted_count'] == 0
    assert len(state['theta']) == 137 and all(v == 0. for v in state['theta'])
    implementation = json.loads((OWN / 'MAIN1440_IMPLEMENTATION.json').read_text())
    hashes = implementation['source_sha256']
    assert all(sha(OWN / 'pytools' / name) == value for name, value in hashes.items())
    assert hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() == (
        implementation['bundle_sha256'])
    ledger = [json.loads(s) for s in (OWN / 'accounting.jsonl').read_text().splitlines() if s]
    assert len(ledger) == len({r['id'] for r in ledger}) == 53
    charged = sum(r['sim_s'] for r in ledger)
    assert charged == payload['budget']['charged_sim_s'] == 47700 <= 650000
    actual, worlds, results = 0, {}, []
    safety = dict(collisions=0, teleports=0, dropped_vehicle_steps=0, lane_calls=0)
    for row in ledger:
        result_path = OWN / 'jobs' / row['id'] / 'result.json'
        r = json.loads(result_path.read_text())
        assert r['ok'] is True and r['actual_sim_s'] == row['sim_s'] == 900
        assert r['job']['id'] == row['id']
        actual += r['actual_sim_s']
        metrics, seed = r['metrics'], r['job']['seed']
        for key in ('collisions', 'teleports', 'dropped_vehicle_steps'):
            safety[key] += metrics[key]
        safety['lane_calls'] += sum(r['lane_audit'].values())
        digest = metrics['initial_state_sha256']
        assert worlds.setdefault(seed, digest) == digest == payload['world_hash'][str(seed)]
        results.append(dict(job=row['id'], seed=seed, J600=r['J600'],
                            initial_state_sha256=digest, result_sha256=sha(result_path)))
    assert actual == payload['budget']['actual_sim_s'] == charged
    assert not any(safety.values())
    canary = receipts[0]['canary']
    assert canary['identical'] and canary['ctrl0']['J600'] == canary['ours0']['J600']
    assert canary['ctrl0']['initial_state_sha256'] == canary['ours0']['initial_state_sha256']
    curve, screens = [], []
    for r in receipts[1:]:
        u = r['update']
        curve.append(dict(update=u, accepted=r['accepted'], no_step=r.get('no_step', False),
                          panel_before=r['panel_mean_before'], panel_candidate=r['panel_mean_after'],
                          panel_retained=sum(state['panel_j']) / 8,
                          pair_deltas=r['pair_deltas'], sigma=r['sigma'],
                          charged=r['budget']['charged_sim_s'], actual=r['budget']['actual_sim_s'],
                          driver_wall_s=r.get('wall_s')))
        if u == 0:
            continue
        req = OWN / 'screens' / f'request_u{u:02d}.json'
        screen_path = OWN / 'screens' / f'screen_u{u:02d}.json'
        screen = json.loads(screen_path.read_text())
        assert screen['request_sha256'] == sha(req)
        assert screen['schema'] == 'sumoes_screen/2' and screen['amendment'] == 'MAIN1440'
        assert screen['kept'] == [] and len(screen['decision_rows']) == 8
        assert all(not row['confident'] and row['median'] < 0 for row in screen['decision_rows'])
        assert screen['rollouts'] == 136 and screen['steps'] == 120
        screens.append(dict(update=u, medians=[x['median'] for x in screen['decision_rows']],
                            q10=[x['q10'] for x in screen['decision_rows']],
                            kept=screen['kept'], wall_s=screen['wall_s'],
                            request_sha256=sha(req), screen_sha256=sha(screen_path)))
    assert not (OWN / 'screens/request_u04.json').exists()
    assert not (OWN / 'STOP').exists(), 'unexpected mechanical STOP'
    # Check owned Python jobs only. No process is signalled.
    inventory = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o',
                                         'pid=,comm=,args='], text=True)
    live = [line for line in inventory.splitlines() if len(line.split(None, 2)) == 3
            and line.split(None, 2)[1].startswith('python')
            and any('/sumoes/remote/' + name in line
                    for name in ('run.py', 'wm_screen.py', 'worker.py'))]
    assert not live, live
    head = dict(schema='sumoes_final_head/1', theta=state['theta'],
                sha256=theta_sha(state['theta']), dimension=137, updates_consumed=4,
                accepted_updates=0, equivalent_to='CTRL0 (zero residual)',
                trunk_sha256='64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c',
                law='MAIN1440 capped-CTRL0-relative; zero/no-effect retains native-base write',
                implementation_bundle_sha256=implementation['bundle_sha256'])
    export(OWN / 'FINAL_HEAD.json', head)
    final = dict(schema='sumoes_final_reconciliation/1', reason=reason,
                 budget=dict(cap=650000, charged=charged, actual=actual, unspent=650000-charged),
                 completed_episodes=len(results), distinct_sumo_worlds=len(worlds),
                 safety=safety, final_head_sha256=head['sha256'],
                 final_head_file_sha256=sha(OWN / 'FINAL_HEAD.json'),
                 state_file_sha256=sha(OWN / 'state.json'), ledger_sha256=sha(OWN / 'accounting.jsonl'),
                 learning_curve=curve, amended_screens=screens, episode_results=results,
                 live_task_python_jobs=live,
                 caution='Updates1-3 are screen-only; corrected-law SUMO training step never dispatched. '
                         'Update0 used historical base-relative law; no learned gain demonstrated. '
                         'Panel values after screen-only updates are cached, not new SUMO measurements.')
    export(OWN / 'FINAL_RECONCILIATION.json', final)
    print(json.dumps({k: v for k, v in final.items() if k not in (
        'episode_results', 'learning_curve', 'amended_screens')}, indent=1))


if __name__ == '__main__':
    main()
