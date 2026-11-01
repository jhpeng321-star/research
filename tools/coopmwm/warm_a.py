"""A-only adoption of already charged, never-attempted reservations after22:08."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time
import torch
from tools.coopmwm.accel_runner import timestamp, wave, panel_report, routing
from tools.coopmwm.accel_storage import read, write, ledger, charged
from tools.coopmwm.accel_learner import load, save, update, evidence, viability, advantages, UNIT
from tools.coopmwm.acceleration import Policy
from tools.coopmwm.accel_protocol import SEED, digest, worlds
from tools.coopmwm.protocol import batch_worlds


def verify_a(root):
    """Prospective A-only release of a bridge STOP; preserve the original STOP."""
    from tools.coopmwm.accel_protocol import budget, sha
    from tools.coopmwm.plans import SEARCH
    b = read(root/'BINDING_TRAINING.json')
    assert b['budget'] == budget() and b['search'] == SEARCH
    assert b['amendment_sha256'] in (root/b.get('authority_file','AUTHORITY.md')).read_text()
    assert b['cpu']['pass_'] and not b['cpu']['skips'] and not b['cpu']['sim_s']
    assert all(v['pass_'] for v in b['gpu_qualification'].values())
    for name, h in b['hashes'].items():
        assert sha(root/name) == h, 'immutable source mismatch '+name
    assert not (root/'VIABILITY_PAUSED.json').exists()
    amendment = read(root/'A_WARM2_AMENDMENT.json')
    assert amendment['arm'] == 'A' and amendment['stop_update'] == 4
    assert sha(root/'STOP.json') == amendment['preserved_stop_sha256']
    assert read(root/'STOP.json') == dict(error="No module named 'torch'", source='owned plan bridge; no retry')
    reg = read(root/'A_WARM2_REGISTRATION.json')
    assert sha(root/'A_WARM2_AMENDMENT.json') == reg['amendment_sha256']
    assert reg['amendment_sha256'] in (root/'A_WARM2_AUTHORITY.md').read_text()
    assert not (root/'A_WARM_CONTINUATION.claim').exists(), 'original continuation attempted physics'
    return b


def adoptable(root, key, charges):
    """A completed result is reused; an attempted unfinished job is never retried."""
    path = root/'jobs'/key
    if (path/'RESULT.json').exists():
        assert (path/'CLOSE.json').exists()
        return 'complete'
    assert key in charges and charges[key]['sim_s'] == 1200
    assert '_A_' in key and not (path/'CLAIM.json').exists()
    assert not (path/'CLOSE.json').exists()
    return 'unattempted'


def run(root):
    import fcntl
    import os
    import numpy as np
    b = verify_a(root)
    assert (root/'HOLD_WARM2208.json').exists()
    d = read(root/'TRAIN_DISPATCH.json')
    s = (Path('/proc')/str(d['pid'])/'stat').read_text().split()
    assert s[21] == d['start_ticks'] and s[2] == 'T'
    lock = (root/'A_WARM_CONTINUATION.claim').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not list(root.glob('A_UPDATE_*.json'))
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    policy, opt, rng, payload = load(root/'A_checkpoint_000.pt')
    assert payload['update'] == 0
    controls = read(root/'MAPPO_CHECK.json')['records']
    try:
        for u in range(1, 5):
            start = time.monotonic()
            if u == 1:
                charges = {c['id']: c for c in ledger(root)}
                eps = []; jobs = []
                for w in batch_worlds(u):
                    key = f'u{u:03}_A_{w["seed"]}'
                    if adoptable(root, key, charges) == 'complete':
                        eps.append(read(root/'jobs'/key/'RESULT.json')); continue
                    jobs.append(dict(id=key, path=str(root/'jobs'/key), world=w,
                        role='train', arm='A', weights=policy.state_dict(), runtime=b['runtime'],
                        checkpoint=b['checkpoint'], noise_seed=int(digest(dict(seed=SEED,world=w['seed'],wave=f'u{u:03}'))[:15],16),
                        deterministic=False, sim_s=1200, queue_root=str(root), accel_bounds=b['accel_bounds'],
                        gpu_deadline=b['gpu_deadline'], guided=False, update=u))
                # These exact reservations were charged before the original hold.
                from tools.coopmwm.runner import admission
                from tools.coopmwm.accel_plant import episode
                admission(b)
                with ProcessPoolExecutor(max_workers=b['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
                    eps.extend(pool.map(episode,jobs))
            else:
                eps = wave(root,b,{'A':policy},batch_worlds(u),'train',f'u{u:03}','train',False,dict(update=u))
            eps = sorted(eps,key=lambda e:e['world']['seed'])
            routes = routing(eps)
            write(root/f'A_VIABILITY_{u:03}.json',dict(timestamp_utc=timestamp(),update=u,arm='A',
                viability=viability(evidence(eps)),routing=routes))
            metrics = update(policy,opt,eps,rng)
            save(root/f'A_checkpoint_{u:03}.pt',policy,opt,rng,u,charged(root))
            raw = [v*UNIT for e in eps for v,_ in advantages(e['steps'])]
            write(root/f'A_UPDATE_{u:03}.json',dict(timestamp_utc=timestamp(),update=u,arm='A',training_seed=SEED,
                ppo=metrics,trust=dict(beta=0.),beta_used=0.,proposal_routing=routes,
                team_only_noise=dict(advantage_variance_veh_s_squared=float(np.var(raw,ddof=1))),
                wall_s=time.monotonic()-start,charged_sim_s=charged(root),reward_source='SUMO source-inclusive team J only'))
            if u%4==0:
                check = wave(root,b,{'A':policy},worlds('check'),'check',f'check{u:03}_A','checks')
                panel_report(root,f'A_CHECK_{u:03}.json',check,controls,'check')
        write(root/'A_WARM_CONTINUATION_PAUSED.json',dict(timestamp_utc=timestamp(),update=4,
            charged_sim_s=charged(root),reason='A preserved through u4; B/R warm qualification independent'))
    except BaseException as e:
        write(root/'A_WARM_CONTINUATION_FAILED.json',dict(timestamp_utc=timestamp(),error=repr(e)))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
