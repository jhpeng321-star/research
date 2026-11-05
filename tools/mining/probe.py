"""Cooperation feasibility probe (owner 2026-11-06 05:10; phase coop_probe <= 250,000 sim-s).

Question: does real GROUP cooperation exist? synergy = J(best single-CAV solo) - J(group) on the
common window, per CRN seed, within-seed per-second circular-block paired bootstrap (the mining
per_seed_v1 statistic, bank_statistics._stats); admitted only if the one-sided 95% lower bound > 0
on that seed. Settings: wall (lead R 986061 t455 + similar moments), hp (highest registered
demand x CAV share TRAIN worlds), lowcoop (owner-approved HDV lane-change variant, probe only).

WM proposes/ranks (gpuA, probe_wm.py), SUMO verifies here: one sv.verify per (context, seed) holds
the policy-continues branch, every group candidate and every member solo (shared CRN window).
Seed 1 = the collection CRN; seeds r >= 2 = rb.replicate(snapshot, 'coopprobe#<sid>#r<r>')
(evaluation seed ranges are never a generator seed). Charges: own ledger, hard cap, never replayed.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('PROBE_ROOT', '/home/ubuntu/coopprobe20261106'))
CAP = int(os.environ.get('PROBE_CAP', '250000'))
TAIL = int(os.environ.get('PROBE_TAIL', '120'))
EVAL = ((962000, 962999), (964000, 964199), (977000, 977999))
SETTINGS = ('wall', 'hp', 'lowcoop')


def _write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, sort_keys=True))
    os.replace(tmp, p)


def check_world(world):
    from tools.coopmwm.protocol import worlds
    s = int(world['seed'])
    assert not any(lo <= s <= hi for lo, hi in EVAL), 'evaluation seed'
    assert any(int(w['seed']) == s for w in worlds('train')), 'not a registered TRAIN world'
    assert 986000 <= s <= 986071


def charge(ident, sim_s, setting, branches):
    import fcntl
    from tools.coopmwm.exact_grade import BudgetBlocked
    with (ROOT/'LEDGER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = ROOT/'CHARGES.jsonl'
        rows = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
        if any(r['id'] == ident for r in rows):
            raise BudgetBlocked('already charged; never replay '+ident)
        if sum(r['sim_s'] for r in rows)+int(sim_s) > CAP:
            raise BudgetBlocked(f'coop_probe cap {CAP} reached')
        with ledger.open('a') as f:
            f.write(json.dumps(dict(id=ident, phase='coop_probe', setting=setting, sim_s=int(sim_s),
                                    branches=branches, unix=time.time()))+'\n')
            f.flush(); os.fsync(f.fileno())


def spent():
    ledger = ROOT/'CHARGES.jsonl'
    rows = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
    out = {}
    for r in rows:
        out[r['setting']] = out.get(r['setting'], 0)+r['sim_s']
    return out


# ------------------------------------------------------------------ collection (settings hp/lowcoop)

def collect(setting, arm, seed, keep=3):
    """One plain frozen-u0 policy episode (as mining collect2) on a registered TRAIN world; keeps up to
    `keep` merge-pressure decision snapshots (observable selection only) with per-second policy
    reference commands of EVERY CAV within the sensed band s in [-400, 100] for 130 s."""
    import torch
    from tools.coopmwm import accel_plant, sumo_verify as sv
    from tools.coopmwm.protocol import worlds
    from tools.mining.collect_contexts import FIELDS, opportunities
    world = next(w for w in worlds('train') if int(w['seed']) == int(seed))
    check_world(world)
    variant = os.environ.get('PROBE_HDV_VARIANT')
    assert (setting == 'lowcoop') == bool(variant), 'variant only for the lowcoop setting'
    ident = f'cp_{setting}_{arm}_train_{seed}'
    work = ROOT/'collect'/ident
    work.mkdir(parents=True)
    weights = torch.load(ROOT/'frozen_u0.pt', weights_only=False, map_location='cpu')['policy']
    job = dict(id=ident, path=str(work), world=world, role='train', arm=arm, update=0, training_seed=20261101,
               weights=weights, runtime='/home/ubuntu/e3final',
               checkpoint='/home/ubuntu/mappospeed/production2/policy_00169.pt',
               noise_seed=20261106+int(seed)+(0 if arm == 'B' else 500000)+7000,
               deterministic=False, guided=False, b_replay=True, v2v=True, health_guards=False,
               sim_s=1200, stop=1200, warm_history=True, accel_bounds=[-3., 2.6])
    kept, profiles = [], {}
    PROFILE_S, SPACING = 130, 20

    class Capture:
        def __call__(self, controller, job_, prefixes):
            self.c, self.job, self.prefixes = controller, job_, prefixes

        def after_step(self, t, after):
            e = self.c.events[-1]
            t = int(e['time_s'])
            members = e['sample']['context']['members']
            action = e['sample']['action']
            for t0, prof in profiles.items():
                k = t-t0
                if 0 <= k < PROFILE_S:
                    for vid, row in prof.items():
                        row.append(float(e['commands'][vid])+float(action[members.index(vid)][0])
                                   if vid in members and vid in e['commands'] else None)
            if not 300 <= t <= 600 or t % 5:
                return
            rows = e['rows']
            options = opportunities(rows)
            if not options:
                return
            band = [r for r in rows if r['is_cav'] and -400 <= r['s_m'] <= 100]
            main_up = [r for r in band if r['lane'] in (0, 1)]
            # observable pressure + multi-CAV availability (>= 3 mainline CAVs in the band)
            pressure, main, ramp, local = options[0]
            score = (min(len(main_up), 6),)+tuple(pressure)
            if len(main_up) < 3:
                return
            if any(abs(x[1]-t) < SPACING and x[0] >= score for x in kept):
                return
            kept[:] = [x for x in kept if abs(x[1]-t) >= SPACING]
            if len(kept) >= keep and min(x[0] for x in kept) >= score:
                return
            snap = sv.capture(self.c, self.prefixes, self.job, t, e['sample'], self.c.generator)
            sensed = {r['vid'] for c in band for r in rows if abs(r['s_m']-c['s_m']) <= 100}
            ctx_rows = [{k: r[k] for k in FIELDS} for r in rows if r['vid'] in sensed]
            cavs = [r['vid'] for r in band]
            context = dict(state_id=f'{ident}_t{t}', world=world, arm=arm, t0=t, group=[main, ramp],
                           pressure=list(pressure), rows=ctx_rows, cavs=cavs, setting=setting,
                           variant=variant, snapshot_sha256=sv.snapshot_sha(snap),
                           selection='(min(#mainline CAVs in s[-400,100],6), local ramp count, local mainline '
                                     'count, sum 1/(1+v_ramp), -|ds|); CAV-sensed rows only; no outcome')
            profiles[t] = {v: [float(e['commands'][v])+float(action[members.index(v)][0])
                               if v in members and v in e['commands'] else None] for v in cavs}
            kept.append([score, t, snap, context])
            kept.sort(key=lambda x: x[0], reverse=True)
            del kept[keep:]

    charge('collect_'+ident, 1200, setting, 1)
    result = accel_plant.episode(job, controller_hook=Capture())
    variant_count = None
    if variant:
        from traffic_shepherd.c4 import population as pop
        variant_count = getattr(pop.vtype_attrs, 'count', None)
        if not variant_count:
            raise RuntimeError('variant requested but no HDV vType was patched')
    saved = []
    for score, t0, snap, context in kept:
        d = work/f'ctx_t{t0}'
        d.mkdir()
        torch.save(snap, d/'SNAPSHOT.pt')
        context['policy_profile'] = profiles[t0]
        _write(d/'CONTEXT.json', context)
        saved.append(dict(t0=t0, dir=str(d), score=list(score)))
    _write(work/'COLLECTION_RECEIPT.json', dict(world=world, arm=arm, setting=setting, variant=variant,
           variant_hdv_vtypes_patched=variant_count, actual_sim_s=result['actual_sim_s'], contexts=saved,
           unix=time.time()))
    return saved


# ------------------------------------------------------------------ candidates

def _pol(ctx, vids, duration):
    from tools.mining.mine_verify import _pol as pol
    return pol(ctx, vids, duration)


def ease_plan(ctx, family, members, onset, hold, depth, tag='g'):
    """All members ease by `depth` during [onset, onset+hold) on top of the seed-1 policy reference
    (wall = >= 3 mainline CAVs, easing = mixed lanes/small depth, ramp_assist = main + ramp CAV)."""
    import numpy as np
    duration = int(onset)+int(hold)
    pol = _pol(ctx, list(members), duration)
    if pol is None:
        return None
    off = np.zeros((len(members), duration))
    off[:, int(onset):] = -float(depth)
    base = np.asarray(pol, float)
    pid = f'{tag}_{family}_{"_".join(members)}_o{onset}_h{hold}_d{depth:g}'
    return dict(id=pid, family=family, members=list(members), duration_s=duration, origin='mining_op2',
                policy_profile=base.tolist(), profile=(base+off).tolist(), perturbation_profile=off.tolist(),
                probe=dict(onset=int(onset), hold=int(hold), depth=float(depth)))


def slate(ctx, families=('wall', 'easing', 'ramp_assist')):
    """Deterministic, observable-only candidate slate (the WM ranks it; SUMO verifies the top)."""
    rows = ctx['rows']
    cav = {r['vid']: r for r in rows if r['is_cav'] and r['vid'] in ctx['policy_profile']}
    ramp_q = sorted([r for r in cav.values() if r['lane'] == 2 and -100 <= r['s_m'] <= 75.44],
                    key=lambda r: -r['s_m'])
    head = ramp_q[0]['s_m'] if ramp_q else 75.44
    mains = sorted([r for r in cav.values() if r['lane'] in (0, 1) and r['s_m'] < head+5
                    and head-r['s_m'] <= 300], key=lambda r: -r['s_m'])
    lane1 = [r for r in mains if r['lane'] == 1]
    out = []

    def add(p):
        if p is not None and p['id'] not in {x['id'] for x in out}:
            out.append(p)
    if 'wall' in families:
        for k in (3, 4, 5, 6):
            if len(mains) >= k:
                m = tuple(r['vid'] for r in mains[:k])
                for hold, depth in ((20, .6), (30, 1.)):
                    add(ease_plan(ctx, 'wall', m, 0, hold, depth))
        if len(lane1) >= 3:
            m = tuple(r['vid'] for r in lane1[:min(6, len(lane1))])
            add(ease_plan(ctx, 'wall', m, 0, 30, 1.))
    if 'easing' in families:
        for k in (3, 6):
            if len(mains) >= k:
                add(ease_plan(ctx, 'easing', tuple(r['vid'] for r in mains[:k]), 0, 30, .3))
    if 'ramp_assist' in families and ramp_q:
        behind = [r for r in mains if r['lane'] == 1][:2]
        if behind:
            m = tuple(r['vid'] for r in behind)+(ramp_q[0]['vid'],)
            add(ease_plan(ctx, 'ramp_assist', m, 0, 20, .6))
    return out


def g5_slate(ctx, prow):
    """Accepted-G5 open_gap_merge / yield_lane / gap_open candidates exactly as mining op2."""
    from tools.mining.mine_verify import candidates
    c = dict(ctx)
    cands, notes = candidates(c, None, prow)
    return cands, notes


def solos(plan):
    from tools.coopmwm import replay_bank as rb
    out = []
    for m in plan['members']:
        s = rb.solo_candidate(plan, m)
        if plan.get('probe'):   # ease solo depends only on (member, onset, hold, depth): share it
            q = plan['probe']
            s['id'] = f"solo_{m}_o{q['onset']}_h{q['hold']}_d{q['depth']:g}"
        out.append(s)
    return out


# ------------------------------------------------------------------ verification

def seed_snapshot(snap, sid, r):
    from tools.coopmwm import replay_bank as rb
    from tools.coopmwm.protocol import digest
    if r == 1:
        return snap, None
    salt = f'coopprobe#{sid}#r{r}'
    for n in range(100):
        s = salt if not n else f'{salt}#x{n}'
        g = int(digest(dict(replication=s))[:15], 16)
        if not any(lo <= g <= hi for lo, hi in EVAL):
            return rb.replicate(snap, s), s
    raise ValueError('salt')


def verify(setting, ctx_dir, groups, seeds, workers, tag):
    """Per seed: ONE sv.verify with policy + groups + all member solos. Returns result rows."""
    import torch
    from tools.coopmwm import sumo_verify as sv, bank_statistics as bs, replay_bank as rb
    ctx = json.loads((Path(ctx_dir)/'CONTEXT.json').read_text())
    check_world(ctx['world'])
    snap = torch.load(Path(ctx_dir)/'SNAPSHOT.pt', weights_only=False)
    assert sv.snapshot_sha(snap) == ctx['snapshot_sha256']
    sid = ctx['state_id']
    cands, owner = [], {}
    for g in groups:
        cands.append(g)
        for s in solos(g):
            if s['id'] not in {c['id'] for c in cands}:
                cands.append(s)
            owner.setdefault(g['id'], []).append(s['id'])
    rows = []
    for r in seeds:
        snap_r, salt = seed_snapshot(snap, sid, r)
        work = ROOT/'verify'/setting/tag/f'r{r}'
        runner = sv.SubprocessBranchRunner(work/'branches', workers=workers, timeout_s=3600)

        def chg(vid, specs, _r=r):
            charge(f'{tag}_r{_r}_{vid}', sum(int(s['sim_s']) for s in specs), setting, len(specs))
        rec = sv.verify(snap_r, cands, runner, chg, 'wm', workers=workers, root=work, tail_s=TAIL)
        base = dict(setting=setting, state_id=sid, world=ctx['world'], arm=ctx['arm'], t0=ctx['t0'],
                    seed_index=r, salt=salt, verification_id=rec['verification_id'], status=rec['status'],
                    charged_sim_s=rec.get('charged_sim_s'), window=rec.get('window'), tag=tag)
        if rec['status'] not in ('accepted', 'rejected'):
            rows.append(dict(base, error=rec.get('reason')))
            with (ROOT/'results'/f'{setting}.jsonl').open('a') as f:
                f.write(json.dumps(rows[-1])+chr(10))
            continue
        bdir = work/'branches'
        br = rec['branches']
        idx = {c['id']: i+1 for i, c in enumerate(cands)}
        tapes = {}

        def tape(i):
            if i not in tapes:
                tapes[i] = bs._read_result(bdir, rec['verification_id'], br[i])
            return tapes[i]
        Jp = rec['J_policy_veh_s']
        for g in groups:
            gi = idx[g['id']]
            Jg = br[gi]['J_veh_s']
            sj = {sid_: br[idx[sid_]]['J_veh_s'] for sid_ in owner[g['id']]}
            row = dict(base, group_id=g['id'], family=g['family'], members=g['members'],
                       n_members=len(g['members']), J_policy=Jp, J_group=Jg, J_solos=sj,
                       group_adv=None if Jg is None else Jp-Jg,
                       safety={k: (None if Jg is None else float(br[gi]['window'][k])-float(br[0]['window'][k]))
                               for k in rb.SAFETY_KEYS})
            if Jg is None or any(v is None for v in sj.values()):
                row['synergy_status'] = 'incomplete'
                rows.append(row)
                continue
            best = min(sj, key=lambda k: (sj[k], k))
            row.update(best_solo=best, best_solo_adv=Jp-sj[best], synergy=sj[best]-Jg)
            try:
                st = bs._stats([(tape(idx[best]), tape(gi), rec['window'])])
                row['synergy_stats'] = {k: st[k] for k in st if not isinstance(st[k], list)}
                row['synergy_lb95'] = st['lower95_one_sided_veh_s']
                adv = bs._stats([(tape(0), tape(gi), rec['window'])])
                row['adv_lb95'] = adv['lower95_one_sided_veh_s']
                row['synergy_status'] = 'admitted' if st['lower95_one_sided_veh_s'] > 0 else 'not_admitted'
            except Exception as e:
                row['synergy_status'] = 'data_missing'
                row['error'] = repr(e)
            rows.append(row)
        with (ROOT/'results'/f'{setting}.jsonl').open('a') as f:
            for x in rows:
                if x.get('seed_index') == r:
                    f.write(json.dumps(x)+'\n')
    return rows


def main(argv):
    cmd = argv[0]
    (ROOT/'results').mkdir(parents=True, exist_ok=True)
    if cmd == 'collect':
        setting, arm, seeds = argv[1], argv[2], [int(x) for x in argv[3].split(',')]
        for s in seeds:
            if (ROOT/'STOP').exists():
                break
            try:
                print(json.dumps(dict(seed=s, saved=collect(setting, arm, s))), flush=True)
            except Exception as e:
                import traceback
                print(json.dumps(dict(seed=s, error=repr(e), tb=traceback.format_exc()[-2000:])), flush=True)
    elif cmd == 'verify':
        # verify <setting> <jobs.json>: [{ctx_dir, group_ids or plans, seeds, tag}]
        setting, jobs = argv[1], json.loads(Path(argv[2]).read_text())
        workers = int(os.environ.get('PROBE_WORKERS', '14'))
        for job in jobs:
            if (ROOT/'STOP').exists() or (ROOT/f'STOP_{setting}').exists():
                break
            claim = ROOT/'claims'/(job['tag']+'.claim')
            claim.parent.mkdir(exist_ok=True)
            try:
                with claim.open('x') as f:
                    f.write(str(os.getpid()))
            except FileExistsError:
                continue
            try:
                rows = verify(setting, job['ctx_dir'], job['plans'], job['seeds'], workers, job['tag'])
                print(json.dumps(dict(tag=job['tag'], n=len(rows))), flush=True)
            except Exception as e:
                import traceback
                print(json.dumps(dict(tag=job['tag'], error=repr(e), tb=traceback.format_exc()[-3000:])), flush=True)
                if 'cap' in str(e):
                    break


if __name__ == '__main__':
    main(sys.argv[1:])
