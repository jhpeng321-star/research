"""Mining operator 2: native all8 SUMO verification of G5 + control candidates.

One decision per captured TRAIN context (collect2). Candidates are built from
the observable context and the recorded seed-1 policy reference, with the
accepted G5 (b903c6c2) executable fields. sv.verify + verify_perseed.confirm_all8
run group/policy on both CRN seeds, extended later-harm windows and per-member
solos; synergy = best solo J - group J on the common window (per seed).
Charges go to the mining ledger (lock, never replayed, cap MINING_CAP).
"""
import json
import os
from pathlib import Path
import sys
import time

MINING_CAP = int(os.environ.get('MINING_CAP', '490000'))
TAIL = 120


def _charge_factory(parent, prefix=''):
    import fcntl
    from tools.coopmwm.exact_grade import BudgetBlocked

    def charge(ident, specs):
        ident = prefix+ident
        amount = sum(int(s['sim_s']) for s in specs)
        with (parent/'MINING_BUDGET.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = parent/'CHARGES.jsonl'
            rows = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()]
            if any(r['id'] == ident for r in rows):
                raise BudgetBlocked('attempt already charged; never replay '+ident)
            if sum(r['sim_s'] for r in rows)+amount > MINING_CAP:
                raise BudgetBlocked(f'mining cap {MINING_CAP} reached')
            with ledger.open('a') as f:
                f.write(json.dumps(dict(id=ident, phase='mining', sim_s=amount, branches=len(specs),
                                        unix=time.time()))+'\n')
                f.flush(); os.fsync(f.fileno())
    return charge


def _pol(ctx, vids, duration):
    # Seconds where the CAV is outside the policy's controlled context execute the native
    # command (plan accelerations are applied only to context members); record that.
    prof = ctx['policy_profile']
    cmds = ctx.get('_cmds') or {}
    t0 = int(ctx['t0'])
    rows = []
    for v in vids:
        row = list(prof.get(v) or [])
        if len(row) < duration:
            return None
        out = []
        for k, x in enumerate(row[:duration]):
            if x is None:
                x = (cmds.get(t0+k) or {}).get(v)
            if x is None and out and v not in {r['vid'] for r in (ctx.get('_rows') or {}).get(t0+k, [])}:
                x = out[-1]   # vehicle has left the network: inert placeholder, never applied
            if x is None:
                return None
            out.append(float(x))
        rows.append(out)
    return rows


def policy_rows(ctx_dir, t0, span=40):
    """Seed-1 policy reference rows (the collection episode itself, same CRN as the branch)."""
    import gzip, hashlib
    marker = json.loads((ctx_dir.parent/'RESULT.json').read_text())
    raw = gzip.decompress((ctx_dir.parent/marker['file']).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == marker['sha256']
    tape = json.loads(raw)
    steps = [s for s in tape['steps'] if t0 <= int(s['time_s']) <= t0+span]
    return ({int(s['time_s']): s['rows'] for s in steps},
            {int(s['time_s']): s['commands'] for s in steps})


def feasible(rows, actor):
    """Accepted G5 envelope on the actor's own 100 m packet (native verifier reconstruction)."""
    from tools.coopmwm import group_lane
    from tools.coopmwm.merge_features import SENSOR_RANGE_M
    by = {r['vid']: r for r in rows}
    if actor not in by or not by[actor].get('is_cav'):
        return False
    a = by[actor]
    sensed = [{k: r[k] for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')}
              for r in rows if abs(r['s_m']-a['s_m']) <= SENSOR_RANGE_M]
    e = group_lane.envelope(dict(self=actor, rows=sensed, observed_lanes=(0, 1, 2)), actor)
    return bool(e and e['left'])


def candidates(ctx, wm_rank=None, prow=None):
    from tools.coopmwm import group_lane, moves
    rows = {r['vid']: r for r in ctx['rows']}
    main, ramp = ctx['group']
    t0 = int(ctx['t0'])
    out, notes = [], []

    def add(family, members, onset, hold, depth, tag):
        move = moves.Move(family, tuple(members), int(onset), int(hold), float(depth))
        duration = int(onset)+int(hold)
        pol = _pol(ctx, members, duration)
        if pol is None:
            notes.append(f'{tag}{family}o{onset}h{hold}: policy reference incomplete'); return
        pid = f'{tag}_{family}_{"_".join(members)}_o{onset}_h{hold}_d{depth:g}'
        plan = dict(id=pid, family=family, members=list(members), duration_s=duration, origin='mining_op2')
        if family in group_lane.FAMILIES:
            plan.update(group_lane.plan_fields(move, duration))
            plan.update(group_lane.executable_fields(move, duration, pol))
            group_lane.from_plan(plan)
        else:
            off = group_lane.macro_offsets(move, duration)
            import numpy as np
            base = np.asarray(pol, float)
            plan.update(policy_profile=base.tolist(), profile=(base+off).tolist(), perturbation_profile=off.tolist())
        if pid not in {c['id'] for c in out}:
            out.append(plan)

    z0, z1 = moves._ZONE_S0, moves._ZONE_S1
    if os.environ.get('MINE_PROBE') == '1':
        # Non-G5 multi-CAV probe (report-only; not MINED_MOVES-eligible): both-easing
        # ramp_assist pairs and an up-to-6-CAV mainline wall behind the merging ramp CAV.
        cav0 = [r for r in ctx['rows'] if r['is_cav']]
        rr = [r for r in cav0 if r['lane'] == 2 and z0 <= r['s_m'] <= z1]
        for ramp_r in sorted(rr, key=lambda r: -r['s_m'])[:1]:
            behind = sorted([r for r in cav0 if r['lane'] in (0, 1) and r['s_m'] < ramp_r['s_m']
                             and ramp_r['s_m']-r['s_m'] <= 100], key=lambda r: -r['s_m'])
            if behind:
                add('ramp_assist', (behind[0]['vid'], ramp_r['vid']), 0, 20, .6, 'p')
                add('ramp_assist', (behind[0]['vid'], ramp_r['vid']), 0, 30, 1.0, 'p')
            if len(behind) >= 2:
                add('wall', tuple(r['vid'] for r in behind[:6]), 0, 20, .6, 'p')
                add('wall', tuple(r['vid'] for r in behind[:6]), 0, 30, 1.0, 'p')
        return out, notes
    cav = [r for r in ctx['rows'] if r['is_cav'] and z0 <= r['s_m'] <= z1]
    ramps = [r for r in cav if r['lane'] == 2]
    mains = [r for r in cav if r['lane'] == 1]
    link = float(moves.VARIETY_REGISTERED['variety_gap_link_m'])

    def at(t, vid):
        return {r['vid']: r for r in (prow or {}).get(t, [])}.get(vid)
    pairs = []
    for rr in ramps:
        for mm in mains:
            if not (mm['s_m'] < rr['s_m'] and rr['s_m']-mm['s_m'] <= link):
                continue      # accepted G5 geometry: the gap-opening yielder is BEHIND the merger
            ok = []
            for k in range(0, 31):
                r1, m0 = at(t0+k+1, rr['vid']), at(t0+k, mm['vid'])
                if (r1 and m0 and r1['lane'] == 2 and m0['lane'] == 1 and m0['s_m'] < r1['s_m']
                        and feasible(prow.get(t0+k+1, []), rr['vid'])):
                    ok.append(k)
            if ok:
                pairs.append((rr['s_m']-mm['s_m'], mm['vid'], rr['vid'], ok))
    pairs.sort()
    notes.append('ogm pairs (gap, main, ramp, feasible onsets): ' +
                 str([(round(g, 1), m, r, o[:4]) for g, m, r, o in pairs[:3]]))
    if wm_rank:
        # WM proposer (GPU ranker): its top predicted-synergy G5 moves go first.
        for w in [w for w in wm_rank if w['family'] in group_lane.FAMILIES][:4]:
            try:
                add(w['family'], tuple(w['members']), int(w['onset']), int(w.get('hold', 20)),
                    float(w['depth']), 'w')
            except ValueError as err:
                notes.append(f'WM candidate rejected: {err}')
    for g, m, r, ok in pairs[:2]:
        picks = [ok[0]] + ([ok[min(len(ok)-1, 3)]] if len(ok) > 1 else [])
        for k in dict.fromkeys(picks):
            add('open_gap_merge', (m, r), k, 20, 1.0, 'a')
        add('open_gap_merge', (m, r), ok[0], 20, .6, 'a')
    # yield_lane: lane-1 CAV whose own left move is gate-feasible, any ramp CAV within the link
    for mm in mains:
        partner = next((rr for rr in ramps if abs(rr['s_m']-mm['s_m']) <= link), None)
        if partner is None:
            continue
        ks = [k for k in range(0, 31) if at(t0+k, mm['vid']) and at(t0+k, mm['vid'])['lane'] == 1
              and feasible(prow.get(t0+k, []), mm['vid'])]
        if ks:
            add('yield_lane', (mm['vid'], partner['vid']), ks[0], 20, .3, 'a')
            break
    if not pairs:
        notes.append('no lane-1 CAV behind a lane-2 CAV with a gate-feasible attempt tick (onset 0..30)')
    if pairs:
        add('gap_open', (pairs[0][1], pairs[0][2]), 0, 20, 1.0, 'd')
    if not out:
        return out, notes
    # (c) merge-timed upstream wall: two mainline CAVs upstream of the merge zone,
    # held only while both are still upstream (release at/before the merge).
    up = sorted([r for r in ctx['rows'] if r['is_cav'] and r['lane'] in (0, 1) and r['s_m'] < moves._ZONE_S0],
                key=lambda r: -r['s_m'])
    if len(up) >= 2:
        a, b = up[0], up[1]
        reach = (moves._ZONE_S0-a['s_m'])/max(a['v_mps'], 1.)
        hold = int(min(90, reach))
        if hold >= 20:
            add('wall', (a['vid'], b['vid']), 0, hold, .6, 'c')
        else:
            notes.append(f'wall: leader reaches merge in {reach:.1f}s < 20s hold minimum')
    else:
        notes.append('wall: <2 upstream mainline CAVs sensed')
    return out, notes


def run_context(ctx_dir, bank_root_dir, arm, parent, wm_dir, workers):
    import torch
    from tools.coopmwm import sumo_verify as sv, replay_bank as rb, verify_perseed as vp
    from tools.coopmwm.storage import write as swrite
    ctx = json.loads((ctx_dir/'CONTEXT.json').read_text())
    assert ctx['arm'] == arm and 986000 <= int(ctx['world']['seed']) <= 986071
    rank = None
    rp = wm_dir/(ctx['state_id'].replace('op2_collect01_', '')+'.rank.json')
    for cand in (wm_dir/(ctx['state_id']+'.rank.json'), rp):
        if cand.exists():
            rank = json.loads(cand.read_text()).get('ranked'); break
    prow, ctx['_cmds'] = policy_rows(ctx_dir, int(ctx['t0']), 130)
    ctx['_rows'] = prow
    cands, notes = candidates(ctx, rank, prow)
    bank = Path(rb.bank_root(bank_root_dir, arm))
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    out = dict(state_id=ctx['state_id'], arm=arm, t0=ctx['t0'], world=ctx['world'], group=ctx['group'],
               pressure=ctx['pressure'], wm_ranked=rank is not None, notes=notes,
               candidates=[c['id'] for c in cands])
    if not cands:
        out['status'] = 'no_candidates'; return out
    snap = torch.load(ctx_dir/'SNAPSHOT.pt', weights_only=False)
    assert sv.snapshot_sha(snap) == ctx['snapshot_sha256']
    sid = ctx['state_id']
    if not (bank/'snapshots'/(sid+'.pt')).exists():
        torch.save(snap, bank/'snapshots'/(sid+'.pt'))
        swrite(bank/'snapshots'/(sid+'.json'), dict(state_id=sid, arm=arm, job=sid.rsplit('_t', 1)[0],
               world=ctx['world'], t0=int(ctx['t0']), group=dict(members=ctx['group']), regime=None,
               policy_update=0, record_sha256=ctx['snapshot_sha256'], unix=time.time(),
               mining=dict(pressure=ctx['pressure'], selection=ctx['selection'], wm_ranked=rank is not None)))
    charge = _charge_factory(parent, os.environ.get('MINE_TAG', 'v1')+'_')
    runner = sv.SubprocessBranchRunner(bank/'sumo_verification'/'branches', workers=workers, timeout_s=3600)
    proposer = 'wm' if rank is not None else 'random'
    r = sv.verify(snap, cands, runner, lambda vid, specs: charge('first_'+vid, specs), proposer,
                  workers=workers, root=bank/'sumo_verification', tail_s=TAIL)
    (bank/'decisions').mkdir(exist_ok=True)
    (bank/'decisions'/(sid+'.json')).write_text(json.dumps(dict(state_id=sid, verification_id=r['verification_id'],
        status=r['status'], policy_update=0, arm=arm, available_k=len(cands), issued_unix=time.time(),
        matched_B=None)))
    out.update(verification_id=r['verification_id'], first_status=r['status'])
    if r['status'] not in ('accepted', 'rejected'):
        out['status'] = 'first_seed_'+r['status']; out['reason'] = r.get('reason'); return out
    rb._confirm_charge = lambda run_root, phase, kind: (lambda vid, specs: charge(f'{kind}_{vid}', specs))

    def runner_for(work):
        return sv.SubprocessBranchRunner(Path(work)/'branches', workers=workers, timeout_s=3600)
    receipt = dict(r, _branches=str(bank/'sumo_verification'/'branches'), _policy_version='u0')
    conf = vp.confirm_all8(bank_root_dir, arm, receipt, snap, runner_for, str(bank_root_dir), 'mining',
                           workers=workers, version='u0', candidates=cands, slim_rejected_tapes=False)
    out.update(status='confirmed', admitted=conf['admitted'], confirmation_sha256=conf['confirmation_sha256'],
               charged_sim_s=r.get('charged_sim_s', 0)+conf.get('charged_sim_s', 0))
    return out


def main(root, arm, ctx_list, workers):
    parent = Path('/root/mining20261106')
    root.mkdir(exist_ok=True)
    wm_dir = parent/'wm_lane04'
    os.nice(10)
    log = root/f'RUN_{arm}_{os.getpid()}.jsonl'
    tag = os.environ.get('MINE_TAG', 'v1')
    bank_root_dir = root/('bank' if tag == 'v1' else 'bank_'+tag)
    for line in Path(ctx_list).read_text().split():
        ctx_dir = Path(line)
        if (root/('STOP_'+tag)).exists():
            break
        claim = root/('claims_'+tag)/(ctx_dir.parent.name+'_'+ctx_dir.name+'.claim')
        claim.parent.mkdir(exist_ok=True)
        try:
            with claim.open('x') as f:
                f.write(str(os.getpid()))
        except FileExistsError:
            continue
        started = time.time()
        try:
            res = run_context(ctx_dir, bank_root_dir, arm, parent, wm_dir, workers)
        except Exception as e:
            import traceback
            res = dict(state_id=ctx_dir.parent.name+'/'+ctx_dir.name, status='error', error=repr(e),
                       traceback=traceback.format_exc()[-3000:])
        res.update(wall_s=time.time()-started, unix=time.time())
        with log.open('a') as f:
            f.write(json.dumps(res)+'\n')
        if 'mining cap' in str(res.get('reason', '')):
            break


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
