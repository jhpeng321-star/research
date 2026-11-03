"""MAIN12:05 policy cooperation diagnostic; independent of every training owner.

Non-group cars use the new policy, exactly as in the matched lesson check.
The all-new branch is the matched lesson branch; solos switch only the other
group members to u3. The identical captured t0 action is retained, as in lessons;
policy interventions begin at t0+1. Every branch shares the prefix and noise draws.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from . import replay_bank as rb, sumo_verify as sv
from .exact_grade import _seal, _write_new, verify_sealed
from .protocol import digest, sha


class MixedPolicy:
    """Two policy evaluations, identical CRN draws; u3 only on reverted members."""
    def __init__(self, new, old, reverted, new_members_only=None):
        self.new, self.old, self.reverted = new, old, set(reverted)
        self.new_members_only = None if new_members_only is None else set(new_members_only)

    def __call__(self, *args, **kwargs):
        return self.new(*args, **kwargs)

    def sample(self, context, generator, deterministic=False, memory=None):
        state = generator.get_state().clone()
        a, ma = self.new.sample(context, generator, deterministic, memory)
        after = generator.get_state().clone()
        other = torch.Generator().set_state(state)
        b, mb = self.old.sample(context, other, deterministic, memory)
        if not torch.equal(after, other.get_state()):
            raise ValueError('policy synergy CRN draw counts differ')
        for i, vid in enumerate(context['members']):
            if (vid in self.reverted if self.new_members_only is None
                    else vid not in self.new_members_only):
                a['action'][i] = b['action'][i]
                a['own_action'][i] = b['own_action'][i]
                ma[vid] = mb[vid]
        # These samples never enter training or any PPO ratio calculation.
        a['policy_synergy_diagnostic_only'] = True
        return a, ma


def controller_class():
    parent = sv.verify_controller_class()

    class SynergyController(parent):
        def __init__(self, policy, generator, snapshot, plan):
            from .acceleration import Policy
            cfg = snapshot['policy_synergy']
            old = Policy();old.load_state_dict(snapshot['u3_weights']);old.eval()
            if sv._state_sha(old.state_dict()) != cfg['u3_state_sha256']:
                raise ValueError('policy synergy u3 weights mismatch')
            if sv._state_sha(policy.state_dict()) != cfg['new_state_sha256']:
                raise ValueError('policy synergy new weights mismatch')
            reverted = set(cfg['group'])-set(cfg['new_members'])
            selected = cfg['new_members'] if cfg.get('background_old') else None
            super().__init__(MixedPolicy(policy, old, reverted, selected), generator, snapshot, plan)
            self.audit['policy_synergy'] = {k: v for k, v in cfg.items()}
    return SynergyController


class BranchRunner(sv.SubprocessBranchRunner):
    def run(self, spec, snapshot):
        d = self.work_root/spec['verification_id']
        if not (d/'SNAPSHOT.pt').exists():self.prepare([spec], snapshot)
        sp = d/(spec['branch']+'.spec.json');out = d/(spec['branch']+'.result.json')
        _write_new(sp, spec)
        p = subprocess.Popen([self.python, '-B', '-u', '-m', 'tools.coopmwm.policy_synergy',
            'branch', '--snapshot', str(d/'SNAPSHOT.pt'), '--spec', str(sp),
            '--out', str(out), '--work', str(self.work_root)])
        from .replay_start import owner
        with (self.work_root/'WORKERS.jsonl').open('a') as f:
            f.write(json.dumps(dict(owner(p.pid), branch=spec['branch'], verification_id=spec['verification_id']))+'\n')
        try:code = p.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            p.terminate();p.wait()
            raise RuntimeError('policy synergy own branch timed out; never retried')
        if not out.exists():raise RuntimeError(f'policy synergy branch exit {code} without result')
        v = json.loads(out.read_text())
        if v.get('prefix_mismatch'):raise sv.PrefixMismatch(v['error'])
        if code or v.get('error'):raise RuntimeError(v.get('error', f'exit {code}'))
        return v


def specs_for(snapshot, receipt, update, arm, new_weights, old_weights, isolated=False):
    group = list(receipt['chosen_plan']['members'])
    if len(group)<2 or len(set(group))!=len(group):raise ValueError('invalid cooperative group')
    if not 986000 <= snapshot['episode']['world']['seed'] <= 986071:
        raise ValueError('policy synergy requires registered TRAIN worlds')
    new_sha, old_sha = map(sv._state_sha, (new_weights, old_weights))
    out = []
    choices = [('all', group)]+[(f'solo{i}', [m]) for i, m in enumerate(group)]
    if isolated:choices += [('background', [])]
    for label, selected in choices:
        snap = dict(snapshot, weights=new_weights, u3_weights=old_weights)
        cfg = dict(group=group, new_members=selected, new_state_sha256=new_sha,
                   u3_state_sha256=old_sha, update=update, arm=arm,
                   non_group='new policy, as in lesson check', intervention_start_s=snapshot['t0']+1)
        if isolated:
            cfg.update(background_old=label!='background',
                non_group='u3 policy' if label!='background' else 'new policy; group u3')
        snap['policy_synergy'] = cfg
        # snapshot_sha binds the extra weights and the exact member intervention.
        snap['policy'] = dict(state_dict_sha256=new_sha, policy_synergy=cfg)
        spec = rb.lesson_policy_spec(snap, receipt['chosen_plan'], receipt)
        spec.pop('branch_sha256')
        prefix = 'psi' if isolated else 'ps'
        spec.update(verification_id=f'{prefix}_u{update:03}_{arm}_{receipt["verification_id"]}_{label}', branch=label)
        out.append((_seal(spec, 'branch_sha256'), snap))
    return out


def adopted_all(lesson, receipt, snapshot, new_weights):
    """Reuse only the sealed matched-window all-new lesson for this exact checkpoint/state."""
    if lesson is None:return None
    verify_sealed(lesson, 'lesson_check_sha256')
    snap = dict(snapshot, weights=new_weights, policy=dict(state_dict_sha256=sv._state_sha(new_weights)))
    for e in lesson['entries']:
        b = e.get('new_policy_branch', {})
        if (e['verification_id']==receipt['verification_id'] and e.get('status')=='graded'
                and b.get('window')==receipt['window'] and b.get('snapshot_sha256')==sv.snapshot_sha(snap)):
            return dict(status='graded', J_veh_s=e['J_new_policy_veh_s'],
                        adopted_lesson_sha256=lesson['lesson_check_sha256'], branch='all', actual_sim_s=0)
    return None


def metrics(J_u3, results, group):
    if any(v['status']!='graded' for v in results):
        return dict(status='incomplete', policy_cooperation_gain_veh_s=None,
                    reason='at least one branch ungraded; never imputed')
    deltas = [J_u3-float(v['J_veh_s']) for v in results]
    solos = dict(zip(group, deltas[1:]))
    gain = deltas[0]-max(solos.values())
    return dict(status='graded', delta_all_new_veh_s=deltas[0], delta_only_member_new_veh_s=solos,
                policy_cooperation_gain_veh_s=gain, positive_policy_cooperation=bool(gain>0))


def run_update(root, arm, update, cfg):
    root = Path(root);control = Path(cfg['control_root']);base = control/f'u{update:03}_{arm}'
    base.mkdir(parents=True, exist_ok=True)
    isolated = cfg.get('isolated', False)
    suffix = '_ISOLATED' if isolated else ''
    claim = base/'CLAIM.json';result_path = root/f'POLICY_SYNERGY_u{update:03}_{arm}{suffix}.json'
    if result_path.exists():return json.loads(result_path.read_text())
    if claim.exists():return dict(status='interrupted_claim_not_replayed', arm=arm, update=update)
    from .replay_start import owner, accounting
    accounting(cfg)
    checkpoint = root/f'{arm}_checkpoint_{update:03}.pt'
    new = torch.load(checkpoint, weights_only=False)['policy']
    old = torch.load(root/f'{arm}_checkpoint_003.pt', weights_only=False)['policy']
    sources = [cfg['collection_root']] if arm=='B' else []
    confirmed = rb.confirmations(root, arm)
    receipts = [r for r in rb.bank_decisions(root, arm, sources=sources)
        if confirmed.get(r['verification_id'], {}).get('admitted')
        and confirmed[r['verification_id']].get('cooperation', {}).get('cooperative') is True]
    if isolated:
        # Use the same update's frozen state list, not a later enlarged bank.
        legacy = json.loads((root/f'POLICY_SYNERGY_u{update:03}_{arm}.json').read_text())
        verify_sealed(legacy, 'policy_synergy_sha256')
        if legacy['checkpoint_sha256'] != sha(checkpoint):
            raise ValueError('isolated diagnostic checkpoint differs from original')
        wanted = {r['verification_id'] for r in legacy['entries']}
        receipts = [r for r in receipts if r['verification_id'] in wanted]
        if {r['verification_id'] for r in receipts} != wanted:
            raise ValueError('isolated diagnostic missing original cooperative states')
    _write_new(claim, dict(owner(), arm=arm, update=update, checkpoint_sha256=sha(checkpoint),
                          verification_ids=[r['verification_id'] for r in receipts]))
    by_vid = {v['verification_id']:v for v in
        (json.loads(p.read_text()) for p in (rb.bank_root(root,arm)/'decisions').glob('*.json'))}
    lesson_path = root/f'LESSON_CHECK_u{update:03}_{arm}.json'
    lesson = json.loads(lesson_path.read_text()) if lesson_path.exists() else None
    entries = []
    for r in receipts:
        vid = r['verification_id'];entry_path = base/(vid+'.json')
        try:
            snap = rb._decision_snapshot(root, arm, r, by_vid)
            pairs = specs_for(snap, r, update, arm, new, old, isolated=isolated)
            adopt = None if isolated else adopted_all(lesson, r, snap, new)
            runner = BranchRunner(base/'branches', workers=1)
            pending = pairs[1:] if adopt else pairs
            reference_adopt = None
            if isolated:
                reference_adopt = next(e for e in legacy['entries'] if e['verification_id']==vid)
                if (reference_adopt['status']!='graded' or reference_adopt['window']!=r['window']
                        or reference_adopt['group']!=r['chosen_plan']['members']):
                    raise ValueError('original u3 reference ungraded or different state/window/group')
            fresh_reference = not isolated and sv._state_sha(snap['weights']) != sv._state_sha(old)
            if fresh_reference:
                # Later-policy snapshots cannot reuse a policy J as a u3 label.
                ref_spec, ref_snap = specs_for(snap, r, update, arm, old, old)[0]
                ref_spec = dict(ref_spec);ref_spec.pop('branch_sha256')
                ref_spec.update(verification_id=ref_spec['verification_id']+'_u3_reference', branch='u3_reference')
                pending = pending+[(_seal(ref_spec, 'branch_sha256'),ref_snap)]
            specs = [s for s,_ in pending]
            blocked = runner.admit(specs)
            if blocked:raise RuntimeError(blocked)
            _write_new(base/(vid+'.QUOTE.json'),dict(verification_id=vid,
                branch_ids=[s['verification_id'] for s in specs],
                sim_s=sum(s['sim_s'] for s in specs),phase='bank_verification_ext',
                adopted_all=adopt is not None,fresh_u3_reference=fresh_reference))
            # Complete member set precharged atomically before the first branch.
            rb._confirm_charge(root, 'bank_verification_ext', f'policy_synergy_u{update:03}_{arm}')(vid, specs)
            results = [adopt] if adopt else []
            for spec, snapshot in pending:
                from .warm_health import assert_running
                assert_running(root,arm)
                runner.prepare([spec], snapshot)
                results.append(sv.run_branch(spec, snapshot, runner))
            reference_result=results.pop() if fresh_reference else None
            J_u3=(reference_adopt['J_u3_veh_s'] if reference_adopt else
                  reference_result['J_veh_s'] if reference_result and reference_result['status']=='graded'
                  else float(r['branches'][0]['J_veh_s']) if not fresh_reference else None)
            background = results.pop() if isolated else None
            measure=(metrics(J_u3,results,r['chosen_plan']['members']) if J_u3 is not None
                     else dict(status='incomplete',policy_cooperation_gain_veh_s=None,reason='u3 reference ungraded'))
            if isolated:
                measure['background_only_delta_veh_s'] = (J_u3-background['J_veh_s']
                    if J_u3 is not None and background['status']=='graded' else None)
                if background['status']!='graded':
                    measure.update(status='incomplete',reason='background branch ungraded')
            row = dict(verification_id=vid, group=r['chosen_plan']['members'],
                window=r['window'], reference='u3 policy, same captured state/prefix/CRN',
                J_u3_veh_s=J_u3,fresh_u3_reference=reference_result,
                plan_cooperation_gain_veh_s=confirmed[vid]['cooperation']['cooperation_gain_veh_s'],
                results=results, charged_sim_s=sum(s['sim_s'] for s in specs),
                **measure)
            if isolated:
                row.update(background_only_result=background,
                    adopted_u3_reference_sha256=legacy['policy_synergy_sha256'])
        except Exception as e:
            row = dict(verification_id=vid, status='failed', reason=repr(e), policy_cooperation_gain_veh_s=None)
        if row['status']!='graded':
            _write_new(base/(vid+'.ALERT.json'),dict(unix=time.time(),evidence=row,
                alert='policy synergy incomplete; excluded, no retry or imputation'))
        _write_new(entry_path, row);entries.append(row)
    out = dict(arm=arm, update=update, unix=time.time(), entries=entries,
        status='complete' if all(e['status']=='graded' for e in entries) else 'incomplete',
        admitted_cooperative_entries=len(receipts), checkpoint_sha256=sha(checkpoint),
        phase='bank_verification_ext', rule='delta(all new) - max delta(only member new); positive means multi-car improvement',
        background=('non-group u3 for group/solo, non-group new for background-only; captured t0 action, intervention t0+1'
                    if isolated else 'non-group new policy exactly as lesson; common captured t0 action, intervention from t0+1'),
        outcomes_opened=False)
    _write_new(result_path, _seal(out, 'policy_synergy_sha256'))
    if isolated:
        combined = dict(arm=arm, update=update, original=legacy,
            isolated=json.loads(result_path.read_text()),
            rule='MAIN12:26; original and isolated variants retained; no training change')
        _write_new(root/f'POLICY_SYNERGY_u{update:03}_{arm}_WITH_ISOLATED.json',
            _seal(combined, 'policy_synergy_sha256'))
    return out


def watch(root, cfg):
    import fcntl
    from .replay_start import atomic, owner
    from .warm_health import assert_running
    c = Path(cfg['control_root']);c.mkdir(exist_ok=True)
    lock = (c/'WATCH.claim').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    _write_new(c/'OWNER.json', owner())
    while time.time()<cfg['gpu_deadline']:
        progress = {}
        suffix = '_ISOLATED' if cfg.get('isolated') else ''
        for u in range(cfg.get('first_update',8),25):
            for arm in ('B','R'):
                key=f'{arm}{u}'
                if not (Path(root)/f'{arm}_UPDATE_{u:03}.json').exists():continue
                if not (Path(root)/f'LESSON_CHECK_u{u:03}_{arm}.json').exists():continue
                if cfg.get('isolated') and not (Path(root)/f'POLICY_SYNERGY_u{u:03}_{arm}.json').exists():continue
                try:
                    assert_running(root,arm)
                    result=run_update(root,arm,u,cfg);progress[key]=result['status']
                except Exception as e:
                    progress[key]=repr(e)
                    alert=c/f'ALERT_{key}.json'
                    if not alert.exists():_write_new(alert,dict(unix=time.time(),error=repr(e)))
        atomic(c/'STATUS.json',dict(unix=time.time(),progress=progress,training_owners_untouched=True))
        if all((Path(root)/f'POLICY_SYNERGY_u024_{a}{suffix}.json').exists() for a in ('B','R')):return
        time.sleep(30)


def main():
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=('branch','watch'))
    for name in ('root','binding','snapshot','spec','out','work'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.cmd=='watch':
        cfg=json.loads(Path(a.binding).read_text())
        assert cfg['amendment_sha256'] in (Path(a.root)/'AUTHORITY.md').read_text()
        watch(a.root,cfg);return
    spec=json.loads(Path(a.spec).read_text());snap=torch.load(a.snapshot,weights_only=False)
    original=sv.verify_controller_class
    try:
        sv.verify_controller_class=original;cls=controller_class();sv.verify_controller_class=lambda:cls
        out=sv.LocalBranchRunner(a.work).run(spec,snap);code=0
    except sv.PrefixMismatch as e:out=dict(error=str(e),prefix_mismatch=True);code=3
    except Exception as e:
        import traceback
        out=dict(error=repr(e),traceback=traceback.format_exc());code=4
    finally:sv.verify_controller_class=original
    _write_new(a.out,out)
    raise SystemExit(code)


if __name__=='__main__':main()
