"""Budgeted native all8 admission, with coordinator-owned dispatch and charges."""
from copy import deepcopy
import json
from pathlib import Path
import types

import torch

from . import rerun_cpus as rc
from .exact_grade import _seal


def verify(backend, payload, cfg, execute):
    from . import sumo_verify as sv, verify_perseed as vp, replay_bank as rb
    from .accel_storage import read
    path = Path(rc.guard_path(payload['snapshot_path']))
    if rc.sha(path) != payload['snapshot_file_sha256']:
        raise ValueError('REFUSED: augmentation snapshot changed')
    snapshot = torch.load(path, map_location='cpu', weights_only=False)
    if snapshot['episode']['role'] != 'train' or snapshot['episode']['world']['seed'] != payload['seed']:
        raise ValueError('REFUSED: augmentation requires source-bound TRAIN state')
    root = Path(cfg['run_root'])/'augmentation'/payload['id']
    bank = rb.bank_root(root, payload['arm'])
    plan = payload['plan']
    # Keep the complete registered original window, even when native default tails
    # would finish earlier. Extensions can only lengthen this window.
    required_tail = max(0, int(payload['window']['end_s'])-int(snapshot['t0'])-int(plan['duration_s']))
    calls = 0
    def verification(snap, candidates, directory, proposer, tail_s):
        nonlocal calls
        suffix = f'wave{calls:03}'
        calls += 1
        directory = Path(directory)
        completed = {}
        def build(snap, candidates, proposer='wm', tail_s=sv.TAIL_S):
            out = []
            for spec in sv.build_branches(snap, candidates, proposer, tail_s):
                body = {k: v for k, v in spec.items() if k != 'branch_sha256'}
                body['t2c_artifact_root'] = str(directory/'branches')
                out.append(_seal(body, 'branch_sha256'))
            return out
        def charge(vid, specs):
            # The entire native wave is reserved before its first dispatch.
            results = execute([(spec, snap) for spec in specs], suffix)
            completed.update({spec['branch']: result for spec, result in zip(specs, results)})
        class Runner:
            def run(self, spec, snap):
                result = completed[spec['branch']]
                return read(Path(rc.guard_path(result['result_path'])))
        native_verify = types.FunctionType(sv.verify.__code__, dict(sv.verify.__globals__, build_branches=build),
                                          argdefs=sv.verify.__defaults__)
        return native_verify(snap, candidates, Runner(), charge, proposer,
                         workers=1, root=directory, tail_s=max(tail_s, required_tail))
    proposer = 'random' if payload['arm'] == 'R' else 'wm'
    first_root = bank/'sumo_verification'
    receipt = verification(snapshot, [plan], first_root, proposer, sv.TAIL_S)
    if receipt['status'] not in ('accepted', 'rejected'):
        return dict(examples=[], receipt=receipt)
    # Bind the native confirmation implementation to a local dispatcher. No module
    # monkeypatches, global charge hooks or unaccounted native child dispatches.
    def confirm_verify(kind, snapshot, candidates, root_dir, *, proposer, tail_s, charges, **kw):
        r = verification(snapshot, candidates, root_dir, proposer, tail_s)
        charges.append(dict(kind=kind, receipt_sha256=r['receipt_sha256'], charged_sim_s=r['charged_sim_s'],
                            actual_sim_s=r.get('actual_sim_s'), status=r['status'],
                            phase='augment_verify', verification_id=r['verification_id'],
                            n_branches=len(r.get('branches', []))))
        return r
    confirm = types.FunctionType(vp.confirm_all8.__code__, dict(vp.confirm_all8.__globals__, _verify=confirm_verify),
                                 argdefs=vp.confirm_all8.__defaults__)
    record = confirm(root, payload['arm'], receipt, snapshot, None, cfg['run_root'], 'augment_verify',
                     workers=1, candidates=[plan], branches_dir=first_root/'branches', health=False,
                     later_harm_seed1=True, later_harm_seed2=True, slim_rejected_tapes=False)
    examples = materialise(backend, payload, cfg, record, snapshot, root)
    return dict(examples=examples, receipt=record)


def materialise(backend, payload, cfg, record, snapshot, root):
    """Use only each successful modified plan's OWN executed tape and AWR weight."""
    from . import replay_bank as rb, bank_v2v, exact_grade as xg
    from .effective_imitation import attach_limits
    from .v2v_intent import refresh_bank_intents, attach_bank_targets, settings
    from .accel_storage import read
    from .protocol import digest
    from .rerun_tracks_backend import observation
    from .exact_grade import verify_sealed
    from .sumo_verify import snapshot_sha
    from .warm_protocol import SEED
    temperature = rb.temperature(Path(cfg['run_root']), payload['arm'], SEED,
        rb.bank_decisions(cfg['run_root'], payload['arm'], sources=cfg.get('replay_sources', {}).get(payload['arm'], ())))
    out = []
    for ex in record['per_seed_v1']['examples']:
        if not ex['admitted'] or ex.get('cooperative') is not True:
            continue
        tape_ref = ex['tape']
        verification_path = Path(tape_ref['branches_dir']).parent/'verifications'/(tape_ref['verification_id']+'.json')
        rc.guard_path(verification_path.resolve())
        verification = json.loads(verification_path.read_text(encoding='utf-8'))
        verify_sealed(verification, 'receipt_sha256')
        d = Path(tape_ref['branches_dir'])/tape_ref['verification_id']/tape_ref['branch']['branch']
        rc.guard_path(d.resolve())
        tape, trace = read(d/'RESULT.json'), d/'observable.jsonl.gz'
        if digest(tape) != tape_ref['branch'].get('result_sha256'):
            raise ValueError('REFUSED: augmented execution tape differs from all8 receipt')
        plan = ex['plan']
        weight = xg.weight(ex['advantage_veh_s'], temperature['temperature_veh_s'], xg.WEIGHT_CLIP)
        entry = dict(start_s=snapshot['t0'], duration_s=plan['duration_s'], plan_id=plan['id'], weight=weight)
        pairs = xg.imitation_pairs(tape, entry)
        pairs, _, limits = attach_limits(pairs, tape, trace, entry)
        s = settings(cfg)
        pairs = attach_bank_targets(pairs, tape, entry, s['horizon'], s['bounds'], limits=limits)
        pairs, _ = bank_v2v.attach_pairs(pairs, trace, snapshot['t0'], None,
            intent=dict(horizon=s['horizon'], entry=tape_ref['intent_entry']))
        refresh_bank_intents(backend.models[payload['arm']], pairs)
        if not pairs:
            continue
        frames = bank_v2v.trace_frames(trace)
        # A decision target uses the first actually executed effective sample.
        sample, _ = pairs[0]
        t = int(sample['intent_link']['t'])
        obs = observation(frames[t]['vehicles'], frames.get(t-1), t, snapshot['sumo']['accel_bounds'], ego=plan['members'][0])
        features = backend.components[payload['arm']]['merge_features'].encode(obs)
        group = plan['members']
        indices = [sample['context']['members'].index(v) for v in group]
        effective = dict(action=[float(sample['action'][i][0]) for i in indices],
            binding_limits={v: sample['binding_limits'][v] for v in group},
            behaviour=dict(executed=True, proposal_members=group),
            context=dict(members=group, observation=obs, features=features), native_sample=sample,
            plan_sha256=_seal(dict(plan=plan), 'sha256')['sha256'])
        snap = rb.replicate(snapshot, record['verification_id']) if ex['seed_index'] == 2 else snapshot
        bank = rb.bank_root(root, payload['arm'])
        (bank/'snapshots').mkdir(parents=True, exist_ok=True)
        original_path = bank/'snapshots'/'source.pt'
        if not original_path.exists():
            torch.save(snapshot, original_path)
        own_path = bank/'snapshots'/f'seed{ex["seed_index"]}.pt'
        torch.save(snap, own_path)
        effective_sha = _seal(dict(sample=effective), 'sha256')['sha256']
        weight_sha = _seal(dict(weight=weight, example=ex), 'sha256')['sha256']
        e = dict(payload, id=payload['proposal_id']+f'_s{ex["seed_index"]}', plan=plan,
            weight=weight, observation=obs, effective_sample=effective, statistics='per_seed_v1',
            snapshot_identity_sha256=snapshot_sha(snap),
            snapshot_path=str(own_path), snapshot_file_sha256=rc.sha(own_path), practice_root=str(root),
            practice_row=dict(verification_id=record['verification_id'], example_id=ex['example_id'],
                seed_index=ex['seed_index'], salt=record['verification_id'], state_id='source', fallback=False),
            provenance=dict(verification_sha256=record['confirmation_sha256'],
                            weight_sha256=weight_sha, effective_sha256=effective_sha))
        e['augmentation_admission'] = dict(confirmation=record, verification_receipt=verification,
            example_id=ex['example_id'], seed=e['seed'],
            snapshot_file_sha256=e['snapshot_file_sha256'], effective_sha256=effective_sha, weight_sha256=weight_sha)
        out.append(e)
    return out
