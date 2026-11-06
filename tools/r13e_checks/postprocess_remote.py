"""After the three-arm study: same-event gs differential and same-panel round-9 BASE."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/c4/v5')
REPO = ROOT/'repo_r13e_8c7ff0a1'
WORK = ROOT/'diag_emerg'
OUT = WORK/'offline_train48_workers4'
sys.path.insert(0, str(REPO))
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))


def main():
    # One idle coordinator; no additional numerical worker while the main Pool runs.
    receipt = json.loads((WORK/'offline_owned_process.json').read_text())
    while not (OUT/'report.json').exists():
        stat = Path(f"/proc/{receipt['pid']}/stat")
        if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0] == 'Z':
            raise RuntimeError('study exited without report; no substitute results')
        time.sleep(15)
    # The report is written only after both Pools have joined.
    from tools.r13e_checks import offline as study
    study.cpu_limits()
    import numpy as np
    import torch
    from tools.r13e_checks import gs_equivalence as diff
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles

    docs, worlds = study.train_manifests([ROOT/'data/v5_native', ROOT/'data/genrand_plain'])
    records = study.load_train_states(WORK/'train_states48.json.gz', worlds)
    predictor, capture, weights = study.load_exact_predictor(ROOT/'runs/c4_wm_v10_round12c/wm_s.pt')
    reference = diff.module_at(WORK/'gapacc_ref/factworker.py', 'gapacc_reference')
    extractor = diff.module_at(WORK/'gapacc_ref/ins.py', 'gapacc_events')
    cases = []
    captured = {}
    original_forward = predictor.model.forward

    def forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        captured['dp'] = out['driver_parameters'].detach().cpu().numpy()[0]
        return out

    predictor.model.forward = forward
    for index, record in enumerate(records):
        study.seed_all(study.RNG_SEED+2*index)
        predictor.predict_states(record, {'arm': record['arm']}, .5, study.RNG_SEED+2*index)
        dp = captured['dp']
        t0, ids, kin, cav = t0_vehicles(record)
        teacher, present, _ = recorded_teacher(record, t0, ids, kin, 240, record['arm'], predictor.backend.tools)
        traj = teacher[0].numpy()
        events = extractor.events(traj[...,0], traj[...,1], present[0].numpy(), traj[...,2], cav)
        for event in events:
            r, f, leader = [int(event[k]) for k in ('r','f','l')]
            c = diff.case(f"state{index}:row{event['tm']}:slot{r}",
                          v=float(event['vR']), vf=float(event['vF']), vl=float(event['vL']),
                          gf=float(event['gf']), gl=float(event['gl']), has_f=f>=0, has_l=leader>=0,
                          dp=dp[[r,max(f,0),max(leader,0)]].copy())
            c['hdv'] = bool(event['hdvR'])
            cases.append(c)
    predictor.model.forward = original_forward
    assert study.state_digest(predictor.model.state_dict()) == weights
    gs = {'verdict':'NOT_EQUIVALENT', 'same_event_inputs':True, 'recorded_events':len(cases),
          'parameter_source':'S-R12C encoder, exact pre-t0 history, captured first forward; identical per-slot T,b,s0 supplied to both gates',
          'event_source':'same 48 TRAIN snapshots, recorded lane2->1 post-insertion geometry, both methods on every event',
          'reference_sha256':study.file_digest(WORK/'gapacc_ref/factworker.py'),
          'extractor_sha256':study.file_digest(WORK/'gapacc_ref/ins.py'),
          'definition_difference':'R13E adds rear learned min_gap; GAPACC gs has no min_gap term',
          'events':{}}
    for name, selected in [('all',cases),('hdv',[c for c in cases if c['hdv']])]:
        gs['events'][name] = {str(dtype):diff.compare(reference,selected,dtype,predictor.model.merge_pivot)
                              for dtype in (torch.float32,torch.float64)} if selected else {}
    gs_text = json.dumps(study.json_safe(gs),indent=2,allow_nan=False)+'\n'
    gs_path = OUT/'gs_equivalence.json'
    if gs_path.exists():
        assert gs_path.read_text() == gs_text, 'same-event gs evidence must reproduce exactly'
    else:
        gs_path.write_text(gs_text)
    print('GS_COMPLETE',json.dumps(study.json_safe(gs)),flush=True)
    # Independent strict round-9 AS-IS comparator, SAME 48 replay windows.
    from tools.wm_eval_common import load_predictor
    from wms.config.load import load_train_config
    from tools.r12_checks import base_calibration as base
    checkpoint = study.checked_path(ROOT/'runs/c4_wm_v10_round9/wm_s_stage300.pt')
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    r9, _ = load_predictor(str(checkpoint),'AS_IS',device='cpu',threads=1,hdv_exits='none',entries='off')
    r9.model.eval().requires_grad_(False)
    assert study.state_digest(r9.model.state_dict()) == study.state_digest(payload['model']), 'round9 strict bytes'
    original_r9_forward = r9.model.forward

    def r9_replay_metadata(*args, **kwargs):
        # R9 predates courtesy. Supply the BASE collector's descriptive fields
        # without adding a network, changing acceleration, or altering any tensor.
        out = original_r9_forward(*args, **kwargs)
        out.setdefault('accel_pre_courtesy_mps2', out['accel_mps2'])
        out.setdefault('courtesy_accel_mps2', torch.zeros_like(out['accel_mps2']))
        return out

    r9.model.forward = r9_replay_metadata
    cfg = load_train_config(REPO/'configs/c4_wm_v10round12c.toml',['runtime.device=cpu'])
    sets = study.train_datasets(cfg,docs)
    original_arms = study.ARMS
    study.ARMS = {'off':original_arms['off']}
    try:
        packed, cols, windows = study.run_onestep(r9,cfg,sets,workers=4,records=records)
    finally:
        study.ARMS = original_arms
    reference_dir = OUT/'round9_reference'
    reference_dir.mkdir()
    study.save_onestep(reference_dir, packed, cols)
    data = {'r9':{c:packed['off'][:,i].astype(float) for i,c in enumerate(cols)}}
    del packed
    for arm in original_arms:
        data[arm] = base.load_rows(str(OUT),arm)
    stats, encoded = {}, {}
    for arm, rows in data.items():
        base.A.RNG = np.random.default_rng(study.RNG_SEED)
        stats[arm],encoded[arm] = base.class_stats(rows,1000),base.encoded_params(rows)
        for name, selected in base.row_classes(rows).items():
            finite = selected & np.isfinite(rows['fin']-rows['rec'])
            clusters = len(np.unique(rows['win'][finite]))
            stats[arm][name]['clusters'] = clusters
            if clusters < 2:
                stats[arm][name]['lo'] = stats[arm][name]['hi'] = float('nan')
    verdicts = {}
    for arm in original_arms:
        rules = base.decide({'r9':stats['r9'],'r10b':stats[arm],'headzero':stats[arm]},
                            {'r9':encoded['r9'],'r10b':encoded[arm],'headzero':encoded[arm]})
        verdicts[arm] = {'overall':base.overall(rules),'rules':rules}
    contrasts = {}
    for contrast,(later,earlier) in study.CONTRASTS.items():
        contrasts[contrast] = {}
        for name,select in base.row_classes(data[earlier]).items():
            errors = [(data[a]['fin']-data[a]['rec'])[select] for a in (earlier,later)]
            contrasts[contrast][name] = {'bias':study._estimate(errors[1]-errors[0],data[earlier]['win'][select]),
                                        'mae':study._estimate(np.abs(errors[1])-np.abs(errors[0]),data[earlier]['win'][select])}
    result = {'round9_checkpoint_sha256':study.file_digest(checkpoint),'round9_strict_exact_weights':True,
              'round9_collector_metadata':'no courtesy network in R9: pre=unchanged final accel, courtesy=zeros; no tensor or accel modification',
              'round9_npz_sha256':study.file_digest(reference_dir/'onestep_off.npz'),
              'one_step_windows':windows,'classes':stats,'encoded':encoded,'verdicts':verdicts,
              'contrasts':contrasts,'bootstrap_draws':1000,'replay_present':'filled'}
    (OUT/'base_checks.json').write_text(json.dumps(study.json_safe(result),indent=2,allow_nan=False)+'\n')
    print('BASE_COMPLETE',json.dumps(verdicts),flush=True)
    # Linux-specific fork tests run only after numerical study workers finish.
    env = dict(os.environ,PYTHONPATH=str(REPO/'src')+':'+str(REPO),PYTHONDONTWRITEBYTECODE='1')
    command = ['/root/miniconda3/bin/python','-B','-m','pytest','-q','-p','no:cacheprovider','--noconftest',
               'tests/test_r13e_offline.py']
    run = subprocess.run(command,cwd=REPO,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (OUT/'linux_tests.log').write_text(run.stdout)
    print('LINUX_TESTS',run.returncode,run.stdout,flush=True)
    assert run.returncode == 0
    (OUT/'POSTPROCESS_COMPLETE.json').write_text(json.dumps({'status':'complete','linux_tests_exit':run.returncode})+'\n')


if __name__ == '__main__':
    main()
