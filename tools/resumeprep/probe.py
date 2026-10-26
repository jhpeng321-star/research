"""Matched, bounded real-checkpoint policy and WM timing probes; never SUMO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--returns', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    runtime, base = Path(a.runtime), Path(a.base)
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                                     '--format=csv'], text=True)
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    sys.path[:0] = [str(runtime / 'src'), str(runtime)]
    os.chdir(runtime)
    os.environ.update(WMS_SPEED_PIVOT_DIAG_OFF='1', WMS_SPEED_GRAD_AUDIT_EVERY='10',
                      C4_PROBES_DEVICE='cpu', C4_ALLOWED_ROOTS=str(base)+':'+str(out)+':'+str(Path(a.returns).parent))
    from wms.control import policy14_cli as cli
    # Process-local synthetic sampler injection; the CLI's --smoke intentionally
    # refuses CUDA. No production loader or validation source file is changed.
    cli._load_starts = lambda raw, models, seed, entries: (cli.SmokeSampler(models[0]),cli.SmokeSampler(models[0]))
    cli.check_train_seeds = lambda sampler: None
    # Synthetic starts, real round16, unchanged policy package/config. Gates remain
    # enabled, but no frozen *real* panel is passed to the synthetic sampler.
    torch.cuda.synchronize()
    start = time.perf_counter()
    rc = cli.main(['--config', 'configs/c4_policy14_main14_nativelane.toml', '--real-returns', a.returns,
                 '--wm-ckpt', str(base/'wm_s_best.pt'), '--wm-config', str(base/'config.resolved.json'),
                 '--device', 'cuda', '--seed', '965140', '--compact',
                 '--max-updates', '4', '--output-dir', str(out/'policy')])
    torch.cuda.synchronize()
    summary = json.loads((out/'policy/RUN_SUMMARY.json').read_text())
    rows = summary.get('updates', [])
    receipt = {'nvidia_smi_before': query, 'checkpoint_sha256': hashlib.sha256((base/'wm_s_best.pt').read_bytes()).hexdigest(),
               'policy_config_sha256': hashlib.sha256((runtime/'configs/c4_policy14_main14_nativelane.toml').read_bytes()).hexdigest(),
               'policy_exit': rc, 'policy_total_s': time.perf_counter()-start, 'policy_updates': rows,
               'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
               'workload': 'synthetic starts; real round16; full policy update, 10s; not real-start throughput'}
    (out/'POLICY_TIMING.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt), flush=True)
    # Matched short WM probe uses the *loop FT recipe*, checkpoint and frozen
    # prefixes; synthetic 96-slot/24-present recordings bound data/memory costs.
    # This is an explicit workload qualifier, never production-throughput evidence.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    from wms.config.load import load_train_config
    from wms.train.settings import TrainSettings
    from wms.train.losses.compose import build_loss
    from wms.train.unroll import UnrollRunner
    from tools.r10_attribution import load_model
    from tools.speed2_measure import scene
    cfg = load_train_config(runtime/'configs/c4_wm_loop14_r16_ft.toml')
    settings = TrainSettings.from_config(cfg)
    _, _, model, report = load_model(str(base/'config.resolved.json'), str(base/'wm_s_best.pt'))
    assert not report['new'] and not report['unused'], report
    model.to('cuda').train()
    for name, param in model.named_parameters():
        param.requires_grad_(not any(name.startswith(x) for x in cfg.optim.freeze_prefixes))
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device('cuda'))
    optimizer = torch.optim.Adam([x for x in model.parameters() if x.requires_grad], lr=3e-5)
    kin, present, cav = scene('cuda', k=96, p=24)
    steps = 240
    kin, present, cav = kin.expand(4,-1,-1), present.expand(4,-1), cav.expand(4,-1)
    rec = kin[:,None].expand(-1,steps+1,-1,-1).clone()
    rec[...,0] += torch.arange(steps+1,device='cuda')[None,:,None]*.5*kin[:,None,:,4]
    batch = dict(states=rec, present=present[:,None].expand(-1,steps+1,-1),
                 is_cav=cav[:,None].expand(-1,steps+1,-1))
    batch.update(driver_hist=torch.zeros(4,96,61,9,device='cuda'),
                 driver_hist_mask=present[:,:,None].expand(-1,-1,61),
                 hist=torch.zeros(4,96,10,6,device='cuda'),hist_mask=present[:,:,None].expand(-1,-1,10))
    batch['driver_hist'][...,0]=.5
    batch['driver_hist'][...,2]=1.
    os.environ.update(WMS_SPEED_GRAD_AUDIT_EVERY='1', WMS_SPEED_FAST_CAV_DRIVE='1', WMS_SPEED_FAST_LONG_HISTORY='1')
    times=[]
    for i in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(743)
        torch.cuda.synchronize()
        start=time.perf_counter()
        metrics=runner.run(batch, unroll_steps=steps, backward=True)
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        print(json.dumps({'wm_step':i,'seconds':times[-1],'loss':metrics['loss']}),flush=True)
    result={'warmup_s':times[0],'measured_step_s':times[1], 'checkpoint_sha256':receipt['checkpoint_sha256'],
            'wm_config_sha256':hashlib.sha256((runtime/'configs/c4_wm_loop14_r16_ft.toml').read_bytes()).hexdigest(),
            'workload':'synthetic recorded kinematics; B4 K96 P24,120s, same FT settings/frozen prefixes; not production-data throughput',
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
    (out/'WM_TIMING.json').write_text(json.dumps(result,indent=1))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
