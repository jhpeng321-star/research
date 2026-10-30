"""Validation-only, mix-ON scorer with an explicit default-OFF lane switch."""
import os
import sys
from pathlib import Path

OWN = Path('/root/autodl-tmp/c4/wmrequal3_20261030')
OLD = Path('/root/autodl-tmp/c4/wmrequal20261030')
RT = Path('/root/autodl-tmp/c4/restart20261026/runtime_longint')


def bootstrap():
    sys.path[:0] = [str(OWN/'tools/wmrequal'), str(OWN), str(OWN/'src'), str(RT)]
    import score
    # score.py retains its historical default paths for exact OFF users.
    sys.path[:0] = [str(OWN), str(OWN/'src'), str(OWN/'tools/wmrequal'), str(RT/'src'), str(RT)]
    score.OUT = OWN
    return score


def main():
    import argparse
    import hashlib
    import json
    import time
    import subprocess
    from dataclasses import replace
    ap = argparse.ArgumentParser()
    ap.add_argument('--native-lanes', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--budget-mb', type=int, default=11000)
    args = ap.parse_args()
    # Guard before importing any data loader; filenames/assignment files refused.
    bad = ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
    def guard(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            name = os.fsdecode(values[0]).lower()
            if any(x in name for x in bad) or name.endswith('.xml'):
                raise RuntimeError('prohibited input path')
    sys.addaudithook(guard)
    score = bootstrap()
    import torch
    torch.set_num_threads(2)
    assert args.budget_mb <= 11000
    total = torch.cuda.get_device_properties(0).total_memory
    # Leave >=1 GB for context/libraries, in addition to capped allocator.
    torch.cuda.set_per_process_memory_fraction(args.budget_mb*1e6/total)
    e3 = score.bootstrap_runtime()
    ckpt = score.BASE/'restart20261026/arm/loop/wm/c02/wm_s.pt'
    cfg = RT/'configs/c4_wm_loop14_r16_ft.toml'
    assert score.digest(ckpt) == 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
    _, settings, model = e3.load_for_scoring(cfg, ckpt)
    design = json.loads((score.BASE/'overnight20261028/W5_COMBINED/DESIGN.json').read_text())
    controllers = {a: design['controllers'][a] for a in ('mappo','ours_c1','coopoff_c1')}
    settings = replace(settings, seed=140014, entrant_control=True, entrant_controllers=controllers,
                       entry_queue_state=True, entry_pad_auto=True, entry_pad_max=1024,
                       entry_insert_rule='sumo_like_v2')
    import wms.control.entrant as entrant
    from grouped_feedback import feedback
    entrant.feedback_from_specs = feedback
    model = model.to('cuda').eval().requires_grad_(False)
    model.cav_exec_cap = 'idm'
    rows = json.loads((OWN/'REAL.json').read_text())['panels']['validation']
    source, indexes = score.source_for('validation', rows)
    assert len(indexes) == 36
    steps = 20 if args.smoke else 600
    if args.smoke:
        indexes = indexes[:1]
        score.HORIZONS = (10,)
    target = OWN/('smoke' if args.smoke else 'validation')
    assert target.parent.is_dir()
    target.mkdir(exist_ok=True)
    receipt = dict(pid=os.getpid(), wm_sha256=score.digest(ckpt), config_sha256=score.digest(cfg),
        source_sha256=score.digest(OWN/'SOURCE.json'), real_sha256=score.digest(OWN/'REAL.json'),
        K=1 if args.smoke else 8, steps=steps, physical_exits=True, queue_state=True,
        mix='ON', native_lanes=args.native_lanes, allocator_budget_bytes=args.budget_mb*1000000,
        total_owned_budget_bytes=12000000000, source_receipt=source.receipt,
        futures=dict(driver_scale=1.,embedding_std=.05,accel_noise=False))
    with (target/f'RUN_{os.getpid()}.json').open('x') as f:
        json.dump(receipt, f, indent=1)
    print('OWN_PID',os.getpid(),'groups',len(indexes),'steps',steps,flush=True)
    for i, gi in enumerate(indexes):
        dest = target/f'batch_{i:03d}.json'
        if dest.exists():
            saved = json.loads(dest.read_text())
            assert all(saved['receipt'][k] == receipt[k] for k in receipt if k != 'pid')
            continue
        # Pilots retain priority: before each world wait >=10 minutes if our
        # allocator budget plus physical reserve no longer fits beside others.
        while True:
            free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1048576
            if free + torch.cuda.memory_reserved() >= args.budget_mb*1000000 + 1000000000:
                break
            print('WAIT_HEADROOM',i,free,flush=True)
            torch.cuda.empty_cache()
            time.sleep(600)
        begin = time.monotonic()
        result = score.score_batch(model, settings, e3, source, [gi], rows,
            torch.device('cuda'), steps, args.smoke, native_lanes=args.native_lanes, mix_modes=(True,))
        memory = dict(allocated_peak=torch.cuda.max_memory_allocated(), reserved_peak=torch.cuda.max_memory_reserved())
        with dest.open('x') as f:
            json.dump(dict(receipt=receipt,rows=result,seconds=time.monotonic()-begin,memory=memory), f, allow_nan=False)
        print('DONE_BATCH',i,round(time.monotonic()-begin,2),memory,flush=True)
    with (target/'DONE.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),groups=len(indexes),steps=steps), f)


if __name__ == '__main__':
    main()
