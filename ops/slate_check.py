"""Guaranteed-offer slate regeneration check (CPU, zero SUMO, read-only). See clean5c/guaranteed/CRITERIA.json.

  python ops/slate_check.py <run_root> <update> <job_dir> a|b > SLATE_<job>_<mode>.json
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

torch.set_num_threads(1)


def main(root, u, job, mode):
    from tools.coopmwm import clean4_restart as c4, launch_production as production, demo_extract as dx
    from tools.coopmwm import clean5b_restart as c5b, fullpol_learn as fl
    from tools.coopmwm.bank_growth import update_view
    from tools.coopmwm.lc_families import CONFIG_KEY
    from tools.coopmwm.launch_training import read
    from tools.coopmwm.v2v_runtime import load_checkpoint
    cfg = read(Path(root)/'CONFIG.json')
    b = cfg.get('clean4_budget') or {}
    if b.get('run') in c4.APPROVED_BY_RUN:
        c4.APPROVED_BY_RUN[b['run']] = b['approved_sim_s']
    native = production.backend_config(cfg)
    native['update'] = u
    native, _ = update_view(native, 'B', u)
    if mode == 'b':
        args = native[CONFIG_KEY]['generator_args']
        args.update(classes=['verified'], structured=[], guaranteed_offer=True, slate_rule=c5b.SLATE_RULE)
    base = load_checkpoint(os.path.join(root, f'B_checkpoint_{u-1:03}.pt'), config=native)[0]
    state = torch.load(os.path.join(root, f'B_track2_{u-1:03}.pt'), map_location='cpu', weights_only=False)['worker']
    bank = {t['family'] for t in native['clean5c']['templates']}
    seconds = dx.tape_seconds(os.path.join(job, 'observable.jsonl.gz'))
    recs = dx.featurize(native, base, state, seconds, set(range(300, max(seconds)+1)))
    slates = offering = rows_offering = free = rows = 0
    for t, r in recs.items():
        if not r:
            continue
        slates += 1
        hit = False
        for table, mask in zip(r['tables'], r['mask']):
            rows += 1
            labels = [set(fl.option_labels(e)) if (e and m) else set() for e, m in zip(table, mask)]
            if any(lb & bank for lb in labels):
                rows_offering += 1
                hit = True
            free += sum(1 for e, m, lb in zip(table, mask, labels) if e and m and e.get('family', '').startswith('free_program')
                        and not lb & bank)
        offering += hit
    print(json.dumps(dict(job=os.path.basename(job), mode=mode, decision_slates=slates, slates_offering_bank=offering,
                          rows=rows, rows_offering_bank=rows_offering, nonverified_free_program_options=free,
                          bank=sorted(bank))))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
