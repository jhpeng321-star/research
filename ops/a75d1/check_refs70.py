"""A75-D1 confirm pieces, zero SUMO: the stored A70 MAPPO60C / DUO_HOLD_60C waves bind from the confirm tree.
usage: check_refs70.py <a70_root> <a66_root> <ref70_reports>      (run from the tree's top directory)"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import a75_directhold as B  # noqa: E402


def main(a70_root, a66_root, ref70):
    D, panel, D70 = B.setup('c')
    A70 = B._a70()
    worlds = [r['world_id'] for r in panel['worlds']]
    sel = A70.selection(a70_root)
    pins = B.ref_pins()
    out = dict(panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), runner_ranges=B.RANGES['c'])
    for arm in B.REF_ARMS:
        have, src = D70.C37._load(a70_root, f'A70C_{arm}', A70.confirm_specs(D70, arm, panel, a70_root, a66_root, sel))
        pin = pins[f'a70c_{arm.lower()}']
        got = src['receipt_sha256']
        if (got if isinstance(got, list) else [got]) != pin['wave_receipt_sha256']:
            raise SystemExit(f'A70C_{arm}: not the pinned wave receipt')
        reps = B.pinned_reports(ref70, have, worlds, pin)
        out[arm] = dict(n=len(have), receipt=pin['wave_receipt_sha256'][0][:16], reports_ok=all(x['ok'] for x in reps.values()),
                        mean_J_veh_s=sum(have[w]['J_veh_s'] for w in worlds)/len(worlds))
    print(json.dumps(out))


if __name__ == '__main__':
    main(*sys.argv[1:4])
