"""E2 stage 2: choose the ONE uniform-hold speed H* on the Development worlds and write the confirmation waves
(zero SUMO; reads only the Development tuning results, never a confirmation outcome).

Input : E2_MANIFEST.json (e2_build --tune-worlds N) and E2T_RESULTS.json
        (box_driver collect --kind e2main on block `dev`: dict(u0={world: J}, banks={sha: {world: J}}, arms={arm: sha})).
Rule  : REGISTERED BEFORE THE TUNING RUN. H* = the common speed whose bank has the LOWEST mean total delay over the
        Development worlds (i.e. the strongest resource-matched uniform hold = the hardest control for F);
        exact ties -> the lower speed. Nothing else is tuned; F and S are not run on the Development worlds here.
Output: E2_SELECTION.json with the table of the five speeds and `confirm_waves` on the shared confirmation block
        `conf`: CONF_U0 (the same wave E1 uses), E2C_F (the original bank, byte-identical), E2C_S, E2C_<H*>.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402

CONFIRM_BLOCK = 'conf'
RULE = ('H* = the common speed with the lowest mean total delay on the Development worlds (strongest uniform hold); '
        'exact ties -> the lower speed')


def select(manifest, res):
    worlds = sorted(res['u0'])
    if manifest.get('tune_worlds') and len(worlds) != int(manifest['tune_worlds']):
        raise L.Refused(f'tuning results have {len(worlds)} worlds, the manifest registered {manifest["tune_worlds"]}')
    by_arm = {v['arm']: v['sha256'] for v in manifest['banks'].values()}
    table = []
    for v in manifest['v_common']:
        arm = f'H{v}'
        if res['arms'].get(arm) != by_arm[arm]:
            raise L.Refused(f'{arm}: tuning results are not for the manifest bank')
        src = res['banks'][by_arm[arm]]
        if sorted(src) != worlds:
            raise L.Refused(f'{arm}: worlds differ from U0')
        mean = sum(float(src[w]) for w in worlds)/len(worlds)
        table.append(dict(arm=arm, v_common=v, mean_total_delay=mean,
                          mean_minus_u0=mean-sum(float(res['u0'][w]) for w in worlds)/len(worlds)))
    best = min(table, key=lambda r: (r['mean_total_delay'], r['v_common']))
    return best['arm'], table


def confirm_waves(manifest, arm, n_worlds):
    ob = manifest['original_bank']
    waves = [dict(wave='CONF_U0', kind='U0', block=CONFIRM_BLOCK, n_worlds=n_worlds),
             dict(wave='E2C_F', kind='BANK', block=CONFIRM_BLOCK, n_worlds=n_worlds, bank=ob['name'],
                  bank_sha256=ob['sha256'], arm='F')]
    by_arm = {b['arm']: (name, b) for name, b in manifest['banks'].items()}
    for name, b in (by_arm[x] for x in ('S', arm) if x in by_arm):     # fixed order (the sealed file is key-sorted)
        waves.append(dict(wave=f'E2C_{b["arm"]}', kind='BANK', block=CONFIRM_BLOCK, n_worlds=n_worlds, bank=name,
                          bank_sha256=b['sha256'], arm=b['arm']))
    if [w.get('arm') for w in waves[1:]] != ['F', 'S', arm]:
        raise L.Refused('manifest lacks the S bank or the selected H bank')
    return waves


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--results', required=True, help='E2T_RESULTS.json (Development tuning)')
    p.add_argument('--confirm-worlds', type=int, required=True, help='size of the shared confirmation block (96)')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    man = L.verify_seal(L.read_json(Path(a.out)/'e2'/'E2_MANIFEST.json'))
    arm, table = select(man, L.read_json(a.results))
    w = confirm_waves(man, arm, a.confirm_worlds)
    doc = L.seal(dict(schema='claim_validation.e2.selection.v1', run_id=L.RUN_ID, manifest_sha256=man['doc_sha256'],
                      tuning_results_sha256=L.sha_file(a.results), rule=RULE, table=table, selected_arm=arm,
                      confirm_waves=w, sim_s=dict(new=3*a.confirm_worlds*L.HORIZON_S,
                                                  u0_shared_with_e1=a.confirm_worlds*L.HORIZON_S)))
    state = 'dry-run' if a.dry_run else L.write_once(Path(a.out)/'e2'/'E2_SELECTION.json', L.text_of(doc))
    print(L.text_of(dict(stage='e2_select', dry_run=a.dry_run, file=state, selected_arm=arm, table=table,
                         doc_sha256=doc['doc_sha256'])), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
