"""E2 stage 3: whole-episode contrasts of the original bank F against its matched controls (zero SUMO).

Input: E2_MANIFEST.json (e2_build), the bank files, and E2M_RESULTS.json (box_driver collect --kind e2main):
    dict(u0={world: J}, banks={bank sha: {world: J}}, arms={arm: bank sha}, acts={bank sha: {world: [act]}}, ...)

Output: every arm - U0, F - S (timing: staggered onsets vs synchronised copy), F - H<v> for each common speed
(resource-matched uniform hold), all by the registered paired percentile bootstrap (diff = treatment - reference,
negative = less delay), plus the ACTUAL control effort of each arm from the executor receipts (hold vehicle-seconds,
unique CAVs, activations, full / truncated groups) so 'resource matched' is checked on what was executed, not only
on the descriptors.

REGISTERED E2 PRIMARY (with --selection E2_SELECTION.json): total delay F - H* on the confirmation worlds, where
H* is the one uniform-hold speed chosen on the Development worlds by e2_select BEFORE any confirmation episode
(paired percentile bootstrap, two-sided p). Only F, S and H* are run on the confirmation block.
Without --selection (all five H arms on one block) 'F beats the uniform hold' is the worst case over the speeds.
`--primaries name=p ...` applies Holm to the named p-values (the E1 and the E2 primary).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import stats as S  # noqa: E402


def effort(acts_by_world, specs):
    rows = [S.control_effort(a, specs) for a in acts_by_world.values()]
    n = len(rows)
    keys = ('activations', 'hold_vehicle_seconds', 'unique_cavs', 'full_groups', 'truncated_groups')
    return dict(n_worlds=n, per_episode_mean={k: sum(r[k] for r in rows)/n for k in keys},
                total={k: sum(r[k] for r in rows) for k in keys})


def analyse(manifest, res, specs, selected=None):
    worlds = sorted(res['u0'])
    arms = dict(res['arms'])

    def J(arm):
        src = res['u0'] if arm == 'U0' else res['banks'][arms[arm]]
        if sorted(src) != worlds:
            raise L.Refused(f'{arm}: worlds differ from U0')
        return [float(src[w]) for w in worlds]
    hs = [selected] if selected else [f'H{v}' for v in manifest['v_common']]
    for need in ['F', 'S']+hs:
        if need not in arms:
            raise L.Refused(f'results lack arm {need}')
    out = dict(convention='diff = treatment - reference; negative = less delay', n_worlds=len(worlds),
               vs_u0={a: S.paired_bootstrap(J(a), J('U0')) for a in ['F', 'S']+hs},
               F_minus_S=S.paired_bootstrap(J('F'), J('S')),
               F_minus_H={h: S.paired_bootstrap(J('F'), J(h)) for h in hs})
    worst = max(hs, key=lambda h: out['F_minus_H'][h]['ci95'][1])
    out['F_minus_H_worst_case'] = dict(arm=worst, ci95_upper=out['F_minus_H'][worst]['ci95'][1],
                                       F_better_than_every_H=bool(out['F_minus_H'][worst]['ci95'][1] < 0),
                                       p_two_sided_max=max(out['F_minus_H'][h]['p_two_sided'] for h in hs))
    if selected:
        out['primary'] = dict(contrast=f'F - {selected} total delay (confirmation worlds)', arm=selected,
                              **out['F_minus_H'][selected])
    if res.get('acts'):
        out['control_effort'] = {a: effort(res['acts'][arms[a]], specs) for a in ['F', 'S']+hs}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--bank', required=True, help='the original bank (specs of F)')
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--results', required=True, help='E2C_RESULTS.json (collect --kind e2main on block conf)')
    p.add_argument('--selection', help='E2_SELECTION.json (e2_select): the registered primary F - H*')
    p.add_argument('--primaries', nargs='*', default=[], help='name=p pairs for the Holm adjustment')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    man = L.verify_seal(L.read_json(Path(a.out)/'e2'/'E2_MANIFEST.json'))
    specs = {x['family']: x['strategy'] for x in L.bank_strategies(L.read_json(a.bank))}
    for name in man['banks']:
        specs.update({x['family']: x['strategy']
                      for x in L.bank_strategies(L.read_json(Path(a.out)/'e2'/'banks'/name))})
    sel = L.verify_seal(L.read_json(a.selection)) if a.selection else None
    if sel is not None and sel['manifest_sha256'] != man['doc_sha256']:
        raise L.Refused('selection belongs to another manifest')
    body = analyse(man, L.read_json(a.results), specs, sel['selected_arm'] if sel else None)
    if sel is not None:
        body['selection_sha256'] = sel['doc_sha256']
    if a.primaries:
        ps = {k: float(v) for k, v in (x.split('=') for x in a.primaries)}
        body['holm'] = dict(raw=ps, adjusted=S.holm(ps))
    doc = L.seal(dict(schema='claim_validation.e2.analysis.v1', run_id=L.RUN_ID, manifest_sha256=man['doc_sha256'],
                      results_sha256=L.sha_file(a.results), **body))
    state = 'dry-run' if a.dry_run else L.write_once(Path(a.out)/'e2'/'E2_ANALYSIS.json', L.text_of(doc))
    print(L.text_of(dict(stage='e2_analyse', dry_run=a.dry_run, file=state, F_minus_S=doc['F_minus_S'],
                         F_minus_H_worst_case=doc['F_minus_H_worst_case'], primary=doc.get('primary'))), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
