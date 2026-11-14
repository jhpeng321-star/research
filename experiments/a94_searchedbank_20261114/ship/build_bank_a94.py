"""A94/A95 build_bank_a94 (zero SUMO, deterministic): the BANK_SEARCHED and BANK_HAND bank files.

usage (inside the A94 tree, env sourced):
  python -I build_bank_a94.py searched --s5-bank F --s4-bank F --out-dir D
  python -I build_bank_a94.py hand --hand ROOT=SHA [--hand ROOT=SHA] --out-dir D
    ROOT = an admission final root (ROOT/a80_banks/A80_BANK_D2800_p60.json + ROOT/A80_ADMISSION_D2800_p60.json);
    SHA  = the expected sha256 of that root's bank file, or "none" when that root admitted no plan.

Rules (Amendment 95, registered 2026-11-14 02:09; overrides Amendment 94 where they differ):
  searched: BANK_SEARCHED = the rows (and templates) of the S5 bank (577a0564...) followed by those of the S4 bank
    (a428cd1d...), both pinned, COPIED UNCHANGED (canonical-JSON equal), duplicates by identical strategy key removed
    (first occurrence kept). Executor priority = the executor's own a75_bank_plans order from the carried fields.
  hand, one root: BANK_HAND = that root's bank file rows unchanged (the output bytes are re-serialized with the source
    serializer and checked byte-equal to the source file).
  hand, two roots: the admission code's own BANK_RULE (a75_d2forms.bank_choice, unchanged, with the a80 FORMS and
    MAX_BANK = 4) applied to the union of both receipts' verdict rows: per form family the admitted (PASS),
    non-duplicate plan with the most negative admission CI95 upper bound of (plan - MAPPO60C) total delay, verdict
    field ci95[1] (= -lb95_veh_s carried in the bank row), ties: strategy key; then the (at most) 4 family bests with
    the most negative CI95 upper bound. Duplicate detection (identical 24 per-seed results) follows the code's order
    (draw_rank, key) with root order first (root 1 before root 2; draw_rank offset by 1000 per root, in a COPY of the
    verdict rows used only for the choice). The chosen plans' rows / templates are copied unchanged from their own
    root's bank file, in bank_plans order (lb95 desc, then key); a chosen plan absent from its root's bank file stops.
  hand, no plan admitted in any root: no BANK_HAND file; the receipt says NO_HAND (the BANK_HAND arm is not run).
Stops (non-zero exit, nothing written) on any pin / consistency failure. Outputs are write-once (exclusive create),
serializer indent=1, sort_keys, trailing LF. Prints plan ids and sha256 of every output.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

CELL = 'D2800_p60'
S5_BANK_SHA = '577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6'
S4_BANK_SHA = 'a428cd1d320196578d55c84eea4b9b391f5e5c03928692a90dff0ac96667d7a8'
S5_FAMILIES = ('rand_8b08c77a63f7d622', 'mut_069966492c9abdcc', 'mut_97dda2be6b45ccf0')
SEARCHED = f'A94_BANK_SEARCHED_{CELL}.json'
HAND = f'A94_BANK_HAND_{CELL}.json'
BANK_NAME = f'A80_BANK_{CELL}.json'
ADM_NAME = f'A80_ADMISSION_{CELL}.json'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canon(x):
    return json.dumps(x, sort_keys=True)


def text_of(doc):
    return json.dumps(doc, indent=1, sort_keys=True)+'\n'


def stop(code, msg):
    print(f'BUILD_BANK_A94 STOP: {msg}')
    sys.exit(code)


def tpl_of(doc, fam):
    got = [t for t in doc['clean5c']['templates'] if t['family'] == fam]
    if len(got) != 1:
        stop(5, f'{fam}: {len(got)} templates')
    return got[0]


def key_of(row, tpl):
    k = row.get('strategy_key')
    kt = (tpl['template'].get('source') or {}).get('strategy_key')
    if k is None or k != kt:
        stop(5, f'{row["family"]}: row key {k} != template key {kt}')
    return k


def read_bank(p, pin):
    b = Path(p).read_bytes()
    if sha(b) != pin:
        stop(2, f'{p}: sha {sha(b)} != expected {pin}')
    doc = json.loads(b)
    if text_of(doc).encode() != b:
        stop(2, f'{p}: not in the source serializer layout')
    return doc, b


def executor_view(doc):
    from tools.coopmwm import a75_holdforms as H
    plans, refused = H.a75_bank_plans(doc, CELL)
    if refused or sorted(p['family'] for p in plans) != sorted(r['family'] for r in doc['rows']):
        stop(5, f'executor refused {refused}')
    core = H.make_core('A75_BANK', CELL, doc)
    return [dict(family=p['family'], lb95=p['lb95'], key=p['key'], extended=p['extended'], core=type(c).__name__,
                 n_roles=len(p['spec']['roles']), zone=list(H.zone_bounds(p['spec'])))
            for p, c in zip(core.plans, core.cores)]


def merge(sources):
    """[(name, doc)] -> (doc, dropped, proof): rows / templates in source order, identical keys deduplicated."""
    rows, tpls, seen, fam_key, dropped, origin = [], [], {}, {}, [], {}
    for name, doc in sources:
        for r in doc['rows']:
            t = tpl_of(doc, r['family'])
            k = key_of(r, t)
            if r['family'] in fam_key and fam_key[r['family']] != k:
                stop(5, f'family {r["family"]} carries two keys')
            fam_key[r['family']] = k
            if k in seen:
                dropped.append(dict(family=r['family'], key=k, source=name, duplicate_of=seen[k],
                                    row_equal=canon(r) == canon(next(x for x in rows if x['family'] == seen[k]))))
                continue
            seen[k] = r['family']
            rows.append(r)
            tpls.append(t)
            origin[r['family']] = (name, doc)
    if not rows:
        stop(5, 'empty bank')
    schema = {d['schema'] for _, d in sources}
    if len(schema) != 1:
        stop(5, f'bank schemas differ {schema}')
    out = dict(schema=schema.pop(), rows=rows, clean5c=dict(templates=tpls))
    proof = {r['family']: dict(source=origin[r['family']][0],
                               row_equal=canon(r) == canon(next(x for x in origin[r['family']][1]['rows']
                                                                if x['family'] == r['family'])),
                               template_equal=canon(t) == canon(tpl_of(origin[r['family']][1], r['family'])))
             for r, t in zip(rows, tpls)}
    if not all(v['row_equal'] and v['template_equal'] for v in proof.values()):
        stop(5, f'unchanged-copy proof failed {proof}')
    return out, dropped, proof


def write(out_dir, files):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name in files:
        if (d/name).exists():
            stop(7, f'{d/name} exists (write-once)')
    for name, text in files.items():
        with open(d/name, 'x', encoding='utf-8', newline='\n') as f:
            f.write(text)


def cmd_searched(a):
    s5, s5b = read_bank(a.s5_bank, S5_BANK_SHA)
    s4 = read_bank(a.s4_bank, S4_BANK_SHA)[0] if a.s4_bank else None       # --s4-bank omitted: S5 ALONE (owner option)
    if tuple(r['family'] for r in s5['rows']) != S5_FAMILIES:
        stop(2, f'S5 rows {[r["family"] for r in s5["rows"]]}')
    doc, dropped, proof = merge([('S5', s5)] + ([('S4', s4)] if s4 is not None else []))
    prio = executor_view(doc)
    t = text_of(doc)
    rec = dict(schema='coopmwm.a94.bank_build.v2', amendment='95', arm='BANK_SEARCHED', cell=CELL,
               mode='s5+s4' if s4 is not None else 's5only',
               rule=('rows of the S5 (577a0564) and S4 (a428cd1d) admission bank files merged unchanged, identical '
                     'strategy keys deduplicated (first kept)') if s4 is not None else
                    'S5-only option: the rows of the S5 (577a0564) admission bank file, unchanged (no S4 rows)',
               inputs=dict(S5=dict(path=a.s5_bank, sha256=S5_BANK_SHA, families=[r['family'] for r in s5['rows']]),
                           S4=None if s4 is None else dict(path=a.s4_bank, sha256=S4_BANK_SHA,
                                                           families=[r['family'] for r in s4['rows']])),
               bytes_equal_s5_bank=(t.encode() == text_of(s5).encode()),
               plan_ids=[r['family'] for r in doc['rows']], keys=[r['strategy_key'] for r in doc['rows']],
               dropped_duplicates=dropped, unchanged_copy_proof=proof, executor_priority=prio,
               output=dict(file=SEARCHED, sha256=sha(t.encode())))
    write(a.out_dir, {SEARCHED: t, 'A94_BANK_SEARCHED_BUILD.json': text_of(rec)})
    print(text_of(rec), end='')
    print('BANK_SEARCHED plan ids:', ' '.join(rec['plan_ids']))
    print(f'sha256 {SEARCHED} {rec["output"]["sha256"]}')
    print('BUILD_BANK_A94_OK')


def load_root(root, want):
    root = Path(root)
    adm_p, bank_p = root/ADM_NAME, root/'a80_banks'/BANK_NAME
    if not adm_p.is_file():
        stop(3, f'{adm_p}: admission receipt absent (admission not finished?)')
    adm_b = adm_p.read_bytes()
    adm = json.loads(adm_b)
    if adm.get('schema') != 'coopmwm.a75d2.admission.v1' or adm.get('cell') != CELL:
        stop(3, f'{adm_p}: schema / cell')
    meta = adm.get('bank')
    if want == 'none':
        if meta is not None or bank_p.exists():
            stop(3, f'{root}: expected no bank (sha "none") but the receipt / file has one')
        return dict(root=str(root), adm=adm, adm_sha=sha(adm_b), doc=None, bank_sha=None, forms={})
    if meta is None:
        stop(3, f'{root}: receipt has no bank but a bank sha was given')
    doc, b = read_bank(bank_p, want)
    if meta['file_sha256'] != want or [r['family'] for r in doc['rows']] != list(meta['families']):
        stop(3, f'{root}: bank file / receipt bank mismatch')
    for r in doc['rows']:
        v = next((x for x in adm['verdicts'] if x['family'] == r['family']), None)
        if v is None or not (v['PASS'] and v['IN_BANK']) or abs(float(r['lb95_veh_s'])+float(v['ci95'][1])) > 1e-9:
            stop(3, f'{root}: row {r["family"]} does not match its admitted verdict (lb95 = -ci95 upper)')
    return dict(root=str(root), adm=adm, adm_sha=sha(adm_b), doc=doc, bank_sha=want, forms=meta['forms'])


def cmd_hand(a):
    pairs = []
    for x in a.hand:
        root, _, want = x.partition('=')
        if not root or not (want == 'none' or len(want) == 64):
            stop(2, f'--hand {x}: want ROOT=SHA256|none')
        pairs.append((root, want))
    if not 1 <= len(pairs) <= 2:
        stop(2, '--hand: one or two roots')
    roots = [load_root(r, w) for r, w in pairs]
    if len(roots) == 2:
        for k in ('panel_sha256', 'seeds', 'reference', 'ref_pins_receipt_sha256'):
            if roots[0]['adm'].get(k) != roots[1]['adm'].get(k):
                stop(3, f'hand roots differ in {k}')
    with_bank = [r for r in roots if r['doc'] is not None]
    rec = dict(schema='coopmwm.a94.bank_build.v2', amendment='95', arm='BANK_HAND', cell=CELL,
               inputs=[dict(root=r['root'], admission_sha256=r['adm_sha'], bank_sha256=r['bank_sha'],
                            n_admitted=r['adm'].get('n_admitted'), admitted=r['adm'].get('admitted'),
                            bank_families=[x['family'] for x in r['doc']['rows']] if r['doc'] else None)
                       for r in roots])
    if not with_bank:
        rec.update(rule='no hand plan admitted in any root: no BANK_HAND file; the BANK_HAND arm is not run',
                   status='NO_HAND', plan_ids=[], output=None)
        write(a.out_dir, {'A94_BANK_HAND_BUILD.json': text_of(rec)})
        print(text_of(rec), end='')
        print('BANK_HAND: NO_HAND (no hand plan admitted)')
        print(f'sha256 {HAND} none')
        print('BUILD_BANK_A94_OK')
        return
    if len(with_bank) == 1:
        doc = with_bank[0]['doc']
        rec.update(rule='one root with a bank: its rows unchanged', choice=None)
    else:
        import a75_d2forms as D2
        import a80_s2b as A80
        D2.FORMS = A80.FORMS                                       # as a80_s2b._point_d2 sets it (only this name)
        rows, src_of = [], {}
        for i, r in enumerate(roots):
            for v in r['adm']['verdicts']:
                if v['family'] in src_of:
                    stop(5, f'family {v["family"]} in both hand roots')
                src_of[v['family']] = r
                w = copy.deepcopy(v)
                w['draw_rank'] = 1000*i+int(v['draw_rank'])
                rows.append(w)
        chosen, dup, table = D2.bank_choice(rows)
        picked = []
        for fam in chosen:
            r = src_of[fam]
            if r['doc'] is None or fam not in [x['family'] for x in r['doc']['rows']]:
                stop(5, f'chosen plan {fam} is not a row of its root bank file (cannot copy it unchanged)')
            picked.append((fam, r))
        picked.sort(key=lambda fr: (-float(next(x for x in fr[1]['doc']['rows'] if x['family'] == fr[0])['lb95_veh_s']),
                                    next(x for x in fr[1]['doc']['rows'] if x['family'] == fr[0])['strategy_key']))
        doc = dict(schema=with_bank[0]['doc']['schema'],
                   rows=[next(x for x in r['doc']['rows'] if x['family'] == f) for f, r in picked],
                   clean5c=dict(templates=[tpl_of(r['doc'], f) for f, r in picked]))
        rec.update(rule='two roots: a75_d2forms.bank_choice (unchanged; a80 FORMS, MAX_BANK %d) on the union of both '
                        'receipts\' verdict rows; statistic = verdict ci95[1] (admission CI95 upper of plan - '
                        'MAPPO60C total delay; = -lb95_veh_s), most negative first, ties: strategy key; root 1 before '
                        'root 2 in the duplicate order; rows copied unchanged' % D2.MAX_BANK,
                   choice=dict(chosen=chosen, duplicates=dup, per_form=table,
                               forms={f: r['forms'].get(f) for f, r in picked}))
    for r in doc['rows']:
        key_of(r, tpl_of(doc, r['family']))
    srcs = [(x['root'], x['doc']) for x in with_bank]
    proof = {}
    for row in doc['rows']:
        name, d = next((n, d) for n, d in srcs if any(x['family'] == row['family'] for x in d['rows']))
        proof[row['family']] = dict(source=name, row_equal=canon(row) == canon(next(x for x in d['rows'] if x['family'] == row['family'])),
                                    template_equal=canon(tpl_of(doc, row['family'])) == canon(tpl_of(d, row['family'])))
    if not all(v['row_equal'] and v['template_equal'] for v in proof.values()):
        stop(5, f'unchanged-copy proof failed {proof}')
    prio = executor_view(doc)
    t = text_of(doc)
    rec.update(status='HAND', plan_ids=[r['family'] for r in doc['rows']], keys=[r['strategy_key'] for r in doc['rows']],
               unchanged_copy_proof=proof, executor_priority=prio,
               bytes_equal_single_source=(t.encode() == text_of(with_bank[0]['doc']).encode()) if len(with_bank) == 1 else None,
               output=dict(file=HAND, sha256=sha(t.encode())))
    write(a.out_dir, {HAND: t, 'A94_BANK_HAND_BUILD.json': text_of(rec)})
    print(text_of(rec), end='')
    print('BANK_HAND plan ids:', ' '.join(rec['plan_ids']))
    print(f'sha256 {HAND} {rec["output"]["sha256"]}')
    print('BUILD_BANK_A94_OK')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('searched', 'hand'))
    p.add_argument('--s5-bank')
    p.add_argument('--s4-bank')
    p.add_argument('--hand', action='append', default=[])
    p.add_argument('--out-dir', required=True)
    a = p.parse_args()
    if a.command == 'searched':
        if not a.s5_bank:
            stop(2, 'searched needs --s5-bank (and --s4-bank unless the S5-only option)')
        cmd_searched(a)
    else:
        cmd_hand(a)


if __name__ == '__main__':
    main()
