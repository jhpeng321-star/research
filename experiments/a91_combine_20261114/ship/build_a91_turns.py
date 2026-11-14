"""A91 (zero SUMO, deterministic): build the two-plan bank A91_TURNS_D2800_p60.json for the BANK_TURNS arm.

usage (inside the A91 tree, env sourced):  python -I build_a91_turns.py A89_ARMS_BANK OUT_JSON
  A89_ARMS_BANK  the pinned A89 arms bank A89_ARMS_D2800_p60.json (sha256 de9a30a9..., checked)

Output (cell D2800_p60, schema of the source bank, serializer of the source bank: indent=1, sort_keys, trailing LF):
  rows / templates = the a89l2_d86b1d6ac653e8ad (ROLL_L2_ONLY) and mut_1fca373046a3c66e (TAPER) row and template
  objects COPIED UNCHANGED (canonical-JSON equal) from the A89 arms bank. Nothing is retuned, nothing renamed, no
  threshold added. The bank is executed by the UNCHANGED a75 executor arm A75_BANK (HoldFormCellBankCore ->
  oracle_plan.OracleCore.step): one session ledger per plan (own slots / cooldown), plans tried each second in the
  executor's own priority order (a75_bank_plans sort: -lb95 then key), a CAV in one plan's session is never offered
  to the other. The carried lb95_veh_s fields therefore fix the priority (L2 row 2917.05 > TAPER row 0.0): L2 first.
Refuses unless both plans are accepted by a75_holdforms.a75_bank_plans and make_core('A75_BANK') admits exactly both.
No SUMO outcome is read.
"""
import copy
import hashlib
import json
import sys

A89_ARMS = 'de9a30a944eda470777cbd3247c6ca81db2745547549cde3e2bff154e23824c2'
CELL = 'D2800_p60'
L2 = 'a89l2_d86b1d6ac653e8ad'
TAPER = 'mut_1fca373046a3c66e'
FAMILIES = (L2, TAPER)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canon(x):
    return json.dumps(x, sort_keys=True)


def one(xs, fam):
    got = [x for x in xs if x['family'] == fam]
    if len(got) != 1:
        raise SystemExit(f'{fam}: {len(got)} entries')
    return got[0]


def build(arms_bytes):
    from tools.coopmwm import a75_holdforms as H
    if sha(arms_bytes) != A89_ARMS:
        raise SystemExit('A89 arms bank is not de9a30a9')
    src = json.loads(arms_bytes)
    tpls = [copy.deepcopy(one(src['clean5c']['templates'], f)) for f in FAMILIES]
    rows = [copy.deepcopy(one(src['rows'], f)) for f in FAMILIES]
    doc = dict(schema=src['schema'], rows=rows, clean5c=dict(templates=tpls))
    proof = dict(schema=doc['schema'])
    for f, t, r in zip(FAMILIES, tpls, rows):
        proof[f] = dict(template_equal_a89=canon(t) == canon(one(src['clean5c']['templates'], f)),
                        row_equal_a89=canon(r) == canon(one(src['rows'], f)))
    plans, refused = H.a75_bank_plans(doc, CELL)
    if refused or sorted(p['family'] for p in plans) != sorted(FAMILIES):
        raise SystemExit(f'executor refused {refused}')
    core = H.make_core('A75_BANK', CELL, doc)
    proof['executor_priority_order'] = [dict(family=p['family'], lb95=p['lb95'], key=p['key'], extended=p['extended'],
                                             core=type(c).__name__, n_roles=len(p['spec']['roles']))
                                        for p, c in zip(core.plans, core.cores)]
    if not all(v for f in FAMILIES for v in proof[f].values()):
        raise SystemExit(f'proof failed {proof}')
    return doc, proof


def main():
    arms_p, out_p = sys.argv[1:3]
    doc, proof = build(open(arms_p, 'rb').read())
    text = json.dumps(doc, indent=1, sort_keys=True)+'\n'
    with open(out_p, 'x', encoding='utf-8', newline='\n') as f:      # write-once
        f.write(text)
    print(json.dumps(dict(out=out_p, sha256=sha(text.encode()), proof=proof), indent=1))


if __name__ == '__main__':
    main()
