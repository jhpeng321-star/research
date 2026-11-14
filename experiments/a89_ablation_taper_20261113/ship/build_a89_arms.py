"""A89 (zero SUMO, deterministic): build the four-plan arms bank A89_ARMS_D2800_p60.json.

usage (inside the A89 tree, env sourced):  python -I build_a89_arms.py BANK_C325 E1R_CANDS_BANK OUT_JSON
  BANK_C325       the frozen A75-D2 bank A75D2_BANK_D2800_p60.json (sha256 c325ec1d..., checked)
  E1R_CANDS_BANK  the E1R candidate bank /root/a80e1r/root/a80_banks/A80_CANDS_D2800_p60.json (sha256 pinned below)

Rows / templates of the output (cell D2800_p60, schema of the source banks):
  ROLL_BOTH     family d2_d86b1d6ac653e8ad: template + row objects COPIED UNCHANGED from the c325ec1d bank
  ROLL_L0_ONLY  the same template with ONLY the lane-2 role deleted (roles[i] with sumo_lane == 2); every other field
                unchanged; new family a89l0_d86b1d6ac653e8ad and the executor's own strategy_key of the result
  ROLL_L2_ONLY  the same template with ONLY the lane-0 role deleted (sumo_lane == 0); onset_s 20 kept
  TAPER         family mut_1fca373046a3c66e: template + row objects COPIED UNCHANGED from the E1R candidate bank
Nothing is retuned; no SUMO outcome is read (the lb95_veh_s field of the copied bank row is carried, never used: with
one plan per A75_PLAN core it does not affect execution). Refuses unless every plan is accepted by
a75_holdforms.a75_bank_plans and C.restrict yields exactly the one family.
"""
import copy
import hashlib
import json
import sys

C325 = 'c325ec1da034d0ad22d242f97e926c30cfa998981f16cef935b1b6498565c30c'
E1R_CANDS = 'PINNED_AT_SETUP'          # replaced by the sha256 checked in setup_a89.sh (argument 4 if given)
CELL = 'D2800_p60'
ROLL = 'd2_d86b1d6ac653e8ad'
TAPER = 'mut_1fca373046a3c66e'
ARM_FAMILY = dict(ROLL_BOTH=ROLL, ROLL_L0_ONLY='a89l0_d86b1d6ac653e8ad', ROLL_L2_ONLY='a89l2_d86b1d6ac653e8ad',
                  TAPER=TAPER)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canon(x):
    return json.dumps(x, sort_keys=True)


def one(xs, fam):
    got = [x for x in xs if x['family'] == fam]
    if len(got) != 1:
        raise SystemExit(f'{fam}: {len(got)} entries')
    return got[0]


def drop_role(tpl, row, lane, fam, H):
    """Delete exactly the one role with sumo_lane == lane; rename the family; re-key with the executor."""
    t, r = copy.deepcopy(tpl), copy.deepcopy(row)
    roles = t['offer_condition']['roles']
    hit = [i for i, x in enumerate(roles) if int(x['sumo_lane']) == lane]
    if len(hit) != 1 or len(roles) != 2:
        raise SystemExit(f'role lane {lane}: {len(hit)} matches in {len(roles)} roles')
    del roles[hit[0]]
    key = H.strategy_key(H.spec_of(fam, t['offer_condition']))
    t['family'], r['family'] = fam, fam
    t['template']['source']['strategy_key'], r['strategy_key'] = key, key
    return t, r


def build(bank_bytes, cands_bytes, cands_sha):
    from tools.coopmwm import a75_holdforms as H
    from tools.coopmwm import a37_cellbank as C
    if sha(bank_bytes) != C325:
        raise SystemExit('A75-D2 bank is not c325ec1d')
    if sha(cands_bytes) != cands_sha:
        raise SystemExit('E1R candidate bank sha mismatch')
    bank, cands = json.loads(bank_bytes), json.loads(cands_bytes)
    rt, rr = one(bank['clean5c']['templates'], ROLL), one(bank['rows'], ROLL)
    tt, tr = one(cands['clean5c']['templates'], TAPER), one(cands['rows'], TAPER)
    l0 = drop_role(rt, rr, 2, ARM_FAMILY['ROLL_L0_ONLY'], H)
    l2 = drop_role(rt, rr, 0, ARM_FAMILY['ROLL_L2_ONLY'], H)
    tpls = [copy.deepcopy(rt), l0[0], l2[0], copy.deepcopy(tt)]
    rows = [copy.deepcopy(rr), l0[1], l2[1], copy.deepcopy(tr)]
    if bank['schema'] != cands['schema']:
        raise SystemExit(f'schemas differ {bank["schema"]} / {cands["schema"]}')
    doc = dict(schema=bank['schema'], rows=rows, clean5c=dict(templates=tpls))
    # executor acceptance, one plan per arm
    out = {}
    for arm, fam in ARM_FAMILY.items():
        plans, refused = H.a75_bank_plans(C.restrict(doc, fam), CELL)
        if refused or [p['family'] for p in plans] != [fam]:
            raise SystemExit(f'{arm}: refused {refused}')
        core = H.make_core('A75_PLAN', CELL, doc, family=fam)
        out[arm] = dict(family=fam, strategy_key=plans[0]['key'], extended=plans[0]['extended'],
                        n_roles=len(plans[0]['spec']['roles']), executor_core=type(core.cores[0]).__name__,
                        spec_sha256=sha(canon(plans[0]['spec']).encode()))
    # byte-for-byte proofs of the copied rows (same serializer as the source files: indent=1, sort_keys)
    out['ROLL_BOTH']['template_equal_c325'] = canon(tpls[0]) == canon(rt)
    out['ROLL_BOTH']['row_equal_c325'] = canon(rows[0]) == canon(rr)
    out['TAPER']['template_equal_e1r'] = canon(tpls[3]) == canon(tt)
    out['TAPER']['row_equal_e1r'] = canon(rows[3]) == canon(tr)
    # the deleted-role arms differ from ROLL_BOTH only by the removed role + family/key
    for arm, keep in (('ROLL_L0_ONLY', 0), ('ROLL_L2_ONLY', 2)):
        t = tpls[1 if keep == 0 else 2]
        same = {k: v for k, v in t['offer_condition'].items() if k != 'roles'} == \
            {k: v for k, v in rt['offer_condition'].items() if k != 'roles'}
        role_same = [x for x in rt['offer_condition']['roles'] if int(x['sumo_lane']) == keep] == t['offer_condition']['roles']
        out[arm]['condition_unchanged_except_roles'] = same
        out[arm]['kept_role_identical'] = role_same
    if not all(v for a in out.values() for k, v in a.items() if k.endswith(('equal_c325', 'equal_e1r', 'unchanged_except_roles', 'identical'))):
        raise SystemExit(f'proof failed {out}')
    return doc, out


def main():
    bank_p, cands_p, out_p = sys.argv[1:4]
    cands_sha = sys.argv[4] if len(sys.argv) > 4 else E1R_CANDS
    doc, proof = build(open(bank_p, 'rb').read(), open(cands_p, 'rb').read(), cands_sha)
    text = json.dumps(doc, indent=1, sort_keys=True)+'\n'
    with open(out_p, 'x', encoding='utf-8', newline='\n') as f:      # write-once
        f.write(text)
    print(json.dumps(dict(out=out_p, sha256=sha(text.encode()), arms=proof), indent=1))


if __name__ == '__main__':
    main()
