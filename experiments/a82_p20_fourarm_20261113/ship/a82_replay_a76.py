"""A82 part A, zero-SUMO EQUIVALENCE CHECK of the re-pointed admission path (not an analysis; prints booleans only).

Feeds the A76 records themselves (36 A76V plan waves + the stored A76V_COMP wave, all read-only from the A76 root)
through exactly the code path A82 part A uses for admission - a76_d2p20.admit with its names re-pointed by
a82_s3b20._point, the stored-reference loader and the routed report reader - writing ONLY into a throwaway root, and
compares the result with the sealed A76 admission receipt: per plan CI95 / group-rule / PASS / duplicate / IN_BANK, the
chosen bank and the bank file sha (expected f8b5237e). Prints equal / not equal per field and counts; no outcome
number is printed. usage: a82_replay_a76.py --tmp-root /dev/shm/<new dir> [--a76-root /root/a76p20/root]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a82_s3b20 as M  # noqa: E402
import a76_d2p20 as M76  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--tmp-root', required=True)
    p.add_argument('--a76-root', default='/root/a76p20/root')
    a = p.parse_args(argv)
    N = M._N()
    tmp, root76 = Path(a.tmp_root), Path(a.a76_root)
    if tmp.exists() or not str(tmp).startswith('/dev/shm/'):
        raise SystemExit('tmp root must be a NEW dir under /dev/shm')
    tmp.mkdir(parents=True)
    for f in ('BOUND_U0.json', 'ORACLE_BINDING.json', 'PRERUN_PLAN.json', 'prerun_plant/CONFIG.json'):
        (tmp/f).parent.mkdir(parents=True, exist_ok=True)
        (tmp/f).write_bytes((root76/f).read_bytes())
    cands = M76.candidates()                      # the 36 A76 plans (A75-D2 doc 9025e3b1), before any re-pointing
    D76 = M76.core()                              # the A76 core / names (for the A76 plan specs) BEFORE re-pointing
    adm76 = N.verify(N.read(root76/'A76_ADMISSION_D2800_p20.json'), 'receipt_sha256', 'A76 admission')
    pins = M.make_pins(tmp, root76)               # panel byte copy + pinned stored reference (as part A `pin`)
    M._point(root76, pins)
    D, panel = M76.setup('v', tmp)
    comp = M76.comparator(tmp)
    a76_cands_bank = 'A76_CANDS_D2800_p20.json'

    def waves(D_, panel_, root_, comp_, cands_):
        ref = [(M.REF_WAVE, 'episode', M76.u0_specs(panel_, root76, comp_, M.REF_TAG))]
        return ref + [(f'A76V_{c["family"]}', 'episode',
                       M76.plan_specs(D76, panel_, root76, a76_cands_bank, 'A75_PLAN', c['family'], f'a76v_{c["key"]}',
                                      comp_)) for c in cands_]

    orig_load = M._ORIG['load']

    def load(root_, wave, specs):
        return orig_load(root76, wave, specs)     # every wave from the A76 root (read-only)

    class AP(M._AProxy):
        def report_of(self, root_, rec):
            M.ref_episode_check(root76, rec, None)  # refuses if a raw log is present (no cache write into A76)
            return self._A.report_of(root76, rec)
    M76.verify_waves = waves
    M76._load = load
    M76._mods = lambda: (M._ORIG['mods']()[0], AP(M._ORIG['mods']()[1], root76, pins), M._ORIG['mods']()[2])
    (tmp/M.MECH_FILE).write_text(json.dumps(N.seal(dict(verdict='PASS', note='replay fixture only'), 'receipt_sha256')))
    v = M76.admit(D, panel, tmp, comp, cands)
    old = {r['family']: r for r in adm76['verdicts']}
    new = {r['family']: r for r in v['verdicts']}
    fields = ('ci95', 'allowed', 'PASS', 'delay_test_pass', 'group_rule_pass', 'duplicate_of', 'IN_BANK', 'per_seed')
    eq = {f: sum(json.dumps(old[k].get(f), sort_keys=True) == json.dumps(new[k].get(f), sort_keys=True) for k in old)
          for f in fields}
    out = dict(n_plans=len(old), same_families=sorted(old) == sorted(new), equal_counts=eq,
               chosen_equal=adm76['a76_bank'] == v['a76_bank'], n_admitted_equal=adm76['n_admitted'] == v['n_admitted'],
               bank_file_sha_equal=(adm76['bank'] or {}).get('file_sha256') == (v['bank'] or {}).get('file_sha256'),
               per_form_equal_on_a76_forms=all(adm76['per_form'][f] == v['per_form'][f] for f in adm76['per_form']),
               ref_mean_equal=adm76['ref_mean_J_veh_s'] == v['ref_mean_J_veh_s'])
    out['ALL_EQUAL'] = bool(out['same_families'] and all(n == len(old) for n in eq.values()) and out['chosen_equal']
                            and out['n_admitted_equal'] and out['bank_file_sha_equal']
                            and out['per_form_equal_on_a76_forms'] and out['ref_mean_equal'])
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
