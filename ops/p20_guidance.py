"""p20 guidance docs (CPU, no physics; owner REGISTRATION 2026-11-08 17:22 + low-congestion follow-up).

Training cells D2400_p20 / D2800_p20 / D3200_p20; per-cell priority clean5b_restart.P20_PRIORITY_RULE.
rule: MULTICRIT files (fullpol_fair.build_guidance; LOWCAV today) -> families wall_g_duo12 / wall_g_solo2,
      CVaR90 delay attached from each file's metrics, p20 priority + prefer-both -> F8 p20 guidance_path.
wm:   WMVERIFY bank(s) (fullpol_bank.build with the p20 eligibility; parity + WM-group rules unchanged) with the
      p20 rule doc as the labelled fallback -> clean5c p20 guidance_path. Pass more --bank / --analysis /
      --arms / --verify files (a later p20 WMVERIFY, any cell) to swap the bank before launch.

  python ops/p20_guidance.py rule --multicrit MULTICRIT_confirm.json --out GUIDANCE_P20_RULE.json --provenance ...
  python ops/p20_guidance.py wm --bank BANK_MC.jsonl --analysis ANALYSIS.json --arms PLAN.json --verify VERIFY.json \
      --fallback GUIDANCE_P20_RULE.json --out GUIDANCE_P20_WM.json --provenance ...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.coopmwm import clean5b_restart as c5b, fullpol_bank, fullpol_fair  # noqa: E402


def _cvar90(multicrit):
    """{(cell, family): CVaR90 per-vehicle delay diff vs u0 (s)}; the most conservative (max) over files."""
    out = {}
    for path in multicrit:
        for cell, c in json.loads(Path(path).read_text(encoding='utf-8'))['cells'].items():
            for name, a in c['arms'].items():
                m = (a.get('metrics') or {}).get('cvar90_delay')
                if isinstance(m, dict) and m.get('diff') is not None:
                    out[(cell, name)] = max(out.get((cell, name), float('-inf')), float(m['diff']))
    return out


def rule_doc(multicrit, *, provenance):
    doc = fullpol_fair.build_guidance(multicrit, provenance=provenance)
    cvar = _cvar90(multicrit)
    for r in doc['rows']:
        if (r['regime'], r['family']) in cvar:
            r['cvar90_diff_s'] = cvar[(r['regime'], r['family'])]
    return c5b.p20_select(doc, c5b.P20_CELLS, families=c5b.P20_RULE_FAMILIES,
                          note='p20 rule guidance: MULTICRIT '+'/'.join(c5b.P20_RULE_FAMILIES))


def wm_doc(bank, *, analysis, arms, verify, fallback, provenance):
    merged = dict(families={})
    for path in verify:                    # several VERIFY files (initial + later p20 WMVERIFY): families merged
        merged['families'].update(json.loads(Path(path).read_text(encoding='utf-8')).get('families') or {})
    doc = fullpol_bank.build(bank, fallback=fallback, analysis_paths=analysis, provenance=provenance,
                             arms=[json.loads(Path(p).read_text(encoding='utf-8')) for p in arms], verify=merged,
                             eligible_fn=c5b.p20_eligible)
    return c5b.p20_select(doc, c5b.P20_CELLS, note='p20 WM guidance: WMVERIFY-verified p20 multi-CAV plans, '
                          'labelled rule fallback elsewhere')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('kind', choices=['rule', 'wm'])
    ap.add_argument('--multicrit', nargs='*', default=())
    ap.add_argument('--bank', nargs='*', default=())
    ap.add_argument('--analysis', nargs='*', default=())
    ap.add_argument('--arms', nargs='*', default=())
    ap.add_argument('--verify', nargs='*', default=())
    ap.add_argument('--fallback')
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit('refusing to overwrite '+str(out))
    if a.kind == 'rule':
        doc = rule_doc(a.multicrit, provenance=a.provenance)
    else:
        fallback = json.loads(Path(a.fallback).read_text(encoding='utf-8'))
        c5b.check_p20_guidance(fallback, c5b.P20_CELLS, wm=False)
        doc = wm_doc(a.bank, analysis=a.analysis, arms=a.arms, verify=a.verify, provenance=a.provenance,
                     fallback=fallback)
    none = c5b.check_p20_guidance(doc, c5b.P20_CELLS, wm=a.kind == 'wm')
    out.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding='utf-8')
    print(json.dumps(dict(out=str(out), sha256=doc['source']['sha256'], none=list(none),
                          cells={c: (v['source'], [t['family'] for t in v['targets']])
                                 for c, v in doc['p20']['cells'].items()})))


if __name__ == '__main__':
    main()
