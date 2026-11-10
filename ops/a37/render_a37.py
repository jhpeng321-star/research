"""Render the amendment-37 laptop analysis outputs (A37_RESULTS_<cell>.json, ADMISSION_<cell>.json, A37B_NATIVE.json,
IDENTITY_GATE.json) into A37_RESULTS.md. Zero physics. Usage: python render_a37.py <out_dir>"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])


def ci(s, k='total_minus_ref', f='{:+,.0f}'):
    x = s[k]
    pct = f' ({x["pct"]:+.1f}%)' if x.get('pct') is not None else ''
    return f'{f.format(x["diff"])} [{f.format(x["ci95"][0])}, {f.format(x["ci95"][1])}]{pct}, p={x.get("p_two_sided", float("nan")):.3f}'


lines = ['# AMENDMENT 37 results (condition tailoring; cells never pooled)', '',
         'diff = treatment - reference (veh-s per world; negative = less delay); paired percentile bootstrap 10,000, '
         'seed 20261104. Admission = UNC rule (24 TRAIN worlds, (plan - U0) CI95 upper < 0 AND ALLOWED).', '']
g = out/'IDENTITY_GATE.json'
if g.is_file():
    gate = json.loads(g.read_text())
    lines += [f'Plant identity gate: {gate["verdict"]} {gate["per_panel"]} (max |dJ| '
              f'{max(w["abs_diff"] for w in gate["worlds"]):.2e})', '']
for cell in ('D3200_p20', 'D2400_p20'):
    a = out/f'ADMISSION_{cell}.json'
    if a.is_file():
        adm = json.loads(a.read_text())
        lines += [f'## {cell}', '', f'Admitted: {adm["admitted"] or "none (CELL_BANK := U0)"}', '',
                  '| plan | WM rank | plan - U0 mean [CI95] | worst group inc (s) | ALLOWED | PASS | act/ep |',
                  '|---|---|---|---|---|---|---|']
        for v in adm['verdicts']:
            lines.append(f'| {v["family"]} | {v["wm_rank"]} | {v["mean"]:+,.0f} [{v["ci95"][0]:+,.0f}, {v["ci95"][1]:+,.0f}] | '
                         f'{v["worst_group"]} {v["worst_group_inc"] if v["worst_group_inc"] is None else round(v["worst_group_inc"], 1)} | '
                         f'{v["allowed"]} | {v["PASS"]} | {v["activations_per_ep"]:.1f} |')
        lines.append('')
    r = out/f'A37_RESULTS_{cell}.json'
    if r.is_file():
        res = json.loads(r.read_text())
        p = res['primary']
        lines += [f'PRIMARY CELL_BANK - LEARNED_TRIGGER: {ci(res["contrasts"][0])} -> condition tailoring '
                  f'{"SUPPORTED" if p["supported"] else "NOT supported"} ({p["note"]})', '',
                  '| treatment - reference | kind | total delay | worst group inc (s) | groups pass | CVaR90 | hard brakes | TTC |',
                  '|---|---|---|---|---|---|---|---|']
        for c in res['contrasts']:
            grp = c.get('groups') or {}
            worst = max(((k, v['increase_mean_s']) for k, v in (grp.get('groups') or {}).items()
                         if v.get('registered') and 'increase_mean_s' in v), key=lambda x: x[1], default=('-', 0))
            tail = c.get('tail_cvar90') or {}
            tc = tail.get('diff_ci95') or tail.get('ci95')
            tv = tail.get('diff')
            lines.append(f'| {c["treatment"]} - {c["reference"]} | {c["kind"]} | {ci(c)} | {worst[0]} {worst[1]:+.1f} | '
                         f'{grp.get("all_pass")} | {tv if tv is None else round(tv, 1)} {tc and [round(x, 1) for x in tc]} | '
                         f'{c["hard_brakes_minus_ref"]["diff"]:+.1f} | {c["ttc_events_minus_ref"]["diff"]:+.2f} |')
        lines += ['', 'Safety totals: ' + '; '.join(f'{k} {v["safety_totals"]}' for k, v in res['arms'].items()), '']
n = out/'A37B_NATIVE.json'
if n.is_file():
    nat = json.loads(n.read_text())
    c = nat['contrast']
    lines += ['## 37b NATIVE - U0 (a33 fresh panel; context, no pass/fail)', '',
              f'{nat["n_worlds"]} worlds, U0 source: {nat["u0_source"]}; total delay {ci(c)}; mean J {nat["mean_J"]}', '']
(out/'A37_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines))
