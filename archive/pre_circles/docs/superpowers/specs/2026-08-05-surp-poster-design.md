# SURP 2026 Poster — Design Spec

Date: 2026-08-05. Author: Jinghan Peng (UCLA). Supervisor: Prof. Dongkun Han (CUHK).

## Constraints (from SURP Poster Presentation Guidelines PDF)

- Board/print size: **90 cm W x 140 cm H, portrait** (OAL prints 1:1 at this size)
- Colour model: **CMYK**; output **PDF** (LaTeX PDF is fine)
- Submission deadline: **5:00 pm, 6 Aug 2026** (free OAL printing)
- Header must contain: title, author name + home university, CUHK supervisor name, CUHK emblem + UCLA emblem
- Content: brief intro, clear explanation, results; columns + point form; layperson-readable
- Optional: "summer in HK" photo section; acknowledgements welcome
- Audience votes for Best Poster — most voters are non-experts

## Headline / story arc

**"Traffic Shepherd: Can a few robot cars make a highway merge safer and fairer?"**

1. Hook: merges are where highways break down; ramp drivers wait far longer than mainline; stop-and-go waves cause near-misses.
2. Naive fix fails: fixed turn-taking metering = +14.5 s delay, +17 s gap (rejected).
3. Method: MAPPO (multi-agent RL), 10% CAVs, 5 semantic actions (coast / fair_balance / recover_speed / vacate_drop / protect), reward priorities safety > fairness > do-no-harm, with "learned abstention" (intervention tax + fair-win bonus).
4. Verified results (champion: v10c iter_90 + step soft mask + recover sign gate; 50 held-out seeds 101-150, dense 3500 veh/h, dt 0.5 s, TTC ruler 1.5 s, paired no_op):
   - Fairness gap 6.94 -> 4.57 s (**-34%**)
   - Mean delay 23.43 -> 20.96 s (**-10.5%**)
   - Near-misses (TTC<1.5 s) 14.90 -> 12.76/seed (**-14%**)
   - Collisions 0 -> 0; strict domination 28/50 seeds; full win-rule 26/50
5. Key finding: structural safety-flow coupling; ~85% of excess TTC from HDV-HDV compression waves; champion sits at a Pareto corner — 11+ follow-up experiments across every lever family failed to beat it (negative results panel).
6. All-HDV context (TT/delay/collisions only — its TTC pipeline is not comparable): TT 81.40 s, loss 28.75 s, 1.34 collisions/seed.

Data source of record: `runs/eval/fair_safe_v10c_iter90_signgate15_dt05_holdout101_150/summary.json` (+ per-seed JSONs).

## Layout (beamerposter, custom 90x140 cm, 3 columns)

- Header: title, Jinghan Peng (UCLA), supervisor Dongkun Han (CUHK), CUHK + UCLA emblems
- Col 1: Motivation; merge scenario schematic; experimental setup (SUMO, demand, 10% CAV, seed protocol, paired no_op ruler)
- Col 2: Method — pipeline diagram; 5 semantic actions table; reward priorities + learned abstention
- Col 3: Results — headline bar chart (no_op vs shepherd); per-seed paired scatter (delta delay vs delta gap, win colouring); safety-flow coupling finding; ablation/negative-results mini-table; conclusions
- Footer: acknowledgements; (optional HK photo corner if user supplies photos)

## Build plan

- Figures: matplotlib from summary.json per_seed data (bar chart, paired scatter); TikZ schematic for merge + pipeline (no external images needed)
- `poster/` dir in repo root: `poster.tex`, `figures/`, `venv/` for matplotlib
- Compile: MiKTeX pdflatex (xelatex fallback); `\usepackage[cmyk]{xcolor}`
- Verify: pdfinfo page size = 90x140 cm; skill quality checklist (`scripts/review_poster.sh` guidance); guidelines checklist above

## Explicitly excluded

- Ongoing campaign-3 hybrid results (smoke tests only)
- peek101_120 / seeds42_91 / smoke12 numbers (not holdout)
- All-HDV TTC comparison (incompatible pipeline)
