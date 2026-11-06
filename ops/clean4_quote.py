"""Write CLEAN4_QUOTE.json: full u1-u24 per-phase quote (>=15% margin), zero physics.

Usage: python ops/clean4_quote.py OUT.json [--native NATIVE_QUOTES.json] [--cap N]

Without --native the diagnostic phases (lessons/probes/replicated) use the measured
clean3 merge-map native manifests x1.5 (wz3to2 windows not yet expanded) and the
quote is marked ESTIMATE. With --native (written by the clean4 zero-physics prepare:
{lesson_sim_s, probe_sim_s, replicated_sim_s, gate_sim_s}) the bases are the actual
wz3to2 manifests. The chain re-asserts coverage against the native manifests before
launch and refuses if any phase falls short.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.coopmwm import clean4_budget as b  # noqa: E402

# Registered phases/charges under the owner cap (REGISTRATION.md, 2026-11-06). Caps are
# counted in full where the phase is still open or its final spend is not ledger-verified.
PRIOR = {
    'clean1+clean2+parity (ledgers)': 3_098_640,
    'clean3 stopped pre-launch roots r1-r4 (ledgers)': 22_800,
    'clean3_r6 (ledger, owner-paused)': 120_760,
    'clean3 G1 certification (actual)': 0,
    'mining s_mining (reserved phase)': 500_000,
    'coop_probe (phase cap; actual 240,078)': 250_000,
    'hicav_probe (phase cap)': 150_000,
    'newmap_probe (phase cap; actual 148,520)': 150_000,
    'wallmine (phase cap; actual 147,550)': 150_000,
    'upper_bound (phase cap; actual 148,995)': 150_000,
    'newmap2 incl. newmap2_smoke 40,000 (phase cap)': 300_000,
    'coopkinds N8 (phase cap)': 200_000,
}
CAP = 8_200_000
CLEAN3_NATIVE = dict(lesson_sim_s=35_040, probe_sim_s=45_480, replicated_sim_s=143_760, gate_sim_s=15_160)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--native')
    p.add_argument('--cap', type=int, default=CAP)
    p.add_argument('--episode-sim-s', type=int, default=1200)
    args = p.parse_args()
    if args.native:
        native = json.loads(Path(args.native).read_text(encoding='utf-8'))
        factor, status = 1.0, 'NATIVE'
        basis = dict(lessons='wz3to2 native lesson manifests (zero-physics prepare)',
                     probes='wz3to2 native probe manifests u2+u4', replicated='wz3to2 native replicated manifests u6+u12+u24')
    else:
        native, factor, status = CLEAN3_NATIVE, 1.5, 'ESTIMATE'
        basis = dict(lessons='ESTIMATE: clean3 merge-map native 4 x 8760 x1.5 (wz3to2 manifests pending)',
                     probes='ESTIMATE: clean3 native 2 x 22740 x1.5', replicated='ESTIMATE: clean3 native 3 x 47920 x1.5')
    basis.update(g4='G4 two-actor full quote 153600 (clean3); registered hard cap 230400',
                 final='A/B/R u24 + base (frozen MAPPO u0) + native on the 100 registered final-range worlds 989600-989699')
    phases = b.quote(episode_sim_s=args.episode_sim_s, lesson_sim_s=native['lesson_sim_s'],
        probe_sim_s=native['probe_sim_s'], replicated_sim_s=native['replicated_sim_s'],
        g4_base_sim_s=153_600, g4_cap_sim_s=230_400, diagnostic_factor=factor, basis=basis)
    notes = [f'status={status}',
        'interim u8 paired evaluation OFF: the registered final range is evaluated once, at u24',
        'smoke+gate charged to newmap2_smoke (<=40,000, pre-approved); not part of the training total',
        'owner cap ambiguity: coop_probe 250,000 (05:10 entry 7.3M->7.55M) is not in the 12:46-14:37 cap chain '
        '(7.3M->7.6M->...->8.2M); it is counted here as prior; coopkinds 200,000 counted as prior',
        'if short: drop B lessons (-%d) or base/native final arms (-%d each)' % (
            phases['v2v_lessons']['allocated_sim_s'], b.allocation(100*args.episode_sim_s))]
    doc = b.document(phases, prior=PRIOR, cap=args.cap,
                     created=datetime.now().astimezone().isoformat(timespec='seconds'), notes=notes)
    doc_path = Path(args.out)
    with doc_path.open('x', encoding='utf-8') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write('\n')
    print(json.dumps({k: doc[k] for k in ('total_training_sim_s', 'available_sim_s', 'fits',
                                          'shortfall_sim_s', 'receipt_sha256')}))


if __name__ == '__main__':
    main()
