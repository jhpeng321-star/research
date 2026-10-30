"""Verify a completed measurement and render the requested <=70-line report, offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.candB import rankcheck
from tools.distill.runtime import check_metadata, sha256


def verify(document):
    assert document['complete'] and document['measure_only']
    assert document['passed'] is False and document['launch_authorized'] is False
    assert document['n_starts'] >= 16 and len(document['rows']) == document['n_starts']
    assert document['K'] == 8 and document['horizons_s'] == [90, 300]
    assert document['wm_sha256'] == rankcheck.WM_SHA
    assert document['mappo_sha256'] == rankcheck.MAPPO_SHA
    assert document['mix_connected'] and sum(map(sum, document['mix_assignment_receipts'])) > 0
    assert document['gpu_memory_fraction'] <= .40
    assert document['mappo_speed_partial']['iteration'] == 25
    assert document['mappo_speed_partial']['sha256'] == (
        '69e414e53e7f520bbd112ff292aa207d3da6388d73c3ac91e894079dfe40a543')
    for row in document['rows']:
        check_metadata(row['metadata'])
        assert len(row['metadata']) == 1
        assert set(row['arms']) == set(document['arms'])
        for arm in row['arms'].values():
            assert len(arm['J90']) == len(arm['J300']) == 8
    summary, worlds = rankcheck.summarize(document['rows'], document['arms'])
    assert worlds == document['n_worlds']
    for metric in summary:
        for group in ('arms', 'vs_CTRL0'):
            for stat in ('mean', 'lo', 'hi'):
                np.testing.assert_allclose(summary[metric][group][stat],
                    document['summary'][metric][group][stat], rtol=1e-12, atol=1e-9)
    return summary


def number(stats, index, *, signed=False):
    sign = '+' if signed else ''
    return (f"{stats['mean'][index]:{sign}.1f} "
            f"[{stats['lo'][index]:{sign}.1f},{stats['hi'][index]:{sign}.1f}]")


def report(document, evidence, source_sha, final_sha):
    summary = verify(document)
    arms = document['arms']
    labels = dict(native='native', CTRL0='CTRL0', MAPPO_native_lanes='MAPPO proxy aecd7d70',
        CIRCLES='CIRCLES damping', PI='PI external-U', current='current residual (u191)',
        candB_zero='fresh-zero B', MAPPO_speed_partial='speed-only MAPPO partial i25')
    def ranking(metric):
        return ' < '.join(labels[arms[i]] for i in
            sorted(range(len(arms)), key=lambda i: summary[metric]['arms']['mean'][i]))
    caps = sorted({row['arms']['CTRL0']['dense_horizon_s'] for row in document['rows']})
    proxy = arms.index('MAPPO_native_lanes')
    native = arms.index('native')
    native_delta = summary['J300']['vs_CTRL0']
    ctrl0_native = (-native_delta['mean'][native], -native_delta['hi'][native],
                   -native_delta['lo'][native])
    condition = summary['J300']['vs_CTRL0']['hi'][proxy] <= 0
    reason = ('HOLDS: registered proxy upper95 dJ300 <=0.' if condition else
              'NOT DEMONSTRATED: proxy upper95 dJ300 >0; no numeric real speed-only PI/CIRCLES comparison is available for the alternative.')
    lines = [
        '# S-RANKCHECK — exact candidate-B WM ranking (measurement only)',
        f'Completed {document["n_starts"]} consecutive actual TRAIN-sampler starts / {document["n_worlds"]} worlds; no training or SUMO.',
        f'Final measured source SHA: `{source_sha}`; final code+evidence SHA: `{final_sha}`.',
        'Report-delivery SHA: `git log -1 --format=%H -- docs/paper_results/RANKCHECK.md`.',
        f'Raw evidence: `{evidence.as_posix()}` (SHA256 `{sha256(evidence)}`).',
        'WM2 `b8797242` + learned mix ON; paired K8; CTRL0 long reference; all requested vehicles + future entrants + road deficit + source waiting.',
        f'Full resolved config SHA256 `{document["config_sha256"]}`; inference-config SHA256 `{document["wm_config_sha256"]}`.',
        'Native lanes throughout; MAPPO proxy is lane-on aecd7d70 with only its speed output executed; PI/CIRCLES reuse the published TimelineControllers bridge.',
        'cpuQ read-only copy: `~/mappospeed/production2/policy_00025.pt`, iteration25/169, PARTIALLY TRAINED; frozen at first inspection, no outcome selection.',
        'Partial export SHA256 `69e414e53e7f520bbd112ff292aa207d3da6388d73c3ac91e894079dfe40a543`; no lane-head tensors, lane_control=native.',
        'Shared paired WORLD-cluster percentile CIs: 5,000 draws, seed20261030; average K draws/repeated starts within world, then equal-world mean. All starts retained.',
        '## Physical J (veh-s; lower is better)',
        'Every cell is mean [95% CI]; dJ=treatment-CTRL0. J90/J300 use the same 300s closed-loop futures.',
        '| Controller | J90 | dJ90 vs CTRL0 | J300 | dJ300 vs CTRL0 |',
        '|---|---:|---:|---:|---:|',
    ]
    for i, arm in enumerate(arms):
        fields = [labels[arm]]
        for metric in ('J90', 'J300'):
            fields += [number(summary[metric]['arms'], i),
                       number(summary[metric]['vs_CTRL0'], i, signed=True)]
        lines.append('| '+' | '.join(fields)+' |')
    lines += [f'Physical ranking, 90s: {ranking("J90")}.',
              f'Physical ranking, 300s: {ranking("J300")}.',
              'Rankings order point estimates; overlapping intervals do not establish every pairwise order.',
              '## Actual actor objective (loss at scale1; lower is better)',
              f'L90=dense suffix/joint-q10 loss at the TRAIN-resolved nominal90s regime (observed dense caps {caps}s); L300=that dense loss + registered gain-gated accrued300s loss.',
              'These are weighted actor losses, not physical J; no critic/bootstrap, entropy or safety-Lagrange term is added. Every cell is mean [95% world-cluster CI].',
              '| Controller | L90 | dL90 vs CTRL0 | L300 | dL300 vs CTRL0 |',
              '|---|---:|---:|---:|---:|']
    for i, arm in enumerate(arms):
        fields = [labels[arm]]
        for metric in ('objective90_loss', 'objective300_loss'):
            fields += [number(summary[metric]['arms'], i),
                       number(summary[metric]['vs_CTRL0'], i, signed=True)]
        lines.append('| '+' | '.join(fields)+' |')
    lines += [f'Actor-loss ranking, dense: {ranking("objective90_loss")}.',
              f'Actor-loss ranking, dense+300s: {ranking("objective300_loss")}.',
              '## Existing real-SUMO evidence and registered R2',
              'CTRLARM (600s, 23 reused exploratory worlds): CTRL0-native -1,956.53 [-4,086.66,+144.50]; lane-on MAPPO-CTRL0 -623.95 [-2,044.25,+664.51] veh-s; both CIs include0.',
              'Exact source: `E:/research-dense/docs/paper_results/CTRLARM.md:9-17` (published aggregate only; no underlying evaluation tapes read).',
              f'WM CTRL0-native at300s: {ctrl0_native[0]:+.1f} [{ctrl0_native[1]:+.1f},{ctrl0_native[2]:+.1f}] veh-s; its point direction {"agrees with" if ctrl0_native[0] < 0 else "opposes"} CTRLARM, whose real interval includes0.',
              f'WM proxy-CTRL0 at300s point direction {"agrees with" if native_delta["mean"][proxy] < 0 else "opposes"} the historical lane-on MAPPO-CTRL0 point estimate; lane-control and horizon mismatches prevent treating this as matched evidence.',
              'Broader lane-on MAPPO-native: 300s -437.8 [-632.0,-241.4], 600s -1,777.4 [-2,336.8,-1,231.3] veh-s; source `E:/research-dense/docs/paper_results/E3FINAL_A.md:24,30`.',
              'These lane-on/600s results are contextual, not a matched native-lane/300s CTRL0 qualification. A true speed-only partial checkpoint has no published real result here.',
              'No numeric PI/CIRCLES SUMO contrast found in either named paper_results directory or the supplied registrations. `C4-V5-20261014/REGISTRATION.md:96-99` only registers damping/PI arms; C4-EVAL directories contain no REGISTRATION.md.',
              'Binding rule: `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md:4951-4956` (2026-10-30 16:30 R2).',
              f'**R2 numerical condition: {reason}**',
              f'Tool decision: `{document["ranking_passed"]}` — {document["gate_reason"]}.',
              'Measurement-only receipt: passed=false, launch_authorized=false; historical mismatched CCT-2 is not upgraded. Full training qualification remains separate.',
              '## Execution and verification',
              'Fresh immutable staging via tools/candB/stage.sh; one serial CUDA worker, allocator fraction.40; existing S-WMREQUAL PID938487 untouched. No other job signalled, no push/deletions.',
              'Checkpoint initialization verifies original CRLF driver bytes f70c2aaf; exact CRLF-to-LF equality restores committed driver16bd8278 for every rollout. Both full hashes are in the receipt.',
              f'Exact forward flags: `{document["execution_speed_flags"]}`; full argv/environment/source manifest and retained attempts are beside the raw evidence.',
              'Verification: 14 focused CPU tests pass, Ruff/diff checks pass; complete numerical summary independently recomputed from raw starts/futures before publication.',
              ]
    assert len(lines) <= 70, len(lines)
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--final-sha', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.evidence.read_text())
    text = report(document, args.evidence, args.source_sha, args.final_sha)
    args.out.write_text(text, encoding='utf-8')
    print(f'Wrote {args.out}: {len(text.splitlines())} lines; ranking_passed={document["ranking_passed"]}')


if __name__ == '__main__':
    main()
