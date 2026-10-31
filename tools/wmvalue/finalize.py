"""Generate and commit the once-only frozen verdict after all inference completes."""
import argparse
import json
from pathlib import Path
import subprocess

from .audit import digest, safe


def policy_plan(result, task_root):
    """Explicit conditional engineering budget, based on measured inference cost."""
    rates = []
    for name in result['task_sha256']:
        d = json.loads((task_root/name).read_text())
        if d['device'] == 'cuda':
            rates.append(d['rollout_seconds']/(5*(900-d['t0'])))
    benchmark = 'this test: frozen WM2+mix GPU tasks'
    if not rates:
        # Throughput evidence is independent of real outcomes / efficacy. The
        # already audited cached A is still a valid old GPU timing measurement.
        reference = safe('E:/agent-work/c4-wmcalib/tools/wmcalib/data/wm40/wm_gpu_shard3/w32_k0.json')
        assert digest(reference) == 'b4501bc129c2039407c9fef4bd5f50d1580946ae2a277cd697e8cfc1ca6b1c70'
        d = json.loads(reference.read_text())
        rates = [d['seconds']/(len(d['arms'])*d['steps']*.5)]
        benchmark = 'prior WMCALIB WM2+mix GPU w32_k0,484.714s/4arms/600s; includes contention'
    import numpy as np
    horizon = 60 if 'B' in result['winners'] else 90
    # Prefer the cheaper horizon only when it passes the complete gate.
    # 25 updates, 1 state, K8, 2 ES directions (4 signed perturbations),
    # paired two-CAV group credit (both, solo1, solo2, none): 3200 arm rollouts.
    lo, hi = np.percentile(rates, [25, 75])
    hours_lo, hours_hi = 3200*horizon*lo/3600, 2*3200*horizon*hi/3600
    return '\n'.join([
        '# Conditional WMVALUE policy integration plan (PASS only)', '',
        'A separate owner-visible registration is required before any policy run.',
        '1. In c4-pilotdiag/c4-candB policy14_cli, keep WM2+mix frozen. Replace',
        f'   the300s ES long-credit scorer with the passing{horizon}s imagination.',
        '2. Score each future as accrued full-team cost + frozen V(projected terminal).',
        '   Average K8 AFTER value evaluation. Reuse the audited entry/cost/CRN adapter.',
        '3. For a two-CAV group use both/solo1/solo2/none WM counterfactuals.',
        '   Group marginal gain = J_none-J_both; cooperation interaction =',
        '   J_solo1+J_solo2-J_both-J_none. Use the same terminal-V scorer for all.',
        '4. Keep scalar ES through this tree V (no differentiability claim). Retain',
        '   short-credit/safety terms, and pre-register actor-credit weights before runs.',
        '5. Initial engineering pilot:25updates,1FITstate/update,K8,2ESdirections',
        f'   (4signed perturbations),4group variants =3200arm rollouts at{horizon}s.',
        f'   Measured forward-cost estimate: {hours_lo:.2f}..{hours_hi:.2f} GPU-hours.',
        '   Benchmark: '+benchmark+'.',
        '   Lower = benchmark GPU per-arm-second p25; upper =2x p75, including',
        '   host/controller overhead. This is a throughput estimate, not an allocation.',
        '   Re-benchmark 5FIT-only updates before registering a longer training budget.',
        '6. Freeze new development worlds before selecting checkpoints. Use real',
        '   SUMO only in a separately authorized verifier; compare to the oracle',
        '   teacher and report fraction of oracle gain. Do not reuse these exposed',
        '   eight reserved FIT worlds for any subsequent tuning.',
        '', 'Inference receipts bind source throughput; no policy job launched here.',
    ])+'\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', required=True)
    p.add_argument('--real', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--commit', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = root/'tools/wmvalue/evidence'
    result_path = evidence/'frozen_test.json'
    report = root/'docs/paper_results/WMVALUE.md'
    if result_path.exists():
        raise ValueError('frozen test already executed; never repeat')
    command = [__import__('sys').executable, '-B', '-m', 'tools.wmvalue.evaluate_frozen',
        '--tasks', str(safe(a.tasks)), '--real', str(safe(a.real)), '--model', str(safe(a.model)),
        '--out', str(result_path), '--report', str(report)]
    subprocess.run(command, cwd=root, check=True)
    result = json.loads(result_path.read_text())
    files = [report, result_path, evidence/'arm_predictions.json']
    if result['verdict'] == 'PASS':
        plan = policy_plan(result, safe(a.tasks))
        if plan is not None:
            path = root/'docs/paper_results/WMVALUE_POLICY_PLAN.md'
            with path.open('x') as stream:
                stream.write(plan)
            files.append(path)
    line = (f"- S-WMVALUE FROZEN DECISIVE VERDICT {result['verdict']} (c4/wmvalue; "
            '8reservedFITworlds969432..969439,200arm cells,400pairs,K8,starts300..540 to900): '
            'V/D/E preserved SHAef3e7bce,111causal sensor/flow features,no refit. '
            f"Passing short methods={result['winners']}; strict world-bootstrap10000 gate "
            'requires BOTH sign and MAE positive95% lower bounds against EACH A/D/E. '
            'Complete table/contrasts in docs/paper_results/WMVALUE.md; '
            f"resultSHA{digest(result_path)}. No SUMO/eval/prohibited paths/policy training/peer signals/push. "
            'Any policy pilot needs separate owner-visible registration.\n')
    registration = safe('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    with registration.open('a', encoding='utf-8') as stream:
        stream.write(line)
    receipt = evidence/'final_verdict_registration.md'
    with receipt.open('x') as stream:
        stream.write(line)
    files.append(receipt)
    if a.commit:
        # Do not capture pre-existing staged user work in an unattended commit.
        staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root, text=True)
        if staged.strip():
            raise RuntimeError('results written; commit paused because unrelated staged work exists')
        subprocess.run(['git', 'status', '--short'], cwd=root, check=True)
        subprocess.run(['git', 'diff', '--', str(report.relative_to(root))], cwd=root, check=True)
        subprocess.run(['git', 'log', '--oneline', '-10'], cwd=root, check=True)
        subprocess.run(['git', 'add', *[str(path.relative_to(root)) for path in files]], cwd=root, check=True)
        subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, check=True)
        subprocess.run(['git', 'commit', '-m', 'WMVALUE: report frozen decisive '+result['verdict']+' verdict'],
                       cwd=root, check=True)
    print('FROZEN VERDICT DELIVERED', result['verdict'], flush=True)


if __name__ == '__main__':
    main()
