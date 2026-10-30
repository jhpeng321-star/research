"""Append the owner's fixed training-only parity allocation, then bind its ACK."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from tools.candB.parity import JOBS

LEDGER = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
HEADER = '## 2026-10-30 S-CAPREL-PARITY ALLOCATION — registered BEFORE ANY SUMO'


def blob_sha(name):
    return hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:'+name])).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--amend-infrastructure', action='store_true')
    a = p.parse_args()
    if a.out.exists() or not a.out.parent.is_dir():
        raise ValueError('registration ACK must be fresh')
    registered = HEADER.encode() in LEDGER.read_bytes()
    amendment_header = '## S-CAPREL-PARITY infrastructure amendment — before further SUMO'
    if registered and not a.amend_infrastructure:
        raise ValueError('allocation already registered; no implicit repeat')
    if a.amend_infrastructure and (not registered or amendment_header.encode() in LEDGER.read_bytes()):
        raise ValueError('missing original or duplicate amendment')
    code = blob_sha('tools/candB/parity.py')
    laws = {n: blob_sha('src/wms/control/'+n) for n in ('residual.py', 'alwayscmd.py', 'sumo_idm.py')}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    text = '\n\n'+HEADER+'\n'+(
        '- Owner authorization 2026-10-30 16:30, S-INTEG step 5. Remaining exec-parity allocation '
        '**3,600 sim-s of the approved 7,200**, hard cap including warmup and failed attempts. '
        'One sequential CPU worker on **gpuR**, **nice 19**, OMP/MKL/OpenBLAS=1, CUDA hidden; '
        'no GPU job is touched, no cpuQ/gpuA job is touched. No training.\n'
        '- TRAIN seeds only: **963110** base-law CTRL0 zero and cap-relative zero '
        '(paired canary; two episodes), **963111/963112** cap-relative fixed **-0.01 m/s²** '
        'engineering probes (two episodes). Each episode is **300 s native warmup + 600 s '
        'intervention = 900 sim-s**; four episodes total. Demand 2400 veh/h, CAV share .20, '
        'textbook population, native lanes, speedMode27, .5 s physics, 1 s control. '
        'Never 964000-964199, 962xxx or 977xxx. No real-evaluation tape or truth/assignment file is read.\n'
        '- The CAPREL.md historical 977xxx/cpuP/cpuQ worker is incompatible with this task. '
        'Use tools/candB/parity.py on the integrated runtime instead: same shared law, '
        'actual CavSumoEnv/CommandOwnership sole writer, fresh paired training-seed zero identity. '
        'No production probes unless the paired zero canary is bitwise identical. '
        'No automatic retries; a failed attempt retains its full 900 s charge.\n'
        '- Registered PASS: bitwise zero CTRL0 scientific arrays AND pooled executed-acceleration '
        '**RMSE <= 0.1603 m/s²**, on the established stable-present/native-frame row filter. '
        'Retain exclusions, per-world counts/RMSE, source and episode/ledger hashes. '
        'The existing deployment receipt additionally requires the measured same-state native-head '
        'band; fixed-bound PASS alone is not promoted if that guard fails. No tuning after results.\n'
        '- Runtime **/root/autodl-tmp/c4/candB_caprel_20261030**, source bundle '
        '**53b73b2253f4cc1a7da1047f0454a97c33c7c13f** (2,309 source files verified); '
        'outputs **parity_run/** under that fresh root. Installed eclipse-sumo **1.27.1** '
        'and environment import verified before simulation. '
        f'Registration-tool commit **{commit}**, worker SHA256 **{code}**.\n'
        f'- Law SHA256 bindings: `{json.dumps(laws, sort_keys=True)}`. '
        'Worker records PID/start UTC, source manifest and append-only per-attempt budget before reset. '
        'Results and actual/charged seconds will be appended after completion.\n')
    if a.amend_infrastructure:
        text = '\n\n'+amendment_header+'\n'+(
            '- Original owned worker **436778**, start ticks4197738403, exited before '
            'the first simulationStep: reset failed importing the transitive '
            '`output.controller_verification_v7.pi_owner_adapter`. **Actual 0 sim-s**, '
            '**900 sim-s charge retained**, outputs/log/runtime preserved unchanged.\n'
            '- Explicit budget-preserving amendment: fresh runtime '
            '**/root/autodl-tmp/c4/candB_caprel_20261030_r2**, fresh **parity_run/**. '
            'Stage the three existing published PI Python dependencies; import the '
            'controller factory BEFORE starting/charging any simulation. No scientific '
            'law/driver/threshold/seed-selection change. SUMO_HOME points to the installed package.\n'
            '- Remaining **2,700 sim-s**: 963110 base-law zero, 963110 cap-relative zero, '
            'then ONLY 963111 fixed -0.01 probe, each300+600=900. The previously planned '
            '963112 probe is not run. Original charge900 + new cap2700 = **3,600 total**, '
            'no refund, no automatic retries. One worker, gpuR CPUs, nice19, CUDA hidden.\n'
            '- Bitwise paired-zero identity and fixed pooled RMSE<=0.1603 bound unchanged; '
            'one probe world is disclosed as limited execution-parity evidence, not efficacy. '
            f'Source/registration-tool commit **{commit}**, worker SHA256 **{code}**; '
            f'unchanged law SHA256 `{json.dumps(laws, sort_keys=True)}`. '
            'Full staged source manifest and CPU import preflight must pass before reset.\n')
    with LEDGER.open('ab', buffering=0) as stream:
        stream.write(text.encode('utf-8'))
        os.fsync(stream.fileno())
    digest = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    doc = dict(allocation='S-CAPREL-PARITY', host='gpuR', cap_sim_s=3600,
        pooled_rmse_bound=.1603, jobs=list(JOBS[:3] if a.amend_infrastructure else JOBS),
        registration_sha256=digest,
        source_sha256=code, caprel_source_sha256=laws, registration_commit=commit)
    if a.amend_infrastructure:
        doc['prior_charged_sim_s'] = 900
    with a.out.open('x', encoding='utf-8') as stream:
        json.dump(doc, stream, indent=1)
        stream.write('\n')
    print(json.dumps(doc, indent=1))


if __name__ == '__main__':
    main()
