"""Append the completed, budget-bound parity receipt to the named shared ledger."""
import hashlib
import json
import os
from pathlib import Path

from tools.candB.register_parity import LEDGER


def main():
    root = Path('tools/candB/evidence')
    report_path = root/'REPORT.json'
    report = json.loads(report_path.read_text())
    identity = json.loads((root/'CTRL0_IDENTITY.json').read_text())
    receipt = json.loads((root/'CAPREL_PARITY.json').read_text())
    assert report['passed'] and identity['bit_identical'] and receipt['passed']
    assert report['charged_sim_s'] == 3600 and report['actual_sim_s'] == 2700
    assert report['pooled_rmse'] <= report['registered_bound'] == .1603
    header = '## S-CAPREL-PARITY RESULT — allocation complete'
    assert header.encode() not in LEDGER.read_bytes(), 'result already appended'
    text = '\n\n'+header+'\n'+(
        '- **PASS**: paired training-seed963110 zero-output scientific arrays '
        'bitwise identical; pooled executed acceleration RMSE '
        f'**{report["pooled_rmse"]:.12f} <= 0.1603 m/s²** on '
        '**18,702** stable rows of the fixed -0.01 probe963111. '
        '104 nonstable rows excluded by the established filter; zero speed/ID mismatches. '
        'Same-state native-head RMSE **0.127698706935**; the existing '
        '`wms.control.execution.verify_parity` deployment guard also **PASS** on the LF runtime.\n'
        '- Charged **3,600 sim-s total** (failed900 + successful2700), actual **2,700**. '
        'No remaining S-CAPREL-PARITY budget; no further SUMO under this allocation. '
        'Single nice19 CPU worker, eclipse-sumo1.27.1, native lanes, all training seeds. '
        'Owned worker437069/start ticks4197765052 independently verified exited; '
        'original436778 also exited. No other process was signalled; no GPU jobs touched.\n'
        '- Remote immutable evidence: gpuR '
        '`/root/autodl-tmp/c4/candB_caprel_20261030_r2/parity_run/`; '
        'small JSON/identity/budget/ownership/log receipts copied into '
        '`E:/agent-work/c4-candB/tools/candB/evidence/`. '
        f'REPORT SHA256 **{hashlib.sha256(report_path.read_bytes()).hexdigest()}**; '
        f'amended registration SHA256 **{report["registration_sha256"]}**. '
        'One probe world is limited execution-parity evidence, not efficacy or WM qualification.\n')
    with LEDGER.open('ab', buffering=0) as stream:
        stream.write(text.encode('utf-8'))
        os.fsync(stream.fileno())
    print(header)


if __name__ == '__main__':
    main()
