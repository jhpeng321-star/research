"""Append one durable owner2305 receipt to the named shared registration."""
import json
import os
from pathlib import Path


def main():
    ledger = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    marker = 'S-PILOTB OWNER2305 OVERLAY ACK'
    assert marker not in ledger.read_text(encoding='utf-8')
    receipt = json.loads((Path(__file__).parent / 'pilot_deploy_2305_receipt.json').read_text())
    assert receipt['returncode'] == 0
    line = ('\n- '+marker+'23:08 +08: exploratory overlay5118d220 staged before any continuation dispatch; '
        'original scientific source da0bc366/input/config/WM/mix/parity authority independently PASS again '
        '(authority file SHA a089c5fb, canonical verification digest187cd8d4). Original dry985766/start874675005 '
        'and exclusive controller999762/start875186271 preserved; frozen baseline still computing,0 updates. '
        '14 CPU checks PASS, CUDA hidden. At first clean boundary, ordinary same-phase full-Adam/head/RNG '
        'continuation, byte-identical frozen baseline and original phase10 R4 retained; maximum safe forward '
        'parallelism requested4, bounded to3 on48GiB at.24 each when coordinator reservation permits '
        '(min90%/45GiB including contexts;2/serial fallback);3 dense workers with memory-bounded1..3 VJP slots. '
        'Adds raw group D_G/C_G/sum_D_i/D_all and confident-gain rows, executed acting-CAV braking minima '
        'and per-worker allocator peak/PID receipts; existing dJ/ES/gradient/s-update telemetry retained. '
        'Original live dry adapter untouched; ten-minute GPU polling; no new GPU job, SUMO, peer signal or push. '
        'SPEEDPROF currently diagnosis only; awaiting exact published application diff.\n')
    with ledger.open('a', encoding='utf-8') as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    print(marker)


if __name__ == '__main__':
    main()
