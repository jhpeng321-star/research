"""Append one durable owner2305 receipt to the named shared registration."""
import json
import os
from pathlib import Path
import sys


def main():
    ledger = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    if sys.argv[1:] == ['--monitor']:
        marker = 'S-PILOTB OWNER2305 OBSERVER ACK'
        assert marker not in ledger.read_text(encoding='utf-8')
        lease = json.loads((Path(__file__).parent / 'pilot_watch_2305_OWNERSHIP.json').read_text())
        line = ('\n- '+marker+'23:17 +08: exclusive local observer'+str(lease['pid'])+
            '/creation23:13:43 has taken over after exact-owned predecessor43860 cooperatively exited '
            '(Windows process handle waited; no signals). Receipt tools/candB/pilot_watch_2305_OWNERSHIP.json. '
            'It writes per-update PILOT/JSON metrics and commits, launch/every5 registration entries, '
            'and pilot_update3_ACK.json only after actual phase3; monitoring continues thereafter for MAIN. '
            'First observation23:17:01 dry0/pilot0, baseline still computing; phase3 requirement PENDING. '
            'Remote trainer985766 and controller999762 preserved. Final operational code5118d220 + '
            'native-budget correction07899bcf;16 CPU PASS. Ordinary native continuation owns the '
            '150-total phase budget (149 remaining after1), exact baseline/full-Adam preservation and original R4. '
            'SPEEDPROF still has no exact application diff published. No duplicate GPU dispatch/SUMO/push.\n')
        with ledger.open('a', encoding='utf-8') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        print(marker)
        return
    assert not sys.argv[1:]
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
