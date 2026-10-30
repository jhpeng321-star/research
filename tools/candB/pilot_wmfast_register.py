"""Append a single recovery/pre-dispatch disclosure to the named registration."""
from datetime import datetime, timezone
import json
from pathlib import Path

from tools.candB.pilot_lean_observe import append, LEDGER, DIRECTORY


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    if args.control:
        marker = DIRECTORY/'pilot_wmfast_CONTROL_REGISTERED.json'
        assert not marker.exists(), 'already registered'
        launch = json.loads((DIRECTORY/'pilot_wmfast_CONTROL_DISPATCH.json').read_text())
        assert launch['returncode'] == 0
        doc = dict(label='EXPLORATORY-LEAN', utc=datetime.now(timezone.utc).isoformat(),
            runtime=launch['runtime'], controller=json.loads(launch['stdout']),
            source='61bfa61236bbce4054994294cbe17350f1dfdccf', files=2625,
            authority_sha256='cc7dfb5bcd9075f872cc74a5a1afbd39c4fb3b30ade125b4b0e12cddfefd9096',
            CPU='14 PASS,1.56s,CUDA hidden, actual CLI specialization handles unscored update1',
            policy='wait existing repair35928 and supervisor35923 EXIT; require clean finite phase1 '
                'and durable baseline; no duplicate trainer; ordinary resume with last ACTUALLY '
                'measured independent score, full Adam/head/RNG/sampler/origin191/R4phase10/150total',
            unchanged_live_repair=True, no_extra_GPU_query=True)
        append(LEDGER, '\n- S-PILOTB EXPLORATORY-LEAN WMFAST BOUNDARY WAITER '+json.dumps(doc)+'\n')
        with marker.open('x') as stream:
            json.dump(doc,stream,indent=1)
        print(json.dumps(doc)); return
    marker = DIRECTORY/'pilot_wmfast_RECOVERY_REGISTERED.json'
    assert not marker.exists(), 'already registered; recover marker'
    doc = dict(label='EXPLORATORY-LEAN', utc=datetime.now(timezone.utc).isoformat(),
        old_exit_epoch=1793379795.4174902, old_trainer=17859, old_phase_updates=0,
        error='native gate requires16 despite approved lean panel4', sunk_wall_s=3216.882277,
        durable_frozen_baseline=False, saved_independent_score_reused=True,
        lost_native_panel='four scored results not persisted by old adapter; no complete baseline existed',
        source='e113760fce3b9e10a0e604611675a3f46526c314', files=2609,
        runtime='/root/autodl-tmp/c4/candB_pilot_WMFAST_EXPLORATORY-LEAN_20261031',
        lean_authority_sha256='cd0c98015ccb562ae91af5d1f898db9b1edae0eae567ec599f686db821ed0c96',
        wmfast_authority_sha256='1edba7a95b6386fb354ea256065107526349e25f36fc7d45762cfa404b52abe8',
        CPU='13 PASS,1.51s,CUDA hidden; gate, real resume loader/budget, original score reuse without forward',
        plan='repair native panel4 only with durable per-score receipts; original ES OFF to update1; '
            'own boundary STOP then S2 one .50 forward worker, ordinary full continuation; '
            'same baseline bytes/Adam/head/RNG/TRAIN sampler/origin191/R4phase10/150total',
        constraints='GPU polling>=600s; <=45GiB/.90 including peers+growth; peers untouched; no SUMO/push/deletion',
        rescue='owned CPU pending-call test refused ptrace; trainer never attached or signalled; no repeat')
    append(LEDGER, '\n- S-PILOTB EXPLORATORY-LEAN STARTUP FAILURE / BOUND RECOVERY '+json.dumps(doc)+'\n')
    with marker.open('x') as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps(doc))


if __name__ == '__main__':
    main()
