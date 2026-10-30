"""Durably receipt the actual failed update without dispatch or baseline scoring."""
import json
from pathlib import Path

from tools.candB.pilot_lean_observe import append, DIRECTORY, LEDGER


def main():
    doc = json.loads((DIRECTORY/'pilot_wmfast_latest.json').read_text())
    data = doc['outputs']['pilot_EXPLORATORY-LEAN']
    summary = data['files']['RUN_SUMMARY.json']
    assert summary['status'] == 'failed' and not data['rows']
    assert summary['error'] == "TypeError: 'NoneType' object is not subscriptable"
    assert 'frozen_scoring_baseline.json' in data['files']
    marker = DIRECTORY/'pilot_repaired_UPDATE1_FAILURE.json'
    with marker.open('x') as stream:
        json.dump(doc, stream, indent=1)
    text = ('\n- S-PILOTB EXPLORATORY-LEAN UPDATE1 FAILED epoch1793385850: '
        "TypeError: 'NoneType' object is not subscriptable in dense autograd backward; "
        'ZERO completed optimizer rows. Frozen baseline744b30b8 remains durable; '
        'all original outputs/startup scores retained. Trainer35928 and supervisor35923 '
        'exited naturally; old S2 waiter37982 and prepared S4 waiter60032 both refused '
        'the non-clean parent, no continuation trainer dispatched. S4 runtimeaf78fda7 '
        'authority07f5e05b and10 CPU contracts passed but S4 production not launched. '
        'Investigating a baseline-preserving recovery; NO baseline recomputation or '
        'completed-update claim. Full traceback/receipt '
        '`E:/agent-work/c4-candB/tools/candB/pilot_repaired_UPDATE1_FAILURE.json`.\n')
    append(DIRECTORY/'PILOT.md', text)
    append(LEDGER, text)
    print(summary['traceback'])


if __name__ == '__main__':
    main()
