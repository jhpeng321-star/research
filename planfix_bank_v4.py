"""Remainder-only bank continuation while clean1 is explicitly owner-stopped."""
import json
from pathlib import Path
import planfix_run_bank as bank


def inactive_training(root):
    root=Path(root)
    stop=root/'STOPPED_BY_OWNER_CLEAN2_RESTART.json'
    if not stop.is_file() or json.loads(stop.read_text()).get('owner_verbatim')!='restart':
        return False
    if not (root/'STOP.json').is_file():
        return False
    if (root.parent/'coopmwm20261105_clean2').exists():
        return False
    owners=json.loads((root/'RESUMED_1434.json').read_text())['owners']
    for owner in owners.values():
        path=Path('/proc')/str(owner['pid'])/'stat'
        try:
            fields=path.read_text().split()
        except FileNotFoundError:
            continue
        if fields[21]==owner['start_ticks'] and fields[2]!='Z':
            return False
    return True


def pace_while_owner_stopped():
    bank.own_health()
    if not inactive_training(bank.ROOT):
        raise RuntimeError('PLANFIX remainder hold: training changed; require fresh active-training pace registration')


if __name__=='__main__':
    bank.ARTIFACTS='bank_verification_v4'
    bank.check_pace=pace_while_owner_stopped
    bank.main()
