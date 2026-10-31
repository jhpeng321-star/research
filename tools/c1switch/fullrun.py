"""One-worker, precharged Step0 v2; preserved legacy STOP is never cleared."""
import json
import os
from pathlib import Path
import time
import traceback

from .design import CONTROLLERS, STEP0_SEEDS, STEP0_TIMES, worlds
from .rerun import candidates
from .storage import Ledger, canonical, exclusive, write


def main():
    root = Path.cwd().parent
    binding = json.loads((root/'BINDING.json').read_text())
    if binding['step0_total_sim_s'] != 47010 or binding['main_authorized']:
        raise ValueError('unregistered budget binding')
    with exclusive(root):
        from .sumo import Backend, setup
        inputs = setup()
        ledger = Ledger(root)
        if ledger.rows() != [binding['retained_debit']]:
            raise ValueError('existing attempt; no retry')
        jobs = root/'jobs'
        jobs.mkdir(exist_ok=False)
        all_rows = []
        try:
            for world in worlds('step0'):
                if world['seed'] not in STEP0_SEEDS:
                    continue
                for decision in STEP0_TIMES:
                    # Yield at state boundaries and before each fresh prefix;
                    # never signal or inspect a peer process.
                    while os.getloadavg()[0] > 20 or (root/'YIELD').exists():
                        write(root/'STATUS.json', dict(status='YIELD', world=world, decision=decision))
                        time.sleep(60)
                    ident = f's{world["seed"]}_t{decision}'
                    write(root/'STATUS.json', dict(status='RUNNING', world=world, decision=decision))
                    def factory_for(name):
                        def create():
                            if (root/'STOP').exists():
                                raise ValueError('durable STOP')
                            while os.getloadavg()[0] > 20 or (root/'YIELD').exists():
                                time.sleep(60)
                            job = jobs/f'{ident}_{name}'
                            job.mkdir(exist_ok=False)
                            write(job/'START.json', dict(world=world, t_s=decision, alternative=name,
                                                         pid=os.getpid(), unix=time.time()))
                            return Backend(job, world, ledger, 'step0', f'{ident}_{name}', inputs)
                        return create
                    rows = candidates(factory_for, ['native']*((decision-300)//60))
                    write(root/f'{ident}.json', dict(world=world, t_s=decision, branches=rows))
                    all_rows.append(dict(world=world, t_s=decision, branches=rows))
                    write(root/'TRUNK_GATE.json', dict(pass_=True, states=len(all_rows),
                                                       receipt_sha256=canonical(rows)))
            noise = max(abs(r['branches']['native']['90']['team_delay_vs']-
                            r['branches']['native_repeat']['90']['team_delay_vs']) for r in all_rows)
            threshold = max(1e-6, 2*noise)
            wins = {clock:dict.fromkeys(CONTROLLERS, 0) for clock in ('60','90')}
            resolved_other = dict.fromkeys(('60','90'), 0)
            # Dominant identity defined from the full panel separately at each clock.
            for clock in wins:
                for row in all_rows:
                    costs = sorted((row['branches'][c][clock]['team_delay_vs'], c) for c in CONTROLLERS)
                    if costs[1][0]-costs[0][0] > 1e-6:
                        wins[clock][costs[0][1]] += 1
                dominant = max(CONTROLLERS, key=wins[clock].get)
                for row in all_rows:
                    costs = sorted((row['branches'][c][clock]['team_delay_vs'], c) for c in CONTROLLERS)
                    resolved_other[clock] += costs[0][1] != dominant and costs[1][0]-costs[0][0] > threshold
            nearly_everywhere = any(wins['60'][c] >= .95*16 and wins['90'][c] >= .95*16 for c in CONTROLLERS)
            unresolved = any(n < .05*16 for n in resolved_other.values())
            gate = dict(pass_=not (nearly_everywhere or unresolved), unique_wins=wins,
                        native_repeat_noise_vs=noise, margin_threshold_vs=threshold,
                        resolved_non_dominant=resolved_other, states=len(all_rows),
                        charged_sim_s=sum(r['sim_s'] for r in ledger.rows()), main_authorized=False)
            write(root/'STEP0_GATE.json', gate)
            write(root/'STATUS.json', dict(status='COMPLETE', **gate))
        except BaseException:
            write(root/'STOP', dict(error=traceback.format_exc(), pid=os.getpid(),
                                   charged_sim_s=sum(r['sim_s'] for r in ledger.rows())))
            raise


if __name__ == '__main__':
    main()
