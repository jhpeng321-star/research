"""Single detached preparation -> sequential training/scoring -> mirror -> DONE."""
import json
import os
from pathlib import Path
import sys
import traceback

from common import CONFIG, ID, OUT, RT, VARIANTS, disk, event, mirror, now, run, write


def score(design, variant):
    from paired_statistics import bias, compare
    checkpoint = OUT/variant/'wm_s.pt'
    gate = OUT/(variant+'_gate.json')
    code = run([sys.executable, '-B', str(RT/'tools/coopw/horizon_gate.py'),
                '--wm-ckpt', str(checkpoint), '--wm-config', str(CONFIG),
                '--pairs', design['score_pairs'], '--groups', '48',
                '--horizons', '10', '30', '60', '90', '120', '180', '300',
                '--threshold', str(RT/'configs/coopw_horizon_gate_13E_b2.json'),
                '--out', str(gate)], variant+'_gate')
    mirror()
    acceptance = run([sys.executable, '-B', str(RT/'tools/loop14_checks/accept.py'),
                      '--candidate', str(checkpoint), '--reference', str(OUT/'wm1/wm_s.pt'),
                      '--config', str(CONFIG), '--roots', design['score_loop'],
                      '--pairs', design['score_loop'], '--panel', str(OUT/'accept_panel.json'),
                      '--out', str(OUT/(variant+'_accept')), '--onestep-windows', '16',
                      '--rollout-windows', '16', '--workers', '1'], variant+'_accept')
    result = {'verdict': 'unavailable', 'gate_exit': code, 'accept_exit': acceptance}
    if code == 0:
        reference = json.loads((OUT/'wm1_gate.json').read_text())
        candidate = json.loads(gate.read_text())
        result.update(compare(reference, candidate))
        result['paired_bias_and_spread'] = {'wm1': bias(reference), variant: bias(candidate)}
    write(variant+'/RESULTS.json', result)
    return result


def main():
    status, error, results = 'failed', None, {}
    try:
        token = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write('RUNNING.json', {'pid': os.getpid(), 'start_ticks': token, 'started': now(),
              'session': os.getsid(0), 'ppid': os.getppid(), 'stdin': os.readlink('/proc/self/fd/0')})
        mirror()
        if not (OUT/'DESIGN.json').is_file():
            assert run([sys.executable, '-B', str(OUT/'code/prepare.py')], 'prepare', 1800) == 0
        design = json.loads((OUT/'DESIGN.json').read_text())
        mirror()
        for variant in VARIANTS:
            disk()
            write('STATE.json', {'phase': 'training', 'variant': variant, 't': now()})
            code = run([sys.executable, '-B', str(OUT/'code/train.py'), variant],
                       variant+'_train', 12600)  # hard backstop; SIGALRM enforces 10800 s
            mirror()
            if (OUT/variant/'wm_s.pt').is_file():
                write('STATE.json', {'phase': 'scoring', 'variant': variant, 't': now()})
                results[variant] = score(design, variant)
            else:
                results[variant] = {'verdict': 'unavailable', 'train_exit': code}
            write('RESULTS.json', results)
            mirror()
        status = 'complete' if all(r['verdict'] != 'unavailable' for r in results.values()) else 'partial'
    except BaseException:
        error = traceback.format_exc()
        write('ERROR.json', {'error': error})
        event(error=error)
    finally:
        receipt = {'id': ID, 'status': status, 'finished': now(),
                   'pid': os.getpid(), 'error': error, 'results': results}
        receipt['mirror'] = 'complete' if mirror() else 'failed_see_EVENTS'
        write('DONE.json', receipt)
        mirror()


if __name__ == '__main__':
    main()
