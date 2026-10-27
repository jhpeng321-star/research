"""Single detached preparation -> training -> long-horizon scoring -> mirror -> DONE.

Scoring (all held-out): the E3DATA validation worlds (36, never fitted) via
tools/wmcombined/e3_score.py and the registered pairs300 48-group horizon-gate panel
for WM2 (reference) and the selected-best + final-step candidates; accept.py one-step
checks. The candidates carry the (zero-at-init) mainline head, so both accept arms run
under the candidate's resolved config: the reference is ``combined/wm_s_initial.pt``
(step-0 save = WM2 weights + exactly-zero head = bit-identical WM2 behaviour), never a
modified registered tool. Verdicts: the owner-amended long-horizon rule (registered
21:38, primary) and the original registered rule, via the unchanged S-LONGLOSSTEST
tooling on both panels.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from common import E3, ID, LOOP_CONFIG, OUT, RT, THRESH, WM2_SHA256
from common import event, mirror, now, run, write

HORIZONS = ('10', '30', '60', '90', '120', '180', '300')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def gate(ckpt, config, out, label):
    design = json.loads((OUT/'DESIGN.json').read_text())
    code = run([sys.executable, '-B', str(RT/'tools/coopw/horizon_gate.py'),
                '--wm-ckpt', str(ckpt), '--wm-config', str(config),
                '--pairs', design['score_pairs'], '--groups', '48',
                '--horizons', *HORIZONS, '--threshold', str(THRESH),
                '--out', str(out)], label)
    if code != 0:
        raise RuntimeError(f'{label} horizon gate exit {code}')
    return json.loads(Path(out).read_text())


def e3_gate(ckpt, config, role, out, label, device='cuda'):
    code = run([sys.executable, '-B', str(OUT/'code/e3_score.py'), '--role', role,
                '--wm-ckpt', str(ckpt), '--wm-config', str(config),
                '--pairs', str(E3), '--threshold', str(THRESH),
                '--device', device, '--out', str(out)], label)
    if code != 0:
        raise RuntimeError(f'{label} exit {code}')
    return json.loads(Path(out).read_text())


def accept(ckpt, resolved, label):
    design = json.loads((OUT/'DESIGN.json').read_text())
    out = OUT/(label+'_accept')
    code = run([sys.executable, '-B', str(RT/'tools/loop14_checks/accept.py'),
                '--candidate', str(ckpt), '--reference', str(OUT/'combined/wm_s_initial.pt'),
                '--config', str(resolved), '--roots', design['score_loop'],
                '--pairs', design['score_loop'], '--panel', str(OUT/'accept_panel.json'),
                '--out', str(out), '--onestep-windows', '16',
                '--rollout-windows', '16', '--workers', '1'], label)
    return {'exit': code, 'out': str(out),
            'reference': 'combined/wm_s_initial.pt (= WM2 weights + zero mainline head, '
                         'bit-identical behaviour)'}


def amended(reference, candidate, out):
    sys.path.insert(0, str(OUT/'code'))
    from amended import evaluate as amended_evaluate
    doc = amended_evaluate(reference, candidate)
    Path(out).write_text(json.dumps(doc, indent=1))
    return doc


def summarize_gate(doc):
    return {'threshold': doc.get('threshold'),
            'sign_agreement': {h: v.get('sign_agreement')
                               for h, v in doc.get('horizons', {}).items()}}


def main():
    status, error, results = 'failed', None, {}
    try:
        token = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write('RUNNING.json', {'pid': os.getpid(), 'start_ticks': token, 'started': now(),
              'session': os.getsid(0), 'ppid': os.getppid(),
              'stdin': os.readlink('/proc/self/fd/0')})
        mirror()
        if not (OUT/'DESIGN.json').is_file():
            assert run([sys.executable, '-B', str(OUT/'code/prepare.py')], 'prepare',
                       1800) == 0
        assert sha256(OUT/'wm2/wm_s.pt') == WM2_SHA256
        best = OUT/'combined/wm_s.pt'
        final = OUT/'combined/wm_s_final.pt'
        initial = OUT/'combined/wm_s_initial.pt'
        resolved = OUT/'combined/config.resolved.json'
        for name in (best, final, initial, resolved):
            assert name.is_file(), f'{name} missing (training failed?)'

        write('STATE.json', {'phase': 'scoring_e3', 't': now()})
        ref_e3 = e3_gate(OUT/'wm2/wm_s.pt', LOOP_CONFIG, 'validation',
                         OUT/'e3val_wm2.json', 'e3val_wm2')
        cand_e3 = {'combined': e3_gate(best, resolved, 'validation',
                                       OUT/'e3val_combined.json', 'e3val_combined'),
                   'combined_final': e3_gate(final, resolved, 'validation',
                                             OUT/'e3val_combined_final.json',
                                             'e3val_combined_final')}
        for name, ckpt in (('combined', best), ('combined_final', final)):
            cand_e3[name]['calibration'] = e3_gate(ckpt, resolved, 'calibration',
                                                   OUT/f'e3cal_{name}.json',
                                                   f'e3cal_{name}')
        mirror()

        write('STATE.json', {'phase': 'scoring_gate', 't': now()})
        ref_gate = gate(OUT/'wm2/wm_s.pt', LOOP_CONFIG, OUT/'wm2_gate.json', 'wm2_gate')
        mirror()
        cand = {'combined': gate(best, resolved, OUT/'combined_gate.json', 'combined_gate'),
                'combined_final': gate(final, resolved, OUT/'combined_final_gate.json',
                                       'combined_final_gate')}
        mirror()

        write('STATE.json', {'phase': 'acceptance', 't': now()})
        verdicts = {}
        for name, ckpt in (('combined', best), ('combined_final', final)):
            verdicts[name] = {
                'pairs300_amended': amended(ref_gate, cand[name],
                                            OUT/f'AMENDED_pairs300_{name}.json'),
                'e3_validation_amended': amended(ref_e3, cand_e3[name],
                                                 OUT/f'AMENDED_e3val_{name}.json'),
                'accept': accept(ckpt, resolved, f'{name}_accept')}
            mirror()

        results = {'id': ID, 'wm2_sha256': WM2_SHA256,
                   'best_checkpoint': str(best), 'best_checkpoint_sha256': sha256(best),
                   'final_checkpoint': str(final), 'final_checkpoint_sha256': sha256(final),
                   'training': json.loads((OUT/'combined/TRAINING.json').read_text()),
                   'pairs300': {name: {'wm2': summarize_gate(ref_gate),
                                       'candidate': summarize_gate(cand[name]),
                                       'amended': verdicts[name]['pairs300_amended']}
                                for name in ('combined', 'combined_final')},
                   'e3_validation': {name: {'wm2': summarize_gate(ref_e3),
                                            'candidate': summarize_gate(cand_e3[name]),
                                            'amended': verdicts[name]['e3_validation_amended'],
                                            'arm_analysis': cand_e3[name].get('arm_analysis'),
                                            'wm2_arm_analysis': ref_e3.get('arm_analysis')}
                                     for name in ('combined', 'combined_final')},
                   'accept': {name: verdicts[name]['accept']
                              for name in ('combined', 'combined_final')}}
        write('RESULTS.json', results)
        status = 'complete'
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
