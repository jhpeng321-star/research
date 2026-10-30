"""Exclusive observer successor; cooperative predecessor STOP, no trainer dispatch."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.candB import pilot_continue_observe, pilot_observe
from tools.candB.pilot_watch_2240 import append


DIRECTORY = Path(__file__).resolve().parent
ROOT = '/root/autodl-tmp/c4/candB_pilot_final_run_EXPLORATORY_20261030'
LEDGER = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def retire_predecessor():
    previous = json.loads((DIRECTORY / 'pilot_watch_2240_OWNERSHIP.json').read_text())
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, previous['pid'])
    if not handle:
        raise RuntimeError('predecessor unavailable; independently verify before changing observer lease')
    try:
        command = ('Get-CimInstance Win32_Process -Filter "ProcessId='+str(previous['pid'])+'" | '
                   'Select-Object ProcessId,CommandLine,@{Name="utc";Expression={$_.CreationDate.ToUniversalTime().ToString("o")}} | ConvertTo-Json -Compress')
        result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
            text=True, capture_output=True, check=True, timeout=20)
        actual = json.loads(result.stdout)
        if ('-m tools.candB.pilot_watch_2240 ' not in actual['CommandLine']
                or ROOT not in actual['CommandLine']
                or abs((datetime.fromisoformat(actual['utc'])-datetime.fromisoformat(previous['utc'])).total_seconds()) > 3):
            raise ValueError('predecessor exact command/creation mismatch')
        (DIRECTORY / 'pilot_watch_2240_STOP').touch(exist_ok=True)
        if kernel.WaitForSingleObject(handle, 660000) != 0:
            raise TimeoutError('predecessor did not cooperatively exit; no duplicate observer')
        return actual
    finally:
        kernel.CloseHandle(handle)


def commit_observation(paths, phase):
    worktree = DIRECTORY.parents[1]
    for command in (['git', 'status', '--short'], ['git', 'diff', '--', 'tools/candB/PILOT.md'],
                    ['git', 'log', '--oneline', '-10']):
        subprocess.run(command, cwd=worktree, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=worktree).returncode:
        raise RuntimeError('staged work present; preserve it and leave observer evidence uncommitted')
    subprocess.run(['git', 'add', '--', *[str(p.relative_to(worktree)) for p in paths]], cwd=worktree, check=True)
    subprocess.run(['git', 'commit', '-m', f'S-PILOTB: observe exploratory phase update {phase}'], cwd=worktree, check=True)


def watch():
    predecessor = retire_predecessor()
    lease = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
                 root=ROOT, poll_s=600, predecessor=predecessor, trainer_dispatch=False)
    with (DIRECTORY / 'pilot_watch_2305_OWNERSHIP.json').open('x') as stream:
        json.dump(lease, stream, indent=1)
    recorded = set()
    proof_path = DIRECTORY / 'pilot_samephase_PROOF.json'
    ack = DIRECTORY / 'pilot_update3_ACK.json'
    while not (DIRECTORY / 'pilot_watch_2305_STOP').exists():
        try:
            doc = pilot_continue_observe.observe('gpuA', ROOT)
            pilot_observe.record(doc, DIRECTORY, LEDGER)
            for mode, stage in doc['stages'].items():
                for row in stage['rows']:
                    phase = row['phase_update']
                    if (mode, phase) in recorded:
                        continue
                    snapshot = DIRECTORY / f'pilot_watch2305_{mode}_phase{phase}.json'
                    with snapshot.open('x') as stream:
                        json.dump(dict(label='EXPLORATORY', observed_utc=datetime.now(timezone.utc).isoformat(),
                            row=row, trainer=doc.get('continue_trainer_OWNERSHIP.json'),
                            continuation_proof=stage.get('CONTINUATION_PROOF.json')), stream, indent=1)
                    score = row.get('independent_J', {})
                    append(DIRECTORY / 'PILOT.md', f'\n### {mode} phase{phase} — EXPLORATORY\n'
                        f'- Raw dJ dense/300s: {score.get("raw_dJ_vs_CTRL0_dense")}/{score.get("raw_dJ_vs_CTRL0_long")}; '
                        f'ES accepted={row.get("eslong_accepted_pairs")}; gradients actor/dense/long='
                        f'{row.get("actor_grad_norm")}/{row.get("grad_norm_dense")}/{row.get("grad_norm_long")}\n'
                        f'- Optimizer/incl-rescore seconds: {row.get("optimizer_update_s")}/{row.get("update_including_rescore_s")}; '
                        f'peak allocated/reserved bytes={row.get("cuda_peak_allocated_bytes")}/{row.get("cuda_peak_reserved_bytes")}\n'
                        f'- Executed residual: {json.dumps(score.get("executed_residual"))}; minima={json.dumps(score.get("executed_residual_minima"))}\n'
                        f'- Group credit: {json.dumps(row.get("group_credit_terms"))}; confident gains={json.dumps(row.get("confident_group_gain_rows"))}\n'
                        f'- Recipe/worker peaks: {json.dumps(row.get("pilot_recipe"))}/{json.dumps(row.get("pilot_worker_memory"))}; '
                        f'R4={json.dumps(row.get("R4"))}. Full observed row: `{snapshot.name}`.\n')
                    paths = [DIRECTORY / 'PILOT.md', snapshot]
                    proof = stage.get('CONTINUATION_PROOF.json')
                    if proof and not proof_path.exists():
                        with proof_path.open('x') as stream:
                            json.dump(proof, stream, indent=1)
                        paths.append(proof_path)
                    if mode == 'pilot' and phase >= 3 and not ack.exists():
                        with ack.open('x') as stream:
                            json.dump(dict(label='EXPLORATORY', row=row, proof=proof,
                                observed_utc=datetime.now(timezone.utc).isoformat()), stream, indent=1)
                        paths.append(ack)
                    recorded.add((mode, phase))
                    commit_observation(paths, phase)
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                phases={m:[r['phase_update'] for r in s['rows']] for m,s in doc['stages'].items()},
                trainer=doc.get('continue_trainer_OWNERSHIP.json'), exit=doc.get('continue_EXIT.json'))), flush=True)
            if 'continue_EXIT.json' in doc:
                return
        except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as error:
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), error=str(error))), flush=True)
        time.sleep(600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spawn', action='store_true')
    args = parser.parse_args()
    if args.spawn:
        with (DIRECTORY / 'pilot_watch_2305_DISPATCH.json').open('x') as lease:
            child = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.candB.pilot_watch_2305'],
                cwd=DIRECTORY.parents[1], stdin=subprocess.DEVNULL,
                stdout=(DIRECTORY / 'pilot_watch_2305.log').open('x'), stderr=subprocess.STDOUT,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
            json.dump(dict(pid=child.pid, utc=datetime.now(timezone.utc).isoformat(),
                           root=ROOT, dispatch='observer only; wait predecessor exit'), lease, indent=1)
        print(json.dumps(dict(pid=child.pid, root=ROOT)))
    else:
        watch()


if __name__ == '__main__':
    main()
