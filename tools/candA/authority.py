"""S-INTEGRATE candidate-A launch authority: bind every hash the morning GO needs.

One template, fail-closed on ANY mismatch. ``build`` records:

* source commit + runtime manifest (execution-law / resume / monitor files),
* the resolved candidate-A config chain hashes (candA -> c4prep_on -> ...),
* the C1 actor (64f70af4...) resumed with --new-phase,
* the chosen WM checkpoint sha256 + WM config sha256 + learned-mix flag state,
* CCT-1 (informational) and CCT-2 (gain-only long credit) authority files,
* the always-commanded execution law + shared sumo_idm mirror source hashes,
* the c4/execparity3 SUMO lane/link leader rule (BOUND since S-INTEGRATE2:
  source hashes of lane_chain/sumo_leader/neighbours/contract, the
  candidate-A config's execution.leader_rule = true binding, and the merged
  commit), and
* the mode-2 execution parity receipt (wms.control.execution.verify_parity).

``verify`` recomputes every binding against the tree and refuses on mismatch.
No GPU, no SUMO, no dispatch: artifact verification only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'src')]

from wms.control.gated300 import CCT2_SCHEMA, H_G_HORIZONS, MIN_GAIN, SIGMA  # noqa: E402

SCHEMA = 'candA_launch_authority/1'
C1_ACTOR_SHA256 = '64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'
CONFIG_CHAIN = ['configs/c4_policy14_candA.toml', 'configs/c4_policy14_c4prep_on.toml',
                'configs/c4_policy14_c4prep_off.toml', 'configs/c4_policy14_c3longint.toml']
# Execution law, shared mirror, sole writer, deploy fixes, resume, monitors.
MANIFEST_FILES = [
    'src/wms/control/sumo_idm.py', 'src/wms/control/alwayscmd.py',
    'src/wms/control/continuous.py', 'src/wms/control/execution.py',
    'src/wms/control/nativeinit.py', 'src/wms/control/bridge.py',
    'src/wms/control/deploy.py', 'src/wms/control/policy.py',
    'src/wms/control/dreamer.py', 'src/wms/control/policy14_cli.py',
    'src/wms/control/phase_resume.py', 'src/wms/control/gated300.py',
    'src/wms/control/exploit.py', 'src/wms/control/auditfix_authority.py',
    'src/wms/control/parallel_futures.py', 'src/wms/core/speed_flags.py',
    'src/wms/control/zerostart.py',
    'traffic_shepherd/ownership.py', 'traffic_shepherd/c4/branching.py',
    'traffic_shepherd/c4/wms_glue.py', 'traffic_shepherd/c4/env.py',
    'traffic_shepherd/c4/recorder.py',
]
# S-EXECPARITY3 leader rule (imagination mirror; deployment path unchanged).
LEADER_RULE_FILES = ['src/wms/physics/lane_chain.py', 'src/wms/control/sumo_leader.py',
                     'src/wms/physics/neighbours.py', 'src/wms/control/contract.py']
LEADER_RULE_SOURCE_COMMIT = 'a6a1894d'  # c4/execparity3 tip merged into c4/candA-int
MANIFEST_FILES += LEADER_RULE_FILES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit(root: Path) -> str:
    source = root / 'SOURCE.json'
    if source.is_file():
        return json.loads(source.read_text(encoding='utf-8'))['commit']
    out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(root), text=True)
    return out.strip()


def clean_tree(root: Path) -> bool:
    if not (root / '.git').exists():
        return True  # staged immutable runtime: byte manifest is the binding
    out = subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(root), text=True)
    return not out.strip()


def resolve_chain(root: Path, entry: str) -> list[str]:
    """Config chain in extends order, fully hash-bound down to the base preset.

    ``extends`` values resolve exactly like ``train_cli._read_config`` (relative
    to the extending file's directory). Refuses missing files or bases outside
    ``configs/``; the head of the chain must be the candidate-A entry files.
    """
    chain, name = [], entry
    while name:
        path = root / name
        if not path.is_file():
            raise ValueError(f'config chain file missing: {name}')
        chain.append(name)
        base = None
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip().startswith('extends'):
                base = line.split('=', 1)[1].strip().strip('"').strip("'")
                break
        if base:
            parent = str(Path(name).parent)
            resolved = base if '/' in base or parent == '.' else f'{parent}/{base}'
            if not resolved.startswith('configs/') or (root / resolved).is_file() is False:
                raise ValueError(f'config extends outside the registered configs/: {resolved}')
            name = resolved
        else:
            name = ''
    for required in CONFIG_CHAIN:
        if required not in chain:
            raise ValueError(f'candidate-A chain lost its registered head: {required}')
    return chain


def verify_cct2(doc: dict, wm: str, *, learned_mix: bool) -> int:
    if (doc.get('schema') != CCT2_SCHEMA or doc.get('wm_sha256') != wm
            or doc.get('panels') != ['pairs300', 'validation']
            or doc.get('policy_authorized') is not True):
        raise ValueError('CCT-2 must be this WM\'s pooled pairs300+validation verdict authority')
    hg = doc.get('H_G')
    if type(hg) is not int or hg not in (0, *H_G_HORIZONS):
        raise ValueError('invalid CCT-2 H_G')
    largest = 0
    for horizon in H_G_HORIZONS:
        row = doc.get('horizons', {}).get(str(horizon), {})
        gain, agreement = (row.get(k) for k in ('n_gain', 'gain_sign_agreement'))
        passed = (row.get('sigma') == SIGMA[horizon] and row.get('threshold') == 2 * SIGMA[horizon]
                  and all(type(x) in (int, float) and math.isfinite(x) for x in (gain, agreement))
                  and gain >= MIN_GAIN and .8 <= agreement <= 1
                  and row.get('verdict') == 'PASS')
        if passed:
            largest = horizon
    if hg != largest:
        raise ValueError('CCT-2 H_G disagrees with the registered largest passing horizon')
    if learned_mix and doc.get('learned_mix_qualified') is not True:
        raise ValueError('learned-mix WM requires a mix-qualified CCT-2 authority '
                         '(ERRSRC mix qualified on WM2 only; re-qualify the mix under '
                         'the registered K1 contract before gain-only long credit)')
    return hg


def _config_long_credit_enabled(root: Path) -> bool:
    from wms.core.toml import load as _toml_load
    section = _toml_load(root / CONFIG_CHAIN[0]).get('long_credit', {})
    enabled = section.get('enabled')
    if type(enabled) is not bool:
        raise ValueError('candidate-A config must declare long_credit.enabled as a bool')
    return enabled


def build(args) -> dict:
    root = Path(args.root).resolve()
    commit = source_commit(root)
    if not args.allow_dirty and not clean_tree(root):
        raise ValueError('source tree dirty; commit or stage the runtime before binding')
    wm_path, wm_config = Path(args.wm_ckpt), Path(args.wm_config)
    wm = sha256(wm_path)
    actor = sha256(Path(args.actor)) if args.actor else C1_ACTOR_SHA256
    if actor != C1_ACTOR_SHA256:
        raise ValueError(f'C1 actor sha mismatch: {actor}')
    chain = resolve_chain(root, CONFIG_CHAIN[0])
    from wms.core.toml import load as _toml_load
    zero_start = bool(_toml_load(root / CONFIG_CHAIN[0]).get('policy14', {}).get('zero_residual_output'))
    if _toml_load(root / CONFIG_CHAIN[0]).get('execution', {}).get('leader_rule') is not True:
        raise ValueError('candidate-A config must bind execution.leader_rule = true: the '
                         'mode-2 parity receipt is only valid with C4_SUMO_LEADER_RULE=1 '
                         'in the imagination mirror (deployment path unchanged)')
    # Declared credit mode must match the bound config bytes (fail-closed both
    # ways): gain_only requires long_credit.enabled = true; the registered
    # dense-90s phase 1 (23:25-rule second-phase deferral) requires false.
    enabled_in_config = _config_long_credit_enabled(root)
    if args.dense_phase1 and enabled_in_config:
        raise ValueError('--dense-phase1 requires long_credit.enabled = false in the '
                         'candidate-A config (registered 23:25-rule second-phase deferral)')
    if not args.dense_phase1 and not enabled_in_config:
        raise ValueError('long_credit gain_only authority requires long_credit.enabled = '
                         'true; pass --dense-phase1 for the registered dense-90s phase 1')
    manifest = {name: sha256(root / name) for name in MANIFEST_FILES}
    doc = dict(schema=SCHEMA, source_commit=commit, runtime_root=str(root),
               config_chain={name: sha256(root / name) for name in chain},
               actor_c1_sha256=actor, actor_checkpoint=str(args.actor_checkpoint or ''),
               wm_sha256=wm, wm_checkpoint=str(wm_path), wm_config=str(wm_config),
               wm_config_sha256=sha256(wm_config),
               learned_mix=bool(args.learned_mix),
               learned_mix_module=args.mix_module or '',
               learned_mix_module_sha256=(sha256(root / args.mix_module)
                                          if args.learned_mix and args.mix_module else ''),
               new_phase=True, checkpoint_every=1,
               zero_start={'enabled': zero_start, 'module': 'src/wms/control/zerostart.py',
                           'config_flag': 'policy14.zero_residual_output'},
               monitors={'exploit_monitor': True, 'exploit_disable_direct_credit': True,
                         'exploit_ensemble_pessimism': False, 'exploit_stop_on_conflict': False,
                         'native_anchor': False},
               dense_horizon_s=90, long_credit={'mode': ('dense_90s_phase1' if args.dense_phase1
                                                         else 'gain_only'),
                                                 'cct2_file': str(Path(args.cct2).resolve()),
                                                 'cct2_sha256': sha256(Path(args.cct2))},
               execution_law={'mode': 2, 'backend_contract': 'always_commanded_ctrl0/1',
                              'shared_base': 'src/wms/control/sumo_idm.py',
                              'sole_writer': 'traffic_shepherd/ownership.py',
                              'accel_max_mps2': 1.2, 'speed_mode': 27},
                leader_rule={'source': 'c4/execparity3',
                             'status': args.leader_rule_status or 'OPEN_SLOT',
                             'commit': args.leader_rule_commit or LEADER_RULE_SOURCE_COMMIT,
                             'env': 'C4_SUMO_LEADER_RULE=1 (imagination mirror only; '
                                    'deployment executor path unchanged)',
                             'config_flag': 'configs/c4_policy14_candA.toml '
                                            '[execution] leader_rule = true',
                             'files': {name: manifest[name] for name in LEADER_RULE_FILES}},
               runtime_manifest=manifest)
    if args.learned_mix and not args.mix_module:
        raise ValueError('--learned-mix requires --mix-module (errsrc inference path) '
                         'to be hash-bound')
    if args.cct1:
        doc['cct1'] = {'file': str(Path(args.cct1).resolve()), 'sha256': sha256(Path(args.cct1))}
    hg = verify_cct2(json.loads(Path(args.cct2).read_text(encoding='utf-8')), wm,
                     learned_mix=bool(args.learned_mix))
    doc['long_credit']['H_G'] = hg
    if args.dense_phase1:
        doc['long_credit']['phase2'] = {
            'mode': 'gain_only', 'H_long': hg,
            'rule': 'registered 23:25 MAIN pre-registration: A starts at 90 s; the long '
                    'credit is added as a registered second phase once the speed work '
                    'makes it fit the 24 h projection',
            'projection_s_per_update': {'dense_90s': 350.756, 'H_long_180': 701.5,
                                        'H_long_300': 1165.8},
            'oom_receipt': 'gpuA candA_dry_f176d7ac9a28 attempt 3 (forward worker, '
                           'fraction .04 = 1.89 GiB allowed, 300 s futures)'}
    doc['parity_receipt'] = ''
    if args.parity_receipt:
        os.environ.setdefault('C4_EXECUTION_PARITY', '2')
        if not os.environ.get('C4_EXECUTION_DRIVER'):
            raise ValueError('mode-2 parity receipt validation requires C4_EXECUTION_DRIVER')
        from wms.control.execution import verify_parity
        verify_parity(args.parity_receipt)  # raises on missing/stale/failing receipt
        doc['parity_receipt'] = str(Path(args.parity_receipt).resolve())
        doc['parity_receipt_sha256'] = sha256(Path(args.parity_receipt))
        doc['leader_rule']['status'] = (args.leader_rule_status
                                        if args.leader_rule_status not in (None, 'OPEN_SLOT')
                                        else 'bound-with-receipt')
    if args.out:
        out = Path(args.out)
        if out.exists():
            raise ValueError('authority output must be a NEW file')
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n', encoding='utf-8')
        print(json.dumps({'authority': str(out), 'H_G': hg}, indent=1))
    else:
        print(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def verify(args) -> dict:
    root = Path(args.root).resolve()
    doc = json.loads(Path(args.authority).read_text(encoding='utf-8'))
    if doc.get('schema') != SCHEMA:
        raise ValueError('unknown authority schema')
    if source_commit(root) != doc['source_commit']:
        raise ValueError('source commit mismatch')
    for name, digest in doc['runtime_manifest'].items():
        if not (root / name).is_file() or sha256(root / name) != digest:
            raise ValueError(f'runtime manifest mismatch: {name}')
    leader = doc.get('leader_rule', {})
    if leader.get('status') == 'OPEN_SLOT' or not leader.get('files'):
        raise ValueError('leader-rule slot OPEN: the merged c4/execparity3 sources and a '
                         'passing mode-2 receipt must be bound (S-INTEGRATE2)')
    for name, digest in leader['files'].items():
        if not (root / name).is_file() or sha256(root / name) != digest:
            raise ValueError(f'leader rule source mismatch: {name}')
    if any(name not in doc['runtime_manifest'] for name in LEADER_RULE_FILES):
        raise ValueError('leader-rule files lost from the runtime manifest')
    from wms.core.toml import load as _toml_load
    if _toml_load(root / CONFIG_CHAIN[0]).get('execution', {}).get('leader_rule') is not True:
        raise ValueError('candidate-A config lost execution.leader_rule = true '
                         '(imagination mirror leader rule)')
    if 'zero_start' in doc:
        from wms.core.toml import load as _toml_load2
        flag = bool(_toml_load2(root / CONFIG_CHAIN[0]).get('policy14', {}).get('zero_residual_output'))
        if flag != bool(doc['zero_start']['enabled']):
            raise ValueError('zero-start flag diverges from the bound authority (policy14.zero_residual_output)')
    chain = resolve_chain(root, CONFIG_CHAIN[0])
    for name, digest in doc['config_chain'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'config chain mismatch: {name}')
    if set(chain) != set(doc['config_chain']):
        raise ValueError('config chain shape mismatch')
    # Declared credit mode stays consistent with the bound config bytes.
    mode = doc.get('long_credit', {}).get('mode', 'gain_only')
    enabled_in_config = _config_long_credit_enabled(root)
    if mode == 'dense_90s_phase1' and enabled_in_config:
        raise ValueError('authority declares dense-90s phase 1 but the config enables '
                         'long credit; rebuild the authority')
    if mode == 'gain_only' and not enabled_in_config:
        raise ValueError('authority declares gain_only long credit but the config runs '
                         'dense; rebuild with --dense-phase1 (registered second-phase '
                         'deferral) or re-enable long credit')
    if mode not in ('dense_90s_phase1', 'gain_only'):
        raise ValueError(f'unknown long-credit mode in authority: {mode}')
    if sha256(Path(doc['wm_checkpoint'])) != doc['wm_sha256']:
        raise ValueError('WM checkpoint mismatch')
    if sha256(Path(doc['wm_config'])) != doc['wm_config_sha256']:
        raise ValueError('WM config mismatch')
    if doc['actor_c1_sha256'] != C1_ACTOR_SHA256:
        raise ValueError('C1 actor binding lost')
    cct2 = json.loads(Path(doc['long_credit']['cct2_file']).read_text(encoding='utf-8'))
    if sha256(Path(doc['long_credit']['cct2_file'])) != doc['long_credit']['cct2_sha256']:
        raise ValueError('CCT-2 file changed')
    hg = verify_cct2(cct2, doc['wm_sha256'], learned_mix=doc['learned_mix'])
    if hg != doc['long_credit']['H_G']:
        raise ValueError('CCT-2 H_G changed')
    if doc['learned_mix'] and not doc['learned_mix_module_sha256']:
        raise ValueError('learned-mix requires a bound inference module hash')
    if doc['learned_mix'] and doc['learned_mix_module']:
        if sha256(root / doc['learned_mix_module']) != doc['learned_mix_module_sha256']:
            raise ValueError('learned-mix inference module mismatch')
    if not doc['parity_receipt']:
        raise ValueError('no mode-2 execution parity receipt bound; launch refuses '
                         '(ASTRA4 GO condition 1 unmet until a passing receipt exists)')
    receipt = Path(doc['parity_receipt'])
    if sha256(receipt) != doc.get('parity_receipt_sha256'):
        raise ValueError('parity receipt changed')
    os.environ.setdefault('C4_EXECUTION_PARITY', '2')
    from wms.control.execution import verify_parity
    verify_parity(str(receipt))
    doc['verified'] = True
    print(json.dumps({'verified': True, 'source_commit': doc['source_commit'],
                      'wm_sha256': doc['wm_sha256'], 'H_G': hg,
                      'leader_rule': doc['leader_rule']['status']}, indent=1))
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    build_p = sub.add_parser('build', parents=[common])
    for name in ('wm-ckpt', 'wm-config', 'cct2', 'out'):
        build_p.add_argument('--' + name, required=True)
    build_p.add_argument('--cct1', default=None)
    build_p.add_argument('--actor', default=None,
                         help='checkpoint file to hash against the registered C1 actor')
    build_p.add_argument('--actor-checkpoint', default=None)
    build_p.add_argument('--learned-mix', action='store_true')
    build_p.add_argument('--mix-module', default=None)
    build_p.add_argument('--dense-phase1', action='store_true',
                         help='declare the registered dense-90s phase 1 (23:25-rule '
                              'second-phase deferral); requires long_credit.enabled = false '
                              'in the candidate-A config, keeps the CCT-2 file bound as '
                              'the registered phase-2 authority')
    build_p.add_argument('--parity-receipt', default=None)
    build_p.add_argument('--leader-rule-status', default=None,
                         help='override the recorded leader-rule status '
                              '(default: bound-with-receipt once the mode-2 receipt passes)')
    build_p.add_argument('--leader-rule-commit', default=None)
    build_p.add_argument('--allow-dirty', action='store_true')
    verify_p = sub.add_parser('verify', parents=[common])
    verify_p.add_argument('--authority', required=True)
    args = parser.parse_args(argv)
    (build if args.cmd == 'build' else verify)(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
