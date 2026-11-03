"""Prepare a distinct, immutable Linux qualification after the reviewed merge.

No training, exact-gate execution, reservation, GPU release or launch is exposed.
The current qualified legacy source and all of its owners remain untouched.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

from . import ops, v2v_qualification_ops as qualification

BASE_COMMIT = 'cf3efc9e'
BASE_SOURCE = '/home/ubuntu/coopmwm20261103_v2vcpu8'
BASELINE_NAMES = ('accel_learner', 'exact_grade', 'bank_imitation', 'v2v_obs')
EXTRA_FILES = tuple('tools/coopmwm/'+name+'.py' for name in (
    'v2v_intent', 'test_v2v_intent_cpu', 'coopfix_exact_gate',
    'test_coopfix_exact_gate_cpu', 'solo_seed2_links', 'test_solo_seed2_links_cpu',
    'test_bank_boot_cpu', 'test_imitfix2_cpu', 'v2v_launch3', 'test_v2v_launch3_cpu'))
EXTRA_TESTS = ('test_v2v_intent_cpu', 'test_coopfix_exact_gate_cpu',
    'test_solo_seed2_links_cpu', 'test_bank_boot_cpu', 'test_imitfix2_cpu',
    'test_v2v_launch3_cpu')


def git(*args):
    result = subprocess.run(['git', *args], cwd=ops.LOCAL, capture_output=True, check=True)
    return result.stdout


def release_options(commit):
    """Only a complete committed integration can be staged; no dirty source overlay."""
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('full final integration commit required')
    if git('rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('current HEAD differs from the intended qualification commit')
    required = tuple(dict.fromkeys(qualification.FILES + EXTRA_FILES))
    git('diff', '--exit-code', 'HEAD', '--', *required)
    # Verify every input is tracked at the pinned HEAD. The source reader itself
    # uses the dedicated whitelist and never traverses data or outcome folders.
    for name in required:
        git('cat-file', '-e', commit+':'+name)
    frozen = {}
    for name in BASELINE_NAMES:
        frozen['intent_baseline/'+name+'.py'] = git(
            'show', BASE_COMMIT+':tools/coopmwm/'+name+'.py')
    frozen['intent_baseline/PROVENANCE.json'] = json.dumps(dict(
        commit=git('rev-parse', BASE_COMMIT).decode().strip(),
        rule='exact pre-intent git blobs; qualification source manifest pins each fixture'),
        sort_keys=True).encode()
    return dict(additional_files=EXTRA_FILES, additional_tests=EXTRA_TESTS,
        frozen_files=frozen, source_commit=commit,
        integration='reviewed intent + boot_v2 + full-pass + exact gate + launch3',
        extra_env={'COOPMWM_INTENT_BASELINE_DIR': qualification.SOURCE+'/intent_baseline'})


def configure(generation):
    if not re.fullmatch('I[1-9][0-9]*', generation):
        raise ValueError('distinct intent generation I1, I2, ... required')
    qualification.BASE = BASE_SOURCE
    qualification.GENERATION = 'INTENT_'+generation
    qualification.SOURCE = '/home/ubuntu/coopmwm20261103_intentcpu_'+generation.lower()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'qualify', 'status'))
    parser.add_argument('--generation', required=True)
    parser.add_argument('--commit')
    args = parser.parse_args(argv)
    configure(args.generation)
    if args.action == 'status':
        qualification.status()
        return
    options = release_options(args.commit or '')
    if args.action == 'prepare':
        print(json.dumps(dict(commit=args.commit, source=qualification.SOURCE,
            files=list(qualification.FILES)+list(EXTRA_FILES),
            tests=list(EXTRA_TESTS), frozen_baseline=list(options['frozen_files']),
            new_sim_s=0, dispatched=False)))
    else:
        qualification.qualify(**options)


if __name__ == '__main__':
    main()
