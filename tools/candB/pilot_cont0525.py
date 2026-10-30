"""Owner05:25 same-phase S2 continuation from the RETRY0300 v2 checkpoint192.

The v2 run exited ONLY because the unchanged per-pool peer memory guard refused
the next S4 pool after S-WMFIXMERGE's verdict panel (peer82309, ~12.5GB) started
on gpuA; checkpoint policy_latest.pt (update192) and the frozen baseline were
already durable. This adapter accepts exactly that exit, binds those artifacts
by hash, and otherwise reuses the verified ordinary WMFAST continuation with a
peer-aware S2 recipe. No scientific source or gate is changed.
"""
import argparse
import inspect
from pathlib import Path
from unittest.mock import patch

from tools.candB import pilot_authority, pilot_wmfast
from tools.candB import pilot_cont0525_authority as authority
from tools.candB.pilot_continue import replace_once

PARENT = Path('/root/autodl-tmp/c4/candB_pilot_run_RETRY0300_v2_20261031/pilot_EXPLORATORY-LEAN_RETRY0300')
BASELINE = '744b30b809af60fb11acc54734cbc29bf92f6163793cd306b040ee26e430d680'
CHECKPOINT = 'aa683d4a5c9ea057fecb20da6db6c67761b3440715baf095fc06d9918922479a'
GUARD_EXIT = 'ValueError: lean aggregate memory bound including peers exceeds safe ceiling'
# Capture the physical WMFAST validator BEFORE run() patches the module attribute
# with this adapter; getsource must never observe the patched-in function itself.
_WMFAST_VALIDATE_PARENT = pilot_wmfast.validate_parent


def validate_parent(parent):
    source = inspect.getsource(_WMFAST_VALIDATE_PARENT)
    source = replace_once(source,
        "prior.get('error') != 'RuntimeError: own-run STOP requested; checkpoint saved'",
        'prior.get("error") != GUARD_EXIT')
    source = replace_once(source,
        "raise ValueError('WMFAST requires clean owned STOP and durable original baseline')",
        "raise ValueError('CONT0525 requires the exact peer-induced guard exit '\n"
        "            'and durable original baseline')")
    source = replace_once(source, "rows = prior['updates']\n    origin =",
        "if parent != PARENT:\n"
        "        raise ValueError('CONT0525 is restricted to the RETRY0300 v2 phase')\n"
        "    for name, digest in (('frozen_scoring_baseline.json', BASELINE),\n"
        "            ('policy_latest.pt', CHECKPOINT)):\n"
        "        if pilot_authority.sha256(parent/name) != digest:\n"
        "            raise ValueError('CONT0525 bound artifact drift: '+name)\n"
        "    rows = prior['updates']\n    origin =")
    namespace = dict(pilot_wmfast.__dict__, PARENT=PARENT, BASELINE=BASELINE,
        CHECKPOINT=CHECKPOINT, GUARD_EXIT=GUARD_EXIT, authority=authority,
        pilot_authority=pilot_authority)
    del namespace['validate_parent']
    exec(compile(source, __file__, 'exec'), namespace)
    return namespace['validate_parent'](parent)


def run(path, output, parent, *, exploratory=False):
    from tools.candB import pilot_wmfast_authority
    parent = Path(parent).resolve()
    if parent != PARENT:
        raise ValueError('CONT0525 is restricted to the RETRY0300 v2 phase')
    # No vjp_guard: it substitutes the switch-less raw bridge factory for the
    # CLI's config-bound one and the raw factory refuses the candidate-B
    # native-residual + lane_rules call (bridge.py 'native residual needs native
    # command...'). Production evidence (RETRY0300 v2 update192, CPU bitwise
    # contracts) shows the c73498cd passthrough markers alone keep telemetry
    # alive across replay release in exactly this adapter chain.
    with patch.object(pilot_wmfast, 'validate_parent', validate_parent), patch.object(
            pilot_wmfast_authority, 'ENV', authority.ENV), patch.object(
            pilot_wmfast_authority, 'recipe', authority.recipe):
        return pilot_wmfast.run(path, output, parent, exploratory=exploratory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(pilot_authority.FLAG, action='store_true', required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    raise SystemExit(run(args.authority, args.output, args.parent, exploratory=True))


if __name__ == '__main__':
    main()
