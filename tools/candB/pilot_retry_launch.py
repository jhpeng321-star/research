"""Launch the owner03:00 zero-progress S4 retry from the bound lean environment."""
import argparse
import os
from pathlib import Path

from tools.candB.pilot_continue import replace_once


def launch_source(source):
    # All source changes are exact operational launcher substitutions, in memory.
    source = replace_once(source, 'cd "${RUNTIME:?}"', 'test "$PWD" = "${RUNTIME:?}"')
    source = replace_once(source, 'PILOT_EXPLORATORY_LEAN_AUTHORITY.json', 'PILOT_RETRY_AUTHORITY.json')
    source = replace_once(source, '-m tools.candB.pilot_lean_authority', '-m tools.candB.pilot_retry_authority')
    source = replace_once(source,
        'export WMS_SPEED_FORWARD_WORKERS=2 WMS_SPEED_FORWARD_MEMORY_FRACTION=.28',
        'export WMS_SPEED_ES_STACK_SIDES=4\nexport WMS_SPEED_FORWARD_WORKERS=1 WMS_SPEED_FORWARD_MEMORY_FRACTION=.75')
    source = replace_once(source,
        'export WMS_SPEED_FUTURE_WORKERS=1 WMS_SPEED_FUTURE_VJP_SLOTS=1 WMS_SPEED_FUTURE_MEMORY_FRACTION=.24',
        'export WMS_SPEED_FUTURE_WORKERS=2 WMS_SPEED_FUTURE_VJP_SLOTS=2 WMS_SPEED_FUTURE_MEMORY_FRACTION=.24')
    source = replace_once(source, '-m tools.candB.pilot_lean --exploratory-owner-20261030',
        '-m tools.candB.pilot_retry --exploratory-owner-20261030')
    source = replace_once(source, '>= 38000', '>= 46000')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    args = parser.parse_args()
    source = launch_source(Path('tools/candB/pilot_lean_launch.sh').read_text())
    # exec bash's final Python directly under this owned process PID.
    os.execvpe('bash', ['bash', '-c', source, 'pilot_retry_launch', args.output,
        '--exploratory-owner-20261030'], dict(os.environ))


if __name__ == '__main__':
    main()
