"""Launch the owner05:25 peer-present S2 continuation from the bound lean environment."""
import argparse
import os
from pathlib import Path

from tools.candB.pilot_continue import replace_once


def launch_source(source):
    # All source changes are exact operational launcher substitutions, in memory.
    source = replace_once(source, 'cd "${RUNTIME:?}"', 'test "$PWD" = "${RUNTIME:?}"')
    source = replace_once(source, 'PILOT_EXPLORATORY_LEAN_AUTHORITY.json', 'PILOT_CONT0525_AUTHORITY.json')
    source = replace_once(source, '-m tools.candB.pilot_lean_authority', '-m tools.candB.pilot_cont0525_authority')
    source = replace_once(source,
        'export WMS_SPEED_FORWARD_WORKERS=2 WMS_SPEED_FORWARD_MEMORY_FRACTION=.28',
        'export WMS_SPEED_ES_STACK_SIDES=2\nexport WMS_SPEED_FORWARD_WORKERS=1 WMS_SPEED_FORWARD_MEMORY_FRACTION=.28')
    source = replace_once(source,
        'export WMS_SPEED_FUTURE_WORKERS=1 WMS_SPEED_FUTURE_VJP_SLOTS=1 WMS_SPEED_FUTURE_MEMORY_FRACTION=.24',
        'export WMS_SPEED_FUTURE_WORKERS=2 WMS_SPEED_FUTURE_VJP_SLOTS=2 WMS_SPEED_FUTURE_MEMORY_FRACTION=.16')
    source = replace_once(source, '-m tools.candB.pilot_lean --exploratory-owner-20261030',
        '-m tools.candB.pilot_cont0525 --parent "$PILOT_PARENT" --exploratory-owner-20261030')
    source = replace_once(source,
        'assert int(os.environ["PILOT_ADMISSION_FREE_MIB"]) >= 38000',
        'assert int(os.environ["PILOT_ADMISSION_FREE_MIB"]) >= 21000; '
        'assert int(os.environ["PILOT_ADMISSION_PEER_MIB"]) <= 23000')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    source = launch_source(Path('tools/candB/pilot_lean_launch.sh').read_text())
    # exec bash's final Python directly under this owned process PID.
    os.execvpe('bash', ['bash', '-c', source, 'pilot_cont0525_launch', args.output,
        '--exploratory-owner-20261030'], dict(os.environ,
        PILOT_PARENT=str(Path(args.parent).resolve())))


if __name__ == '__main__':
    main()
