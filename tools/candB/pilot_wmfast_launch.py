"""Use the bound existing lean launch environment for repair or full continuation."""
import argparse
import os
from pathlib import Path

from tools.candB.pilot_continue import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('repair', 'continue'))
    parser.add_argument('--parent', required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    source = Path('tools/candB/pilot_lean_launch.sh').read_text()
    # All source changes are exact operational launcher substitutions, in memory.
    source = replace_once(source, 'cd "${RUNTIME:?}"', 'test "$PWD" = "${RUNTIME:?}"')
    if args.mode == 'repair':
        source = replace_once(source, '-m tools.candB.pilot_lean --exploratory-owner-20261030',
            '-m tools.candB.pilot_lean_repaired --failed-parent "$PILOT_PARENT" --exploratory-owner-20261030')
        source = replace_once(source, 'export WMS_SPEED_PILOT_FORWARD_POOL=1',
            'export WMS_SPEED_ES_STACK_SIDES=0\nexport WMS_SPEED_PILOT_FORWARD_POOL=1')
    else:
        source = replace_once(source, 'PILOT_EXPLORATORY_LEAN_AUTHORITY.json', 'PILOT_WMFAST_AUTHORITY.json')
        source = replace_once(source, '-m tools.candB.pilot_lean_authority', '-m tools.candB.pilot_wmfast_authority')
        source = replace_once(source, 'export WMS_SPEED_FORWARD_WORKERS=2 WMS_SPEED_FORWARD_MEMORY_FRACTION=.28',
            'export WMS_SPEED_ES_STACK_SIDES=2\nexport WMS_SPEED_FORWARD_WORKERS=1 WMS_SPEED_FORWARD_MEMORY_FRACTION=.50')
        source = replace_once(source, '-m tools.candB.pilot_lean --exploratory-owner-20261030',
            '-m tools.candB.pilot_wmfast --parent "$PILOT_PARENT" --exploratory-owner-20261030')
    env = dict(os.environ, PILOT_PARENT=str(Path(args.parent).resolve()))
    # exec bash's final Python directly under this owned process PID.
    os.execvpe('bash', ['bash', '-c', source, 'pilot_wmfast_launch', args.output,
        '--exploratory-owner-20261030'], env)


if __name__ == '__main__':
    main()
