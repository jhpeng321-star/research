"""Launch only a bound S4 ordinary continuation, never a new phase."""
import argparse
import os
from pathlib import Path

from tools.candB.pilot_continue import replace_once


def launch_source(source):
    for old, new in (
        ('cd "${RUNTIME:?}"', 'test "$PWD" = "${RUNTIME:?}"'),
        ('PILOT_EXPLORATORY_LEAN_AUTHORITY.json', 'PILOT_WMFAST4_AUTHORITY.json'),
        ('-m tools.candB.pilot_lean_authority', '-m tools.candB.pilot_wmfast4_authority'),
        ('export WMS_SPEED_FORWARD_WORKERS=2 WMS_SPEED_FORWARD_MEMORY_FRACTION=.28',
         'export WMS_SPEED_ES_STACK_SIDES=4\nexport WMS_SPEED_FORWARD_WORKERS=1 WMS_SPEED_FORWARD_MEMORY_FRACTION=.75'),
        ('export WMS_SPEED_FUTURE_WORKERS=1 WMS_SPEED_FUTURE_VJP_SLOTS=1',
         'export WMS_SPEED_FUTURE_WORKERS=2 WMS_SPEED_FUTURE_VJP_SLOTS=2'),
        ('-m tools.candB.pilot_lean --exploratory-owner-20261030',
         '-m tools.candB.pilot_wmfast4 --parent "$PILOT_PARENT" --exploratory-owner-20261030'),
        ('>= 38000', '>= 46000'),
    ):
        source = replace_once(source, old, new)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('continue',))
    parser.add_argument('--parent', required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    source = launch_source(Path('tools/candB/pilot_lean_launch.sh').read_text())
    env = dict(os.environ, PILOT_PARENT=str(Path(args.parent).resolve()))
    os.execvpe('bash', ['bash', '-c', source, 'pilot_wmfast4_launch', args.output,
        '--exploratory-owner-20261030'], env)


if __name__ == '__main__':
    main()
