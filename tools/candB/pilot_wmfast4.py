"""S4 execution-only specialization of the verified full same-phase adapter."""
import argparse
from unittest.mock import patch

from tools.candB import pilot_wmfast, pilot_wmfast_authority, pilot_wmfast4_authority


def run(path, output, parent, *, exploratory=False):
    with patch.object(pilot_wmfast_authority, 'ENV', pilot_wmfast4_authority.ENV), patch.object(
            pilot_wmfast_authority, 'recipe', pilot_wmfast4_authority.recipe):
        return pilot_wmfast.run(path, output, parent, exploratory=exploratory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(pilot_wmfast4_authority.base.FLAG, action='store_true', required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    raise SystemExit(run(args.authority, args.output, args.parent, exploratory=True))


if __name__ == '__main__':
    main()
