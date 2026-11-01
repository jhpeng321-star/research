"""Score a registered scope with the original gate and strict CI adapter."""
import argparse
from pathlib import Path
import sys

from tools.rb2int.gate_bootstrap import GateBootstrap
from tools.rb2int.predictions import read_model
from tools.rbeval import score
from tools.rbeval.core import digest, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--real', required=True)
    parser.add_argument('--scope', choices=('primary54', 'all58', 'reserved8'), default='primary54')
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    runs = load(args.runs)
    registration = load(runs['registration_path'])
    if digest(runs['registration_path']) != runs['registration_sha256']:
        raise ValueError('registration binding differs')
    worlds = registration[dict(primary54='replicated_primary_worlds',
        all58='final_comparison_worlds', reserved8='reserved8')[args.scope]]

    def reader(inputs, real, partial):
        return read_model(args.runs, args.model, real, worlds, partial=partial)

    sys.argv = [sys.argv[0], '--inputs', args.runs, '--out', args.out]
    if args.allow_partial:
        sys.argv.append('--allow-partial')
    score.main(prediction_reader=reader, bootstrap_factory=GateBootstrap,
               real_receipt=Path(args.real), split_classifier=lambda s: args.scope)


if __name__ == '__main__':
    main()
