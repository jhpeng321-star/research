"""Package a passing physical motion fit without changing its raw evidence bytes."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.rb2break.run import DEV
from tools.rbeval.core import digest, load, save_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = args.input/'entrant_population.json'
    artifact = load(source)
    assert artifact['schema'] == 'wm_phys.entrant_population.v1'
    assert artifact['motion_validation_improved'] and artifact['fit_worlds'] == list(DEV)
    assert artifact['conditioning_end_s'] == 300
    support, weights = np.array(artifact['support']), np.array(artifact['weights'])
    assert support.shape == (len(weights), 5) and np.isfinite(support).all() and (support > 0).all()
    assert np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0
    root = Path(__file__).resolve().parents[2]
    assert artifact['source_population_sha256'] == digest(root/'artifacts/rbfit/shrinkage_v1/idm_population.json')
    assert args.output.parent.is_dir() and not args.output.exists()
    args.output.mkdir()
    (args.output/'entrant_population.json').write_bytes(source.read_bytes())
    mean = weights@support/weights.sum()
    sd = np.sqrt(weights@((support-mean)**2)/weights.sum())
    receipt = dict(input=str(source), sha256=digest(source), output=str(args.output),
        parameter_order=artifact['parameter_order'], mean=mean.tolist(), sd=sd.tolist(),
        motion_metrics=artifact['metrics'], source_sha256=digest(__file__),
        traffic_qualified=False)
    save_new(args.output/'PACKAGE.json', receipt)
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
