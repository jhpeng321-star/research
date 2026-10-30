"""Additional immutable owner23:50 lean recipe binding over exploratory authority."""
import argparse
import json
from pathlib import Path

from tools.candB import pilot_authority as base

LABEL = 'EXPLORATORY-LEAN'
ENV = dict(WMS_SPEED_REUSE_PANEL='1', WMS_SPEED_SKIP_GRAD_DIAGNOSTICS='1',
    WMS_SPEED_PROFILE='1', WMS_SPEED_RESCORE_EVERY='5',
    WMS_SPEED_PILOT_FORWARD_POOL='1', WMS_SPEED_PARALLEL_FORWARD='1',
    WMS_SPEED_FORWARD_WORKERS='2', WMS_SPEED_FORWARD_MEMORY_FRACTION='.28',
    WMS_SPEED_FUTURE_BUDGET_CAP='.90')


def recipe(doc):
    panel = json.loads(base.safe_path(doc['inputs']['gate_panel']).read_text())
    if len(panel['starts']) != 16:
        raise ValueError('lean requires registered sixteen-start original panel')
    return dict(label=LABEL, owner_amendment='2026-10-30 23:50 MAIN',
        original_panel_sha256=doc['input_sha256']['gate_panel'],
        panel_starts=panel['starts'][:4], panel_selection='first4 of original ordered16',
        es_future_indices=[0, 1, 2, 3], dense_futures=8, scoring_futures=8,
        es_selection='first4 cached ORIGINAL K8 CRN plans; no resampling',
        environment=ENV, fresh_zero_start=True, phase_update_limit=150, R4_phase_update=10,
        pool_memory_rule='coordinator+worker caps+contexts+sampled peers+2GiB <= min(90% device,45GiB)')


def verify(path, *, exploratory=False):
    doc = base.verify(path, exploratory=exploratory)
    if doc.get('lean_recipe') != recipe(doc):
        raise ValueError('lean recipe/panel/CRN identity mismatch')
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(base.FLAG, action='store_true', required=True)
    parser.add_argument('command', choices=('build', 'verify'))
    parser.add_argument('authority')
    parser.add_argument('--parent-authority')
    args = parser.parse_args()
    if args.command == 'build':
        parent = json.loads(base.safe_path(args.parent_authority).read_text())
        doc = base.build(parent['inputs'], args.authority, exploratory=True)
        doc['lean_recipe'] = recipe(doc)
        base.safe_path(args.authority).write_text(json.dumps(doc, indent=1)+'\n')
    doc = verify(args.authority, exploratory=True)
    print(json.dumps(dict(label=LABEL, source_commit=doc['source_commit'],
        authority_sha256=base.sha256(args.authority), recipe=doc['lean_recipe']), indent=1))


if __name__ == '__main__':
    main()
