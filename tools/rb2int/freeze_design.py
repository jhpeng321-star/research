"""Bind the prospective evaluation design and outcome-blind metadata selection."""
import json
from datetime import datetime
from pathlib import Path

from tools.rb2int.audit import ROOT, load, sha


def main():
    base = ROOT/'docs/rb2int'
    metadata = base/'PANEL_METADATA.json'
    design = base/'EVALUATION_DESIGN.md'
    panel = load(metadata)
    assert panel['selection_used_outcomes'] is False
    assert len(panel['final_comparison_worlds']) == 58
    assert len(panel['replicated_primary_worlds']) == 54
    result = dict(schema='rb2int-evaluation-registration/1',
        registered_at=datetime.now().astimezone().isoformat(),
        design=str(design.relative_to(ROOT)), design_sha256=sha(design),
        metadata=str(metadata.relative_to(ROOT)), metadata_sha256=sha(metadata),
        final_comparison_worlds=panel['final_comparison_worlds'],
        replicated_primary_worlds=panel['replicated_primary_worlds'],
        descriptive_singleton_worlds=panel['descriptive_singleton_worlds'],
        reserved8=panel['reserved8'], cells=panel['cells'], arms=panel['arms'],
        K=8, horizons_s=[120, 300, 600], models=['WM2', 'FULL-v4', 'FULL-v5', 'FULL-v5-LP'],
        fitting_worlds=list(range(969400, 969432)), bootstrap=10000,
        bootstrap_seed=20261101, primary_strata=['demand', 'penetration'],
        opened_new_final_outcomes=False, full12_cell_qualification_available=False,
        final_inference_authorized=False,
        release_requires=['exact delivered worker exposure/lineage clearance',
                          'immutable candidate source/recipe/artifact freeze',
                          'development smoke and CPU suite receipts',
                          'resource and exclusive task ownership admission'])
    out = base/'EVALUATION_REGISTRATION.json'
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1, allow_nan=False)
    print(json.dumps(dict(receipt=str(out), sha256=sha(out),
                         design_sha256=sha(design), metadata_sha256=sha(metadata))))


if __name__ == '__main__':
    main()
