"""Saved OE41 results reconcile without loading the large raw trace fixture."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / 'output/formation_harm_v1'


def test_v3_category_costs_reconcile_to_source_interval_analysis():
    result=json.loads((ROOT/'interactions_phase_v3.json').read_text())
    source=json.loads((ROOT/'RESULTS.json').read_text())
    refs={c['seed']:c for c in source['cases'] if c['action']=='h20'}
    assert len(result['cases'])==32
    for case in result['cases']:
        for metric in ('braking_vehicle_seconds','acceleration_energy','ttc_below_3_vehicle_seconds'):
            actual=sum(d['metric_deltas'][metric] for d in case['assembly'].values())
            expected=refs[case['seed']]['phases']['assembly']['delta'][metric]
            assert abs(actual-expected)<1e-6


def test_missing_coordinate_identity_is_reported_not_assumed():
    result=json.loads((ROOT/'interactions_phase_v3.json').read_text())
    assert all(s['position_pairs_available']==0 for s in result['summary'].values())
    assert result['summary']['cohort']['endpoint_speed_pairs']==64
