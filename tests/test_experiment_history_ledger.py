"""Read-only integrity tests for the experiment-history ledger.

These tests never run SUMO, never train, and never open sealed outcome contents:
they parse protocols/experiment_history.json, run the ledger consistency validator,
and run the scoped source verifier (which reads only the registered allowed
sources). Copy-mutations below are applied to in-memory copies only.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'protocols' / 'experiment_history.json'


def _tool():
    spec = importlib.util.spec_from_file_location(
        'experiment_history_tool', ROOT / 'tools' / 'experiment_history.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load():
    with open(LEDGER, encoding='utf-8') as f:
        return json.load(f)


def test_ledger_loads_and_schema_is_versioned():
    d = _load()
    assert str(d['schema']).startswith('experiment_history/2')
    assert len(d['entries']) >= 102
    ids = [e['id'] for e in d['entries']]
    assert len(ids) == len(set(ids)), 'duplicate entry ids'


def test_coverage_metadata_and_evidence_vocabulary():
    d = _load()
    cov = d['coverage']
    assert cov['included_families'] and cov['last_reviewed'] and cov['cutoff']
    vocab = set(d['conventions']['evidence_status'])
    assert {'consistency_verified', 'source_values_verified', 'integrity_receipt_verified',
            'report_only_indexed', 'unknown_or_pending'} <= vocab
    for e in d['entries']:
        st = e.get('structured') or {}
        status = st.get('evidence_status') or e.get('evidence_status')
        if st:
            assert status in vocab, (e['id'], status)


def test_all_referenced_paths_exist():
    d = _load()
    tool = _tool()

    def present(p):
        return Path(tool.resolve(p)).exists() or tool.recorded_deletion(p) or bool(tool.offloaded(p))

    missing = []
    for e in d['entries']:
        for p in e.get('paths', []) or []:
            if not present(p.split(' (')[0]):
                missing.append((e['id'], p))
        st = e.get('structured') or {}
        for p in st.get('artifacts', []) or []:
            if not present(p):
                missing.append((e['id'], p))
        ns = st.get('numbers_source')
        if ns and not present(ns):
            missing.append((e['id'], ns))
    assert not missing, missing[:10]


def test_sign_arithmetic_on_structured_outcomes():
    d = _load()
    checked = 0
    for e in d['entries']:
        for o in (e.get('structured') or {}).get('outcomes', []) or []:
            for m, v in (o.get('outcomes') or {}).items():
                if not isinstance(v, dict) or 'cost_difference' not in v:
                    continue
                checked += 1
                assert v.get('verified') is True, (e['id'], m)
                if 'cost_treatment' in v and 'cost_reference' in v:
                    assert abs((v['cost_treatment'] - v['cost_reference'])
                               - v['cost_difference']) < 1e-6, (e['id'], m)
                if 'improvement' in v:
                    assert abs(v['improvement'] + v['cost_difference']) < 1e-6, (e['id'], m)
    assert checked >= 40


def test_corrected_upward_arm_signs_and_ttc_only_failure():
    """CORR-6 regression: 4107 raise-26 improves service/braking/effort/wait; TTC only fails."""
    d = _load()
    e = next(x for x in d['entries'] if x['id'] == 'TA-03')
    row = next(o for o in e['structured']['outcomes'] if '4107' in (o.get('treatment_id') or ''))
    cd = {m: v['cost_difference'] for m, v in row['outcomes'].items()}
    assert cd['system_seconds_per_request'] == pytest.approx(-8.435689, abs=1e-5)
    assert cd['braking_seconds_per_request'] == pytest.approx(-0.057597, abs=1e-5)
    assert cd['effort_per_request'] == pytest.approx(-5.745149, abs=1e-5)
    assert cd['ttc_seconds_per_request'] == pytest.approx(0.039576, abs=1e-5)
    assert cd['source_wait_seconds_per_request'] == pytest.approx(-0.133569, abs=1e-5)
    assert row['guardrail_failures'] == ['ttc_seconds_per_request']
    assert row['collisions_treatment'] == 0 and row['collisions_reference'] == 4


def test_reused_treatment_runs_carry_zero_new_simulation():
    d = _load()
    runs = {r['run_id']: r for r in d.get('runs', [])}
    reused = ('reused_run', 'trace_equivalent', 'exact_reuse', 'replay_verification',
              'reused_reference', 'alias_of_canonical')
    for c in d.get('comparisons', []):
        if c.get('treatment_class') in reused:
            r = runs[c['treatment_run']]
            assert not r.get('actual_simulated_seconds'), (c['comparison_id'], c['treatment_run'])


def test_original_execution_cost_retained_and_alias_free():
    """Reuse costs nothing additional; the canonical run keeps its full cost."""
    d = _load()
    runs = {r['run_id']: r for r in d.get('runs', [])}
    alias = runs['R-LC-C-4107-reuse']
    canon = runs[alias['canonical_run']]
    assert alias['canonical_run'] == 'R-LC-B-4107'
    assert alias['actual_simulated_seconds'] == 0 and alias['charged_seconds'] == 0
    assert canon['actual_simulated_seconds'] == 2400 and canon['charged_seconds'] == 2400
    for r in d.get('runs', []):
        if r.get('canonical_run'):
            assert (r.get('charged_seconds') or 0) == 0, r['run_id']
            assert runs[r['canonical_run']]['actual_simulated_seconds'], r['run_id']


def test_campaign_component_and_run_charges_reconcile():
    d = _load()
    campaigns = {c['campaign_id']: c for c in d.get('campaigns', [])}
    for cid, c in campaigns.items():
        comp = c.get('components') or {}
        defined = [v for v in comp.values() if v[0] is not None] if comp else []
        if defined and c.get('actual_progress_s') is not None and len(defined) == len(comp):
            assert abs(sum(v[0] for v in defined) - c['actual_progress_s']) < 0.5, cid
    for cid in ('CAMP-GENTLE', 'CAMP-BRAKE-HOLD', 'CAMP-LIFECYCLE', 'CAMP-TARGET-SPEED',
                'CAMP-TARGET-ADJUST-V3', 'CAMP-ESCAL-EXEC', 'CAMP-OBSERVER-REPLAY',
                'CAMP-DISSOLUTION'):
        c = campaigns[cid]
        tot = 0.0
        for r in d.get('runs', []):
            if r['campaign'] == cid:
                tot += r.get('charged_seconds') if r.get('charged_seconds') is not None \
                    else (r.get('allocated_horizon_s') or 0)
        assert abs(tot - c['allocated_charged_s']) < 0.5, (cid, tot, c['allocated_charged_s'])


def test_entry_and_continuation_frames_are_distinct():
    d = _load()
    kinds = {c['comparison_kind'] for c in d.get('comparisons', [])}
    assert 'entry_vs_policy' in kinds
    assert 'continuation_keep_vs_release' in kinds
    assert 'continuation_keep_vs_adjust' in kinds
    assert 'continuation_vs_policy_benchmark' in kinds


def test_continuation_evidence_appears_in_speed_queries():
    tool = _tool()
    d = _load()
    hits = [e['id'] for e in d['entries']
            if any(t in tool._entry_text(e) for t in ('maintained', 'target', 'speed'))]
    assert 'TA-03' in hits and 'TA-01' in hits and 'TH-07' in hits


def test_coverage_warnings_for_relevant_pending_evidence():
    tool = _tool()
    d = _load()
    assert tool.pending_hits(d, 'should we sweep maintained target speed again')
    assert tool.pending_hits(d, 'TTC warning lead times')
    assert not tool.pending_hits(d, 'krauss margin law refit')


def test_validator_reports_no_hard_errors():
    tool = _tool()
    errors, warnings, _, unknowns, _, _ = tool.validate(_load())
    assert errors == [], errors[:10]
    assert unknowns, 'sealed/unknown documentation must be present'


def test_source_verification_clean_on_real_ledger():
    tool = _tool()
    report, mismatches, _, checked = tool.verify_sources(_load())
    assert report is not None and checked >= 150
    assert mismatches == [], mismatches[:5]


def test_source_verification_detects_source_mismatch_and_change():
    """An internally consistent but source-mismatched ledger must be caught, and a
    changed/missing source must be flagged - consistency alone is not verification."""
    tool = _tool()
    d = _load()
    mutated = copy.deepcopy(d)
    for e in mutated['entries']:
        if e['id'] == 'TA-03':
            for o in e['structured']['outcomes']:
                if '4107' in (o.get('treatment_id') or ''):
                    o['outcomes']['effort_per_request']['cost_difference'] = 5.745149
                    o['outcomes']['effort_per_request']['improvement'] = -5.745149
    _, mismatches, _, _ = tool.verify_sources(mutated)
    assert any('effort_per_request' in m and 'TA-03' in m for m in mismatches)
    d2 = copy.deepcopy(_load())
    d2['source_verification']['allowed_sources'][0]['sha256_registered'] = '0' * 64
    _, mism2, _, _ = tool.verify_sources(d2)
    assert any('SOURCE CHANGED' in m for m in mism2)
    d3 = copy.deepcopy(_load())
    d3['source_verification']['allowed_sources'][0]['path'] = 'runs/active/does_not_exist.json'
    _, mism3, _, _ = tool.verify_sources(d3)
    assert any('missing' in m for m in mism3)


def test_pending_update_lifecycle():
    """Pending entries may be resolved, but only to verified extents; blocks still
    carrying unresolved items must not be dropped from coverage."""
    d = _load()
    entries = {e['id'] for e in d['entries']}
    pu = d.get('pending_updates', [])
    assert pu, 'a pending-updates lifecycle must exist (resolved or not)'
    for p in pu:
        assert p.get('status') in ('partially_resolved', 'resolved_incorporated',
                                   'excluded_pending_review'), p['pending_id']
        for it in p.get('resolved_items', []) or []:
            for eid in it.get('entries', []) or []:
                assert eid in entries, (p['pending_id'], eid)
    assert 'TA-03' in entries
    still = json.dumps(d['coverage'].get('excluded_and_pending', []))
    assert 'raise-26' not in still and 'lower-22' not in still
    for p in pu:
        for it in p.get('remaining_pending', []) or []:
            assert it.get('why'), (p['pending_id'], it.get('item'))


def test_ttc_diagnosis_recorded_with_prohibitions():
    d = _load()
    ttc = d.get('ttc_diagnosis') or {}
    assert 'superseded' in ttc.get('status', '') or 'under_versioned_review' in ttc.get('status', '')
    gt = ttc['ground_truth_4107_raise_vs_damping']
    assert gt['only_violated_guardrail'] == 'ttc'
    assert gt['effort'] < 0 and gt['wait'] < 0 and gt['ttc'] > 0
    assert len(ttc.get('known_issues', [])) >= 4
    joined = json.dumps(ttc)
    assert 'NOT action-validated' in joined or 'not action-validated' in joined.lower()
    assert 'withdrawn' in joined and 'aggregate post-release predictability is open' in joined


# ---------------------------------------------------------------- final pass

def test_artifact_review_registry_assigns_statuses_from_files():
    """Version-level review statuses must exist and v6 must NOT be treated as done."""
    d = _load()
    ar = d.get('artifact_review', {})
    arts = {a['path']: a for a in ar.get('artifacts', [])}
    v6 = arts['output/wall_harm_speed_v1/fusion_warning_v6.py']
    assert v6['status_overall'] == 'correction_pending'
    assert v6.get('report_published') is False
    v3 = arts['output/ttc_diagnosis_v1/REPORT_TTC_V3.md']
    assert v3['status_overall'] == 'superseded' and v3.get('replacement')
    phases = arts['output/ttc_diagnosis_v1/TTC_PHASE_V4.md']
    assert phases['status_overall'] == 'current'
    assert phases.get('numerical_findings_verified') is True
    for a in ar['artifacts']:
        assert a.get('status_overall') in ('current', 'partially_superseded',
                                           'superseded', 'correction_pending')


def test_verification_fails_when_required_source_metric_deleted(tmp_path):
    """Deleting a required source metric must fail verification (no silent skip)."""
    tool = _tool()
    d = _load()
    src_id = 'SRC-DISS'
    src = next(x for x in d['source_verification']['allowed_sources']
               if x['source_id'] == src_id)
    real = ROOT / src['path']
    text = json.loads(real.read_text(encoding='utf-8'))
    for r in text['results']:
        r['effect']['costs_control'].pop('source_wait_seconds_per_request', None)
    mutated_path = tmp_path / 'diss_mutated.json'
    mutated_path.write_text(json.dumps(text), encoding='utf-8')
    d2 = copy.deepcopy(d)
    for x in d2['source_verification']['allowed_sources']:
        if x['source_id'] == src_id:
            x['path'] = str(mutated_path)
            x['sha256_registered'] = None
    report, mismatches, _, _ = tool.verify_sources(d2)
    assert any('source_wait_seconds_per_request' in m and 'missing' in m
               for m in mismatches), mismatches[:5]
    rec = next(r for r in report['sources'] if r['source_id'] == src_id)
    assert rec['status'] == 'failure'
    assert any(not sk.get('intentional') for sk in rec['skipped_checks'])


def test_mismatched_metric_cannot_be_fully_verified():
    """A stored value that disagrees with the source blocks clean verification."""
    tool = _tool()
    mutated = copy.deepcopy(_load())
    e = next(x for x in mutated['entries'] if x['id'] == 'CUR-12')
    row = next(o for o in e['structured']['outcomes']
               if '4107' in (o.get('treatment_id') or ''))
    row['outcomes']['braking_seconds_per_request']['cost_difference'] = 0.999
    _, mismatches, _, _ = tool.verify_sources(mutated)
    assert any('braking_seconds_per_request' in m for m in mismatches)


def test_nan_and_infinity_rejected():
    """Non-finite stored values must fail both consistency validation and verification."""
    tool = _tool()
    mutated = copy.deepcopy(_load())
    for e in mutated['entries']:
        if e['id'] == 'TA-03':
            for o in e['structured']['outcomes']:
                if '4107' in (o.get('treatment_id') or ''):
                    o['outcomes']['effort_per_request']['cost_difference'] = float('nan')
                    o['outcomes']['effort_per_request']['improvement'] = float('-inf')
    errors, _, _, _, _, _ = tool.validate(mutated)
    assert any('non-finite' in m for m in errors)
    _, mismatches, _, _ = tool.verify_sources(mutated)
    assert any('non-finite' in m for m in mismatches)


def test_checksum_change_is_detected(tmp_path):
    tool = _tool()
    d = _load()
    src = d['source_verification']['allowed_sources'][0]
    real = ROOT / src['path']
    obj = json.loads(real.read_text(encoding='utf-8'))
    obj['probe'] = 'checksum-changed'
    p = tmp_path / 'changed_source.json'
    p.write_text(json.dumps(obj), encoding='utf-8')
    d2 = copy.deepcopy(d)
    d2['source_verification']['allowed_sources'][0]['path'] = str(p)
    report, mismatches, _, _ = tool.verify_sources(d2)
    assert any('SOURCE CHANGED' in m for m in mismatches)
    rec = next(r for r in report['sources'] if r['source_id'] == src['source_id'])
    assert rec['status'] == 'failure'


def test_partial_review_stays_partial(tmp_path):
    """A source whose checks include an intentional exclusion is 'partial', never
    'complete'; the exclusion carries a recorded reason and adds no mismatch.
    Built by removing one metric from the SOURCE duration_contrast only."""
    tool = _tool()
    d = _load()
    src = next(x for x in d['source_verification']['allowed_sources']
               if x['source_id'] == 'SRC-DUR')
    text = json.loads((ROOT / src['path']).read_text(encoding='utf-8'))
    for r in text['results']:
        dc = (r.get('duration_contrast') or {}).get('cost_release_minus_cost_continue')
        if dc is not None:
            dc.pop('source_wait_seconds_per_request', None)
    p2 = tmp_path / 'dur_mutated.json'
    p2.write_text(json.dumps(text), encoding='utf-8')
    d2 = copy.deepcopy(d)
    for x in d2['source_verification']['allowed_sources']:
        if x['source_id'] == 'SRC-DUR':
            x['path'] = str(p2)
            x['sha256_registered'] = None
    report, mismatches, _, _ = tool.verify_sources(d2)
    assert mismatches == [], mismatches[:3]
    rec = next(r for r in report['sources'] if r['source_id'] == 'SRC-DUR')
    assert rec['status'] == 'partial'
    assert any(sk.get('intentional') for sk in rec['skipped_checks'])


def test_superseded_reports_marked_in_navigation():
    """Current navigation must visibly mark superseded/disputed items."""
    idx = (ROOT / 'docs' / 'INDEX.md').read_text(encoding='utf-8')
    assert 'superseded' in idx.lower()
    for marker in ('PART3_RESULTS.md', 'REPORT_TTC_V3.md', 'OBSERVATION_INVENTORY_V3.md'):
        line = next(ln for ln in idx.splitlines() if marker in ln)
        assert 'superseded' in line.lower(), line
    route = (ROOT / 'docs' / 'ROUTE.md').read_text(encoding='utf-8')
    assert 'Current state' in route and 'read this first' in route


def test_generated_summary_has_units_and_comparators():
    text = (ROOT / 'docs' / 'RESULTS_SUMMARY.md').read_text(encoding='utf-8')
    assert 'cost difference' in text and 'negative = improvement' in text
    assert 'sha256' in text
    assert 'vehicle-seconds' in text and 'm^2/s^3' in text
    assert 'vs keep-24 source + reused damping' in text
    assert 'frame: entry_vs_policy' in text and 'frame: continuation' in text
    assert 'frame: characterization' in text
    assert '[None]' not in text


def test_generated_summary_is_current():
    """summary --check invariant: the committed file matches the structured records."""
    tool = _tool()
    regenerated = tool.build_summary_text(_load(), tool.sha256_of(tool.LEDGER))
    current = (ROOT / 'docs' / 'RESULTS_SUMMARY.md').read_text(encoding='utf-8')
    assert current == regenerated, 'RESULTS_SUMMARY.md is stale; run summary --write'


def test_multi_comparison_reuse_is_informational_not_error():
    tool = _tool()
    errors, warnings, infos, _, _, _ = tool.validate(_load())
    assert any('R-LC-D-4107' in m and 'distinct frames' in m for m in infos)
    assert not any('R-LC-D-4107' in m and 'double counting' in m
                   for m in errors + warnings)


def test_double_charging_remains_an_error():
    tool = _tool()
    d = copy.deepcopy(_load())
    d['runs'].append({'run_id': 'R-FAKE-DOUBLE', 'campaign': 'CAMP-LIFECYCLE',
                      'kind': 'corridor_episode', 'status': 'complete', 'role': 'treatment',
                      'treatment_class': 'fresh', 'actual_simulated_seconds': 2400,
                      'charged_seconds': 2400,
                      'path': 'runs/active/duration_release_v1/jobs/s4107_m2700_p20_t600_release_h5_wall'})
    errors, _, _, _, _, _ = tool.validate(d)
    assert any('double charging' in m for m in errors)


def test_offloaded_paths_resolve_to_their_drive_copy(tmp_path):
    tool = _tool()
    gone = tmp_path / 'output' / 'study_v1'
    m = {'schema': 'offloaded_paths/1', 'entries': [
        {'local': str(gone), 'drive': 'gdrive:research-archive/repo-evidence/output/study_v1/study_v1.tar.zst'}]}
    (tmp_path / 'map.json').write_text(json.dumps(m), encoding='utf-8')
    tool.OFFLOADED = str(tmp_path / 'map.json')
    tool._offloaded = None
    assert tool.offloaded(str(gone / 'REPORT.md')).endswith('study_v1.tar.zst')
    assert tool.offloaded(str(gone)) is not None
    assert tool.offloaded(str(tmp_path / 'output' / 'study_v10' / 'x.json')) is None
    assert tool.offloaded('') is None


def test_offload_map_is_well_formed():
    spec = importlib.util.spec_from_file_location('offload_map_tool', ROOT / 'tools' / 'offload_map.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check() == []


def test_offloaded_repository_paths_are_worktree_independent(tmp_path):
    tool = _tool()
    receipt = {'schema': 'offloaded_paths/1', 'entries': [
        {'local': 'E:/research-dense/output/study_v1',
         'drive': 'gdrive:research-archive/study_v1.tar.zst'}]}
    path = tmp_path / 'map.json'
    path.write_text(json.dumps(receipt), encoding='utf-8')
    tool.OFFLOADED = str(path)
    expected = receipt['entries'][0]['drive']
    for source in ('output/study_v1/REPORT.md', str(ROOT / 'output/study_v1/REPORT.md'),
                   'D:/research-dense/output/study_v1/REPORT.md'):
        assert tool.offloaded(source) == expected
    assert tool.offloaded('output/study_v10/REPORT.md') is None
    assert tool.offloaded('output/unregistered_study/REPORT.md') is None
    assert tool.offloaded(str(tmp_path / 'output/study_v1/REPORT.md')) is None


def test_relocated_maps_paths_inside_a_moved_directory():
    tool = _tool()
    tool._relocated = {'output/old_study': 'sealed/relocated_20261010/E/research-dense/output/old_study'}
    tool._relocated_dirs = [('output/old_study', 'sealed/relocated_20261010/E/research-dense/output/old_study')]
    assert tool.relocated('output/old_study') == 'sealed/relocated_20261010/E/research-dense/output/old_study'
    assert tool.relocated('output/old_study/sub/x.json') == 'sealed/relocated_20261010/E/research-dense/output/old_study/sub/x.json'
    assert tool.relocated('output/old_study_v2/x.json') is None
