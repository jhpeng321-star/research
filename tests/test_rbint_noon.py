import json
from pathlib import Path

import pytest

from tools.rbint import continue_noon as collector
from tools.rbint.noon_report import render


def receipt(root, tasks):
    root.mkdir()
    hashes = {'shard0/IDENTITY.json': 'source'}
    hashes.update({f'shard0/w{seed}_k{k}.json': f'{seed}:{k}' for seed, k in tasks})
    (root/'SNAPSHOT_RECEIPT.json').write_text(json.dumps(dict(hashes=hashes)))
    return root


def test_only_complete_ensembles_change_the_cached_scientific_score(tmp_path):
    complete = [(969432, k) for k in range(8)]
    a = collector.fingerprint(receipt(tmp_path/'a', complete+[(969433, 0)]))
    b = collector.fingerprint(receipt(tmp_path/'b', complete+[(969433, k) for k in range(7)]))
    c = collector.fingerprint(receipt(tmp_path/'c', complete+[(969433, k) for k in range(8)]))
    assert a['sha256'] == b['sha256'] and a['tasks'] != b['tasks']
    assert c['sha256'] != a['sha256'] and c['complete'] == [969432, 969433]


def test_milestone_wakes_do_not_freeze_or_finalize_early():
    assert collector.next_wake(collector.INTERIM-1) == collector.INTERIM
    assert collector.next_wake(collector.FREEZE-1) == collector.FREEZE
    assert collector.next_wake(collector.FINAL-1) == collector.FINAL


def test_foreign_staged_or_uncommitted_work_stops_publication(monkeypatch):
    def git(*args):
        if args[0] == 'branch':
            return 'c4/rb-int'
        if args[:2] == ('diff', '--cached'):
            return ''
        if args[0] == 'status':
            return ' M HANDOFF.md\n?? docs/rbint/noon/round/a.json'
        raise AssertionError(args)
    monkeypatch.setattr(collector, 'git', git)
    collector.guard(allowed=['HANDOFF.md', 'docs/rbint/noon/round'])
    with pytest.raises(RuntimeError, match='foreign/uncommitted'):
        collector.guard(allowed=['docs/rbint/noon/round'])


def test_handoff_updates_only_owned_block():
    original = 'Peer ownership must survive.\n'
    first = collector.update_handoff(original, 'first')
    final = collector.update_handoff(first, 'second')
    assert final.startswith(original)
    assert final.count('<!-- NOON-COLLECTOR:BEGIN -->') == 1
    assert 'second' in final and 'first' not in final


def test_missing_variant_and_cis_render_as_pending_within_sixty_lines(tmp_path):
    publication = dict(variants={}, label='interim', local_time='2026-11-01T08:00:00+08:00',
                       receipt_directory='docs/rbint/noon/example')
    text = render(publication)
    assert len(text.splitlines()) <= 60
    assert 'C2:PENDING' in text and '95%CI NA' in text
    assert 'no automatic S-C1SWITCH adoption' in text
    assert 'RETRACTED' in text


def test_render_uses_same_candidate_and_common_cohort_not_a_different_source(tmp_path):
    source = tmp_path/'comparison.json'
    comparisons = [dict(metric='delay_crps_difference', horizon_s=h, mean=-10*h, low=None, high=None)
                   for h in (120, 300, 600)]
    comparisons.append(dict(metric='upstream_rate_distance_difference', horizon_s=300, mean=.1, low=None, high=None))
    source.write_text(json.dumps(dict(summaries=dict(reserved8=dict(worlds=[969432], comparisons=comparisons)))))
    item = dict(comparison=str(source), tasks=8)
    publication = dict(variants={'FULL-v4': item}, label='final', local_time='date', receipt_directory='receipt')
    text = render(publication)
    assert 'current paired diagnostic below is FULL-v4' in text
    assert 'FULL-v4−WM2 delay CRPS' in text
    assert 'HDV60 non-regression PENDING' in text
    assert len(text.splitlines()) <= 60
