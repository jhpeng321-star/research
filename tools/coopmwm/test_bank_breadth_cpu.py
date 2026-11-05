"""F3 contracts with native bank admission, real trigger learning and CPU fakes."""
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from . import bank_breadth as breadth, exact_grade as xg, practice_starts as ps
from . import replay_bank as rb, track2_backend as p1, launch_production as launch
from .test_track2_backend_cpu import cfg, cpu_threads, sample, rows  # noqa: F401


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_plan_fix_missing_empty_and_explicit_verified_output_contract(tmp_path):
    config = dict(run_root=str(tmp_path), training_seed=1)
    original = breadth.inputs(config)
    ready = tmp_path/'PLAN_FIX_READY.json'
    write(ready, dict(schema=breadth.SCHEMA, outputs=[]))
    assert breadth.inputs(config) == original
    write(ready, dict(schema=breadth.SCHEMA, outputs=[dict(run_root='library',
        replay_sources=dict(B=['variety/B'], R=['variety/R']))]))
    sources = breadth.inputs(config)
    assert len(sources) == 2
    assert sources[1]['run_root'] == str((tmp_path/'library').resolve())
    assert sources[1]['replay_sources']['R'] == [str((tmp_path/'variety/R').resolve())]


def test_handoff_is_not_allowed_to_supply_unverified_labels(tmp_path):
    write(tmp_path/'PLAN_FIX_READY.json', dict(outputs=[dict(moves=['wall'])]))
    with pytest.raises(ValueError, match='contract'):
        breadth.inputs(dict(run_root=str(tmp_path)))


def test_restricted_handoff_paths_refused_before_read(tmp_path):
    # No file or directory at this synthetic restricted path is created/opened.
    with pytest.raises(ValueError, match='restricted'):
        breadth.inputs(dict(run_root=str(tmp_path), plan_fix_ready=str(tmp_path/'sealed'/'x.json')))


def test_native_admission_all_examples_plus_plan_fix_and_snapshot_integrity(tmp_path, monkeypatch):
    from .test_practice_starts_cpu import bank_with_snapshots
    from .test_bank_perseed_cpu import make, lose, noisy
    from . import effective_imitation as effective, bank_v2v, bank_statistics
    bank_statistics._CACHE.clear()
    # Only observable feature attachment is faked: native sealed receipts,
    # per-seed confidence/safety admission and tape hashes are actually read.
    def limits(pairs, tape, trace, entry, cap):
        assert cap == 1.2
        return pairs, dict(pairs=len(pairs), kept=len(pairs), cap_mps2=cap,
            excluded_missing_limit=0, excluded_command_mismatch=0), {}
    def attach(pairs, trace, t0, cfg=None, intent=None):
        for i, (s, _) in enumerate(pairs):
            s['intent_link'] = dict(t=t0+i, entry='cpu', slots={})
        return pairs, dict(pairs=len(pairs), kept=len(pairs),
            excluded_missing_prev_second=0, excluded_missing_partner_command=0, trace_missing=False)
    monkeypatch.setattr(effective, 'attach_limits', limits)
    monkeypatch.setattr(bank_v2v, 'attach_pairs', attach)
    source = tmp_path/'verified_library'
    for root, ids in ((tmp_path, ('old1', 'old2', 'old3', 'old4')),
                      (source, ('library', 'variety'))):
        for arm in ('B', 'R'):
            bank_with_snapshots(root, arm, vids=ids, snapshot_seed=1 if root == tmp_path else 30)
            for path in (rb.bank_root(root, arm)/'snapshots').glob('*.pt'):
                snap = torch.load(path, weights_only=False)
                snap['episode']['arm'] = arm
                torch.save(snap, path)
            make(root, 'rejected', lose(), lose(), arm=arm,
                 syn1=noisy(2., .5, 9), syn2=noisy(2., .5, 10))
            for path in (rb.bank_root(root, arm)/'sumo_verification'/'verifications').glob('*.json'):
                receipt = json.loads(path.read_text())
                receipt['chosen_plan']['perturbation_profile'] = [[-.3]*20]*2
                write(path, xg._seal(receipt, 'receipt_sha256'))
    write(tmp_path/'PLAN_FIX_READY.json', dict(schema=breadth.SCHEMA,
        outputs=[dict(run_root='verified_library')]))
    # Also exercise the existing LS import format, formerly forbidden for
    # practice even when confirmed. No own-bank decision link for old4 remains.
    import gzip
    import io
    replay_sources = {}
    for arm in ('B', 'R'):
        bank = rb.bank_root(tmp_path, arm)
        external = tmp_path/'legacy_labels'/arm
        receipt_path = bank/'sumo_verification'/'verifications'/'old4.json'
        receipt = json.loads(receipt_path.read_text())
        receipt_path.unlink()
        # Native LS tapes reside beneath the label source.
        import shutil
        shutil.copytree(bank/'sumo_verification'/'branches'/'old4', external/'branches'/'old4')
        write(external/'labels'/'search.json', dict(search_id='search', verification=receipt))
        link = ps._links(tmp_path, arm)['old4']
        (bank/'decisions'/(link['state_id']+'.json')).unlink()
        snap = torch.load(bank/'snapshots'/(link['state_id']+'.pt'), weights_only=False)
        episode = tmp_path/'episodes'/arm
        write(episode/'RESULT.json', dict(id=snap['episode']['id'], world=snap['episode']['world'],
            prefixes=snap['prefixes'], steps=[dict(time_s=int(t), sample=s) for t, s in snap['samples'].items()]))
        snap['current_sample'] = snap['samples'][str(snap['t0'])]
        payload = io.BytesIO(); torch.save(snap, payload)
        (external/'snapshots').mkdir(parents=True)
        (external/'snapshots'/'state.pt.gz').write_bytes(gzip.compress(payload.getvalue()))
        (external/'pending').mkdir()
        (external/'pending'/'search.json.gz').write_bytes(gzip.compress(json.dumps(
            dict(snapshot='state', episode_path=str(episode))).encode()))
        replay_sources[arm] = [str(external)]
    config = dict(run_root=str(tmp_path), training_seed=1, replay_sources=replay_sources)
    pool = breadth.practice_pool(config, 'B')
    assert {r['verification_id'] for r in pool} == {'old1', 'old2', 'old3', 'old4', 'library', 'variety'}
    assert len(pool) == 12  # six contexts x two admitted verified seeds, not three contexts
    assert all('rejected' != r['verification_id'] for r in pool)
    ls_row = next(r for r in pool if r['verification_id'] == 'old4' and r['seed_index'] == 1)
    assert ls_row['state_id'].startswith('imported_')
    assert ps.load_snapshot(tmp_path, 'B', ls_row)['t0'] == 400
    imported = next(r for r in pool if r['verification_id'] == 'library' and r['seed_index'] == 2)
    snap = ps.load_snapshot(tmp_path, 'B', imported)
    assert snap['rng']['replication_salt'] == 'library'
    broken = dict(imported, snapshot_sha256='bad')
    with pytest.raises(ValueError, match='snapshot_sha256'):
        ps.load_snapshot(tmp_path, 'B', broken)
    branch = rb.bank_root(source, 'B')/'sumo_verification'/'branches'/'library'/'cand1'/'RESULT.json'
    tape = json.loads(branch.read_text()); tape['tampered'] = True; write(branch, tape)
    with pytest.raises(ValueError, match='tape differs|result.*hash|sha256|differs'):
        breadth.practice_pool(config, 'B')


def fake_pool(arm, nwall=7, ngap=4):
    return [dict(context_id=f'{arm}_{family}_{i}', family=family,
        example_id=f'{arm}_{family}_{i}', fallback=False)
        for family, n in (('wall', nwall), ('gap_open', ngap)) for i in range(n)]


def test_practice_uses_full_pool_before_repetition_and_rotates_across_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(breadth, 'practice_pool', lambda cfg, arm: fake_pool(arm))
    config = dict(run_root=str(tmp_path))
    common = dict(fraction=.5, repeat_cap=2, count=6, training_seed=1, breadth_cfg=config)
    seen = set()
    for update in range(1, 5):
        b = ps.plan_for_update(tmp_path, 'B', update, **common)
        r = ps.plan_for_update(tmp_path, 'R', update, **common)
        assert b['breadth']['family_counts'] == r['breadth']['family_counts']
        assert b['breadth']['unique_selected'] == 6
        assert b['registry']['pool_examples'] == 11
        seen.update(row['context_id'] for row in b['rows'])
        assert ps.plan_for_update(tmp_path, 'B', update, **common) == b
    assert seen == {row['context_id'] for row in fake_pool('B')}


def test_family_distribution_matches_when_bank_sizes_differ(tmp_path, monkeypatch):
    monkeypatch.setattr(breadth, 'practice_pool', lambda cfg, arm:
        fake_pool(arm, nwall=10 if arm == 'B' else 2, ngap=2 if arm == 'B' else 9))
    config = dict(run_root=str(tmp_path))
    kwargs = dict(cfg=config, fraction=.5, repeat_cap=2, count=8, training_seed=3)
    b = breadth.practice_plan(tmp_path, 'B', 1, **kwargs)
    r = breadth.practice_plan(tmp_path, 'R', 1, **kwargs)
    assert Counter(row['family'] for row in b['rows']) == Counter(row['family'] for row in r['rows'])
    assert b['breadth']['family_counts'] == {'gap_open': 4, 'wall': 4}
    assert max(Counter(row['context_id'] for row in r['rows']).values()) <= 2


def test_no_silent_family_bias_or_old_plan_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(breadth, 'practice_pool', lambda cfg, arm:
        fake_pool(arm, ngap=0 if arm == 'R' else 1))
    with pytest.raises(ValueError, match='family support differs'):
        breadth.matched_pools(dict(run_root=str(tmp_path)))
    write(ps.plan_path(tmp_path, 'B', 1), xg._seal(dict(arm='B', update=1, rows=[]), 'practice_plan_sha256'))
    with pytest.raises(ValueError, match='predates breadth'):
        ps.plan_for_update(tmp_path, 'B', 1, fraction=.5, repeat_cap=2,
            count=8, training_seed=1, breadth_cfg=dict(run_root=str(tmp_path)))


def test_r_matches_realized_b_families_after_normal_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(breadth, 'practice_pool', lambda cfg, arm: fake_pool(arm))
    config = dict(run_root=str(tmp_path))
    common = dict(fraction=.5, repeat_cap=2, count=6, training_seed=1, breadth_cfg=config)
    b = ps.plan_for_update(tmp_path, 'B', 1, **common)
    actual = [r for i, r in enumerate(b['rows']) if i != 0]  # failed first slot became normal
    write(tmp_path/'B_UPDATE_001.json', dict(practice=dict(practice=5,
        episodes=[dict(breadth=dict(family=r['family'])) for r in actual])))
    r = ps.plan_for_update(tmp_path, 'R', 1, **dict(common, count=5))
    assert [row['family'] for row in r['rows']] == [row['family'] for row in actual]
    assert r['breadth']['family_counts'] == dict(Counter(row['family'] for row in actual))


def test_realized_breadth_report_includes_identity_and_family_counts():
    episodes = [dict(id=str(i), world=dict(seed=986000+i), steps=[dict(reward=-1.)],
        practice=dict(verification_id='v', example_id='v#s1', seed_index=1, t0=400,
            breadth=dict(context_id='same' if i < 2 else 'other', family='wall', source='verified_breadth')))
        for i in range(3)]
    report = ps.update_report(episodes)
    assert report['breadth'] == dict(practice=3, unique_contexts=2,
        context_counts=dict(same=2, other=1), family_counts=dict(wall=3))
    assert report['episodes'][0]['breadth']['context_id'] == 'same'


def test_a_never_reads_banks_or_handoff(cfg, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A must not read guidance')
    monkeypatch.setattr(breadth, 'inputs', forbidden)
    breadth.visit(cfg, 'A', forbidden)
    runtime = SimpleNamespace(arm='A')
    assert launch.bank_move_supervision(runtime) == dict(examples=0, source='A_no_bank')
    assert ps.enabled(dict(practice_starts=True), 'A') is False


def test_supervision_scans_past_missing_first_frame_and_never_commands_hidden_members(cfg):
    from . import moves
    policy, recorded, _ = sample(cfg)
    plan = dict(members=['c1', 'c2'], duration_s=40,
        perturbation_profile=[[0.]*5+[-.3]*35]*2)
    pairs = [(dict(recorded, intent_link=dict(t=300+i)), 20.) for i in (0, 1, 2)]
    # The first sample has no trace frame. At 301 c2 is out of range;
    # at 302 both are visible. This formerly lost the entire example.
    far = deepcopy(rows()); far[1]['s_m'] = 10000.
    frames = {301: dict(vehicles=far), 302: dict(vehicles=rows())}
    got = list(breadth.observable_contexts(policy, plan, pairs, frames, t0=300))
    assert len(got) == 1
    context, target, reader, _, t = got[0]
    assert t == 302 and set(context['members']) == {'c1', 'c2'}
    assert target.onset == max(0, moves.from_plan(plan).onset-2)
    assert set(target.members) <= set(reader.visible['c1'])
    frames[302]['vehicles'] = far
    assert list(breadth.observable_contexts(policy, plan, pairs, frames, t0=300)) == []


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_all_context_ce_and_family_mass_not_legacy_weight_concentration(cfg, monkeypatch, arm):
    from . import bank_v2v
    policy, recorded, _ = sample(cfg)
    before = deepcopy(policy.layer.base.state_dict())
    plans = [dict(id='wall', members=['c1', 'c2'], duration_s=30,
                  perturbation_profile=[[-.3]*30]*2),
             dict(id='ease', members=['c1'], duration_s=30,
                  perturbation_profile=[[-.3]*30])]
    def visit(config, requested_arm, consume):
        for i in range(8):
            consume(cfg=config, receipt=dict(verification_id=f'{requested_arm}_{i}', t0=300),
                example=dict(example_id=f'{requested_arm}_{i}#s1'), plan=plans[i == 7],
                pairs=[(dict(recorded, intent_link=dict(t=300)), 20. if i < 7 else 1.)], trace='CPU')
    monkeypatch.setattr(breadth, 'visit', visit)
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda path:
        {299: dict(vehicles=rows()), 300: dict(vehicles=rows())})
    result = launch.bank_move_supervision(SimpleNamespace(cfg=cfg, policy=policy, arm=arm))
    assert result['examples'] == 8
    assert len({row['example_id'] for row in result['contexts']}) == 8
    assert result['family_counts'] == {'wall': 7, 'ease_off': 1}
    assert result['family_mass'] == {'wall': .5, 'ease_off': .5}
    assert result['final_ce'] < result['initial_ce']
    assert all(torch.equal(v, policy.layer.base.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in policy.layer.base.parameters())
