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


def bind_fixture_snapshots(root, arm):
    """Give the old statistical fixtures original collector-grade evidence.

    The fixtures predate snapshot provenance and intentionally do not model it.
    Stamp receipts once during setup, never when a pool reads the bank.
    """
    from . import sumo_verify as sv, bank_statistics
    for vid, link in ps._links(root, arm).items():
        snap_path = rb.bank_root(root, arm)/'snapshots'/(link['state_id']+'.pt')
        snap = torch.load(snap_path, weights_only=False)
        snap['episode']['arm'] = arm
        torch.save(snap, snap_path)
        for kind, own in (('sumo_verification', snap), ('confirm_replication', rb.replicate(snap, vid))):
            path = rb.bank_root(root, arm)/kind/'verifications'/(vid+'.json')
            receipt = json.loads(path.read_text())
            receipt.update(snapshot_sha256=sv.snapshot_sha(own), episode=own['episode'], t0=own['t0'])
            receipt['chosen_plan']['perturbation_profile'] = [[-.3]*20]*2
            receipt = xg._seal(receipt, 'receipt_sha256')
            write(path, receipt)
            if kind == 'confirm_replication':
                cpath = rb.bank_root(root, arm)/'confirmations'/(vid+'.json')
                confirmation = json.loads(cpath.read_text())
                confirmation['replication']['receipt_sha256'] = receipt['receipt_sha256']
                write(cpath, xg._seal(confirmation, 'confirmation_sha256'))
    bank_statistics._CACHE.clear()


@pytest.fixture
def observable_bank(monkeypatch):
    """Only observation attachment is fake; admission/evidence/tapes are native."""
    from . import effective_imitation as effective, bank_v2v, bank_statistics
    bank_statistics._CACHE.clear()
    def limits(pairs, tape, trace, entry, cap):
        assert cap == 1.2
        return pairs, dict(pairs=len(pairs), kept=len(pairs), cap_mps2=cap,
            excluded_missing_limit=0, excluded_command_mismatch=0), {}
    def attach(pairs, trace, t0, cfg=None, intent=None):
        for i, (sample, _) in enumerate(pairs):
            sample['intent_link'] = dict(t=t0+i, entry='cpu', slots={})
        return pairs, dict(pairs=len(pairs), kept=len(pairs),
            excluded_missing_prev_second=0, excluded_missing_partner_command=0, trace_missing=False)
    monkeypatch.setattr(effective, 'attach_limits', limits)
    monkeypatch.setattr(bank_v2v, 'attach_pairs', attach)


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
            bind_fixture_snapshots(root, arm)
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


def augmented_breadth_bank(tmp_path, arm, location):
    from .test_practice_starts_cpu import bank_with_snapshots
    from .test_practice_weighting_cpu import augment
    from .protocol import digest
    root = tmp_path if location == 'current' else tmp_path/'library_variety'
    for own_arm in ('B', 'R'):
        bank_with_snapshots(root, own_arm, vids=('v0',))
    _, manifest = augment(root, arm)
    for own_arm in ('B', 'R'):
        bind_fixture_snapshots(root, own_arm)
    bank = rb.bank_root(root, arm)
    receipt = json.loads((bank/'sumo_verification'/'verifications'/'aug1.json').read_text())
    manifest['rows'][0]['sumo_evidence'].update(receipt_sha256=receipt['receipt_sha256'],
        plan_sha256=digest(receipt['chosen_plan']))
    write(bank/'PRACTICE_AUGMENTED.json', xg._seal(manifest, 'practice_augmented_sha256'))
    config = dict(run_root=str(tmp_path), training_seed=4,
        augmentation_prelaunch=manifest['prelaunch'])
    if location != 'current':
        write(tmp_path/'PLAN_FIX_READY.json', dict(schema=breadth.SCHEMA,
            outputs=[dict(run_root='library_variety')]))
    return root, config


def corrupt_augmentation(root, arm, failure):
    bank = rb.bank_root(root, arm)
    path = root/'CHARGES.jsonl'
    charges = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == 'absent_ledger':
        path.unlink()
    elif failure == 'empty_ledger':
        path.write_text('')
    elif failure in ('wrong_phase', 'wrong_namespace', 'undercharged', 'over_budget'):
        if failure == 'wrong_phase':
            charges[0]['phase'] = 'other_phase'
        elif failure == 'wrong_namespace':
            charges[0]['id'] = 'unrelated_branch'
        elif failure == 'undercharged':
            charges[0]['sim_s'] = 1
        else:
            charges.append(dict(id='other_work', phase=charges[0]['phase'], sim_s=200000))
        path.write_text(''.join(json.dumps(c)+'\n' for c in charges))
    elif failure == 'changed_evidence':
        path = bank/'sumo_verification'/'verifications'/'aug1.json'
        receipt = json.loads(path.read_text())
        receipt['snapshot_sha256'] = '0'*64
        write(path, xg._seal(receipt, 'receipt_sha256'))
    elif failure == 'unregistered':
        path = bank/'PRACTICE_AUGMENTED.json'
        manifest = json.loads(path.read_text())
        manifest['prelaunch']['allocation_sim_s'] -= 1
        write(path, xg._seal(manifest, 'practice_augmented_sha256'))
    else:
        raise AssertionError(failure)


PLAN_ARGS = dict(fraction=.5, repeat_cap=2, count=20, training_seed=4, weighting=True)
AUGMENT_FAILURES = ['absent_ledger', 'empty_ledger', 'wrong_phase', 'wrong_namespace',
                    'undercharged', 'over_budget', 'changed_evidence', 'unregistered']


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('location', ['current', 'plan_fix'])
@pytest.mark.parametrize('failure', AUGMENT_FAILURES)
def test_production_weighting_plus_breadth_rejects_invalid_children_before_selection(
        tmp_path, observable_bank, arm, location, failure):
    root, config = augmented_breadth_bank(tmp_path, arm, location)
    corrupt_augmentation(root, arm, failure)
    with pytest.raises((ValueError, FileNotFoundError)):
        ps.plan_for_update(tmp_path, arm, 1, breadth_cfg=config, **PLAN_ARGS)
    assert not ps.plan_path(tmp_path, arm, 1).exists()
    # Trigger teaching shares admission, including PLAN_FIX sources.
    with pytest.raises((ValueError, FileNotFoundError)):
        breadth.visit(config, arm, lambda **row: None)


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('location', ['current', 'plan_fix'])
@pytest.mark.parametrize('failure', AUGMENT_FAILURES[:-1])
def test_frozen_breadth_preserves_child_provenance_and_revalidates_before_dispatch(
        tmp_path, observable_bank, arm, location, failure):
    root, config = augmented_breadth_bank(tmp_path, arm, location)
    plan = ps.plan_for_update(tmp_path, arm, 1, breadth_cfg=config, **PLAN_ARGS)
    child = next(r for r in plan['rows'] if r['example_id'] == 'aug1#s1')
    assert child['source'] == 'augmented' and child['parent_example_id'] == 'v0#s1'
    assert child['prelaunch'] == config['augmentation_prelaunch']
    assert child['augmentation_sha256'] and child['sumo_evidence']['charge_ids']
    assert child['snapshot_lookup']['run_root'] == str(root.resolve())
    assert ps.load_snapshot(tmp_path, arm, child)['episode']['world']['seed'] == 986020
    frozen = ps.plan_path(tmp_path, arm, 1).read_bytes()
    corrupt_augmentation(root, arm, failure)
    resumed = ps.plan_for_update(tmp_path, arm, 1, breadth_cfg=config, **PLAN_ARGS)
    assert resumed == plan and ps.plan_path(tmp_path, arm, 1).read_bytes() == frozen
    with pytest.raises((ValueError, FileNotFoundError)):
        ps.wave_call(tmp_path, arm, 'training', 'train', child, 0, {})


def import_ls_context(root, arm, vid='v0'):
    import gzip
    import io
    import shutil
    bank = rb.bank_root(root, arm)
    external = root/'ls_labels'/arm
    path = bank/'sumo_verification'/'verifications'/(vid+'.json')
    receipt = json.loads(path.read_text()); path.unlink()
    shutil.copytree(bank/'sumo_verification'/'branches'/vid, external/'branches'/vid)
    write(external/'labels'/'search.json', dict(search_id='search', verification=receipt))
    link = ps._links(root, arm)[vid]
    (bank/'decisions'/(link['state_id']+'.json')).unlink()
    snap = torch.load(bank/'snapshots'/(link['state_id']+'.pt'), weights_only=False)
    episode = root/'episodes'/arm
    write(episode/'RESULT.json', dict(id=snap['episode']['id'], world=snap['episode']['world'],
        prefixes=snap['prefixes'], steps=[dict(time_s=int(t), sample=s) for t, s in snap['samples'].items()]))
    snap['current_sample'] = snap['samples'][str(snap['t0'])]
    payload = io.BytesIO(); torch.save(snap, payload)
    snapshot_path = external/'snapshots'/'state.pt.gz'
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(gzip.compress(payload.getvalue()))
    (external/'pending').mkdir()
    (external/'pending'/'search.json.gz').write_bytes(gzip.compress(json.dumps(
        dict(snapshot='state', episode_path=str(episode))).encode()))
    return external, snapshot_path, episode


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('source', ['native', 'imported', 'plan_fix'])
@pytest.mark.parametrize('change', ['world', 'episode', 'time', 'substitution', 'generator'])
def test_original_snapshot_binding_rejects_preselection_and_resume_changes(
        tmp_path, observable_bank, arm, source, change):
    import gzip
    import io
    from .test_practice_starts_cpu import bank_with_snapshots, snapshot
    root = tmp_path/'library' if source == 'plan_fix' else tmp_path
    bank_with_snapshots(root, arm, vids=('v0',))
    bind_fixture_snapshots(root, arm)
    config = dict(run_root=str(tmp_path), training_seed=4)
    episode = None
    if source == 'imported':
        external, snapshot_path, episode = import_ls_context(root, arm)
        config['replay_sources'] = {arm: [str(external)]}
    else:
        snapshot_path = rb.bank_root(root, arm)/'snapshots'/f'state_{arm}_v0.pt'
    if source == 'plan_fix':
        write(tmp_path/'PLAN_FIX_READY.json', dict(schema=breadth.SCHEMA,
            outputs=[dict(run_root='library')]))
    pool = breadth.practice_pool(config, arm)
    assert {r['seed_index'] for r in pool} == {1, 2}
    assert all(ps.load_snapshot(tmp_path, arm, r)['t0'] == 400 for r in pool)
    if source == 'imported':
        snap = torch.load(io.BytesIO(gzip.decompress(snapshot_path.read_bytes())), weights_only=False)
    else:
        snap = torch.load(snapshot_path, weights_only=False)
    if change == 'world':
        snap['episode']['world']['seed'] += 1  # still an ordinary TRAIN seed
    elif change == 'episode':
        snap['episode']['id'] = 'other_training_episode'
    elif change == 'time':
        snap['t0'] -= 1
    elif change == 'substitution':
        snap = snapshot(world_seed=986021, gen_seed=37)
        snap['episode']['arm'] = arm
        if source == 'imported':
            snap['current_sample'] = snap['samples'][str(snap['t0'])]
    else:
        snap['generator_state'] = torch.Generator().manual_seed(99).get_state()
    if source == 'imported':
        payload = io.BytesIO(); torch.save(snap, payload)
        snapshot_path.write_bytes(gzip.compress(payload.getvalue()))
        record = json.loads((episode/'RESULT.json').read_text())
        record.update(id=snap['episode']['id'], world=snap['episode']['world'])
        write(episode/'RESULT.json', record)
    else:
        torch.save(snap, snapshot_path)
    with pytest.raises(ValueError, match='verification snapshot'):
        breadth.practice_pool(config, arm)
    for frozen_row in pool:
        with pytest.raises(ValueError, match='verification snapshot'):
            ps.load_snapshot(tmp_path, arm, frozen_row)


@pytest.mark.parametrize('source', ['native', 'imported', 'plan_fix'])
def test_unbound_old_contexts_are_excluded_with_an_explicit_report(tmp_path, observable_bank, source):
    from .test_practice_starts_cpu import bank_with_snapshots
    root = tmp_path/'library' if source == 'plan_fix' else tmp_path
    bank_with_snapshots(root, 'B', vids=('v0',))
    bind_fixture_snapshots(root, 'B')
    path = rb.bank_root(root, 'B')/'sumo_verification'/'verifications'/'v0.json'
    receipt = json.loads(path.read_text()); receipt.pop('snapshot_sha256')
    write(path, xg._seal(receipt, 'receipt_sha256'))
    config = dict(run_root=str(tmp_path), training_seed=4)
    if source == 'imported':
        external, _, _ = import_ls_context(root, 'B')
        config['replay_sources'] = {'B': [str(external)]}
    if source == 'plan_fix':
        write(tmp_path/'PLAN_FIX_READY.json', dict(schema=breadth.SCHEMA,
            outputs=[dict(run_root='library')]))
    with pytest.warns(RuntimeWarning, match='F3 excludes B/v0: original verification has no immutable'):
        assert breadth.practice_pool(config, 'B') == []


def all8_breadth_bank(root, arm):
    from .test_practice_starts_cpu import bank_with_snapshots
    from . import verify_perseed as vp, sumo_verify as sv
    import shutil
    bank_with_snapshots(root, arm, vids=('v0',))
    bind_fixture_snapshots(root, arm)
    bank = rb.bank_root(root, arm)
    with p1._bank_view(dict(run_root=str(root), training_seed=4)):
        record = deepcopy(rb.confirmations(root, arm)['v0'])
    first = json.loads((bank/'sumo_verification'/'verifications'/'v0.json').read_text())
    second = json.loads((bank/'confirm_replication'/'verifications'/'v0.json').read_text())
    base = torch.load(bank/'snapshots'/f'state_{arm}_v0.pt', weights_only=False)
    extra_root = bank/'confirm_seed3_all8'
    shutil.copytree(bank/'confirm_replication', extra_root)
    salt = vp.extra_seed_salt('v0', 3)
    third = xg._seal(dict(second, snapshot_sha256=sv.snapshot_sha(rb.replicate(base, salt))), 'receipt_sha256')
    write(extra_root/'verifications'/'v0.json', third)
    record.update(verify_mode=vp.ALL8, arm=arm,
        first_seed=dict(verification_id='v0', receipt_sha256=first['receipt_sha256']),
        seed2=dict(verification_id='v0', receipt_sha256=second['receipt_sha256'],
                   snapshot_sha256=second['snapshot_sha256'], salt='v0'),
        extra={'3': dict(verification_id='v0', receipt_sha256=third['receipt_sha256'],
                         snapshot_sha256=third['snapshot_sha256'], salt=salt)})
    ex = deepcopy(record['per_seed_v1']['examples'][1])
    ex.update(seed_index=3, example_id='v0#s3')
    ex['tape']['branches_dir'] = str(extra_root/'branches')
    record['per_seed_v1']['examples'].append(ex)
    write(bank/vp.RECORDS/'v0.json', xg._seal(record, 'confirmation_sha256'))
    return dict(run_root=str(root), training_seed=4), record


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_native_all8_seed_evidence_and_extra_seed_salt_restore_exactly(tmp_path, observable_bank, arm):
    from . import verify_perseed as vp, sumo_verify as sv
    config, _ = all8_breadth_bank(tmp_path, arm)
    pool = breadth.practice_pool(config, arm)
    assert {r['seed_index'] for r in pool} == {1, 2, 3}
    row = next(r for r in pool if r['seed_index'] == 3)
    base = torch.load(rb.bank_root(tmp_path, arm)/'snapshots'/f'state_{arm}_v0.pt', weights_only=False)
    expected = rb.replicate(base, vp.extra_seed_salt('v0', 3))
    actual = ps.load_snapshot(tmp_path, arm, row)
    assert row['replication_salt'] == 'v0#seed3'
    assert sv.snapshot_sha(actual) == sv.snapshot_sha(expected) == row['snapshot_sha256']
    assert torch.equal(actual['generator_state'], expected['generator_state'])
    call = ps.wave_call(tmp_path, arm, 'training', 'train', row, 0, {})
    assert 'replicate(snapshot, v0#seed3)' in call['extras']['practice']['crn_seed']


@pytest.mark.parametrize('failure', ['first_receipt', 'seed_salt', 'seed_hash', 'seed_receipt_world',
                                     'seed_receipt_time', 'tape_branch'])
def test_native_all8_rejects_mismatched_original_per_seed_evidence(tmp_path, observable_bank, failure):
    from . import verify_perseed as vp
    config, record = all8_breadth_bank(tmp_path, 'B')
    frozen = breadth.practice_pool(config, 'B')
    bank = rb.bank_root(tmp_path, 'B')
    if failure == 'first_receipt':
        record['first_seed']['receipt_sha256'] = '0'*64
    elif failure == 'seed_salt':
        record['seed2']['salt'] = 'wrong'
    elif failure == 'seed_hash':
        record['seed2']['snapshot_sha256'] = '0'*64
    elif failure == 'tape_branch':
        record['per_seed_v1']['examples'][1]['tape']['branch']['result_sha256'] = '0'*64
    else:
        path = bank/'confirm_replication'/'verifications'/'v0.json'
        receipt = json.loads(path.read_text())
        if failure == 'seed_receipt_world':
            receipt['episode']['world']['seed'] += 1
        else:
            receipt['t0'] -= 1
        receipt = xg._seal(receipt, 'receipt_sha256')
        record['seed2']['receipt_sha256'] = receipt['receipt_sha256']
        write(path, receipt)
    write(bank/vp.RECORDS/'v0.json', xg._seal(record, 'confirmation_sha256'))
    with pytest.raises(ValueError):
        breadth.practice_pool(config, 'B')
    row = next(r for r in frozen if r['seed_index'] == 2)
    with pytest.raises(ValueError):
        ps.load_snapshot(tmp_path, 'B', row)


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('location', ['current', 'plan_fix'])
def test_augmented_snapshot_cannot_be_certified_by_refreezing_current_hash(
        tmp_path, observable_bank, arm, location):
    from . import sumo_verify as sv
    root, config = augmented_breadth_bank(tmp_path, arm, location)
    bank = rb.bank_root(root, arm)
    path = bank/'snapshots'/f'aug_state_{arm}.pt'
    snapshot = torch.load(path, weights_only=False)
    snapshot['episode']['world']['seed'] += 1  # ordinary TRAIN, wrong verified context
    torch.save(snapshot, path)
    # Even a freshly sealed manifest/current hash cannot change original evidence.
    manifest_path = bank/'PRACTICE_AUGMENTED.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['rows'][0]['snapshot_sha256'] = sv.snapshot_sha(snapshot)
    write(manifest_path, xg._seal(manifest, 'practice_augmented_sha256'))
    with pytest.raises(ValueError, match='SUMO receipt binding|verification snapshot'):
        ps.plan_for_update(tmp_path, arm, 1, breadth_cfg=config, **PLAN_ARGS)
    assert not ps.plan_path(tmp_path, arm, 1).exists()


def test_old_f3_frozen_lookup_requires_a_new_update(tmp_path):
    row = dict(seed_index=1, snapshot_lookup=dict(run_root=str(tmp_path),
        receipt=dict(verification_id='old', _policy_version='unknown')))
    with pytest.raises(ValueError, match='predates original verification binding'):
        ps.load_snapshot(tmp_path, 'B', row)


def test_augmented_native_link_cannot_omit_prelaunch_manifest(tmp_path, observable_bank):
    root, config = augmented_breadth_bank(tmp_path, 'B', 'current')
    bank = rb.bank_root(root, 'B')
    path = bank/'decisions'/'aug_state_B.json'
    link = json.loads(path.read_text())
    link['augmentation'] = dict(parent_example_id='v0#s1')
    write(path, link)
    (bank/'PRACTICE_AUGMENTED.json').unlink()
    with pytest.raises(ValueError, match='missing verified prelaunch manifest'):
        ps.plan_for_update(tmp_path, 'B', 1, breadth_cfg=config, **PLAN_ARGS)


def test_resume_binds_original_own_seed_receipt_even_if_confirmation_is_replaced(tmp_path, observable_bank):
    from . import verify_perseed as vp
    config, record = all8_breadth_bank(tmp_path, 'B')
    pool = breadth.practice_pool(config, 'B')
    frozen = next(r for r in pool if r['seed_index'] == 2)
    bank = rb.bank_root(tmp_path, 'B')
    path = bank/'confirm_replication'/'verifications'/'v0.json'
    receipt = json.loads(path.read_text())
    old_hash = receipt['receipt_sha256']
    receipt['timing']['finished_unix'] += 1
    receipt = xg._seal(receipt, 'receipt_sha256')
    write(path, receipt)
    record['seed2']['receipt_sha256'] = receipt['receipt_sha256']
    write(bank/vp.RECORDS/'v0.json', xg._seal(record, 'confirmation_sha256'))
    # This is a self-consistent *new* bank. It cannot replace frozen evidence.
    updated = next(r for r in breadth.practice_pool(config, 'B') if r['seed_index'] == 2)
    assert updated['snapshot_sha256'] == frozen['snapshot_sha256']
    assert frozen['snapshot_lookup']['verification_binding']['own_receipt_sha256'] == old_hash
    assert updated['snapshot_lookup']['verification_binding']['own_receipt_sha256'] != old_hash
    with pytest.raises(ValueError, match='immutable verification evidence changed'):
        ps.load_snapshot(tmp_path, 'B', frozen)
