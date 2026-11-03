"""CPU regression for FIX 2, the statistical bank filter (option ``bank_filter='statistical'``,
DEFAULT OFF = byte-identical old behaviour).

Synthetic entries cover the registered verdict rules:
- zero-gain replicated advantage -> dropped;
- noisy positive (positive aggregates, mixed-sign paired per-vehicle differences) -> dropped;
- clear positive -> kept;
- ~zero synergy (recorded gain > 0 but the group-minus-best-solo bound and the
  group-minus-sum-of-solos synergy fail) -> NOT cooperative;
- clear synergy -> cooperative; the 50% oversampling then applies to the filtered set.
No SUMO, no GPU: sv.verify runs on the in-process fake plant below.
"""
import gzip
import json
from pathlib import Path

import pytest

from tools.coopmwm import bank_filter as bf
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.protocol import digest
from tools.coopmwm.storage import ledger
from tools.coopnative.storage import canonical


@pytest.fixture(autouse=True)
def bank_budget(tmp_path):
    from .warm_protocol import budget
    b = budget()
    b['total'] += 2600000
    b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (tmp_path/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (tmp_path/'AUTHORITY.md').write_text('test-bank-budget')


def fp(t, delay):
    hexf = lambda x: float(x).hex()
    return dict(t_s=t, metrics=dict(team_delay_vs=hexf(delay), arrived=hexf(t), collisions=hexf(0.),
                                    teleports=hexf(0.), ttc_below_3_vs=hexf(0.), hard_brake=hexf(0.)))


def snapshot(t0=420):
    return dict(episode=dict(id='warm_u005_B_7', world=dict(seed=7), arm='B', update=5, role='train'),
                policy=dict(state_dict_sha256='p'), sumo=dict(runtime='rt', checkpoint='ck', warm_history=True,
                                                             accel_bounds=[-3, 2], noise_seed=11),
                t0=t0, end_s=1200, prefixes={str(t): fp(t, 3.*7+0.5*t*t/100.) for t in range(300, t0+1)},
                samples={}, rng=dict(generator_state_sha256='g'), committed_plans=[])


def plan(pid, gain, members=('c1', 'c2')):
    return dict(id=pid, members=list(members), duration_s=120,
                profile=[[gain]*120 for _ in members],
                saving_vs_mappo=gain, saving_vs_policy=gain)


def duo_plan(pid, g1, g2):
    return dict(id=pid, members=['c1', 'c2'], duration_s=120, profile=[[g1]*120, [g2]*120],
                saving_vs_mappo=(g1+g2)/2, saving_vs_policy=(g1+g2)/2)


SLOW = [0.]*4                  # every vehicle at a standstill: full deficit 1 veh-s per tick
FAST = [31.29]*4               # every vehicle at the registered limit: zero deficit


class StatRunner:
    """Fake plant: J(plan) = J_policy - sum(profile) - SYN*max(0, len(members)-1); every branch
    writes ``n_s`` observable trace seconds whose per-vehicle speeds are looked up by the
    executed plan id ('policy' for the policy-continues branch), so the paired per-vehicle
    deficit differences (the filter's input) are exactly controlled per branch. Same-lane
    vehicles always share one speed per branch (no closing pair, no TTC events);
    fingerprints carry zero collisions/teleports/hard-brake in every branch (safety passes)."""

    def __init__(self, work, syn=0., speeds=None, n_s=3, J_policy=1000.):
        from tools.coopmwm.accel_storage import write as awrite
        self.work = Path(work);self.awrite = awrite
        self.syn, self.J_policy, self.n_s = syn, J_policy, n_s
        self.speeds = dict(speeds or {})          # plan id or 'policy' -> [speeds per vehicle]

    def J(self, p):
        if p is None:
            return self.J_policy
        base = float(sum(x for row in p['profile'] for x in row))
        return self.J_policy-base-self.syn*max(0, len(p['members'])-1)

    def speeds_for(self, p):
        default = self.speeds.get('policy', SLOW)
        return self.speeds.get('policy' if p is None else p['id'], default)

    def run(self, spec, snap):
        t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        d = self.work/'branches'/spec['verification_id']/spec['branch']
        d.mkdir(parents=True, exist_ok=True)
        sp = self.speeds_for(spec['plan'])
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            for k in range(self.n_s):
                f.write(canonical(dict(time_s=t0+k, vehicles=[
                    dict(vid=f'v{i}', lane=('m1' if i % 2 == 0 else 'm2'), s_m=10.*i,
                         v_mps=float(sp[i % len(sp)]), length_m=5., is_cav=True)
                    for i in range(4)]))+'\n')
        w = snap['episode']['world']['seed']
        r = dict(prefixes={str(t): fp(t, 3.*w+0.5*t*t/100.) for t in range(300, t0+1)},
                 final=fp(t_end, 3.*w+0.5*t0*t0/100.+self.J(spec['plan'])), actual_sim_s=t_end-300)
        p = spec['plan']
        if p is not None:
            tape = dict(steps=[dict(time_s=t, sample=dict(plan_executions=[
                dict(plan_id=p['id'], proposal_members=list(p['members']))],
                absolute_accels={v: 0.5 for v in p['members']}))
                for t in range(t0, t0+p['duration_s'])])
            self.awrite(d/'RESULT.json', tape)
            r['result_sha256'] = digest(tape)
        return r


def _confirm(root, bank, t0, runner_factory, make_plan):
    first = sv.verify(snapshot(t0), [make_plan()], runner_factory(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    assert first['status'] == 'accepted', first.get('reason')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(t0), runner_factory, str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['admitted'] is True, out
    return out


def _bank(tmp_path):
    """Five admitted entries: clear positive (cooperative), zero-gain, noisy positive,
    ~zero-synergy (recorded cooperative, C_G = 0), and a single-CAV plan."""
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _confirm(root, bank, 420, lambda work: StatRunner(work, syn=50., speeds=dict(
        policy=SLOW, clear=FAST, **{f'clear__solo_{m}': SLOW for m in ('c1', 'c2')})),
        lambda: duo_plan('clear', 1., .5))
    # zero-gain: tiny positive aggregates on both replicates, zero paired per-vehicle signal
    _confirm(root, bank, 460, lambda work: StatRunner(work, syn=0., speeds=dict(
        policy=SLOW, zero=SLOW, **{f'zero__solo_{m}': SLOW for m in ('c1', 'c2')})),
        lambda: duo_plan('zero', 1e-9, 1e-9))
    # noisy: aggregates are large and positive, but half the vehicles are saved while the
    # other half get SLOWER than their already-moving policy state (mixed-sign paired
    # differences; same-lane speeds stay equal per branch, so no TTC event ever fires)
    _confirm(root, bank, 500, lambda work: StatRunner(work, syn=0., speeds=dict(
        policy=[20.]*4, noisy=[31.29, 0., 31.29, 0.], **{
            f'noisy__solo_{m}': [31.29, 0., 31.29, 0.] for m in ('c1', 'c2')})),
        lambda: duo_plan('noisy', 1., .5))
    # ~zero synergy: linear plant -> recorded gain > 0, C_G = 0, clear group advantage
    _confirm(root, bank, 540, lambda work: StatRunner(work, syn=0., speeds=dict(
        policy=SLOW, lin=FAST, **{f'lin__solo_{m}': SLOW for m in ('c1', 'c2')})),
        lambda: duo_plan('lin', 1., .5))
    # single-CAV plan: cooperative=False by definition, clear advantage -> kept
    _confirm(root, bank, 580, lambda work: StatRunner(work, syn=0., speeds=dict(
        policy=SLOW, solo1=FAST)), lambda: plan('solo1', 1., members=('c1',)))
    return root


# ---------------------------------------------------------------- pure statistics

def test_one_sided_lb95_basic_and_degenerate():
    assert bf.one_sided_lb95([1., 1., 1.]) == 1.
    assert bf.one_sided_lb95([3., 3.]) == 3.
    assert bf.one_sided_lb95([1., 1., -1.]) < 0        # noisy sample fails the bound
    assert bf.one_sided_lb95([5.]) is None             # a single value is never significant
    assert bf.one_sided_lb95([]) is None


def test_paired_differences_pairs_vehicles_and_skips_missing_traces(tmp_path):
    def trace(name, speeds):
        p = tmp_path/name
        with gzip.open(p, 'wt', encoding='utf-8') as f:
            for k, v in enumerate(speeds):
                f.write(canonical(dict(time_s=400+k, vehicles=[
                    dict(vid=f'v{i}', lane='m1', v_mps=v[i], s_m=1.) for i in range(len(v))]))+'\n')
        return p
    limit = bf.speed_limit_mps()
    a = trace('a.gz', [[0., 0.], [0., 0.]])            # policy: full deficit both seconds
    b = trace('b.gz', [[limit, 0.], [limit, limit]])   # plan: saved v0 (both s), v1 (second s)
    d = bf.paired_differences(a, b, 400, 402, limit)
    assert d == dict(v0=2., v1=1.)
    assert bf.paired_differences(tmp_path/'missing.gz', b, 400, 402, limit) is None


def test_advantage_verdict_rules():
    clear = [dict(name='first', aggregate_veh_s=4430., paired={f'v{i}': 10. for i in range(4)}),
             dict(name='replication', aggregate_veh_s=4000., paired={f'v{i}': 9. for i in range(4)})]
    v = bf.advantage_verdict(clear)
    assert v['keep'] is True and v['bound'] > 0
    zero = [dict(name='first', aggregate_veh_s=0.0005, paired={f'v{i}': 0. for i in range(4)}),
            dict(name='replication', aggregate_veh_s=0.0004, paired={f'v{i}': 0. for i in range(4)})]
    v = bf.advantage_verdict(zero)
    assert v['keep'] is False and v['bound'] == 0.
    noisy = [dict(name='first', aggregate_veh_s=40., paired=dict(v0=1., v1=-1., v2=1., v3=-1.)),
             dict(name='replication', aggregate_veh_s=35., paired=dict(v0=1., v1=-1., v2=1., v3=-1.))]
    assert bf.advantage_verdict(noisy)['keep'] is False
    nonpositive = [dict(name='first', aggregate_veh_s=10., paired=dict(v0=5., v1=5.)),
                   dict(name='replication', aggregate_veh_s=-2., paired=dict(v0=5., v1=5.))]
    v = bf.advantage_verdict(nonpositive)
    assert v['keep'] is False and 'not > 0' in v['reason']
    # aggregate fallback: no vehicle data anywhere -> bound over the >= 2 replicate aggregates
    agg = [dict(name='first', aggregate_veh_s=4430., paired=None),
           dict(name='replication', aggregate_veh_s=959., paired=None)]
    assert bf.advantage_verdict(agg)['keep'] is True
    single = [dict(name='first', aggregate_veh_s=10., paired=None)]
    assert bf.advantage_verdict(single)['keep'] is False


def test_cooperation_verdict_rules():
    clear = dict(v0=5., v1=5.)
    v = bf.cooperation_verdict(clear, 60., True, 30.)
    assert v['cooperative'] is True and v['bound'] > 0
    # ~zero synergy: recorded gain > 0 but the bound is 0 and the sum-of-solos synergy is 0
    v = bf.cooperation_verdict(dict(v0=0., v1=0.), 60., True, 0.)
    assert v['cooperative'] is False
    # noisy synergy: bound <= 0 beats the recorded gain
    v = bf.cooperation_verdict(dict(v0=-1., v1=0., v2=1.), 60., True, 30.)
    assert v['cooperative'] is False
    # sum-of-solos synergy available and non-positive blocks the label even with a clear bound
    v = bf.cooperation_verdict(clear, 60., True, 0.)
    assert v['cooperative'] is False and 'sum-of-solos' in v['reason']
    v = bf.cooperation_verdict(clear, 60., True, None)      # incomplete solos: not available
    assert v['cooperative'] is True
    # no usable trace pair: documented fallback to the recorded aggregate rule
    v = bf.cooperation_verdict(None, 60., True, 0.)
    assert v['cooperative'] is False and 'aggregate' in v['reason']
    v = bf.cooperation_verdict(None, 60., False, None)
    assert v['cooperative'] is True


# ---------------------------------------------------------------- end-to-end on a synthetic bank

def test_filter_drops_zero_and_noisy_keeps_clear_and_relables_synergy(tmp_path):
    root = _bank(tmp_path)
    pairs_off, s_off = rb.bank_pairs(root, 'B', 12345)
    assert len(s_off['entries']) == 5 and s_off['counts']['cooperative'] == 4
    assert s_off['exposure']['cooperative_entries'] == 4      # old label: gain > 0 for all four duos
    assert 'bank_filter' not in s_off and 'filter_dropped' not in s_off['counts']
    pairs_none, s_none = rb.bank_pairs(root, 'B', 12345, bank_filter=None)
    assert pairs_none == pairs_off and s_none == s_off        # explicit off == old behaviour
    with pytest.raises(ValueError):
        rb.bank_pairs(root, 'B', 12345, bank_filter='thresholds')

    pairs, s = rb.bank_pairs(root, 'B', 12345, bank_filter='statistical')
    f = s['bank_filter']
    assert f['mode'] == 'statistical' and f['rule'] == bf.RULE
    assert f['counts'] == dict(kept=3, dropped=2)
    by_id = {row['verification_id']: row for row in f['entries']}
    dropped = [row for row in f['entries'] if row['decision'] == 'dropped']
    assert len(dropped) == 2 and all('lower bound <= 0' in row['reason'] for row in dropped)
    assert all(row['n_paired'] == 8 for row in dropped)        # 4 paired vehicles x 2 replicate seeds
    kept = [row for row in f['entries'] if row['decision'] == 'kept']
    assert len(kept) == 3 and all(row['bound'] > 0 for row in kept)
    assert s['counts']['used_admitted'] == 3 and s['counts']['filter_dropped'] == 2
    assert len(s['entries']) == 3 and len(pairs) == 480
    # the ~zero-synergy entry is among the kept for imitation but is NOT cooperative any more
    lin = next(e for e in s['entries'] if e['group_size'] == 2 and not e['cooperative'])
    row = by_id[lin['verification_id']]
    assert row['keep'] is True and row['cooperation']['cooperative'] is False
    assert row['cooperation']['recorded_cooperative'] is True       # the old label said True
    assert row['cooperation']['sum_synergy_veh_s'] == 0.            # linear plant: C_G = 0
    # the clear-synergy entry stays cooperative (bound > 0, C_G = 50 > 0) and the 50%
    # oversampling applies to the filtered cooperative set
    coop = [e for e in s['entries'] if e['cooperative']]
    assert len(coop) == 1 and coop[0]['cooperation_gain_veh_s'] == 110.
    assert s['exposure']['cooperative_entries'] == 1 and s['exposure']['non_cooperative_entries'] == 2
    assert s['exposure']['duplicated_pairs'] == 120
    assert s['exposure']['cooperative_share_of_samples'] == 0.5
    assert all(w == 1. for _, w in pairs)                           # weights unchanged


def test_statistical_label_uses_paired_traces_not_only_gain(tmp_path):
    """A ~zero group-vs-best-solo trace must unlabel the entry even when the recorded
    aggregate gain is positive; the clear-synergy plant keeps it."""
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _confirm(root, bank, 420, lambda work: StatRunner(work, syn=50., speeds=dict(
        policy=SLOW, syn=FAST, syn__solo_c1=SLOW, syn__solo_c2=[0., 0., 0., 20.])),
        lambda: duo_plan('syn', 1., .5))
    pairs, s = rb.bank_pairs(root, 'B', 12345, bank_filter='statistical')
    row = s['bank_filter']['entries'][0]
    assert row['keep'] is True and row['cooperation']['cooperative'] is True
    assert row['cooperation']['bound'] > 0 and row['cooperation']['sum_synergy_veh_s'] == 50.
    root2 = tmp_path/'quiet';root2.mkdir();bank2 = rb.bank_root(root2, 'B');bank2.mkdir(parents=True)
    from .warm_protocol import budget
    b = budget();b['total'] += 2600000;b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (root2/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (root2/'AUTHORITY.md').write_text('test-bank-budget')
    # same aggregate plant, but the best-solo branch nearly matches the group trace (only v2
    # differs): the paired bound is 0 and the statistical label flips despite gain = +110
    _confirm(root2, bank2, 420, lambda work: StatRunner(work, syn=50., speeds=dict(
        policy=SLOW, syn2=FAST, syn2__solo_c1=[31.29, 31.29, 0., 31.29],
        syn2__solo_c2=SLOW)), lambda: duo_plan('syn2', 1., .5))
    pairs2, s2 = rb.bank_pairs(root2, 'B', 12345, bank_filter='statistical')
    row2 = s2['bank_filter']['entries'][0]
    assert row2['keep'] is True and row2['cooperation']['recorded_cooperative'] is True
    assert row2['cooperation']['cooperative'] is False and row2['cooperation']['bound'] == 0.
    assert [e['cooperative'] for e in s2['entries']] == [False]
    assert s2['exposure']['cooperative_entries'] == 0 and s2['exposure']['duplicated_pairs'] == 0


def test_unreadable_traces_degrade_to_documented_aggregate_rules(tmp_path):
    """No traces at all: admission falls back to the replicate-aggregate bootstrap (both
    replicates > 0) and the cooperative label to the recorded aggregate gain rule - exactly
    the documented degradation, never an imputed vehicle value."""
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _confirm(root, bank, 420, lambda work: StatRunner(work, syn=50.), lambda: duo_plan('d', 1., .5))
    for sub in ('sumo_verification', 'confirm_replication', 'confirm_solo'):
        for p in (bank/sub/'branches').rglob('observable.jsonl.gz'):
            p.unlink()
    pairs, s = rb.bank_pairs(root, 'B', 12345, bank_filter='statistical')
    row = s['bank_filter']['entries'][0]
    assert row['decision'] == 'kept' and row['first_seed_traces_readable'] is False
    assert 'replicate-aggregate' in row['reason'] and row['bound'] > 0
    assert row['cooperation']['cooperative'] is True and row['cooperation']['bound'] is None
    assert 'aggregate' in row['cooperation']['reason']
    assert len(pairs) == 120 and s['exposure']['cooperative_entries'] == 1
    # charges untouched by the filter (read-only): only the confirmation branches ran
    assert all(e['phase'] == 'bank_verification_ext' for e in ledger(root) if e['id'].startswith('confirm_'))
