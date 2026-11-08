"""DEMOX: verified-seed demonstration extractor for c4/learnfix demo-BC (MAIN 2026-11-08; scope update 23:0x:
WM and rule plans, every row tagged source wm|rule; only ADMITTED (plan, bin) rows).

Pipeline (CPU, zero SUMO, read-only on every input):
  1. ``plan_activations``: privileged onset rows (ops/sit_reduce PONSET files, mode 'privileged',
     fullpol_situational.onset_bin('privileged')) whose (family, bin) is admitted in the guidance doc, joined to the
     per-seed paired effect from the WMVERIFY BANK files (diff_vs_incumbent = J(plan) - J(incumbent), veh-s;
     negative = plan better). Rule seeds are allowed only when the doc admits them (MAIN scope update).
  2. ``work_list``: the tapes needed. Helped seeds (diff < 0): the plan's own tape. Hurt seeds (diff > 0): the same
     seed's incumbent (wall_g_solo2) and u0 tapes at the same onset second (never the plan's own arm).
  3. ops/demo_extract_host.py trims those tapes on the hosts (stdlib), copied here.
  4. ``featurize``: each tape is replayed second by second (300 .. last wanted second) through the EXACT training
     featurizer: track2_backend.episode_components over the v2v Controller (controller_class(Controller)), same
     calls as accel_plant (controller.apply(rows, commands, t), then observe_issued(the tape line)); the trigger
     is forced to 'none' so the replaying policy never starts a session of its own.
  5. ``build_items``: at the onset second, every acting member's row that offers the plan's template
     (demo_bc.plan_slot semantics, per member) gives one item:
       act  (helped seed): label = that slot; weight = (inc - pick) / max over helped seeds / offered members
       none (hurt seed):   label = none;      weight = (pick - inc) / max over hurt seeds  / offered rows / 2 tapes
     Weights are in (0, 1]. Rows whose slate does not offer the plan, or whose tick has no trigger decision, are
     skipped and counted.
  6. Outputs: DEMO.json (demo_bc.SCHEMA / RULE, LF bytes; demo_bc.load works unchanged), DEMO.npz (flat arrays +
     the trigger records as JSON strings) and DEMO.manifest.json (sha256 of both, counts, seeds, plans, cells,
     inputs and their sha256).
Seeds: every seed must lie in the verification ranges 989950-989999 or 991000-991999 and be disjoint from the
held-out panel (map_scenarios.HELDOUT_RANGE, 9896xx) - asserted twice (activations and items).
"""
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path

SCHEMA_NPZ = 'coopmwm.demo_extract.v1'
VERIFY_RANGES = ((989950, 989999), (991000, 991999))
INCUMBENT = 'wall_g_solo2'
REFERENCE_ARMS = (INCUMBENT, 'u0')
BANNED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'driver_params',
          'vtype')
TAPE_KEYS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')
LOOKBACK_S = 60  # = trigger_head.decision_interval_s of the CLEAN5C config


# ----------------------------------------------------------------------------------------------- guards
def guard_path(path):
    low = str(path).lower().replace('\\', '/')
    for tok in BANNED:
        if tok in low:
            raise ValueError(f'refusing banned path token {tok!r}: {path}')
    return path


def assert_verification_seeds(seeds):
    """Every seed in VERIFY_RANGES and disjoint from the held-out panel."""
    from .map_scenarios import HELDOUT_RANGE
    lo, hi = HELDOUT_RANGE
    seeds = sorted({int(s) for s in seeds})
    held = [s for s in seeds if lo <= s <= hi]
    if held:
        raise ValueError(f'demo seeds overlap the held-out panel {HELDOUT_RANGE}: {held}')
    out = [s for s in seeds if not any(a <= s <= b for a, b in VERIFY_RANGES)]
    if out:
        raise ValueError(f'demo seeds outside the verification ranges {VERIFY_RANGES}: {out}')
    return True


def source_tag(family):
    return 'wm' if str(family).startswith('mut_') else 'rule'


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(guard_path(path)).read_bytes())


def bin_key(b):
    return json.dumps(b, sort_keys=True)


def root_name(root):
    return str(root).rstrip('/').split('/')[-1]


# ----------------------------------------------------------------------------------------------- inputs
def admitted_pairs(guidance):
    """{family: {bin_key}} of the admitted situational pairs (any plan source)."""
    out = {}
    for p in guidance['situational']['pairs']:
        if p.get('admitted'):
            out.setdefault(p['family'], set()).add(bin_key(p['bin']))
    return out


def plan_sources(guidance):
    """{family: 'wm'|'rule'} from the doc rows' plan_source when present, else by name."""
    out = {}
    for p in guidance['situational']['pairs']:
        if p.get('plan_source') in ('wm', 'rule'):
            out[p['family']] = p['plan_source']
    return out


def read_onsets(paths):
    rows = []
    for f in paths:
        with open(guard_path(f), encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                if r.get('mode') == 'privileged':
                    rows.append(r)
    return rows


def read_bank(paths):
    """(seed effects, members): effects[(root, cell, seed, family)] = diff_vs_incumbent (veh-s, plan - incumbent);
    members[(root, cell, seed, family, t)] = member vids of that activation."""
    effects, members = {}, {}
    for f in paths:
        with open(guard_path(f), encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                key = (root_name(r['root']), r['cell'], int(r['seed']), r['family'])
                if r.get('diff_vs_incumbent') is not None:
                    d = float(r['diff_vs_incumbent'])
                    if key in effects and abs(effects[key]-d) > 1e-6:
                        raise ValueError(f'inconsistent per-seed effect for {key}')
                    effects[key] = d
                members[key+(int(r['t']),)] = list(r.get('members') or [])
    return effects, members


def plan_activations(guidance, onsets, effects, members):
    """(activations, skipped): admitted-bin privileged onsets with their per-seed paired effect."""
    adm = admitted_pairs(guidance)
    tags = plan_sources(guidance)
    acts, skipped = [], []
    for r in sorted(onsets, key=lambda r: (r['root'], r['cell'], int(r['seed']), r['arm'], int(r['t']))):
        fam = r['arm']
        if fam not in adm or bin_key(r['bin']) not in adm[fam]:
            continue
        key = (r['root'], r['cell'], int(r['seed']), fam)
        base = dict(root=r['root'], cell=r['cell'], seed=int(r['seed']), family=fam, t=int(r['t']),
                    source=tags.get(fam, source_tag(fam)), bin=r['bin'], holder=r.get('holder'))
        if fam == INCUMBENT:
            skipped.append(dict(base, reason='incumbent has no paired effect vs itself'))
            continue
        d = effects.get(key)
        if d is None:
            skipped.append(dict(base, reason='no per-seed bank effect'))
            continue
        mem = members.get(key+(int(r['t']),)) or ([r['holder']] if r.get('holder') else [])
        acts.append(dict(base, diff=d, members=mem))
    assert_verification_seeds({a['seed'] for a in acts})
    return acts, skipped


def unconditional_activations(guidance, bank_paths, cells=None):
    """(activations, skipped) of the doc's UNCONDITIONAL rows (situational.unconditional '<cell>/<family>'):
    every executed activation of that plan in the given BANK files (any onset bin), paired effect = the seed's
    episode-level diff_vs_u0 (the unconditional confirmation is vs u0), reference tape for none rows = u0."""
    unc = {tuple(k.split('/', 1)) for k in (guidance['situational'].get('unconditional') or {})}
    tags = plan_sources(guidance)
    acts, skipped, effects = [], [], {}
    rows = []
    for f in bank_paths:
        with open(guard_path(f), encoding='utf-8') as fh:
            rows += [json.loads(line) for line in fh]
    for r in rows:
        if (r['cell'], r['family']) in unc and r.get('diff_vs_u0') is not None:
            key = (root_name(r['root']), r['cell'], int(r['seed']), r['family'])
            d = float(r['diff_vs_u0'])
            if key in effects and abs(effects[key]-d) > 1e-6:
                raise ValueError(f'inconsistent per-seed effect for {key}')
            effects[key] = d
    for r in sorted(rows, key=lambda r: (r['root'], r['cell'], int(r['seed']), r['family'], int(r['t']))):
        if (r['cell'], r['family']) not in unc or (cells and r['cell'] not in cells):
            continue
        key = (root_name(r['root']), r['cell'], int(r['seed']), r['family'])
        base = dict(root=key[0], cell=r['cell'], seed=key[2], family=r['family'], t=int(r['t']),
                    source=tags.get(r['family'], source_tag(r['family'])), bin='unconditional',
                    holder=(r.get('members') or [None])[0], admission='unconditional', ref_arms=['u0'])
        if key not in effects:
            skipped.append(dict(base, reason='no per-seed u0 effect'))
            continue
        acts.append(dict(base, diff=effects[key], members=list(r.get('members') or [])))
    assert_verification_seeds({a['seed'] for a in acts})
    return acts, skipped


def job_name(cell, seed, arm):
    return f'{cell}_s{int(seed)}_{arm}'


def window(t, lookback=None):
    """The policy decision seconds that may carry an activation at onset second t: [t - lookback, t] (>= 300).
    The trigger decides every decision_interval_s (60 s) plus event seconds, so the plant's onset second is
    usually not a decision second. The extractor makes a decision due at every second of this window
    (featurize force_decision) and the demo uses the LATEST second whose slate offers the plan to an acting
    member (normally the onset second itself; pre-onset state, the plan has not acted yet)."""
    lb = LOOKBACK_S if lookback is None else int(lookback)
    return list(range(max(300, int(t)-lb), int(t)+1))


def work_list(acts, lookback=None):
    """{tape key: {root, job, until, ticks}} needed by build_items; tape key = <root>__<job>."""
    want = {}

    def add(root, job, t):
        k = f'{root}__{job}'
        w = want.setdefault(k, dict(root=root, job=job, until=0, ticks=set()))
        w['until'] = max(w['until'], int(t))
        w['ticks'].update(window(t, lookback))
    for a in acts:
        if a['diff'] < 0:
            add(a['root'], job_name(a['cell'], a['seed'], a['family']), a['t'])
        elif a['diff'] > 0:
            for arm in a.get('ref_arms') or REFERENCE_ARMS:
                if arm != a['family']:
                    add(a['root'], job_name(a['cell'], a['seed'], arm), a['t'])
    return {k: dict(v, ticks=sorted(v['ticks'])) for k, v in sorted(want.items())}


# ----------------------------------------------------------------------------------------------- featurizer
def tape_seconds(path):
    """{second: tape line dict} of an observable tape (full or host-trimmed)."""
    out = {}
    with gzip.open(guard_path(path), 'rt', encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            out[int(round(float(d['time_s'])))] = d
    return out


def tape_inputs(line, *, strip=True):
    """(rows, commands, issued line) the plant handed the controller at that second, rebuilt from a tape line.
    rows: the observation keys; issued line: the tape line itself (what accel_plant's ObservedTrace passes to
    observe_issued; verification tapes carry the same per-vehicle fields). commands: each CAV's MAPPO raw command
    from the tape audit when audited; otherwise (strip=True, verification tapes audit only rule-acting CAVs)
    the row's recorded policy-mean / issued command, else 0. strip=False: audit only, else 0."""
    rows = [{k: r[k] for k in TAPE_KEYS if k in r} for r in line['vehicles']]
    audit = {a['vid']: a.get('mappo_raw') for a in line.get('actions') or () if 'vid' in a}
    commands = {}
    for r in line['vehicles']:
        if not r.get('is_cav'):
            continue
        c = audit.get(r['vid'])
        if c is None and strip:
            c = r.get('policy_mean_acceleration_mps2', r.get('issued_acceleration_mps2'))
        commands[r['vid']] = float(c or 0.)
    return rows, commands, line


def featurizer(cfg, base, worker_state, *, seed=0):
    """The training worker's controller (track2_backend.episode_components over the v2v Controller, the same
    construction as accel_plant), with the trigger always choosing 'none' (no session of its own starts).
    The recorded logp of a replayed record is therefore that of the forced choice (~0); values are the real
    critic values. Context/tables/descriptors/mask are untouched."""
    import torch
    from .acceleration import Controller
    from .track2_backend import episode_components
    from .v2v_runtime import controller_class
    job = dict(track2_backend=True, track2_config=cfg, arm='B', track2_state=worker_state, noise_seed=int(seed))
    policy, ctype = episode_components(job, base, controller_class(Controller))
    trig = policy.trigger
    real = trig.distribution

    def distribution(context, descriptors, mask):
        logits, values = real(context, descriptors, mask)
        forced = torch.full_like(logits, -1e9)
        forced[:, 0] = 0.
        return forced, values
    trig.distribution = distribution
    ctl = ctype(policy, torch.Generator().manual_seed(int(seed)), True, guide=None, beta=1., replay=None, branch=None)
    return policy, ctl


def featurize(cfg, base, worker_state, seconds, want, *, seed=0, strip=True, inputs=None, force_decision=()):
    """{t: track2_trigger record or None} for t in ``want``, replaying every second from 300 to max(want) exactly
    as accel_plant does: controller.apply(rows, commands, t) then observe_issued(the second's tape line).
    ``inputs`` (parity only): {t: (rows, commands)} = the recorded training controller inputs.
    ``force_decision``: seconds at which the trigger's periodic decision clock is made due (last_decision reset),
    so a trigger record (the full slate) exists there; only the decision PHASE changes (the trigger picks none,
    so no session or state follows from it). Training decides every decision_interval_s from its own phase."""
    policy, ctl = featurizer(cfg, base, worker_state, seed=seed)
    out = {}
    last = max(want) if want else 299
    for t in range(300, last+1):
        if t not in seconds:
            raise ValueError(f'tape second {t} missing')
        rows, commands, issued = tape_inputs(seconds[t], strip=strip)
        if inputs is not None:
            rows, commands = inputs[t]
        if t in force_decision:
            policy.last_decision = None
        n = len(ctl.events)
        ctl.apply(rows, commands, t)
        if len(ctl.events) != n+1:
            raise ValueError('expected one controller event per second')
        if hasattr(ctl, 'observe_issued'):
            ctl.observe_issued(issued)
        if t in want:
            out[t] = deepcopy(ctl.events[-1]['sample'].get('track2_trigger'))
    return out


def runtime(run_root, guidance=None):
    """(training config with the guidance applied, base policy, B worker state) of a CLEAN5C run root."""
    import torch
    from . import clean4_restart as c4, launch_production as production
    from .launch_training import read
    from .v2v_runtime import load_checkpoint
    cfg = read(Path(guard_path(run_root))/'CONFIG.json')
    budget = cfg.get('clean4_budget') or {}
    if budget.get('run') in c4.APPROVED_BY_RUN:
        c4.APPROVED_BY_RUN[budget['run']] = budget['approved_sim_s']
    native = production.backend_config(cfg)
    if guidance is not None:
        from .bank_growth import apply_guidance
        native = apply_guidance(native, guidance)
    base = load_checkpoint(os.path.join(run_root, 'B_checkpoint_000.pt'), config=native)[0]
    state = torch.load(os.path.join(run_root, 'B_track2_000.pt'), map_location='cpu', weights_only=False)['worker']
    return native, base, state


# ----------------------------------------------------------------------------------------------- items
def member_slots(trigger, family, members, cfg=None):
    """[(row, slot, vid)] for every member row whose table offers ``family`` (first offering slot per row);
    labels by fullpol_learn.option_labels(entry, cfg) (the EV-BC label function; cfg resolves N2 labels)."""
    from .fullpol_learn import option_labels
    vids = list(trigger['context']['members'])
    out = []
    for vid in members:
        if vid not in vids:
            continue
        i = vids.index(vid)
        for j, entry in enumerate(trigger['tables'][i]):
            if j and trigger['mask'][i][j] and family in option_labels(entry, cfg):
                out.append((i, j, vid))
                break
    return out


def build_items(acts, triggers, cfg=None, lookback=None):
    """(items, skipped). ``triggers``: {(tape key, t): trigger record or None}, or a list of such dicts = the
    same replay under several slate draws (candidate_rng seeds; which candidates a slate offers is random in
    training too). Per activation and tape, the first draw (in list order) with a decision second in the
    onset window whose slate offers the plan to an acting member is used (meta slate_draw)."""
    slates = triggers if isinstance(triggers, list) else [triggers]
    helped = [-a['diff'] for a in acts if a['diff'] < 0]
    hurt = [a['diff'] for a in acts if a['diff'] > 0]
    help_max, hurt_max = max(helped or [1.]), max(hurt or [1.])
    items, skipped = [], []
    for a in acts:
        if a['diff'] == 0:
            skipped.append(dict(_meta(a), reason='zero paired effect'))
            continue
        kind = 'act' if a['diff'] < 0 else 'none'
        arms = ([a['family']] if kind == 'act' else
                [x for x in (a.get('ref_arms') or REFERENCE_ARMS) if x != a['family']])
        base_w = (-a['diff']/help_max) if kind == 'act' else (a['diff']/hurt_max)/len(arms)
        for arm in arms:
            job = job_name(a['cell'], a['seed'], arm)
            key = f"{a['root']}__{job}"
            meta = dict(_meta(a), job=job, tape_key=key, tape_arm=arm)
            chosen = trig = None
            any_decision = False
            for k, per_seed in enumerate(slates):
                decisions = [t for t in reversed(window(a['t'], lookback)) if per_seed.get((key, t)) is not None]
                any_decision |= bool(decisions)
                for t in decisions:
                    hits = member_slots(per_seed[(key, t)], a['family'], a['members'], cfg)
                    if hits:
                        chosen, trig = t, per_seed[(key, t)]
                        meta.update(slate_draw=k)
                        break
                if chosen is not None:
                    break
            if not any_decision:
                skipped.append(dict(meta, kind=kind, reason='no trigger decision in the onset window'))
                continue
            if chosen is None:
                skipped.append(dict(meta, kind=kind,
                                    reason='plan not offered to any member row in the onset window'))
                continue
            meta.update(decision_t=chosen, onset_offset_s=chosen-a['t'])
            for row, slot, vid in hits:
                w = base_w/len(hits)
                if not (0. < w <= 1.):
                    skipped.append(dict(meta, kind=kind, reason=f'weight {w} outside (0, 1]'))
                    continue
                items.append(dict(kind=kind, weight=float(w), row=int(row), slot=int(slot), trigger=trig,
                                  meta=dict(meta, member=vid)))
    assert_verification_seeds({x['meta']['seed'] for x in items})
    return items, skipped


def _meta(a):
    unc = a.get('admission') == 'unconditional'
    return dict(root=a['root'], cell=a['cell'], seed=a['seed'], t=a['t'], family=a['family'], source=a['source'],
                bin=a['bin'], admission=a.get('admission', 'situational'),
                effect_reference='u0' if unc else INCUMBENT,
                **{('diff_vs_u0' if unc else 'diff_vs_incumbent'): a['diff']}, members=list(a['members']))


def summary(items):
    return dict(rows=len(items), act=sum(x['kind'] == 'act' for x in items),
                none=sum(x['kind'] == 'none' for x in items),
                act_weight=sum(x['weight'] for x in items if x['kind'] == 'act'),
                none_weight=sum(x['weight'] for x in items if x['kind'] == 'none'),
                seeds=sorted({x['meta']['seed'] for x in items}),
                plans=sorted({x['meta']['family'] for x in items}),
                cells=sorted({x['meta']['cell'] for x in items}),
                sources=sorted({x['meta']['source'] for x in items}),
                by_plan={f: dict(act=sum(x['kind'] == 'act' and x['meta']['family'] == f for x in items),
                                 none=sum(x['kind'] == 'none' and x['meta']['family'] == f for x in items))
                         for f in sorted({x['meta']['family'] for x in items})})


def write_outputs(items, skipped, out_dir, provenance):
    """DEMO.json (demo_bc schema, LF), DEMO.npz, DEMO.manifest.json. Returns the manifest."""
    import numpy as np
    from . import demo_bc
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(schema=demo_bc.SCHEMA, rule=demo_bc.RULE, extractor=SCHEMA_NPZ, provenance=provenance,
               items=items, skipped=skipped, summary=summary(items))
    raw = (json.dumps(doc, sort_keys=True)+'\n').encode('utf-8')
    (out/'DEMO.json').write_bytes(raw)
    arrays = dict(
        kind=np.array([x['kind'] for x in items], dtype='U4'),
        weight=np.array([x['weight'] for x in items], dtype=np.float64),
        row=np.array([x['row'] for x in items], dtype=np.int64),
        slot=np.array([x['slot'] for x in items], dtype=np.int64),
        seed=np.array([x['meta']['seed'] for x in items], dtype=np.int64),
        t=np.array([x['meta']['t'] for x in items], dtype=np.int64),
        family=np.array([x['meta']['family'] for x in items], dtype='U64'),
        source=np.array([x['meta']['source'] for x in items], dtype='U4'),
        cell=np.array([x['meta']['cell'] for x in items], dtype='U32'),
        job=np.array([x['meta']['job'] for x in items], dtype='U128'),
        member=np.array([x['meta']['member'] for x in items], dtype='U64'),
        trigger_json=np.array([json.dumps(x['trigger'], sort_keys=True) for x in items], dtype=object).astype(str))
    with open(out/'DEMO.npz', 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    manifest = dict(schema=SCHEMA_NPZ, json=dict(path=str(out/'DEMO.json'), sha256=sha256_bytes(raw)),
                    npz=dict(path=str(out/'DEMO.npz'), sha256=sha256_file(out/'DEMO.npz')),
                    summary=doc['summary'], skipped=len(skipped),
                    skipped_reasons={r: sum(s['reason'] == r for s in skipped) for r in sorted({s['reason'] for s in skipped})},
                    provenance=provenance)
    (out/'DEMO.manifest.json').write_bytes((json.dumps(manifest, indent=1, sort_keys=True)+'\n').encode('utf-8'))
    return manifest


def load_npz(path, sha256=None):
    """Items (demo_bc format) from DEMO.npz; optional sha256 check."""
    import numpy as np
    raw = Path(path).read_bytes()
    if sha256 is not None and sha256_bytes(raw) != sha256:
        raise ValueError('demo npz sha256 differs')
    z = np.load(path, allow_pickle=False)
    items = []
    for i in range(len(z['kind'])):
        items.append(dict(kind=str(z['kind'][i]), weight=float(z['weight'][i]), row=int(z['row'][i]),
                          slot=int(z['slot'][i]), trigger=json.loads(str(z['trigger_json'][i])),
                          meta=dict(seed=int(z['seed'][i]), t=int(z['t'][i]), family=str(z['family'][i]),
                                    source=str(z['source'][i]), cell=str(z['cell'][i]), job=str(z['job'][i]),
                                    member=str(z['member'][i]))))
    assert_verification_seeds({x['meta']['seed'] for x in items})
    return items


# ----------------------------------------------------------------------------------------------- parity
PARITY_KEYS = ('context', 'tables', 'descriptors', 'mask')


def _j(x):
    return json.dumps(x, sort_keys=True)


def bank_families(cfg):
    """Families of the guidance bank templates offered on the slate (lc_families generator_args templates, B)."""
    from . import lc_families
    raw = ((cfg.get(lc_families.CONFIG_KEY) or {}).get('generator_args') or {}).get('templates') or {}
    return sorted({t['family'] for t in raw.get('B') or ()})


def compare_records(got, rec, families=()):
    """Comparison of two track2_trigger records. Exact: context (the policy observation), row members, and per
    row slot 0 ('none': entry, mask, descriptor). Bank-template entries (``families``, the demo label space)
    offered in BOTH records on the same row must be identical (entry + descriptor). WHICH candidates a slate
    offers (N2 group_hold / lane_drop_initiate parameters, LCEXEC free_program and verified-template draws) is
    drawn from candidate_rng, whose training stream is not recoverable from the RESULT; offered-only-on-one-side
    counts are reported, not compared."""
    from .fullpol_learn import option_labels
    fams = set(families)

    def bank(r, i):
        out = {}
        for j, e in enumerate(r['tables'][i]):
            if j and r['mask'][i][j] and e is not None:
                for f in fams & set(option_labels(e)):
                    out.setdefault(f, (_j(e), _j(r['descriptors'][i][j])))
        return out
    out = dict(context=_j(got['context']) == _j(rec['context']), rows=len(got['tables']) == len(rec['tables']))
    slot0 = both = same = one_side = 0
    if out['rows']:
        for i in range(len(rec['tables'])):
            slot0 += (_j(got['tables'][i][0]) == _j(rec['tables'][i][0]) and got['mask'][i][0] == rec['mask'][i][0]
                      and _j(got['descriptors'][i][0]) == _j(rec['descriptors'][i][0]))
            bg, br = bank(got, i), bank(rec, i)
            for f in set(bg) | set(br):
                if f in bg and f in br:
                    both += 1
                    same += bg[f] == br[f]
                else:
                    one_side += 1
    n = len(rec['tables'])
    out.update(slot0_equal=f'{slot0}/{n}', bank_entries_both=both, bank_entries_equal=same,
               bank_entries_one_side_rng=one_side, full_tables=_j(got['tables']) == _j(rec['tables']))
    out['ok'] = bool(out['context'] and out['rows'] and slot0 == n and same == both)
    return out


def parity_recorded(cfg, base, state, result, tape, *, noise_seed, t=300):
    """Training-path parity: the featurizer's record at the episode's first decision second (before any of the
    episode's own sessions exist) vs the RECORDED training record, from the recorded controller inputs (exact)
    and from the tape alone (strip, as for verification tapes)."""
    steps = {int(s['time_s']): s for s in result['steps'] if int(s['time_s']) <= t}
    rec = steps[t]['sample'].get('track2_trigger')
    if rec is None:
        raise ValueError(f'no recorded trigger record at {t}')
    exact = featurize(cfg, base, state, tape, {t}, seed=noise_seed, strip=False,
                      inputs={k: (s['rows'], s['commands']) for k, s in steps.items()})[t]
    strip = featurize(cfg, base, state, tape, {t}, seed=noise_seed, strip=True)[t]
    fams = bank_families(cfg)
    return dict(t=t, rows=len(rec['context']['members']), bank_families=fams,
                exact=compare_records(exact, rec, fams), strip=compare_records(strip, rec, fams))


def parity_strip(cfg, base, state, result, tape, *, noise_seed, until=1199):
    """Input-path parity over a whole episode: inputs rebuilt from the tape alone with its audit REMOVED (the
    harshest verification-tape case: no MAPPO raw command for any CAV, so every command is the row's recorded
    policy-mean command) vs the recorded training inputs (step rows/commands), same featurizer, same seed,
    trigger forced to none; FULL record equality (all four keys) at every second."""
    steps = {int(s['time_s']): s for s in result['steps']}
    last = min(until, max(steps), max(tape))
    want = set(range(300, last+1))
    a = featurize(cfg, base, state, tape, want, seed=noise_seed, strip=False,
                  inputs={k: (s['rows'], s['commands']) for k, s in steps.items()})
    no_audit = {t: dict(line, actions=[]) for t, line in tape.items()}
    b = featurize(cfg, base, state, no_audit, want, seed=noise_seed, strip=True)
    bad = [t for t in sorted(want) if (a[t] is None) != (b[t] is None) or
           (a[t] is not None and any(_j(a[t][k]) != _j(b[t][k]) for k in PARITY_KEYS))]
    return dict(ticks=len(want), decision_ticks=sum(a[t] is not None for t in want), mismatched=len(bad),
                first_mismatches=bad[:10])


# ----------------------------------------------------------------------------------------------- CLI
HOST_ROOTS = {  # WMVERIFY root name -> (host, job root); REGISTRATION.md WMVERIFY records (tapes moved to cpuW)
    'wmv_mc': ('cpuW', '/home/ubuntu/wmverify_mc_tapes/wmv_mc'),
    'wmv_rep': ('cpuW', '/home/ubuntu/wmverify_mc_tapes/wmv_rep'),
    'wmv_p20': ('cpuW', '/home/ubuntu/wmverify_p20/wmv_p20'),
    'wmv_unc': ('cpuW', '/home/ubuntu/wmverify_mc_tapes/wmv_unc'),
}


def host_root(root):
    if root in HOST_ROOTS:
        return HOST_ROOTS[root]
    if root.startswith('wmv_r3_'):
        return ('gpuA', f'/root/autodl-fs/wmverify_20261108/{root}')
    raise ValueError(f'no host mapping for WMVERIFY root {root!r}')


def _inputs_receipt(paths):
    return [dict(path=str(p), sha256=sha256_file(p)) for p in paths]


def cmd_plan(a):
    guidance = json.loads(Path(guard_path(a.guidance)).read_bytes())
    effects, members = read_bank(a.bank)
    acts, skipped = plan_activations(guidance, read_onsets(a.ponset), effects, members)
    for x in acts:
        x['admission'] = 'situational'
    if a.cells:
        skipped += [dict(x, reason='cell not selected') for x in acts if x['cell'] not in a.cells]
        acts = [x for x in acts if x['cell'] in a.cells]
    if a.unconditional_bank:
        ua, us = unconditional_activations(guidance, a.unconditional_bank, a.cells)
        acts, skipped = acts+ua, skipped+us
    work = work_list(acts)
    reqs = {}
    for key, w in work.items():
        host, root = host_root(w['root'])
        reqs.setdefault(host, []).append(dict(key=key, tape=f"{root}/jobs/{w['job']}/plant/observable.jsonl.gz",
                                              until=w['until']))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(schema=SCHEMA_NPZ+'.plan', guidance=dict(path=a.guidance, sha256=sha256_file(a.guidance)),
                ponset=_inputs_receipt(a.ponset), bank=_inputs_receipt(a.bank),
                unconditional_bank=_inputs_receipt(a.unconditional_bank or ()), cells=a.cells, activations=acts,
                skipped=skipped, work=work, requests={h: len(r) for h, r in reqs.items()})
    (out/'PLAN.json').write_bytes((json.dumps(plan, indent=1, sort_keys=True)+'\n').encode('utf-8'))
    for host, jobs in reqs.items():
        (out/f'REQUEST_{host}.json').write_bytes((json.dumps(dict(jobs=jobs), indent=1)+'\n').encode('utf-8'))
    print(json.dumps(dict(activations=len(acts), helped=sum(x['diff'] < 0 for x in acts),
                          hurt=sum(x['diff'] > 0 for x in acts), skipped=len(skipped), tapes=len(work),
                          requests=plan['requests'])))


_WORKER = {}


def _worker_init(run_root, guidance_path, threads):
    import torch
    torch.set_num_threads(int(threads))
    guidance = json.loads(Path(guidance_path).read_bytes())
    _WORKER['rt'] = runtime(run_root, guidance)


def _worker_tape(path, ticks, seed):
    cfg, base, state = _WORKER['rt']
    return featurize(cfg, base, state, tape_seconds(path), set(ticks), seed=seed, force_decision=set(ticks))


def slim_cache(path, families, cfg=None):
    """{t: trigger} of a cached draw, only the records offering one of ``families`` on some row."""
    from .fullpol_learn import option_labels
    families, keep = set(families), {}
    if Path(path).exists():
        for t, trig in json.loads(gzip.decompress(Path(path).read_bytes()))['triggers'].items():
            if trig is not None and any(e is not None and families & set(option_labels(e, cfg))
                                        for row in trig['tables'] for e in row):
                keep[int(t)] = trig
    return keep


def _worker_slim(path, families):
    return slim_cache(path, families, _WORKER['rt'][0])


class LazySlates:
    """{(tape key, t): trigger} view of one draw's trigger cache. Each tape file is read once and only the
    records whose slate offers one of ``families`` on some row are kept (all others can never give an item)."""

    def __init__(self, cache, seed, families, cfg=None):
        self.cache, self.seed, self.families, self.cfg, self.data = Path(cache), int(seed), set(families), cfg, {}

    def _load(self, key):
        self.data[key] = slim_cache(self.cache/f'{key}__seed{self.seed}.json.gz', self.families, self.cfg)

    def get(self, k, default=None):
        key, t = k
        if key not in self.data:
            self._load(key)
        return self.data[key].get(int(t), default)

    def __getitem__(self, k):
        v = self.get(k)
        if v is None:
            raise KeyError(k)
        return v


def cmd_extract(a):
    from concurrent.futures import ProcessPoolExecutor
    plan = json.loads(Path(a.plan).read_bytes())
    guidance = json.loads(Path(guard_path(plan['guidance']['path'])).read_bytes())
    if sha256_file(plan['guidance']['path']) != plan['guidance']['sha256']:
        raise ValueError('guidance changed since PLAN')
    cfg = runtime(a.run_root, guidance)[0]
    cache = Path(a.cache) if a.cache else Path(a.out)/'triggers'
    cache.mkdir(parents=True, exist_ok=True)
    tape_receipts, todo = {}, []
    for key, w in sorted(plan['work'].items()):
        path = Path(a.tapes)/f'{key}.jsonl.gz'
        if not path.exists():
            tape_receipts[key] = dict(missing=True)
            continue
        tape_receipts[key] = dict(sha256=sha256_file(path), ticks=w['ticks'])
        cpath = cache/f'{key}__seed{a.seed}.json.gz'
        if a.build_only:
            continue
        if cpath.exists():
            c = json.loads(gzip.decompress(cpath.read_bytes()))
            ok = c['tape_sha256'] == tape_receipts[key]['sha256'] and c['ticks'] == w['ticks'] and c['seed'] == a.seed
            del c
            if ok:
                continue
        todo.append((key, str(path), w['ticks']))
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_worker_init,
                             initargs=(a.run_root, plan['guidance']['path'], a.threads)) as pool:
        futs = {pool.submit(_worker_tape, path, ticks, a.seed): (key, ticks) for key, path, ticks in todo}
        for fut in list(futs):
            key, ticks = futs.pop(fut)
            got = fut.result()
            cpath = cache/f'{key}__seed{a.seed}.json.gz'
            cpath.write_bytes(gzip.compress(json.dumps(dict(tape_sha256=tape_receipts[key]['sha256'], ticks=ticks,
                                                            seed=a.seed, triggers={str(t): v for t, v in got.items()}),
                                                       sort_keys=True).encode('utf-8')))
            print(json.dumps(dict(tape=key, ticks=len(ticks))), flush=True)
            del got, fut
    if a.no_build:
        return
    if a.seed not in a.draw_seeds:
        raise ValueError('--seed must be one of --draw-seeds')
    fams = sorted({x['family'] for x in plan['activations']})
    slates = [LazySlates(cache, seed, fams, cfg) for seed in a.draw_seeds]
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_worker_init,
                             initargs=(a.run_root, plan['guidance']['path'], a.threads)) as pool:
        futs = {pool.submit(_worker_slim, str(cache/f'{key}__seed{sl.seed}.json.gz'), fams): (sl, key)
                for sl in slates for key in plan['work']}
        for fut in list(futs):
            sl, key = futs.pop(fut)
            sl.data[key] = fut.result()
    items, skipped = build_items(plan['activations'], slates, cfg)
    skipped = plan['skipped']+skipped+[dict(tape_key=k, reason='tape missing') for k, r in tape_receipts.items()
                                       if r.get('missing')]
    provenance = dict(rule='DEMOX demo_extract v1 (MAIN 2026-11-08; wm+rule admitted pairs)',
                      plan=dict(path=a.plan, sha256=sha256_file(a.plan)), guidance=plan['guidance'],
                      ponset=plan['ponset'], bank=plan['bank'], tapes=tape_receipts,
                      run_root=dict(path=str(a.run_root), **{n: sha256_file(Path(a.run_root)/n) for n in
                                    ('CONFIG.json', 'B_checkpoint_000.pt', 'B_track2_000.pt')}),
                      featurizer='track2_backend.episode_components(controller_class(acceleration.Controller)), '
                                 'trigger forced none, decision forced due at every window second, slate draws (noise_seed) %s' % a.draw_seeds,
                      parity=json.loads(Path(a.parity).read_bytes()) if a.parity else None)
    manifest = write_outputs(items, skipped, a.out, provenance)
    print(json.dumps(dict(json=manifest['json'], npz=manifest['npz'], **{k: manifest['summary'][k] for k in
                          ('rows', 'act', 'none', 'seeds', 'plans', 'cells', 'sources')},
                          skipped=manifest['skipped_reasons'])))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--guidance', required=True)
    p.add_argument('--ponset', nargs='+', required=True)
    p.add_argument('--bank', nargs='+', required=True)
    p.add_argument('--unconditional-bank', nargs='*', default=[])
    p.add_argument('--cells', nargs='*', default=[])
    p.add_argument('--out', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--plan', required=True)
    p.add_argument('--run-root', required=True)
    p.add_argument('--tapes', required=True)
    p.add_argument('--seed', type=int, default=0, help='replay noise_seed computed in this call')
    p.add_argument('--draw-seeds', type=int, nargs='+', default=[0], help='cached slate draws used, in order')
    p.add_argument('--parity')
    p.add_argument('--no-build', action='store_true', help='featurize + cache only')
    p.add_argument('--cache', help='trigger cache dir (default <out>/triggers)')
    p.add_argument('--build-only', action='store_true', help='no featurization; build from the cached draws')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    (cmd_plan if a.cmd == 'plan' else cmd_extract)(a)


if __name__ == '__main__':
    main()
