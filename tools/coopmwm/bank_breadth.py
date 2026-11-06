"""F3: one verified, observable context pool for practice and trigger teaching.

PLAN_FIX_READY.json is a handoff, not verification evidence. Its optional
``outputs`` list names bank run roots and their B/R replay_sources. Every entry
still goes through the existing per_seed_v1, safety, tape-hash and effective
target readers. No WM-only proposal or arbitrary move label is admitted here.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import warnings

from .protocol import digest

SCHEMA = 'coopmwm-plan-fix-banks-v1'
RULE = ('F3 owner 2026-11-05 15:15: all usable admitted verified bank contexts; '
        'distinct contexts before repeats; least-used contexts across updates; '
        'uniform supported-family mass, identical B/R family distribution')


def inputs(cfg):
    """Current bank plus optional PLAN_FIX/S-VARIETY bank outputs (CPU reads).

    Contract: {schema: SCHEMA, outputs: [{run_root: ..., replay_sources:
    {B: [...], R: [...]}}]}. Paths are relative to the handoff's directory or
    absolute. Missing handoff / empty outputs means the current bank only.
    Both arms use the same handoff, with their own verified guidance and CRN.
    """
    from .track2_backend import _path
    root = _path(cfg['run_root']).resolve()
    ready = _path(cfg.get('plan_fix_ready', root/'PLAN_FIX_READY.json'))
    raw = [dict(run_root=str(root), replay_sources=cfg.get('replay_sources', {}))]
    if ready.exists():
        record = json.loads(ready.read_text(encoding='utf-8'))
        if record.get('schema') != SCHEMA or not isinstance(record.get('outputs'), list):
            raise ValueError('PLAN_FIX_READY requires the F3 bank-output contract')
        for item in record['outputs']:
            if not isinstance(item, dict) or 'run_root' not in item:
                raise ValueError('PLAN_FIX output must name a verified bank run_root')
            base = _path(item['run_root'])
            base = base if base.is_absolute() else ready.parent/base
            sources = item.get('replay_sources', {})
            if not isinstance(sources, dict) or set(sources)-{'B', 'R'}:
                raise ValueError('PLAN_FIX replay_sources must be common B/R bank inputs')
            converted = {}
            for arm, paths in sources.items():
                if not isinstance(paths, list):
                    raise ValueError('PLAN_FIX replay_sources must contain path lists')
                converted[arm] = [str(_path(p if Path(p).is_absolute() else ready.parent/p).resolve())
                                  for p in paths]
            raw.append(dict(run_root=str(_path(base).resolve()), replay_sources=converted))
    from . import group_lane_bank
    raw.extend(group_lane_bank.inputs(cfg))
    seen, out = set(), []
    for item in raw:
        for paths in item['replay_sources'].values():
            for path in paths:
                _path(path)
        key = digest(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _mined_row(rows, arm, key, seed_index):
    """The MINED_MOVES row binding one example: a walls row binds its OWN seed exactly;
    a v1 pointer row (no seed field) binds any example of its plan. Ambiguous
    duplicates of one example are refused."""
    hits = [r for r in rows if r['arm'] == arm
            and (r['verification_id'], r['plan_sha256']) == key]
    hits = ([r for r in hits if r.get('seed') == seed_index]
            or [r for r in hits if 'seed' not in r])
    if len({digest(r) for r in hits}) > 1:
        raise ValueError('MINED_MOVES binds one example ambiguously: '+key[0])
    return hits[0] if hits else None


def visit(cfg, arm, consumer):
    """Visit ALL usable verified examples, including legacy and imported banks.

    The callback also receives the bank configuration that resolves its snapshot.
    Deduplication is by verification/seed/plan, not a path or policy-version tag.
    """
    if arm not in ('B', 'R'):
        return
    from . import track2_backend as p1
    certified = None
    if cfg.get('executable_label_manifest'):
        from . import executable_labels as el
        el._sha(cfg.get('executable_label_runtime_sha256'))
        certified = el.read_manifest(cfg['executable_label_manifest'])
    seen = set()
    for source in inputs(cfg):
        local = dict(cfg, **source)
        # Breadth and trigger teaching must not bypass the prelaunch admission
        # that weighted practice uses. Validate ALL children before any callback.
        from . import practice_starts as ps, practice_weighting as pw
        with p1._bank_view(local):
            augmented, manifest_sha = pw.augmented_examples(local['run_root'], arm,
                ps.examples(local['run_root'], arm), deduplicate=False)
        if any(not local.get('augmentation_prelaunch') or
               r['prelaunch'] != local['augmentation_prelaunch'] for r in augmented):
            raise ValueError('augmented breadth needs parent registered prelaunch provenance')
        local['_breadth_augmented'] = {r['example_id']: dict(r,
            augmentation_sha256=manifest_sha) for r in augmented}
        links = ps._links(local['run_root'], arm)
        mined = {(r['verification_id'], r['plan_sha256'])
                 for r in local.get('_g5_mined', ()) if r['arm'] == arm}
        matched_mined = set()

        def consume(*, receipt, example, plan, pairs, trace):
            # bank_pairs has already checked legacy admission/cooperation too.
            if not pairs:
                return
            bound = local
            if '_g5_mined' in local:
                mined_key = (receipt['verification_id'], digest(plan))
                # per_seed_v1 may admit DIFFERENT plans on different seeds of
                # one verification. Import only the explicitly mined bindings.
                if mined_key not in mined:
                    return
                mined_move(plan)  # exact G5 macro, or an executable verified acceleration plan
                from . import group_lane_bank
                row = _mined_row(local['_g5_mined'], arm, mined_key,
                                 int((example or {}).get('seed_index', 1)))
                if row is None:
                    # An admitted example on a seed the file does not bind is not a
                    # registered mined example; only bound rows are ingested.
                    return
                matched_mined.add(mined_key)
                value = group_lane_bank.binding(row)
                if value is not None:
                    if (list(value['members']) != list(plan['members'])
                            or int(value['duration_s']) != int(plan['duration_s'])
                            or ('profile' in plan and value['profile'] != [[float(x) for x in line]
                                                                          for line in plan['profile']])):
                        raise ValueError('MINED_MOVES binding differs from the verified plan it names')
                    bound = dict(local, _mined_binding=value)
            if plan.get('g5_move') and not local.get('group_lane_moves'):
                return
            ident = (example or {}).get('example_id', receipt['verification_id']+'#s1')
            if (links.get(receipt['verification_id'], {}).get('augmentation')
                    and ident not in local['_breadth_augmented']):
                raise ValueError('augmented breadth child missing verified prelaunch manifest')
            key = (receipt['verification_id'], int((example or {}).get('seed_index', 1)), digest(plan))
            if key in seen:
                return
            seen.add(key)
            p1._path(trace)
            # MINED_MOVES rows are admitted by the mining operator's own full-window
            # group/base/each-solo CRN verification (group>best-solo CI95 low>0); the
            # G1 manifest certifies the existing bank lessons only.
            if certified is not None and '_g5_mined' not in local:
                try:
                    snap, _ = verified_restart(local['run_root'], arm, receipt, example, plan)
                except UnboundContext:
                    return
                lesson = el.bank_certificate(certified, arm=arm, receipt=receipt, example=example,
                    plan=plan, **el.restart_binding(snap), runtime_sha256=cfg['executable_label_runtime_sha256'])
                if lesson is None:
                    return
                bound = dict(local, _executable_lesson=lesson)
            consumer(cfg=bound, receipt=receipt, example=example, plan=plan, pairs=pairs, trace=trace)
        with p1._bank_view(local) as rb:
            rb.bank_pairs(Path(local['run_root']), arm, local['training_seed'],
                sources=local.get('replay_sources', {}).get(arm, ()), v2v=True,
                v2v_intent=local if local.get('v2v_intent') else None,
                imitation_bank_filter='cooperative_only',
                imitation_target_space='effective', example_consumer=consume)
        if set(mined)-matched_mined:
            raise ValueError('MINED_MOVES reference has no usable native verified example')


def mined_move(plan):
    """Executable binding of a MINED_MOVES plan: G5 exact macro, else verified accel plan.

    Owner ~07:2x walls-only clean3: mined wall examples are ordinary verified acceleration
    plans (no lane actions). They must be executable by plans.acceleration_at over their
    whole duration; the move label is the shared-vocabulary projection.
    """
    if plan.get('g5_move') or plan.get('uses_lane_actions'):
        from .group_lane import from_plan as g5
        return g5(plan)
    from . import moves
    from .plans import acceleration_at
    duration = plan.get('duration_s')
    if not plan.get('members') or type(duration) is not int or duration <= 0:
        raise ValueError('mined acceleration plan needs members and an integer duration_s')
    for t in range(duration):
        accels = acceleration_at(plan, t)
        if set(accels) - set(plan['members']):
            raise ValueError('mined acceleration plan commands a non-member')
    return moves.from_plan(plan)


class UnboundContext(ValueError):
    """An old source has no immutable snapshot binding; never certify its current file."""


def verified_restart(root, arm, receipt, example, plan, mined=None):
    """Restore ONLY a context bound by original sealed first/own-seed evidence.

    Called at pool construction and dispatch. No freshly measured hash can stand
    in for a missing receipt hash, including for legacy and imported LS sources.
    ``mined`` (walls-only clean3): the validated MINED_MOVES binding of this exact
    example; without a decision link the restart snapshot comes from that row's own
    sealed SUMO verification. None keeps the byte-identical decision-link path."""
    import hashlib
    from . import replay_bank as rb, practice_starts as ps, sumo_verify as sv
    from . import exact_grade as xg, verify_perseed as vp
    from .track2_backend import _path
    first = {k: v for k, v in receipt.items() if not k.startswith('_')}
    xg.verify_sealed(first, 'receipt_sha256')
    if not first.get('snapshot_sha256'):
        raise UnboundContext('original verification has no immutable snapshot_sha256')
    snap = rb._decision_snapshot(root, arm, receipt, ps._links(root, arm), mined)

    def check_snapshot(evidence, snapshot):
        if (evidence.get('snapshot_sha256') != sv.snapshot_sha(snapshot)
                or evidence.get('episode') != snapshot['episode']
                or evidence.get('t0') != snapshot['t0']):
            raise ValueError('verification snapshot identity/world/decision time mismatch')
        # snapshot_sha projects the RNG metadata. Also bind the actual tensor
        # that the runner will restore to that verified metadata.
        if hashlib.sha256(snapshot['generator_state'].numpy().tobytes()).hexdigest() != (
                snapshot['rng'].get('generator_state_sha256')):
            raise ValueError('verification snapshot generator state mismatch')

    check_snapshot(first, snap)
    episode = snap['episode']
    from .map_runtime import primary_train_world  # N1: merge 986000-986071 or the map's TRAIN registry
    if (episode.get('role') != 'train' or episode.get('arm') != arm
            or not primary_train_world(dict(episode['world'], seed=int(episode['world']['seed'])))):
        raise ValueError('F3 practice requires ordinary TRAIN snapshots')
    vid = first['verification_id']
    confirmation = rb.confirmations(root, arm).get(vid, {})
    if confirmation.get('t0') != snap['t0']:
        raise ValueError('verification confirmation decision time mismatch')
    if not confirmation.get('admitted') or (example is None and
            (confirmation.get('cooperation') or {}).get('cooperative') is not True):
        raise ValueError('verified restart confirmation no longer admitted/cooperative')
    seed = int((example or {}).get('seed_index', 1))
    if seed < 1:
        raise ValueError('invalid verified restart seed index')
    salt = None if seed == 1 else vid if seed == 2 else vp.extra_seed_salt(vid, seed)
    if salt is not None:
        snap = rb.replicate(snap, salt)
    own = first
    if example is not None:
        current = next((e for e in (confirmation.get('per_seed_v1') or {}).get('examples', [])
                        if e['example_id'] == example['example_id']), None)
        if current != example or not example.get('admitted') or example.get('cooperative') is not True:
            raise ValueError('verified restart example evidence changed/not admitted')
        tape = example['tape']
        if seed > 1:
            path = _path(Path(tape['branches_dir']).parent/'verifications'/
                         (tape['verification_id']+'.json'))
            own = json.loads(path.read_text(encoding='utf-8'))
            xg.verify_sealed(own, 'receipt_sha256')
            if not own.get('snapshot_sha256'):
                raise UnboundContext('own-seed verification has no immutable snapshot_sha256')
        if (own['verification_id'] != tape['verification_id']
                or tape['window'] != own['window'] or tape['window']['start_s'] != snap['t0']
                or tape['branch'] not in own['branches']):
            raise ValueError('verified restart own-seed tape binding mismatch')
        branch_index = own['branches'].index(tape['branch'])-1
        hashes = own.get('candidate_plan_sha256')
        if branch_index < 0 or (hashes and hashes[branch_index] != digest(plan)) or (
                branch_index == own['chosen'] and digest(own['chosen_plan']) != digest(plan)):
            raise ValueError('verified restart own-seed plan binding mismatch')
    if confirmation.get('verify_mode') == vp.ALL8:
        meta = (confirmation.get('first_seed') if seed == 1 else confirmation.get('seed2')
                if seed == 2 else (confirmation.get('extra') or {}).get(str(seed))) or {}
        first_meta = confirmation.get('first_seed') or {}
        if (first_meta.get('receipt_sha256') != first['receipt_sha256']
                or first_meta.get('verification_id') != vid
                or meta.get('receipt_sha256') != own['receipt_sha256']
                or meta.get('verification_id') != own['verification_id']
                or (seed > 1 and (meta.get('salt') != salt
                    or meta.get('snapshot_sha256') != sv.snapshot_sha(snap)))):
            raise ValueError('verified restart native per-seed CRN evidence mismatch')
    elif seed > 1:
        if seed != 2 or (confirmation.get('replication') or {}).get('receipt_sha256') != own['receipt_sha256']:
            raise ValueError('verified restart legacy replication evidence mismatch')
    check_snapshot(own, snap)
    binding = dict(first_receipt_sha256=first['receipt_sha256'], own_receipt_sha256=own['receipt_sha256'],
        confirmation_sha256=confirmation.get('confirmation_sha256'), example_sha256=digest(example),
        seed_index=seed, replication_salt=salt)
    return snap, binding


def load_verified_restart(arm, lookup):
    """Re-read the original source and compare immutable evidence on resume."""
    from . import replay_bank as rb
    from .track2_backend import _path, _bank_view
    root = _path(lookup['run_root'])
    expected = lookup['receipt']
    for key in ('_source', '_branches'):
        if key in expected:
            _path(expected[key])
    if lookup.get('binding_version') != 1:
        raise ValueError('F3 practice plan predates original verification binding; use a new update')
    sources = [expected['_source']] if expected.get('_source') else []
    with _bank_view(lookup['bank_cfg']):
        current = next((r for r in rb.bank_decisions(root, arm, sources=sources)
                        if r['verification_id'] == expected['verification_id']
                        and r['receipt_sha256'] == expected['receipt_sha256']), None)
        if current is None:
            raise ValueError('verified restart original receipt evidence changed/missing')
        snapshot, binding = verified_restart(root, arm, current, lookup['example'], lookup['plan'],
                                             lookup.get('mined_binding'))
        if binding != lookup.get('verification_binding'):
            raise ValueError('verified restart immutable verification evidence changed')
        if lookup['bank_cfg'].get('executable_label_manifest'):
            from . import executable_labels as el
            value = el.read_manifest(lookup['bank_cfg']['executable_label_manifest'])
            lesson = el.bank_certificate(value, arm=arm, receipt=current, example=lookup['example'],
                plan=lookup['plan'], **el.restart_binding(snapshot),
                runtime_sha256=lookup['bank_cfg']['executable_label_runtime_sha256'])
            if lesson is None or lesson['lesson_sha256'] != lookup.get('executable_lesson_sha256'):
                raise ValueError('G1 frozen practice executable certificate changed/missing')
    return snapshot


def _certified_contexts(policy, pairs, frames, certification):
    """Replay a decision at t from its actual bound t-1 local observation.

    Keep the observer at t-1: observing t would overwrite both geometry and
    local-intent features. The recorded pair's x is also from t, so reconstruct
    sensor features from the validated local rows instead. The t frame/pair is
    used only to establish that all macro members can act at this decision.
    """
    from tools.coopnative.policy import encode
    from . import executable_labels as el, moves
    label = moves.Move.from_dict(certification['macro'])
    local = certification['local']
    time, observed_at = certification['decision_s'], local['observed_at_s']
    if observed_at != time-1 or any(t not in frames or frames[t].get('time_s', t) != t
                                   for t in (observed_at, time)):
        return
    for sample, weight in sorted(pairs, key=lambda pair: int(pair[0]['intent_link']['t'])):
        if int(sample['intent_link']['t']) != time:
            continue
        present = {r['vid'] for r in frames[time]['vehicles'] if r['is_cav']}
        eligible = present & set(sample['context']['members'])
        if not set(label.members) <= eligible:
            continue
        reader = deepcopy(policy)
        reader.observe(frames[observed_at]['vehicles'], observed_at)
        ego = local['self']
        if (ego not in reader.local_intents or ego not in reader.local_states
                or not set(local['acquirable_cav_ids']) <= set(reader.visible.get(ego, ()))):
            continue
        def observed_rows(rows):
            return sorted(({k: r[k] for k in moves._ROW_KEYS} for r in rows), key=lambda r: r['vid'])
        state = reader.local_states[ego]
        if observed_rows(state['rows']) != observed_rows(local['rows']):
            continue
        try:
            el._local(label, local, time)
        except ValueError:
            continue
        # Availability belongs to the certified decision, not the training
        # policy's current episode or all visible CAVs by default.
        state['acquirable_cav_ids'] = [v for v in local['acquirable_cav_ids'] if v in eligible]
        encoded = encode(state['rows'])
        index = encoded['members'].index(ego)
        context = dict(encoded, members=[ego], x=[encoded['x'][index]],
            anchors=[encoded['anchors'][index]], rsu=[0.]*len(encoded['rsu']),
            trigger_v2v={ego: reader.local_intents[ego]},
            observed_at_s=observed_at, decision_s=time)
        yield context, label, reader, float(weight), time
        return  # only this exact proposer/decision is certified


def observable_contexts(policy, plan, pairs, frames, *, t0, certification=None):
    """Earliest usable frame per acting member, rather than pairs[0] only.

    Each member is supervised once per example. A later frame gets the remaining
    onset/hold, never a fresh full-duration move. Frames with less than the
    registered minimum hold remaining are not representable. Own/partner packets are
    reconstructed with the same causal policy observer as online execution.
    Certified lessons use their explicit t-1 observation for BOTH binding and
    teacher inputs; they never reuse a current-time pair's sensor features.
    """
    if certification is not None:
        yield from _certified_contexts(policy, pairs, frames, certification)
        return
    from . import moves
    label = moves.from_plan(plan)
    remaining = set(label.members)
    reader = deepcopy(policy)
    observed = set()
    for sample, weight in sorted(pairs, key=lambda pair: int(pair[0]['intent_link']['t'])):
        time = int(sample['intent_link']['t'])
        elapsed = time-int(t0)
        # A macro already under way cannot be relabelled with a fresh yield /
        # gap-then-merge sequence at onset=0 in a later context.
        if label.family in moves.LANE_FAMILIES and elapsed > label.onset:
            continue
        active = max(0, elapsed-label.onset)
        if elapsed < 0 or (label.family != 'none' and label.hold-active < moves._HOLD_LO):
            continue
        if time not in frames:
            continue
        for t in (time-1, time):
            if t in frames and t not in observed:
                reader.observe(frames[t]['vehicles'], t)
                observed.add(t)
        source = sample['context']
        keep = [v for v in source['members'] if v in remaining
                and set(label.members) <= set(reader.visible.get(v, ()))
                and v in reader.local_intents]
        if not keep:
            continue
        # Remaining onset is relative to this observed context, not original t0.
        from dataclasses import replace
        target = (replace(label, onset=max(0, label.onset-elapsed), hold=label.hold-active)
                  if label.family != 'none' else label)
        context = dict(source, members=keep,
            x=[source['x'][source['members'].index(v)] for v in keep],
            trigger_v2v={v: reader.local_intents[v] for v in keep})
        yield context, target, reader, float(weight), time
        remaining.difference_update(keep)
        if not remaining:
            break


def practice_pool(cfg, arm, *, deduplicate=True):
    """Verified restarts including imported contexts; preserve exact seed salts."""
    from . import practice_starts as ps, moves, sumo_verify as sv
    rows = []
    def consume(*, cfg, receipt, example, plan, pairs, trace):
        root = cfg['run_root']
        links = ps._links(root, arm)
        try:
            snap, binding = verified_restart(root, arm, receipt, example, plan,
                                             cfg.get('_mined_binding'))
        except UnboundContext as error:
            warnings.warn(f'F3 excludes {arm}/{receipt["verification_id"]}: {error}', RuntimeWarning)
            return
        seed = int((example or {}).get('seed_index', 1))
        vid = receipt['verification_id']
        bank_cfg = {k: cfg[k] for k in ('run_root', 'training_seed', 'replay_sources', 'bank_rule',
            'executable_label_manifest', 'executable_label_runtime_sha256') if k in cfg}
        lesson = cfg.get('_executable_lesson')
        family = lesson['macro']['family'] if lesson else moves.from_plan(plan).family
        row = dict(verification_id=vid,
            example_id=(example or {}).get('example_id', f'{vid}#s1'), seed_index=seed, salt=vid,
            state_id=links.get(vid, {}).get('state_id', f'imported_{vid}'), fallback=False,
            source='verified_breadth', family=family,
            replication_salt=binding['replication_salt'],
            snapshot_lookup=dict(binding_version=1, run_root=str(root), bank_cfg=bank_cfg,
                receipt=receipt, example=example, plan=plan, verification_binding=binding),
            snapshot_sha256=sv.snapshot_sha(snap),
            context_id=digest(dict(snapshot=sv.snapshot_sha(snap), family=family,
                                   **({'mined_plan': digest(plan)} if '_g5_mined' in cfg else {}))))
        if cfg.get('_mined_binding') is not None:
            # The binding is revalidated on resume from the frozen plan itself.
            row['snapshot_lookup']['mined_binding'] = cfg['_mined_binding']
        if '_g5_mined' in cfg:
            # Owner 05:05: mined (move, seed) situations are shown, not only imitated.
            row['mined'] = True
            row['mined_plan_sha256'] = digest(plan)
        if row.get('mined') or family in set(cfg.get('demonstrate_families') or ()):
            # Same rule for B and R: practice starts of a mined family (each arm's own
            # verified examples) execute their own verified plan from t0.
            row['demonstrate'] = True
        if lesson is not None:
            row['snapshot_lookup']['executable_lesson_sha256'] = lesson['lesson_sha256']
            row['executable_lesson_sha256'] = lesson['lesson_sha256']
        child = cfg.get('_breadth_augmented', {}).get(row['example_id'])
        if child is not None:
            if child['snapshot_sha256'] != row['snapshot_sha256']:
                raise ValueError('augmented breadth verified snapshot_sha256 mismatch')
            row.update(child)
        rows.append(row)
    visit(cfg, arm, consume)
    # Multiple verified plans/links can describe the very same restart context.
    if not deduplicate:
        # Every verified example row (reporting/coverage); the sampler uses the
        # deduplicated pool, where one restart context may describe several plans.
        return sorted(rows, key=lambda row: (row['context_id'], row['verification_id'], row['example_id']))
    unique = {}
    for row in rows:
        unique.setdefault(row['context_id'], row)
    return sorted(unique.values(), key=lambda row: row['context_id'])


def matched_pools(cfg):
    pools = {arm: practice_pool(cfg, arm) for arm in ('B', 'R')}
    supports = {arm: {r['family'] for r in pool} for arm, pool in pools.items()}
    if supports['B'] != supports['R']:
        raise ValueError(f'F3 B/R verified family support differs: {supports}; add verified guidance')
    return pools


def practice_plan(root, arm, update, *, cfg, fraction, repeat_cap, count, training_seed):
    """Matched family quotas, least-used distinct restarts, frozen by the caller."""
    from . import practice_starts as ps
    pools = matched_pools(cfg)
    cap = int(repeat_cap)
    families = sorted({r['family'] for r in pools[arm]})
    capacities = {f: min(sum(r['family'] == f for r in p) for p in pools.values())*cap
                  for f in families}
    # Identical rotating family sequence for B/R, independent of arm RNGs.
    if families:
        shift = (int(update)-1) % len(families)
        families = families[shift:]+families[:shift]
    if cfg.get('practice_priority_mined'):
        # Registered clean3 smoke rule: mined families first so the end-to-end smoke
        # actually starts from a mined seed situation; identical for B and R.
        mined = sorted({r['family'] for r in pools[arm] if r.get('mined')})
        families = [f for f in families if f in mined]+[f for f in families if f not in mined]
    schedule, totals = [], Counter()
    while len(schedule) < int(count):
        available = [f for f in families if totals[f] < capacities[f]]
        if not available:
            break
        family = min(available, key=lambda f: totals[f])
        schedule.append(family)
        totals[family] += 1
    # R follows B's frozen/realized family sequence even if new contexts have
    # arrived since B collected. A replacement normal episode is not practice.
    b_path = ps.plan_path(root, 'B', update)
    if arm == 'R' and b_path.exists():
        from . import exact_grade as xg
        b_plan = json.loads(b_path.read_text(encoding='utf-8'))
        xg.verify_sealed(b_plan, 'practice_plan_sha256')
        if 'breadth' not in b_plan:
            raise ValueError('F3 R requires a breadth-aware B practice plan')
        sequence = [r['family'] for r in b_plan['rows']]
        b_update = Path(root)/f'B_UPDATE_{int(update):03}.json'
        if b_update.exists():
            from .accel_storage import read
            report = read(b_update)['practice']
            sequence = [r['breadth']['family'] for r in report['episodes']]
        if len(sequence) != int(count):
            raise ValueError('F3 R practice family count must match B realized practice')
        schedule = sequence
        totals = Counter(schedule)
        own_capacity = Counter(r['family'] for r in pools[arm])
        if any(n > own_capacity[f]*cap for f, n in totals.items()):
            raise ValueError('F3 R verified family capacity cannot match B practice')
    prior, used = Counter(), Counter()
    for path in sorted(ps.plan_path(root, arm, 0).parent.glob(f'PRACTICE_PLAN_{arm}_*.json')):
        old = json.loads(path.read_text(encoding='utf-8'))
        from . import exact_grade as xg
        xg.verify_sealed(old, 'practice_plan_sha256')
        if int(old['update']) < int(update):
            prior.update(r.get('context_id', r['example_id']) for r in old['rows'])
    selected = []
    for family in schedule:
        eligible = [r for r in pools[arm] if r['family'] == family and used[r['context_id']] < cap]
        first = bool(cfg.get('practice_priority_mined'))  # registered smoke rule only
        row = min(eligible, key=lambda r: (first and not r.get('mined'), used[r['context_id']], prior[r['context_id']],
            digest(dict(context=r['context_id'], seed=int(training_seed)))))
        selected.append(dict(row))
        used[row['context_id']] += 1
    return dict(arm=arm, update=int(update), training_seed=int(training_seed),
        fraction=float(fraction), repeat_cap=cap, requested=int(count), rows=selected,
        registry=dict(pool_examples=len(pools[arm]), cooperative_examples=len(pools[arm]),
            cooperative_capacity=len(pools[arm])*cap, empty_bank=not pools[arm], fallback_used=False),
        breadth=dict(rule=RULE, family_counts=dict(totals),
            eligible_contexts=[r['context_id'] for r in pools[arm]],
            unique_selected=len(used), inputs_sha256=digest(inputs(cfg))), rule=RULE)
