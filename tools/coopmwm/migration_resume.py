"""Operator migration resume: COOPMWM_MIGRATION_RESUME=<tag> (tag = 1..12 chars of [a-z0-9]).

Spot-reclaim migration (owner/MAIN 2026-11-08): the essentials backup keeps every top-level root file and
jobs/*/*.json, but not RESULT.json.gz payloads, tapes, scenarios/ or g4/. An arm's update that was attempted but
not completed at reclaim can never finish under the no-retry rules. With the variable set, an arm owner records the
abandoned attempt of its FIRST not-completed update u in a sealed receipt <root>/MIGRATION_RESUME_<arm>_<u:03>_<tag>.json
and re-attempts u FROM SCRATCH in a new namespace <job_namespace>_warm_u<u:03>m<tag> (job ids, practice slots, G4
inputs/envelope, the fair A-label files and the composite safety record), with the same worlds, the same noise seeds
(derived from the UNTAGGED ident) and the same code. Abandoned ledger rows stay where they are; the new jobs are
charged normally. Completed updates are adopted as before; later updates are untagged.

Resolution is receipt-driven: an (arm, update) WITHOUT a receipt keeps the original ids/paths (one glob, no other
change), so with the variable absent and no receipt every path is byte-identical. An update that has a receipt
resolves to the tag recorded in its sealed UPDATE (completed) or, while in flight, to the variable's tag, which must
then be set and receipted (fail closed otherwise). B and R must use the same tag for a G4-paired update.
"""
import json
import os
from pathlib import Path
import re
import time

ENV = 'COOPMWM_MIGRATION_RESUME'
SCHEMA = 'coopmwm-migration-resume-v1'
REASON = 'spot reclaim migration'
RULE = ('operator migration resume (COOPMWM_MIGRATION_RESUME=<tag>, MAIN 2026-11-08): the abandoned attempt of the '
        "arm's first not-completed update is receipted (its ledger rows stay, never retried or adopted) and the update "
        'is re-attempted from scratch under <namespace>_warm_u<u>m<tag> with the same worlds, noise seeds and code; '
        'new jobs are charged normally')
NOISE_RULE = 'noise seeds derive from the UNTAGGED wave ident (the original update CRN, shared with the other arms)'
G4_RULE = ('G4 migration re-attempt: chained after the abandoned plan of the same update (its reservation stays spent; '
           'last_update rewound to the value before that update was first planned)')
_TAG = re.compile(r'[a-z0-9]{1,12}')
# the tagged ids/paths pass the run's restricted-path guards (track2_backend._path, coopbandit/rerun_cpus FORBIDDEN)
_RESTRICTED = ('sealed', 'holdout', 'heldout', 'chunk5', 'j4b', 'inc1', 'vtype', 'driver', 'truth', 'assignment')
_MARKER_KEYS = {'rule', 'tag', 'receipt', 'receipt_sha256', 'supersedes_plan_sha256', 'rewound_last_update'}


def env_tag():
    """None when the variable is absent; else the validated tag (fail closed on any other format)."""
    value = os.environ.get(ENV)
    if value is None:
        return None
    if not _TAG.fullmatch(value) or any(token in 'm'+value for token in _RESTRICTED):
        raise ValueError(f'{ENV} must be 1..12 characters of [a-z0-9] without a restricted path token '
                         f'(got {value!r}; fail closed)')
    return value


def receipt_name(arm, update, tag):
    return f'MIGRATION_RESUME_{arm}_{int(update):03}_{tag}.json'


def receipt_path(root, arm, update, tag):
    return Path(root)/receipt_name(arm, update, tag)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _verified(path, arm, update, tag):
    from .exact_grade import verify_sealed
    value = _read(path)
    verify_sealed(value, 'receipt_sha256')
    if (value.get('schema') != SCHEMA or value.get('arm') != arm or value.get('update') != int(update)
            or value.get('tag') != tag):
        raise ValueError(f'migration receipt {Path(path).name} does not match its arm/update/tag (fail closed)')
    return value


def receipt_tags(root, arm, update):
    """Sorted tags of the sealed migration receipts of (arm, update); [] when never migrated (the common case)."""
    prefix = f'MIGRATION_RESUME_{arm}_{int(update):03}_'
    tags = []
    for path in sorted(Path(root).glob(prefix+'*.json')):
        tag = path.name[len(prefix):-len('.json')]
        if not _TAG.fullmatch(tag):
            raise ValueError(f'malformed migration receipt name {path.name} (fail closed)')
        _verified(path, arm, update, tag)
        tags.append(tag)
    return tags


def suffix(root, arm, update):
    """'' (original ids/paths) or 'm<tag>': the namespace suffix of ``arm``'s update ``update``."""
    tags = receipt_tags(root, arm, update)
    if not tags:
        return ''
    done = Path(root)/f'{arm}_UPDATE_{int(update):03}.json'
    if done.exists():
        marker = _read(done).get('migration_resume')
        if not isinstance(marker, dict) or marker.get('tag') not in tags:
            raise ValueError(f'{done.name} disagrees with its migration receipts {tags} (fail closed)')
        return 'm'+marker['tag']
    tag = env_tag()
    if tag is None:
        raise ValueError(f'{arm} update {update} has migration receipts {tags}; resuming it requires {ENV} '
                         '(fail closed)')
    if tag not in tags:
        raise ValueError(f'{arm} update {update} is not receipted for {ENV}={tag} (receipts {tags}; fail closed)')
    return 'm'+tag


def base_ident(cfg, update):
    return cfg.get('job_namespace', 'clean1')+f'_warm_u{int(update):03}'


def ident(cfg, arm, update):
    """The wave ident of ``arm``'s update: the original one unless that update has a migration receipt."""
    return base_ident(cfg, update)+suffix(cfg['run_root'], arm, update)


def seed_ident(wave_ident):
    """The ident a wave's noise seeds derive from: the UNTAGGED one (same CRN as the original update).
    Unchanged when the variable is absent or the ident carries no tag."""
    if os.environ.get(ENV) is None:
        return wave_ident
    tag = env_tag()
    return re.sub(r'(_warm_u[0-9]{3})m'+tag+r'(?=_|$)', r'\1', wave_ident, count=1)


def update_marker(cfg, arm, update):
    """The sealed UPDATE's 'migration_resume' block (None = untagged update: no key, byte-identical)."""
    root = cfg['run_root']
    sfx = suffix(root, arm, update)
    if not sfx:
        return None
    tag = sfx[1:]
    receipt = _verified(receipt_path(root, arm, update, tag), arm, update, tag)
    return dict(rule=RULE, tag=tag, receipt=receipt_name(arm, update, tag),
                receipt_sha256=receipt['receipt_sha256'], namespace=base_ident(cfg, update)+sfx)


# ---------------------------------------------------------------- G4 (B = R)

def g4_suffix(cfg, update):
    """The G4 input/envelope suffix of an update, common to B and R. '' when neither arm has a receipt."""
    root = cfg['run_root']
    tags = set(receipt_tags(root, 'B', update)) | set(receipt_tags(root, 'R', update))
    if not tags:
        return ''
    if all((Path(root)/f'{a}_UPDATE_{int(update):03}.json').exists() for a in ('B', 'R')):
        out = suffix(root, 'B', update)
        if suffix(root, 'R', update) != out:
            raise ValueError(f'B and R completed G4 update {update} under different migration tags (fail closed)')
        return out
    tag = env_tag()
    if tag is None or tag not in tags:
        raise ValueError(f'G4 update {update} has migration receipts {sorted(tags)}; {ENV}={tag} is not one of '
                         'them (fail closed)')
    return 'm'+tag


def check_pair(cfg, update):
    """After both G4 inputs exist: B and R re-attempt the update under the SAME tag (fail closed)."""
    sfx = g4_suffix(cfg, update)
    if sfx and not (suffix(cfg['run_root'], 'B', update) == suffix(cfg['run_root'], 'R', update) == sfx):
        raise ValueError(f'G4 update {update}: B and R are not re-attempting under the same migration tag')
    return sfx


def plan_marker(plan):
    marker = plan.get('migration_resume')
    if marker is None:
        return None
    if (not isinstance(marker, dict) or set(marker) != _MARKER_KEYS or marker['rule'] != G4_RULE
            or not isinstance(marker['tag'], str) or not _TAG.fullmatch(marker['tag'])):
        raise ValueError('malformed G4 migration marker (fail closed)')
    return marker


def mark_plan(plan, marker):
    """Bind the marker INTO the plan digest (the envelope carries it to R)."""
    from . import sparse_synergy_credit as credit
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    body['migration_resume'] = marker
    return dict(body, plan_sha256=credit.digest(body))


def chained_before(plan, state, last):
    """The reservation_before a ledger plan must carry. Unmarked plans: ``state`` (the original rule). A marked
    re-attempt that supersedes an abandoned plan of its update directly follows it with last_update rewound."""
    marker = plan_marker(plan)
    if marker is None or marker['supersedes_plan_sha256'] is None:
        if marker is not None and marker['rewound_last_update'] is not None:
            raise ValueError('G4 migration marker rewinds without a superseded plan (fail closed)')
        return state
    if (last is None or last['plan_sha256'] != marker['supersedes_plan_sha256'] or last['update'] != plan['update']
            or state['last_update'] != plan['update']
            or marker['rewound_last_update'] != last['reservation_before']['last_update']):
        raise ValueError('G4 migration re-attempt does not directly follow the abandoned plan it supersedes')
    return dict(state, last_update=marker['rewound_last_update'])


def is_attempt(plan, update, tag):
    """Whether a ledger plan is THIS attempt's reservation (tag None: the original rule, any plan of the update)."""
    if plan['update'] != update:
        return False
    if tag is None:
        return True
    marker = plan_marker(plan)
    return marker is not None and marker['tag'] == tag


def reattempt(cfg, update, tag, state, last):
    """(reservation_before, marker) of a NEW tagged G4 reservation: chained after the receipted abandoned plan of
    the update, if any; the abandoned reservation stays spent."""
    root = cfg['run_root']
    receipt = _verified(receipt_path(root, 'B', update, tag), 'B', update, tag)
    listed = [p['plan_sha256'] for p in receipt['abandoned']['g4_plans'] or ()]
    if state['last_update'] == update:
        if last is None or last['update'] != update or not listed or listed[-1] != last['plan_sha256']:
            raise ValueError('the abandoned G4 plan of this update is not the receipted one (fail closed)')
        before = dict(state, last_update=last['reservation_before']['last_update'])
        superseded, rewound = last['plan_sha256'], before['last_update']
    elif state['last_update'] < update and not listed:
        before, superseded, rewound = state, None, None
    else:
        raise ValueError('G4 ledger state does not match the migration receipt (fail closed)')
    return before, dict(rule=G4_RULE, tag=tag, receipt=receipt_name('B', update, tag),
                        receipt_sha256=receipt['receipt_sha256'], supersedes_plan_sha256=superseded,
                        rewound_last_update=rewound)


# ---------------------------------------------------------------- receipt (arm owner, before the re-attempt)

def _publish_new(path, value):
    """Atomic publication that refuses an existing target (launch_training.write's contract)."""
    path = Path(path)
    temporary = path.with_name('.'+path.name+f'.writer{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _ledger(cfg):
    from .storage import ledger
    return ledger(cfg.get('accounting_root', cfg['run_root']))


def _pattern(namespace, arm):
    # normal / in-run replacement / resume replacement:  <ns>_<arm>_<seed>
    # practice slot <ns>_pNN_<arm>_<seed>; practice replacement <ns>_practice_<arm>_<seed>
    # matched practice A references charged by B/R:       <ns>_<arm>_practiceref_paNN_A_<seed> (and replacements)
    return re.compile(re.escape(namespace)+r'_(?:p[0-9]{2}_|practice_)?'+re.escape(arm)+'_')


def _job_state(root, ident, row):
    path = Path(root)/'jobs'/ident
    state = 'RESULT' if (path/'RESULT.json').exists() else 'CLAIM' if (path/'CLAIM.json').exists() else 'none'
    return dict(id=ident, ledger=row is not None, phase=None if row is None else row['phase'],
                charged_sim_s=0 if row is None else row['sim_s'], state=state,
                episode_failed=(path/'EPISODE_FAILED.json').exists(), close=(path/'CLOSE.json').exists())


def _namespace_jobs(root, rows, namespace, arm):
    pattern = _pattern(namespace, arm)
    by_id = {r['id']: r for r in rows if pattern.match(r['id'])}
    jobs = Path(root)/'jobs'
    ids = set(by_id) | {p.name for p in jobs.glob(namespace+'_*') if pattern.match(p.name)}
    return [_job_state(root, i, by_id.get(i)) for i in sorted(ids)]


def _g4_plans(root, rows, update, tag):
    """Abandoned G4 reservations (B's ledger batches) of the update: every plan of it not of this tag."""
    by_id = {r['id']: r for r in rows}
    out = []
    for row in rows:
        plan = row.get('g4_plan')
        if plan is None or plan['update'] != update:
            continue
        marker = plan_marker(plan)
        if marker is not None and marker['tag'] == tag:
            raise ValueError(f'a G4 plan of update {update} already uses tag {tag} without its receipt (fail closed)')
        ids = (['g4_'+j['job_sha256'] for j in plan['jobs']] if plan['jobs']
               else ['g4_empty_'+plan['plan_sha256']])
        if row['id'] != ids[0] or any(i not in by_id for i in ids):
            raise ValueError('abandoned G4 plan batch is not in the ledger as reserved (fail closed)')
        jobs = [_job_state(root, i, by_id[i]) for i in ids]
        sfx = '' if marker is None else 'm'+marker['tag']
        out.append(dict(plan_sha256=plan['plan_sha256'], tag=None if marker is None else marker['tag'],
                        quote_sim_s=plan['quote_sim_s'], starts=len(plan['starts']), jobs=jobs,
                        charged_sim_s=sum(j['charged_sim_s'] for j in jobs),
                        envelope_present=(Path(root)/'g4'/f'ENVELOPE_{update:03}{sfx}.json').exists()))
    return out


def _check_a_references(cfg, update):
    """B/R safety pairs with A's same-update RESULT payloads (accel_storage RESULT.json -> RESULT.json.gz). When A
    completed the update before the reclaim, those payloads must have been restored (they are not in the
    essentials backup) - fail closed BEFORE any charge."""
    root = Path(cfg['run_root'])
    if not (root/f'A_UPDATE_{update:03}.json').exists():
        return None                                       # A re-attempts this update itself: fresh payloads
    a_ident = ident(cfg, 'A', update)
    missing = []
    composite = (root/'SAFETY_RULE_0957.json').exists()
    cache = root/'safety_diagnosis_0957'/'ttc_cache'
    for world in cfg['worlds_by_update'][str(update)]:
        job = root/'jobs'/f'{a_ident}_A_{world["seed"]}'
        if not (job/'RESULT.json').exists():
            continue                                      # unchanged rule: the safety pairing itself fails closed
        marker = _read(job/'RESULT.json')
        if isinstance(marker, dict) and marker.get('encoding') and not (job/str(marker.get('file'))).exists():
            missing.append(f'jobs/{job.name}/{marker.get("file")}')
        if composite and not (job/'observable.jsonl.gz').exists() and not any(cache.glob(job.name+'_*.json')):
            missing.append(f'jobs/{job.name}/observable.jsonl.gz')
    if missing:
        raise ValueError('the B/R re-attempt needs the completed A update payloads, missing from this root: '
                         + ', '.join(missing[:6])+(' ...' if len(missing) > 6 else '')+' (fail closed)')
    return a_ident


def prepare(cfg, arm, update, last):
    """Arm owner, at its FIRST not-completed update, before that update starts. Variable absent: refuses a
    receipted in-flight update, else returns None (no write). Set: write (or reuse) the sealed receipt."""
    from .protocol import digest
    root = Path(cfg['run_root'])
    tag = env_tag()
    existing = receipt_tags(root, arm, update)
    if tag is None:
        if existing:
            raise ValueError(f'{arm} update {update} is a receipted migration re-attempt {existing}; resume it with '
                             f'{ENV} set (fail closed)')
        return None
    path = receipt_path(root, arm, update, tag)
    if tag in existing:                                   # idempotent: a second resume reuses the receipt
        receipt = _verified(path, arm, update, tag)
        if receipt['config_sha256'] != digest(cfg):
            raise ValueError('migration receipt was written for another runtime config (fail closed)')
        return receipt
    if (root/f'{arm}_UPDATE_{update:03}.json').exists():
        raise ValueError('a completed update is adopted, never re-attempted')
    for name in (f'{arm}_checkpoint_{update:03}.pt', f'{arm}_track2_{update:03}.pt'):
        if (root/name).exists():
            raise ValueError(f'{name} exists without {arm}_UPDATE_{update:03}.json; the re-attempt could not save '
                             '(fail closed)')
    rows = _ledger(cfg)
    base = base_ident(cfg, update)
    for later in range(update+1, last+1):                # exactly one not-completed update may carry attempts
        name = base_ident(cfg, later)
        if ((root/f'{arm}_UPDATE_{later:03}.json').exists() or receipt_tags(root, arm, later)
                or (root/f'{arm}_checkpoint_{later:03}.pt').exists()
                or any((root/'updates').glob(f'{arm}_{later:03}_STARTED*.json'))
                or any(_pattern(name, arm).match(r['id']) for r in rows)
                or (arm == 'B' and any(r.get('g4_plan', {}).get('update') == later for r in rows))):
            raise ValueError(f'{arm} update {later} has an attempt while update {update} is not completed '
                             '(fail closed)')
    if arm in ('B', 'R'):
        from .g4_runtime import enabled
        partner = 'R' if arm == 'B' else 'B'
        if enabled(cfg, update) and (root/f'{partner}_UPDATE_{update:03}.json').exists():
            raise ValueError(f'{partner} completed G4-paired update {update}; its G4 input/envelope cannot be '
                             f'recreated for a re-attempt of {arm} (fail closed)')
    a_reference = _check_a_references(cfg, update) if arm in ('B', 'R') else None
    if _namespace_jobs(root, rows, base+'m'+tag, arm):
        raise ValueError(f'namespace {base}m{tag} already has {arm} attempts without a receipt (fail closed)')
    previous = []
    for other in existing:
        jobs = _namespace_jobs(root, rows, base+'m'+other, arm)
        if jobs and all(j['state'] == 'RESULT' for j in jobs):
            raise ValueError(f'the {other} re-attempt of {arm} update {update} completed its collection; resume '
                             f'with {ENV}={other} (fail closed)')
        previous.append(dict(tag=other, receipt=receipt_name(arm, update, other),
                             receipt_sha256=_verified(receipt_path(root, arm, update, other), arm, update,
                                                      other)['receipt_sha256']))
    namespaces = [base]+[base+'m'+p['tag'] for p in previous]
    jobs = [j for ns in namespaces for j in _namespace_jobs(root, rows, ns, arm)]
    g4_plans = _g4_plans(root, rows, update, tag) if arm == 'B' else None
    records = sorted(p.relative_to(root).as_posix() for p in
                     [*(root/'updates').glob(f'{arm}_{update:03}_STARTED*.json'),
                      *(root/'owners'/'children').glob(f'{arm}_u{update:03}*.json')])
    from .exact_grade import _seal
    receipt = _seal(dict(schema=SCHEMA, rule=RULE, reason=REASON, tag=tag, arm=arm, update=update,
        config_sha256=digest(cfg), unix=time.time(), pid=os.getpid(), noise_seed_rule=NOISE_RULE,
        namespace=dict(reattempt=base+'m'+tag, abandoned=namespaces), a_reference_namespace=a_reference,
        abandoned=dict(jobs=jobs, charged_sim_s=sum(j['charged_sim_s'] for j in jobs),
                       attempted=sum(j['ledger'] or j['state'] != 'none' for j in jobs),
                       g4_plans=g4_plans,
                       g4_charged_sim_s=None if g4_plans is None else sum(p['charged_sim_s'] for p in g4_plans),
                       g4_note=None if arm == 'B' else 'G4 reservations are made and receipted by B',
                       process_records=records),
        previous_receipts=previous), 'receipt_sha256')
    _publish_new(path, receipt)
    return receipt
