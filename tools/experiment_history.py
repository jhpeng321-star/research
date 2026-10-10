"""Read-only lookup, consistency validation and scoped source verification for the
experiment-history ledger.

Master ledger: protocols/experiment_history.json (schema experiment_history/2.x).
Search/card/closest/validate are metadata-only: they never open outcome contents.
`verify-sources` is the ONE exception, explicitly scoped: it reads only the sources
registered in the ledger's source_verification.allowed_sources, compares stored
costs/differences/labels against those source fields with full coverage accounting
(expected / performed / skipped-with-reason / mismatched per source, comparison and
metric), and writes protocols/source_verification_report.json. Sealed outcomes are
never traversed by any mode. A pinned checksum establishes artifact identity only,
not scientific correctness.

Evidence-status vocabulary (ledger conventions.evidence_status):
  consistency_verified | source_values_verified | integrity_receipt_verified |
  report_only_indexed | unknown_or_pending

Usage:
  python tools/experiment_history.py search [--family F] [--kind K] [--status S]
                                           [--seed 4107] [--metric braking] [--text "..."]
  python tools/experiment_history.py card TH-07
  python tools/experiment_history.py closest --text "..." [-n 5]
  python tools/experiment_history.py open-questions | pending
  python tools/experiment_history.py summary [--write | --check]
  python tools/experiment_history.py validate            (ledger consistency validation)
  python tools/experiment_history.py verify-sources      (registered sources only)
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, 'protocols', 'experiment_history.json')
OQ = os.path.join(ROOT, 'docs', 'OPEN_QUESTIONS.md')
REPORT = os.path.join(ROOT, 'protocols', 'source_verification_report.json')
SUMMARY = os.path.join(ROOT, 'docs', 'RESULTS_SUMMARY.md')
SCHEMA_PREFIX = 'experiment_history/2'

STOPWORDS = {'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'and', 'or', 'is',
             'are', 'was', 'were', 'we', 'our', 'it', 'its', 'with', 'from', 'at',
             'be', 'been', 'do', 'does', 'did', 'not', 'no', 'yes', 'what', 'which',
             'how', 'than', 'this', 'that', 'these', 'those', 'have', 'has', 'had'}
PENDING_TRIGGER_TERMS = ('speed', 'continuation', 'observation', 'raise', 'lower',
                         'target', 'adjust', 'ttc', 'warning', 'maintained', 'fusion')
METRIC_ORDER = ('system_seconds_per_request', 'braking_seconds_per_request',
                'effort_per_request', 'ttc_seconds_per_request',
                'source_wait_seconds_per_request')
METRIC_SHORT = {'system_seconds_per_request': 'service', 'braking_seconds_per_request': 'brake',
                'effort_per_request': 'effort', 'ttc_seconds_per_request': 'ttc',
                'source_wait_seconds_per_request': 'wait'}


def load():
    with open(LEDGER, encoding='utf-8') as f:
        return json.load(f)


PATH_MAP = os.path.join(ROOT, 'archive', 'circles_era', 'PATH_MAP.json')
_relocated = None
_relocated_dirs = []


def _path_maps():
    """MIG-03 map first (original precedence), then every other archive/*/PATH_MAP.json."""
    others = sorted(glob.glob(os.path.join(ROOT, 'archive', '*', 'PATH_MAP.json')))
    return [PATH_MAP] + [p for p in others if os.path.normcase(p) != os.path.normcase(PATH_MAP)]


def relocated(rel):
    """Map a frozen ledger path to its archive location (MIG-03 and later moves), or None."""
    global _relocated
    if _relocated is None:
        _relocated = {}
        for path in _path_maps():
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8') as f:
                m = json.load(f)
            for row in m.get('files', []) + m.get('directories', []):
                _relocated.setdefault(row['old'], row['new'])
            for row in m.get('directories', []):
                _relocated_dirs.append((row['old'].strip('/'), row['new'].strip('/')))
    key = rel.replace('\\', '/').strip('/')
    hit = _relocated.get(key)
    if hit is None:
        for old_dir, new_dir in _relocated_dirs:   # a path inside a relocated directory
            if key.startswith(old_dir + '/'):
                return new_dir + key[len(old_dir):]
    return hit


#: WM-55 drive migration: recorded paths keep their HISTORICAL legacy-drive
#: text (records are never rewritten); they are OPENED through the live roots.
#: Longest legacy root first so the backups root wins over research-dense.
_LEGACY_ROOTS = (
    ('d:/research-dense-backups', 'RD_BACKUPS_ROOT', 'E:/research-dense-backups'),
    ('d:/research-dense', 'RESEARCH_DENSE_ROOT', 'E:/research-dense'),
    ('d:/rd-wt', 'RD_WT_ROOT', 'E:/rd-wt'),
)


def _map_legacy_root(rel):
    text = str(rel).replace('\\', '/')
    low = text.lower()
    for legacy, env_key, default in _LEGACY_ROOTS:
        if low == legacy or low.startswith(legacy + '/'):
            return os.environ.get(env_key, default) + text[len(legacy):]
    return rel


def resolve(rel):
    rel = _map_legacy_root(rel)
    path = os.path.normpath(os.path.join(ROOT, rel.replace('/', os.sep)))
    if not os.path.exists(path):
        moved = relocated(rel)
        if moved:
            return os.path.normpath(os.path.join(ROOT, moved.replace('/', os.sep)))
    return path


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def coverage_banner(d):
    cov = d.get('coverage', {})
    lines = ['coverage cutoff: %s | last reviewed: %s' % (cov.get('cutoff'), cov.get('last_reviewed'))]
    for b in cov.get('excluded_and_pending', []):
        lines.append('EXCLUDED/PENDING: %s [%s]' % (b.get('block'), b.get('review_status', '')))
    return lines


def pending_hits(d, text):
    t = (text or '').lower()
    if not any(term in t for term in PENDING_TRIGGER_TERMS):
        return []
    hits = []
    for p in d.get('pending_updates', []):
        for item in p.get('remaining_pending', []) or []:
            hits.append('%s remaining pending: %s (%s)' % (p['pending_id'], item.get('item'), item.get('why', '')))
    for b in d.get('coverage', {}).get('excluded_and_pending', []):
        for s in b.get('still_pending', []) or []:
            hits.append('still pending: %s' % s)
    for a in (d.get('artifact_review', {}) or {}).get('artifacts', []):
        if a.get('status_overall') in ('correction_pending', 'unknown_or_pending'):
            hits.append('artifact under correction: %s (%s)' % (a.get('path'), a.get('status_overall')))
    return hits


def evidence_line(e):
    st = e.get('structured') or {}
    return st.get('evidence_status') or e.get('evidence_status') or 'report_only_indexed'


def _entry_text(e):
    st = e.get('structured') or {}
    return ' '.join(str(x) for x in (e.get('name'), e.get('question'), e.get('results'),
                                     e.get('controller'), e.get('criteria_verdict'),
                                     st.get('scientific_verdict'), st.get('action'),
                                     st.get('controller_profile'), e.get('family', ''))).lower()


# ---------------------------------------------------------------- search / card
def cmd_search(a):
    d = load()
    for line in coverage_banner(d):
        print(line)
    hits = []
    for e in d['entries']:
        st = e.get('structured') or {}
        if a.family and e.get('family') != a.family:
            continue
        if a.kind and st.get('record_kind', 'experiment') != a.kind:
            continue
        if a.status and e.get('status') != a.status:
            continue
        blob = json.dumps(e, ensure_ascii=False).lower()
        if a.seed and a.seed.lower() not in blob:
            continue
        if a.metric and a.metric.lower() not in blob:
            continue
        if a.text:
            terms = [t for t in a.text.lower().split() if t not in STOPWORDS]
            if not any(t in blob for t in terms):
                continue
        hits.append(e)
    for e in hits:
        st = e.get('structured') or {}
        print('%-8s [%s/%s/%s] %s' % (e['id'], e['family'], e.get('status'),
                                      evidence_line(e), e.get('name')))
        if st.get('scientific_verdict'):
            print('         verdict: %s' % st['scientific_verdict'][:150])
    print('(%d entries)' % len(hits))
    for h in pending_hits(d, ' '.join(x for x in (a.text, a.metric) if x)):
        print('NOTICE:', h)


def cmd_card(a):
    d = load()
    e = next((x for x in d['entries'] if x['id'] == a.id), None)
    if not e:
        sys.exit('no such entry: %s' % a.id)
    st = e.get('structured') or {}
    for line in coverage_banner(d):
        print(line)
    for h in pending_hits(d, json.dumps(e)[:4000]):
        print('NOTICE (relevant pending work):', h)
    print('== %s | %s | %s' % (e['id'], e['family'], e.get('date')))
    print('name   :', e.get('name'))
    print('status :', e.get('status'), '| kind:', st.get('record_kind', 'experiment (prose-only)'),
          '| execution:', st.get('execution_status', 'n/a'),
          '| evidence:', st.get('evidence_validity', 'n/a'),
          '| review:', evidence_line(e))
    print('question:', e.get('question'))
    print('verdict:', st.get('scientific_verdict') or e.get('criteria_verdict'))
    if st.get('controller_profile'):
        print('controller:', st['controller_profile'])
    if st.get('comparison_kind'):
        print('frame  :', st['comparison_kind'])
    if st.get('decision_time_s') is not None:
        print('decision timing:', st['decision_time_s'])
    if st.get('seed_groups'):
        print('seeds  :', st.get('seed_groups'), '| exposure:', st.get('exposure_role'))
    if st.get('changed_vars'):
        print('changed:', st['changed_vars'])
    if st.get('requested_target_and_phases') or st.get('delivered_target_and_phases'):
        print('requested:', st.get('requested_target_and_phases'))
        print('delivered:', st.get('delivered_target_and_phases'))
    for c in st.get('confounders', []) or []:
        print('confound:', c)
    for o in st.get('outcomes') or []:
        vals = o.get('outcomes') or {}
        head = '  %s [%s vs %s] label=%s pop=%s' % (
            o.get('comparison_id') or o.get('job_id'),
            o.get('treatment_id'), o.get('reference_id'), o.get('label'), o.get('population'))
        j = o.get('joint_non_deterioration')
        if j is not None:
            head += ' strict_joint_nonworsening=%s' % j
        print(head)
        cd = ', '.join('%s %+.6f' % (METRIC_SHORT.get(m, m), v['cost_difference'])
                       for m, v in vals.items()
                       if isinstance(v, dict) and 'cost_difference' in v)
        if cd:
            print('     cost differences (treatment-reference; negative=improvement):', cd)
        if o.get('guardrail_failures') is not None:
            print('     guardrail failures:', o['guardrail_failures'])
        if o.get('versus_keep_24'):
            print('     vs keep-24:', o['versus_keep_24'].get('cost_difference_new_minus_keep'))
        if o.get('related_prior_outcome'):
            rpo = o['related_prior_outcome']
            print('     related prior outcome: %s (%s) - %s' % (
                rpo.get('entry_id'), rpo.get('comparison_id'), rpo.get('note', '')[:120]))
        if o.get('note'):
            print('     note:', o['note'])
    print('paths:')
    for p in e.get('paths', []) or []:
        print('   ', p.split(' (')[0])
    for p in st.get('artifacts', []) or []:
        print('   [structured]', p)


def cmd_closest(a):
    d = load()
    print('Lexical retrieval over ledger text - NOT an automated novelty determination; '
          'a proposal sharing an action axis/controller/seeds may still answer a new '
          'question if it isolates a different variable or uses a different comparator/frame.')
    for line in coverage_banner(d):
        print(line)
    terms = [t for t in a.text.lower().split() if t not in STOPWORDS]
    scored = []
    for e in d['entries']:
        blob = _entry_text(e)
        score = sum(blob.count(t) for t in terms)
        if score:
            scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    for score, e in scored[:a.n]:
        st = e.get('structured') or {}
        print('%4d  %-8s [%s|%s] %s' % (score, e['id'], e['family'], evidence_line(e), e.get('name')))
        print('      controller:', (st.get('controller_profile') or e.get('controller') or '-')[:110])
        print('      frame/timing:', st.get('comparison_kind') or '-', '/', st.get('decision_time_s') or '-')
        print('      seeds:', (st.get('seed_groups') or e.get('seeds_splits') or '-')[:90])
        print('      changed:', (st.get('changed_vars') or '-')[:110])
        print('      comparator:', (st.get('reference_id') or e.get('comparators') or '-')[:90])
        for cf in (st.get('confounders') or [])[:2]:
            print('      confound:', cf[:110])
        print('      verdict:', (st.get('scientific_verdict') or e.get('criteria_verdict') or '')[:150])
    if not scored:
        print('no lexical overlap; widen the query')
    for h in pending_hits(d, a.text):
        print('NOTICE:', h)


def cmd_open_questions(a):
    with open(OQ, encoding='utf-8') as f:
        for line in f:
            if line.startswith('## OQ-') or line.startswith('*Any new experiment'):
                print(line.rstrip())


def cmd_pending(a):
    d = load()
    for p in d.get('pending_updates', []):
        print('%s: %s [%s]' % (p['pending_id'], p['block'], p['status']))
        for it in p.get('resolved_items', []) or []:
            print('   resolved:', it.get('item'), '->', (it.get('resolution') or '')[:100])
        for it in p.get('remaining_pending', []) or []:
            print('   PENDING :', it.get('item'), '-', it.get('why', ''))
    for b in d.get('coverage', {}).get('excluded_and_pending', []):
        print('excluded-ish:', b['block'], '-', b.get('review_status', ''))
    ar = (d.get('artifact_review', {}) or {}).get('artifacts', [])
    for art in ar:
        if art.get('status_overall') != 'current':
            print('artifact [%s]: %s (%s)' % (art.get('status_overall'), art.get('path'),
                                              '; '.join(art.get('defects', []) or [])[:120]))


# ---------------------------------------------------------------- summary
def _frame_of(d, comparison_id):
    for c in d.get('comparisons', []):
        if c.get('comparison_id') == comparison_id:
            return c.get('comparison_kind', 'unrecorded frame')
    return 'unrecorded frame'


def build_summary_text(d, ledger_sha):
    lines = ['# Current results summary (generated)', '',
             'Generated by `tools/experiment_history.py summary --write` from structured '
             'ledger records. Ledger: schema %s, sha256 %s. Sign convention: '
             '**cost difference = treatment - reference; negative = improvement.** '
             'Computed facts only - manual interpretation lives in '
             'docs/EXPERIMENT_HISTORY.md and docs/OPEN_QUESTIONS.md. Fields not present '
             'in the structured record are omitted (not zero). Frames: '
             'entry_vs_policy / continuation_* / characterization - see the ledger '
             'conventions. This file is mechanical and regenerable via `summary --check`.'
             % (d.get('schema'), ledger_sha[:16] + '...'), '']
    for e in d['entries']:
        outs = (e.get('structured') or {}).get('outcomes') or []
        if not outs:
            continue
        lines.append('## %s - %s' % (e['id'], e['name']))
        lines.append('review: %s' % evidence_line(e))
        for o in outs:
            cid = o.get('comparison_id') or o.get('job_id')
            frame = _frame_of(d, cid)
            pop = o.get('population')
            pop_s = ('%d requested vehicles' % pop) if pop else 'population not recorded'
            lines.append('- **%s** | frame: %s | %s vs %s | %s'
                         % (cid, frame, o.get('treatment_id'), o.get('reference_id'), pop_s))
            label = o.get('label') or (o.get('benefit') or {}).get('label')
            if label:
                line = '  - original label: **%s**' % label
                gf = o.get('guardrail_failures') or (o.get('benefit') or {}).get('violated_guardrails')
                if gf:
                    line += ' | guardrail failures: %s' % ', '.join(gf)
                else:
                    line += ' | guardrail failures: none'
                lines.append(line)
            vals = o.get('outcomes') or {}
            for m in METRIC_ORDER:
                v = vals.get(m)
                if not (isinstance(v, dict) and 'cost_difference' in v):
                    continue
                unit = v.get('unit', 'unit not recorded')
                denom = v.get('normalization', '')
                lines.append('  - %s (%s%s): treatment %+.6f vs reference %+.6f -> '
                             'cost difference **%+.6f**%s'
                             % (m, unit, ('; ' + denom) if denom else '',
                                v.get('cost_treatment', float('nan')),
                                v.get('cost_reference', float('nan')),
                                v['cost_difference'],
                                '  [source: %s]' % o['verified_against']
                                if o.get('verified_against') and m == METRIC_ORDER[0] else ''))
            if o.get('versus_keep_24'):
                lines.append('  - vs keep-24 (new minus keep, per request): %s'
                             % o['versus_keep_24'].get('cost_difference_new_minus_keep'))
            if o.get('collisions_treatment') is not None:
                lines.append('  - collision involvements: treatment %s vs reference %s'
                             % (o.get('collisions_treatment'), o.get('collisions_reference', 'not recorded')))
            if o.get('related_prior_outcome'):
                rpo = o['related_prior_outcome']
                lines.append('  - related prior outcome: %s (%s) - %s'
                             % (rpo.get('entry_id'), rpo.get('comparison_id'),
                                rpo.get('note', '')[:200]))
        lines.append('')
    return '\n'.join(lines) + '\n'


def cmd_summary(a):
    d = load()
    ledger_sha = sha256_of(LEDGER)
    text = build_summary_text(d, ledger_sha)
    if a.write:
        with open(SUMMARY, 'w', encoding='utf-8') as f:
            f.write(text)
        print('written: %s' % os.path.relpath(SUMMARY, ROOT))
    elif a.check:
        if not os.path.exists(SUMMARY):
            print('STALE: %s does not exist; run summary --write' % os.path.relpath(SUMMARY, ROOT))
            sys.exit(1)
        with open(SUMMARY, encoding='utf-8') as f:
            current = f.read()
        if current == text:
            print('CURRENT: %s matches the structured ledger records '
                  '(ledger sha %s...)' % (os.path.relpath(SUMMARY, ROOT), ledger_sha[:12]))
        else:
            print('STALE: %s differs from regenerated content; run summary --write'
                  % os.path.relpath(SUMMARY, ROOT))
            sys.exit(1)
    else:
        print(text)


# ---------------------------------------------------------------- validation
def _iter_paths(entry):
    for p in entry.get('paths', []) or []:
        yield p.split(' (')[0], 'entries.paths'
    st = entry.get('structured') or {}
    for p in st.get('artifacts', []) or []:
        yield p, 'structured.artifacts'
    ns = st.get('numbers_source')
    if ns:
        yield ns, 'structured.numbers_source'


DELETED = os.path.join(ROOT, 'protocols', 'deleted_evidence.json')
_deleted = None


def recorded_deletion(rel):
    """True if rel lies under evidence recorded as permanently deleted
    (MIG-04 incident or an owner-directed retirement recorded in
    protocols/deleted_evidence.json; see that file's per-entry content)."""
    global _deleted
    if _deleted is None:
        _deleted = ([], [])
        if os.path.exists(DELETED):
            with open(DELETED, encoding='utf-8') as f:
                m = json.load(f)
            _deleted = ([r['path'] for r in m.get('fully_deleted_roots', [])],
                        [r['path'] for r in m.get('partially_deleted_dirs', [])])
    key = (rel or '').split(' (')[0].replace('\\', '/').strip('/')
    under = lambda root: key == root or key.startswith(root + '/')  # noqa: E731
    return any(under(r) for r in _deleted[0]) or any(
        under(r) and not os.path.exists(resolve(key)) for r in _deleted[1])


OFFLOADED = os.path.join(ROOT, 'protocols', 'offloaded_paths.json')
_offloaded = None


def offloaded(rel):
    """Drive location if rel lies under evidence that an owner-run clean-up moved
    to Google Drive after a live size+MD5 check (protocols/offloaded_paths.json;
    restore with tools/fetch_data.py / `python -m wms fetch`), else None. Unlike a
    recorded deletion the bytes still exist; verify-sources still needs its
    registered source files locally."""
    global _offloaded
    if _offloaded is None:
        _offloaded = []
        if os.path.exists(OFFLOADED):
            with open(OFFLOADED, encoding='utf-8') as f:
                m = json.load(f)
            for r in m.get('entries', []):
                _offloaded.append((os.path.normcase(os.path.normpath(r['local'])), r['drive']))
    key = (rel or '').split(' (')[0].strip()
    if not key:
        return None
    path = os.path.normcase(resolve(key))
    for root, drive in _offloaded:
        if path == root or path.startswith(root + os.sep):
            return drive
    return None


def known_missing(d, entry_id, rel):
    for m in (d.get('coverage', {}).get('known_missing_evidence', []) or []):
        if m.get('id') == entry_id and m.get('item', '').startswith(rel.split(' (')[0][:40]):
            return True
    return False


def validate(d):
    """Ledger CONSISTENCY validation: stored arithmetic (finite values enforced),
    relationships, path existence, reuse/alias semantics, budget scope, pending
    lifecycle. Informational reuse notes are separated from warnings; real
    discrepancies are never suppressed. Does NOT compare stored values against
    source-analysis contents (that is verify-sources)."""
    errors, warnings, infos, unknowns = [], [], [], []

    def err(m): errors.append(m)
    def warn(m): warnings.append(m)
    def info(m): infos.append(m)
    def unk(m): unknowns.append(m)

    if not str(d.get('schema', '')).startswith(SCHEMA_PREFIX):
        err('schema %r not supported (expected %s*)' % (d.get('schema'), SCHEMA_PREFIX))

    for key, idf in (('entries', 'entry'), ('runs', 'run'), ('comparisons', 'comparison'),
                     ('campaigns', 'campaign'), ('pending_updates', 'pending')):
        seen = set()
        for it in d.get(key, []):
            i = it.get(idf + '_id') or it.get('id')
            if i in seen:
                err('duplicate %s id: %s' % (idf, i))
            seen.add(i)

    entries = {e['id']: e for e in d['entries']}
    runs = {r['run_id']: r for r in d.get('runs', [])}
    campaigns = {c['campaign_id']: c for c in d.get('campaigns', [])}

    n_paths = 0
    for e in d['entries']:
        for rel, where in _iter_paths(e):
            n_paths += 1
            if not os.path.exists(resolve(rel)):
                if known_missing(d, e['id'], rel):
                    unk('documented missing path (%s/%s): %s' % (e['id'], where, rel))
                elif recorded_deletion(rel):
                    unk('recorded deletion (%s/%s): %s' % (e['id'], where, rel))
                elif offloaded(rel):
                    unk('offloaded to Drive (%s/%s): %s' % (e['id'], where, rel))
                else:
                    err('missing path (%s/%s): %s' % (e['id'], where, rel))
    for r in d.get('runs', []):
        n_paths += 1
        if not os.path.exists(resolve(r['path'])):
            if recorded_deletion(r['path']):
                unk('recorded deletion (%s): %s' % (r['run_id'], r['path']))
            elif offloaded(r['path']):
                unk('offloaded to Drive (%s): %s' % (r['run_id'], r['path']))
            else:
                err('missing run path (%s): %s' % (r['run_id'], r['path']))
    for p in d.get('pending_updates', []):
        for it in p.get('resolved_items', []) or []:
            for eid in it.get('entries', []) or []:
                if eid not in entries:
                    err('pending %s resolves to unknown entry %s' % (p['pending_id'], eid))
        for rel in p.get('authoritative_artifacts', []) or []:
            n_paths += 1
            if not os.path.exists(resolve(rel)):
                if recorded_deletion(rel):
                    unk('recorded deletion (%s): %s' % (p['pending_id'], rel))
                elif offloaded(rel):
                    unk('offloaded to Drive (%s): %s' % (p['pending_id'], rel))
                else:
                    err('missing pending-update artifact (%s): %s' % (p['pending_id'], rel))

    n_out = 0
    for e in d['entries']:
        st = e.get('structured') or {}
        for o in st.get('outcomes', []) or []:
            n_out += 1
            for m, v in (o.get('outcomes') or {}).items():
                if not isinstance(v, dict) or 'cost_difference' not in v:
                    continue
                if v.get('verified') is not True:
                    err('%s/%s: outcome not flagged verified' % (o.get('comparison_id'), m))
                for k in ('cost_treatment', 'cost_reference', 'cost_difference', 'improvement'):
                    if k in v and not _finite(v[k]):
                        err('%s/%s: non-finite value in %s (%r) - rejected'
                            % (o.get('comparison_id'), m, k, v[k]))
                if all(k in v and _finite(v[k]) for k in ('cost_treatment', 'cost_reference', 'cost_difference')):
                    if abs((v['cost_treatment'] - v['cost_reference']) - v['cost_difference']) > 1e-6:
                        err('%s/%s: cost_difference != treatment - reference' % (o.get('comparison_id'), m))
                if 'improvement' in v and _finite(v['improvement']) and _finite(v['cost_difference']):
                    if abs(v['improvement'] + v['cost_difference']) > 1e-6:
                        err('%s/%s: improvement is not the negative of cost_difference' % (o.get('comparison_id'), m))
            if not os.path.exists(resolve(o.get('verified_against', ''))):
                if recorded_deletion(o.get('verified_against', '')):
                    unk('recorded deletion (%s verified_against): %s'
                        % (o.get('comparison_id'), o.get('verified_against')))
                elif offloaded(o.get('verified_against', '')):
                    unk('offloaded to Drive (%s verified_against): %s'
                        % (o.get('comparison_id'), o.get('verified_against')))
                else:
                    err('%s: verified_against path missing: %s' % (o.get('comparison_id'), o.get('verified_against')))

    reused_classes = ('reused_run', 'trace_equivalent', 'exact_reuse', 'replay_verification',
                      'reused_reference', 'alias_of_canonical')
    for c in d.get('comparisons', []):
        if c['entry_id'] not in entries:
            err('comparison %s: unknown entry %s' % (c['comparison_id'], c['entry_id']))
        for side in ('treatment_run', 'reference_run'):
            if c[side] not in runs:
                err('comparison %s: unknown run %s' % (c['comparison_id'], c[side]))
        tcls = c.get('treatment_class', 'fresh')
        tr = runs.get(c['treatment_run'], {})
        if tcls in reused_classes and tr.get('actual_simulated_seconds') not in (0, None):
            err('comparison %s: reused-class treatment run %s has nonzero simulated seconds'
                % (c['comparison_id'], c['treatment_run']))
    # one physical sample may legitimately serve several comparisons when the frame
    # or the reference differs; identical duplicated comparisons remain a warning
    per_run = {}
    for c in d.get('comparisons', []):
        per_run.setdefault(c['treatment_run'], []).append(c)
    for rid, cs in per_run.items():
        if len(cs) < 2:
            continue
        keys = {(c.get('comparison_kind'), c.get('reference_run')) for c in cs}
        if len(keys) < len(cs):
            warn('treatment run %s appears in %d comparisons with duplicated frame/reference '
                 'pairs %s - possible double counting' % (rid, len(cs), sorted(map(str, keys))))
        else:
            info('verified physical run %s serves %d comparisons with distinct frames/references '
                 '(%s) - one physical sample, one execution charge; not independent evidence'
                 % (rid, len(cs), ', '.join(sorted({str(k[0]) for k in keys}))))
    for r in d.get('runs', []):
        if r['campaign'] not in campaigns:
            err('run %s: unknown campaign %s' % (r['run_id'], r['campaign']))
        if r.get('canonical_run'):
            can = runs.get(r['canonical_run'])
            if not can:
                err('run %s: canonical_run %s missing' % (r['run_id'], r['canonical_run']))
            else:
                if (r.get('actual_simulated_seconds') or 0) != 0:
                    err('alias %s carries nonzero simulated seconds (must be 0; canonical keeps the cost)' % r['run_id'])
                if (r.get('charged_seconds') or 0) != 0:
                    err('alias %s carries a nonzero charge (must be 0; canonical keeps the charge)' % r['run_id'])
                if not can.get('actual_simulated_seconds'):
                    warn('alias %s points at canonical %s with no recorded execution cost'
                         % (r['run_id'], r['canonical_run']))
    # double charging: two charged rows aliasing the same canonical, or a canonical
    # charged again under another id with the same physical path
    by_canonical = {}
    by_path = {}
    for r in d.get('runs', []):
        if (r.get('charged_seconds') or 0) > 0:
            if r.get('canonical_run'):
                by_canonical.setdefault(r['canonical_run'], []).append(r['run_id'])
            by_path.setdefault(r['path'], []).append(r['run_id'])
    for can, rids in by_canonical.items():
        err('double charging: runs %s all charge seconds while aliasing canonical %s' % (rids, can))
    for path, rids in by_path.items():
        if len(rids) > 1:
            err('double charging: %d run rows (%s) charge seconds for the same physical path %s'
                % (len(rids), rids, path))

    for cid, c in campaigns.items():
        comp = c.get('components') or {}
        if comp and c.get('actual_progress_s') is not None:
            defined = [v for v in comp.values() if v[0] is not None]
            tot = sum(v[0] for v in defined)
            unresolved = len(comp) - len(defined)
            if unresolved == 0:
                if abs(tot - c['actual_progress_s']) > 0.5:
                    err('campaign %s: components actual sum %s != declared actual %s' % (cid, tot, c['actual_progress_s']))
            else:
                warn('campaign %s: %d component(s) have unknown actual progress; partial sum %s vs declared %s'
                     % (cid, unresolved, tot, c['actual_progress_s']))
        cr = [r for r in d.get('runs', []) if r['campaign'] == cid]
        if cr and c.get('allocated_charged_s') is not None:
            tot = 0.0
            for r in cr:
                tot += r.get('charged_seconds') if r.get('charged_seconds') is not None \
                    else (r.get('allocated_horizon_s') or 0)
            if abs(tot - c['allocated_charged_s']) > 0.5:
                warn('campaign %s: run-level charge sum %s != campaign charged %s (check scope)'
                     % (cid, tot, c['allocated_charged_s']))

    for e in d['entries']:
        st = e.get('structured') or {}
        status = st.get('evidence_status') or e.get('evidence_status')
        if st.get('execution_status') == 'completed':
            if not e.get('paths') and not st.get('artifacts'):
                warn('%s: completed without artifact paths' % e['id'])
            if not (st.get('scientific_verdict') or e.get('criteria_verdict')):
                warn('%s: completed without a verdict field' % e['id'])
        if status == 'source_values_verified':
            allowed = [s for s in d.get('source_verification', {}).get('allowed_sources', [])
                       if e['id'] in (s.get('entries') or [])]
            if not allowed:
                err('%s: evidence_status source_values_verified but entry not in any registered allowed source' % e['id'])
            elif not os.path.exists(REPORT):
                warn('%s: source_values_verified but %s does not exist yet (run verify-sources)'
                     % (e['id'], os.path.basename(REPORT)))

    unk('sealed blocks are recorded by metadata only; consistency validation used existence checks (stat) exclusively '
        'and never opened outcome contents; source-value comparison happens only in verify-sources on registered allowed sources')
    return errors, warnings, infos, unknowns, n_paths, n_out


def cmd_validate(a):
    d = load()
    errors, warnings, infos, unknowns, n_paths, n_out = validate(d)
    print('LEDGER CONSISTENCY VALIDATION (stored arithmetic incl. finite-value enforcement, '
          'relationships, paths, reuse/alias, budget scope, pending lifecycle; NOT a '
          'source-value comparison - run verify-sources for that)')
    print('ledger: %s (schema %s)' % (LEDGER, d.get('schema')))
    print('entries=%d runs=%d comparisons=%d campaigns=%d pending=%d | paths checked=%d outcomes checked=%d'
          % (len(d['entries']), len(d.get('runs', [])), len(d.get('comparisons', [])),
             len(d.get('campaigns', [])), len(d.get('pending_updates', [])), n_paths, n_out))
    for title, items, mark in (('HARD ERRORS', errors, '!'), ('WARNINGS', warnings, '~'),
                               ('INFORMATIONAL REUSE NOTES', infos, 'i')):
        print('\n%s (%d):' % (title, len(items)))
        for m in items:
            print('  %s' % mark, m)
    print('\nDOCUMENTED UNKNOWNS (%d):' % len(unknowns))
    for m in unknowns:
        print('  .', m)
    sys.exit(1 if errors else 0)


# ---------------------------------------------------------------- source verification
def _find_row(rows, key_field, key):
    for r in rows:
        if r.get(key_field) == key:
            return r
    return None


#: per-metric sub-assertions verified against the source effect block
METRIC_SUBFIELDS = ('cost_treatment', 'cost_reference', 'cost_difference', 'improvement')


def _is_continuation_outcome(o):
    return (o.get('comparison_kind') == 'continuation_keep_vs_release'
            or any(isinstance(v, dict) and
                   str(v.get('source_field', '')).startswith('duration_contrast')
                   for v in (o.get('outcomes') or {}).values()))


def verify_sources(d):
    """Compare stored outcomes against REGISTERED allowed sources only, with explicit
    coverage accounting. Required fields missing from the source FAIL verification;
    intentional exclusions (with reason) make a source PARTIAL; NaN/infinity fail.
    A pinned sha256 establishes artifact identity, not scientific correctness."""
    reg = d.get('source_verification', {}).get('allowed_sources', [])
    if not reg:
        return None, ['no allowed sources registered'], 0, 0
    entries = {e['id']: e for e in d['entries']}
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    report = {'generated_at': now,
              'scope': 'registered allowed sources only (exposed development artifacts); '
                       'sealed outcomes never traversed',
              'convention': 'cost difference = treatment - reference; negative = improvement',
              'status_definitions': {'complete': 'all expected checks performed, zero mismatches',
                                     'partial': 'all performed checks passed; some checks '
                                                'intentionally excluded (reason recorded)',
                                     'failure': 'any mismatch, missing required source field, '
                                                'or non-finite value',
                                     'deleted': 'source recorded as permanently deleted '
                                                '(protocols/deleted_evidence.json); not re-verifiable'},
              'sources': []}
    all_mismatches = []
    expected_total = performed_total = 0

    for s in reg:
        rec = {'source_id': s['source_id'], 'path': s['path'],
               'sha256_registered': s.get('sha256_registered'), 'entries': s.get('entries', []),
               'field_map': s.get('field_map'),
               'expected_checks': 0, 'performed_checks': 0,
               'skipped_checks': [], 'mismatches': [], 'status': 'failure'}
        path = resolve(s['path'])
        if not os.path.exists(path) and recorded_deletion(s['path']):
            rec['skipped_checks'].append({'reason': 'source file deleted (MIG-04); stored values '
                                                    'stand as last verified', 'count': None})
            rec['status'] = 'deleted'
            report['sources'].append(rec)
            continue
        if not os.path.exists(path):
            rec['skipped_checks'].append({'reason': 'source file missing', 'count': None})
            rec['mismatches'].append('%s: source file missing' % s['source_id'])
            all_mismatches.append('%s: source file missing' % s['source_id'])
            report['sources'].append(rec)
            continue
        rec['sha256_current'] = sha256_of(path)
        rec['verified_at'] = now
        if s.get('sha256_registered') and rec['sha256_current'] != s['sha256_registered']:
            rec['mismatches'].append('%s: SOURCE CHANGED since registration (sha256 mismatch; '
                                     'identity only - re-verify contents)' % s['source_id'])
            all_mismatches.append('%s: SOURCE CHANGED since registration' % s['source_id'])
        try:
            with open(path, encoding='utf-8') as f:
                src = json.load(f)
        except Exception as ex:  # noqa: BLE001
            rec['mismatches'].append('%s: unreadable: %s' % (s['source_id'], ex))
            all_mismatches.append('%s: unreadable' % s['source_id'])
            report['sources'].append(rec)
            continue

        def expect(reason=None):
            nonlocal expected_total
            rec['expected_checks'] += 1
            expected_total += 1

        def performed():
            nonlocal performed_total
            rec['performed_checks'] += 1
            performed_total += 1

        def skip(eid_label, reason, intentional):
            rec['skipped_checks'].append({'reason': eid_label + ': ' + reason,
                                          'intentional': intentional})
            if not intentional:
                msg = '%s: REQUIRED CHECK SKIPPED (%s): %s' % (s['source_id'], eid_label, reason)
                rec['mismatches'].append(msg)
                all_mismatches.append(msg)

        def compare(eid, label, stored, source_val, tol=1e-6):
            expect()
            performed()
            if not (_finite(stored) and _finite(source_val)):
                msg = '%s/%s: non-finite value (stored %r, source %r)' % (eid, label, stored, source_val)
                rec['mismatches'].append(msg)
                all_mismatches.append(msg)
                return
            if abs(stored - source_val) > tol:
                msg = '%s/%s: stored %r != source %r' % (eid, label, stored, source_val)
                rec['mismatches'].append(msg)
                all_mismatches.append(msg)

        for eid in s.get('entries', []):
            st = entries.get(eid, {}).get('structured') or {}
            outcomes = st.get('outcomes', []) or []
            for o in outcomes:
                row = None
                if 'cases' in src:
                    row = _find_row(src['cases'], 'job_id', o.get('treatment_id'))
                elif 'paired_effects' in src:
                    ids = [x.get('comparison_id') for x in outcomes]
                    if o.get('comparison_id') in ids:
                        idx = ids.index(o.get('comparison_id'))
                        row = src['paired_effects'][idx] if idx < len(src['paired_effects']) else None
                elif 'results' in src:
                    row = _find_row(src['results'], 'job_id', o.get('treatment_id'))
                if row is None:
                    expect()
                    skip(o.get('treatment_id') or '?', 'source row not found (required)', intentional=False)
                    all_mismatches.append('%s/%s: row not found' % (s['source_id'], eid))
                    continue
                eff = row.get('effect') or row
                ct, cr = eff.get('costs_treatment') or {}, eff.get('costs_control') or {}
                imp_src = eff.get('improvements') or {}
                continuation = _is_continuation_outcome(o)

                # per-metric numeric checks
                for m, v in (o.get('outcomes') or {}).items():
                    if not isinstance(v, dict) or 'cost_difference' not in v:
                        continue
                    # structural: units and normalization must be recorded
                    expect()
                    if v.get('unit') and v.get('normalization'):
                        performed()
                    else:
                        skip(eid + '/' + m, 'unit/normalization not recorded (structural)', intentional=False)
                    if continuation:
                        # intentional exclusion: continuation deltas come from
                        # duration_contrast, not the row's entry-effect block
                        dc = (row.get('duration_contrast') or {}).get('cost_release_minus_cost_continue') or {}
                        if m in dc:
                            if _finite(dc[m]) and _finite(v['cost_difference']):
                                compare(eid, '%s (duration_contrast)' % m,
                                        v['cost_difference'], dc[m])
                            else:
                                expect()
                                performed()
                                skip(eid + '/' + m, 'non-finite duration_contrast value', intentional=False)
                        else:
                            expect()
                            skip(eid + '/' + m, 'duration_contrast lacks this metric '
                                                '(intentional: not required for continuation rows)', intentional=True)
                        continue
                    for sub in METRIC_SUBFIELDS:
                        if m not in ct or m not in cr or (sub == 'improvement' and m not in imp_src):
                            expect()
                            skip('%s/%s/%s' % (eid, m, sub), 'required source field missing (effect block)', intentional=False)
                            continue
                        if sub == 'cost_treatment':
                            compare(eid, '%s cost_treatment' % m, v.get('cost_treatment'), ct[m])
                        elif sub == 'cost_reference':
                            compare(eid, '%s cost_reference' % m, v.get('cost_reference'), cr[m])
                        elif sub == 'cost_difference':
                            compare(eid, '%s cost_difference' % m, v['cost_difference'], ct[m] - cr[m])
                        else:
                            compare(eid, '%s improvement' % m, v.get('improvement'), imp_src[m])

                # row-level verdict and safety/service evidence
                ben = row.get('benefit') or row.get('physical_label') or {}
                if isinstance(ben, dict) and ben:
                    if o.get('label') is not None:
                        expect()
                        performed()
                        if ben.get('label') != o['label']:
                            msg = '%s/%s: label %r != source %r' % (
                                eid, o.get('treatment_id'), o['label'], ben.get('label'))
                            rec['mismatches'].append(msg)
                            all_mismatches.append(msg)
                    if o.get('guardrail_failures') is not None:
                        expect()
                        performed()
                        if ben.get('violated_guardrails') != o['guardrail_failures']:
                            msg = '%s/%s: guardrail failures %r != source %r' % (
                                eid, o.get('treatment_id'), o['guardrail_failures'],
                                ben.get('violated_guardrails'))
                            rec['mismatches'].append(msg)
                            all_mismatches.append(msg)
                if o.get('population') is not None and eff.get('requested_population') is not None:
                    compare(eid, '%s population' % o.get('treatment_id'),
                            float(o['population']), float(eff['requested_population']))
                if o.get('collisions_treatment') is not None:
                    if eff.get('treatment_collision_involvements') is None:
                        expect()
                        skip(eid + '/' + (o.get('treatment_id') or '?'), 'source collision evidence missing', intentional=False)
                    else:
                        compare(eid, '%s collisions_treatment' % o.get('treatment_id'),
                                float(o['collisions_treatment']),
                                float(eff['treatment_collision_involvements']))
                if o.get('collisions_reference') is not None:
                    if eff.get('control_collision_involvements') is None:
                        expect()
                        skip(eid + '/' + (o.get('treatment_id') or '?'), 'source reference-collision evidence missing', intentional=False)
                    else:
                        compare(eid, '%s collisions_reference' % o.get('treatment_id'),
                                float(o['collisions_reference']),
                                float(eff['control_collision_involvements']))
                if o.get('joint_non_deterioration') is not None:
                    jn = row.get('joint_non_deterioration') or {}
                    expect()
                    if 'all_continuous_costs_nonworsening' not in jn:
                        skip(eid + '/' + (o.get('treatment_id') or '?'), 'source joint_non_deterioration missing', intentional=False)
                    else:
                        performed()
                        if bool(jn['all_continuous_costs_nonworsening']) != bool(o['joint_non_deterioration']):
                            msg = '%s/%s: joint_non_deterioration %r != source %r' % (
                                eid, o.get('treatment_id'), o['joint_non_deterioration'],
                                jn['all_continuous_costs_nonworsening'])
                            rec['mismatches'].append(msg)
                            all_mismatches.append(msg)

        hard_skip = [x for x in rec['skipped_checks'] if not x.get('intentional')]
        if rec['mismatches'] or hard_skip:
            rec['status'] = 'failure'
        elif any(x.get('intentional') for x in rec['skipped_checks']):
            rec['status'] = 'partial'
        else:
            rec['status'] = 'complete'
        report['sources'].append(rec)

    report['summary'] = {
        'sources': len(reg),
        'expected_checks': expected_total,
        'performed_checks': performed_total,
        'skipped_checks': sum(len(x.get('skipped_checks', [])) for x in report['sources']),
        'mismatches': len(all_mismatches),
        'complete': sum(1 for x in report['sources'] if x['status'] == 'complete'),
        'partial': sum(1 for x in report['sources'] if x['status'] == 'partial'),
        'failure': sum(1 for x in report['sources'] if x['status'] == 'failure'),
        'deleted': sum(1 for x in report['sources'] if x['status'] == 'deleted'),
    }
    return report, all_mismatches, expected_total, performed_total


def cmd_verify_sources(a):
    d = load()
    report, mismatches, expected, performed = verify_sources(d)
    if report is None:
        sys.exit('no allowed sources registered')
    with open(REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    s = report['summary']
    print('SOURCE VERIFICATION (registered allowed sources only; sealed outcomes never traversed)')
    print('report written:', REPORT)
    print('sources=%d (complete=%d partial=%d failure=%d deleted=%d) | expected=%d performed=%d '
          'skipped=%d mismatches=%d'
          % (s['sources'], s['complete'], s['partial'], s['failure'], s['deleted'],
             s['expected_checks'], s['performed_checks'], s['skipped_checks'], s['mismatches']))
    for m in mismatches:
        print('  !', m)
    for r in report['sources']:
        print('  [%-7s] %s expected=%d performed=%d skipped=%d (sha %s...)'
              % (r['status'], r['source_id'], r['expected_checks'], r['performed_checks'],
                 len(r.get('skipped_checks', [])), (r.get('sha256_current') or 'n/a')[:12]))
        for sk in r.get('skipped_checks', []):
            kind = 'intentional' if sk.get('intentional') else 'REQUIRED'
            print('           skipped (%s): %s' % (kind, sk.get('reason')))
    sys.exit(1 if mismatches else 0)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('search')
    s.add_argument('--family')
    s.add_argument('--kind')
    s.add_argument('--status')
    s.add_argument('--seed')
    s.add_argument('--metric')
    s.add_argument('--text')
    s.set_defaults(fn=cmd_search)

    c = sub.add_parser('card')
    c.add_argument('id')
    c.set_defaults(fn=cmd_card)

    cl = sub.add_parser('closest')
    cl.add_argument('--text', required=True)
    cl.add_argument('-n', type=int, default=5)
    cl.set_defaults(fn=cmd_closest)

    o = sub.add_parser('open-questions')
    o.set_defaults(fn=cmd_open_questions)
    p = sub.add_parser('pending')
    p.set_defaults(fn=cmd_pending)
    sm = sub.add_parser('summary')
    sm.add_argument('--write', action='store_true')
    sm.add_argument('--check', action='store_true')
    sm.set_defaults(fn=cmd_summary)
    v = sub.add_parser('validate')
    v.set_defaults(fn=cmd_validate)
    vs = sub.add_parser('verify-sources')
    vs.set_defaults(fn=cmd_verify_sources)

    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
