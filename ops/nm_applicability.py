"""N6 amendment 24 (owner main claim "when cooperation applies we can win mappo"): applicability labels.

A held-out world is APPLICABLE iff in its sealed U0 (MAPPO) run at least one decision second t in [300, 1200) has a
CAV ``own`` and an admitted bank plan (WM-sourced, privileged situation) of that world's cell such that
  (i)  fullpol_situational.onset_bin(own, rows, 'privileged') is one of the plan's admitted bins, and
  (ii) fullpol_bank.offer_ok(cond, own, rows) holds (the plan's own-lane / zone / role-lane / spread_m proximity
       rule; gates fail closed, exactly as the bank offers it).
The functions are the ones the training uses (imported from the evaluated source tree, never re-implemented here).

Inputs: ONLY the U0 vehicle logs of the sealed pre-run (located through the U0 wave receipts; from a receipt only
job_id / work_dir / world_id / world / vehicle_log.sha256 are read) and the frozen end-of-training bank doc (sha
logged). Refused when any A/B/R wave of a trained run exists in the root (labels must precede every B outcome).
Never reads any delay/J/safety field, any B, R, A, RULE, ORACLE, NATIVE or comparator file, or any outcome delta.

Outputs (sealed APPLICABILITY_<tag>.json): per world applicable, first applicable second, applicable seconds,
vehicle-seconds; per cell the applicability rate (fraction of worlds; fraction of vehicle-time = vehicle-seconds in
applicable seconds / all vehicle-seconds in [300, 1200)).
"""
import argparse
import gzip
import hashlib
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import nm_finaleval as nm  # noqa: E402

SCHEMA = 'coopmwm.nm.finaleval.v1.applicability'
T0, T1 = 300, 1200
OUTCOME_ARMS = ('A', 'B', 'R')


def _u0_receipts(final_root, prefix):
    """The U0 wave receipts of one panel (whole wave or chunk files), sealed; nothing else is globbed."""
    final_root = Path(final_root)
    whole = final_root/f'WAVE_{prefix}U0.json'
    paths = [whole] if whole.is_file() else sorted(final_root.glob(f'WAVE_{prefix}U0__*.json'))
    if not paths:
        raise nm.Blocked(f'no sealed U0 wave for prefix {prefix!r}')
    out = []
    for p in paths:
        value = nm.verify(nm.read(p), 'receipt_sha256', p.name)
        out.append((p, value))
    return out


def u0_logs(final_root, prefix):
    """[(world_id, world, log path, sha)] from the U0 receipts; each log file sha re-verified."""
    rows, receipts = [], []
    for path, value in _u0_receipts(final_root, prefix):
        receipts.append(dict(file=path.name, receipt_sha256=value['receipt_sha256']))
        for r in value['records']:
            if r.get('arm') != 'U0':
                raise nm.Blocked(f'{path.name}: non-U0 record')
            log = Path(final_root)/'evaluation'/r.get('work_dir', r['job_id'])/nm.VEHICLE_LOG
            sha = r['vehicle_log']['sha256']
            if nm.file_sha(log) != sha:
                raise nm.Blocked(f'{r["job_id"]}: vehicle log differs from its sealed record')
            rows.append((r['world_id'], r['world'], log, sha))
    return rows, receipts


def ticks(path, t0=T0, t1=T1):
    """(t, rows) per logged second: rows = [dict(vid, s_m, v_mps, lane (WM), is_cav)]."""
    heads = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            if 'veh' in x:
                heads[x['veh']] = x
                continue
            t = int(x['t'])
            if not t0 <= t < t1:
                continue
            yield t, [dict(vid=heads[k]['id'], s_m=float(s), v_mps=float(v), lane=int(ln), is_cav=bool(heads[k]['cav']))
                      for k, s, v, ln in zip(x['k'], x['s'], x['v'], x['lane'])]


def bank_entries(bank, cell_key):
    """Admitted privileged-situation plans of this cell: [(family, cond, bins)]; other rows are not WM plans."""
    # FULLPOL (2026-11-08): the frozen bank = the run's bank_growth snapshot guidance doc; templates live in
    # doc['clean5c']['templates'] = [{cell, family, template, offer_condition}] (a top-level 'templates' is accepted)
    templates = (bank.get('clean5c') or {}).get('templates', bank.get('templates'))
    if templates is None:
        raise nm.Blocked('frozen bank doc has no clean5c.templates list')
    out = []
    for e in templates:
        cond = e.get('offer_condition') or {}
        sit = cond.get('situation')
        if e.get('cell') != cell_key or not sit or sit.get('mode') != 'privileged':
            continue
        item = (e['family'], cond, [dict(cong=b['cong'], lane_diff=b['lane_diff']) for b in sit['bins']])
        if item not in out:                                # duplicate rows of the same (family, cond) count once
            out.append(item)
    return out


def label_world(path, entries, onset_bin, offer_ok):
    first, seconds, veh_total, veh_app, fams = None, 0, 0, 0, {}
    for t, rows in ticks(path):
        veh_total += len(rows)
        hit = False
        for own in rows:
            if not own['is_cav']:
                continue
            for fam, cond, bins in entries:
                lanes = [int(x) for x in cond.get('own_wm_lanes') or ()]
                zone = cond.get('zone_s')
                if own['lane'] not in lanes or (zone is not None and not float(zone[0]) <= own['s_m'] <= float(zone[1])):
                    continue                              # offer_ok's own-lane / zone test first (cheap, same rule)
                b = onset_bin(own, rows, 'privileged')
                if not any(b.get('cong') == w['cong'] and b.get('lane_diff') == w['lane_diff'] for w in bins):
                    continue
                if offer_ok(cond, own, rows):
                    hit = True
                    fams[fam] = fams.get(fam, 0)+1
                    break
            if hit:
                break
        if hit:
            seconds += 1
            veh_app += len(rows)
            first = t if first is None else first
    return dict(applicable=first is not None, first_t=first, applicable_seconds=seconds, vehicle_seconds=veh_total,
                applicable_vehicle_seconds=veh_app, families=dict(sorted(fams.items())))


def code_digest(onset_bin, offer_ok):
    parts = [inspect.getsource(f) for f in (onset_bin, offer_ok, label_world, ticks, bank_entries)]
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


def label(*, final_root, panels, bank_path, tag, functions=None):
    """``panels``: [(panel dict, wave prefix)]. Sealed APPLICABILITY_<tag>.json in the final root (never overwritten)."""
    final_root = Path(final_root)
    target = final_root/f'APPLICABILITY_{tag}.json'
    if target.exists():
        return nm.verify(nm.read(target), 'receipt_sha256', target.name)
    trained = [p.name for p in final_root.glob('WAVE_*.json') if any(
        p.name.startswith(f'WAVE_{pre}{a}_u') for a in OUTCOME_ARMS for pre in ('', 'C3_'))
        or any(p.name.startswith(f'WAVE_{a}') for a in nm.BANK_ARMS)]          # amendment 27: before bank outcomes
    if trained:
        raise nm.Blocked(f'trained-run waves already exist ({trained[:3]}): labels must precede every B outcome')
    if functions is None:
        from tools.coopmwm.fullpol_bank import offer_ok
        from tools.coopmwm.fullpol_situational import onset_bin
    else:
        onset_bin, offer_ok = functions
    bank = json.loads(Path(bank_path).read_text(encoding='utf-8'))
    cells, worlds, sources = {}, {}, []
    for panel, prefix in panels:
        key = panel['cell']['cell_key']
        entries = bank_entries(bank, key)
        logs, receipts = u0_logs(final_root, prefix)
        sources.append(dict(panel_sha256=panel['panel_sha256'], cell=key, prefix=prefix, u0_receipts=receipts,
                            plans=[f for f, _, _ in entries]))
        if [w for w, _, _, _ in logs] != [r['world_id'] for r in panel['worlds']]:
            raise nm.Blocked(f'{key}: U0 logs do not cover the registered panel in order')
        rows = {}
        for world_id, _, log, sha in logs:
            rows[world_id] = dict(cell=key, log_sha256=sha, **label_world(log, entries, onset_bin, offer_ok))
        worlds.update(rows)
        n = len(rows)
        app = [r for r in rows.values() if r['applicable']]
        vt = sum(r['vehicle_seconds'] for r in rows.values())
        cells[key] = dict(worlds=n, applicable_worlds=len(app), rate_worlds=len(app)/n if n else None,
                          vehicle_seconds=vt, applicable_vehicle_seconds=sum(r['applicable_vehicle_seconds'] for r in rows.values()),
                          rate_vehicle_time=(sum(r['applicable_vehicle_seconds'] for r in rows.values())/vt) if vt else None,
                          admitted_plans=len(entries))
    value = nm.seal(dict(schema=SCHEMA, tag=tag, rule=__doc__.split('\n\n')[1], bank=dict(path=str(bank_path),
                         sha256=nm.file_sha(bank_path)), code_sha256=code_digest(onset_bin, offer_ok),
                         sources=sources, cells=cells, worlds=worlds,
                         reads='U0 vehicle logs + the frozen bank only (no outcome, no B/R/A file)'), 'receipt_sha256')
    nm.publish(target, value)
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description='N6 amendment 24 applicability labels (zero SUMO)')
    p.add_argument('--setting', required=True, choices=tuple(nm.SETTINGS))
    p.add_argument('--final-root', required=True)
    p.add_argument('--bank', required=True, help='the frozen end-of-training bank doc (templates + offer_condition)')
    p.add_argument('--tag', required=True)
    a = p.parse_args(argv)
    nm.apply_setting(a.setting)
    if nm.EVAL_EXTRA_RANGE is not None:
        from tools.coopmwm import map_scenarios
        from tools.coopmwm.nm_eval_runner import install_eval_ranges
        install_eval_ranges(map_scenarios)
    panel, panel_3c = nm.load_panel(), nm.load_panel_3c()
    panels = [(x, pre) for x, pre in ((panel, ''), (panel_3c, 'C3_')) if x is not None]   # p20x: one panel (23b)
    value = label(final_root=a.final_root, panels=panels, bank_path=a.bank, tag=a.tag)
    print(json.dumps(dict(receipt_sha256=value['receipt_sha256'], cells=value['cells']), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
