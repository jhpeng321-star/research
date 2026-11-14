"""A102TS post-run check (zero SUMO, read only; stdlib only). Prints every number; a mismatch is REPORTED, never hidden.
(a) executor_sha256_loaded == MAPPO60C 2abed317 in both re-run records;
(b) each re-run's total delay J_veh_s vs the stored A94 per-world value for world 988700 (the A94 wave record whose
    receipt is cited in A94_RESULTS_D2800_p60.json 'sources'), MATCH / MISMATCH with both numbers; plus safety
    counters and (bank arm) the a75 behavioural receipts (activations, activations_by_family, issued, acts);
(c) the raw files (name, bytes, sha256) of each re-run episode, and a format probe of observable.jsonl.gz for the
    time-space figure (records, vehicles with s_m / v_mps, actions carrying oracle_plan).
Exit 0 iff (a) and (b-J) hold for both arms and the raw files exist; prints A102TS_CHECK_DONE last in every case."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/root/a102ts/root')
A94 = Path('/root/a94/root')
MAPPO60C = '2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8'
ARMS = {
    'MAPPO60C': dict(job='a94c_mappo60c_0000', new='WAVE_A94C_MAPPO60C__0000-0000.json',
                     old='WAVE_A94C_MAPPO60C__0000-0095.json',
                     old_sha='e126789b682c391159555d42a4d4bc0f6e134e26fd8765b087a0b12d0d4d626b',
                     old_receipt='693be701751e57d22551f6c191168deeecf04a21a2ee1627eda6868ed6d2ec6b',
                     J_stored=42646.69246811499),
    'BANK_SEARCHED': dict(job='a94c_bank_searched_0000', new='WAVE_A94C_BANK_SEARCHED__0000-0000.json',
                          old='WAVE_A94C_BANK_SEARCHED__0000-0095.json',
                          old_sha='44951b2a6a9a737ffe37be3ad8d8fc223e82fc7c574ddefebd6635e17a689cc3',
                          old_receipt='9230f4ae53f7067f8753fb47f47d866ea6ea739123da8a60e69fa93a9d743586',
                          J_stored=27625.72399731639),
}
RAW = ('RESULT.json.gz', 'vehicles.jsonl.gz', 'observable.jsonl.gz')
IDENT = ('activations', 'activations_by_family', 'issued', 'acts')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rec_of(path, job):
    d = json.loads(Path(path).read_text())
    r = [x for x in d['records'] if x['job_id'] == job]
    return d, (r[0] if len(r) == 1 else None)


def probe(p):
    n = veh = acts = plan_acts = 0
    keys, members, t0, t1 = set(), set(), None, None
    with gzip.open(p, 'rt') as fh:
        for line in fh:
            d = json.loads(line)
            n += 1
            t = float(d['time_s'])
            t0 = t if t0 is None else min(t0, t)
            t1 = t if t1 is None else max(t1, t)
            for v in d.get('vehicles', []):
                veh += 1
                if veh <= 3:
                    keys |= set(v)
            for a in d.get('actions', []):
                acts += 1
                if a.get('oracle_plan'):
                    plan_acts += 1
                    members.add(a['vid'])
    return dict(records=n, t=[t0, t1], vehicle_rows=veh, vehicle_keys=sorted(keys), actions=acts,
                actions_with_oracle_plan=plan_acts, plan_member_vids=len(members),
                figure_fields_ok={'s_m', 'v_mps', 'vid'} <= keys)


ok = True
print('A102TS post-run check', ROOT)
for arm, c in ARMS.items():
    print(f'==== {arm} ({c["job"]}, world 988700)')
    newp = ROOT/c['new']
    if not newp.is_file():
        print(f'  MISSING re-run wave record {newp}')
        ok = False
        continue
    _, new = rec_of(newp, c['job'])
    oldp = A94/c['old']
    od, old = rec_of(oldp, c['job'])
    osha = sha(oldp)
    print(f'  stored record {oldp} sha {osha[:12]} pinned {"OK" if osha == c["old_sha"] else "DIFFERS"}; receipt '
          f'{od["receipt_sha256"][:12]} cited in A94 results {"OK" if od["receipt_sha256"] == c["old_receipt"] else "DIFFERS"}')
    if new is None or old is None:
        print('  record for the job missing (new or stored)')
        ok = False
        continue
    exe = new.get('executor_sha256_loaded')
    a_ok = exe == MAPPO60C
    print(f'  (a) executor_sha256_loaded {exe} -> {"OK" if a_ok else "NOT MAPPO60C"}')
    jn, jo = new.get('J_veh_s'), old.get('J_veh_s')
    if jo != c['J_stored']:
        print(f'  NOTE stored J {jo!r} differs from the pinned constant {c["J_stored"]!r}')
    b_ok = jn == jo
    print(f'  (b) total delay J_veh_s: re-run {jn!r} stored A94 {jo!r} diff {None if jn is None else jn - jo!r} -> '
          f'{"MATCH" if b_ok else "MISMATCH"}')
    print(f'      safety re-run {new.get("safety")} stored {old.get("safety")} -> '
          f'{"MATCH" if new.get("safety") == old.get("safety") else "MISMATCH"}')
    print(f'      vehicle_log ticks/vehicles re-run {[new["vehicle_log"].get(k) for k in ("ticks", "vehicles")]} stored '
          f'{[old["vehicle_log"].get(k) for k in ("ticks", "vehicles")]}; gzip sha re-run '
          f'{new["vehicle_log"].get("sha256", "")[:12]} stored {old["vehicle_log"].get("sha256", "")[:12]} '
          '(informational: gzip header bytes may differ)')
    if 'a75' in old or 'a75' in new:
        for k in IDENT:
            x, y = (new.get('a75') or {}).get(k), (old.get('a75') or {}).get(k)
            print(f'      a75 {k}: {"MATCH" if x == y else "MISMATCH"}'
                  + ('' if x == y else f' (re-run {str(x)[:200]} | stored {str(y)[:200]})'))
    ok = ok and a_ok and b_ok
    work = ROOT/'evaluation'/c['job']
    print(f'  (c) raw files in {work}:')
    for f in sorted(work.iterdir()) if work.is_dir() else []:
        if f.is_file():
            print(f'      {f.name:28s} {f.stat().st_size:>12,d} B  sha {sha(f)[:16]}')
    for f in RAW:
        if not (work/f).is_file():
            print(f'      MISSING {f}')
            ok = False
    drop = A94/'evaluation'/c['job']/'DROPPED_RAW.json'
    if drop.is_file():
        sealed = {x['file']: x for x in json.loads(drop.read_text())['removed']}
        for f in RAW:
            if (work/f).is_file() and f in sealed:
                print(f'      vs A94 sealed (dropped) {f}: bytes {sealed[f]["bytes"]:,} sha {sealed[f]["sha256"][:16]} -> '
                      f'{"BYTE-IDENTICAL" if sha(work/f) == sealed[f]["sha256"] else "bytes differ (informational)"}')
    if (work/'observable.jsonl.gz').is_file():
        print(f'      observable probe: {json.dumps(probe(work/"observable.jsonl.gz"))}')
print('A102TS_CHECK', 'PASS' if ok else 'FAIL (see MISMATCH / MISSING lines above)')
print('A102TS_CHECK_DONE')
sys.exit(0 if ok else 3)
