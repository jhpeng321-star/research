"""Complete-panel exam export and independent ledger/label reconciliation."""
import csv
import json
from pathlib import Path
from .protocol import TIMES, DATA_CAP, canonical, sha, digest
from .storage import read, write


def csvwrite(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,keys)
        writer.writeheader()
        writer.writerows(rows)


def export(root, partial=False, name=None):
    root = Path(root)
    selection = read(root/'SELECTED.json')
    dest = root/(name or ('exam-partial' if partial else 'exam'))
    dest.mkdir()
    states, labels, credits, source = [], [], [], []
    for entry in selection['worlds']:
        w = entry['world']
        baseline_path = root/'jobs'/f's{w["seed"]}_screen'/'RESULT.json'
        baseline = read(baseline_path)
        source.append(dict(path=str(baseline_path.relative_to(root)),sha256=sha(baseline_path)))
        for t in TIMES:
            state_id = f's{w["seed"]}_t{t}'
            if partial and not all((root/'jobs'/f'{state_id}_{oid}'/'RESULT.json').exists() and
                                   (root/'jobs'/f'{state_id}_{oid}'/'CLOSE.json').exists()
                                   for oid in ('O1','O2','O3','O4')):
                continue
            s = baseline['checkpoints'][str(t)]
            states.append(dict(state_id=state_id,world_id=w['seed'],t0=t,
                               features=s['features'],observed_vehicles=s['rows'],
                               prefix_sha256=digest(s['prefix'])))
            base = {str(h):{k:v-s['totals'][k] for k,v in baseline['checkpoints'][str(t+h)]['totals'].items()}
                    for h in (120,300,600)}
            recs = {}
            for oid in ('O0','O1','O2','O3','O4'):
                if oid == 'O0':
                    rec = None
                else:
                    path = root/'jobs'/f'{state_id}_{oid}'/'RESULT.json'
                    rec = read(path)
                    recs[oid] = rec
                    source.append(dict(path=str(path.relative_to(root)),sha256=sha(path)))
                    if rec['prefix_sha256'] != digest(s['prefix']) or rec['features'] != s['features']:
                        raise ValueError('wrong state prefix/features')
                for h in (120,300,600):
                    j = base[str(h)] if rec is None else rec['metrics'][str(h)]
                    d = {k:base[str(h)][k]-v for k,v in j.items()}
                    if rec is not None and d != rec['D'][str(h)]:
                        raise ValueError('label arithmetic')
                    if abs(d['team_delay_vs']-d['road_delay_vs']-d['source_wait_vs']) > 1e-6:
                        raise ValueError('team/source conservation')
                    labels.append(dict(state_id=state_id,world_id=w['seed'],cell=w['cell'],
                        t0=t,setting=int(s['features']['setting']),option=oid,horizon_s=h,
                        available=True if rec is None else rec['plan']['available'],
                        D=d['team_delay_vs'],D_road=d['road_delay_vs'],D_source=d['source_wait_vs'],
                        J=j['team_delay_vs'],collisions=j['collisions'],teleports=j['teleports'],
                        ttc_below_3_vs=j['ttc_below_3_vs'],hard_brake_vs=j['hard_brake'],
                        override_share=0. if rec is None else rec['safety']['override_share']))
            if w['seed'] in selection['solo_worlds'] and (not partial or all(
                    (root/'jobs'/f'{state_id}_{oid}_solo{i}'/'RESULT.json').exists()
                    for oid in ('O2','O3') for i in (0,1))):
                for oid in ('O2','O3'):
                    solos = [read(root/'jobs'/f'{state_id}_{oid}_solo{i}'/'RESULT.json') for i in (0,1)]
                    for h in (120,300,600):
                        dg = recs[oid]['D'][str(h)]['team_delay_vs']
                        ds = [r['D'][str(h)]['team_delay_vs'] for r in solos]
                        credits.append(dict(state_id=state_id,world_id=w['seed'],option=oid,horizon_s=h,
                            available=recs[oid]['plan']['available'],D_G=dg,D_0=ds[0],D_1=ds[1],
                            sum_D_i=sum(ds),C_G=dg-sum(ds)))
    if not partial and (len(states) != 300 or len(labels) != 4500 or len(credits) != 240):
        raise ValueError('incomplete panel')
    ledger = [json.loads(s) for s in (root/'CHARGES.jsonl').read_text().splitlines()]
    if len({r['id'] for r in ledger}) != len(ledger):
        raise ValueError('duplicate ledger')
    charged, actual = 0, 0.
    for row in ledger:
        if row['phase'] != 'data':
            continue
        close_path=root/'jobs'/row['id']/'CLOSE.json'
        if partial and not close_path.exists():
            continue
        closed = read(close_path)
        if closed['charged_sim_s'] != row['sim_s'] or closed['actual_sim_s'] != row['sim_s']:
            raise ValueError('reservation/actual reconciliation')
        charged += row['sim_s']
        actual += closed['actual_sim_s']
    if not partial and charged != DATA_CAP:
        raise ValueError('exact dataset budget')
    with (dest/'states.jsonl').open('x',encoding='utf-8') as f:
        for state in states:
            f.write(canonical(state)+'\n')
    csvwrite(dest/'labels.csv',labels)
    csvwrite(dest/'group_credit.csv',credits)
    write(dest/'worlds.json',selection)
    write(dest/'provenance.json',dict(design_sha256=sha(root/'COOPBANDIT_DESIGN.md'),sources=source,
                                    charged_sim_s=charged,actual_sim_s=actual,partial=partial,states=len(states)))
    with (dest/'README.md').open('x',encoding='utf-8') as f:
        f.write('# COOPBANDIT world-model EXAM v1\n\n'
                'states.jsonl contains numeric observable features and the observed vehicle roster.\n'
                'labels.csv is long format: state_id, option O0..O4, horizon_s 120/300/600;\n'
                'D=J(O0)-J(option), positive saves fleet plus source delay (vehicle-seconds).\n'
                'Unavailable options are explicit exact no-ops. J, D_road, D_source, safety and\n'
                'override shares are included. group_credit.csv has C_G=D_G-sum(D_i).\n'
                'world_id is grouping metadata, never a predictor; cell is descriptive only.\n'
                'Group every split by world_id. Dev worlds 981080..981091 are excluded.\n'
                'The roster uses only RSU detections and CAV telemetry, no driver truth.\n'
                'provenance.json and MANIFEST.json bind the sources and design with SHA256.\n'
                'Partial exports are exploratory, completion-order subsets, not final panel estimates.\n')
    write(dest/'MANIFEST.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
