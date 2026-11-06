import json, sys, statistics as st
ev = []; geo = dict(calls=0, rows=0, scene_invalid_rows=0, ramp_cars=0, ramp_scene_invalid=0, ramp_overlap=0, ramp_phys_ok=0)
rs = dict(nonfinite=0, no_lane=0, len0=0); nb = 0; nfb = 0; tfzm = 0
for fn in sys.argv[1:]:
    for l in open(fn):
        r = json.loads(l); nb += 1
        tfzm += sum(z['path'] == 'tf' for z in r['zero_mass'])
        nfb += r['nonfinite_chunks'] > 0
        for e in r.get('on_events', []): e['batch'] = (fn[-10:], r['step']); ev.append(e)
        for k in geo: geo[k] += r.get('geo', {}).get(k, 0)
        for k in rs: rs[k] += r.get('geo', {}).get('reasons', {}).get(k, 0)
print('batches', nb, 'with non-finite chunk', nfb, 'tf zero-mass', tfzm)
print('rollout geometry', geo, rs)
zm = [e for e in ev if not e['elig']]; ok = [e for e in ev if e['elig']]
print('on-policy event cells', len(ev), 'vetoed (zero mass)', len(zm), 'eligible', len(ok), 'support true among vetoed', sum(e['support'] for e in zm))
def q(xs):
    xs = sorted(abs(x) for x in xs)
    return None if not xs else dict(n=len(xs), med=round(xs[len(xs)//2], 2), p25=round(xs[len(xs)//4], 2), p75=round(xs[3*len(xs)//4], 2), max=round(xs[-1], 2))
print('|dpos| vetoed', q([e['dpos'] for e in zm])); print('|dpos| eligible', q([e['dpos'] for e in ok]))
print('|dv| vetoed', q([e['dv'] for e in zm])); print('|dv| eligible', q([e['dv'] for e in ok]))
for lim in (2, 5, 10, 25, 50):
    a = sum(abs(e['dpos']) <= lim for e in zm); b = sum(abs(e['dpos']) <= lim for e in ok)
    print(f'|dpos|<={lim}: vetoed {a}/{len(zm)} eligible {b}/{len(ok)}')
