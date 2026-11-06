import json, sys
tot = dict(tf=0, on=0)
for fn in sys.argv[1:]:
    for l in open(fn):
        r = json.loads(l); zm = r['zero_mass']
        tf = sum(z['path'] == 'tf' for z in zm); on = sum(z['path'] == 'on' for z in zm)
        tot['tf'] += tf; tot['on'] += on
        print(fn[-12:], r['step'], 'nonfin', r['nonfinite_chunks'], '/', r['chunks'], 'zm tf', tf, 'on', on,
              'err', (r['error'] or '')[:50], 'gradmax', max((v or 0) for v in r['grad_norms'].values()))
print(tot)
