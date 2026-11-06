import json, sys
for fn in sys.argv[1:]:
    for l in open(fn):
        r = json.loads(l)
        for z in r['zero_mass']:
            if z['path'] == 'tf':
                print(fn[-10:], r['step'], {k: v for k, v in z.items() if k != 'wm'})
