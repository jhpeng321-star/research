import json, sys
for fn in sys.argv[1:]:
    for l in open(fn):
        r = json.loads(l)
        g = r['grad_norms']
        print(fn[-10:], r['step'], 'nonfin', r['nonfinite_chunks'], 'err', bool(r['error']), 'nonzero tensors', sum(1 for v in g.values() if v), '/', len(g),
              {k.split('.')[1]: ('%.2e' % v if v else 0) for k, v in g.items()} if any(g.values()) else '')
