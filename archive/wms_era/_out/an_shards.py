import json, collections, os
for p in ('E:/rd-wt/dspeed-data/_out/pack_dspeed', 'E:/rd-wt/damping-snap/_out/pack_damping',
          'D:/rd-staging/wmn1_feat/pack_v3'):
    ix = json.load(open(p + '/INDEX.json'))
    c = collections.Counter()
    ex = None
    for r in ix['runs']:
        sd = (r.get('shard_dir') or 'NONE').replace(chr(92), '/')
        ok = os.path.exists(sd + '/slot_index.npy') if sd != 'NONE' else False
        c[(os.path.dirname(os.path.dirname(sd)) if sd != 'NONE' else sd, ok)] += 1
        ex = ex or sd
    print(p, len(ix['runs']), c.most_common(6), ex)
d = 'E:/rd-wt/dspeed-data/_out/shards_dspeed/chunk4/w8660m2700p10t300/arm0'
print(os.listdir(d))
print(open(d + '/ALIGN.json').read()[:600])
