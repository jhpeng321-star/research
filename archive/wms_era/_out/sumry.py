import json, collections, re, sys
D = sys.argv[1]
r = json.load(open(D + '/BUILD_RECEIPT.json')); print(r['status'])
for k, v in r['sets'].items():
    print(k, {a: b for a, b in v.items() if a != 'counts'}); print('  ', v['counts'])
c = collections.Counter()
for l in open(D + '/REJECTIONS.jsonl'):
    x = json.loads(l); c[(x['set'], re.sub(r"window '[^']*'", 'window', x['error']))] += 1
for k, v in c.items(): print(v, k)
