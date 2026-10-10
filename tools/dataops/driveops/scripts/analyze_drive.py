import json, collections, sys, io

path = sys.argv[1] if len(sys.argv) > 1 else r'E:/rd-wt/reorg3/driveops/_work/drive_recursive.json'
d = json.load(open(path, encoding='utf-8-sig'))
files = [x for x in d if not x.get('IsDir')]
agg = collections.defaultdict(lambda: [0, 0])
for f in files:
    p = f['Path'].split('/')
    key = '/'.join(p[:2]) if len(p) > 1 else p[0]
    agg[key][0] += 1
    agg[key][1] += f['Size']
out = []
out.append('== by top-2 dir (bytes, files) ==')
for k in sorted(agg):
    out.append(f"{agg[k][1]:>14} {agg[k][0]:>7}  {k}")
out.append(f"TOTAL {sum(x['Size'] for x in files)} bytes in {len(files)} files")
out.append('== archives/bundles (size, md5, path) ==')
for f in sorted(files, key=lambda x: x['Size'], reverse=True):
    p = f['Path'].lower()
    if p.endswith('.tar.zst') or p.endswith('.bundle') or p.endswith('.tar') or p.endswith('.zst'):
        md5 = (f.get('Hashes') or {}).get('MD5', '-')
        out.append(f"{f['Size']:>14} {md5}  {f['Path']}")
sys.stdout.reconfigure(encoding='utf-8')
print('\n'.join(out))
