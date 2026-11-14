"""A95 zero-SUMO mapping pre-check: a80_s2b.map_row (MAPPING_RULE, pure) per stream row; writes nothing in any root."""
import hashlib, json, sys
sys.path[:0] = ['ops', '.']
import a80_s2b as M
p, out = sys.argv[1], sys.argv[2]
raw = open(p, 'rb').read()
rows = json.loads(raw)
res = []
for i, r in enumerate(rows):
    c, why = M.map_row(i, r)
    res.append(dict(family=r['family'], reason=why, form=c['form'] if c else None,
                    uncovered=sorted(list(t) for t in M.form_tokens(c['strategy']) - M.covered_tokens()) if c else None))
open(out, 'w').write(json.dumps(dict(stream_sha256=hashlib.sha256(raw).hexdigest(),
                                     a80_s2b_sha256=hashlib.sha256(open('ops/a80_s2b.py', 'rb').read()).hexdigest(),
                                     rows=res), indent=1))
print('refused', sum(1 for x in res if x['reason']), 'of', len(res))
