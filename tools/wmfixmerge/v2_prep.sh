set -e
cd /root/autodl-tmp/c4/wmfixmerge20261031
export PYTHONPATH=src:.
PY=/root/miniconda3/bin/python
$PY - <<'P' > v2/WMDATA_VIEWS_VERIFY.json
import json, hashlib, pathlib
root = pathlib.Path("/root/autodl-tmp/c4/wmdata20261031/views")
s = json.load(open(root / "SHA256SUM.json"))["files"]
bad = []
for rel, meta in s.items():
    h = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    if h != meta["sha256"]:
        bad.append(rel)
print(json.dumps(dict(files=len(s), mismatches=bad, sha256sum_json=hashlib.sha256((root/"SHA256SUM.json").read_bytes()).hexdigest()), indent=1))
P
$PY -c "import json,sys;d=json.load(open('v2/WMDATA_VIEWS_VERIFY.json'));sys.exit(1 if d['mismatches'] else 0)"
$PY -m tools.wmfixmerge.merge_corpus --fit-root /root/autodl-tmp/c4/e3data --wmdata-root /root/autodl-tmp/c4/wmdata20261031/views --out /root/autodl-tmp/c4/wmfixmerge20261031/v2/corpus > v2/merge_corpus.json
$PY -m tools.wmfixmerge.prepare --root /root/autodl-tmp/c4/wmfixmerge20261031/v2/corpus --out /root/autodl-tmp/c4/wmfixmerge20261031/v2/TRAIN_PREPARED_V2.json
sha256sum v2/TRAIN_PREPARED_V2.json v2/corpus/C4_EPISODES_MANIFEST.json v2/corpus/SPLIT.json > v2/PREP_SHA256.txt
$PY -c "
import json,collections;d=json.load(open('v2/TRAIN_PREPARED_V2.json'))
print(json.dumps(dict(seeds=len(d['seeds']),episodes=len(d['episodes']),arms=collections.Counter(e['arm'] for e in d['episodes']),scales=d['scales'],anchor_scales=d['anchor_scales'],crossing_width=d['crossing_width'])))" > v2/PREP_SUMMARY.json
echo PREP_DONE
