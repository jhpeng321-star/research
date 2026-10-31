set -e
SRC=/root/autodl-tmp/c4/speedhead20261030/e3final/runtime
DST=/root/autodl-tmp/c4/wmfixmerge20261031/_in/pi_deps
test ! -e $DST
mkdir -p $DST/output
cp -a $SRC/traffic_shepherd $DST/
for v in controller_verification_v2 controller_verification_v4 controller_verification_v7; do cp -a $SRC/output/$v $DST/output/; done
[ -f $SRC/output/__init__.py ] && cp -a $SRC/output/__init__.py $DST/output/ || true
cd /root/autodl-tmp/c4/wmfixmerge20261031
/root/miniconda3/bin/python - <<'P'
import json, hashlib, pathlib
src = pathlib.Path("/root/autodl-tmp/c4/speedhead20261030/e3final/runtime")
dst = pathlib.Path("/root/autodl-tmp/c4/wmfixmerge20261031/_in/pi_deps")
m = json.loads((src / "SOURCE_MANIFEST.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checked = bad = 0
for n, h in m.items():
    p = dst / n
    if p.is_file():
        checked += 1
        bad += sha(p) != h
print(json.dumps(dict(manifest_sha256=sha(src / "SOURCE_MANIFEST.json"), checked=checked, mismatches=bad)))
P
PYTHONPATH=src:. /root/miniconda3/bin/python -c "from wms import sim_bridge; d=sim_bridge.load_controller_dependencies(); print('deps OK', d.circles_type, d.pi_type)"
