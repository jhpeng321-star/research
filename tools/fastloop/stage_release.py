"""Stage one clean committed release and an explicit CPU contract; does not arm/launch."""
import argparse
import hashlib
import json
from pathlib import Path

from remote_verify import BLOCK, bundle, remote


def stage(contract):
    for text in json.dumps(contract).split():
        if any(token in text.lower() for token in BLOCK) or "FILL_FROM_" in text:
            raise ValueError("CPU contract contains an unresolved/restricted path")
    data = bundle(release=True)
    for box in ("gpuR", "cpuP", "cpuQ"):
        cpu = box.startswith("cpu")
        root = "/home/ubuntu/c4-fastloop-sfastloop" if cpu else "/root/autodl-tmp/c4-fastloop-sfastloop"
        fixture = contract["scenario_fixture"][box] if cpu else None
        script = f"""
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path({root!r}); assert root.parent.is_dir()
assert not (root/'arm/waiter.pid').exists(), 'release already armed; refuse mutation'
root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
 for item in archive:
  assert item.isfile() and (root/item.name).resolve().is_relative_to(root.resolve())
  archive.extract(item,root,filter='data')
source=json.loads((root/'FASTLOOP_SOURCE.json').read_text())
assert source['release']
for rel,expected in source['files'].items(): assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected,rel
fixture={fixture!r}
if fixture:
 src=pathlib.Path(fixture); assert src.is_dir()
 dst=root/'scenarios'
 if not dst.exists(): dst.symlink_to(src,target_is_directory=True)
 assert dst.resolve()==src.resolve()
if {not cpu!r}: (root/'CPU_CONTRACT.runtime.json').write_text({json.dumps(contract)!r})
print(json.dumps({{'host':{box!r},'code_sha':source['code_sha'],'source_verified':True}}))
"""
        result = remote(box, script, data=data)
        if result.returncode:
            raise RuntimeError(f"{box}: {result.stderr.decode(errors='replace')}")
        print(result.stdout.decode(), end="")
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-contract", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps({"bundle_sha256": stage(json.loads(args.cpu_contract.read_text())), "armed": False}))
