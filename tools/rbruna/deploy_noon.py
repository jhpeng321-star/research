"""One owner-authorized CPU-monitor dispatch; no inference dispatch or retry."""
import hashlib
import json
from pathlib import Path
import subprocess

from tools.rbruna import watch


def main():
    local = watch.STATE/'WM-PHYS-A2'
    original = json.loads(json.loads((local/'DISPATCH.json').read_text())['receipt'])
    assert original['pid'] == 334996 and original['variant'] == 'WM-PHYS-A2'
    files = [watch.HERE/'continue_noon.py', watch.HERE/'remote.py']
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    tag = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    stage = '/root/rbeval20261031/runa/noon_'+tag
    root = original['root']
    marker = local/'NOON_DISPATCH.json'
    assert not marker.exists(), 'existing continuation; inspect it, never replay'
    watch.ssh(f"import json\nfrom pathlib import Path\np=Path({root!r})\n"
        f"assert json.loads((p/'DISPATCH.json').read_text())=={original!r}\n"
        "assert not any((p/n).exists() for n in ('FAILED.json','RESULT.json','NOON_HANDOVER_AUTHORIZED.json'))\n"
        f"s=Path({stage!r})\nassert s.parent.is_dir()\ns.mkdir()\n")
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
        *map(str, files), f'gpuA:{stage}/'], check=True, capture_output=True, text=True, timeout=60)
    receipt = watch.ssh(f"import sys,hashlib,json,subprocess\nfrom pathlib import Path\ns=Path({stage!r})\n"
        f"for name,digest in {hashes!r}.items():\n assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest\n"
        "sys.path.insert(0,str(s))\nimport remote\n"
        f"argv=['/root/miniconda3/bin/python','-B','-u',str(s/'continue_noon.py'),'--root',{root!r},'--expected-supervisor',{json.dumps(original)!r}]\n"
        "with (s/'monitor.log').open('x') as log:\n p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=s,start_new_session=True)\n"
        f"receipt=dict(**remote.incarnation(p.pid),stage=str(s),argv=argv,source_hashes={hashes!r})\n"
        "remote.save(s/'DISPATCH.json',receipt)\nprint(json.dumps(receipt))\n")
    watch.save_new(marker, json.loads(receipt))
    print(receipt)


if __name__ == '__main__':
    main()
