"""Prepare a deterministic, conditional re-arm package without changing gpuR's armed files."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from remote import remote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.main14.deploy import export  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip(), "commit candidate first"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    source, files = export(commit)
    old = json.loads(remote("gpuR", '''
import base64,json,hashlib
from pathlib import Path
a=Path('/root/autodl-tmp/c4/v5/main14_arm'); raw=(a/'ARMED.json').read_bytes(); arm=json.loads(raw)
assert not any((a/x).exists() for x in ['LAUNCHED.json','WAIT_OWNER.json','STOP_ARM','EXPIRED.json'])
owner=json.loads((a/'WAITER_INSTALLED.json').read_text())
f=Path('/proc/'+str(owner['pid'])+'/stat').read_text().rsplit(')',1)[1].split(); assert f[19]==owner['start_ticks'] and f[0]!='Z'
print(json.dumps({'arm_sha256':hashlib.sha256(raw).hexdigest(),'owner':owner,
 'registration':base64.b64encode(Path(arm['registration']).read_bytes()).decode(),
 'config':base64.b64encode(Path(arm['config']).read_bytes()).decode()}))
'''))
    old_reg, old_cfg = [base64.b64decode(old[k]) for k in ("registration", "config")]
    reg, cfg = json.loads(old_reg), json.loads(old_cfg)
    assert reg["sources"]["code_sha"] == "fe7b5146ca1aa712bd6a4503e421c31866e7b738"
    cfg["policy"]["config"] = "configs/c4_policy14_main14_nativelane.toml"
    config_text = json.dumps(cfg, indent=1) + "\n"
    cfg_sha = sha(config_text.encode())
    reg["sources"].update(code_sha=commit, config_sha256=cfg_sha,
                          source_manifest_sha256=sha(json.dumps(files, sort_keys=True).encode()))
    reg["registered_at"] = datetime.now(timezone.utc).isoformat()
    reg["native_lane_amendment"] = {
        "registration": "S-NATIVELANE-1", "flag": "policy14.native_lane_change=true",
        "conditional": "Install only after all registered 13E FINAL paired evidence is complete and adoption rule passes before 09:00; no PASS->FAIL exploratory pulse_m/gap_open sign unit; tests and real-WM 2-update CPU prearm smoke pass before documented disarm",
        "deadline": "2026-10-26T09:00:00+08:00", "new_sumo_sim_s": 0,
        "preserved": "base selection, frozen gate panel, seeds/exclusions, budgets, acceptance/incident/update-10 stops, readiness and allocator fixes",
        "failure": "Do not disarm on failed/incomplete/late scientific or prearm evidence; no autonomous retry after operational failure"}
    reg_text = json.dumps(reg, indent=1) + "\n"
    req = {"code_sha": commit, "source_files": files, "source_bundle_sha256": sha(source),
           "config_text": config_text, "config_sha256": cfg_sha,
           "registration_text": reg_text, "registration_sha256": sha(reg_text.encode()),
           "expected_old_arm_sha256": old["arm_sha256"], "expected_old_waiter": old["owner"],
           "expected_old_registration_sha256": sha(old_reg), "expected_old_config_sha256": sha(old_cfg)}
    adopt_code = subprocess.check_output(["git", "show", f"{commit}:tools/nativelane/adopt.py"], cwd=ROOT)
    outer = io.BytesIO()
    with tarfile.open(fileobj=outer, mode="w:gz") as t:
        for name, data in (("source.tar.gz", source), ("request.json", json.dumps(req, indent=1).encode()),
                           ("adopt.py", adopt_code)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            t.addfile(info, io.BytesIO(data))
    bundle = outer.getvalue()
    print(remote("cpuX", '''
import sys,hashlib,json
from pathlib import Path
p=Path.home()/'nativelane'; assert p.is_dir(); data=sys.stdin.buffer.read()
f=p/'rearm_package.tar.gz'; assert not f.exists(), 'retain any previous candidate rather than overwrite'
f.write_bytes(data); print(json.dumps({'package_sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}))
''', bundle).decode())
    receipt = {k: req[k] for k in ("code_sha", "config_sha256", "registration_sha256", "source_bundle_sha256",
                                  "expected_old_arm_sha256", "expected_old_registration_sha256", "expected_old_config_sha256")}
    receipt.update(package_sha256=sha(bundle), source_files=len(files), registered_at=reg["registered_at"],
                   package_path="/home/ubuntu/nativelane/rearm_package.tar.gz", conditional=True)
    out = ROOT / "tools/nativelane/REARM_PACKAGE.json"
    out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
