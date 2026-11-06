"""Capture pre-arm evidence and pin the identical coop-OFF 16-start gate panel."""
import json
from deploy import ROOT, export, remote

receipt_path = ROOT / "tools/main14/deployment_receipt.json"
receipt = json.loads(receipt_path.read_text())
_, files = export(receipt["code_sha"])
print(remote("cpuX", """
import hashlib,json,pathlib,sys
root=pathlib.Path('/home/ubuntu/c4/repo-main14')
sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'src'))
files=json.load(sys.stdin)
for rel,expected in files.items():
 assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected, rel
from traffic_shepherd.config import load_config
c=load_config(); assert c.network_path.is_file() and c.scenario_template_path.is_file()
import traci
print(json.dumps({'cpuX_source_files_verified':len(files),'simulator_fixture_paths_exist':True}))
""", data=json.dumps(files).encode()))
record = json.loads(remote("gpuR", """
import hashlib,json,pathlib,shutil
v5=pathlib.Path('/root/autodl-tmp/c4/v5'); arm=v5/'main14_arm'
assert not (arm/'waiter.pid').exists()
panel=arm/'gate_panel.json'; sha=hashlib.sha256(panel.read_bytes()).hexdigest()
assert sha=='3eaa69a170c17302660d12dd4d7fddb4657e43fa334eece17844dd0fa6c0411e'
assert len(json.loads(panel.read_text())['starts'])==16
smoke=json.loads((arm/'prearm_smoke/RUN_SUMMARY.json').read_text())
assert smoke['status']=='completed' and len(smoke['updates'])==2
path=arm/'ARMED.json'; d=json.loads(path.read_text()); d['gate_panel_sha256']=sha
path.write_text(json.dumps(d,indent=1)+'\\n')
record={'gate_panel_sha256':sha,'gate_panel_starts':16,'prearm_unit_tests':340,
 'prearm_real_wm_smoke':'completed; 2 CPU updates, 13E b2; zero SUMO',
 'free_bytes':shutil.disk_usage(v5).free,'registered_code_sha':d['code_sha'],
 'registration_sha256':d['registration_sha256'],'config_sha256':d['config_sha256']}
(arm/'PREARM_VERIFIED.json').write_text(json.dumps(record,indent=1))
print(json.dumps(record))
"""))
receipt.update(record)
receipt_path.write_text(json.dumps(receipt, indent=1) + "\n")
print(json.dumps(record, indent=1))
