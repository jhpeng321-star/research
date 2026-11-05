from pathlib import Path
for name in ('/home/ubuntu/cpus_env/env.sh','/home/ubuntu/coopmwm20261104_launchcoop_src7/runtime_src/traffic_shepherd/config.py'):
 print('QUALIFIED_ENV_SOURCE',name,flush=True)
 print(Path(name).read_text(),flush=True)
print('PHYSICAL_ADAPTER_EXISTS',Path('/home/ubuntu/e3final/runtime/baselines/circles/adapter.json').is_file(),flush=True)
