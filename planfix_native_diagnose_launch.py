from pathlib import Path
from planfix_ops import remote
source=Path(__file__).with_name('planfix_native_diagnose.py').read_text()
remote('cpuV','import os,subprocess;env=dict(os.environ,LD_PRELOAD="/home/ubuntu/cpus_env/libmkl_intel_shim.so"); subprocess.run(["/home/ubuntu/miniconda3/bin/python","-B","-c",'+repr(source)+'],env=env,check=True)',timeout=25)
