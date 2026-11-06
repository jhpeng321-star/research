import subprocess, json, os, time, shutil
SRC='/home/ubuntu/coopmwm20261106_clean3_src_7b8cbaad'
CTRL='/home/ubuntu/cpuv_setup_20261105/clean3_chain_7b8cbaad'
q=json.load(open('/home/ubuntu/cpuv_setup_20261105/clean3_tests_7b8cbaad/RESULT.json'))
assert q['returncode']==0 and q['source_commit'].startswith('7b8cbaad'), q
assert not os.path.exists('/home/ubuntu/coopmwm20261106_clean3_r3') and not os.path.exists('/home/ubuntu/coopmwm20261106_clean3_g1cert_r9')
os.mkdir(CTRL)
shutil.copy('/home/ubuntu/cpuv_setup_20261105/clean3_upload_7b8cbaad/AUTHORITY.json', CTRL+'/AUTHORITY.json')
cmd = f". /home/ubuntu/cpus_env/env.sh {SRC} && exec nice -n 19 /home/ubuntu/miniconda3/bin/python -B -u {SRC}/ops/clean3_chain.py --source {SRC} --control {CTRL}"
json.dump(dict(unix=time.time(), command=cmd, qualification=q), open(CTRL+'/DISPATCH_STARTED.json','x'))
p = subprocess.Popen(['setsid','bash','-c',cmd], stdin=subprocess.DEVNULL, stdout=open(CTRL+'/CHAIN.log','x'),
    stderr=subprocess.STDOUT, start_new_session=True, cwd='/home/ubuntu')
time.sleep(5)
st=open(f'/proc/{p.pid}/stat').read().split()
json.dump(dict(pid=p.pid, start_ticks=st[21], state=st[2], source_root=SRC, control=CTRL, status='CHAIN_STARTED', unix=time.time()), open(CTRL+'/CHAIN_DISPATCH.json','x'))
print(p.pid, st[2], open(CTRL+'/CHAIN.log').read()[-500:], os.listdir(CTRL))
