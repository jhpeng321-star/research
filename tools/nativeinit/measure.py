"""Launch ONLY the registered offline imagination probe, in task-owned scratch."""
import subprocess

SCRIPT = r'''
import hashlib,json,os,pathlib,subprocess
root=pathlib.Path('/root/autodl-tmp/c4/nativeinit_s1')
cfg=json.loads(pathlib.Path('/root/autodl-tmp/c4/v5/mainstart/mainr16/restart/CONFIG.json').read_text())
state=json.loads(pathlib.Path('/root/main14_loop/LOOP14_STATE.json').read_text())
out=root/'measure30'
assert not out.exists() and not (root/'MEASURE_PID.json').exists(), 'never relaunch an existing probe'
panel=pathlib.Path('/root/autodl-tmp/c4/v5/mainstart/mainr16/restart/gate_panel.json')
assert hashlib.sha256(panel.read_bytes()).hexdigest()=='3eaa69a170c17302660d12dd4d7fddb4657e43fa334eece17844dd0fa6c0411e'
env=dict(os.environ,**cfg['policy']['env'])
env.update(PYTHONPATH=str(root/'runtime/src')+':'+str(root/'runtime'),
           C4_GPU_MEM_FRACTION='0.40',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
           PYTHONDONTWRITEBYTECODE='1')
env['C4_ALLOWED_ROOTS']=':'.join([str(root),cfg['loop_root'],*cfg['policy']['allowed_roots']])
wm=state['accepted_wm']
assert wm['sha256']=='5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2'
argv=['/root/miniconda3/bin/python','-m','wms.control.policy14_cli','--config',cfg['policy']['config'],
      '--wm-ckpt',wm['checkpoint'],'--wm-config',wm['config'],'--real-returns',state['real_returns'],
      '--device','cuda','--seed','965140','--max-updates','30','--output-dir',str(out),
      '--native-init','--compact',*cfg['policy']['extra_args']]
with (root/'measure30.log').open('xb') as log:
    p=subprocess.Popen(argv,cwd=root/'runtime',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'start':pathlib.Path(f'/proc/{p.pid}/stat').read_text().split()[21],
         'argv':argv,'offline_only':True,'sumo_seconds':0,'allocator_fraction':.40}
(root/'MEASURE_PID.json').write_text(json.dumps(receipt,indent=1))
print(json.dumps(receipt))
'''

if __name__ == '__main__':
    import sys
    script = SCRIPT
    if '--crn' in sys.argv:
        script = script.replace("root/'runtime", "root/'runtime_measure_crn")
        script = script.replace('measure30', 'measure30_crn').replace('MEASURE_PID', 'MEASURE_CRN_PID')
    if '--final' in sys.argv:
        script = script.replace("root/'runtime", "root/'runtime_measure_final")
        script = script.replace('measure30', 'measure30_final').replace('MEASURE_PID', 'MEASURE_FINAL_PID')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'gpuR', '/root/miniconda3/bin/python -'],
                            input=script.encode(), capture_output=True, check=True)
    print(result.stdout.decode())
