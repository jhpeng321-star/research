"""Run a module from an immutable remote task snapshot, with explicit argv."""
import argparse
import json
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('host', choices=('gpuR', 'gpuA', 'cpuQ'))
    p.add_argument('tag')
    p.add_argument('module')
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    home = '/home/ubuntu' if a.host == 'cpuQ' else '/root'
    root = home+'/wmvalue20261031/'+a.tag
    code = ('import sys,runpy; sys.path.insert(0,'+repr(root)+'); sys.argv='+
            repr([a.module, *a.args])+'; runpy.run_module('+repr(a.module)+",run_name='__main__')")
    command = ('nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+
               home+'/miniconda3/bin/python -B -')
    print(json.dumps(dict(host=a.host, snapshot=a.tag, module=a.module, args=a.args)), flush=True)
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', a.host, command],
                       input=code, text=True)
    raise SystemExit(r.returncode)


if __name__ == '__main__':
    main()
