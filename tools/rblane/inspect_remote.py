"""Read only the six permitted observable arrays of the first FIT episode."""
import subprocess


def main():
    code = r'''
import json, pathlib, numpy as np, os
p=pathlib.Path('/home/ubuntu/wmdata20261031/views/episodes/w969400m1900p05t300__native')
for name in ('time_s.npy','veh_kin.npy','veh_mask.npy','is_cav.npy','vid.npy'):
    x=np.load(p/name,allow_pickle=False,mmap_mode='r')
    print(name,x.shape,x.dtype)
    if name=='veh_kin.npy':
        m=np.load(p/'veh_mask.npy',allow_pickle=False,mmap_mode='r')[120].astype(bool)
        print('sample at t300', x[120][m][:8].tolist())
    if name=='time_s.npy': print('time',x[0],x[120],x[-1])
    if name=='vid.npy': print('vid sample',x.reshape(-1)[:8].tolist())
print('load',os.getloadavg())
'''
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                    'cpuQ', '/home/ubuntu/miniconda3/bin/python -B -'],
                   input=code, text=True, check=True, timeout=30)


if __name__ == '__main__':
    main()
