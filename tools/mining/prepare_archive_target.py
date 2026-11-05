from pathlib import Path
r=Path('/root/mining20261106')
assert (r/'EXPERIMENT_ID').read_text()=='S-MINING-20261106\n'
(r/'cpuv_parity.tar.gz').touch(exist_ok=False)
print('empty SFTP resume target created; zero simulation')
