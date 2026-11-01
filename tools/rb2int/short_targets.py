"""Read named recorded short-horizon clocks only; no simulation/model execution."""
import argparse
import json
from pathlib import Path

from tools.rbint.remote import ssh
from tools.rb2int.audit import ROOT
from tools.rbeval.core import digest, save_new


def extract(host, views, worlds, derive=False):
    code = f'''
from pathlib import Path
import hashlib,json
import numpy as np
BAD=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params','vtype')
def safe(p):
 p=Path(p);assert not any(x in str(p).lower() for x in BAD);return p
def load(p): return json.loads(safe(p).read_bytes())
def sha(p): return hashlib.sha256(safe(p).read_bytes()).hexdigest()
rows=[];schema=[]
for seed in {worlds!r}:
 manifest=safe(Path({views!r})/f'world_{{seed}}.json');world=load(manifest)
 for arm in ('native','ctrl0','mappo','pi','circles'):
  record=world[arm];job=safe(record['job']);clocks={{}};hashes={{}};missing=[]
  if seed=={worlds[0]} and arm=='native':
   for name in ('traj_base.npz','deploy/episode/episode.npz'):
    path=safe(job/name)
    if path.is_file():
     with np.load(path,allow_pickle=False) as f: schema.append(dict(path=str(path),keys=f.files))
   result=load(job/'result.json');schema.append(dict(result_metrics_keys=list(result['metrics'])))
   schema.append(dict(observable_names=[p.name for p in job.iterdir()
    if not any(x in p.name.lower() for x in BAD)],window_keys=list(result['metrics'].get('windows',{{}}))))
  for h in (30,60,90):
   path=job/f'cumulative_{{h}}.json'
   if path.is_file():
    c=load(path);assert c['t0_s']==300 and c['t_end_s']==300+h
    assert abs(c['team_delay_vs']-c['road_delay_vs']-c['source_wait_vs'])<1e-6
    clocks[str(h)]={{k:c[k] for k in ('team_delay_vs','road_delay_vs','source_wait_vs')}}
    hashes[str(h)]=sha(path)
   else: missing.append(h)
  road={{}};bounds={{}};road_check=None;observable_hashes={{}}
  if {derive!r}:
   ep=safe(Path({views!r})/'episodes'/record['episode']);meta=load(ep/'episode.json')
   assert meta['seed']==seed and meta['arm']==arm
   assert meta['source']['initial_state_sha256']==record['initial_state_sha256']
   arrays={{n:np.load(safe(ep/(n+'.npy')),mmap_mode='r',allow_pickle=False)
    for n in ('time_s','veh_kin','veh_mask','vid')}}
   times=arrays['time_s'];start=np.flatnonzero(times==300.)
   assert len(start)==1;start=int(start[0]);stop=start+240
   assert np.array_equal(times[start:stop+1],300+np.arange(241)*.5)
   kin=arrays['veh_kin'];mask=arrays['veh_mask'];vid=arrays['vid']
   m0=np.asarray(mask[start:stop],bool);m1=np.asarray(mask[start+1:stop+1],bool)
   ids0=vid[start:stop] if vid.ndim==2 else vid[None,:]
   ids1=vid[start+1:stop+1] if vid.ndim==2 else vid[None,:]
   cont=m0&m1&(ids0==ids1)
   before=kin[start:stop,:,4];after=kin[start+1:stop+1,:,4]
   increments=.5*np.where(m0,np.maximum(0.,1.-np.where(cont,after,before)/31.29),0.).sum(1,dtype=np.float64)
   road={{str(h):float(increments[:2*h].sum()) for h in (30,60,90,120)}}
   factual=load(job/'cumulative_120.json')
   assert sha(job/'cumulative_120.json')==meta['source']['clocks_sha256']['120']
   road_check=road['120']-factual['road_delay_vs']
   assert np.isclose(road['120'],factual['road_delay_vs'],rtol=1e-6,atol=.01),(seed,arm,road_check)
   source_max=factual['source_wait_vs'];assert source_max>=0
   bounds={{str(h):[road[str(h)],road[str(h)]+source_max] for h in (30,60,90)}}
   observable_hashes={{n:sha(ep/(n+'.npy')) for n in arrays}}
   observable_hashes['cumulative_120']=sha(job/'cumulative_120.json')
  rows.append(dict(seed=seed,arm=arm,initial_sha256=record['initial_state_sha256'],
   clocks=clocks,clock_sha256=hashes,missing_horizons=missing,job=str(job),manifest_sha256=sha(manifest),
   road_delay_vs=road,team_delay_bounds_vs=bounds,road120_reconstruction_error_vs=road_check,
   observable_hashes=observable_hashes))
print(json.dumps(dict(host={host!r},episodes=rows,observable_schema=schema,read_only=True,simulation_runs=0)))
'''
    return json.loads(ssh(host, code, timeout=180))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--derive', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    if not out.parent.is_dir() or out.exists():
        raise ValueError('exclusive receipt in existing parent required')
    parts = [extract('cpuQ', '/home/ubuntu/wmdata20261031/views', list(range(969432,969440)), args.derive),
             extract('gpuR', '/root/wmdata2/views', list(range(969550,969600)), args.derive)]
    rows = [r for p in parts for r in p['episodes']]
    save_new(out, dict(schema='rb2int-short-targets/1', episodes=rows, sources=parts,
        source_sha256=digest(__file__), evaluation_only=True, fitting=False,
        registration_sha256=digest(ROOT/'docs/rb2int/EVALUATION_REGISTRATION.json')))
    print(json.dumps(dict(episodes=len(rows), missing={h:sum(h in r['missing_horizons'] for r in rows)
        for h in (30,60,90)})))


if __name__ == '__main__':
    main()
