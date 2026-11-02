"""Read-only throughput from completed it2_K2 forecasts; no worker/job control."""
import json
import time


def summarize(rows, now):
    """Keep configuration variants separate; expose startup/gap censoring explicitly."""
    selected = [r for r in rows if r.get('search_config', {}).get('name') == 'it2_K2']
    groups = {}
    for r in selected:
        config = r['search_config']
        assert config['overrides'] == dict(iterations=2, screen_K=2)
        assert config['screen_H'] == 300 and config['final_K'] == 8
        assert config['full_population'] and r['wall_s'] > 0
        assert r['top8_count'] == 8
        key = f"{r['host']}/n{r['group_size']}"
        groups.setdefault(key, []).append(r)
    by_group = {}
    for key, rs in sorted(groups.items()):
        wall = sum(r['wall_s'] for r in rs)
        by_group[key] = dict(completed=len(rs), total_query_wall_s=wall,
            mean_query_wall_s=wall / len(rs),
            reciprocal_mean_query_wall_per_hour=3600 * len(rs) / wall)
    start = min((r['unix'] - r['wall_s'] for r in selected), default=None)
    hours = (now - start) / 3600 if start is not None else None
    return dict(unix=now, search_config='it2_K2', completed=len(selected),
        by_host_group_size=by_group, window_start_unix=start, window_hours=hours,
        observed_completed_per_hour=len(selected) / hours if hours else None,
        calendar_rate_contract='since first completed query start; includes rollout gaps '
            'and unfinished queries; not steady-state or a quality comparison',
        reciprocal_rate_contract='3600/mean completed query wall; not aggregate GPU throughput',
        new_sumo_sim_s=0, new_wm_searches=0, rows=selected)


def main():
    from .ls_validity_ops import remote
    from . import ops
    from .storage import write
    roots = dict(gpuA=('/root/coopmwm20261102_incident1',
                       '/root/coopmwm20261102_deadlineA2'),
                 gpuR=('/root/coopmwm20261102_concurrent3',
                       '/root/coopmwm20261102_deadlineR2'))
    rows = []
    for host, paths in roots.items():
        code = f'''from pathlib import Path
import json,hashlib
rows=[]
for name in {paths!r}:
 r=Path(name)
 for p in sorted((r/'forecasts').glob('*.json')):
  f=json.loads(p.read_text());v=f.get('result',{{}})
  if v.get('search_config',{{}}).get('name')!='it2_K2':continue
  q=json.loads((r/'inbox'/p.name).read_text())
  assert f['key']==p.stem==q['state_id']
  rows.append(dict(host={host!r},root=name,key=p.stem,
   forecast_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
   world_id=q['world_id'],time_s=q['time_s'],group_size=len(q['group']),
   search_config=v['search_config'],wall_s=f['wall_s'],unix=f['unix'],
   top8_count=len(v['collection_top8'])))
print(json.dumps(rows))'''
        rows.extend(json.loads(remote(host, code)))
    value = summarize(rows, time.time())
    path = ops.EVIDENCE / f'IT2_K2_LIVE_TIMING_{time.time_ns()}.json'
    write(path, value)
    print(json.dumps(dict(receipt=str(path), **value)))


if __name__ == '__main__':
    main()
