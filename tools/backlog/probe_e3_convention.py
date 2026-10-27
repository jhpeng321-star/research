"""S-BACKLOG probe: which clock convention reproduces e3long cumulative clocks.

Read-only. Computes, over the [t0, t0+300] window of a raw e3long tape:
  road  = .5 * ((1 - v/31.29)+ * mask).sum() over candidate tick alignments
  source= .5 * backlog.sum() for (a) interval waiting-time integral, (b) integer count
and compares with the job's cumulative_300.json (authoritative)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wms.data.c4_episodes import read_requests  # noqa: E402

tape, cum = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
with np.load(tape / "episode.npz", allow_pickle=False) as d:
    ep = {k: d[k] for k in d.files}
meta = json.loads(str(ep["meta_json"]))
t0, H, DT = float(meta["warmup_s"]), 300, 0.5
times = np.asarray(ep["time_s"], np.float64)
present = np.asarray(ep["present"], bool)
vid = np.asarray(ep["vid_index"], np.int32)
v = np.where(present, ep["v_mps"], 0.0)

names, depart, entrance = read_requests(tape / "scenario" / "requests.rou.xml")
first = {}
for t in range(len(times)):
    for i in vid[t][present[t]].tolist():
        if i >= 0 and i not in first:
            first[i] = float(times[t])
by_name = {str(n): first[i] for i, n in enumerate(ep["vehicle_ids"]) if i in first}
ins = np.asarray([by_name.get(str(n), np.inf) for n in names], np.float64)
dep, ent = np.asarray(depart, np.float64), np.asarray(entrance, np.int64)

tick = int(np.searchsorted(times, t0 - 1e-6))
n = int(round(H / DT))
print(f"tape t0={t0} tick={tick} times[{tick}]={times[tick]} n_req={len(names)} "
      f"never_inserted={(np.isposinf(ins)).sum()}")
target_road = float(cum["road_delay_vs"])
target_src = float(cum["source_wait_vs"])
for kin_off, q_off, tag in ((1, 0, "loop14 (kin+1, queue+0)"), (0, 0, "aligned (kin+0)"),
                            (1, 1, "(kin+1, queue+1)")):
    kin_sl = slice(tick + kin_off, tick + kin_off + n)
    road = .5 * np.maximum(0., 1. - v[kin_sl] / 31.29)[present[kin_sl]].sum()
    q_sl = slice(tick + q_off, tick + q_off + n)
    backlog_int = np.zeros((n, 2))
    backlog_intgr = np.zeros((n, 2))
    for j, tt in enumerate(range(n)):
        now = float(times[tick + q_off + tt])
        waiting = np.maximum(0.0, np.minimum(ins, now + DT) - np.maximum(dep, now))
        backlog_intgr[tt] = [waiting[ent == e].sum() / DT for e in (0, 1)]
        due = dep <= now + 1e-9
        backlog_int[tt] = [(due & (ins > now + 1e-9) & (ent == e)).sum() for e in (0, 1)]
    src_int = .5 * backlog_intgr.sum()
    src_cnt = .5 * backlog_int.sum()
    print(f"[{tag}] road={road:.4f} vs {target_road:.4f} (d={road - target_road:+.4f}) | "
          f"src_integral={src_int:.4f} src_count={src_cnt:.4f} vs {target_src:.4f} "
          f"(d_int={src_int - target_src:+.4f} d_cnt={src_cnt - target_src:+.4f})")
