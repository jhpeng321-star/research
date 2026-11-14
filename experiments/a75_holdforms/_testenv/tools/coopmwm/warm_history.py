"""Causal observable-only rolling history, with transmitted identity continuity."""
from collections import deque
import copy
import math

FIELDS = frozenset(('vid','s_m','v_mps','lane','is_cav','length_m'))
HISTORY_S = 60.


def sensor_rows(rows):
    if any(set(r) != FIELDS for r in rows):
        raise ValueError('history sensor whitelist violation')
    if len({r['vid'] for r in rows}) != len(rows):
        raise ValueError('duplicate observed identity')
    for r in rows:
        if (not isinstance(r['vid'], str) or not isinstance(r['is_cav'], bool)
                or r['lane'] not in (-1,0,1,2)
                or not all(math.isfinite(float(r[k])) for k in ('s_m','v_mps','length_m'))
                or r['v_mps'] < 0 or r['length_m'] <= 0):
            raise ValueError('invalid sensed history row')
    return copy.deepcopy(rows)


class ObservableHistory:
    def __init__(self):
        self.frames = deque()

    def push(self, time_s, rows, entrance_counts=None):
        now = float(time_s)
        if not math.isfinite(now) or (self.frames and now <= self.frames[-1]['time_s']):
            raise ValueError('history clock must increase')
        rows = sensor_rows(rows)
        counts = None
        if entrance_counts is not None:
            if (len(entrance_counts) != 2 or any(isinstance(x,bool) or int(x) != x or x < 0 for x in entrance_counts)):
                raise ValueError('observable entrance counts require two nonnegative integers')
            counts = list(entrance_counts)
        self.frames.append(dict(time_s=now,rows=rows,entrance_counts=counts))
        while self.frames and self.frames[0]['time_s'] < now-HISTORY_S:
            self.frames.popleft()

    def snapshot(self, time_s):
        return validate_history(list(self.frames), time_s)


def validate_history(frames, time_s):
    if not isinstance(frames,list):
        raise ValueError('history frames must be an explicit list')
    previous = -math.inf
    for frame in frames:
        if set(frame) != {'time_s','rows','entrance_counts'}:
            raise ValueError('history frame whitelist violation')
        now = float(frame['time_s'])
        if not math.isfinite(now) or not time_s-HISTORY_S <= now <= time_s or now <= previous:
            raise ValueError('noncausal or unordered history')
        sensor_rows(frame['rows'])
        counts = frame['entrance_counts']
        if counts is not None and (len(counts)!=2 or any(isinstance(x,bool) or int(x)!=x or x<0 for x in counts)):
            raise ValueError('invalid observed entrance counters')
        previous = now
    if frames and frames[-1]['time_s'] != time_s:
        raise ValueError('history must end at the query clock')
    return copy.deepcopy(frames)


def arrays(record):
    """C1 layout; actual observation cadence, no interpolation or fabricated past."""
    import numpy as np
    frames = validate_history(record.get('history',[]), record['time_s'])
    if not frames:
        return None
    rows = sensor_rows(record['rows'])
    if {r['vid']:r for r in frames[-1]['rows']} != {r['vid']:r for r in rows}:
        raise ValueError('history terminal roster differs from current observation')
    ids = [r['vid'] for r in rows]
    ids += sorted({r['vid'] for f in frames for r in f['rows']}-set(ids))
    slots = {v:i for i,v in enumerate(ids)}
    kin = np.zeros((len(frames),len(ids),7),dtype=np.float32)
    mask = np.zeros(kin.shape[:2],dtype=bool)
    cav = np.zeros(len(ids),dtype=bool)
    times = np.array([f['time_s'] for f in frames],dtype=float)
    last = {}
    for t,frame in enumerate(frames):
        for row in frame['rows']:
            i = slots[row['vid']];lane=row['lane'];s=row['s_m'];speed=row['v_mps']
            # C1 causal carry through a sensed junction; never back-fill a birth.
            if lane < 0:
                if row['vid'] not in last:continue
                s,speed,lane=last[row['vid']]
            else:
                last[row['vid']]=(s,speed,lane)
            kin[t,i,0]=s;kin[t,i,1+lane]=1.;kin[t,i,4]=speed;kin[t,i,6]=row['length_m']
            if t and mask[t-1,i]:
                kin[t,i,5]=(speed-kin[t-1,i,4])/(times[t]-times[t-1])
            mask[t,i]=True;cav[i]=row['is_cav']
    vids = np.broadcast_to(np.arange(len(ids)),mask.shape).copy()
    return dict(time_s=times,kin=kin,mask=mask,vid_index=vids,ids=ids,is_cav=cav,
                target_mask=mask[:-1]&mask[1:],current_slots=len(rows))
