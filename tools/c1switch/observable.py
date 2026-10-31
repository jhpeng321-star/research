"""Anonymous RSU assimilation; model tensors never use simulator roster identities."""
import math


def tracker_from_payload(payload):
    from wms.control.belief import BeliefConfig, BeliefTracker, CavReport, RsuDetection, Detection
    if payload.get('schema') != 'c1switch_observable/1':
        raise ValueError('observable receipt schema required')
    history = payload['history']
    times = [h['now_s'] for h in history]
    if len(times) != 121 or any(abs(b-a-.5) > 1e-8 for a,b in zip(times,times[1:])):
        raise ValueError('genuine full60s half-second observable history required')
    tracker = BeliefTracker(BeliefConfig(max_slots=192, history_steps=121))
    for row in history:
        reports = []
        for r in row['self_reports']:
            def neighbour(ahead):
                sign = 1 if ahead else -1
                seen = [d for d in row['detections'] if d['lane'] == r['lane']
                        and sign*(d['s_m']-r['s_m']) > 0]
                if not seen:
                    return None
                d = min(seen,key=lambda x:sign*(x['s_m']-r['s_m']))
                gap = max(0.,sign*(d['s_m']-r['s_m'])-5.)
                return Detection(gap,d['v_mps']-r['v_mps']) if gap <= 100 else None
            reports.append(CavReport(cav_id=r['vehicle_id'], s_m=r['s_m'], lane=r['lane'],
                                     v_mps=r['v_mps'], a_mps2=r['a_mps2'], age_s=r['age_s'],
                                     front=neighbour(True),rear=neighbour(False)))
        # The capture-side track IDs are never handed to association or the model.
        # RSU vehicles are anonymous and CAV identity comes exclusively from self.
        detections = [RsuDetection(s_m=r['s_m'], lane=r['lane'], v_mps=r['v_mps'],
                                  a_mps2=r['a_mps2'], age_s=r['age_s']) for r in row['detections']]
        if any(not math.isfinite(r.age_s) or r.age_s < 0 for r in reports+detections):
            raise ValueError('invalid observable receipt age')
        tracker.update(row['now_s'], reports, detections)
    if tracker.stats['slot_overflow']:
        raise ValueError('belief slot overflow')
    return tracker


def rollout_inputs(payload, model):
    """Native 60s history with missing sightings masked, shared by WM and BB APIs."""
    from wms.data.history import long_history_features
    import torch
    tracker = tracker_from_payload(payload)
    state = tracker.state()
    short, smask = tracker.driver_history(history_len=int(model.history_len))
    kwargs = dict(history=short, history_mask=smask)
    if getattr(model, 'long_history', False):
        raw = tracker.history(history_s=60, history_stride_s=float(model.history_stride_s))
        long, lmask = long_history_features(torch.as_tensor(raw['kin']), torch.as_tensor(raw['mask']),
            torch.as_tensor(raw['target_mask']), state['present'][0], history_s=60,
            history_stride_s=float(model.history_stride_s), dt_s=.5)
        kwargs.update(driver_history=long[None], driver_history_mask=lmask[None])
    return state, kwargs, tracker
