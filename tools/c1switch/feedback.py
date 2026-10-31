"""Observable controller memory: explicit fields, self IDs only, no simulator state."""
import math

from .design import CONTROLLERS


def capture(backend):
    payload = backend.belief.payload()
    own = {r['vehicle_id'] for r in payload['history'][-1]['self_reports']}
    env, controller = backend.env, backend.handover.controller
    memory = dict(incumbent=backend.handover.name, now_s=float(env.now), pi={}, commands={})
    if backend.handover.name == 'pi' and env.pi_session is not None:
        adapter = env.pi_session.adapter
        if adapter.stopped or adapter._pending or adapter.pi._pending:
            raise ValueError('PI has uncommitted/stopped state at decision')
        memory['pi'] = {vid:list(adapter.pi._state[vid]) for vid in sorted(own)
                        if vid in adapter._initialized and vid in adapter.pi._state}
    if backend.handover.name == 'mappo' and controller is not None:
        mem = controller.exe.memory
        if mem.present is not None:
            # Slot identity is used solely to recover our own previously issued
            # command. Neither raw roster nor any HDV slot is serialized.
            ids = env.builder.vehicle_ids
            for i, index in enumerate(env.frame.vid_index):
                if index < 0 or ids[int(index)] not in own:
                    continue
                if (controller.previous is not None and controller.previous[i] == index
                        and bool(mem.present[0, i])):
                    memory['commands'][ids[int(index)]] = [float(mem.accel[0, i]), float(mem.intent[0, i])]
    payload['controller_state'] = memory
    validate(payload)
    return payload


def validate(payload):
    mem = payload['controller_state']
    if set(mem) != {'incumbent','now_s','pi','commands'} or mem['incumbent'] not in CONTROLLERS:
        raise ValueError('controller memory allowlist')
    if mem['now_s'] != payload['history'][-1]['now_s']:
        raise ValueError('controller memory clock')
    own = {r['vehicle_id'] for r in payload['history'][-1]['self_reports']}
    for key, arm in [('pi','pi'),('commands','mappo')]:
        if set(mem[key])-own or (mem[key] and mem['incumbent'] != arm):
            raise ValueError('memory must belong to live self-reporting incumbent CAVs')
        if any(len(v) != 2 or any(not math.isfinite(x) for x in v) for v in mem[key].values()):
            raise ValueError('finite causal controller state required')
    if any(abs(v[1]-(mem['now_s']-1)) > 1e-8 for v in mem['pi'].values()):
        raise ValueError('PI recurrence must continue from preceding control tick')
    return mem


def seed_state(state, tracker, payload, arm):
    """Bridge command history resets on change, retains causal commands otherwise."""
    import torch
    mem = validate(payload)
    state['cmd_accel_mps2'] = torch.zeros_like(state['s_m'])
    state['lane_intent'] = torch.zeros_like(state['s_m'])
    if arm == mem['incumbent'] == 'mappo':
        slots = {tr.cav_id:tr.slot for tr in tracker.tracks.values() if tr.is_cav}
        for vid, values in mem['commands'].items():
            slot = slots[vid]
            state['cmd_accel_mps2'][0,slot], state['lane_intent'][0,slot] = values
    return state


def seed_pi(timeline, tracker, payload, arm):
    mem = validate(payload)
    if arm == mem['incumbent'] == 'pi':
        slots = {tr.cav_id:tr.slot for tr in tracker.tracks.values() if tr.is_cav}
        for vid, (raw, last) in mem['pi'].items():
            key = f'0:{slots[vid]}'
            timeline.pi._state[key] = (raw, last-mem['now_s'])
            timeline.pi._generation[key] = 1
            timeline.pi_active.add(key)
