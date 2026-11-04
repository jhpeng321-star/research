"""CPU observable V2V continuation for the opt-in online planner.

Each imagined branch owns its radio history. Features use the live bank/runtime
builders; commands are fed back only AFTER safety/ownership, and publications
become visible at the next whole second. No WM latent/truth fields are inputs.
"""
import copy
import math

import torch

from .bank_v2v import DEFAULT_CFG, _features, _intent_features, causal_frame
from .policy_wm import load_actor
from .warm_history import sensor_rows


def snapshot(controller, frame, previous, board):
    """Capture pre-step radio inputs (not this second's newly published intents)."""
    from .v2v_intent import intent_model
    if int(controller.policy.actor[0].weight.shape[1]) == 33:
        return None
    intent = intent_model(controller.policy)
    state = dict(frame=copy.deepcopy(frame), previous_commands=dict(previous),
                 deterministic=controller.deterministic,
                 noise=dict(controller.memory),
                 generator_state=controller.generator.get_state().tolist(),
                 log_std=controller.policy.log_std.detach().cpu().tolist(), intent=None)
    if intent is not None:
        state['intent'] = dict(horizon=intent.intent_horizon, board=copy.deepcopy(board),
                               weight=intent.intent_head.weight.detach().cpu().tolist(),
                               bias=intent.intent_head.bias.detach().cpu().tolist(),
                               bounds=intent.intent_bounds.detach().cpu().tolist())
    return state


def local_snapshot(state, rows):
    """Keep only the acting ego's already-observed geometry and radio reports."""
    state = copy.deepcopy(state)
    visible = {r['vid'] for r in rows}
    cavs = {r['vid'] for r in rows if r['is_cav']}
    state['frame']['vehicles'] = [r for r in state['frame']['vehicles'] if r['vid'] in visible]
    for key in ('previous_commands', 'noise'):
        state[key] = {v: x for v, x in state[key].items() if v in cavs}
    if state['intent'] is not None:
        state['intent']['board'] = {v: x for v, x in state['intent']['board'].items() if v in cavs}
    return state


def validate(record):
    """Fail closed on missing/inconsistent radio state, including a zero actor."""
    width = len(record['policy_state'].get('actor.0.weight', [[0.]*33])[0])-33
    radio = record.get('policy_radio')
    if radio is None:
        if width:
            raise ValueError('V2V actor requires observable policy_radio state')
        return
    required = {'frame', 'previous_commands', 'deterministic', 'noise',
                'generator_state', 'log_std', 'intent'}
    if set(radio) != required:
        raise ValueError('policy radio whitelist violation')
    frame = radio['frame']
    if set(frame) != {'time_s', 'vehicles'} or frame['time_s'] != record['time_s']:
        raise ValueError('radio frame clock/whitelist violation')
    rows = []
    for row in frame['vehicles']:
        if set(row) != set(record['rows'][0]) | {'issued_acceleration_mps2'}:
            raise ValueError('radio sensor whitelist violation')
        accel = row['issued_acceleration_mps2']
        if accel is not None and not math.isfinite(accel):
            raise ValueError('nonfinite sensed acceleration')
        rows.append({k: v for k, v in row.items() if k != 'issued_acceleration_mps2'})
    if sensor_rows(rows) != record['rows']:
        raise ValueError('radio geometry differs from observable rows')
    cavs = {r['vid'] for r in rows if r['is_cav']}
    if not cavs <= set(record['policy_residuals']):
        raise ValueError('radio continuation requires every observed CAV current sample')
    for key in ('previous_commands', 'noise'):
        if not set(radio[key]) <= cavs or any(not math.isfinite(x) for x in radio[key].values()):
            raise ValueError('radio history must contain only finite observed CAV values')
    if type(radio['deterministic']) is not bool:
        raise ValueError('radio deterministic flag required')
    generator = torch.Generator()
    generator.set_state(torch.tensor(radio['generator_state'], dtype=torch.uint8))
    if len(radio['log_std']) != 1 or not all(math.isfinite(x) for x in radio['log_std']):
        raise ValueError('finite residual noise scale required')
    from .v2v_obs import V2V_DIM, v2v_intent_dim
    intent = radio['intent']
    if intent is None:
        if width not in (0, V2V_DIM):
            raise ValueError('intent actor requires timestamped intent state')
    else:
        if set(intent) != {'horizon', 'board', 'weight', 'bias', 'bounds'}:
            raise ValueError('intent state whitelist violation')
        h = intent['horizon']
        if type(h) is not int or h < 1 or width != v2v_intent_dim(horizon=h):
            raise ValueError('intent horizon/actor width mismatch')
        for key, shape in (('weight', (h, 64)), ('bias', (h,)), ('bounds', (2,))):
            values = torch.tensor(intent[key])
            if tuple(values.shape) != shape or not bool(values.isfinite().all()):
                raise ValueError('invalid intent head/bounds')
        if not intent['bounds'][0] < intent['bounds'][1]:
            raise ValueError('invalid intent bounds')
        if not set(intent['board']) <= cavs:
            raise ValueError('intent board exceeds observable CAVs')
        for timestamp, values in intent['board'].values():
            if (type(timestamp) is not int or timestamp >= record['time_s']
                    or len(values) != h or any(not math.isfinite(x) for x in values)):
                raise ValueError('noncausal or invalid intent publication')


class Continuation:
    """One instance per candidate batch; tick zero always resets all branch state."""
    def __init__(self, record, policy=None):
        validate(record)
        self.record = copy.deepcopy(record)
        if policy is None:
            with torch.random.fork_rng(devices=[]):
                policy = load_actor(record['policy_state'], 'cpu')
        self.policy = policy
        self.branches = None

    @torch.no_grad()
    def residuals(self, state, ids, tick):
        from .acceleration import PHI, RHO
        from .acceleration import encode
        from .v2v_integrate import widen_context
        if tick % 2:
            raise ValueError('radio policy advances only on whole seconds')
        batch, slots = state['s_m'].shape
        if tick == 0:
            self.branches = []
            for _ in range(batch):
                radio = copy.deepcopy(self.record['policy_radio'])
                radio['generator'] = torch.Generator().set_state(
                    torch.tensor(radio['generator_state'], dtype=torch.uint8))
                self.branches.append(radio)
        if self.branches is None or len(self.branches) != batch:
            raise ValueError('radio continuation requires a tick-zero decision')
        out = torch.zeros_like(state['s_m'])
        for b, radio in enumerate(self.branches):
            # Initial identities keep input insertion order (stable V2V ties).
            # Predicted entrants use fresh slot identities; no hidden identity data.
            names = [ids[i] if i < len(ids) else 'new:'+str(i) for i in range(slots)]
            radio['slot_names'] = names
            rows = [dict(vid=names[i], s_m=float(state['s_m'][b, i]),
                         v_mps=float(state['v_mps'][b, i]), lane=int(state['lane'][b, i]),
                         is_cav=bool(state['is_cav'][b, i]), length_m=5.)
                    for i in range(slots) if bool(state['present'][b, i])]
            now = self.record['time_s']+tick//2
            if tick == 0:
                frame = radio['frame']
            else:
                frame = causal_frame(dict(time_s=now, vehicles=rows), radio['frame'])
            context = encode(rows)
            intent = radio['intent']
            blocks = {v: (_intent_features(frame, v, radio['previous_commands'], DEFAULT_CFG, intent)
                          if intent else _features(frame, v, radio['previous_commands'], DEFAULT_CFG))
                      for v in context['members']}
            wide = widen_context(context, blocks)
            x = torch.tensor(wide['x'], dtype=torch.float32).reshape(
                -1, int(self.policy.actor[0].weight.shape[1])-4)
            latent = self.policy.broadcast(torch.tensor(context['rsu'])).expand(len(x), -1)
            hidden = self.policy.actor[:-1](torch.cat((x, latent), -1))
            mean = self.policy.actor[-1](hidden)
            if tick == 0:
                # The decision follows this second's real sample. Reuse it in
                # every arm, including stochastic policies, without resampling.
                action = [self.record['policy_residuals'].get(v, float(mean[j, 0]))
                          for j, v in enumerate(context['members'])]
            elif radio['deterministic']:
                action = mean.flatten().tolist()
            else:
                innovation = math.sqrt(RHO)*torch.randn((1, 1), generator=radio['generator'])
                innovation = innovation+math.sqrt(1-RHO)*torch.randn(mean.shape, generator=radio['generator'])
                previous = torch.tensor([[radio['noise'].get(v, 0.)] for v in context['members']]).reshape(-1, 1)
                noise = previous*PHI+torch.tensor(radio['log_std']).exp()*math.sqrt(1-PHI**2)*innovation
                action = (mean+noise).flatten().tolist()
                radio['noise'] = dict(zip(context['members'], noise.flatten().tolist()))
            for vid, value in zip(context['members'], action):
                out[b, names.index(vid)] = value
            if intent is not None:
                values = torch.nn.functional.linear(hidden, torch.tensor(intent['weight']),
                                                     torch.tensor(intent['bias'])).clamp(*intent['bounds'])
                intent['board'] = {v: (int(now), value) for v, value in
                                   zip(context['members'], values.tolist())}
            radio['frame'] = frame
        return out

    def observe_issued(self, state, command, tick):
        """Post-safety issued acceleration, not raw request or speed difference."""
        if tick % 2:
            return
        for b, radio in enumerate(self.branches):
            rows = radio['frame']['vehicles']
            # The frame is in slot order with absent slots removed.
            live = state['present'][b].nonzero().flatten().tolist()
            by_id = {row['vid']: row for row in rows}
            radio['previous_commands'] = {radio['slot_names'][slot]: float(command[b, slot])
                                          for slot in live if radio['slot_names'][slot] in by_id
                                          and by_id[radio['slot_names'][slot]]['is_cav']
                                          and by_id[radio['slot_names'][slot]]['lane'] >= 0}
