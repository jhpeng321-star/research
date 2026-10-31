"""Keep the frozen v7 anonymous history/key projection, extend command memory only."""
import copy
import math

from tools.c1switch.observable import anonymous_payload as v7_anonymous
from tools.c1switch.storage import canonical

from .design import CONTROLLERS


def anonymous_payload(payload, validate_wall):
    # v7 validates its own five-way memory separately; never bypass its history
    # allowlist or introduce an HDV ID through a second serialization boundary.
    history_only = {k: payload[k] for k in ('schema', 'history')}
    result = v7_anonymous(history_only)
    times = [h['now_s'] for h in result['history']]
    if len(times) != 121 or any(b-a != .5 for a, b in zip(times, times[1:])):
        raise ValueError('genuine 60s half-second history required')
    memory = payload['controller_state']
    if set(memory) != {'incumbent', 'now_s', 'pi', 'commands', 'wall'}:
        raise ValueError('v2 command memory allowlist')
    if memory['incumbent'] not in CONTROLLERS or memory['now_s'] != times[-1]:
        raise ValueError('invalid command memory clock/incumbent')
    legacy = {k: copy.deepcopy(v) for k, v in memory.items() if k != 'wall'}
    if legacy['incumbent'] == 'wall':
        legacy['incumbent'] = 'native'
    from tools.c1switch.feedback import validate
    validate(dict(history=result['history'], controller_state=legacy))
    own = {r['vehicle_id'] for r in result['history'][-1]['self_reports']}
    if memory['incumbent'] != 'wall' and memory['wall'] is not None:
        raise ValueError('inactive wall cannot retain controller state')
    # S-WALLSW must validate its exact causal state schema and self-ID membership.
    # Equality rejects extra fields even if a plugin attempts to silently strip them.
    wall = validate_wall(copy.deepcopy(memory['wall']), own, times[-1])
    if canonical(wall) != canonical(memory['wall']):
        raise ValueError('wall memory must already be allowlisted')
    result['controller_state'] = copy.deepcopy(memory)
    # JSON finiteness including nested group memory, independent of plugin checks.
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite observable memory')
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)
    finite(result)
    return result


def future_seed(payload):
    """Only call with the fully anonymous/validated payload, never a world or RPC ID."""
    return int(canonical(payload)[:8], 16) & 0x7fffffff
