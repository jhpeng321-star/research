"""Source-checked longitudinal decoder around ORIGINAL FULL-v4 option_controller.

No lane surrogate is provided: native_lane_mask/native target overwrite and the
post-longitudinal WM ramp-merge delegation lack a qualified per-action G5 hook.
"""
import ast
from contextlib import contextmanager
import copy
import inspect
from unittest.mock import patch

from .wm_contract import LANE_FAMILIES, file_sha

RUNTIME_SHA = '2893e762741a61a791ce467493e0a35290e6ad4c893e2909b70bfa0fe1df50d8'


class UnsupportedLaneABI(ValueError):
    pass


def require_longitudinal(actions):
    for action in actions:
        move = action.get('wm_move', {})
        if (move.get('family') in LANE_FAMILIES or action.get('g5_move')
                or action.get('uses_lane_actions') or action.get('lane_actions')):
            raise UnsupportedLaneABI('FULL-v4 native_lane_mask + native target overwrite + '
                'post-motion ramp delegation have no qualified per-candidate G5 command hook')
        if set(action) - {'id', 'wm_move', 'wm_active_members', 'mining_family', 'solo_of'}:
            raise ValueError('unrecognized action fields; absolute profiles are not dynamic moves')
        if 'wm_move' not in action and action.get('id') not in ('policy', 'MAPPO'):
            raise ValueError('only explicit library moves or baseline comparators supported')


class U0Sampler:
    """Use the original Policy.sample AR(1) algorithm, not WM's mean-only actor.

    The exact checkpoint must have a zero terminal actor layer. Its forward mean
    is then identically zero even for V2V columns; critic/intent outputs cannot
    affect acceleration. Real noise/RNG/memory and lexical CAV order survive.
    """
    def __init__(self, continuation, current, weights, sample_method):
        import torch
        if not all(k in weights and not bool(weights[k].count_nonzero())
                   for k in ('actor.4.weight', 'actor.4.bias')):
            raise ValueError('only exact zero-actor frozen u0 is supported')
        if any(k.startswith(('p1.', 't2c.')) for k in weights):
            raise ValueError('cooperation weights are not a frozen base')
        if (weights['log_std'].numel() != 1 or not bool(torch.isfinite(weights['log_std']).all())):
            raise ValueError('one finite native acceleration log_std required')
        self.log_std = weights['log_std'].detach().cpu().clone()
        self.generator = torch.Generator()
        self.generator.set_state(torch.tensor(continuation['generator_state_after_t0'], dtype=torch.uint8))
        self.memory = copy.deepcopy(continuation['memory_after_t0'])
        self.current = dict(current)
        self.deterministic = continuation['deterministic']
        self.sample_method = sample_method

    def __call__(self, context):
        import torch
        return torch.zeros((len(context['members']), 1)), torch.tensor(0.)

    def at(self, rows, second):
        members = sorted(r['vid'] for r in rows if r['is_cav'])
        if second == 0:
            if set(members) != set(self.current):
                raise ValueError('captured t0 sample does not match live CAV roster')
            return dict(self.current)
        sample, self.memory = self.sample_method(self, {'members': members}, self.generator,
                                                self.deterministic, self.memory)
        return {v: float(x[0]) for v, x in zip(members, sample['own_action'])}


def compile_controller(original, decode):
    """Replace ONLY the inherited pre-safety acceleration decoder; fail on ABI drift."""
    tree = ast.parse(inspect.getsource(original))
    outer = tree.body[0]; matches = {'fixed': 0, 'injected': 0}
    for i, node in enumerate(outer.body):
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'fixed' for x in node.targets):
            matches['fixed'] += 1
            outer.body[i] = ast.parse('decode_state = make_decoder(candidate_actions, ids)').body[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'injected':
            matches['injected'] += 1
            node.body = ast.parse('raw = decode_state(state, accel, tick)\nreturn safe_accel(state, raw, **kw)').body
    if matches != {'fixed': 1, 'injected': 1}:
        raise ValueError('frozen option_controller injection ABI changed')
    ast.fix_missing_locations(tree)
    ns = dict(original.__globals__, make_decoder=decode)
    exec(compile(tree, '<mining-source-bound-dynamic-offset-decoder>', 'exec'), ns)
    return ns[original.__name__]


def decoder_factory(continuation, current, weights, sample_method, moves, roster, telemetry):
    def factory(actions, ids):
        require_longitudinal(actions)
        samplers = [U0Sampler(continuation, current, weights, sample_method) for _ in actions]
        parsed = [moves.Move.from_dict(a['wm_move']) if 'wm_move' in a else None for a in actions]
        for a, move in zip(actions, parsed):
            if move and (not a['wm_active_members']
                         or len(set(a['wm_active_members'])) != len(a['wm_active_members'])
                         or not set(a['wm_active_members']) <= set(move.members)
                         or not set(move.members) <= set(ids)):
                raise ValueError('move/solo role binding outside observed identities')
        held = [None] * len(actions)
        previous_tick = -1

        def decode(state, accel, tick):
            import torch
            nonlocal previous_tick
            if tick != previous_tick + 1 or accel.shape[0] != len(actions):
                raise ValueError('controller tick/batch ABI changed')
            previous_tick = tick
            out = accel.clone()
            if tick % 2 == 0:
                for b, (action, move) in enumerate(zip(actions, parsed)):
                    row = {k: v[b:b+1] if torch.is_tensor(v) and v.ndim
                           and v.shape[0] == len(actions) else v for k, v in state.items()}
                    names = ids + ['new:' + str(i) for i in range(len(ids), accel.shape[1])]
                    rows = roster(row, names)
                    own = samplers[b].at(rows, tick // 2)
                    if action['id'] == 'MAPPO':
                        own = {v: 0. for v in own}
                    table = {r['vid']: r for r in rows}
                    correction = {}
                    for vid, base in own.items():
                        offset = 0.
                        if move and vid in action['wm_active_members'] and vid in table:
                            ego = table[vid]
                            positions = {v: r['s_m'] for v, r in table.items()
                                         if abs(r['s_m'] - ego['s_m']) <= 100.}
                            offset = moves.offsets(move, tick // 2, state={'s_m': positions})[vid]
                        # ProductionLayer composes own sampled residual + move in float32.
                        value = torch.tensor(base, dtype=torch.float32)
                        if offset:
                            value = value + torch.tensor(offset, dtype=torch.float32)
                        correction[vid] = float(value)
                        if move and vid in action['wm_active_members']:
                            telemetry['active_member_seconds'] += int(offset != 0.)
                    held[b] = correction
            for b, correction in enumerate(held):
                for slot, vid in enumerate(ids + ['new:' + str(i) for i in range(len(ids), accel.shape[1])]):
                    if vid not in correction:
                        continue
                    live = bool(state['present'][b, slot] and state['is_cav'][b, slot]
                                and state['lane'][b, slot] >= 0)
                    value = correction[vid]
                    if live and value != 0.:
                        # Native requests use Python float(base) + float(float32 residual).
                        out[b, slot] = (accel[b, slot].double() + value).to(accel.dtype)
            return out
        return decode
    return factory


@contextmanager
def dynamic_executor(rt, query, weights, *, telemetry=None):
    """In-process patch only; original safety/cap/bridge/WM and CRN untouched.

    Use full-roster, unoptimized evaluate until this NEW decoder has its own
    full-vs-batch trajectory qualification. Original gates run WITHOUT this patch.
    """
    from tools.wmteach import runtime
    from tools.coopmwm import plan_wm, moves
    from tools.coopmwm.acceleration import Policy
    if file_sha(runtime.__file__) != RUNTIME_SHA:
        raise ValueError('original FULL-v4 runtime source SHA mismatch')
    stats = telemetry if telemetry is not None else {'active_member_seconds': 0}
    original = runtime.option_controller
    make_decoder = decoder_factory(query['base_continuation'], query['record']['policy_residuals'],
                                   weights, Policy.sample, moves, runtime.roster, stats)
    compiled = compile_controller(original, make_decoder)

    @contextmanager
    def executor(horizon=300, policy=None, trace=None, final_chain=False):
        if policy is not None or trace is not None or final_chain:
            raise ValueError('mean-only policy/trace/changed command chain not qualified here')
        with patch.object(runtime, 'option_controller', compiled):
            yield
    with patch.object(plan_wm, 'plan_executor', executor):
        yield stats
