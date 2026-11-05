"""Additive native-lane ownership adapter. No imports boot a model or a service.

The original bridge is compiled IN MEMORY with two lane insertion boundaries:
native_lane_mask excludes only safety-approved command slots; after native
target assignment those slots receive the explicit target. The original old-
and target-lane acceleration safety, execution cap and final WM conflict path
remain intact. Absent/blocked actors and every nonactor retain native lanes.

Use AFTER the running coordinator's original gates, in a separately bound later
qualification. This module neither accesses that worker nor dispatches work.
"""
import ast
import copy
from contextlib import contextmanager
import inspect
from unittest.mock import patch

from .wm_adapter import RUNTIME_SHA, U0Sampler, require_longitudinal
from .wm_contract import file_sha
from .wm_lane_schedule import validate_schedule, targets_at, offsets_at

SOURCE_PINS = {
    'bridge': '053bb01eaca21107a62b96b3d8c0e8a93f92395200e9676da50961c79f213a6c',
    'lanes': 'f3e16476e58341f649b08f071530b88c66db3fb5283a4cca511d685a3400a230',
    'cav': 'f249b481a2f4af3766ddc2a06a8f25bc261796f9abc36fc908806a6e08ac0e71',
    'wms': 'f55534e7d70bafc6e9b01886555d36e195d32dd115e8bb3ebea7cd62496971fb',
}
KIN = ('s_m', 'v_mps', 'lane', 'present', 'is_cav', 'length_m')


def check_actions(actions, ids):
    for a in actions:
        if 'wm_role_schedule' not in a:
            require_longitudinal([a])
            continue
        if set(a) - {'id', 'wm_role_schedule', 'wm_active_members', 'solo_of'}:
            raise ValueError('role schedule cannot masquerade as a G5/profile plan')
        s = validate_schedule(a['wm_role_schedule'])
        active = a['wm_active_members']
        if (not active or len(set(active)) != len(active) or not set(active) <= set(s['members'])
                or not set(s['members']) <= set(ids)):
            raise ValueError('role/solo identities must bind the original observed roster')


class LaneRuntime:
    """One rollout callback's deterministic schedule, pure pre-forward mask.

    coverage is a producer-bound map {observed_actor_id: [observed lane ids]}.
    It is NEVER inferred from absent rows. MAIN must establish the commanded
    actor's own sensing coverage, not reuse another ego's partial packet.
    Imagined sensing uses only six kinematic fields clipped to that actor100m.
    """
    def __init__(self, actions, ids, coverage, lanes, telemetry=None):
        check_actions(actions, ids)
        self.actions, self.ids = copy.deepcopy(actions), list(ids)
        self.coverage = copy.deepcopy(coverage)
        if (not isinstance(coverage, dict) or set(coverage) - set(ids)
                or any(not isinstance(v, (list, tuple)) or len(set(v)) != len(v)
                       or any(type(x) is not int or x not in (0, 1, 2) for x in v)
                       for v in coverage.values())):
            raise ValueError('explicit actor sensing coverage map required (empty denies all)')
        self.lanes = lanes
        self.rules = lanes.LaneRules()
        # Original G5 rules use no inferred hidden fields for command support.
        if getattr(self.rules, 'secure_gap_hook', '') or getattr(self.rules, 'consequence_mask', False):
            raise ValueError('observable-only G5 LaneRules defaults changed')
        self.dwell = lanes.LaneDwell(self.rules.dwell_s, .5)
        self.tick = 0
        self.held = []
        self.pending = []
        self.events = telemetry if telemetry is not None else []

    def sensor_state(self, state):
        import torch
        if any(k not in state for k in KIN):
            raise ValueError('missing observable lane geometry; never guess lengths/coverage')
        result = {k: state[k] for k in KIN}
        shape = result['s_m'].shape
        if (len(shape) != 2 or shape[0] != len(self.actions) or shape[1] < len(self.ids)
                or any(v.shape != shape for v in result.values())):
            raise ValueError('lane roster/batch ABI drift')
        present = result['present'].bool()
        if any(not bool(torch.isfinite(result[k][present]).all()) for k in ('s_m', 'v_mps', 'length_m')):
            raise ValueError('nonfinite observable geometry')
        if (bool((result['v_mps'][present] < 0).any()) or bool((result['length_m'][present] <= 0).any())
                or bool(((result['lane'][present] < 0) | (result['lane'][present] > 2)).any())):
            raise ValueError('invalid sensed lane/speed/length geometry')
        return result

    def planned(self, state):
        """Pure: no RNG, counter increments, telemetry or dwell-memory mutation."""
        import torch
        sensed = self.sensor_state(state)
        next_dwell = copy.deepcopy(self.dwell)
        ages = next_dwell.step(sensed)
        mask = torch.zeros_like(sensed['present'], dtype=torch.bool)
        targets = sensed['lane'].long().clone()
        attempts = []
        if self.tick % 2:
            for b, slot, target in self.held:
                if bool(sensed['present'][b, slot] and sensed['is_cav'][b, slot]
                        and sensed['lane'][b, slot] == target):
                    mask[b, slot] = True
                    targets[b, slot] = target
            return mask, targets, attempts, next_dwell
        for b, a in enumerate(self.actions):
            if 'wm_role_schedule' not in a:
                continue
            requested = targets_at(a['wm_role_schedule'], self.tick // 2, a['wm_active_members'])
            for vid, (source, target) in requested.items():
                slot = self.ids.index(vid)
                reason = None
                if target not in self.coverage.get(vid, ()):
                    reason = 'missing_actor_target_lane_coverage'
                elif not bool(sensed['present'][b, slot] and sensed['is_cav'][b, slot]):
                    reason = 'absent_or_not_cav'
                elif int(sensed['lane'][b, slot]) != source:
                    reason = 'source_lane_changed'
                else:
                    # An actor's own sensor packet includes ALL local obstacles,
                    # not only the two acquired CAVs. No inferred driver fields.
                    local = {k: v[b:b+1] for k, v in sensed.items()}
                    local['present'] = local['present'].bool() & (
                        local['s_m'] - local['s_m'][0, slot]).abs().le(100.)
                    feasible = self.lanes.lane_feasibility(local, ages[b:b+1], self.rules)
                    choices = torch.zeros_like(local['lane'], dtype=torch.long)
                    choices[0, slot] = 1  # both registered commands are LEFT
                    acting = torch.zeros_like(local['present'], dtype=torch.bool)
                    acting[0, slot] = True
                    assigned, changed = self.lanes.apply_lane_choices(
                        local, choices, feasible, self.rules, acting=acting)
                    if bool(changed[0, slot]) and int(assigned[0, slot]) == target:
                        mask[b, slot] = True
                        targets[b, slot] = target
                    else:
                        reason = 'original_lane_safety_or_dwell_or_edge'
                attempts.append(dict(batch=b, action=a['id'], actor=vid, tick=self.tick,
                                     source_lane=source, target_lane=target,
                                     accepted_in_imagination=reason is None, reason=reason))
        return mask, targets, attempts, next_dwell

    def native_mask(self, state, original):
        mask, _, _, _ = self.planned(state)
        return original & ~mask

    def apply(self, state, native_targets):
        import torch
        mask, targets, attempts, dwell = self.planned(state)
        for event in self.pending:
            b, slot = event['batch'], self.ids.index(event['actor'])
            event['next_halfstep_lane_confirmed'] = bool(
                state['present'][b, slot] and state['lane'][b, slot] == event['target_lane'])
        self.pending = [event for event in attempts if event['accepted_in_imagination']]
        self.events.extend(attempts)
        self.dwell = dwell
        self.held = [(b, self.ids.index(e['actor']), e['target_lane'])
                     for e in attempts if e['accepted_in_imagination'] for b in [e['batch']]] if self.tick % 2 == 0 else []
        self.tick += 1
        intent = (targets - state['lane']).to(state['s_m'].dtype)
        return torch.where(mask, targets, native_targets), mask, intent


def _compile(tree, original, namespace, filename):
    import linecache
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree) + '\n'
    linecache.cache[filename] = (len(text), None, text.splitlines(keepends=True), filename)
    ns = dict(original.__globals__, **namespace)
    exec(compile(text, filename, 'exec'), ns)
    return ns[original.__name__]


def compile_bridge(original):
    """Retain frozen bridge math; patch only the exact lane ownership boundaries."""
    tree = ast.parse(inspect.getsource(original)); fn = tree.body[0]
    fn.args.kwonlyargs.append(ast.arg(arg='mining_lane'))
    fn.args.kw_defaults.append(ast.Constant(None))
    counts = {'mask': 0, 'assignment': 0}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'native_lane_mask':
            for ret in ast.walk(node):
                if isinstance(ret, ast.Return):
                    expr = ast.unparse(ret.value)
                    ret.value = ast.parse('mining_lane.native_mask(state, (' + expr + ')) '
                        'if mining_lane is not None else (' + expr + ')', mode='eval').body
                    counts['mask'] += 1
        if isinstance(node, ast.FunctionDef) and node.name == 'controller':
            for block in ast.walk(node):
                if isinstance(block, ast.If) and ast.unparse(block.test) == 'native_lane_change':
                    matches = [i for i, stmt in enumerate(block.body) if isinstance(stmt, ast.Assign)
                               and ast.unparse(stmt).startswith('lane_now = torch.where(native_lanes &')]
                    for i in reversed(matches):
                        counts['assignment'] += 1
                        block.body[i+1:i+1] = ast.parse('''
if mining_lane is not None:
    lane_now, mining_owned, mining_intent = mining_lane.apply(state, lane_now)
    native_lanes = native_lanes & ~mining_owned
    intent = torch.where(mining_owned, mining_intent, intent)
    if log_entry is not None:
        log_entry['lane_changed'] = log_entry.get('lane_changed', torch.zeros_like(mining_owned)) | mining_owned
''').body
    if counts != {'mask': 2, 'assignment': 1}:
        raise ValueError('frozen bridge lane ABI drift: ' + str(counts))
    return _compile(tree, original, {}, '<mining-role-lane-bridge-v1>')


def compile_option(original, make_decoder, make_lane_runtime):
    tree = ast.parse(inspect.getsource(original)); fn = tree.body[0]
    counts = {'ids': 0, 'fixed': 0, 'injected': 0, 'bridge': 0}
    for i in range(len(fn.body) - 1, -1, -1):
        stmt = fn.body[i]
        if isinstance(stmt, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'ids' for x in stmt.targets):
            counts['ids'] += 1
            fn.body[i+1:i+1] = ast.parse('mining_lane = make_lane_runtime(candidate_actions, ids)').body
        if isinstance(stmt, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'fixed' for x in stmt.targets):
            counts['fixed'] += 1
            fn.body[i] = ast.parse('decode_state = make_decoder(candidate_actions, ids)').body[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'injected':
            counts['injected'] += 1
            node.body = ast.parse('raw = decode_state(state, accel, tick)\nreturn safe_accel(state, raw, **kw)').body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'make_cav_controller':
            counts['bridge'] += 1
            node.keywords.append(ast.keyword(arg='mining_lane', value=ast.Name(id='mining_lane', ctx=ast.Load())))
    if counts != {'ids': 1, 'fixed': 1, 'injected': 1, 'bridge': 1}:
        raise ValueError('frozen option controller lane ABI drift: ' + str(counts))
    return _compile(tree, original, dict(make_decoder=make_decoder, make_lane_runtime=make_lane_runtime),
                    '<mining-role-lane-option-v1>')


def raw_decoder_factory(query, weights, sample_method, moves, roster, telemetry):
    """Exact u0 sampling plus per-role float32 composition; no policy approximation."""
    def factory(actions, ids):
        check_actions(actions, ids)
        samplers = [U0Sampler(query['base_continuation'], query['record']['policy_residuals'],
                              weights, sample_method) for _ in actions]
        parsed = [moves.Move.from_dict(a['wm_move']) if 'wm_move' in a else None for a in actions]
        for a, move in zip(actions, parsed):
            if move and (not a['wm_active_members'] or not set(a['wm_active_members']) <= set(move.members)
                         or not set(move.members) <= set(ids)):
                raise ValueError('longitudinal role binding mismatch')
        held = [None] * len(actions); previous = -1
        def decode(state, accel, tick):
            import torch
            nonlocal previous
            if tick not in (previous, previous + 1) or accel.shape[0] != len(actions):
                raise ValueError('callback tick/batch ABI drift')
            fresh = tick != previous
            previous = tick
            names = ids + ['new:' + str(i) for i in range(len(ids), accel.shape[1])]
            out = accel.clone()
            # A changed lane invokes ORIGINAL safety in target AND old lanes.
            # Both calls must reuse one sampled control decision/RNG advance.
            if tick % 2 == 0 and fresh:
                for b, (a, move) in enumerate(zip(actions, parsed)):
                    row = {k: v[b:b+1] if torch.is_tensor(v) and v.ndim and v.shape[0] == len(actions)
                           else v for k, v in state.items()}
                    rows = roster(row, names); table = {r['vid']: r for r in rows}
                    own = samplers[b].at(rows, tick // 2)
                    if a['id'] == 'MAPPO': own = {v: 0. for v in own}
                    boxes = offsets_at(a['wm_role_schedule'], tick // 2, a['wm_active_members']) if 'wm_role_schedule' in a else {}
                    held[b] = {}
                    for vid, base in own.items():
                        offset = boxes.get(vid, 0.)
                        if move and vid in a['wm_active_members']:
                            positions = {v: r['s_m'] for v, r in table.items()
                                         if abs(r['s_m'] - table[vid]['s_m']) <= 100.}
                            offset = moves.offsets(move, tick // 2, state={'s_m': positions})[vid]
                        value = torch.tensor(base, dtype=torch.float32)
                        if offset: value = value + torch.tensor(offset, dtype=torch.float32)
                        held[b][vid] = float(value)
            for b, corrections in enumerate(held):
                for i, vid in enumerate(names):
                    value = corrections.get(vid, 0.)
                    if value and bool(state['present'][b, i] and state['is_cav'][b, i] and state['lane'][b, i] >= 0):
                        out[b, i] = (accel[b, i].double() + value).to(accel.dtype)
            return out
        return decode
    return factory


@contextmanager
def lane_executor(rt, query, weights, *, coverage_by_member, telemetry=None):
    """Install only AFTER original gates. Full-roster reference evaluator only.

    Context patches restore even on exceptions. No bootstrap, gate, model call,
    worker access, filesystem edit, network action or simulator is performed.
    MAIN owns later trajectory/no-op/solo qualification and native deployment.
    """
    from tools.wmteach import runtime
    from tools.coopmwm import plan_wm, moves
    from tools.coopmwm.acceleration import Policy
    from wms.control import bridge, lanes
    from wms.model import cav, wms
    if file_sha(runtime.__file__) != RUNTIME_SHA:
        raise ValueError('frozen FULL-v4 option source changed')
    for name, module in (('bridge', bridge), ('lanes', lanes), ('cav', cav), ('wms', wms)):
        if file_sha(module.__file__) != SOURCE_PINS[name]:
            raise ValueError('frozen lane pipeline source changed: ' + name)
    # wm_lane04: the bound FULL-v4 WmSModel has merge_pivot=None (no first-passage
    # pivot module); the lane ownership patch never touches the merge law, so only
    # the backend / native-lane bridge flags are required. The lane-OFF identity
    # qualification (exact scores vs the unpatched executor) still gates ranking.
    if (runtime.BACKEND != 'FULL-v4' or rt.mp.native_lanes is not True
            or rt.mp.residual_on_native):
        raise ValueError('only original FULL-v4 native-lane bridge supported')
    events = telemetry if telemetry is not None else []
    factory = lambda actions, ids: LaneRuntime(actions, ids, coverage_by_member, lanes, events)
    raw = raw_decoder_factory(query, weights, Policy.sample, moves, runtime.roster, events)
    option = compile_option(runtime.option_controller, raw, factory)
    patched_bridge = compile_bridge(bridge.make_cav_controller)
    @contextmanager
    def executor(horizon=600, policy=None, trace=None, final_chain=False):
        if policy is not None or trace is not None or final_chain:
            raise ValueError('explicit u0 sampler only; mean-only/alternate-chain path refused')
        with patch.object(runtime, 'option_controller', option), patch.object(bridge, 'make_cav_controller', patched_bridge):
            yield
    with patch.object(plan_wm, 'plan_executor', executor):
        yield events


def evaluate_lane_grid(rt, query, weights, actions, *, coverage_by_member, H=600, horizons=None):
    """Explicit later-call API: real K8/H600, original causal memory and CRN.

    Does not bootstrap, qualify, dispatch or write files. MAIN must first bind
    this source and qualify the new decoder's no-op/trajectory/solo behavior.
    Importing this function does not run it. Lane events are WM predictions,
    never native/SUMO execution or bank-admission receipts.
    """
    from .wm_contract import HORIZONS, validate_query
    validate_query(query)
    if (not 2 <= len(actions) <= 200 or len({a['id'] for a in actions}) != len(actions)
            or not {'policy', 'MAPPO'} <= {a['id'] for a in actions}):
        raise ValueError('bounded unique slate with BOTH base comparators required')
    from tools.coopmwm import plan_wm
    from tools.coopmwm.plan_service import request_memory
    record = copy.deepcopy(query['record'])
    check_actions(actions, [r['vid'] for r in record['rows']])
    memory = request_memory(rt, record, {'memory_mode': 'warm'})
    horizons = tuple(HORIZONS if horizons is None else horizons)
    with lane_executor(rt, query, weights, coverage_by_member=coverage_by_member) as events:
        costs = plan_wm.evaluate(rt, record, actions, 8, H, chunk_actions=1,
                                trim=False, pad_actions=False, horizons=horizons,
                                memory=memory, speed2=False, transfers=False)
    if H == 600 and horizons == tuple(HORIZONS):
        from .wm_prescreen import validate_scores
        validate_scores(costs, actions)
    return dict(scores=costs, lane_events=events, K=8, H=H,
                horizons=list(horizons), schema='mining-role-lane-wm-result-v1',
                native_admissions=0, new_sumo_sim_s=0,
                interpretation='WM predictions only; new lane-decoder qualification required')
