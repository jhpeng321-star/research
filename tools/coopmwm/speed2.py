"""CPU-only fixed-fixture profiler and old/new parity gate (never SUMO)."""
import argparse
import ast
import cProfile
import inspect
import importlib.util
import json
import os
from pathlib import Path
import pstats
import sys
import time


def cpu_function(fn):
    """Translate device placement only; preserve all computation and draws."""
    tree = ast.parse(inspect.getsource(fn))
    class CPU(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == 'set_per_process_memory_fraction':
                    return None
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if node.value == 'cuda':
                return ast.copy_location(ast.Constant('cpu'), node)
            return node
    tree = CPU().visit(tree)
    ast.fix_missing_locations(tree)
    ns = dict(fn.__globals__)
    exec(compile(tree, '<speed2-cpu-' + fn.__name__ + '>', 'exec'), ns)
    return ns[fn.__name__]


def setup(root):
    gpu = os.environ.get('SPEED2_DEVICE') == 'cuda'
    assert gpu or os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA before CPU Python starts'
    import torch
    from tools.wmteach import runtime
    from . import plan_wm
    from .wm import cost_adapter
    torch.set_num_threads(1)
    rt = runtime.bootstrap() if gpu else cpu_function(runtime.bootstrap)()
    if gpu:
        # Warm buffers need the registered persistent-worker allocation, applied
        # after the frozen bootstrap's smaller inherited8GiB cap.
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('SPEED2_MEMORY_FRACTION','.65')))
    if not gpu:runtime.move = cpu_function(runtime.move)
    rt.frozen.costs = cost_adapter(rt.frozen.costs, json.loads((root/'FROZEN_COSTS.json').read_text())['code'])
    original = plan_wm.adapted_futures
    def adapted(horizons):
        ns = original(horizons)
        # This function was AST-generated, so change its constant device literal
        # through its code object, without depending on inspect's synthetic source.
        import types
        fn = ns['futures']
        code = fn.__code__.replace(co_consts=tuple('cpu' if x == 'cuda' else x for x in fn.__code__.co_consts))
        ns['futures'] = types.FunctionType(code, ns, fn.__name__, fn.__defaults__)
        return ns
    if not gpu:plan_wm.adapted_futures = adapted
    source = json.loads((root/'INPUTS.json').read_text())['records'][int(os.environ.get('SPEED2_FIXTURE_INDEX','0'))]
    commands = {r['vid']: 0. for r in source['rows'] if r['is_cav']}
    record = dict(state_id=source['state_id'], world_id=source['world_id'], rows=source['rows'],
                  group=source['group'], time_s=source['time_s'], commands=commands,
                  policy_residuals={v: -.25 for v in commands})
    from .plans import baseline_actions, family_seeds
    bounds = (float(rt.mp.policy.accel_min), float(rt.mp.policy.accel_max))
    acts = baseline_actions(record) + family_seeds(record['rows'], record['group'], commands, bounds)[:2]
    # Some qualified rosters have no inherited family for their particular
    # selected pair. Still exercise their ramp/background geometry with fixed
    # absolute profiles, without drawing anything or changing search behaviour.
    while len(acts) < 4:
        j = len(acts)-2
        values = (-1.,.5) if j == 0 else (.5,-1.)
        acts.append(dict(id=f'fixture_profile_{j}',members=record['group'],duration_s=60,
                         knots=[[max(bounds[0],min(bounds[1],v))]*7 for v in values]))
    assert len(acts) == 4
    if os.environ.get('SPEED2_WARM')=='1':
        from .speed2_memory import native_fixture
        record,rt.speed2_memory=native_fixture(rt)
        commands=record['commands']
    group_size=int(os.environ.get('SPEED2_GROUP_SIZE','0'))
    duration=int(os.environ.get('SPEED2_DURATION','60'))
    if group_size or os.environ.get('SPEED2_WARM')=='1':
        group_size=group_size or 2
        record['group']=list(commands)[:group_size]
        assert len(record['group'])==group_size
        acts=baseline_actions(record)
        for j in range(2):
            acts.append(dict(id=f'long_group_{j}',members=record['group'],duration_s=duration,
                knots=[[(-.7 if (i+t+j)%2 else .4) for t in range(duration//10+1)]
                       for i in range(group_size)]))
    return rt, record, acts


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2)


class LineTimer:
    """Statement wall timing for selected Python hotspots, including tensor ops."""
    def __init__(self):
        self.active = {}
        self.totals = {}

    def trace(self, frame, event, arg):
        code = frame.f_code
        if Path(code.co_filename).name not in ('lc2013_geometry.py', 'lc2013.py', 'lanes.py'):
            return None
        now = time.perf_counter()
        previous = self.active.pop(id(frame), None)
        if previous is not None:
            key, start = previous
            seconds, count = self.totals.get(key, (0., 0))
            self.totals[key] = (seconds + now-start, count+1)
        if event != 'return':
            self.active[id(frame)] = ((code.co_filename, code.co_name, frame.f_lineno), now)
        return self.trace

    def result(self):
        functions = {}
        for (file, name, line), (seconds, count) in self.totals.items():
            functions[file, name] = functions.get((file, name), 0.) + seconds
        return [dict(file=file, name=name, line=line, seconds=seconds, calls=count,
                     percent_of_function=100*seconds/functions[file, name])
                for (file, name, line), (seconds, count) in
                sorted(self.totals.items(), key=lambda item: -item[1][0])[:60]]


def reference(root):
    base = root.parent/'pytools/tools/coopmwm'
    def load(name, filename):
        spec = importlib.util.spec_from_file_location('tools.coopmwm._speed2_'+name, base/filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    old = load('reference', 'plan_wm.py')
    old.FusedModel = load('fused_reference', 'fused.py').FusedModel
    from . import plan_wm
    old.adapted_futures = plan_wm.adapted_futures
    return old.evaluate


def parity(root, H, K, width):
    import torch
    print('BOOTSTRAP',os.environ.get('SPEED2_DEVICE','cpu'), flush=True)
    rt, record, acts = setup(root)
    from .plan_wm import evaluate
    old = reference(root)
    memory=getattr(rt,'speed2_memory',None)
    tapes = []
    costs = rt.frozen.costs
    def capture(roll, entries, initial):
        tapes.append({k: roll[k].detach().cpu().clone() for k in ('positions','speeds','lanes','masks','accel')})
        return costs(roll, entries, initial)
    rt.frozen.costs = capture
    hs = tuple(h for h in (60,120,300) if h <= H)
    def sync():
        if os.environ.get('SPEED2_DEVICE') == 'cuda':torch.cuda.synchronize()
    sync();start = time.perf_counter()
    print('REFERENCE', H, K, width, flush=True)
    from contextlib import nullcontext
    with (memory.context(rt,record) if memory is not None else nullcontext()):
        before = old(rt, record, acts, K, H, chunk_actions=width, trim=False,
                     pad_actions=width, horizons=hs)
    sync();old_wall = time.perf_counter()-start
    expected = tapes[:]
    tapes.clear()
    sync();start = time.perf_counter()
    print('OPTIMIZED', H, K, width, flush=True)
    after = evaluate(rt, record, acts, K, H, chunk_actions=width, trim=False,
                     pad_actions=width, horizons=hs, speed2=True,memory=memory)
    sync();new_wall = time.perf_counter()-start
    worst = max(abs(x['team_delay']-y['team_delay']) for h in before for a in before[h]
                for x,y in zip(before[h][a], after[h][a]))
    rank = lambda values: sorted(values, key=lambda a: (sum(v['team_delay'] for v in values[a]), a))
    rankings = all(rank(before[h]) == rank(after[h]) for h in before)
    draws = all(x['draw_id']==y['draw_id'] for h in before for a in before[h]
                for x,y in zip(before[h][a],after[h][a]))
    from .speed2_runtime import trajectory_errors
    errors = trajectory_errors(expected,tapes)
    result = dict(pass_=worst<=.001 and rankings and draws and max(errors.values())<=.001,
        max_cost_error_veh_s=worst, identical_candidate_ranking=rankings,
        identical_draw_ids=draws,
        trajectory_errors=errors, reference_wall_s=old_wall, optimized_wall_s=new_wall,
        speedup=old_wall/new_wall, reference_future_steps_per_s=K*max(4,width)*2*H/old_wall,
        optimized_future_steps_per_s=K*max(4,width)*2*H/new_wall,
        H=H,K=K,width=width,horizons=hs,state_id=record['state_id'],
        device=os.environ.get('SPEED2_DEVICE','cpu'),timing_includes_trajectory_capture=True,
        group_size=len(record['group']),plan_duration_s=max(a.get('duration_s',0) for a in acts),
        reference=before,optimized=after,new_sumo_sim_s=0)
    if memory is not None:
        result['memory']=dict(memory.receipt,encoder_calls=len(memory.encoder_seconds),
                              encoder_wall_s=memory.encoder_seconds)
    save(root/'PARITY.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('reference','optimized')}),flush=True)
    assert result['pass_'], 'speed2 parity failed'


def profile(root, compact=False, speed2=False, lines=False):
    print('BOOTSTRAP_CPU', flush=True)
    rt, record, acts = setup(root)
    print('ROLLOUT_CPU', flush=True)
    from .plan_wm import evaluate
    profiler = cProfile.Profile()
    timer = LineTimer()
    start = time.perf_counter()
    if lines:sys.settrace(timer.trace)
    try:
        with profiler:
            result = evaluate(rt, record, acts, 2, 60, chunk_actions=4, trim=compact, pad_actions=4,speed2=speed2,
                              memory=getattr(rt,'speed2_memory',None))
    finally:
        sys.settrace(None)
    wall = time.perf_counter() - start
    profiler.dump_stats(str(root/'BASELINE.prof'))
    stats = pstats.Stats(profiler)
    top = []
    for (file, line, name), (cc, nc, tt, ct, _) in sorted(stats.stats.items(), key=lambda x: -x[1][2])[:40]:
        top.append(dict(file=file, line=line, name=name, calls=nc, self_s=tt,
                        cumulative_s=ct, self_percent=100*tt/stats.total_tt))
    receipt=dict(wall_s=wall, future_steps_per_s=8*120/wall,
        H=60, K=2, width=4, top=top, lines=timer.result(), result=result, new_sumo_sim_s=0)
    if hasattr(rt,'speed2_memory'):
        memory=rt.speed2_memory
        receipt['memory']=dict(memory.receipt,encoder_calls=len(memory.encoder_seconds),encoder_wall_s=memory.encoder_seconds)
    save(root/'BASELINE.json',receipt)
    print(json.dumps(dict(wall_s=wall, top=top[:20])), flush=True)


def tests(root):
    import unittest
    from unittest.mock import patch
    setup(root)
    os.environ['SPEED2_EXPORT_BOUND'] = '/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json'
    suite = unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.test_speed2_cpu')
    with patch('subprocess.Popen', side_effect=AssertionError('physics forbidden')):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    save(root/'TESTS.json',dict(pass_=result.wasSuccessful(),tests=result.testsRun,
                              skips=len(result.skipped),new_sumo_sim_s=0))
    assert result.wasSuccessful() and not result.skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('profile','parity','tests','qualification','gpu-profile','gpu-experiment'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--compact', action='store_true')
    p.add_argument('--speed2', action='store_true')
    p.add_argument('--lines', action='store_true')
    p.add_argument('--H',type=int,default=60)
    p.add_argument('--K',type=int,default=2)
    p.add_argument('--width',type=int,default=4)
    p.add_argument('--fixture-index',type=int,choices=range(4),default=0)
    a = p.parse_args()
    os.nice(19)
    os.environ['SPEED2_FIXTURE_INDEX'] = str(a.fixture_index)
    import torch
    import platform
    save(a.root/'ENVIRONMENT.json',dict(torch=torch.__version__,python=sys.version,
        platform=platform.platform(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        requested_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),new_sumo_sim_s=0))
    try:
        if a.mode == 'profile':profile(a.root, a.compact, a.speed2, a.lines)
        elif a.mode == 'gpu-profile':
            from .speed2_gpu import profile as gpu_profile
            gpu_profile(a.root,a.H,a.K,a.width)
        elif a.mode == 'gpu-experiment':
            from .speed2_gpu import experiment
            experiment(a.root,a.H,a.K,a.width)
        elif a.mode == 'parity':parity(a.root,a.H,a.K,a.width)
        elif a.mode == 'qualification':
            from .speed2_runtime import qualify
            rt,record,actions=setup(a.root)
            result=qualify(rt,record,actions,a.K,a.H,a.width)
            save(a.root/'QUALIFICATION.json',result)
            print(json.dumps(result),flush=True)
            assert result['pass_']
        else:tests(a.root)
    except BaseException:
        import traceback
        save(a.root/'FAILED.json',dict(traceback=traceback.format_exc(),new_sumo_sim_s=0))
        raise


if __name__ == '__main__':
    main()
