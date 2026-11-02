"""MAIN00:48 it2_K2 boundary wrapper; inherited numerical sources stay immutable."""
import copy

CONFIG = dict(iterations=2, screen_K=2)
NAME = 'it2_K2'


def configured_search(search):
    def run(*args, **kwargs):
        kwargs['config'] = dict(kwargs.get('config') or {}, **CONFIG)
        result = search(*args, **kwargs)
        result['search_config'] = dict(name=NAME, overrides=CONFIG.copy(),
            full_population=True, screen_H=300, final_K=8,
            final_H='min(600,max(300,duration+120))', mirror_active=False)
        return result
    return run


def owner_key(owner):
    return tuple(owner[k] for k in ('boot_id', 'pid', 'start_ticks'))


def drain_applies(value, owner):
    return value is not None and owner_key(value['owner']) == owner_key(owner)


def pending_claims(root):
    """Do not retire a worker with any uncompleted submitted request."""
    return sorted(p.name for p in (root/'inbox').glob('*.json')
        if not (root/'forecasts'/p.name).exists())


def run(root, old, deadline, amendment, overlay):
    import json
    import sys
    import time
    from pathlib import Path
    from unittest.mock import patch
    root = Path(root)
    sys.path[:0] = [str(Path(overlay)), str(root/'pytools')]
    import ls_gpu
    import tools.coopmwm
    tools.coopmwm.__path__.insert(0, str(Path(overlay)))
    from tools.coopmwm import warm_service
    capture = ls_gpu.capture_search
    ls_gpu.capture_search = lambda search, **kw: configured_search(capture(search, **kw))
    original_main = warm_service.main

    def qualified_main(r):
        from tools.wmteach import runtime
        from tools.coopmwm.plan_service import configured_runtime, batch_width
        from tools.coopmwm.plan_wm import evaluate
        from tools.coopmwm.speed2_memory import native_fixture
        from tools.coopmwm.speed2_runtime import qualify
        from tools.coopmwm.mppi_search import group_components, comparators, sample_unscored
        from tools.coopmwm.wm import cost_adapter
        binding = warm_service.read(r/'GPU_BINDING.json')
        rt = configured_runtime(runtime.bootstrap, binding.get('memory_fraction', .65))
        costs = rt.frozen.costs
        rt.frozen.costs = cost_adapter(costs, warm_service.read(r/'FROZEN_COSTS.json')['code'],
            horizons=(60,120,300,360,420,600))
        try:
            record, memory = native_fixture(rt)
            group = max(group_components(record['rows']), key=lambda g:g['size'])
            bounds = (float(rt.mp.policy.accel_min), float(rt.mp.policy.accel_max))
            acts = comparators(record, 600) + [sample_unscored(group,
                [[0.]*300]*group['size'], bounds, 'ordinary', 20261101, duration_s=300)]
            width = batch_width(binding, 2)
            begin = time.monotonic()
            full_fast = qualify(rt, record, acts, 2, 300, width, memory=memory, transfers=True)
            single = evaluate(rt, record, acts, 2, 300, chunk_actions=1,
                pad_actions=width, trim=False, speed2=True, transfers=True, memory=memory)
            batch = evaluate(rt, record, acts, 2, 300, chunk_actions=width,
                pad_actions=width, trim=False, speed2=True, transfers=True, memory=memory)
            error = max(abs(a['team_delay']-b['team_delay'])
                for key in single for a,b in zip(single[key],batch[key]))
            draws = all(a['draw_id']==b['draw_id']
                for key in single for a,b in zip(single[key],batch[key]))
            gate = dict(pass_=full_fast['pass_'] and full_fast['max_cost_error']==0
                and error==0 and draws, max_cost_error=error,
                full_fast=full_fast, identical_draw_ids=draws, K=2, H=300,
                width=width, search_config=NAME, new_sumo_sim_s=0,
                wall_s=time.monotonic()-begin, amendment_sha256=amendment)
            warm_service.write(r/'STARTUP_VARIANT_IT2K2.json', gate)
            if not gate['pass_']:
                raise ValueError('it2_K2 exact full/fast and warm batch/single gate failed')
        finally:
            rt.frozen.costs = costs
        # Fresh inherited eight gates, including committed K4/H300 and K8/H420,
        # still run. Bootstrap reuses the same loaded model without extra memory.
        with patch.object(runtime, 'bootstrap', return_value=rt):
            return original_main(r)

    warm_service.main = qualified_main
    ls_gpu.run(str(root), copy.deepcopy(old), deadline, amendment,
        generation='_FASTK2', overlay=overlay)
