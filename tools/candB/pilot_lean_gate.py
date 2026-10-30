"""Owner23:50 four-start gate adapter and durable startup-score rescue.

Installed only in an explicitly verified owned lean process. No rollout, sampler,
objective or parameter changes. The library's registered sixteen-start gate stays
unchanged on disk and this distinct receipt cannot authorize deployment.
"""


def panel_gate(policy_j, native_j, *, update, panel_sha256=None):
    # Self-contained globals: this code can specialize an already-bound function.
    import math
    p, n = list(map(float, policy_j)), list(map(float, native_j))
    if len(p) != 4 or len(n) != 4 or not all(map(math.isfinite, p+n)):
        raise ValueError('exploratory lean native gate needs four finite paired costs')
    delta = [a-b for a, b in zip(p, n)]
    return dict(schema='EXPLORATORY-LEAN_native_panel/1', label='EXPLORATORY-LEAN',
        deployment_authorized=False, update=int(update), panel_sha256=panel_sha256,
        policy_J=p, native_J=n, delta_per_start=delta,
        mean_delta=sum(delta)/4, passed=all(x <= 0 for x in delta),
        critic_tail_included=False)


def rescued_gate(policy_j, native_j, *, update, panel_sha256=None):
    import hashlib
    import json
    import sys
    from pathlib import Path
    from tools.candB import pilot_lean_gate as overlay
    from wms.control.nativeinit import PANEL_SHA256
    panel_gate = overlay.panel_gate
    panel_sha256 = PANEL_SHA256 if panel_sha256 is None else panel_sha256
    frame = sys._getframe(1)
    main = frame
    while main and not (main.f_code.co_name == 'main' and 'output' in main.f_locals):
        main = main.f_back
    if main is None or frame.f_code.co_name != 'score_native_panel':
        # The final selection also uses the gate, without rescoring startup.
        return panel_gate(policy_j, native_j, update=update, panel_sha256=panel_sha256)
    output = Path(main.f_locals['output'])
    if 'EXPLORATORY-LEAN' not in output.name:
        raise ValueError('startup rescue escaped owned lean output')
    panel = frame.f_locals['panel']
    if len(panel['starts']) != 4:
        raise ValueError('startup rescue requires exact lean panel prefix')
    result = panel_gate(policy_j, native_j, update=update, panel_sha256=panel_sha256)
    receipt = output / 'pilot_preserved_startup.json'
    if not receipt.exists():
        doc = dict(label='EXPLORATORY-LEAN', baseline_recomputed=False,
            independent=main.f_locals['before'], scores=frame.f_locals['scores'],
            panel=panel, native_gate=result,
            overlay_sha256=hashlib.sha256(Path(overlay.__file__).read_bytes()).hexdigest())
        with receipt.open('x') as stream:
            json.dump(doc, stream, indent=1, allow_nan=False)
        frame.f_locals['trainer'].save_checkpoint(output / 'pilot_startup_preserved.pt')
    return result


def install():
    """Retain existing native-scoring call's bound function, replacing code only."""
    import hashlib
    import inspect
    import json
    import os
    import sys
    import time
    from pathlib import Path
    from wms.control import nativeinit
    frames = list(sys._current_frames().values())
    active = []
    for frame in frames:
        while frame:
            active.append(frame)
            frame = frame.f_back
    target = next((f for f in active if f.f_code.co_name == 'score_native_panel'), None)
    if target is None or len(target.f_locals['panel']['starts']) != 4:
        raise ValueError('owned trainer is not in four-start native startup scoring')
    trainer = target.f_locals['trainer']
    if trainer.update_count != 191 or trainer.policy.residual_reference != 'ctrl0':
        raise ValueError('unexpected startup phase or execution law')
    if os.environ.get('WMS_SPEED_RESCORE_EVERY') != '5':
        raise ValueError('not the registered lean execution environment')
    function = target.f_locals['panel_gate']
    if function is not nativeinit.panel_gate or '16 finite paired costs' not in inspect.getsource(function):
        raise ValueError('unexpected native gate source/binding')
    main = next(f for f in active if f.f_code.co_name == 'main' and 'output' in f.f_locals)
    output = Path(main.f_locals['output'])
    receipt = output / 'pilot_gate_overlay_INSTALLED.json'
    if receipt.exists():
        raise ValueError('gate overlay already installed; never repeat live dispatch')
    # Function has no freevars. Builtins/imports only, even with original globals.
    function.__code__ = rescued_gate.__code__
    function.__kwdefaults__ = rescued_gate.__kwdefaults__
    with receipt.open('x') as stream:
        json.dump(dict(label='EXPLORATORY-LEAN', pid=os.getpid(), epoch=time.time(),
            overlay_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reason='preserve in-flight first4 native scores; explicit owner23:50 panel',
            source_files_modified=False, baseline_recomputed=False), stream, indent=1)
