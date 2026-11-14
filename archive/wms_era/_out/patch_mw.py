p = 'E:/rd-wt/ws-r3-evalstates3/output/wd01/wm_n1/lpwo/materialize_wall.py'
s = open(p, 'rb').read().decode().replace('\r\n', '\n')


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a, s.count(a))
    s = s.replace(a, b)


rep('''def materialize_state(window, packs, identities, guard, *, timeline=None,
                      source="wall_same", world_seen=True, min_vehicles=5,
                      history_s=None):
    checked_record(window, guard)''', '''def _real_state(window, packs, identities, guard, real_steps, history_s, min_vehicles):
    """Build one state at the fixed real t0 (see ``real_t0``); returns it with ``_physical``."""
    from . import real_t0
    if str(window.get("arm")) in real_t0.BASE_ARMS:
        raise WallMaterializeError(
            "base arm featurised from sim origin 0 (t0 label misaligned): excluded by rule")
    pack, identity = packs[str(window["pack"])], identities[str(window["pack"])]
    run = _run(window, pack, guard)
    position, item = pack.item_for(run)
    checked_record({**run, **item}, guard)
    shard = real_t0.open_shard(run, guard, checked_path, read_json)
    return real_t0.build_state_real(
        window, pack, identity, run, position, item, shard, steps=real_steps,
        history_s=history_s, min_vehicles=min_vehicles)


def materialize_state(window, packs, identities, guard, *, timeline=None,
                      source="wall_same", world_seen=True, min_vehicles=5,
                      history_s=None, real_steps=None):
    checked_record(window, guard)''')
rep('''    physical = _geometry(window, pack, identity, guard)
    descriptor = {**window, "_extra_baselines": []}
    state = build_state(descriptor, packs, identities, horizon_s=HORIZON_S,
                        extra_baselines=(), shift_steps=SHIFT_STEPS,
                        min_vehicles=min_vehicles, history_s=history_s)
    _validate_state(state, history_s)
''', '''    if real_steps is None:
        physical = _geometry(window, pack, identity, guard)
        descriptor = {**window, "_extra_baselines": []}
        state = build_state(descriptor, packs, identities, horizon_s=HORIZON_S,
                            extra_baselines=(), shift_steps=SHIFT_STEPS,
                            min_vehicles=min_vehicles, history_s=history_s)
    else:
        state = _real_state(window, packs, identities, guard, real_steps, history_s,
                            min_vehicles)
        physical = state.pop("_physical")
    _validate_state(state, history_s)
''')
rep('''        "assembly" if timeline.get("assembled_at_s") is None
        or t < float(timeline["assembled_at_s"]) else "hold")
''', '''        "assembly" if timeline.get("assembled_at_s") is None
        or t < float(timeline["assembled_at_s"]) else "hold")
    if (real_steps is not None and timeline["owned"]
            and state["wall_phase_at_t0"] not in ("assembly", "hold")):
        raise WallMaterializeError(
            "wall not owned at the real t0 (" + state["wall_phase_at_t0"] + ")")
''')
rep('''        try:
            native_physical = _geometry(native_window, pack, identity, guard)
            if native_physical != physical:
                state["native_pair_status"] = "initial_physical_state_mismatch"
            else:
                native = build_state(native_window, packs, identities,
                                     horizon_s=HORIZON_S, extra_baselines=(),
                                     shift_steps=SHIFT_STEPS, min_vehicles=min_vehicles,
                                     history_s=history_s)
                _validate_state(native, history_s)''', '''        try:
            if real_steps is None:
                native_physical = _geometry(native_window, pack, identity, guard)
            else:
                native = _real_state(native_window, packs, identities, guard, real_steps,
                                     history_s, min_vehicles)
                native_physical = native.pop("_physical")
            if native_physical != physical:
                state["native_pair_status"] = "initial_physical_state_mismatch"
            else:
                if real_steps is None:
                    native = build_state(native_window, packs, identities,
                                         horizon_s=HORIZON_S, extra_baselines=(),
                                         shift_steps=SHIFT_STEPS, min_vehicles=min_vehicles,
                                         history_s=history_s)
                _validate_state(native, history_s)''')
rep('''                 history_s=None, out_roots=()):
    counts = Counter()''', '''                 history_s=None, out_roots=(), real_steps=None):
    counts = Counter()''')
rep('''                    min_vehicles=min_vehicles, history_s=history_s)
            except MaterializeError as exc:''', '''                    min_vehicles=min_vehicles, history_s=history_s,
                    real_steps=real_steps)
            except MaterializeError as exc:''')
rep('''            if accepted_targets is not None:
                accepted_targets.add''', '''            if real_steps is not None:
                counts["n_roster_vehicles"] += int(record["history"]["n_roster"])
                counts["n_roster_full_history"] += int(record["history"]["n_roster_full_history"])
                counts["n_states_all_roster_full"] += int(
                    record["history"]["n_roster"] == record["history"]["n_roster_full_history"])
                if name != "seen_controls":
                    counts["phase:" + record["wall_phase_at_t0"]] += 1
            if accepted_targets is not None:
                accepted_targets.add''')
rep('''        for k in ("n_states_full_history", "n_states_lt_history_s"):''',
    '''        for k in ("n_states_full_history", "n_states_lt_history_s", "n_roster_vehicles",
                  "n_roster_full_history", "n_states_all_roster_full"):''')
rep('''    out_roots = [source_path(p) for p in (getattr(args, "out_root", None) or [])]
    if guard is None:''', '''    real_steps = None
    if getattr(args, "t0_after_origin_s", None) is not None:
        real_steps = int(round(float(args.t0_after_origin_s) / DT_S))
        if history_s is None or real_steps < int(round(history_s / DT_S)):
            raise WallMaterializeError(
                "--t0-after-origin-s needs --history-s no longer than the offset")
    out_roots = [source_path(p) for p in (getattr(args, "out_root", None) or [])]
    if guard is None:''')
rep('''               "history_steps": HISTORY_STEPS, "history_s": history_s,''',
    '''               "history_steps": HISTORY_STEPS, "history_s": history_s,
               "t0_mode": ("v2_shift9" if real_steps is None else "real_after_origin"),
               "t0_after_origin_steps": real_steps,
               "t0_rule": (None if real_steps is None else __import__(
                   "lpwo.real_t0", fromlist=["RULE"]).RULE),''')
rep('''                    history_s=history_s, out_roots=out_roots)
        receipt["status"]''', '''                    history_s=history_s, out_roots=out_roots, real_steps=real_steps)
        receipt["status"]''')
rep('''    p.add_argument("--out-dir", required=True,''', '''    p.add_argument("--t0-after-origin-s", type=float, default=None,
                   help="R3 wall_h60: decision tick = manifest item tick + this many seconds "
                        "(fixed, outcome-blind) instead of the v2 +4.5 s shift; needs "
                        "per-run identity shards under a --source-root")
    p.add_argument("--out-dir", required=True,''')
open(p, 'wb').write(s.replace('\n', '\r\n').encode())
print('ok')
