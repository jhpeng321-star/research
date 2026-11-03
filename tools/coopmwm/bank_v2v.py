"""V2V intent sharing on the B-REPLAY bank (MAIN 2026-11-03, part 3/3 BANK/IMITATION).

Flag ``v2v`` (binding key ``v2v``, DEFAULT OFF: every code path in this module is skipped
and all outputs stay byte-identical). When ON:

- BANK PAIRS (replay_bank.bank_pairs): every imitation pair of an ADMITTED bank entry also
  carries the partners' V2V features of that second, computed from the winning plan's
  recorded branch trace (observable.jsonl.gz of the chosen branch): partner CAV positions,
  speeds and accelerations, plus their LAST ISSUED command at the PREVIOUS second of the
  same trace. Pairs whose trace lacks the partners' commands are EXCLUDED and counted
  (never imputed).
- RUNTIME (lesson check + policy-synergy NEW-policy branches): the branch runner computes
  last_commands per step (the commands in effect at the previous second) and adds the v2v
  block to the policy observation (context['v2v'], {member vid: features}) before the policy
  runs. Wired through bank_v2v.branch_controller_class() (sumo_verify.LocalBranchRunner);
  the env flag propagates to branch subprocesses via bank_v2v.enabled().

The feature function is the OBSERVATION part's tools/coopmwm/v2v_obs.py
(v2v_features(state, cav_id, last_commands, cfg), V2V_DIM). Until that module lands, a
local dependency-free STUB with the same signature is used (STUB=True is disclosed in the
bank-pairs summary). The stub block: for each of the max_partners nearest partner CAVs
(by |s_m - ego s_m|, vid order tie-break) 4 floats - relative position (s_m - ego s_m),
speed (v_mps), acceleration at that second (issued_acceleration_mps2, 0 when absent) and
the partner's last issued command at the previous second (0 when absent) - then zero padded.
"""
import os

ENV = 'C4NM_V2V'             # branch-runner flag; set inside bank_v2v.enabled() only
DEFAULT_CFG = dict(max_partners=4)

try:
    from .v2v_obs import v2v_features, V2V_DIM      # OBSERVATION part (other workstream)
    STUB = False
except Exception:                                    # not present yet: local stub, same signature
    STUB = True
    MAX_PARTNERS = DEFAULT_CFG['max_partners']
    V2V_DIM = 4*MAX_PARTNERS

    def v2v_features(state, cav_id, last_commands, cfg=None):
        cfg = dict(DEFAULT_CFG, **(cfg or {}))
        vehicles = state.get('vehicles', []) if isinstance(state, dict) else list(state)
        me = next((v for v in vehicles if v.get('vid') == cav_id), None)
        if me is None:
            return [0.]*V2V_DIM
        partners = sorted((v for v in vehicles if v.get('is_cav') and v.get('vid') != cav_id),
                          key=lambda v: (abs(float(v['s_m'])-float(me['s_m'])), str(v['vid'])))
        out = []
        for v in partners[:int(cfg['max_partners'])]:
            accel = v.get('issued_acceleration_mps2')
            cmd = last_commands.get(v['vid'])
            out += [float(v['s_m'])-float(me['s_m']), float(v['v_mps']),
                    float(accel) if accel is not None else 0.,
                    float(cmd) if cmd is not None else 0.]
        return out+[0.]*(V2V_DIM-len(out))


def enabled(flag):
    """Context manager: set the branch-runner env flag for the enclosed region only
    (branch subprocesses inherit it); restores the previous value afterwards."""
    import contextlib
    @contextlib.contextmanager
    def go():
        old = os.environ.get(ENV)
        if flag:
            os.environ[ENV] = '1'
        try:
            yield
        finally:
            if flag:
                if old is None:
                    os.environ.pop(ENV, None)
                else:
                    os.environ[ENV] = old
    return go()


# ---------------------------------------------------------------- bank pairs (offline, from the trace)

def trace_frames(trace_path):
    """{int second: frame} of a branch's observable.jsonl.gz (same reader family as
    replay_bank.ttc_events: lines {time_s, vehicles[{vid, s_m, v_mps, lane, is_cav,
    length_m, issued_acceleration_mps2, command_present}], actions})."""
    import gzip
    import json
    frames = {}
    with gzip.open(trace_path, 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            frames[int(x['time_s'])] = x
    return frames


def _last_commands(frame):
    """vid -> last issued command of every CAV in a frame (None where it was not recorded)."""
    out = {}
    for v in frame.get('vehicles', []):
        if not v.get('is_cav'):
            continue
        cmd = v.get('issued_acceleration_mps2')
        out[v['vid']] = float(cmd) if cmd is not None else None
    return out


def _frame_table(frame):
    """``{vid: row}`` view of a trace frame for the OBSERVATION part's v2v_features (which takes
    a vid -> record mapping, not the raw trace frame); the partner acceleration feature a_mps2
    maps the row's recorded issued_acceleration_mps2 (0.0 when absent - only the PREVIOUS-second
    command gates exclusion, exactly as before)."""
    vehicles = frame.get('vehicles', []) if isinstance(frame, dict) else list(frame)
    table = {}
    for v in vehicles:
        row = dict(v)
        accel = row.get('issued_acceleration_mps2')
        row['a_mps2'] = float(accel) if accel is not None else 0.
        table[v['vid']] = row
    return table


def _obs_cfg(cfg=None):
    """The real v2v_features takes a V2VConfig; the bank cfg carries max_partners/range_m."""
    from .v2v_obs import V2V_RANGE_M, V2VConfig
    if isinstance(cfg, V2VConfig):
        return cfg
    merged = dict(DEFAULT_CFG, **(cfg or {}))
    return V2VConfig(v2v=True, k=int(merged['max_partners']),
                     range_m=float(merged.get('range_m', V2V_RANGE_M)))


def _features(frame, ego, known, cfg):
    """Dispatch on the feature source: the local STUB takes the raw frame + dict cfg; the
    OBSERVATION part's v2v_features takes the {vid: row} table + a V2VConfig. Both return
    plain Python floats (bank pairs are JSON-serialised into receipts)."""
    if STUB:
        return [float(x) for x in v2v_features(frame, ego, known, cfg)]
    return [float(x) for x in v2v_features(_frame_table(frame), ego, known, _obs_cfg(cfg))]


def _intent_features(frame, ego, known, cfg, intent):
    """Planned-command intent variant of :func:`_features` (v2v_intent ON only): the base block
    interleaved with the partner intents published at ``time_s-1`` (``intent['board']``)."""
    if STUB:
        raise ValueError('v2v intent requires the OBSERVATION part (tools/coopmwm/v2v_obs.py)')
    from .v2v_obs import v2v_intent_features
    return [float(x) for x in v2v_intent_features(_frame_table(frame), ego, known, intent['board'],
            int(frame.get('time_s', 0)), int(intent['horizon']), _obs_cfg(cfg))]


def _block_dim(cfg, intent):
    if not intent:
        return V2V_DIM
    from .v2v_obs import v2v_intent_dim
    return v2v_intent_dim(_obs_cfg(cfg).k, int(intent['horizon']))


def pair_v2v_block(frame, prev_frame, members, cfg=None, intent=None):
    """The v2v block for one pair second: {member vid: features} for every member present as a
    CAV in ``frame`` (zeros for absent members); partners = the other CAVs of the frame.
    Returns (block, sorted vids of partners whose previous-second command is missing).
    ``intent`` (v2v_intent ON only; default None = unchanged): dict(horizon, board) with
    board = {vid: (published_time_s, H planned commands)} visible at this second."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    last = _last_commands(prev_frame) if prev_frame is not None else {}
    known = {k: c for k, c in last.items() if c is not None}
    frame = causal_frame(frame, prev_frame)
    vehicles = frame.get('vehicles', [])
    missing = sorted({v['vid'] for v in vehicles if v.get('is_cav') and last.get(v['vid']) is None})
    block = {}
    for ego in members:
        me = next((v for v in vehicles if v.get('vid') == ego and v.get('is_cav')), None)
        if me is None:
            block[ego] = [0.]*_block_dim(cfg, intent)
            continue
        block[ego] = list(_intent_features(frame, ego, known, cfg, intent) if intent
                          else _features(frame, ego, known, cfg))
    return block, missing


def causal_frame(frame, previous):
    """Sensed acceleration is a past speed difference, never this step's target."""
    prior = {v['vid']:v for v in (previous or {}).get('vehicles', [])}
    dt = float(frame.get('time_s', 0))-float((previous or {}).get('time_s', 0))
    vehicles = []
    for v in frame.get('vehicles', []):
        old = prior.get(v['vid'])
        accel = ((float(v['v_mps'])-float(old['v_mps']))/dt if old and dt > 0 else None)
        vehicles.append(dict(v, issued_acceleration_mps2=accel))
    return dict(frame, vehicles=vehicles)


def attach_pairs(pairs, trace_path, t0, cfg=None, intent=None):
    """(kept pairs, counts): add ``sample['context']['v2v']`` to every pair of one admitted
    bank entry, from the winning plan's recorded branch trace; pair i is second t0+i.
    Pairs whose trace lacks the partners' commands (no previous second, no trace frame, or a
    partner command not recorded) are EXCLUDED and counted - never imputed.
    ``intent`` (v2v_intent ON only): dict(horizon, entry).  Partner-intent inputs are built EMPTY
    (zeros, intent_valid 0) and each kept pair gets ``sample['intent_link']`` = dict(entry, t,
    slots={member: partner vids in slot order}); v2v_intent.refresh_bank_intents fills them each
    update from the CURRENT intent head on the partners' own t-1 bank contexts (never from
    recorded commands)."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    counts = dict(pairs=len(pairs), kept=0, excluded_missing_prev_second=0,
                  excluded_missing_partner_command=0, trace_missing=False)
    try:
        frames = trace_frames(trace_path)
    except (FileNotFoundError, OSError):
        counts['trace_missing'] = True
        return [], counts
    kept = []
    for i, (sample, w) in enumerate(pairs):
        t = int(t0)+i
        frame, prev = frames.get(t), frames.get(t-1)
        if frame is None or prev is None:
            counts['excluded_missing_prev_second'] += 1
            continue
        context = sample.setdefault('context', {})
        members = list(context.get('members') or sample['behaviour']['proposal_members'])
        block, missing = pair_v2v_block(frame, prev, members, cfg, None if intent is None else
                                        dict(horizon=intent['horizon'], board={}))
        if missing:
            counts['excluded_missing_partner_command'] += 1
            continue
        context['v2v'] = block
        if intent is not None:
            from .v2v_obs import v2v_partner_ids
            seen = causal_frame(frame, prev)  # the exact frame pair_v2v_block ranked partners on
            table, ocfg = _frame_table(seen), _obs_cfg(cfg)
            cavs = {v.get('vid') for v in seen.get('vehicles', []) if v.get('is_cav')}
            sample['intent_link'] = dict(entry=str(intent['entry']), t=t, slots={
                ego: list(v2v_partner_ids(table, ego, ocfg)) for ego in members if ego in cavs})
        kept.append((sample, w))
    counts['kept'] = len(kept)
    return kept, counts


def merge_counts(total, counts):
    for k in ('pairs', 'kept', 'excluded_missing_prev_second', 'excluded_missing_partner_command'):
        total[k] = total.get(k, 0)+counts[k]
    total['trace_missing'] = total.get('trace_missing', False) or counts['trace_missing']
    return total


# ---------------------------------------------------------------- runtime (branch runner)

def branch_controller_class():
    """A VerifyController subclass that computes last_commands per step and adds the v2v block
    to the policy observation - ONLY when the env flag is set. Default (unset): returns None
    and the branch runner uses the plain controller class (byte-identical)."""
    if os.environ.get(ENV) != '1':
        return None
    from . import sumo_verify as sv
    from .v2v_runtime import controller_class
    return controller_class(sv.verify_controller_class())

