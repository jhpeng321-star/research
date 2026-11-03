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


def pair_v2v_block(frame, prev_frame, members, cfg=None):
    """The v2v block for one pair second: {member vid: features} for every member present as a
    CAV in ``frame`` (zeros for absent members); partners = the other CAVs of the frame.
    Returns (block, sorted vids of partners whose previous-second command is missing)."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    last = _last_commands(prev_frame) if prev_frame is not None else {}
    known = {k: c for k, c in last.items() if c is not None}
    vehicles = frame.get('vehicles', [])
    missing = sorted({v['vid'] for v in vehicles if v.get('is_cav') and last.get(v['vid']) is None})
    block = {}
    for ego in members:
        me = next((v for v in vehicles if v.get('vid') == ego and v.get('is_cav')), None)
        if me is None:
            block[ego] = [0.]*V2V_DIM
            continue
        block[ego] = list(v2v_features(frame, ego, known, cfg))
    return block, missing


def attach_pairs(pairs, trace_path, t0, cfg=None):
    """(kept pairs, counts): add ``sample['context']['v2v']`` to every pair of one admitted
    bank entry, from the winning plan's recorded branch trace; pair i is second t0+i.
    Pairs whose trace lacks the partners' commands (no previous second, no trace frame, or a
    partner command not recorded) are EXCLUDED and counted - never imputed."""
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
        block, missing = pair_v2v_block(frame, prev, members, cfg)
        if missing:
            counts['excluded_missing_partner_command'] += 1
            continue
        context['v2v'] = block
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
    base = sv.verify_controller_class()

    class _V2VPolicy:
        """Injects context['v2v'] just before the wrapped policy samples; everything else is
        delegated (the current model ignores the extra context key, so pre-model runs and
        sealed digests are unaffected)."""

        def __init__(self, outer, real):
            self.outer, self.real = outer, real

        def sample(self, context, *a, **k):
            context['v2v'] = self.outer.v2v_block(context)
            return self.real.sample(context, *a, **k)

        def __call__(self, *a, **k):
            return self.real(*a, **k)

        def __getattr__(self, name):
            return getattr(self.real, name)

    class V2VController(base):
        """last_commands per step = the commands in effect at the previous second (the apply()
        ``commands`` argument, i.e. what was issued at t-1); partner accelerations = the
        per-second finite difference of the recorded speeds (the offline trace records the
        issued command in the same slot)."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.v2v_cfg = dict(DEFAULT_CFG)
            self.v2v_last, self._v2v_prev_v, self._v2v_state = {}, None, None

        def apply(self, rows, commands, now):
            self._v2v_state = (list(rows), dict(commands), int(now))
            saved = self.policy
            self.policy = _V2VPolicy(self, saved)
            try:
                return super().apply(rows, commands, now)
            finally:
                self.policy = saved
                self.v2v_last = {r['vid']: float(commands[r['vid']]) for r in rows
                                 if r.get('is_cav') and r['vid'] in commands}
                self._v2v_prev_v = {r['vid']: float(r['v_mps']) for r in rows}

        def v2v_block(self, context):
            rows, commands, now = self._v2v_state
            vehicles = []
            for r in rows:
                v = dict(r)
                if v.get('is_cav'):
                    prev = (self._v2v_prev_v or {}).get(v['vid'])
                    a = float(v['v_mps'])-prev if prev is not None else None
                    v['issued_acceleration_mps2'] = a
                    v['command_present'] = a is not None
                vehicles.append(v)
            known = {k: c for k, c in self.v2v_last.items() if c is not None}
            frame = dict(time_s=now, vehicles=vehicles)
            block = {}
            for m in context['members']:
                if any(v.get('vid') == m and v.get('is_cav') for v in vehicles):
                    block[m] = list(v2v_features(frame, m, known, self.v2v_cfg))
                else:
                    block[m] = [0.]*V2V_DIM
            return block

    return V2VController
