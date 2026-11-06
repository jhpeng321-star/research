"""N2 RSU-initiated lane-drop group moves (MAIN 2026-11-06 evening), DEFAULT OFF.

Why: walls failed before because no single CAV could see all members. An RSU at the
lane-drop bottleneck can coordinate within its radio range (the project's original
CAV-observable RSU-escalation design).

Inputs (observable only; AGENTS.md RSU constraint): the RSU field is built from the CAVs'
OWN timestamped uplink packets -- each CAV's self state plus its locally detected neighbours
(``SensorReport``) published at t-1, from CAVs within ``range_m`` of the RSU position --
with receipt age, coverage fractions and missing bits. No ground truth, true penetration,
driver type, full-state rollout or global traffic metric is read.

Suggestion: at the CAV-trigger decision cadence the RSU may choose ONE suggestion from a
registered slate ``none`` + up to RSU_CANDIDATES "form a pacing row (abreast_wall) /
closing pair at position x" entries. Entries are built from the reporting CAVs' self states
(role lanes, registered windows/span, same parameter draws as the CAV moves); a chosen
suggestion opens an RSU session on the SAME lane-drop ledger (lane_drop.Protocol) and is
broadcast at t over the V2V/RSU channel, visible at t+1: CAVs within the RSU range and
within the join radius of the (propagated) anchor, in an uncovered role lane, get a
``lane_drop_join`` option in their trigger slate. Execution is the existing single-writer
path (MoveExecutor commitment + committed-offset column; lane-drop speed law). HDVs never
receive anything; ownership and the session cap are the protocol's.

Learnable: ``RSUHead`` scores each slate entry from [scaled RSU field | entry descriptor]
with a small MLP whose OUTPUT layer is zero-initialised (an untrained head is uniform over
the available entries, none included). Rollout decisions are recorded (``RSUBatch``) for the
existing per-decision PPO (``trigger_policy.ppo_update`` interface) and for CE guidance.
B and R are identical: the same field, slate, head and loss. Guidance labels differ only by
their source -- B: a verified closing-pair/wall example mapped to the slate entry of the same
kind nearest its group centre (``teacher_target``); R: the random teacher, uniform over the
same available slate (``random_teacher_target``).

Config (inside lane_drop; requires lane_drop.v2v): rsu = True | dict(range_m=100..1000,
position_m=float (default: the lane-drop point)).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from . import lane_drop as ld
from . import moves
from .merge_features import (CONTROL_DT_S, N_LANES, QUEUE_SPEED_MPS, SENSOR_RANGE_M,
                             _local_rows, _valid_report)

RSU_RULE = 'n2-rsu-suggest-v1'
RSU_RANGE_WINDOW_M = (100., 1000.)
DEFAULT_RSU_RANGE_M = 1000.    # DSRC-class upper range: covers the abreast window from 400 m
RSU_CANDIDATES = 4             # registered slate size (= the clean3 trigger_candidates)
RSU_HIDDEN = 32
JOIN_RADIUS_M = ld.ABREAST_SPAN_M   # joiners within the registered span of the anchor

LANE_FIELD = ('lane_present', 'count_veh', 'queue_veh', 'mean_speed_mps', 'coverage_frac',
              'speed_missing', 'cav_reports')
GLOBAL_FIELD = ('report_count', 'report_age_s', 'reports_missing', 'closing_queue_tail_m',
                'closing_queue_veh', 'open_sessions', 'rsu_session_active',
                'time_since_suggestion_s', 'suggestion_missing')
RSU_FIELD = tuple(f'lane_{lane}_{name}' for lane in range(N_LANES) for name in LANE_FIELD)+GLOBAL_FIELD
RSU_FIELD_DIM = len(RSU_FIELD)
_FIELD_SCALE = dict(count_veh=10., queue_veh=10., mean_speed_mps=30., cav_reports=10., report_count=10.,
                    closing_queue_tail_m=1000., closing_queue_veh=10., open_sessions=4.,
                    time_since_suggestion_s=600.)
FIELD_SCALE = np.array([_FIELD_SCALE.get(name.split('_', 2)[-1] if name.startswith('lane_') else name, 1.)
                        for name in RSU_FIELD], dtype=np.float32)
SUGGESTION_DESCRIPTOR = ('kind_closing_pair', 'kind_abreast_wall', 'anchor_dist_to_drop_frac',
                         'target_frac', 'gap_dv_frac', 'zip', 'onset_frac', 'hold_frac', 'depth_frac',
                         'group_size_over_lanes')
DESCRIPTOR_DIM = len(SUGGESTION_DESCRIPTOR)
_GF = {name: i for i, name in enumerate(RSU_FIELD)}


def rsu_settings(raw, view):
    if raw is True:
        raw = {}
    if not isinstance(raw, dict) or set(raw)-{'range_m', 'position_m'}:
        raise ValueError('lane_drop rsu must be True or dict(range_m, position_m)')
    range_m = raw.get('range_m', DEFAULT_RSU_RANGE_M)
    if not ld._finite(range_m) or not RSU_RANGE_WINDOW_M[0] <= float(range_m) <= RSU_RANGE_WINDOW_M[1]:
        raise ValueError(f'lane_drop rsu range_m must be in {RSU_RANGE_WINDOW_M} m')
    position = raw.get('position_m', view.lane_drop_s_m)
    if not ld._finite(position):
        raise ValueError('lane_drop rsu position_m must be finite')
    return dict(rule=RSU_RULE, range_m=float(range_m), position_m=float(position),
                candidates=RSU_CANDIDATES, hidden=RSU_HIDDEN)


# ---------------------------------------------------------------- field (observable only)

def missing_field():
    out = np.zeros(RSU_FIELD_DIM, dtype=np.float32)
    for lane in range(N_LANES):
        out[_GF[f'lane_{lane}_speed_missing']] = 1.
    out[_GF['reports_missing']] = 1.
    out[_GF['suggestion_missing']] = 1.
    return out


def _union(intervals):
    return ld._union_length(intervals)


def rsu_field(reports, now_s, view, rsu, *, open_sessions=0, rsu_active=False, last_suggestion_s=None):
    """[RSU_FIELD_DIM] physical field from the CAVs' own t-1 uplink packets in RSU range."""
    out = missing_field()
    for lane in range(N_LANES):
        out[_GF[f'lane_{lane}_lane_present']] = float(lane in view.lane_ids)
    out[_GF['open_sessions']] = open_sessions
    out[_GF['rsu_session_active']] = float(rsu_active)
    if last_suggestion_s is not None and last_suggestion_s <= now_s:
        out[_GF['time_since_suggestion_s']] = now_s-last_suggestion_s
        out[_GF['suggestion_missing']] = 0.
    accepted = uplink(reports, now_s, rsu)
    if not accepted:
        return out
    out[_GF['report_count']] = len(accepted)
    out[_GF['report_age_s']] = CONTROL_DT_S
    out[_GF['reports_missing']] = 0.
    lo, hi = rsu['position_m']-rsu['range_m'], min(rsu['position_m']+rsu['range_m'], view.lane_drop_s_m)
    visible, spans = {}, {lane: [] for lane in range(N_LANES)}
    for vid, report in accepted:
        for row in _local_rows(report):
            visible.setdefault(row.vehicle_id, row)
        a, b = max(lo, report.source.s_m-SENSOR_RANGE_M), min(hi, report.source.s_m+SENSOR_RANGE_M)
        if b > a:
            for lane in set(report.observed_lanes):
                spans[lane].append((a, b))
    rows = [visible[v] for v in sorted(visible)]
    sources = {vid: report.source for vid, report in accepted}
    for lane in range(N_LANES):
        if not spans[lane] or lane not in view.lane_ids:
            continue
        inside = [r for r in rows if r.lane == lane and any(a <= r.s_m <= b for a, b in spans[lane])]
        out[_GF[f'lane_{lane}_count_veh']] = len(inside)
        out[_GF[f'lane_{lane}_queue_veh']] = sum(r.v_mps < QUEUE_SPEED_MPS for r in inside)
        out[_GF[f'lane_{lane}_coverage_frac']] = _union(spans[lane])/max(hi-lo, 1e-9)
        out[_GF[f'lane_{lane}_cav_reports']] = sum(s.lane == lane for s in sources.values())
        if inside:
            out[_GF[f'lane_{lane}_mean_speed_mps']] = float(np.mean([r.v_mps for r in inside]))
            out[_GF[f'lane_{lane}_speed_missing']] = 0.
    queued = [r for r in rows if r.lane == view.dropped_lane and r.v_mps < QUEUE_SPEED_MPS
              and 0 <= view.distance_to_drop(r.s_m) <= ld.CLOSING_ROI_M
              and any(a <= r.s_m <= b for a, b in spans[view.dropped_lane])]
    out[_GF['closing_queue_veh']] = len(queued)
    out[_GF['closing_queue_tail_m']] = max((view.distance_to_drop(r.s_m) for r in queued), default=0.)
    return out


def uplink(reports, now_s, rsu):
    """[(cav, report)] valid own t-1 packets whose publisher is within the RSU range."""
    out = []
    for vid, report in sorted((reports or {}).items()):
        if _valid_report(report, vid, float(now_s)-CONTROL_DT_S) \
                and abs(report.source.s_m-rsu['position_m']) <= rsu['range_m']:
            out.append((vid, report))
    return out


def scaled_field(field):
    value = np.asarray(field, dtype=np.float32)
    if value.shape != (RSU_FIELD_DIM,) or not np.isfinite(value).all():
        raise ValueError('RSU field must be finite [RSU_FIELD_DIM]')
    return value/FIELD_SCALE


# ---------------------------------------------------------------- suggestion slate

def _rows(accepted):
    return {vid: dict(vid=vid, s_m=float(r.source.s_m), v_mps=float(r.source.v_mps), lane=int(r.source.lane),
                      is_cav=True) for vid, r in accepted}


def suggestion_pools(accepted, view):
    """{kind: [role-ordered group rows]} from the reporting CAVs' self states (registered
    windows/span/roles; groups whose schedule cannot fit before the release point dropped)."""
    rows = _rows(accepted)
    pools = {}
    for kind in ld.KINDS:
        found, seen = [], set()
        for vid in sorted(rows):
            own = rows[vid]
            partners = [rows[v] for v in sorted(rows) if v != vid]
            for group in ld.joint_groups(kind, view, own, partners):
                key = tuple(r['vid'] for r in group)
                if key in seen or ld._arrival(view, group) < moves._HOLD_LO:
                    continue
                seen.add(key)
                found.append(group)
        if found:
            pools[kind] = found
    return pools


def _suggestion(kind, group, view, rng):
    onset, hold, depth, target, gap_dv, zipped = ld._params(rng, kind)
    clipped = ld.clip_schedule(onset, hold, ld._arrival(view, group))
    if clipped is None:
        return None
    return dict(kind=kind, onset=clipped[0], hold=clipped[1], depth=depth, target_mps=target,
                gap_dv_mps=gap_dv, zip=zipped, anchor_s_m=float(np.mean([r['s_m'] for r in group])),
                anchor_v_mps=float(np.mean([r['v_mps'] for r in group])), radius_m=JOIN_RADIUS_M,
                group_size=len(group))


def slate(accepted, view, rsu, rng, n=RSU_CANDIDATES):
    """Balanced slate (F2 rule): random kind order each cycle, a uniform group of that kind
    and fresh registered parameter draws per slot (the CAV move draws)."""
    pools = suggestion_pools(accepted, view)
    out = []
    while pools and len(out) < n:
        for kind in rng.permutation(sorted(pools)):
            groups = pools[str(kind)]
            out.append(_suggestion(str(kind), groups[int(rng.integers(len(groups)))], view, rng))
            if len(out) == n:
                break
    return out


def descriptor(suggestion, view):
    if suggestion is None:
        return [0.]*DESCRIPTOR_DIM
    lo, hi = ld.TARGET_WINDOW_MPS
    return [float(suggestion['kind'] == 'closing_pair'), float(suggestion['kind'] == 'abreast_wall'),
            view.distance_to_drop(suggestion['anchor_s_m'])/ld.ABREAST_WINDOW_M[1],
            (suggestion['target_mps']-lo)/(hi-lo), suggestion['gap_dv_mps']/ld.GAP_DV_WINDOW_MPS[1],
            float(suggestion['zip']),
            (suggestion['onset']-moves._ONSET_LO)/(moves._ONSET_HI-moves._ONSET_LO),
            (suggestion['hold']-moves._HOLD_LO)/(moves._HOLD_HI-moves._HOLD_LO),
            (suggestion['depth']-moves._DEPTH_LO)/(moves._DEPTH_HI-moves._DEPTH_LO),
            suggestion['group_size']/len(view.lane_ids)]


# ---------------------------------------------------------------- learnable head

@dataclass
class RSUBatch:
    """Recorded RSU decisions: field [B,F], descriptors [B,N+1,D], mask, actions, logp, values."""
    field: torch.Tensor
    descriptors: torch.Tensor
    mask: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    slates: tuple = ()


class RSUHead(nn.Module):
    """Per-entry scorer over [scaled RSU field | entry descriptor]; zero-initialised output."""

    def __init__(self, n_candidates=RSU_CANDIDATES, hidden=RSU_HIDDEN):
        super().__init__()
        self.n_candidates = n_candidates
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)   # registered, arm-independent init (B = R)
            self.score = nn.Sequential(nn.Linear(RSU_FIELD_DIM+DESCRIPTOR_DIM, hidden), nn.Tanh(),
                                       nn.Linear(hidden, 1))
            self.value = nn.Sequential(nn.Linear(RSU_FIELD_DIM, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            for head in (self.score, self.value):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def distribution(self, field, descriptors, mask):
        count = self.n_candidates+1
        if (field.dim() != 2 or field.shape[1] != RSU_FIELD_DIM
                or descriptors.shape != (len(field), count, DESCRIPTOR_DIM) or mask.shape != (len(field), count)):
            raise ValueError('RSU table shape mismatch')
        if mask.dtype != torch.bool or (len(field) and not bool(mask[:, 0].all())):
            raise ValueError('none must always be available to the RSU')
        inputs = torch.cat((field[:, None, :].expand(-1, count, -1), descriptors), -1)
        logits = self.score(inputs).squeeze(-1).masked_fill(~mask, -torch.inf)
        return logits, self.value(field).squeeze(-1)

    def evaluate(self, batch):
        logits, values = self.distribution(batch.field, batch.descriptors, batch.mask)
        if (batch.actions.shape != values.shape or bool(((batch.actions < 0)
                | (batch.actions > self.n_candidates)).any())
                or not bool(batch.mask.gather(1, batch.actions[:, None]).all())):
            raise ValueError('invalid or masked RSU action')
        dist = Categorical(logits=logits)
        return dist.log_prob(batch.actions), values, dist.entropy()

    def learning_parameters(self):
        return list(self.parameters())

    def optimizer(self, *, lr):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('learning rate must be positive and finite')
        return torch.optim.Adam(self.learning_parameters(), lr=lr)

    def imitation_loss(self, batch, targets):
        """CE on recorded slates; targets index THIS recorded slate (B verified / R random)."""
        logits, _ = self.distribution(batch.field, batch.descriptors, batch.mask)
        targets = torch.as_tensor(targets, dtype=torch.long)
        if (targets.shape != (len(logits),) or bool(((targets < 0) | (targets > self.n_candidates)).any())
                or not bool(batch.mask.gather(1, targets[:, None]).all())):
            raise ValueError('invalid or masked RSU imitation target')
        return nn.functional.cross_entropy(logits, targets) if len(logits) else logits.sum()


def batch(decisions):
    """Stack recorded decision dicts (``RSU.decide`` records) into an RSUBatch."""
    return RSUBatch(torch.tensor([d['field'] for d in decisions], dtype=torch.float32).reshape(-1, RSU_FIELD_DIM),
                    torch.tensor([d['descriptors'] for d in decisions], dtype=torch.float32).reshape(
                        len(decisions), RSU_CANDIDATES+1, DESCRIPTOR_DIM),
                    torch.tensor([d['mask'] for d in decisions], dtype=torch.bool).reshape(
                        len(decisions), RSU_CANDIDATES+1),
                    torch.tensor([d['action'] for d in decisions], dtype=torch.long),
                    torch.tensor([d['logp'] for d in decisions], dtype=torch.float32),
                    torch.tensor([d['value'] for d in decisions], dtype=torch.float32),
                    tuple(tuple(d['slate']) for d in decisions))


def teacher_target(decision, kind, anchor_s_m, *, radius_m=JOIN_RADIUS_M):
    """B guidance: the available entry of the verified example's kind nearest its group
    centre (within the join radius), else none (0). Same function for both arms."""
    best = (math.inf, 0)
    for i, entry in enumerate(decision['slate']):
        if i and entry is not None and decision['mask'][i] and entry['kind'] == kind:
            gap = abs(float(entry['anchor_s_m'])-float(anchor_s_m))
            if gap <= radius_m and (gap, i) < best:
                best = (gap, i)
    return best[1]


def random_teacher_target(decision, rng):
    """R guidance: uniform over the SAME available non-none slate entries (none if empty)."""
    available = [i for i in range(1, len(decision['mask'])) if decision['mask'][i]]
    return int(available[int(rng.integers(len(available)))]) if available else 0


def rollout_records(samples, returns):
    """(decision, return) for every recorded RSU decision whose uplink CAVs have returns.

    The RSU's return is the mean per-CAV return of its own uplink CAVs at that tick (its
    observable coverage), the same per-CAV SUMO returns the trigger PPO uses."""
    out = []
    for key in sorted(returns):
        decision = (((samples[key].get('track2') or {}).get('lane_drop') or {}).get('rsu') or {}).get('decision')
        if not decision:
            continue
        members = samples[key]['context']['members']
        values = [returns[key][members.index(v)] for v in decision['uplink'] if v in members]
        if values:
            out.append((decision, float(np.mean(values))))
    return out


def train_rsu(head, optimizer, records, *, ppo, kl_budget):
    """One clipped-PPO step over this update's RSU decisions with an exact KL guard (reverted
    when above the registered budget, like the trigger). Identical for B and R."""
    from copy import deepcopy
    from .trigger_policy import ppo_update
    if not records:
        return dict(decisions=0, accepted=None)
    decisions = [d for d, _ in records]
    rollout = batch(decisions)
    returns = [r for _, r in records]
    advantages = [r-float(v) for r, v in zip(returns, rollout.values.tolist())]
    with torch.no_grad():
        before_logits = head.distribution(rollout.field, rollout.descriptors, rollout.mask)[0]
    weights, state = deepcopy(head.state_dict()), deepcopy(optimizer.state_dict())
    metrics = ppo_update(head, optimizer, rollout, advantages, returns, **ppo)
    with torch.no_grad():
        after = head.distribution(rollout.field, rollout.descriptors, rollout.mask)[0]
        kl = float(torch.distributions.kl_divergence(Categorical(logits=before_logits),
                                                     Categorical(logits=after)).mean())
    accepted = math.isfinite(kl) and kl <= kl_budget
    if not accepted:
        head.load_state_dict(weights)
        optimizer.load_state_dict(state)
    return dict(metrics, decisions=len(decisions), exact_kl=kl, accepted=accepted)


def supervision_batch(decisions, targets):
    """CE guidance on recorded slates (B: teacher_target of verified examples; R: the random
    teacher). Returns (RSUBatch, targets) for ``RSUHead.imitation_loss``."""
    if len(decisions) != len(targets):
        raise ValueError('one guidance target per RSU decision')
    return batch(decisions), list(targets)


# ---------------------------------------------------------------- runtime

class RSU:
    """Lane-drop RSU: observable field, suggestion slate, learnable head, session opening."""

    def __init__(self, lane_drop_settings, protocol):
        if protocol is None:
            raise ValueError('RSU suggestions require the lane-drop V2V join protocol')
        self.view = lane_drop_settings['view']
        self.settings = dict(lane_drop_settings['rsu'])
        self.protocol = protocol
        protocol.rsu_reception = (self.settings['position_m'], self.settings['range_m'])
        self.head = RSUHead(self.settings['candidates'], self.settings['hidden'])
        self.reseed(0)
        self.accepted, self.now, self.field = [], None, missing_field()
        self.last_suggestion_s = None

    def reseed(self, seed):
        self.generator = torch.Generator().manual_seed(int(seed))
        self.rng = np.random.default_rng(int(seed))

    def observe(self, previous_reports, now):
        """Uplink at ``now``: the CAVs' own packets published at now-1 (receipt age 1 s)."""
        self.now = int(now)
        self.accepted = uplink(previous_reports, now, self.settings)
        sessions = [s for s in self.protocol.sessions.values() if not s['released']]
        self.field = rsu_field(previous_reports, float(now), self.view, self.settings,
            open_sessions=len(sessions), rsu_active=any(s['initiator'] == ld.RSU_ID for s in sessions),
            last_suggestion_s=self.last_suggestion_s)

    def reports_for_publish(self):
        return dict(self.accepted)

    @torch.no_grad()
    def decide(self, now, *, deterministic=False):
        """One suggestion decision at ``now`` (field from t-1 uplink); opens an RSU session."""
        if self.now != int(now):
            raise ValueError('RSU must observe the same second before deciding')
        entries = slate(self.accepted, self.view, self.settings, self.rng, self.settings['candidates'])
        table = [None]+entries+[None]*(self.settings['candidates']-len(entries))
        mask = [True]*(len(entries)+1)+[False]*(self.settings['candidates']-len(entries))
        field = scaled_field(self.field)
        descriptors = [descriptor(e, self.view) for e in table]
        logits, value = self.head.distribution(torch.tensor(field)[None], torch.tensor([descriptors]),
                                               torch.tensor([mask]))
        action = (int(logits.argmax(-1)) if deterministic else
                  int(torch.multinomial(logits.softmax(-1), 1, generator=self.generator)))
        logp = float(Categorical(logits=logits).log_prob(torch.tensor([action])))
        opened = False
        if action:
            opened = self.protocol.open_rsu(table[action], now)
            if opened:
                self.last_suggestion_s = float(now)
        return dict(rule=RSU_RULE, time_s=int(now), field=field.tolist(), descriptors=descriptors,
                    mask=mask, slate=table, action=action, logp=logp, value=float(value[0]),
                    opened=opened, uplink=[vid for vid, _ in self.accepted])

    def receipt(self, decision):
        return dict(rule=RSU_RULE, position_m=self.settings['position_m'], range_m=self.settings['range_m'],
                    uplink=[vid for vid, _ in self.accepted], decision=decision)
