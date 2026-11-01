"""SUMO IDM departure checks using population or motion-inferred settings only."""
from __future__ import annotations

import hashlib
import math

from .entries import NOMINAL_LENGTH_M
from wms.physics.merge_network import POSITION_EPS, lane_desired_speed

def follow_speed(speed, bumper_gap, leader_speed, params, dt):
    """Scalar IDM::_v, matching default .25s substeps."""
    n = max(1, int(dt / .25 + .5))
    gap = bumper_gap
    for _ in range(n):
        desired = params['min_gap'] + max(0., speed * params['time_headway']
            + speed * (speed - leader_speed) / (2 * math.sqrt(params['accel'] * params['decel'])))
        acceleration = params['accel'] * (1 - (speed / params['v0']) ** 4
                                         - (desired / max(1e-3, gap)) ** 2)
        speed = max(0., speed + dt / n * acceleration)
        gap -= max(0., speed - leader_speed) * dt / n
    return speed


def insertion_speed(speed, bumper_gap, leader_speed, params, dt):
    """MSLane::safeInsertionSpeed -> IDM::insertionFollowSpeed, not Krauss.

    None means physical collision/minGap rejection. No end-step extrapolation:
    the insertion caller supplies the current, already updated lane snapshot.
    """
    gap = bumper_gap - params['min_gap']
    if gap < 0:
        return None
    while True:
        secure = max(0., speed * params['time_headway'] + speed * (speed - leader_speed)
                     / (2 * math.sqrt(params['accel'] * params['decel'])))
        if gap >= secure:
            return speed
        speed2 = follow_speed(speed, bumper_gap, leader_speed, params, dt)
        speed3 = follow_speed(speed2, bumper_gap, leader_speed, params, dt)
        # SUMO source's convergence tolerance is ACCEL2SPEED(1).
        if speed2 - speed3 < dt:
            return speed2
        speed = speed2


class SumoInsertion:
    """No learned admission/hazard module. Optional mix is existing encoder output.

    Uses its exact identity-keyed sample for admission and records that ordinal
    for newborn motion, preserving the source ledger and FIFO refusal semantics.
    departLane supports free (max rear gap), best_free, first and a fixed WM lane.
    departSpeed supports max (patched), last/desired (unpatched), and explicit numeric
    speeds (unpatched, refusal if unsafe). Default SUMO departSpeed is 0.
    """
    def __init__(self, physics, mix=None, *, depart_lane='free', depart_speed='max', departure_profile=None):
        self.physics = physics
        self.mix = mix
        self.depart_lane = depart_lane
        self.depart_speed = depart_speed if depart_speed in ('max', 'last', 'desired') else float(depart_speed)
        self.departure_profile = departure_profile
        if depart_lane not in ('free', 'best_free', 'first', 0, 1, 2):
            raise ValueError('unsupported SUMO departLane')
        if depart_speed not in ('max', 'last', 'desired'):
            value = float(depart_speed)
            if not math.isfinite(value) or value < 0:
                raise ValueError('departSpeed must be nonnegative')

    def candidate(self, b, entrance, ordinal, cav=False):
        a, decel, tau, gap, _ = self.physics.idm()
        params = dict(v0=self.physics.v0_mps, accel=a, decel=decel,
                      time_headway=tau, min_gap=gap)
        if self.mix is not None:
            if callable(getattr(self.mix, 'candidate', None)):
                params.update(self.mix.candidate(b, entrance, ordinal, cav))
            else:
                pool = self.mix.pool_for(b, cav)
                key = f'errsrc-mix:{self.mix.seed}:{self.mix.row_keys[b]}:{entrance}:{ordinal}'
                index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % len(pool)
                params.update(zip(self.mix.names, pool[index].detach().cpu().tolist()))
        if any(not math.isfinite(params[n]) or params[n] <= 0
               for n in ('v0', 'accel', 'decel', 'time_headway', 'min_gap')):
            raise ValueError('nonphysical motion-inferred insertion settings')
        return params

    @staticmethod
    def gap(process, work, b, lane, length):
        rear = process._lane_rear(work, b, lane)
        if rear is None:
            return 1e9, 0.
        s, leader_length, leader_speed, _ = rear
        return s - leader_length - process.entry_s[lane] - length - POSITION_EPS, leader_speed

    def place(self, process, work, b, entrance, lanes, cav, ordinal):
        if not process.cfg.conserve_queue or process.cfg.learned_entry is not None or process.cfg.learned_admission is not None:
            raise ValueError('SUMO insertion requires exact queue and learned admission OFF')
        p = self.candidate(b, entrance, ordinal, cav)
        depart_lane, depart_speed = self.depart_lane, self.depart_speed
        if self.departure_profile is not None:
            depart_lane, depart_speed = self.departure_profile.modes(self.mix, b, entrance, ordinal)
        length = float(NOMINAL_LENGTH_M)
        lanes = sorted(lanes, reverse=True)  # SUMO rightmost first, WM indexes reversed
        if isinstance(depart_lane, int):
            lanes = [depart_lane] if depart_lane in lanes else []
        elif depart_lane == 'first':
            lanes = lanes[:1]
        else:
            # On this network both mainline entrance lanes have the same
            # 3000m-capped best continuation; ramp has only one allowed lane.
            def free_space(lane):
                rear = process._lane_rear(work, b, lane)
                return float('inf') if rear is None else rear[0] - process.entry_s[lane]
            # MSEdge::getFreeLane chooses ONE lane, not a fallback search over
            # every safe lane. Failed insertions remain pending this step.
            lanes = [max(lanes, key=free_space)]
        for lane in lanes:
            gap, leader = self.gap(process, work, b, lane, length)
            cap = lane_desired_speed(p['v0'], float(process.cfg.sumo_lane_speed_mps[lane]),
                                     self.physics.speed_limit_mps)
            if depart_speed in ('max', 'desired'):
                requested = cap
            elif depart_speed == 'last':
                requested = min(cap, leader if gap < 1e9 else cap)
            else:
                requested = float(depart_speed)
                if requested > cap:
                    continue
            # The IDM departure check uses this lane's desired speed too, while
            # the coupled newborn retains the original reference-speed setting.
            speed = insertion_speed(requested, gap, leader, {**p, 'v0': cap}, process.dt)
            if speed is None or (depart_speed != 'max' and speed < requested):
                continue
            if process._place(work, b, entrance, lane, cav, speed, length):
                slot = work.inserted[-1][1]
                work.birth_ordinals[(b, slot)] = int(ordinal)
                work.s[b, slot] = process.entry_s[lane] + length + POSITION_EPS
                return True
        return False


def configure_entries(process, physics, sampler=None):
    """Install once before a SUMO-mode rollout; retain arrival RNG and ledger."""
    if not process.cfg.conserve_queue:
        raise ValueError('merge_model=sumo requires entries.conserve_queue=True')
    if not getattr(process, 'couple_draws', False):
        raise ValueError('SUMO insertion requires TorchEntryProcessV2(couple_draws=True, row_seeds=...)')
    mix = sampler if sampler is not None and hasattr(sampler, 'pool_for') else None
    if sampler is not None and mix is None:
        raise ValueError('SUMO insertion needs identity-keyed observable entrant sampler')
    process.cfg.learned_entry = None
    process.cfg.learned_admission = None
    from wms.physics.learned_parameters import departure_parameters
    profile = departure_parameters(physics)
    process.cfg.sumo_insertion = SumoInsertion(physics, mix, departure_profile=profile)
    process.cfg.insert_rule = 'sumo_like_v2'  # reuse FIFO/ordinal/ledger plumbing only
