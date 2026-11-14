"""WD-01 partA_act: engine kind 'idm_action' - the structured Part A
engine (part3 kind 'idm_merge') with CAV action conditioning through the
verified partA_ctrl controller emulators (A2 step 2).

Behaviour contract:
- Humans evolve exactly as the idm_merge model (per-type IDM + measured
  dawdle, M1 lane-end obstacle, M2 mandatory gap-accepted merge, M3
  cooperative leaders, M4 hazard lane changes) with the FROZEN p3m
  selected-member knobs as defaults.
- Each CAV follows its per-CAV action (native / damping / pi-external-u,
  scheduled switches via the branch dispatcher schedule):
    native        -> the idm_merge human dynamics with the CAV's OWN
                     vType (unchanged vs idm_merge);
    damping / pi  -> the partA_ctrl emulator computes the commanded speed
                     once per 1 s control tick on ideal-sensing frames of
                     the PREDICTED world state; the command is executed
                     every 0.5 s physics step by the MEASURED SUMO
                     execution rule (results/exec_rule_fit.json,
                     speedMode 31):
                       v' = max(0, min(clip(T, v - b*DT, v + a*DT),
                                       v + idm_acc*DT))
                     where the IDM safe-speed term uses the CAV's own
                     vType and the current predicted leader.  Fallback
                     ticks (no command) revert to the native path.
- PI recurrence state advances EVERY control tick for the whole live
  roster (adapter invalidate semantics), keyed by vehicle id.
- Common random numbers: human vTypes (posterior at t0, prior at birth)
  and the per-step dawdle / lane-change uniforms are hash draws keyed by
  (member_seed, vid, tick, salt), identical across arms - the
  DESIGN_PARTS_A_B.md section 2.3 common-random-number rule; the arm
  pairing therefore isolates the action conditioning.

ZERO SIMULATION: pure torch/numpy; the only external data are the
frozen p3d weights and the partA_ctrl emulators.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import act_lib as L  # noqa: E402
import emulators  # noqa: E402
import engine as E  # noqa: E402
import p3lib as P3  # noqa: E402

DT = P3.DT
GAP_MAX = P3.GAP_MAX
OFFSETS = P3.OFFSETS
MERGE_ZONE_S0 = E.MERGE_ZONE_S0

# ---- the selected p3m member (part3m results/fit_p3m.json "selected") --

MERGE_PARAMS_SELECTED = dict(tau_accept=1.9, b_accept=2.0,
                             p_coop_scale=0.0, end_force_s=150.0,
                             v_merge_blend=0.0)
LC_SCALE_SELECTED = 0.002
DAWDLE_SCALE_SELECTED = 1.0

# ---- measured SUMO execution rule (results/exec_rule_fit.json) ---------

EXEC_RULE = "kin_idm"     # max(0, min(kin clip, v + idm_acc*DT))


def execution_speed(T, v, gap, dv, a_max, b, tau, s0, v0, rule=EXEC_RULE):
    """One 0.5 s execution of the commanded target T under the CAV's
    speedMode (pure torch, vectorized).  MEASURED on chunk-1 TRAIN
    command pairs (measure_execution.py, results/exec_rule_fit.json):
    speedMode 31 - v' = max(0, min(clip(T, v - b*DT, v + a*DT),
    v + idm_acc*DT)) with the CAV's own vType IDM safe-speed term
    (p50 residual 1.7e-3 m/s; emergency braking never observed)."""
    kin = torch.clamp(T, (v - b * DT).clamp_min(0.0), v + a_max * DT)
    if rule == "kin_idm":
        acc = P3.idm_acc(v, gap, dv, a_max, b, tau, s0, v0)
        return torch.clamp(torch.minimum(kin, v + acc * DT), min=0.0)
    return torch.clamp(kin, min=0.0)


class ActionRollout:
    """One 30 s commit-window rollout of kind 'idm_action' from a
    chunk-1 branch restore state (L.ActWorld).  force_native=True is the
    CAVs-as-native naive baseline (identical dynamics, no emulator)."""

    def __init__(self, world: L.ActWorld, arm: str, *,
                 member_seed=0, lc_scale=LC_SCALE_SELECTED,
                 merge_params=None, dawdle_scale=DAWDLE_SCALE_SELECTED,
                 dev=None, force_native=False, exec_rule=EXEC_RULE):
        assert exec_rule in ("kin_idm", "kin"), "unknown execution rule"
        self.w = world
        self.arm = arm
        self.dev = dev or torch.device("cpu")
        self.kind = "idm_action"
        self.t0 = world.t0
        self.member_seed = int(member_seed)
        self.lc_scale = float(lc_scale)
        self.dawdle_scale = float(dawdle_scale)
        self.force_native = bool(force_native)
        self.exec_rule = exec_rule
        self.mp = dict(MERGE_PARAMS_SELECTED)
        if merge_params:
            self.mp.update(merge_params)
        self.emu = None
        if not self.force_native:
            self.emu = emulators.ControllerEmulators(L.refload.MODEL_ONNX,
                                                     device=self.dev)
        self.anomalies = defaultdict(int)
        self.pi_store = {}           # name -> [raw_prev, last_ts, active]
        self.cmd = {}                # name -> commanded target (finite)
        self.last_actions = {}       # name -> action applied at last tick
        self.last_actions_tick = None
        self.actions = world.actions(arm)
        # ---- state containers from the t0 restore row
        t0_map = world.t0_map
        cav_set = set(world.cav_ids)
        vids, lanes, ss, vv, aa, cavs = [], [], [], [], [], []
        for name in sorted(t0_map):
            lane, s, v, a, inj = t0_map[name]
            if lane < 0:
                self.anomalies["t0_in_junction"] += 1
                continue
            vids.append(name)
            lanes.append(lane)
            ss.append(s)
            vv.append(v)
            aa.append(a)
            cavs.append(1 if name in cav_set else 0)
        self.names = np.array(vids, dtype=object)     # authoritative ids
        self.vnum = np.array([int(n.split(".")[1]) for n in vids],
                             dtype=np.int64)          # CRN/metric key
        self.lane = torch.tensor(lanes, dtype=torch.long, device=self.dev)
        self.s = torch.tensor(ss, dtype=torch.float64, device=self.dev)
        self.v = torch.tensor(vv, dtype=torch.float64, device=self.dev)
        self.a_prev = torch.tensor(aa, dtype=torch.float64,
                                   device=self.dev)
        self.cav = torch.tensor(cavs, dtype=torch.float32, device=self.dev)
        # vTypes: CAVs keep their LOGGED type (their SUMO car-following
        # model is that vType); humans are drawn from the chunk-1
        # scenario prior 10/70/20 (CRN).  Declared deviation: the p3d
        # posterior is p2C-trained and degenerates on chunk-1 states
        # (~77% cautious draws, measured; the chunk-1 true type mix is
        # 0.10/0.72/0.18 == the prior), so the prior is the honest
        # marginal sampler here.
        true_types = world.true_types_t0()
        self.vtype = np.empty(len(vids), dtype=np.int64)
        prior = np.cumsum([0.1, 0.7, 0.2])
        for r, name in enumerate(vids):
            if cavs[r]:
                self.vtype[r] = int(true_types.get(name, 1))
            else:
                u = L.crn_uniform(self.member_seed, name, 0, "vtype")
                self.vtype[r] = int(np.searchsorted(prior, u, "left"))
        self.vtype_map = {vids[r]: int(self.vtype[r])
                          for r in range(len(vids))}
        self.lim = torch.as_tensor(E.lane_speed_limits(
            self.lane.cpu().numpy(), self.s.cpu().numpy()),
            device=self.dev, dtype=torch.float64)
        self.vcap = 1.2 * self.lim
        ll = P3.lane_lengths()
        self.llen = torch.tensor([ll[0], ll[1], ll[2]],
                                 device=self.dev,
                                 dtype=torch.float64)[self.lane].clone()
        # ---- births (arm-specific truth states; CRN vtypes)
        self.births = defaultdict(list)
        for j, rows in world.births(arm).items():
            for (name, lane, s, v, is_cav, typ) in rows:
                vt = (int(L.TYPE_OF.get(typ, 1)) if is_cav
                      else L.crn_vtype_birth(self.member_seed, name))
                self.births[j].append((name, lane, s, v, is_cav, vt))
        # ---- counters / records
        self.collisions = 0
        self.capviol = 0
        self.lc_count = 0
        self.n_birthed = 0
        self.n_exited = 0
        self.n_commands = 0
        self.snapshots = {}
        self.merge_d = []

    # ------------------------------------------------------------- utils
    def _slots_t(self):
        sl = P3.neighbour_slots(self.lane.cpu().numpy(),
                                self.s.detach().cpu().numpy(),
                                self.v.detach().cpu().numpy(),
                                self.cav.cpu().numpy())
        return torch.as_tensor(sl, device=self.dev)

    def _per_type(self, key):
        return np.array([P3.VTYPES[int(t)][key] for t in self.vtype],
                        dtype=np.float64)

    def snapshot(self):
        return dict(
            vid=self.vnum.copy(),
            names=self.names.copy(),
            lane=self.lane.detach().cpu().numpy().copy(),
            s=self.s.detach().cpu().numpy().copy(),
            v=self.v.detach().cpu().numpy().copy(),
            a=self.a_prev.detach().cpu().numpy().copy(),
            collisions=self.collisions, capviol=self.capviol,
            lc=self.lc_count, n_birthed=self.n_birthed)

    def snap_dicts(self):
        """{j: {name: (lane, s, v)}} for metrics."""
        out = {}
        for j, sn in self.snapshots.items():
            out[j] = {str(name): (int(l), float(ss), float(vv))
                      for name, l, ss, vv in zip(sn["names"], sn["lane"],
                                                 sn["s"], sn["v"])}
        return out

    # --------------------------------------------------------- control
    @torch.no_grad()
    def _control(self, tick):
        """One 1 s control tick: per-CAV action -> emulator -> commanded
        targets for the next two physics steps.  The emulator is ticked
        once for the WHOLE live CAV roster (PI invalidate semantics)."""
        if self.force_native or len(self.names) == 0:
            self.cmd.clear()
            return
        cav_rows = torch.nonzero(self.cav > 0.5).squeeze(1).tolist()
        if not cav_rows:
            self.cmd.clear()
            return
        tick6 = round(float(tick), 6)
        kinds = []
        any_ctrl = False
        last_actions = {}
        for r in cav_rows:
            name = str(self.names[r])
            action = self.actions.get((tick6, name))
            if action is None:
                self.anomalies["missing_action_row"] += 1
                action = "native"
            if action not in emulators.ACTION_CODES:
                raise ValueError(f"unsupported action {action!r} "
                                 "(chunk-1 schedules carry native / "
                                 "damping / pi-external-u only)")
            if action != "native":
                any_ctrl = True
            kinds.append(emulators.ACTION_CODES[action])
            last_actions[name] = action
        self.last_actions = last_actions
        self.last_actions_tick = tick6
        new_cmd = {}
        if any_ctrl:
            lane_np = self.lane.detach().cpu().numpy()
            s_np = self.s.detach().cpu().numpy()
            v_np = self.v.detach().cpu().numpy()
            llen_np = self.llen.detach().cpu().numpy()
            # SumoLocalSensor-exact ideal sensing (E4-verified): a
            # sensor sees only its own SUMO LANE - in model-world
            # coordinates the containing topology SEGMENT - with
            # coverage min(100, lane_end - pos) front and
            # min(100, pos - lane_start) rear.  Cross-edge neighbours
            # are invisible to the real sensor and must be invisible
            # here (the frame-sanity proof of partA_ctrl E4 pins this).
            seg_tbl = E._seg_table()
            seg_idx = np.full(len(s_np), -1, dtype=np.int64)
            seg_lo = np.zeros(len(s_np))
            seg_hi = np.zeros(len(s_np))
            seg_lim = np.zeros(len(s_np))
            for li in (0, 1, 2):
                m = np.where(lane_np == li)[0]
                if not len(m):
                    continue
                starts, lims = seg_tbl[li]
                i = np.searchsorted(starts, s_np[m], side="right") - 1
                i = np.clip(i, 0, len(lims) - 1)
                seg_idx[m] = i
                seg_lo[m] = starts[i]
                hi_idx = np.minimum(i + 1, len(starts) - 1)
                seg_hi[m] = np.where(i + 1 < len(starts),
                                     starts[hi_idx], llen_np[m])
                seg_lim[m] = lims[i]
            frames = []
            for r in cav_rows:
                lane_i = int(lane_np[r])
                pos = float(s_np[r])
                speed = float(v_np[r])
                sf = P3.VTYPES[int(self.vtype[r])]["sf"]
                allowed = float(seg_lim[r]) * sf
                front_cov = min(100.0, max(0.0, float(seg_hi[r]) - pos))
                rear_cov = min(100.0,
                               max(0.0, pos - L.VEHICLE_LENGTH_M))
                front = rear = None
                fb = rb = None
                for q in range(len(s_np)):
                    if q == r or lane_np[q] != lane_i \
                            or seg_idx[q] != seg_idx[r]:
                        continue
                    if s_np[q] >= pos:
                        gap = float(s_np[q]) - L.VEHICLE_LENGTH_M - pos
                        if gap <= front_cov and (fb is None or
                                                 (gap, q) < fb):
                            fb = (gap, q)
                            front = dict(
                                gap_m=gap,
                                relative_speed_mps=float(v_np[q]) - speed)
                    else:
                        gap = pos - L.VEHICLE_LENGTH_M - float(s_np[q])
                        if gap <= rear_cov and (rb is None or
                                                (gap, q) < rb):
                            rb = (gap, q)
                            rear = dict(
                                gap_m=gap,
                                relative_speed_mps=float(v_np[q]) - speed)
                frames.append(dict(
                    now_s=tick6, ts_s=tick6, speed_mps=speed,
                    allowed_speed_mps=allowed,
                    front_coverage_m=front_cov, rear_coverage_m=rear_cov,
                    front=front, rear=rear))
            batch = emulators.FrameBatch.stack(frames, device=self.dev)
            defaults = [0.0, float("nan"), False]
            pi_state = emulators.ControllerState(pi=emulators.PIState(
                raw_prev=torch.tensor(
                    [self.pi_store.get(str(self.names[r]), defaults)[0]
                     for r in cav_rows], dtype=torch.float64,
                    device=self.dev),
                last_ts=torch.tensor(
                    [self.pi_store.get(str(self.names[r]), defaults)[1]
                     for r in cav_rows], dtype=torch.float64,
                    device=self.dev),
                active=torch.tensor(
                    [self.pi_store.get(str(self.names[r]), defaults)[2]
                     for r in cav_rows], device=self.dev)))
            kinds_t = torch.tensor(kinds, device=self.dev)
            out = self.emu.emulate(kinds_t, pi_state, batch)
            npi = out["new_state"].pi
            for i, r in enumerate(cav_rows):
                name = str(self.names[r])
                self.pi_store[name] = [float(npi.raw_prev[i]),
                                       float(npi.last_ts[i]),
                                       bool(npi.active[i])]
                tgt = float(out["target_speed_mps"][i])
                self.n_commands += 1
                if tgt == tgt:            # finite: a command was issued
                    new_cmd[name] = tgt
        self.cmd = new_cmd

    # ------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, j):
        t_now = self.t0 + (j - 1) * DT
        for (name, lane, s, v, is_cav, vt) in self.births.get(j, ()):
            self._insert(name, lane, s, v, is_cav, vt)
        if len(self.names) == 0:
            return
        if float(t_now).is_integer():
            self._control(t_now)
        self.lim = torch.as_tensor(E.lane_speed_limits(
            self.lane.detach().cpu().numpy(),
            self.s.detach().cpu().numpy()), device=self.dev,
            dtype=torch.float64)
        self.vcap = 1.2 * self.lim
        pre_lane = self.lane.clone()
        pre_s = self.s.clone()
        self.vid_of = torch.as_tensor(self.vnum, device=self.dev)
        pre_leader = E.same_lane_leader_idx(pre_lane, pre_s)

        # -- M2 mandatory gap-accepted merge (accel -> main_right)
        merged = self._gap_accept_merge()
        # -- M4 hazard lane changes (0<->1) on the p3d hazard
        n = len(self.names)
        sl = self._slots_t()
        feats = torch.zeros((n, 19), device=self.dev)
        feats[:, 0] = self.v.float() / 40.0
        feats[:, 1] = torch.clamp((454.10 - self.s.float()) / 500.0,
                                  0.0, 1.0)
        feats[:, 2:14] = torch.clamp(
            sl[:, :, 1:3].reshape(n, 12).float() / 250.0, -1.0, 1.0)
        feats[:, 14] = ((self.lane == 2)
                        & (self.s >= MERGE_ZONE_S0)).double()
        feats[:, 15:19] = 0.0
        logits = E._p3d_hazard()(feats)
        pl = torch.sigmoid(logits[:, 0]).double()
        pr_ = torch.sigmoid(logits[:, 1]).double()
        tick6 = round(t_now, 6)
        ul = torch.as_tensor(
            [L.crn_uniform(self.member_seed, str(self.names[r]), tick6,
                           "lcl") for r in range(n)], device=self.dev)
        ur = torch.as_tensor(
            [L.crn_uniform(self.member_seed, str(self.names[r]), tick6,
                           "lcr") for r in range(n)], device=self.dev)
        fire_l = (self.lane == 1) & (ul < pl * self.lc_scale)
        fire_r = (self.lane == 0) & (ur < pr_ * self.lc_scale)
        off = torch.as_tensor(OFFSETS, device=self.dev, dtype=torch.float64)
        new_lane = self.lane.clone()
        new_lane = torch.where(merged, torch.ones_like(new_lane),
                               new_lane)
        new_lane = torch.where(fire_l, self.lane - 1, new_lane)
        new_lane = torch.where(fire_r, self.lane + 1, new_lane)
        changed = new_lane != self.lane
        self.lc_count += int(changed.sum())
        mg = merged.detach().cpu().numpy()
        if mg.any():
            d_m = (OFFSETS[2] - self.s.detach().cpu().numpy())[mg]
            self.merge_d.extend(float(x) for x in d_m)
        lim = torch.as_tensor(E.lane_speed_limits(
            self.lane.detach().cpu().numpy(),
            self.s.detach().cpu().numpy()), device=self.dev,
            dtype=torch.float64)
        if changed.any():
            self.s = torch.where(changed, self.s + off[new_lane]
                                 - off[self.lane], self.s)
            self.lane = new_lane
            lim = torch.as_tensor(E.lane_speed_limits(
                self.lane.detach().cpu().numpy(),
                self.s.detach().cpu().numpy()), device=self.dev,
                dtype=torch.float64)
            self.lim = lim
            self.vcap = 1.2 * lim
            ll = P3.lane_lengths()
            self.llen = torch.where(
                changed,
                torch.as_tensor([ll[0], ll[1], ll[2]], device=self.dev,
                                dtype=torch.float64)[self.lane],
                self.llen)
        # -- M3 cooperative leaders (selected member: p_coop_scale 0)
        fi, fg, fv = self._coop_leaders()
        slots = self._slots_t()
        gap = slots[:, 0, 1].double()
        rel = slots[:, 0, 2].double()
        free = slots[:, 0, 0] < 0.5
        gap = torch.where(free, torch.full_like(gap, GAP_MAX), gap)
        dv = torch.where(free, torch.zeros_like(rel), -rel)
        if len(fi):
            ft = torch.as_tensor(fi, device=self.dev, dtype=torch.long)
            gap = gap.clone()
            dv = dv.clone()
            gap[ft] = torch.as_tensor(fg, device=self.dev,
                                      dtype=torch.float64).clamp(min=0.3)
            dv[ft] = self.v[ft] - torch.as_tensor(
                fv, device=self.dev, dtype=torch.float64)
        a_max = torch.as_tensor(self._per_type("a_max"), device=self.dev)
        b = torch.as_tensor(self._per_type("b"), device=self.dev)
        tau = torch.as_tensor(self._per_type("tau"), device=self.dev)
        s0v = torch.as_tensor(self._per_type("s0"), device=self.dev)
        sf = torch.as_tensor(self._per_type("sf"), device=self.dev)
        v0 = sf * lim
        acc = P3.idm_acc(self.v, gap, dv, a_max, b, tau, s0v, v0)
        # -- M1 lane-end standing obstacle (queues the unmerged)
        on2 = self.lane == 2
        if bool(on2.any()):
            acc = torch.where(
                on2, torch.minimum(acc, E.lane_end_obstacle_acc(
                    self.v, self.s, on2, a_max, b, tau, s0v, v0)), acc)
        acc = acc.clamp(-9.0, 3.0)
        # -- SUMO dawdle (CRN) for the IDM path
        dawd_np = np.array([
            L.crn_uniform(self.member_seed, str(self.names[r]), tick6,
                          "dawdle") * E.SIGMA_ACCEL[int(self.vtype[r])]
            * DT for r in range(n)])
        v_dawdled = np.maximum(
            self.v.detach().cpu().numpy() + acc.detach().cpu().numpy()
            * DT - dawd_np * self.dawdle_scale, 0.0)
        v_new = torch.as_tensor(v_dawdled, device=self.dev,
                                dtype=torch.float64)
        # -- commanded CAVs: the measured SUMO execution rule
        if self.cmd:
            rows = [r for r in range(n) if str(self.names[r]) in self.cmd]
            if rows:
                T = torch.as_tensor(
                    [self.cmd[str(self.names[r])] for r in rows],
                    device=self.dev, dtype=torch.float64)
                v_new[rows] = execution_speed(
                    T, self.v[rows], gap[rows], dv[rows], a_max[rows],
                    b[rows], tau[rows], s0v[rows], v0[rows],
                    rule=self.exec_rule)
        s_new = self.s + v_new * DT
        self.capviol += int((v_new > self.vcap + 1e-6).sum())
        self.a_prev = (v_new - self.v) / DT
        self.v, self.s = v_new, s_new
        self._count_collisions(pre_lane, pre_s, pre_leader)
        self._drop_exits()

    # -------------------------------------------------------- helpers
    def _insert(self, name, lane, s, v, is_cav, vt):
        dev = self.dev
        lane_t = torch.tensor([lane], device=dev, dtype=torch.long)
        s_t = torch.tensor([float(s)], device=dev, dtype=torch.float64)
        lim = torch.as_tensor(E.lane_speed_limits(
            lane_t.cpu().numpy(), s_t.detach().cpu().numpy()), device=dev,
            dtype=torch.float64)
        v_t = torch.tensor([float(v)], device=dev,
                           dtype=torch.float64).clamp(0.0,
                                                      float(1.2 * lim))
        ll = P3.lane_lengths()
        self.names = np.append(self.names, name)
        self.vnum = np.append(self.vnum, int(name.split(".")[1]))
        self.lane = torch.cat([self.lane, lane_t])
        self.s = torch.cat([self.s, s_t])
        self.v = torch.cat([self.v, v_t])
        self.a_prev = torch.cat([self.a_prev, torch.zeros(
            1, device=dev, dtype=torch.float64)])
        self.cav = torch.cat([self.cav, torch.tensor(
            [1.0 if is_cav else 0.0], device=dev, dtype=torch.float32)])
        self.vtype = np.append(self.vtype, vt)
        self.vtype_map[name] = vt
        self.lim = torch.cat([self.lim, lim])
        self.vcap = torch.cat([self.vcap, 1.2 * lim])
        self.llen = torch.cat([self.llen, torch.tensor(
            [ll[lane]], device=dev, dtype=torch.float64)])
        self.n_birthed += 1

    def _gap_accept_merge(self):
        """M2 (engine semantics, numpy): mandatory gap-accepted merge
        accel(2)->main_right(1) inside the merge zone; front-most
        candidate first; a just-accepted vehicle occupies the target
        lane for later candidates."""
        lane_np = self.lane.detach().cpu().numpy()
        s_np = self.s.detach().cpu().numpy()
        v_np = self.v.detach().cpu().numpy()
        out = np.zeros(len(lane_np), dtype=bool)
        cand = np.where((lane_np == 2) & (s_np >= MERGE_ZONE_S0))[0]
        if not len(cand):
            return torch.as_tensor(out, device=self.dev)
        cand = cand[np.argsort(-s_np[cand])]
        on1 = np.where(lane_np == 1)[0]
        occ_d = np.array([OFFSETS[1] - x for x in s_np[on1]],
                         dtype=np.float64) if len(on1) else np.zeros(0)
        occ_v = v_np[on1].astype(np.float64).copy()
        occ_s0 = (np.array([P3.VTYPES[int(t)]["s0"] for t in
                            self.vtype[on1]], dtype=np.float64)
                  if len(on1) else np.zeros(0))
        tau_a = float(self.mp["tau_accept"])
        b_a = float(self.mp["b_accept"])
        force_s = float(self.mp.get("end_force_s", 0.0))
        for i in cand:
            d_i = OFFSETS[2] - s_np[i]
            tau_i, b_i = E.end_force_params(s_np[i], tau_a, b_a, force_s)
            ahead = occ_d < d_i
            gap_lead = (GAP_MAX if not ahead.any()
                        else float(d_i - occ_d[ahead].max()))
            behind_idx = np.where(occ_d > d_i)[0]
            if len(behind_idx):
                jj = behind_idx[np.argmin(occ_d[behind_idx])]
                gap_foll = float(occ_d[jj] - d_i)
                v_f = float(occ_v[jj])
                s0_f = float(occ_s0[jj])
            else:
                gap_foll, v_f, s0_f = GAP_MAX, 0.0, 0.0
            s0_i = P3.VTYPES[int(self.vtype[i])]["s0"]
            ok = E.merge_acceptance(
                np.array([gap_lead]), np.array([v_np[i]]),
                np.array([s0_i]), np.array([gap_foll]),
                np.array([v_f]), np.array([s0_f]), tau_i, b_i)
            if bool(ok[0]):
                out[i] = True
                occ_d = np.append(occ_d, d_i)
                occ_v = np.append(occ_v, float(v_np[i]))
                occ_s0 = np.append(occ_s0, s0_i)
        return torch.as_tensor(out, device=self.dev)

    def _coop_leaders(self):
        """M3 bridge (selected member p_coop_scale 0 -> empty)."""
        u = np.array([
            L.crn_uniform(self.member_seed, str(self.names[r]), 0, "coop")
            for r in range(len(self.names))])
        p = np.array([E.LC_COOP_PRIOR[int(t)] for t in self.vtype],
                     dtype=np.float64) * float(self.mp["p_coop_scale"])
        return E.coop_leader_map(
            self.lane.detach().cpu().numpy(),
            self.s.detach().cpu().numpy(),
            self.v.detach().cpu().numpy(), self.vtype, u, p)

    def _count_collisions(self, pre_lane, pre_s, pre_leader):
        """Per-lane order inversions after the step (engine semantics)."""
        post = {}
        for v, ss in zip(self.vnum, self.s.detach().cpu().numpy()):
            post[int(v)] = float(ss)
        post_lane = {}
        for v, l in zip(self.vnum, self.lane.detach().cpu().numpy()):
            post_lane[int(v)] = int(l)
        n = len(pre_lane)
        pl = pre_lane.detach().cpu().numpy()
        ps = pre_s.detach().cpu().numpy()
        for i in range(n):
            jj = int(pre_leader[i])
            if jj < 0:
                continue
            vi, vj = int(self.vid_of[i]), int(self.vid_of[jj])
            if vi not in post or vj not in post:
                continue
            if post_lane[vi] != pl[i] or post_lane[vj] != pl[jj]:
                continue
            if post[vi] >= post[vj] - 1e-9:
                self.collisions += 1

    def _drop_exits(self):
        gone = self.s > self.llen
        if gone.any():
            keep = ~gone
            gone_names = {str(self.names[r]) for r in
                          np.where(gone.detach().cpu().numpy())[0]}
            for name in gone_names:
                self.pi_store.pop(name, None)
                self.cmd.pop(name, None)
            self.names = self.names[keep]
            self.vnum = self.vnum[keep]
            self.lane = self.lane[keep]
            self.s, self.v = self.s[keep], self.v[keep]
            self.a_prev, self.cav = self.a_prev[keep], self.cav[keep]
            self.vtype = self.vtype[keep]
            self.lim, self.vcap, self.llen = (self.lim[keep],
                                              self.vcap[keep],
                                              self.llen[keep])
            self.n_exited += int(gone.sum())

    # ------------------------------------------------- teacher forcing
    def force_state(self, truth_row):
        """Replace the state with the branch truth row for tick
        t0 + j*DT (teacher forcing; tests).  truth_row:
        {name: (lane, s, v, a)}.  Junction rows (lane < 0) are dropped:
        the model world has no junction states (the documented p2C
        convention - junctions span zero physical s), so a CAV inside a
        junction is absent from the state until it re-emerges."""
        rows = []
        for name in sorted(truth_row):
            lane, s, v, a = truth_row[name][:4]
            if lane < 0:
                continue
            rows.append((name, lane, s, v, a))
        rows.sort()
        self.names = np.array([r[0] for r in rows], dtype=object)
        self.vnum = np.array([int(r[0].split(".")[1]) for r in rows],
                             dtype=np.int64)
        self.lane = torch.tensor([r[1] for r in rows], dtype=torch.long,
                                 device=self.dev)
        self.s = torch.tensor([r[2] for r in rows], dtype=torch.float64,
                              device=self.dev)
        self.v = torch.tensor([r[3] for r in rows], dtype=torch.float64,
                              device=self.dev)
        self.a_prev = torch.tensor([r[4] for r in rows],
                                   dtype=torch.float64, device=self.dev)
        cav_set = set(self.w.cav_ids)
        self.cav = torch.tensor([1.0 if r[0] in cav_set else 0.0
                                 for r in rows], dtype=torch.float32,
                                device=self.dev)
        # vtype: keep the sampled map for survivors; CAVs keep their
        # logged type (true_types_t0 is already TYPE_OF-mapped ints)
        vt_new = np.empty(len(rows), dtype=np.int64)
        tt = self.w.true_types_t0()
        for i, r in enumerate(rows):
            if r[0] in cav_set:
                vt_new[i] = int(tt.get(r[0], 1))
            else:
                vt_new[i] = self.vtype_map.get(r[0], 1)
            self.vtype_map[r[0]] = int(vt_new[i])
        self.vtype = vt_new
        self.lim = torch.as_tensor(E.lane_speed_limits(
            self.lane.cpu().numpy(), self.s.cpu().numpy()),
            device=self.dev, dtype=torch.float64)
        self.vcap = 1.2 * self.lim
        ll = P3.lane_lengths()
        self.llen = torch.tensor([ll[0], ll[1], ll[2]], device=self.dev,
                                 dtype=torch.float64)[self.lane].clone()

    def run(self, n_ticks=L.H30_TICKS):
        self.snapshots = {0: self.snapshot()}
        for j in range(1, n_ticks + 1):
            self.step(j)
            self.snapshots[j] = self.snapshot()
        return self

    def run_teacher(self, truth, n_ticks=L.H30_TICKS):
        """Teacher-forced roll: dynamics run per step, then the state is
        overwritten with the truth row - control ticks therefore see
        EXACTLY the logged world state (the E4 frame conditions)."""
        self.snapshots = {0: self.snapshot()}
        for j in range(1, n_ticks + 1):
            self.step(j)
            self.force_state(truth[j])
            self.snapshots[j] = self.snapshot()
        return self
