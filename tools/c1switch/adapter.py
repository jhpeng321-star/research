"""Frozen WM2/BB-traj observable feedback inference, isolated from SUMO truth.

Uses the frozen audited CAV plant for BOTH models. This is a model of native
SUMO execution, not a claim of an exact SUMO surrogate. Entries are OFF.
"""
import copy
import importlib
import os
from pathlib import Path
import sys
from types import MethodType

from .design import CONTROLLERS, FORBIDDEN, sha
from .feedback import seed_pi, seed_state, validate
from .observable import rollout_inputs, anonymous_payload
from .storage import canonical

BASE = Path('/root/autodl-tmp/c4/speedhead20261030')


def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if FORBIDDEN.search(path) or any(x in path.lower() for x in
                ('traj_base.npz','/labels/','/episode/episode.npz')):
            raise PermissionError('forbidden inference input')
    if event == 'subprocess.Popen':
        raise PermissionError('inference cannot start subprocesses')


def move(value, device):
    import torch
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k:move(v,device) for k,v in value.items()}
    return value


class Adapter:
    def __init__(self, binding, device):
        import torch
        self.binding, self.device = binding, device
        self.optimized = True
        torch.set_num_threads(1)
        sys.addaudithook(audit)
        if device == 'cuda':
            # Two model services share one GPU; total allocator ceilings are5GiB.
            torch.cuda.set_per_process_memory_fraction(2.5*2**30/torch.cuda.get_device_properties(0).total_memory)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        sys.path.insert(0,str(BASE/'tools/speedhead'))
        sys.path[:0] = [str(BASE/'e3final/runtime/src'),str(BASE/'e3final/runtime')]
        frozen = importlib.import_module('wm_worker')
        self.frozen = frozen
        torch.manual_seed(965140)
        models, self.receipt = frozen.load_members([str(frozen.WM_PATH)], [str(frozen.WM_CONFIG)], device)
        self.reference = models[0]
        if sha(frozen.WM_PATH) != binding['wm_checkpoint_sha256']:
            raise ValueError('WM checkpoint identity')
        if binding['model'] == 'wm':
            self.model = self.reference
        else:
            from wms.config.train import TrainConfig
            from wms.model.checkpoint import load_checkpoint
            path = Path(binding['checkpoint_path'])
            if sha(path) != binding['bb_checkpoint_sha256']:
                raise ValueError('BB checkpoint identity')
            self.model = load_checkpoint(path, device=device, strict=True,
                model_kind='blackbox_traj',config_factory=TrainConfig.from_legacy_flat).model.eval()
        candidate = torch.load(frozen.CANDIDATE,map_location='cpu',weights_only=False)
        self.cfg = frozen.replace(frozen.DreamerConfig(**candidate['config']),horizon=60,entrant_learned_mix=True)
        self.model.cav_exec_cap = self.cfg.cav_exec_cap
        def cap(this, out, state, params):
            from wms.control.alwayscmd import inferred_cap
            return inferred_cap(state)
        self.model._exec_cap = MethodType(cap,self.model)
        export = __import__('json').loads((BASE/'EXPORT_BOUND.json').read_text())
        if sha(export['path']) != 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca':
            raise ValueError('MAPPO checkpoint')
        self.mp = frozen.load_policy_fn(export['path'],device=device)
        self.mp.policy.eval(); self.mp.group_encoder.eval()
        if not self.mp.native_lanes or self.mp.has_lane_head:
            raise ValueError('speed-only MAPPO required')
        from wms.sim_bridge import load_controller_dependencies
        deps = load_controller_dependencies(BASE/'e3final/runtime')
        self.deps = (deps.config,deps.circles_type,deps.direct_target,deps.neighbor_type,
                     deps.frame_type,deps.pi_type,deps.u_report_type,deps.pi_params_type)
        source_paths = {Path(m.__file__) for name,m in tuple(sys.modules.items())
                        if (name.startswith('wms.') or name.startswith('traffic_shepherd.') or
                            name in ('wm_worker','measure','panel2')) and getattr(m,'__file__',None)
                        and Path(m.__file__).suffix == '.py'}
        self.attestation = dict(model=binding['model'],device=device,entries=False,mix_configured=True,
            mix_active=False,checkpoint_sha256=sha(frozen.WM_PATH) if binding['model']=='wm' else sha(path),
            wm_config_sha256=sha(frozen.WM_CONFIG),candidate_sha256=sha(frozen.CANDIDATE),
            mappo_sha256=sha(export['path']),sources={str(p):sha(p) for p in sorted(source_paths)},
            allocator_limit_gib=2.5,plant='shared inferred-native cap and gap6 ramp; native lanes approximated')
        if binding['model'] == 'bb':
            self.attestation['physics_sha256'] = sha(path.with_name('physics.json'))

    def callback(self, payload, tracker, states):
        import torch
        from wms.control.continuous import project_native
        from wms.control.bridge import make_cav_controller
        from .merge_fast import gap6
        from wms.model.cav import WM_LANE_TARGET_KEY, WM_MERGE_KEY
        from wms.train.cav_drive import TimelineControllers
        frozen = self.frozen
        native = frozen.native_fn('head',a_max=1.2)
        reference = self.reference
        class Plant:
            physics = reference.physics
            def physics_accel(self, state):
                return project_native(state,native(state))
        plant = Plant()
        class ClockPlant:
            physics = reference.physics
            def physics_accel(self,state):
                # Timeline's native output is ignored below; its controller
                # recurrence/sensing are pure CPU, the shared native plant stays
                # on its original device with identical arithmetic.
                return torch.zeros_like(state['s_m'])
        calls = []
        for arm in CONTROLLERS:
            if arm == 'mappo':
                mp = copy.copy(self.mp)
                if self.optimized:
                    # Native-lane speed-only actors never consume lane feasibility
                    # or dwell. The frozen bridge otherwise computes and discards
                    # these at every tick; actor acceleration/command memory agree.
                    mp.lane_rules = None
                calls.append(frozen.callback_for('mappo',None,None,mp,native))
            elif arm == 'ctrl0':
                def zero(obs, signal):
                    return torch.zeros_like(obs.cav_mask,dtype=torch.float32), None
                zero.residual_on_native, zero.accel_max = True, 1.2
                bridge = make_cav_controller(zero,lambda *a:None,frozen.safe_accel,native=native,
                    native_accel_max=1.2,symmetric_exec_cap=True,native_ramp_merge='wm_law',native_lane_change=True)
                calls.append(bridge)
            elif arm == 'native':
                def native_cb(state):
                    mask = state['present'].bool() & state['is_cav'].bool()
                    state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
                    return frozen.parity_motion(state,state['lane'],plant.physics_accel(state))
                calls.append(native_cb)
            else:
                ids = torch.full((1,61,states['s_m'].shape[1]),2 if arm == 'pi' else 1,dtype=torch.long)
                timeline = TimelineControllers(ClockPlant() if self.optimized else plant,dict(controller=ids,
                    acceleration_mps2=torch.zeros_like(ids,dtype=torch.float32),
                    target_speed_mps=torch.zeros_like(ids,dtype=torch.float32)),dependencies=self.deps)
                seed_pi(timeline,tracker,payload,arm)
                def baseline(state,timeline=timeline):
                    state[WM_MERGE_KEY] = state['present'].bool() & state['is_cav'].bool() & (state['lane'] == 2)
                    if self.optimized:
                        keys = ('s_m','v_mps','a_mps2','lane','present','is_cav','length_m')
                        packed = torch.stack([state[k].float() for k in keys]).cpu()
                        sensed = {key:packed[i] for i,key in enumerate(keys)}
                        timeline(sensed)
                        held = torch.full_like(sensed['s_m'],float('inf'))
                        for key,value in timeline.active.items():
                            b,k = map(int,key.split(':'))
                            held[b,k] = value
                        native_accel = state.get('merge_pivot_native_accel')
                        if native_accel is None:
                            native_accel = plant.physics_accel(state)
                        accel = torch.minimum(native_accel,held.to(state['s_m'].device))
                    else:
                        accel = timeline(state)[...,3]
                    return frozen.parity_motion(state,state['lane'],accel)
                calls.append(baseline)
        def callback(state):
            # BB-traj has no WM discretionary lane pipeline. Use the same explicit
            # ramp-gap CAV lane plant for both learned HDV models (C1FAIR contract).
            state[WM_LANE_TARGET_KEY] = state['lane']
            outputs = []
            for i, call in enumerate(calls):
                row = {k:v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0] == 5 else v
                       for k,v in state.items()}
                out = call(row)
                if not self.optimized:
                    from wms.control.bridge import _merge_lane
                    cav = row['present'].bool() & row['is_cav'].bool()
                    out[...,1] = torch.where(cav,_merge_lane(row,cav),out[...,1])
                outputs.append(out)
            state[WM_MERGE_KEY] = torch.zeros_like(state['present'],dtype=torch.bool)
            result = torch.cat(outputs)
            if self.optimized:
                cav = state['present'].bool() & state['is_cav'].bool()
                result[...,1] = torch.where(cav,gap6(state,cav),result[...,1])
            return result
        return callback

    def predict(self, payload, K=8, horizon_s=60, *, steps=None, validation_seed=None):
        import torch
        validate(payload)
        payload = anonymous_payload(payload)
        if K != 8 or horizon_s != 60:
            raise ValueError('registered K8/60s required')
        n = 120 if steps is None else steps  # CPU integration gate may use two ticks.
        state, kwargs, tracker = rollout_inputs(payload,self.model)
        # Free anonymous slots never enter a rollout with admissions OFF. Compact
        # them before pairwise dynamics, preserving every present belief track.
        keep = torch.where(state['present'][0])[0]
        if not len(keep):
            keep = torch.tensor([0])
        remap = {int(old):new for new,old in enumerate(keep)}
        for tr in tracker.tracks.values():
            tr.slot = remap[tr.slot]
        state = {key:value[:,keep] for key,value in state.items()}
        kwargs = {key:value[:,keep] for key,value in kwargs.items()}
        state, kwargs = move(state,self.device), move(kwargs,self.device)
        state['exit_s_m'] = torch.full_like(state['s_m'],self.frozen.road_end_exit_bound(self.reference.physics))
        rows = [seed_state(copy.deepcopy(state),tracker,payload,c) for c in CONTROLLERS]
        initial = {key:torch.cat([row[key] for row in rows]) for key in rows[0]}
        seed = int(canonical(payload)[:8],16) & 0x7fffffff if validation_seed is None else validation_seed
        torch.manual_seed(seed)
        plans = (self.frozen.SampledFutures(self.frozen.FuturesConfig.from_trainer(self.cfg),seed)
                 .plan([self.model],state,kwargs,n)[0]) if self.binding['model'] == 'wm' else [{} for _ in range(8)]
        samples = {c:[] for c in CONTROLLERS}
        for k in range(8):
            st = copy.deepcopy(initial)
            self.frozen.pair_merge_streams(st,torch.zeros(5,dtype=torch.long,device=self.device),1,seed+k)
            kw = self.frozen.expand({**kwargs,**plans[k]},5)
            lanes = self.frozen.lane_uniform_draws([self.model],1,n,state['s_m'].shape[1],seed+1000003*k,device=self.device)
            if lanes is not None:
                kw['lane_uniform'] = self.frozen.expand(lanes,5)
            costs = torch.zeros(5,device=self.device)
            class FleetRecorder:
                def insert(self, current):
                    costs.add_(.5*(torch.clamp(1-current['v_mps']/31.29,min=0)*current['present']).sum(-1))
                    return torch.zeros_like(current['present'],dtype=torch.bool)
            kw['entries'] = FleetRecorder()
            with torch.inference_mode():
                roll = self.model.rollout(st,n_steps=n,cav_controller=self.callback(payload,tracker,st),**kw)
                if not torch.isfinite(roll['speeds']).all() or not torch.isfinite(costs).all():
                    raise ValueError('nonfinite model rollout')
            for i,c in enumerate(CONTROLLERS):
                samples[c].append(float(costs[i]))
        return samples


def create(binding, device='cuda'):
    return Adapter(binding,device)
