"""Observable-start adapter around the frozen SUMO CAV execution path.

Deliberately never calls CavSumoEnv.reset/build_scenario or loadState.
"""
from collections import Counter
from pathlib import Path
import time
import numpy as np

from traffic_shepherd.c4.env import CavSumoEnv


class ObservableEnv(CavSumoEnv):
    def initialize(self, bundle, config, *, seed, arm, out, horizon):
        import traci
        from traffic_shepherd.damping import CirclesController
        from traffic_shepherd.ownership import CommandOwnership
        from traffic_shepherd.c4.sensing import TableSensor
        from traffic_shepherd.c4.sumo_io import VehicleReader
        from traffic_shepherd.c4.state import FrameBuilder
        from traffic_shepherd.scenarios import observation_map
        from traffic_shepherd.simulation import start_sumo
        self.seed=seed
        self.warmup_s=300.
        self.end_s=300.+horizon
        self.mode='pi' if arm=='pi' else 'direct' if arm=='mappo' else 'native'
        self.cav_ids=frozenset(bundle['cav_ids'])
        self._circles=CirclesController(self.config)
        command=[]
        start_sumo(self.binary,config,seed=seed,config=self.config,api=traci,
                   lock_path=out.parent.parent/'launch.lock',
                   extra_cli=['--begin','300'],command_record=command)
        self.api=traci
        self.done=False
        self.actual_steps=0
        self._original_simulation_step=traci.simulationStep
        def counted_step(*args,**kwargs):
            result=self._original_simulation_step(*args,**kwargs)
            self.actual_steps+=1
            return result
        traci.simulationStep=counted_step
        for vehicle in bundle['vehicles']:
            vid=vehicle['id']
            rid='initial.'+vid
            traci.route.add(rid,vehicle['route'])
            traci.vehicle.add(vid,rid,typeID=vehicle['type'],depart='now',
                departLane=str(vehicle['lane_index']),departPos=str(vehicle['position']),
                departSpeed=str(vehicle['speed']))
            # moveTo explicitly inserts pending vehicles without a physics tick.
            traci.vehicle.moveTo(vid,vehicle['lane_id'],vehicle['position'])
            traci.vehicle.setPreviousSpeed(vid,vehicle['speed'])
        live=set(traci.vehicle.getIDList())
        expected={v['id'] for v in bundle['vehicles']}
        assert live==expected, (len(live),len(expected),sorted(expected-live))
        errors=[]
        for v in bundle['vehicles']:
            errors.append(dict(id=v['id'],lane_match=traci.vehicle.getLaneID(v['id'])==v['lane_id'],
                position_error=abs(traci.vehicle.getLanePosition(v['id'])-v['position']),
                speed_error=abs(traci.vehicle.getSpeed(v['id'])-v['speed'])))
        assert all(v['lane_match'] and v['position_error']<1e-5 and v['speed_error']<1e-5 for v in errors), errors
        self.initial_audit=dict(vehicles=len(live),max_position_error=max(v['position_error'] for v in errors),
                               max_speed_error=max(v['speed_error'] for v in errors),all_lanes_match=True,
                               command=command)
        self.ownership=CommandOwnership(traci,self.cav_ids,self._circles,cav_speed_mode=27)
        self.sensor=TableSensor(traci,self.config,observation_map(traci,self.config))
        self.slow_sensor=None
        self.reader=VehicleReader(traci,cav_ids=self.cav_ids,use_subscriptions=False)
        self.builder=FrameBuilder(self.k,self.cav_ids)
        self._table=self.reader.read()
        self.frame=self.builder.build(300.,self._table)
        self.history.clear();self.history.append(self.frame)
        self._previous_live=set(live);self._previous_live_sub=set(live)
        self._retired_cavs=set();self._lc_original={};self._lane_lengths={}
        self.command_counts=Counter();self.fallback_counts=Counter()
        self.totals=Counter();self.window_totals=Counter();self.lane_command_counts=Counter()
        self._setup_controllers(self.mode)
        self.recorder=None
        self.sensor_checks=self.sensor_mismatches=0
        self.rows=[]
        for vid,row in self._table.items():
            slot=self.builder.pool.slot_of(vid)
            self.rows.append((300.,vid,float(self.frame.s[slot]),row['speed_mps'],int(self.frame.lane[slot])))
        self.requests_observable=bundle['requests']
        self.initial_ids=expected
        self.inserted_synthetic=set()
        self.delay=Counter()
        self.cumulative={}
        return self.state()

    def _after_physics(self, after, actions, command_time, commanded):
        table=self.reader.read()
        previous=self._table
        self._table=table
        self.frame=self.builder.build(after,table)
        self.history.append(self.frame)
        assert not self.frame.dropped, self.frame.dropped
        if self.pi_session is not None:
            live=set(table);prev=self._previous_live_sub
            disappeared=(prev-live)&self.cav_ids
            self.pi_session.observe_lifecycle(live_ids=live,disappeared_ids=disappeared,id_reuse_ids=())
            self._previous_live_sub=live
        # RequestLedger convention: a departure at the reported boundary waits
        # through the preceding interval; new requests contribute partial dt.
        source=sum(max(0.,after-max(after-.5,r['depart'])) for r in self.requests_observable
                   if r['id'] not in self.inserted_synthetic)
        self.inserted_synthetic.update(self.api.simulation.getDepartedIDList())
        # S-WMBREAKDOWN BranchTracker carries the previous last speed for exits.
        speeds={v:r['speed_mps'] for v,r in table.items()}
        cohort={v:speeds.get(v,r['speed_mps']) for v,r in previous.items()}
        road=.5*sum(max(0.,1.-v/31.29) for v in cohort.values())
        self.delay.update(source=source,road=road,team=source+road)
        from traffic_shepherd.c4.geometry import lane_and_s
        for vid,row in table.items():
            if row['lane_id'].startswith(':'):
                # Frozen frame carries the last modeled location at a junction.
                slot=self.builder.pool.slot_of(vid)
                lane=-1
                s=float(self.frame.s[slot])
            else:lane,s=lane_and_s(row['lane_id'],row['lane_position_m'])
            self.rows.append((after,vid,s,row['speed_mps'],lane))
        interval=Counter(source_wait_vehicle_seconds=source)
        interval['collision_involvements']=len(self.api.simulation.getCollidingVehiclesIDList())
        interval['starting_teleports']=len(self.api.simulation.getStartingTeleportIDList())
        elapsed=after-300
        if elapsed in (120.,300.,600.):self.cumulative[str(int(elapsed))]=dict(self.delay)
        return interval

    def advance(self, policy=None):
        command=policy(self.state()) if policy is not None else None
        self._control_tick(command)

    def close(self):
        try:
            if self.api is not None and not hasattr(self,'ownership'):
                if self.api.isLoaded():self.api.close()
                self.api=None;self.done=True
            else:super().close()
        finally:
            if hasattr(self,'_original_simulation_step'):
                import traci
                traci.simulationStep=self._original_simulation_step

    def save(self,out):
        rows=self.rows
        ids=sorted({r[1] for r in rows});idx={v:i for i,v in enumerate(ids)}
        np.savez_compressed(out/'trajectory.npz',
            t=np.asarray([r[0] for r in rows]),vid=np.asarray([idx[r[1]] for r in rows],np.int32),
            s=np.asarray([r[2] for r in rows],np.float32),v=np.asarray([r[3] for r in rows],np.float32),
            lane=np.asarray([r[4] for r in rows],np.int8),vehicle_ids=np.asarray(ids))
