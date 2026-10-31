"""Main-only sensor capture at both native physics ticks; old Step0 is immutable."""
from .sumo import Backend


class MainBackend(Backend):
    def inference_payload(self):
        from .feedback import capture
        return capture(self)

    def observe(self, env):
        # Preserve exact evaluation tape accounting, then replace the one-second
        # belief row with both native half-second sensor projections. Never expose
        # the evaluation tape, raw slot IDs or full on-road mask to the RPC.
        super().observe(env)
        self.belief.history.pop()
        from traffic_shepherd.c4.state import RSU_MAINLINE_S_M, RSU_RAMP_S_M, in_rsu_zone
        from wms.control.contract import SENSOR_RANGE_M
        np = self.np
        raw = env.history.raw(2)
        if not hasattr(self, '_sensor_lane_carry'):
            self._sensor_lane_carry = {}
        for index in range(len(raw['kin'])):
            now = env.now-.5*(len(raw['kin'])-1-index)
            if self.belief.history and now <= self.belief.history[-1]['now_s']:
                continue
            kin, present, vids = raw['kin'][index], raw['mask'][index], raw['vid_index'][index]
            lanes = kin[:,1:4].argmax(-1)
            valid_lanes = kin[:,1:4].sum(-1) > 0
            ids = env.builder.vehicle_ids
            is_cav = np.zeros(len(present), bool)
            valid = present.copy()
            for slot in np.flatnonzero(present):
                vid = ids[int(vids[slot])]
                is_cav[slot] = vid in env.builder.cav_ids
                if valid_lanes[slot]:
                    self._sensor_lane_carry[vid] = int(lanes[slot])
                elif vid in self._sensor_lane_carry:
                    lanes[slot] = self._sensor_lane_carry[vid]
                else:
                    valid[slot] = False
            own = valid & is_cav
            detected = valid & in_rsu_zone(lanes,kin[:,0])
            # Existing nearest same-lane front/rear sensing geometry, nominal
            # length prior only; no true length/type is read for this projection.
            for i in np.flatnonzero(own):
                for ahead in (True,False):
                    delta = kin[:,0]-kin[i,0] if ahead else kin[i,0]-kin[:,0]
                    candidates = np.flatnonzero(valid & (lanes == lanes[i]) & (delta > 0))
                    if len(candidates):
                        j = min(candidates,key=lambda k:delta[k])
                        if delta[j]-5 <= SENSOR_RANGE_M:
                            detected[j] = True
            def receipt(i):
                return dict(vehicle_id=ids[int(vids[i])], timestamp_s=now,s_m=float(kin[i,0]),
                            lane=int(lanes[i]),v_mps=float(kin[i,4]),a_mps2=float(kin[i,5]))
            coverage = [dict(lane=l,s0_m=z[0],s1_m=z[1],timestamp_s=now)
                        for l,z in ((0,RSU_MAINLINE_S_M),(1,RSU_MAINLINE_S_M),(2,RSU_RAMP_S_M))]
            self.belief.observe(now,self_reports=[receipt(i) for i in np.flatnonzero(own)],
                                detections=[receipt(i) for i in np.flatnonzero(detected)],coverage=coverage)
