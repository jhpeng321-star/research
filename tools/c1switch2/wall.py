"""S-WALLSW facade and causal, self-ID-only continuation across the RPC boundary."""
import copy
import math

SCHEMA = 'c1switch2.wall/1'
FIELDS = {'schema', 'started_s', 'last_s', 'cooldown_s', 'members', 'speed_mps',
          'phase', 'assembly_since_s', 'assembled_at_s'}


def optional(value):
    return float(value) if math.isfinite(value) else None


class Wall:
    @staticmethod
    def create_sumo():
        from tools.c1switch.library import install_wall_extension
        install_wall_extension()
        from traffic_shepherd.c4.switchable_wall import SwitchableWallController
        return SwitchableWallController()

    @staticmethod
    def capture_memory(controller, env):
        law = controller.law
        exe = None if law is None else law.executor
        own = set(env.cav_ids) & set(env.api.vehicle.getIDList())
        members = []
        for slot, index in ({} if law is None else law.members).items():
            vid = env.builder.vehicle_ids[index]
            # A disappeared/recycled member is represented as absent, never an HDV ID.
            live = (vid in own and env.frame.present[slot] and env.frame.is_cav[slot]
                    and int(env.frame.vid_index[slot]) == index)
            members.append(vid if live else None)
        state = dict(schema=SCHEMA, started_s=controller.started_at_s,
            last_s=controller.last_time_s,
            cooldown_s=None if law is None else optional(law.cooldown_until), members=members,
            speed_mps=None if exe is None else float(exe.speed[0]),
            phase=None if exe is None else int(exe.state.phase[0]),
            assembly_since_s=None if exe is None else optional(exe.state.assembly_since[0]),
            assembled_at_s=None if exe is None else optional(exe.state.assembled_at[0]))
        return Wall.validate_memory(state, own, env.now)

    @staticmethod
    def validate_memory(state, own_ids, now):
        if state is None:
            return None
        if set(state) != FIELDS or state['schema'] != SCHEMA:
            raise ValueError('wall memory allowlist')
        for key in FIELDS-{'schema', 'members', 'phase'}:
            value = state[key]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError('finite wall scalar required')
        if state['started_s'] is None or not 0 <= state['started_s'] <= now:
            raise ValueError('wall start clock')
        for key in ('last_s', 'assembly_since_s', 'assembled_at_s'):
            if state[key] is not None and not state['started_s'] <= state[key] < now:
                raise ValueError('wall continuation clock')
        members = state['members']
        if not isinstance(members, list) or len(members) not in (0, 2):
            raise ValueError('wall atomic pair required')
        live = [v for v in members if v is not None]
        if any(type(v) is not str or v not in own_ids for v in live) or len(live) != len(set(live)):
            raise ValueError('wall members must be distinct self IDs')
        if members:
            if type(state['phase']) is not int or state['phase'] not in (0, 1):
                raise ValueError('wall active phase')
            if state['speed_mps'] is None or not 6 <= state['speed_mps'] <= 25:
                raise ValueError('frozen wall speed bounds')
            if state['assembly_since_s'] is None:
                raise ValueError('wall assembly clock missing')
        elif any(state[k] is not None for k in
                 ('phase', 'speed_mps', 'assembly_since_s', 'assembled_at_s')):
            raise ValueError('inactive wall executor state')
        return copy.deepcopy(state)

    @staticmethod
    def validate_lane_commands(counts):
        if any(counts.values()):
            raise ValueError('WALL has no lane-command head')

    @staticmethod
    def restore(state, *, slots, absent_slots, k):
        """Restore into anonymous rollout slots; no source simulator slots survive."""
        import numpy as np
        from traffic_shepherd.c4.baselines import WallLaw
        from traffic_shepherd.c4.switchable_wall import SwitchableWallController
        from traffic_shepherd.c4.wms_glue import wall_executor
        controller = SwitchableWallController()
        controller.start(now_s=state['started_s'])
        controller.last_time_s = state['last_s']
        controller.law = law = WallLaw(k)
        law.cooldown_until = -math.inf if state['cooldown_s'] is None else state['cooldown_s']
        empty = iter(absent_slots)
        chosen = [slots[v] if v is not None and v in slots else next(empty) for v in state['members']]
        if chosen:
            mask = np.zeros((1, k), bool)
            mask[0, chosen] = True
            law.members = {slot: slot for slot in chosen}
            law.executor = wall_executor(mask, [state['speed_mps']], prior_controller=np.ones((1, k)))
            exe = law.executor.state
            exe.phase[0] = state['phase']
            exe.assembly_since[0] = state['assembly_since_s']
            exe.assembled_at[0] = np.nan if state['assembled_at_s'] is None else state['assembled_at_s']
        return controller
