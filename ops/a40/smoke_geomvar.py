"""A40 smoke (phase geomvar_a40_smoke, cap 30,000 sim-s; Windows laptop; NEVER analysed as evidence).

Functional checks only, on the wz3to2v80 variant:
 (1) the registered generator (map_scenarios, wz3to2v80 profile) writes scenarios SUMO 1.27.1 runs; native driving is
     stable (collisions / teleports counted) at D2400 / D2800 / D3200 p20;
 (2) the frozen DUO_HOLD executor (a32_arms.make_core, duo12_v20 row of UNION_UNC 39f384a2, cell D2800_p20) and the
     frozen LEARNED_TRIGGER core (learned_trigger.LearnedCore, bank fd385a2b + model 8e0ddd3e, threshold from the
     model doc) load unchanged on the variant, see plant-like rows, score / fire and issue hold commands.
Base = NATIVE SUMO CAVs (the frozen MAPPO U0 + leverprobe plant refuse non-POSIX hosts), so the U0 commands passed to
the cores are zeros and NOT the evaluation plant. Command path as the A38 smoke: slowDown(v, clip(v + a*1 s, 0,
allowed), 1 s); release setSpeed(-1); speedMode 27 on insertion; members laneChangeMode 513 while held.
Seed 994399 (last integer of the audited TRAIN block 994200-994399, used for smoke only).

usage: python smoke_geomvar.py OUT_DIR SPEC_JSON  (SPEC_JSON = list of [arm, demand, end_s]; arm NATIVE|DUO_HOLD|LT)
"""
import json, os, sys, time
from pathlib import Path

SRC = Path('E:/agent-work/a40-geomvar')
MAPDIR = SRC/'assets/research/newmap_candidates/wz3to2v80'
os.environ['COOPMWM_MAP_SPEC'] = str(MAPDIR/'MAP_SPEC.json')
sys.path.insert(0, str(SRC))
from tools.coopmwm import map_descriptor as md, map_scenarios as ms, a32_arms, learned_trigger as L  # noqa: E402
import traci  # noqa: E402

SEED, PEN, CELL = 994399, 0.2, 'D2800_p20'
LEDGER_CAP = 30_000
BANKS = Path('E:/agent-work/a40/local_root')
SHAS = dict(bank='fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82',
            model='8e0ddd3e241ce0172736909133f0b745c58b6bf795e06ca5364e5755f7e3b720',
            union='39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239')


def _doc(path, sha):
    import hashlib
    b = path.read_bytes()
    assert hashlib.sha256(b).hexdigest() == sha, path
    return json.loads(b)


def make_core(arm):
    if arm == 'NATIVE':
        return None
    if arm == 'DUO_HOLD':
        return a32_arms.make_core('DUO_HOLD', CELL, union_doc=_doc(BANKS/'a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json',
                                                                  SHAS['union']))
    if arm == 'LT':
        return L.LearnedCore(_doc(BANKS/'oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json', SHAS['bank']), CELL,
                             _doc(BANKS/'oracle_banks/LEARNED_TRIGGER_MODEL.json', SHAS['model']))
    raise SystemExit(f'unknown arm {arm}')


def charged(out):
    p = out/'CHARGES.jsonl'
    return sum(json.loads(l)['sim_s'] for l in p.read_text().splitlines()) if p.exists() else 0


def run(out, arm, demand, end_s):
    desc = md.current()
    assert desc.name == 'wz3to2v80' and ms.MAP == 'wz3to2v80', (desc.name, ms.MAP)
    root = out/f'{arm}_D{demand}_{end_s}'
    spec = json.loads((MAPDIR/'MAP_SPEC.json').read_text())
    scene, reqs = ms.write_scenario(root, MAPDIR, spec, ms.MAP, SEED, {'main': demand}, PEN)
    core = make_core(arm)
    cav = {r['id'] for r in reqs if r['cav']}
    traci.start(['sumo', '-c', str(scene), '--step-length', '0.5', '--seed', str(SEED), '--no-step-log', 'true',
                 '--no-warnings', 'true', '--end', str(end_s)])
    held, member_lanes, coll, tele, ndep, narr = set(), [], 0, 0, 0, 0
    vmax_wz, wz_speeds, up_speeds = 0.0, [], []
    t0, step = time.time(), 0
    while traci.simulation.getTime() < end_s:
        traci.simulationStep()
        step += 1
        for v in traci.simulation.getDepartedIDList():
            ndep += 1
            if v in cav:
                traci.vehicle.setSpeedMode(v, 27)
        narr += traci.simulation.getArrivedNumber()
        coll += traci.simulation.getCollidingVehiclesNumber()
        tele += traci.simulation.getStartingTeleportNumber()
        if step % 2:
            continue
        t = int(round(traci.simulation.getTime()))
        rows = []
        for v in traci.vehicle.getIDList():
            lid = traci.vehicle.getLaneID(v)
            sp = traci.vehicle.getSpeed(v)
            if lid.startswith('wz_'):
                vmax_wz = max(vmax_wz, sp)
                if t >= 300 and t % 10 == 0:
                    wz_speeds.append(sp)
            elif lid.startswith('up_') and t >= 300 and t % 10 == 0:
                up_speeds.append(sp)
            try:
                lane, s = desc.lane_and_s(lid, traci.vehicle.getLanePosition(v))
            except ValueError:
                continue                            # internal junction lane: not in the frame table
            rows.append(dict(vid=v, s_m=s, v_mps=sp, lane=lane, is_cav=v in cav, length_m=traci.vehicle.getLength(v)))
        if core is None:
            continue
        commands = {r['vid']: 0.0 for r in rows if r['is_cav']}
        accel, lanes = core.step(rows, commands, t)
        for vid, a in accel.items():
            v = traci.vehicle.getSpeed(vid)
            allowed = traci.vehicle.getAllowedSpeed(vid)
            traci.vehicle.slowDown(vid, max(0.0, min(v + a, allowed)), 1.0)
            if vid not in held:
                traci.vehicle.setLaneChangeMode(vid, 513)
                member_lanes.append([t, vid, int(traci.vehicle.getLaneIndex(vid))])
            held.add(vid)
        for vid in list(held - set(accel)):
            if vid in traci.vehicle.getIDList():
                traci.vehicle.setSpeed(vid, -1)
            held.discard(vid)
    traci.close()
    mean = lambda x: round(sum(x)/len(x), 2) if x else None  # noqa: E731
    rec = dict(arm=arm, demand=demand, end_s=end_s, seed=SEED, departed=ndep, arrived=narr, collisions=coll,
               teleports=tele, max_speed_wz_mps=round(vmax_wz, 2), mean_speed_wz_mps_300plus=mean(wz_speeds),
               mean_speed_up_mps_300plus=mean(up_speeds), wall_s=round(time.time()-t0, 1),
               not_evidence='Windows smoke, native base; functional only, never analysed')
    if core is not None:
        r = core.receipts()
        rec.update({k: r.get(k) for k in ('activations', 'issued', 'scores', 'fires', 'activations_by_family')},
                   plans=[p['family'] for p in core.plans], refused=getattr(core, 'refused', None),
                   first_members=member_lanes[:6], member_sumo_lanes=sorted({m[2] for m in member_lanes}))
    return rec


def main():
    out, plan = Path(sys.argv[1]), json.loads(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for arm, demand, end_s in plan:
        if charged(out) + end_s > LEDGER_CAP:
            raise SystemExit(f'phase geomvar_a40_smoke cap {LEDGER_CAP} would be exceeded')
        with open(out/'CHARGES.jsonl', 'a') as f:      # precharge before physics
            f.write(json.dumps(dict(phase='geomvar_a40_smoke', arm=arm, demand=demand, sim_s=end_s,
                                    t=time.strftime('%F %T'))) + '\n')
        rec = run(out, arm, demand, end_s)
        print(json.dumps(rec, default=str), flush=True)
        with open(out/'SMOKE.jsonl', 'a') as f:
            f.write(json.dumps(rec, default=str) + '\n')


if __name__ == '__main__':
    main()
