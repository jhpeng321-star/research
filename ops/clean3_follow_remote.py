"""Public receipt-only clean3 follower probe. No runtime imports or process control."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sealed(value, key='receipt_sha256'):
    if value[key] != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError('receipt seal mismatch: '+key)
    return value


def released_pause(path, load, root):
    rules = {
        'PAUSE_SAFETY_COMPOSITE_B_u009.json': ('B', 'B_SAFETY_REPORT_ONLY_1434.json', 'MAIN2026-11-05 14:34'),
        'PAUSE_SAFETY_COMPOSITE_R_u003.json': ('R', 'R_SAFETY_REPORT_ONLY_1230.json', 'MAIN2026-11-05 12:30')}
    if path.name not in rules:
        return False
    arm, rule_name, authority = rules[path.name]
    rule = load(root/rule_name)
    if rule is None:
        return False
    sealed(rule)
    return (rule.get('arm') == arm and rule.get('authority') == authority and
            rule.get('composite_action') == 'report_only' and
            rule.get('collision_teleport_action') == 'pause_on_any' and
            rule.get('original_pause_sha256') == hashlib.sha256(path.read_bytes()).hexdigest() and
            rule.get('amendment_sha256', 'UNREGISTERED') in (root/'AUTHORITY.md').read_text())


def probe(root, control):
    root, control = Path(root), Path(control)
    def load(path):
        if not path.is_file():
            return None
        if not (path.resolve().is_relative_to(root) or path.resolve().is_relative_to(control)):
            raise ValueError('public receipt escaped registered roots')
        return json.loads(path.read_text())
    out = dict(timestamp=datetime.now(timezone.utc).isoformat(), root=str(root),
               chain=load(control/'CHAIN_STATUS.json'), launch=load(root/'LAUNCHED.json'),
               monitors={}, diagnostics={}, mined={}, composites={}, faults={}, owners={})
    for path in [root/'STOP.json', root/'FAULT.json', *root.glob('PAUSE_*.json')]:
        if path.is_file() and not released_pause(path, load, root):
            out['faults'][path.name] = load(path)
    if not out['launch']:
        return out
    launch = out['launch']
    config_bytes = (root/'CONFIG.json').read_bytes()
    cfg = json.loads(config_bytes)
    if (hashlib.sha256(config_bytes).hexdigest() != launch['config_file_sha256'] or
            digest(cfg) != launch['runtime_sha256'] or launch['root'] != str(root)):
        raise ValueError('immutable launch/config/root binding mismatch')
    smoke, gate = (sealed(load(root/name)) for name in ('SMOKE_PASS.json', 'GATE_PASS.json'))
    if (smoke['status'] != 'SMOKE_PASS' or gate['status'] != 'PASS' or
            smoke['receipt_sha256'] != launch['smoke_sha256'] or
            gate['receipt_sha256'] != launch['gate_sha256'] or
            gate['smoke_sha256'] != smoke['receipt_sha256'] or
            gate['safety']['collisions'] != 0 or gate['safety']['teleports'] != 0):
        raise ValueError('launch smoke/gate binding or absolute safety mismatch')
    out['cap_sim_s'] = cfg['clean3_budget']['owner_cap_sim_s']
    phases = {}
    for line in (root/'CHARGES.jsonl').read_bytes().splitlines():
        row = json.loads(line)
        phases[row['phase']] = phases.get(row['phase'], 0)+row['sim_s']
    out['charged_sim_s'] = sum(phases.values())
    out['protected_total_sim_s'] = out['charged_sim_s']+cfg['clean3_budget']['protected_prior_sim_s']
    if out['protected_total_sim_s'] > out['cap_sim_s']:
        raise ValueError('protected cumulative cap exceeded')
    for phase, amount in phases.items():
        cap = cfg['clean3_budget']['g4_cap_sim_s'] if phase == 'g4_synergy_credit' else cfg['clean3_budget']['phases'][phase]
        if amount > cap:
            raise ValueError('phase cap exceeded: '+phase)
    out['phases'] = phases
    for role, owner in launch['owners'].items():
        stat = Path('/proc')/str(owner['pid'])/'stat'
        fields = stat.read_text().rsplit(')', 1)[1].split() if stat.exists() else []
        done = (root/('MONITOR_COMPLETE.json' if role == 'monitor' else role+'_COMPLETE.json')).is_file()
        alive = bool(fields and fields[0] != 'Z' and fields[19] == str(owner['start_ticks']))
        out['owners'][role] = dict(owner, alive=alive, complete=done)
        if not alive and not done:
            out['faults']['OWNER_'+role] = dict(reason='recorded owner exited before completion', **owner)
    for u in range(1, 25):
        monitor = load(root/f'LEARNING_MONITOR_u{u:03}.json')
        if monitor:
            sealed(monitor)
            if monitor['config_sha256'] != launch['runtime_sha256']:
                raise ValueError('monitor launch config binding mismatch')
            out['monitors'][str(u)] = monitor
            if any(row['metrics']['safety'][key] for row in monitor['arms']
                   for key in ('collisions', 'teleports')):
                out['faults'][f'SAFETY_u{u:03}'] = dict(reason='absolute collision/teleport event in sealed monitor')
        out['mined'][str(u)] = {}
        for arm in ('B', 'R'):
            composite = load(root/f'{arm}_SAFETY_COMPOSITE_{u:03}.json')
            if composite:
                out['composites'][arm+str(u)] = composite
            update = load(root/f'{arm}_UPDATE_{u:03}.json')
            if update:
                sealed(update, 'update_sha256')
                out['mined'][str(u)][arm] = update.get('clean3_mined_usage', {})
        if u in (2, 4, 6, 12, 24):
            kind = 'probe' if u in (2, 4) else 'replicated'
            row = load(root/f'track2_diagnostics/{kind}_u{u:03}_live.json')
            if row:
                out['diagnostics'][str(u)] = sealed(row)
    return out
