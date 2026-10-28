"""MAIN15:00 candidate-A resolution and measured running receipt."""
import json
import math
from pathlib import Path
import statistics
import time

from common import ACTORS, CODE, OUT, REL, WM, WM_CONFIG, WM_SHA, accepted, complete_jsonl, now, read, sha, write_new

DEST = OUT / ('resolved_A_MAIN1500_' + read(CODE / 'SOURCE.json')['commit'][:12])


def prepare():
    accepted()
    authority = OUT / 'MAIN_1500.json'
    from wms.control.c4launch_authority import AUTHORITY, WM2_GATE
    assert sha(authority) == AUTHORITY
    fit = OUT / 'launch_fit_v1/FIT_DECISION.json'
    decision = read(fit)
    assert decision['cycles'] == [0, 1, 2] and decision['pessimistic_credit'] is True
    assert not DEST.exists()
    DEST.mkdir()
    gate_path = DEST / 'HORIZON_AUTHORITY.json'
    acceptance = REL / 'arm/loop/checks/c03.a2/ACCEPT.json'
    gate = {'schema': 'c4launch_inherited_horizon/1', 'authority': str(authority),
        'authority_sha256': AUTHORITY, 'wm_sha256': WM_SHA, 'wm_config': str(WM_CONFIG),
        'acceptance': str(acceptance), 'acceptance_sha256': sha(acceptance),
        'inherited_gate': str(REL / 'arm/loop/longint/gate_wm2/HORIZON_GATE_WM2_REG.json'),
        'inherited_gate_sha256': WM2_GATE, 'trusted_horizon_s': 90,
        'entrant_control': True, 'measured_wm3_gate': False,
        'next_boundary': 'corrected entrant-ON gate'}
    write_new(gate_path, gate)
    entrant = DEST / 'ENTRANT_AUTHORITY.json'
    write_new(entrant, {'task': 'S-C4LAUNCH', 'authority': 'MAIN2026-10-28T15:00',
        'selected_entrant_control': True, 'final': False,
        'merged_source_commit': read(CODE / 'SOURCE.json')['commit'],
        'reason': 'correctness repair; actor imagination ALL mask already controls entrants'})
    base = CODE / 'configs/c4_policy14_c4prep_on.toml'
    enabled = decision['native_anchor']
    anchor_path = decision.get('anchor') if enabled else None
    text = (f'extends = {json.dumps(str(base))}\n\n[trainer]\n'
        f'native_anchor = {str(enabled).lower()}\n'
        f'native_anchor_calibration = {json.dumps(anchor_path or "")}\n'
        '\n[data]\nfocus_cell_fraction = 0.6\n'
        '\n[policy14]\nfd_gate = false\nfree_road_gate = false\n'
        '\n[counterfactual]\ncurriculum_updates = [0, 1]\n'
        'w_individual = [8.9, 8.9]\nw_group = [9.9, 9.9]\nw_coop = [8.9, 8.9]\n'
        'w_team = [10.3, 10.3]\nw_all = [1.0, 1.0]\n'
        '\n[horizon]\nenabled = true\nhorizons_s = [10, 90]\ncurriculum_updates = [0, 1]\n'
        'entry_pad_max = [32, 288]\n'
        f'gate_file = {json.dumps(str(gate_path))}\n')
    with (DEST / 'POLICY.toml').open('x') as stream:
        stream.write(text)
    from wms.control.train_cli import _read_config
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.policy14 import guarded_real_credit_weights
    from wms.control.real_credit14 import load_multipliers
    from wms.control.dreamer import DreamerConfig
    raw = _read_config(DEST / 'POLICY.toml')
    cfg = DreamerConfig(**raw['trainer'])
    assert cfg.entrant_control and cfg.deterministic_crn and cfg.real_credit_individual_floor
    assert cfg.native_residual_train_group and cfg.native_residual_train_trunk
    assert not getattr(cfg, 'hold_move', False) and cfg.future_samples == 8 and cfg.pessimistic_credit
    if enabled:
        from wms.control.antiexploit import inference_signature
        artifact = read(anchor_path)
        assert artifact['inference_signature'] == inference_signature(cfg)
        assert all(artifact['horizons'][str(h)]['validated'] for h in (10, 60, 90))
    credit = REL / 'arm/loop/sumocredit/REAL_CREDIT_c03.a1.json'
    cf = CounterfactualConfig(**raw['counterfactual'])
    weights = [guarded_real_credit_weights(cf.weights(k), load_multipliers(credit),
        individual_floor=True)[1] for k in cf.curriculum_updates]
    assert all(w['effective']['individual'] == w['effective']['coop'] == 8.9 for w in weights)
    boundary = read(REL / 'arm/loop/LOOP14_STATE.json')['looplong_boundary']
    boundary.update(wm={'checkpoint': str(WM), 'sha256': WM_SHA, 'config': str(WM_CONFIG), 'source': 'cycle 3'},
        gate_file=str(gate_path), gate_sha256=sha(gate_path), horizon_s=90,
        previous_horizon_s=90, imagination_horizon_s=90, accepted_after_cycle=3,
        pairs=str(REL / 'arm/loop/data/pairs_c03.a1'),
        pairing_receipt=str(REL / 'arm/loop/returns/pairing_c03.a1.json'),
        policy_returns=str(REL / 'arm/loop/returns/returns_c03.a1.npz'),
        native_returns=str(REL / 'arm/loop/returns/native_returns_c03.a1.npz'),
        source_policy_checkpoint=str(ACTORS[3][0]), source_policy_sha256=ACTORS[3][1],
        coop_credit=True, authority='MAIN2026-10-28T15:00')
    boundary['flags']['hot_swap'] = False
    write_new(DEST / 'BOUNDARY.json', boundary)
    from launch import descriptor, validate
    doc = {**descriptor(True), 'authority': 'MAIN2026-10-28T15:00', 'ready': True, 'blockers': [],
        'trusted_horizon_s': 90, 'native_anchor': enabled, 'pessimistic_credit': True,
        'fit_decision': str(fit), 'entrant_receipt': str(entrant), 'gate': str(gate_path),
        'policy_config': str(DEST / 'POLICY.toml'), 'boundary': str(DEST / 'BOUNDARY.json'),
        'real_credit': str(credit), 'real_credit_weight_knots': weights}
    if enabled:
        doc['anchor'] = anchor_path
    for key in ('fit_decision', 'entrant_receipt', 'gate', 'policy_config', 'boundary', 'real_credit') + (('anchor',) if enabled else ()):
        doc[key + '_sha256'] = sha(doc[key])
    write_new(DEST / 'CONFIG.json', doc)
    validate(DEST / 'CONFIG.json')
    from wms.control.looplong import read_boundary
    read_boundary(DEST / 'BOUNDARY.json')
    print(json.dumps({'resolved': str(DEST), 'anchor': enabled, 'weights': weights,
                      'horizon_s': 90, 'ready_for_real_dry_run': True}))


def status(verify=False):
    result = {'t': now(), 'resolved': str(DEST)}
    for name in ('DRY_COORDINATOR_PID.json', 'DRY_RUN_VERIFIED.json', 'MAIN_LAUNCH.json', 'POLICY_COMPLETE.json'):
        if (DEST / name).exists():
            result[name] = read(DEST / name)
    for mode in ('DRY_RUN', 'POLICY'):
        root = DEST / mode
        item = {}
        pid_path = root / 'A_PID.json'
        if pid_path.exists():
            receipt = read(pid_path)
            stat = Path(f"/proc/{receipt['pid']}/stat")
            values = stat.read_text().rsplit(')', 1)[1].split() if stat.exists() else []
            item.update(pid=receipt['pid'], alive=bool(values and values[19] == receipt['start_ticks'] and values[0] != 'Z'))
        updates = root / 'A/updates.jsonl'
        if updates.exists():
            rows = complete_jsonl(updates)
            item.update(updates=len(rows), last=rows[-1] if rows else None)
            if rows:
                item['finite'] = all(all(math.isfinite(row[k]) for k in
                    ('actor_loss', 'critic_loss', 'actor_grad_norm', 'critic_grad_norm')) for row in rows)
                names = ('group_encoder', 'own_encoder', 'report_encoder', 'attention', 'residual_head')
                item['module_gradient_min'] = {k: min(row['actor_module_grad_norms'][k] for row in rows) for k in names}
                item['mean_abs_residual_mps2'] = statistics.mean(row['physical_residual_mps2']['mean_abs'] for row in rows)
                item['mean_minus_pessimistic_credit'] = statistics.mean(row['mean_paired_credit'] - row['pessimistic_paired_credit'] for row in rows)
                item['peak_reserved_bytes'] = max(row['cuda_peak_reserved_bytes'] for row in rows)
                if len(rows) > 1:
                    rate = (len(rows)-1) / (rows[-1]['t'] - rows[0]['t'])
                    item.update(updates_per_hour=rate*3600, eta_seconds=(150-len(rows))/rate)
        log = root / 'A.log'
        if log.exists():
            item['log_tail'] = log.read_text()[-2500:]
        summary = root / 'A/RUN_SUMMARY.json'
        if summary.exists():
            doc = read(summary)
            item.update(summary_status=doc['status'], error=doc.get('error'), start_focus=doc.get('start_focus'))
        gate = root / 'A/gate_panel_result.json'
        if gate.exists():
            item['signal_gate'] = read(gate)
        native = root / 'A/native_curve.jsonl'
        native_rows = complete_jsonl(native)
        if native_rows:
            item['native_panel'] = native_rows[-1]
        result[mode] = item
    if verify:
        item = result['POLICY']
        assert item['updates'] >= 10 and item['finite'] and item['alive']
        assert item.get('signal_gate', {}).get('passed') is True, 'phase-relative signal gate not yet passed'
        assert all(v > 0 for v in item['module_gradient_min'].values())
        assert math.isfinite(item['mean_abs_residual_mps2']) and math.isfinite(item['mean_minus_pessimistic_credit'])
        memory = DEST / 'POLICY/gpu_memory.jsonl'
        samples = complete_jsonl(memory)
        total = int(subprocess_total())
        peak = max(row['bytes'] for row in samples)
        assert 0 < peak <= .45 * total and 0 < item['peak_reserved_bytes'] <= .45 * total
        result.update(task='S-C4LAUNCH', status='C4A_RUNNING', new_sim_s=0,
            source_commit=read(CODE / 'SOURCE.json')['commit'], sampled_process_peak_bytes=peak,
            native_anchor=read(DEST / 'CONFIG.json')['native_anchor'])
        receipt = DEST / 'RUNNING_VERIFIED.json'
        if receipt.exists():
            assert read(receipt)['source_commit'] == result['source_commit']
        else:
            write_new(receipt, result)
        status_path = REL / 'AUTONOMY_STATUS.json'
        current = read(status_path)
        current.setdefault('peer_status_at_c4a_launch', {k: v for k, v in current.items() if k != 'c4a_preparation'})
        current.update(status='C4A_RUNNING', task='S-C4LAUNCH', t=now(),
            pid=item['pid'], cycle=4, c4a_preparation={'status': 'C4A_RUNNING', 'ready': True,
            'receipt': str(receipt), 'source_commit': result['source_commit']})
        temp = status_path.with_suffix('.c4launch.tmp')
        with temp.open('x') as stream:
            json.dump(current, stream, indent=1)
        temp.replace(status_path)
    return result


def subprocess_total():
    import subprocess
    return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total',
        '--format=csv,noheader,nounits'], text=True).strip()) * 1024**2
