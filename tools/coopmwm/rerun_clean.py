"""PREPARE ONLY (MAIN/owner 2026-11-04 14:4x "yes, prepare it now"): clean B-v2v / R-v2v rerun from
MAPPO u0 with every fix ON from u0. Nothing here dispatches, reserves, charges or runs SUMO.

Pinned options (B == R): V2V intent (v2v_dim 48, horizon 5, loss 1.0, detach), per_seed_v1
cooperative_only bank, imitation_step_mode kl_targeted (KL .045), practice_starts 0.25 / cap 2,
imitation_target_space 'effective' (c4/effective-imitation; PINNED COMMIT SLOT - must be filled and
parity-checked before any launch), 24 updates per arm. A is NOT rerun: the running coop3 A-v2v
(u24) stays the control.
"""
import hashlib
import json
from pathlib import Path

EPISODE_SIM_S = 1200
EPISODES_PER_UPDATE = 24
UPDATES = 24
PRACTICE_REFS_PER_UPDATE = 6           # one matched A-policy reference per practice episode
GATE_SIM_S = 35820                     # cooperative_only exact gate (7 states / 55 branches), measured x5
LESSON_SIM_S_MEASURED = 5280           # B u6 V2V lesson (v2v1_lesson_u006_*), measured
ISOLATED_SIM_S_MEASURED = 32610        # B u6 isolated synergy, measured
TEST_SIM_S_PER_ARM = 96 * EPISODE_SIM_S
SCHEDULE = dict(B=dict(lessons=[6, 12, 18, 24], isolated=[6, 12, 18, 24]),
                R=dict(lessons=[], isolated=[12, 24]))
PINS = dict(v2v=True, v2v_dim=48, v2v_intent=True, v2v_intent_horizon=5, v2v_intent_loss_weight=1.0,
            v2v_intent_detach_trunk=True, bank_imitation='separate_full_pass', imitation_kl_cap=.045,
            bank_statistics='per_seed_v1', imitation_bank_filter='cooperative_only',
            imitation_step_mode='kl_targeted', v2v_intent_bank_term=True, practice_starts=True,
            practice_fraction=.25, practice_repeat_cap=2, imitation_target_space='effective',
            stop_after_update=UPDATES, A_stop_after_update=UPDATES)
EFFECTIVE_IMITATION_SLOT = dict(branch='c4/effective-imitation', commit=None,
    parity='raw-mode byte identity + B==R pin + bind_runtime acceptance on the merged source (Linux)')


def quote(measured=None):
    """Full-run sim-s quote for B and R (24 updates each) from the measured u7-u10 receipts."""
    m = dict(train_per_arm_update=EPISODES_PER_UPDATE*EPISODE_SIM_S,
             practice_refs_per_arm_update=PRACTICE_REFS_PER_UPDATE*EPISODE_SIM_S,
             lesson=LESSON_SIM_S_MEASURED, isolated=ISOLATED_SIM_S_MEASURED, gate=GATE_SIM_S,
             test_per_arm=TEST_SIM_S_PER_ARM, **(measured or {}))
    per_update = m['train_per_arm_update'] + m['practice_refs_per_arm_update']
    train = 2*UPDATES*per_update
    lessons = len(SCHEDULE['B']['lessons'])*m['lesson']
    isolated = (len(SCHEDULE['B']['isolated'])+len(SCHEDULE['R']['isolated']))*m['isolated']
    tests = 2*2*m['test_per_arm']                     # u8 interim + u24 final, B and R
    out = dict(per_arm_update_sim_s=per_update, v2v_train=train, v2v_lessons=lessons,
               v2v_isolated=isolated, v2v_interim=2*m['test_per_arm'], v2v_final=2*m['test_per_arm'],
               gate_bank_verification_ext=m['gate'])
    out['total_without_tests'] = train + lessons + isolated + m['gate']
    out['total'] = out['total_without_tests'] + tests
    out['inputs'] = m
    return out


def wall_clock(pace_s=1800, measured_update_wall_s=1800, diagnostic_extra_s=900,
               test_episode_wall_s=300, test_workers=2):
    """Paced updates: one update per pace_s per arm (measured u7-u10 walls ~1,800 s); B's
    diagnostic updates add ~diagnostic_extra_s (u6 measured 2,700 s); R trails B by ~one update."""
    per_update = max(pace_s, measured_update_wall_s)
    b = UPDATES*per_update + len(SCHEDULE['B']['isolated'])*diagnostic_extra_s
    r = b + per_update
    tests = 2*96*test_episode_wall_s/test_workers            # B+R final test, 2 workers
    return dict(per_update_s=per_update, B_s=b, R_s=r, final_test_s=tests,
                total_h=round((r + 2*tests)/3600, 1))


def runtime_config(base, *, root, legacy_root, source_root, common_checkpoint_sha256, budget_transfer=None):
    """Derived from the runtime8 producer: prepare_intent with every option pinned (B == R), the
    24-update cap and the rerun-only keys; refuses while the effective-imitation slot is empty."""
    from . import v2v_runner
    if not EFFECTIVE_IMITATION_SLOT['commit']:
        raise ValueError('effective-imitation commit slot is empty: merge + parity check first')
    cfg = v2v_runner.prepare_intent(base, root=root, legacy_root=legacy_root, source_root=source_root,
        common_checkpoint_sha256=common_checkpoint_sha256, imitation_step_mode='kl_targeted',
        practice_starts=True, practice_fraction=.25, practice_repeat_cap=2,
        budget_transfer=budget_transfer, imitation_target_space='effective')
    cfg.update(rerun_clean=True, rerun_arms=['B', 'R'], control_arm=dict(arm='A', root='coop3', update=24),
               effective_imitation_commit=EFFECTIVE_IMITATION_SLOT['commit'])
    check_pins(cfg)
    return cfg


def check_pins(cfg):
    bad = {k: cfg.get(k) for k, v in PINS.items() if cfg.get(k) != v}
    if bad:raise ValueError('rerun pin mismatch: '+json.dumps(bad, sort_keys=True))
    return True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight_dry(prep_dir, *, budget_request='RERUN_BUDGET_REQUEST_DRAFT.json', approval='OWNER_APPROVAL.json'):
    """Zero-physics dry preflight: refuses unless OWNER_APPROVAL.json binds the exact budget request
    SHA, the owner/evening verdict fields are positive and the effective-imitation slot is pinned."""
    prep = Path(prep_dir);req = prep/budget_request;app = prep/approval
    if not req.is_file():raise SystemExit('REFUSED: budget request missing: '+str(req))
    if not app.is_file():raise SystemExit('REFUSED: no OWNER_APPROVAL.json (owner budget approval required)')
    a = json.loads(app.read_text())
    if a.get('budget_request_sha256') != sha(req):
        raise SystemExit('REFUSED: OWNER_APPROVAL budget sha differs from the request')
    if a.get('owner_approved') is not True or a.get('evening_verdict') != 'positive':
        raise SystemExit('REFUSED: owner approval or post-fix evening verdict not positive')
    if not (a.get('effective_imitation_commit') and a['effective_imitation_commit'] == EFFECTIVE_IMITATION_SLOT['commit']):
        raise SystemExit('REFUSED: effective-imitation commit not pinned/approved')
    return dict(ready_for_registered_launch=True, budget_request_sha256=sha(req), new_sim_s=0)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser();p.add_argument('cmd', choices=('quote', 'preflight'))
    p.add_argument('--prep', default='.');a = p.parse_args()
    print(json.dumps(dict(quote=quote(), wall_clock=wall_clock())) if a.cmd == 'quote'
          else json.dumps(preflight_dry(a.prep)))
