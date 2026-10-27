"""SHA-bound continuation of an operationally interrupted native learning curve."""
import hashlib
import json

from .nativeinit import PANEL_SHA256, select_update
from .start_states import checked_path


def load_recovery(raw, args):
    """A receipt is scoped to explicit retry outputs, never to a later cycle."""
    path = raw.get('run', {}).get('operational_recovery')
    if not path:
        return None
    receipt = json.loads(checked_path(path).read_text())
    if str(checked_path(args.output_dir)) not in receipt['retry_outputs']:
        return None
    for key in ('summary', 'checkpoint', 'config', 'gate_panel'):
        data = checked_path(receipt[key]).read_bytes()
        if hashlib.sha256(data).hexdigest() != receipt[key+'_sha256']:
            raise ValueError('operational recovery hash mismatch: '+key)
    if not args.native_init or args.seed != receipt['seed'] or args.max_updates != receipt['phase_updates']:
        raise ValueError('operational recovery phase identity mismatch')
    if args.config != receipt['config'] or args.gate_panel != receipt['gate_panel']:
        raise ValueError('operational recovery config/panel mismatch')
    summary = json.loads(checked_path(receipt['summary']).read_text())
    cause = summary
    if receipt.get('cause_summary'):
        data = checked_path(receipt['cause_summary']).read_bytes()
        if hashlib.sha256(data).hexdigest() != receipt['cause_summary_sha256']:
            raise ValueError('operational recovery original cause hash mismatch')
        cause = json.loads(data)
        if summary.get('error') != 'RuntimeError: own-run STOP requested; checkpoint saved':
            raise ValueError('operational recovery requires an explicit saved own-run STOP')
    if summary['status'] != 'failed' or 'OutOfMemoryError: CUDA out of memory' not in cause.get('error', ''):
        raise ValueError('operational recovery requires the recorded CUDA OOM')
    if summary['seed'] != args.seed or summary['resumed_from']['checkpoint'] != args.resume:
        raise ValueError('operational recovery original resume mismatch')
    if summary.get('real_returns_file', args.real_returns) != args.real_returns:
        raise ValueError('operational recovery real returns mismatch')
    returns_sha = hashlib.sha256(checked_path(args.real_returns).read_bytes()).hexdigest()
    source_sha = (summary['real_share']['source_sha256'] if 'real_share' in summary
                  else receipt.get('real_returns_sha256'))
    if returns_sha != source_sha:
        raise ValueError('operational recovery real returns hash mismatch')
    for old, checkpoint, config in zip(summary['wm_members'], args.wm_ckpt, args.wm_config, strict=True):
        for path, key in ((checkpoint, 'checkpoint_sha256'), (config, 'config_sha256')):
            if hashlib.sha256(checked_path(path).read_bytes()).hexdigest() != old[key]:
                raise ValueError('operational recovery frozen WM mismatch')
    start = cause['resumed_from']['update_count']
    saved = receipt['saved_update']
    curve = [r for r in summary['native_curve'] if r['update'] <= saved]
    best = select_update(curve)
    if best != receipt.get('best_update', saved) or curve[-1]['panel_sha256'] != PANEL_SHA256:
        raise ValueError('operational recovery must retain the best saved native checkpoint')
    if receipt.get('best_checkpoint'):
        data = checked_path(receipt['best_checkpoint']).read_bytes()
        if hashlib.sha256(data).hexdigest() != receipt['best_checkpoint_sha256']:
            raise ValueError('operational recovery best checkpoint hash mismatch')
    remaining = start + args.max_updates - saved
    if remaining <= 0 or not summary['gate_panel']['passed']:
        raise ValueError('operational recovery incomplete curve/progress gate')
    args.resume = receipt['checkpoint']
    args.max_updates = remaining
    args.critic_warmup_updates = 0
    return {**receipt, 'prior': summary, 'curve': curve, 'phase_start': start,
            'updates': [u for u in summary['updates'] if u['update'] <= saved]}


def restore_looplong(trainer, boundary, receipt, output):
    """A same-phase restart restores its schedule; it is not a fresh WM boundary.

    In particular, calling looplong.configure again would move the 90s knot from
    update463 to597 when resuming at514. Reuse the saved, SHA-bound refusal too,
    rather than re-fitting/qualifying credit on data which may since have changed.
    """
    import shutil
    import torch
    from .coopw import ExplorationConfig, HorizonConfig
    from .dreamer import _restore_rng
    from .looplong_real import RealPairCredit

    path = checked_path(receipt['checkpoint'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['checkpoint_sha256']:
        raise ValueError('looplong recovery checkpoint hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    saved = payload.get('looplong')
    if (not saved or saved['boundary'] != boundary
            or payload['update_count'] != receipt['saved_update']):
        raise ValueError('looplong recovery saved boundary mismatch')
    # Qualified-CQR checkpoints need additional on/off-policy state, which older
    # checkpoint schemas did not persist. Never silently invent it on recovery.
    if saved.get('cqr') is not None:
        raise ValueError('qualified CQR same-phase recovery state is not persisted')
    horizon = dict(saved['horizon'])
    allowed = horizon.pop('allowed_stage')
    trainer.horizon_curriculum = HorizonConfig(**horizon)
    trainer.horizon_curriculum.bind([boundary['wm']['sha256']])
    if trainer.horizon_curriculum.allowed_stage != allowed:
        raise ValueError('looplong recovery trusted horizon gate changed')
    trainer.exploration = ExplorationConfig(**saved['exploration']) if saved['exploration'] else None
    trainer.long_credit_requested = bool(boundary['flags'].get('long_credit'))
    trainer.long_credit_trusted_s = int(boundary['horizon_s'])
    trainer.looplong_real_credit = RealPairCredit.from_boundary(boundary)
    if trainer.looplong_real_credit.excluded_worlds:
        from .real_share import exclude_development_starts
        trainer.start_states = exclude_development_starts(
            trainer.start_states, trainer.looplong_real_credit.excluded_worlds)
        if payload['sampler'] is not None:
            trainer.start_states.load_state_dict(payload['sampler'])
    trainer.long_credit = None
    refusal = receipt['prior']['looplong_long_credit']
    if refusal.get('qualified') is not False:
        raise ValueError('looplong recovery requires the original unqualified credit receipt')
    source = path.parent / f"CQR_{boundary['wm']['sha256'][:16]}_REFUSED.json"
    if hashlib.sha256(checked_path(source).read_bytes()).hexdigest() != refusal['artifact_sha256']:
        raise ValueError('looplong recovery original credit refusal hash mismatch')
    shutil.copyfile(source, output / source.name)
    trainer.long_credit_refusal = dict(refusal)
    trainer.looplong_boundary = boundary
    trainer.looplong_logging = True
    _restore_rng(payload['rng'], trainer.device)
    return {'checkpoint_sha256': receipt['checkpoint_sha256'],
            'horizon': trainer.horizon_curriculum.as_dict(),
            'exploration': trainer.exploration.as_dict() if trainer.exploration else None,
            'credit_refusal_sha256': refusal['artifact_sha256'], 'rng_restored': True}
