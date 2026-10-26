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
    if summary['status'] != 'failed' or 'OutOfMemoryError: CUDA out of memory' not in summary.get('error', ''):
        raise ValueError('operational recovery requires the recorded CUDA OOM')
    if summary['seed'] != args.seed or summary['resumed_from']['checkpoint'] != args.resume:
        raise ValueError('operational recovery original resume mismatch')
    if summary.get('real_returns_file', args.real_returns) != args.real_returns:
        raise ValueError('operational recovery real returns mismatch')
    returns_sha = hashlib.sha256(checked_path(args.real_returns).read_bytes()).hexdigest()
    if returns_sha != summary['real_share']['source_sha256']:
        raise ValueError('operational recovery real returns hash mismatch')
    for old, checkpoint, config in zip(summary['wm_members'], args.wm_ckpt, args.wm_config, strict=True):
        for path, key in ((checkpoint, 'checkpoint_sha256'), (config, 'config_sha256')):
            if hashlib.sha256(checked_path(path).read_bytes()).hexdigest() != old[key]:
                raise ValueError('operational recovery frozen WM mismatch')
    start = summary['resumed_from']['update_count']
    saved = receipt['saved_update']
    curve = [r for r in summary['native_curve'] if r['update'] <= saved]
    if select_update(curve) != saved or curve[-1]['panel_sha256'] != PANEL_SHA256:
        raise ValueError('operational recovery must retain the best saved native checkpoint')
    remaining = start + args.max_updates - saved
    if remaining <= 0 or remaining % 10 or not summary['gate_panel']['passed']:
        raise ValueError('operational recovery incomplete curve/progress gate')
    args.resume = receipt['checkpoint']
    args.max_updates = remaining
    args.critic_warmup_updates = 0
    return {**receipt, 'prior': summary, 'curve': curve, 'phase_start': start,
            'updates': [u for u in summary['updates'] if u['update'] <= saved]}
