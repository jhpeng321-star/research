"""Read-only group credit and final executed braking telemetry for owner2305."""
from contextlib import contextmanager, ExitStack
from functools import wraps
from unittest.mock import patch


def group_credit(imagined, metrics):
    import torch
    from wms.control.counterfactual import decompose
    variants = imagined.get('variants')
    if variants is None or not variants.base_row:
        return {}
    dense = decompose(imagined['returns'][:, 0].detach().mean(0), variants)
    groups = []
    samples = imagined.get('gated300_long_difference')
    on = metrics.get('gated300_active_rows')
    for index, group in enumerate(variants.groups):
        start, row, members = group['start'], group['row'], group['members']
        solo = [variants.individual_rows.get((start, m)) for m in members]
        record = dict(dense['groups'][index], row=row, members=members,
                      reference='native', horizon='dense start-of-rollout accrued return')
        record['sum_D_i'] = (record['D_G']-record['C_G'] if record['C_G'] is not None else None)
        record['D_all'] = dense['D_all'][list(variants.base_row).index(start)]
        groups.append(record)
        if samples is not None:
            values = samples.detach().mean((0, 1))
            total = float(sum(values[r] for r in solo)) if all(r is not None for r in solo) else None
            dg = float(values[row])
            draws = samples.detach()[..., row].reshape(-1)
            groups.append(dict(start=start, row=row, members=members, reference=imagined.get('gated300_reference', 'native'),
                horizon='300s raw accrued return improvement (positive better)', D_G=dg,
                sum_D_i=total, C_G=dg-total if total is not None else None, D_all=float(values[start]),
                confident_gain=bool(on[0][row]) if on is not None else None,
                median=float(torch.quantile(draws, .5)), q10=float(torch.quantile(draws, .1))))
    return dict(group_credit_terms=groups,
                confident_group_gain_rows=[g for g in groups if g.get('confident_gain')])


@contextmanager
def telemetry():
    import torch
    from wms.control import policy14, policy14_cli as cli, execution
    from wms.control.dreamer import DreamerTrainer
    original_controller, original_score = policy14.make_cav_controller, cli.independent_score
    original_actor = DreamerTrainer.actor_objective
    capture = dict(value=None)

    def actor(trainer, imagined, *args, **kwargs):
        loss, metrics = original_actor(trainer, imagined, *args, **kwargs)
        metrics.update(group_credit(imagined, metrics))
        return loss, metrics

    def controller(*args, **kwargs):
        callback = original_controller(*args, **kwargs)
        @wraps(callback)
        def measured(frame):
            result = callback(frame)
            target = capture['value']
            if target is not None:
                with torch.no_grad():
                    zeros = torch.zeros_like(frame['s_m'])
                    reference = execution.executed_residual(frame, zeros, zeros, residual_reference='ctrl0')
                    reference = ((frame['v_mps']+.5*reference).clamp_min(0.)-frame['v_mps'])/.5
                    n = target['n']
                    mask = (frame['present'] & frame['is_cav'] & (frame['lane'] >= 0))[:n]
                    if kwargs.get('act_mask') is not None:
                        mask = mask & kwargs['act_mask'][:n]
                    delta = (result[:n, :, 3]-reference[:n])[mask]
                    if delta.numel():
                        values = torch.stack((delta.min(), (-delta).clamp_min(0.).min(), result[:n, :, 3][mask].min()))
                        target['minima'] = values if target['minima'] is None else torch.minimum(target['minima'], values)
            return result
        measured._wms_checkpoint_passthrough = callback
        return measured

    def score(trainer, batch, *args, **kwargs):
        target = dict(n=batch.state['s_m'].shape[0], minima=None)
        capture['value'] = target
        try:
            result = original_score(trainer, batch, *args, **kwargs)
        finally:
            capture['value'] = None
        if target['minima'] is not None:
            result['executed_residual_minima'] = dict(zip(
                ('min_delta_mps2', 'min_braking_magnitude_mps2', 'min_executed_acceleration_mps2'),
                target['minima'].cpu().tolist()))
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(DreamerTrainer, 'actor_objective', actor))
        stack.enter_context(patch.object(policy14, 'make_cav_controller', controller))
        stack.enter_context(patch.object(cli, 'independent_score', score))
        yield
