"""Training audit startup/selection diagnostics; no training objective changes."""


def require_real_credit_effect(trainer):
    """Reject an explicitly supplied multiplier file erased by guards at every knot."""
    from .policy14 import guarded_real_credit_weights
    if trainer.credit_multipliers is None:
        return None
    rows = []
    for position in trainer.counterfactual.curriculum_updates:
        weights = trainer.counterfactual.weights(position)
        effective, receipt = guarded_real_credit_weights(
            weights, trainer.credit_multipliers,
            individual_floor=trainer.config.real_credit_individual_floor)
        rows.append({'position': position, **receipt})
    if (not trainer.counterfactual.enabled or
            all(row['registered'] == row['effective'] for row in rows)):
        raise ValueError('--real-credit has no effective weight change after guards at any '
                         'curriculum knot; remove --real-credit for unchanged registered credit')
    return rows
