"""V2V integration glue (part 4 INTEGRATOR, branch c4/v2v-int).

The MODEL part widens the per-CAV policy inputs (tools/coopmwm/v2v_model.py:
zero-init extension of the actor/critic first input layer at column 29) and the
BANK part attaches each member's v2v block to the sample context
(``context['v2v'] = {member vid: V2V_DIM floats}``, tools/coopmwm/bank_v2v.py).
This module is the join: it widens a policy context with the member blocks and
routes the resulting call through the widened forward, so imitation
(tools/coopmwm/bank_imitation.py, tools/coopmwm/accel_learner.imitation) and any
scoring caller can keep calling ``policy(sample['context'])`` unchanged.

Everything stays behind the ``v2v`` flag: a context without a v2v payload (or an
unextended model) takes the exact pre-v2v code path.
"""
import torch

FEATURES = 29  # per-CAV sensor block width (tools/coopnative/policy.py), unchanged
V2V_AT = FEATURES  # widened v2v channels start right after the sensor block


def v2v_width(blocks):
    """Width carried by the bank v2v blocks ({vid: sequence}); 0 when absent."""
    for vid in (blocks or {}):
        return len(blocks[vid])
    return 0


def widen_context(context, blocks):
    """Append each member's v2v block to the per-CAV sensor features ``x``.

    ``context`` is the ``acceleration.encode(rows)`` output (``x`` rows follow the
    ``members`` order); ``blocks`` is the BANK contract payload
    ``{member vid: V2V_DIM floats}`` - members without a block get zeros.  Returns a
    NEW context dict (the input is not mutated); all other keys are shared.  Empty
    ``blocks`` returns the context unchanged (flag off / pre-v2v samples).
    """
    width = v2v_width(blocks)
    if width <= 0:
        return context
    out = dict(context)
    out['x'] = [list(map(float, row)) +
                ([0.]*width if blocks.get(vid) is None else [float(v) for v in blocks[vid]])
                for vid, row in zip(context['members'], context['x'])]
    return out


def policy_call(model, context):
    """``(mean, value)`` for a context, widened or not.

    Widened contexts (x rows wider than the 29-wide sensor block) go through
    ``v2v_model.forward_with_v2v``; plain contexts call the model unchanged, so an
    unextended model and a flag-off sample compute exactly the pre-v2v result.
    """
    x = context.get('x') or ()
    if x and len(x[0]) > FEATURES:
        from .v2v_model import forward_with_v2v
        wide = torch.as_tensor(x, dtype=torch.float32)
        rsu = torch.as_tensor(context['rsu'], dtype=torch.float32)
        return forward_with_v2v(model, wide, rsu)
    return model(context)


class WidenedPolicy:
    """Call-through wrapper so ``policy(context)`` works for widened v2v contexts.

    ``__call__`` routes through :func:`policy_call`; every other attribute
    (``log_std``, ``parameters``, ...) is delegated to the wrapped model.  Note
    ``model.sample`` keeps its own internal forward - widened callers that sample
    must use the deterministic mean via this wrapper or extend the sample path.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, context):
        return policy_call(self.model, context)

    def __getattr__(self, name):
        # copy/deepcopy probes dunders on a not-yet-populated instance: without
        # this guard the self.model lookup recurses forever (no implicit dunder
        # delegation existed before - Python binds dunders on the type).
        if name.startswith('__') or 'model' not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.model, name)
