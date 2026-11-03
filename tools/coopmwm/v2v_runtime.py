"""Live V2V policy/observation adapters; used only by explicitly enabled jobs.

The integrated parts expose the features and widened layers independently. This
adapter connects sampling AND scoring to those layers, including spawned SUMO
workers and exact diagnostic branches. No non-V2V policy is modified.
"""
import torch

from .acceleration import Policy as BasePolicy
from .v2v_obs import V2V_DIM, V2V_INTENT_DIM, V2VConfig, V2VIntentBuffer, intent_block_width, \
    is_intent_block_width, publish_intents, v2v_intent_features
from .v2v_model import extend_for_v2v, forward_with_v2v
from .v2v_integrate import widen_context


class Policy(BasePolicy):
    """v2v-widened policy; ``v2v_dim`` is the observation block width.

    Default ``v2v_dim=V2V_DIM`` is the plain 6-feature partner block.
    Intent-sharing arms (option ``v2v_intent``) load ``v2v_dim=V2V_INTENT_DIM``
    (or any ``K*(6+H+1)``): the same zero-initialised extension, wider block.
    """

    def __init__(self, v2v_dim: int = V2V_DIM):
        super().__init__()
        self.v2v_dim = int(v2v_dim)
        extend_for_v2v(self, self.v2v_dim)

    def forward(self, context):
        x = context['x']
        if x and len(x[0]) == 29:
            blocks = context.get('v2v')
            if not blocks:
                raise ValueError('V2V policy requires recorded partner observations')
            context = widen_context(context, blocks)
            x = context['x']
        if not x:
            x = torch.empty((0, 29+self.v2v_dim), dtype=torch.float32)
        else:
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.shape[1] != 29+self.v2v_dim:
            raise ValueError('V2V feature width mismatch')
        return forward_with_v2v(self, x, torch.as_tensor(context['rsu'], dtype=torch.float32))


def from_weights(weights, *, enabled=False):
    width = int(weights['actor.0.weight'].shape[1])
    extra = width - 33
    if extra == V2V_DIM:
        if not enabled:
            raise ValueError('widened policy without explicit V2V job/branch flag')
        model = Policy()
    elif extra > 0 and is_intent_block_width(extra):
        # Intent-widened checkpoint: the block itself is K*(6+H+1) wide.
        if not enabled:
            raise ValueError('widened policy without explicit V2V job/branch flag')
        model = Policy(v2v_dim=extra)
    elif width == 33:
        # A legacy u3 reference in an enabled mixed-policy diagnostic stays plain.
        model = BasePolicy()
    else:
        raise ValueError('unsupported V2V checkpoint width')
    model.load_state_dict(weights)
    return model


def load_checkpoint(path, config=None):
    from .v2v_model import load_checkpoint as load
    model, optimizer, generator, payload = load(path, config=config)
    if config and config.get('v2v'):
        # The loader has already remapped Adam and retained parameter identities.
        model.__class__ = Policy
        model.v2v_dim = int(config.get('v2v_dim') or V2V_DIM)
    return model, optimizer, generator, payload


def controller_class(base, *, intent=False):
    """Add causal partner features before sampling; keep the command chain intact.

    ``intent=True`` swaps the block builder for :func:`v2v_intent_features`
    (same slots/order, plus each partner's ``t-1`` planned-command broadcast
    and its fresh/stale mask), keeps a :class:`V2VIntentBuffer` resettable at
    episode start (:meth:`reset_intents`), drops despawned CAVs each step, and
    exposes :meth:`publish_planned_intents` for the runner's post-step
    broadcast.  Default (``intent=False``) is byte-identical to the plain
    v2v controller.
    """
    from .bank_v2v import _features, _frame_table, DEFAULT_CFG

    class Proxy:
        def __init__(self, real, blocks):
            self.real, self.blocks = real, blocks

        def sample(self, context, *args, **kwargs):
            context = dict(context, v2v={v: self.blocks(v) for v in context['members']})
            return self.real.sample(context, *args, **kwargs)

        def __getattr__(self, key):
            return getattr(self.real, key)

    class Controller(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.previous_frame = None
            self.intent_buffer = None if not intent else V2VIntentBuffer()
            self.intent_clip = None  # optional (lo, hi); same clip as the action

        def observe_issued(self, frame):
            self.previous_frame = frame

        def intent_cfg(self):
            return V2VConfig(v2v=True, v2v_intent=True,
                             v2v_intent_horizon=self.intent_buffer.horizon)

        def v2v_block(self, context):
            present={v['vid'] for v in self.feature_frame['vehicles'] if v.get('is_cav')}
            if self.intent_buffer is not None:
                cfg = self.intent_cfg()
                table = _frame_table(self.feature_frame)
                step = int(self.feature_frame.get('time_s', 0))
                return {v:([0.]*intent_block_width(cfg) if v not in present else
                    [float(t) for t in v2v_intent_features(table, v, self.feature_previous,
                        self.intent_buffer, step, cfg)]) for v in context['members']}
            return {v:(_features(self.feature_frame,v,self.feature_previous,DEFAULT_CFG)
                if v in present else [0.]*V2V_DIM) for v in context['members']}

        def reset_intents(self):
            """Episode start: no broadcast survives into a new episode."""
            if self.intent_buffer is not None:
                self.intent_buffer.reset()

        def publish_planned_intents(self, step_index, planned):
            """Runner hook after each policy step: broadcast `{vid: H planned accels}`."""
            if self.intent_buffer is None or not planned:
                return 0
            return publish_intents(self.intent_buffer, step_index, planned,
                                   clip=self.intent_clip)

        def apply(self, rows, commands, now):
            from .bank_v2v import causal_frame, _last_commands
            # The trace callback observes the actual post-safety command from
            # the PREVIOUS second, never the current imitation target/raw request.
            frame = causal_frame(dict(time_s=now, vehicles=rows), self.previous_frame)
            previous = {k:v for k,v in _last_commands(self.previous_frame or {}).items() if v is not None}
            self.feature_frame,self.feature_previous=frame,previous
            if self.intent_buffer is not None:
                cfg = self.intent_cfg()
                table = _frame_table(frame)
                step = int(frame.get('time_s', 0))
                blocks = lambda vid: [float(t) for t in v2v_intent_features(
                    table, vid, previous, self.intent_buffer, step, cfg)]
            else:
                blocks = lambda vid: _features(frame, vid, previous, DEFAULT_CFG)
            saved = self.policy
            self.policy = Proxy(saved, blocks)
            try:
                desired = super().apply(rows, commands, now)
            finally:
                self.policy = saved
            if self.intent_buffer is not None:
                # Despawned CAVs drop out of the broadcast table.
                self.intent_buffer.retain({v['vid'] for v in frame.get('vehicles', [])
                                           if v.get('is_cav')})
                planned = getattr(self.policy, 'last_planned_intents', None)
                if isinstance(planned, dict):
                    self.publish_planned_intents(int(now), planned)
            return desired

    return Controller


class ObservedTrace:
    """Pass-through writer exposing only the existing observable trace frame."""
    def __init__(self, stream, controller):
        self.stream, self.controller = stream, controller

    def write(self, line):
        import json
        result = self.stream.write(line)
        self.controller.observe_issued(json.loads(line))
        return result
