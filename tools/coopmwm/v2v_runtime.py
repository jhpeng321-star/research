"""Live V2V policy/observation adapters; used only by explicitly enabled jobs.

The integrated parts expose the features and widened layers independently. This
adapter connects sampling AND scoring to those layers, including spawned SUMO
workers and exact diagnostic branches. No non-V2V policy is modified.
"""
import torch

from .acceleration import Policy as BasePolicy
from .v2v_obs import V2V_DIM
from .v2v_model import extend_for_v2v, forward_with_v2v
from .v2v_integrate import widen_context


class Policy(BasePolicy):
    def __init__(self):
        super().__init__()
        extend_for_v2v(self, V2V_DIM)

    def forward(self, context):
        x = context['x']
        if x and len(x[0]) == 29:
            blocks = context.get('v2v')
            if not blocks:
                raise ValueError('V2V policy requires recorded partner observations')
            context = widen_context(context, blocks)
            x = context['x']
        if not x:
            x = torch.empty((0, 29+V2V_DIM), dtype=torch.float32)
        else:
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.shape[1] != 29+V2V_DIM:
            raise ValueError('V2V feature width mismatch')
        return forward_with_v2v(self, x, torch.as_tensor(context['rsu'], dtype=torch.float32))


def from_weights(weights, *, enabled=False):
    if 'intent_head.weight' in weights:  # v2v_intent ON (tools/coopmwm/v2v_intent.py)
        from .v2v_intent import from_weights as intent_weights
        return intent_weights(weights, enabled=enabled)
    width = int(weights['actor.0.weight'].shape[1])
    if width == 33+V2V_DIM:
        if not enabled:
            raise ValueError('widened policy without explicit V2V job/branch flag')
        model = Policy()
    elif width == 33:
        # A legacy u3 reference in an enabled mixed-policy diagnostic stays plain.
        model = BasePolicy()
    else:
        raise ValueError('unsupported V2V checkpoint width')
    model.load_state_dict(weights)
    return model


def load_checkpoint(path, config=None):
    if config and config.get('v2v') and config.get('v2v_intent'):
        from .v2v_intent import load_checkpoint as intent_load
        return intent_load(path, config)
    from .v2v_model import load_checkpoint as load
    model, optimizer, generator, payload = load(path, config=config)
    if config and config.get('v2v'):
        # The loader has already remapped Adam and retained parameter identities.
        model.__class__ = Policy
    return model, optimizer, generator, payload


def controller_class(base):
    """Add causal partner features before sampling; keep the command chain intact."""
    from .bank_v2v import _features, DEFAULT_CFG

    class Proxy:
        def __init__(self, real, blocks, intent=None):
            self.real, self.blocks, self.intent = real, blocks, intent

        def sample(self, context, *args, **kwargs):
            context = dict(context, v2v={v: self.blocks(v) for v in context['members']})
            out = self.real.sample(context, *args, **kwargs)
            if self.intent is not None:
                # Broadcast AFTER the policy step; partners see it at the next second.
                sample = out[0] if isinstance(out, tuple) else out
                if 'published_intent' not in sample:  # replayed samples keep their record
                    sample['published_intent'] = self.intent.publish(context)
            return out

        def __getattr__(self, key):
            return getattr(self.real, key)

    class Controller(base):
        def __init__(self, *args, **kwargs):
            online_planner = kwargs.pop('planner_online', None)  # T1B: default OFF
            super().__init__(*args, **kwargs)
            if online_planner is not None:
                if any(getattr(self, k, None) is not None for k in ('guide', 'replay', 'branch')):
                    raise ValueError('online planner requires sole runtime request ownership')
                self.online_planner = online_planner
            self.previous_frame = None
            self.intent_board = {}  # v2v_intent ON: vid -> (published second, H commands)

        def observe_issued(self, frame):
            self.previous_frame = frame

        def v2v_block(self, context):
            present={v['vid'] for v in self.feature_frame['vehicles'] if v.get('is_cav')}
            return {v:(_features(self.feature_frame,v,self.feature_previous,DEFAULT_CFG)
                if v in present else [0.]*V2V_DIM) for v in context['members']}

        def apply(self, rows, commands, now):
            from .bank_v2v import causal_frame, _last_commands
            # The trace callback observes the actual post-safety command from
            # the PREVIOUS second, never the current imitation target/raw request.
            frame = causal_frame(dict(time_s=now, vehicles=rows), self.previous_frame)
            previous = {k:v for k,v in _last_commands(self.previous_frame or {}).items() if v is not None}
            self.feature_frame,self.feature_previous=frame,previous
            saved = self.policy
            from .v2v_intent import intent_model
            intent = intent_model(saved)
            if intent is None:
                self.policy = Proxy(saved, lambda vid: _features(frame, vid, previous, DEFAULT_CFG))
            else:
                from .bank_v2v import _intent_features
                spec = dict(horizon=intent.intent_horizon, board=dict(self.intent_board))
                self.policy = Proxy(saved, lambda vid: _intent_features(frame, vid, previous, DEFAULT_CFG, spec),
                                    intent)
            try:
                desired = super().apply(rows, commands, now)
            finally:
                self.policy = saved
            if intent is not None:
                # Publish this second's intents (also from replayed samples); despawned or
                # silent CAVs simply have no fresh entry and read as invalid next second.
                published = (self.events[-1]['sample'].get('published_intent') or {}) if self.events else {}
                self.intent_board = {vid: (int(now), list(values)) for vid, values in published.items()}
            if hasattr(self, 'online_planner'):
                # One pre-safety request map still flows to CommandOwnership;
                # this hook never calls TraCI or creates a second writer.
                event = self.events[-1]
                desired = self.online_planner.apply(rows, commands, now,
                    sample=event['sample'], desired=desired, policy=saved)
                event['requests'] = desired
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
