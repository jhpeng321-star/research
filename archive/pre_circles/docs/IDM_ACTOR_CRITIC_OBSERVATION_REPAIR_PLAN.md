# Proposed IDM Actor/Critic Observation Repair

Status: code audit and proposed future repair only. Nothing in this document
was implemented during the live-RSU diagnostic, and no existing checkpoint is
promoted.

## Confirmed current limitations

### Actor neighbor identity

`collect_neighbors` sorts neighbors by absolute distance, but
`FeatureEncoder` applies the same embedding to every neighbor and then performs
a trust-weighted mean. This DeepSets operation is permutation invariant. It
therefore removes explicit nearest-leader, nearest-follower and adjacent-lane
slot identity before the policy head. Signed relative distance remains in the
aggregate, but the actor cannot reliably attribute a particular gap or closing
speed to the immediate follower.

### Missing actor fields

The actor has an explicit leader gap and relative speed in its ego block, but
no explicit follower TTC, follower acceleration, follower secure-gap margin or
follower closing-speed slot. Neighbor slots contain relative distance, relative
speed, relative lane and turn signal, but no acceleration or TTC. The current
context also omits decision-time runway, predicted dose, terminal rear
headroom, target-speed difference and preview projection margin.

### Centralized critic omits ego

The packed MAPPO observation contains `local | peers | weights`, and the packer
correctly excludes the ego from `peers` to prevent duplication. However,
`MAPPOModel.forward` sends only the peer tensor into the critic encoder and
builds `z_critic` from pooled peers. The separately packed local/ego observation
is used by the actor but never concatenated into the critic value input.
Consequently, the current centralized critic cannot condition value on the
decision-making CAV's own state unless another peer happens to carry correlated
information.

## Proposed versioned repair

This repair should be implemented only after a damping treatment passes a
physical mechanism screen.

1. Introduce a new raw schema rather than shifting the current 134 fields.
2. Add explicit masked slots for same-lane leader and follower and nearest
   adjacent-lane front/rear vehicles. Each slot should contain relative
   distance, relative speed, acceleration, TTC or closing-time surrogate, lane
   relation and availability.
3. Retain the pooled neighbor set as background context, but do not use it as a
   substitute for the named immediate follower.
4. Add the five causal action-authority fields only when their exact
   predecision computation is frozen and deployment-available.
5. Build the critic from `ego embedding + pooled peer embedding + RSU/context
   embedding`; do not rely on peers alone.
6. Keep action masks separate from every physical embedding.

## Required tests before training

- perturbing each named actor slot changes logits when available;
- each named field receives a finite nonzero actor gradient;
- swapping leader and follower slots changes the encoding;
- masked/absent slots contribute exactly zero;
- critic value changes for an ego-only perturbation;
- critic value changes independently for a peer-only and RSU-only perturbation;
- actor loss backpropagates through ego/slot/RSU branches;
- critic loss backpropagates through ego/peer/RSU branches;
- no action-mask gradient enters the physical encoder;
- peer truncation is fail-loud and absent in the intended training cells;
- no feature contains seed, configured-demand shortcuts, postdecision dose or
  future outcomes.

## Checkpoint compatibility

The repair changes raw schema and critic input width. It requires a new schema
identifier and fresh model initialization. Older checkpoints must fail with a
clear incompatibility message; zero-padding, field shifting or partial weight
loading must not be automatic. If a deliberate encoder-transfer experiment is
later approved, it should copy only shape-compatible branches and record an
explicit transfer manifest and new checkpoint identity.

