"""Direct real-300s paired secant credit; no value model or WM return target.

This is a local action-contrast surrogate, not an unbiased policy gradient:
complete paired returns weight the observed first-step executed-action contrast
at the common saved start. No invented stochastic behavior likelihood is used
for the deterministic native-residual actor. Zero contrasts supply no gradient.
"""
from __future__ import annotations

import json

import torch

from .start_states import checked_path


def secant_loss(action, native, contrast, benefit, mask, scale):
    direction = torch.where(mask, contrast, torch.zeros_like(contrast)).detach()
    norm = direction.square().sum(-1)
    projection = ((action - native.detach()) * direction).sum(-1)
    usable = norm > 0
    credit = torch.where(usable, projection / norm.clamp_min(torch.finfo(action.dtype).eps),
                         torch.zeros_like(projection))
    return -(credit * benefit.detach()).mean() / scale, int(usable.sum())


class RealPairCredit:
    def __init__(self, sampler, pairs, records):
        self.sampler, self.pairs, self.records = sampler, pairs, records

    @classmethod
    def from_boundary(cls, doc):
        from .paircrit import PairedProvider
        from .real_returns14 import NPZRealReturnProvider
        from .start_states import StartStateSampler
        from wms.config.load import load_train_config
        from wms.data.c4_pairs import PairedBranchSource

        policy = NPZRealReturnProvider(doc['policy_returns'])
        native = NPZRealReturnProvider(doc['native_returns'])
        provider = PairedProvider(policy, native, doc['pairing_receipt'], doc['source_policy_sha256'])
        proof = json.loads(checked_path(doc['pairing_receipt']).read_text())
        cfg = load_train_config(str(checked_path(doc['wm']['config'])))
        pairs = PairedBranchSource(checked_path(doc['pairs']), horizon_steps=600,
            history_s=cfg.data.history_s, history_stride_s=cfg.data.history_stride_s,
            entries_v2=True, junction_fix=cfg.data.junction_fix,
            merge_pivot_history=cfg.model.merge_decision == 'first_passage',
            arm_kinds=('B', 'POLICY'))
        groups = {g['pair_key']: g for g in pairs.groups}
        records = []
        from .long_credit import world_role
        eligible = set(provider.fit_indices.tolist())
        for i, item in enumerate(proof['pairs']):
            # CQR calibration/validation worlds never supply actor gradients.
            if i not in eligible or world_role(item['seed']) != 'fit':
                continue
            if item['horizon_s'] != 300:
                raise ValueError('direct credit requires real paired 300s returns')
            group = groups[item['pair_key']]
            b = group['rows'][0]
            p = next(r for r in group['rows'][1:] if
                pairs.dataset._loader.entries[r]['branch']['kind'] == 'POLICY')
            records.append((p, b, float(policy._returns[i] - native._returns[i])))
        if not records:
            raise ValueError('no world-disjoint fitting pairs for direct real credit')
        result = cls(StartStateSampler(pairs.dataset), pairs, records)
        result.excluded_worlds = {int(item['seed']) for item in proof['pairs']
                                  if world_role(item['seed']) != 'fit'}
        return result

    def loss(self, trainer, scale):
        from .obs import build_observations
        from .rsu_field import build_rsu_field
        # Round-robin all proven TRAIN pairs; no action-dependent selection.
        p, b, benefit = self.records[trainer.update_count % len(self.records)]
        start = self.sampler._batch([self.sampler._load(p)], trainer.device)
        state = start.state
        obs = build_observations(state)
        encoder = trainer.group_encoder
        signal = (encoder(obs, state) if getattr(encoder, 'requires_state', False)
                  else encoder(obs, state['s_m'], build_rsu_field(state)))
        action, _ = trainer.policy(obs, signal, deterministic=True)
        loader = self.pairs.dataset._loader
        executed = []
        for row in (p, b):
            ep, tick = loader.index[row]
            executed.append(torch.as_tensor(loader.episode(ep).array('veh_kin')[tick + 1, :, 5],
                                            dtype=action.dtype, device=action.device)[None])
        native = executed[1]
        if getattr(trainer.policy, 'residual_on_native', False):
            action = action + native
        loss, count = secant_loss(action, native, executed[0] - native,
            action.new_tensor([benefit]), state['present'] & state['is_cav'], scale)
        return loss, {'direct_real_return_share': 1., 'direct_real_horizon_s': 300,
            'direct_real_credit_loss': float(loss.detach()), 'direct_real_nonzero_contrasts': count,
            'direct_real_paired_benefit': benefit, 'direct_real_credit_off_policy': True,
            'direct_real_credit_surrogate': 'common-start executed-acceleration secant'}
