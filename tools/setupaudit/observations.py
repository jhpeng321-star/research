"""Inspect only the actor-visible tensors on the explicitly exported TRAIN starts."""
import argparse
import json
import os
from pathlib import Path

import torch

from wms.control.contract import OWN_FEATURES, REPORT_FEATURES
from wms.control.counterfactual import CounterfactualConfig, build_variants
from wms.control.obs import build_observations
from wms.control.policy import load_policy_components
from wms.control.rsu_field import build_rsu_field
from wms.control.train_cli import _read_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True)
    p.add_argument('--starts', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from inspect_inputs import allowed
    assert all(allowed(x) for x in (a.candidate, a.starts, a.out))
    policy, encoder, _, _ = load_policy_components(a.candidate)
    starts = torch.load(a.starts, weights_only=False)
    rows = []
    cf = CounterfactualConfig(**_read_config('configs/c4_policy14_candA_longgrad_skip.toml')['counterfactual'])
    with torch.no_grad():
        for batch in starts:
            assert all(allowed(m) for m in batch.metadata)
            obs = build_observations(batch.state)
            field = build_rsu_field(batch.state)
            signal = encoder(obs, batch.state['s_m'], field)
            own = obs.own[obs.cav_mask]
            reports = obs.reports[obs.report_mask]
            from wms.control.alwayscmd import inferred_base, inferred_cap, executed
            base, cap = inferred_base(batch.state), inferred_cap(batch.state)
            z = torch.zeros_like(base)
            ctrl = executed(batch.state, z)
            high, low = executed(batch.state, z + 1.2), executed(batch.state, z - 1.2)
            variants = build_variants(batch.state, cf, torch.Generator().manual_seed(965140))
            groups = [{k: batch.state[k][0, g['members']].tolist() for k in ('s_m', 'v_mps', 'lane')}
                      for g in variants.groups]
            rows.append(dict(metadata=batch.metadata,
                cavs=int(obs.cav_mask.sum()), reports=int(obs.report_mask.sum()),
                present=int(batch.state['present'].sum()),
                hdvs=int((batch.state['present'] & ~batch.state['is_cav']).sum()),
                entry_queue=batch.entry_queue.tolist(), groups=groups,
                own_nonzero=dict(zip(OWN_FEATURES, (own != 0).float().mean(0).tolist())),
                report_nonzero=dict(zip(REPORT_FEATURES, (reports != 0).float().mean(0).tolist())),
                rsu_nonzero_share=float((field != 0).float().mean()),
                signal_abs_mean=float(signal[obs.cav_mask].abs().mean()),
                base_below_cap_share=float((base < cap)[obs.cav_mask].float().mean()),
                positive_authority_max=float((high - ctrl)[obs.cav_mask].max()),
                braking_authority_share=float((low < ctrl)[obs.cav_mask].float().mean()),
                base_minus_cap_min=float((base - cap)[obs.cav_mask].min()),
                base_minus_cap_max=float((base - cap)[obs.cav_mask].max()),
                residual_min=float(policy(obs, signal, deterministic=True)[0][obs.cav_mask].min()),
                residual_max=float(policy(obs, signal, deterministic=True)[0][obs.cav_mask].max())))
    out = Path(a.out)
    assert out.parent.is_dir() and not out.exists()
    out.write_text(json.dumps(dict(pid=os.getpid(), rows=rows), indent=2) + '\n')
    print(out.read_text())


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
