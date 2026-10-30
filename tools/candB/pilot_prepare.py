"""Bind named existing TRAIN inputs for the owner-approved exploratory pilot."""
import json
from pathlib import Path

from tools.candB import pilot_authority as a


def main():
    root = Path('/root/autodl-tmp/c4')
    loop = root/'restart20261026/arm/loop'
    inputs = dict(wm=loop/'wm/c02/wm_s.pt',
        wm_config=root/'restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml',
        current_policy=root/'candA_prod_keyerr/policy_000191.pt',
        current_policy_driver=root/'candA_keyerr/configs/execparity_cav_idm_v1.json',
        cct2=Path('authority')/f'CCT2_{a.WM_SHA}_mix.json',
        horizon_gate=loop/'longint/gate_wm2/HORIZON_GATE_WM2_REG.json',
        boundary=root/'candA_keyerr/authority_gen/BOUNDARY_WM2.json',
        pairing_receipt=loop/'returns/pairing_c03.a1.json',
        parity=Path('tools/candB/evidence/CAPREL_PARITY.json'),
        real_returns=loop/'returns/returns_c03.a1.npz',
        native_returns=loop/'returns/native_returns_c03.a1.npz',
        gate_panel=root/'restart20261026/arm/gate_panel.json')
    out = Path('authority_gen/PILOT_EXPLORATORY_AUTHORITY.json')
    out.parent.mkdir(exist_ok=True)
    doc = a.build({k: str(a.safe_path(v)) for k, v in inputs.items()}, out, exploratory=True)
    a.verify(out, exploratory=True)
    print(json.dumps(dict(label='EXPLORATORY', source_commit=doc['source_commit'],
        input_sha256=doc['input_sha256'], config_sha256=doc['config_sha256'],
        authority=str(out), authority_sha256=a.sha256(out)), indent=1))


if __name__ == '__main__':
    main()
