"""Write-once amendment artifacts; no dispatch, inference, checkpoint load or SUMO."""
import json
from pathlib import Path

from tools.c1switch.design import sha
from tools.c1switch.storage import write

from .design import DESIGN, PROTOCOL, budget, manifest, worlds


def main():
    root = Path('docs/paper_results/c1switch2')
    old = json.loads((root/'PANEL.json').read_text())
    if old['worlds'] != worlds():
        raise ValueError('frozen panel identities changed')
    panel = dict(old, design_sha256=sha(DESIGN), prediction_objective='on_road_fleet_delay',
                 evaluation_objective='full_team_delay_fleet_plus_source',
                 library=['native', 'ctrl0', 'mappo', 'pi', 'circles', 'wall'], cadence_s=20,
                 predecessor_panel_sha256=sha(root/'PANEL.json'))
    outputs = {root/'PANEL_v2.json': panel,
               root/'BUDGET_INTEGRATED_v2.json': dict(base=manifest(), conditional_wm_phys=manifest(True))}
    if any(p.exists() for p in outputs):
        raise ValueError('preserve previous amendment artifacts')
    for path, row in outputs.items():
        write(path, row)
    print(json.dumps(dict(protocol=PROTOCOL, design_sha256=sha(DESIGN),
        panel=str(root/'PANEL_v2.json'), panel_sha256=sha(root/'PANEL_v2.json'),
        base_sim_s=sum(budget().values()), conditional_sim_s=sum(budget(True).values()),
        new_sim_s=0, new_gpu_s=0), indent=2))


if __name__ == '__main__':
    main()
