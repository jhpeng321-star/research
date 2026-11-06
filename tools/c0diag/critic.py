"""Score the recorded current-policy states with frozen online/target critics."""
import json
from pathlib import Path
import numpy as np
import torch
from wms.control.real_returns14 import ScaledTeamCritic

torch.set_num_threads(2)
root=Path('/root/main14_loop'); out=Path('/root/c0diag_s1')
p=torch.load(root/'policy/c00.a2/policy_latest.pt',map_location='cpu',weights_only=True)
names=('s_m','v_mps','a_mps2','lane','present','is_cav','length_m','entry_backlog')
with np.load(root/'returns/returns_c00.a1.npz',allow_pickle=False) as d:
    states={n:torch.tensor(d['state__'+n]) for n in names}
    actual=np.array(d['returns'],float)
result={'actual_policy_return_mean':float(actual.mean()),'n':len(actual)}
for name in ('critic','target_critic'):
    model=ScaledTeamCritic(entry_backlog=True); model.load_state_dict(p[name]); model.eval()
    with torch.no_grad():v=model(states).numpy()
    result[name]={'predicted_mean':float(v.mean()),'error_mean':float((v-actual).mean()),
                  'rmse':float(np.mean((v-actual)**2)**.5),'prediction_std':float(v.std()),
                  'actual_std':float(actual.std()),'pearson':float(np.corrcoef(v,actual)[0,1]),
                  'predictions':v.tolist(),'actual_returns':actual.tolist()}
(out/'critic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
