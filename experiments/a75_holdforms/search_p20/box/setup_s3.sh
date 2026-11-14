#!/bin/bash
# A75-S3 staging check on the A800 (CPU only: CUDA hidden, so no GPU context next to S2). Zero SUMO, never deletes.
set -u
P=/root/autodl-tmp/a75p20; A=$P/a75
export CUDA_VISIBLE_DEVICES=""
cd $A || { echo "SETUP FAILED: $A missing"; exit 1; }
sha256sum -c --quiet MANIFEST.sha256 || { echo "SETUP FAILED: manifest mismatch"; exit 1; }
[ "$(sha256sum ckpt/ckpt/wm_s.pt | cut -c1-64)" = 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 ] \
  || { echo "SETUP FAILED: wm_s.pt"; exit 1; }
cd $P && sha256sum -c --quiet STAGE.sha256 || { echo "SETUP FAILED: stage files"; exit 1; }
cd $A/code_split/code_v12r7b
PYTHONPATH=src:.:$A/a75ship /root/miniconda3/bin/python -B -c "
import json, torch, a75ss, a75_wmforms, wm_fix_crn
from tools.wzprop import wm_backend as W; assert hasattr(W.ParallelWM, '_split_map')
S = [m for m in json.load(open('$P/p20/regime_STATES_p20.json')) if m.get('valid') and m['cell'] == 'D2800_p20']
c = json.load(open('$P/cfg/A75S3_SEARCH_CFG_p20.json'))
b = json.load(open('$A/cfg/A75B_SEARCH_CFG.json'))
assert len(S) == 30 and c['seed'] == 20261177 and {k: v for k, v in c.items() if k != 'seed'} == {k: v for k, v in b.items() if k != 'seed'}
print('SETUP_OK torch', torch.__version__, 'cuda_visible', torch.cuda.is_available(), 'states', len(S), 'seed', c['seed'])"
