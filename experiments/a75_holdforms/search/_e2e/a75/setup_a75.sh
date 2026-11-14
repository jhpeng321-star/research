#!/bin/bash
# A75 box setup check (run after: tar xzf a75_search_box.tgz -C /root/autodl-tmp). Zero SUMO, never deletes.
set -u
A=/root/autodl-tmp/a75
cd $A || { echo "SETUP FAILED: $A missing"; exit 1; }
sha256sum -c --quiet MANIFEST.sha256 || { echo "SETUP FAILED: manifest mismatch"; exit 1; }
[ "$(sha256sum ckpt/ckpt/wm_s.pt | cut -c1-64)" = 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 ] \
  || { echo "SETUP FAILED: wm_s.pt"; exit 1; }
cd code_split/code_v12r7b
PYTHONPATH=src:.:$A/a75ship /root/miniconda3/bin/python -B -c "import torch, a75ss, a75_wmforms, wm_fix_crn; \
from tools.wzprop import wm_backend as W; assert hasattr(W.ParallelWM, '_split_map'); \
print('SETUP_OK torch', torch.__version__, 'cuda', torch.cuda.is_available())"
