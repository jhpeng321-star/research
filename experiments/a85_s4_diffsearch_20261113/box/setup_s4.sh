#!/bin/bash
# A85 S4 box setup (CPU only, never touches the GPU: CUDA_VISIBLE_DEVICES=''; nice/ionice; never deletes).
# Expects /root/autodl-tmp/a85/{a75_search_box.tgz,a85_s4_pkg.tar} (scp'd from the laptop).
set -u
B=/root/autodl-tmp/a85
cd $B || { echo "SETUP FAILED: $B missing"; exit 1; }
[ "$(sha256sum a75_search_box.tgz | cut -c1-64)" = 9eb32080ea789219202f61f1819944870bb28c428f4d417762ca310631a63277 ] \
  || { echo "SETUP FAILED: bundle sha"; exit 1; }
[ -e a75 ] && { echo "SETUP FAILED: $B/a75 exists"; exit 1; }
nice -n 19 ionice -c3 tar xzf a75_search_box.tgz -C $B
( cd a75 && sha256sum -c --quiet MANIFEST.sha256 ) || { echo "SETUP FAILED: bundle manifest"; exit 1; }
[ "$(sha256sum a75/ckpt/ckpt/wm_s.pt | cut -c1-64)" = 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 ] \
  || { echo "SETUP FAILED: wm_s.pt"; exit 1; }
sha256sum -c --quiet MANIFEST_A85.sha256 || { echo "SETUP FAILED: a85 manifest"; exit 1; }
for f in a85ship/a75ss.py a85ship/a75_wmforms.py a85ship/wm_fix_crn.py; do
  cmp -s $f a75/a75ship/$(basename $f) || { echo "SETUP FAILED: $f differs from the bundle"; exit 1; }; done
cd a75/code_split/code_v12r7b
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.:$B/a85ship nice -n 19 /root/miniconda3/bin/python -B \
  $B/check_s4_gen0.py /root/autodl-tmp/a75/out_s2b/STRATEGY_SEARCH_D2800_p60.json $B/cfg/A85S4_SEARCH_CFG.json \
  || { echo "SETUP FAILED: gen0 check"; exit 1; }
echo SETUP_OK
