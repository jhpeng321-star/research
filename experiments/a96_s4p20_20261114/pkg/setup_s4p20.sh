#!/bin/bash
# A96 S4-p20 box setup (modelled on a85/setup_s4.sh b8543e77). CPU only (CUDA_VISIBLE_DEVICES=''); nice/ionice;
# never deletes; reads a85/ and a75p20/ only. Expects /root/autodl-tmp/a96_s4p20/{a75_search_box.tgz,a96_s4p20_pkg.tar}
# already extracted (a85ship/, cfg/, *.sh, check_s4p20_gen0.py, MANIFEST_A96.sha256).
set -u
B=/root/autodl-tmp/a96_s4p20
cd $B || { echo "SETUP FAILED: $B missing"; exit 1; }
[ "$(sha256sum a75_search_box.tgz | cut -c1-64)" = 9eb32080ea789219202f61f1819944870bb28c428f4d417762ca310631a63277 ] \
  || { echo "SETUP FAILED: bundle sha"; exit 1; }
[ -e a75 ] && { echo "SETUP FAILED: $B/a75 exists"; exit 1; }
nice -n 19 ionice -c3 tar xzf a75_search_box.tgz -C $B 2>/dev/null
( cd a75 && sha256sum -c --quiet MANIFEST.sha256 ) || { echo "SETUP FAILED: bundle manifest"; exit 1; }
[ "$(sha256sum a75/ckpt/ckpt/wm_s.pt | cut -c1-64)" = 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 ] \
  || { echo "SETUP FAILED: wm_s.pt"; exit 1; }
sha256sum -c --quiet MANIFEST_A96.sha256 || { echo "SETUP FAILED: a96 manifest"; exit 1; }
for f in a75ss.py a75_wmforms.py wm_fix_crn.py; do
  cmp -s a85ship/$f a75/a75ship/$f || { echo "SETUP FAILED: $f differs from the bundle"; exit 1; }; done
for f in a75ss.py a75_wmforms.py wm_fix_crn.py a85s4.py; do
  cmp -s a85ship/$f /root/autodl-tmp/a85/a85ship/$f || { echo "SETUP FAILED: $f differs from a85"; exit 1; }; done
cmp -s cfg/A96S4P20_SEARCH_CFG.json /root/autodl-tmp/a85/cfg/A85S4_SEARCH_CFG.json || { echo "SETUP FAILED: cfg != S4"; exit 1; }
cd a75/code_split/code_v12r7b
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.:$B/a85ship nice -n 19 /root/miniconda3/bin/python -B \
  $B/check_s4p20_gen0.py /root/autodl-tmp/a75p20/out_s3b/STRATEGY_SEARCH_D2800_p20.json $B/cfg/A96S4P20_SEARCH_CFG.json \
  /root/autodl-tmp/a75p20/p20/regime_STATES_p20.json D2800_p20 /root/autodl-tmp/a85/cfg/A85S4_SEARCH_CFG.json \
  || { echo "SETUP FAILED: gen0 check"; exit 1; }
echo SETUP_OK
