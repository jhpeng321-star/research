#!/bin/bash
# A86 S5 box setup on gpuR (CPU only, never touches the GPU: CUDA_VISIBLE_DEVICES=''; nice/ionice; never deletes).
# Expects /root/autodl-tmp/a86/a86_s5_pkg.tar (scp'd from the laptop) unpacked in /root/autodl-tmp/a86. Copies the
# A75 bundle /root/autodl-tmp/a75_search_box.tgz (read only) into a86/ and unpacks its own tree a86/a75.
set -u
B=/root/autodl-tmp/a86
BUNDLE_SHA=9eb32080ea789219202f61f1819944870bb28c428f4d417762ca310631a63277
S5_GEN0_SHA=275c80a9e53ffe65126f25aa69246bcc6ef9f5302413081b00c6c76800bda364
cd $B || { echo "SETUP FAILED: $B missing"; exit 1; }
[ -e SETUP_OK ] && { echo "SETUP FAILED: SETUP_OK already exists"; exit 1; }
sha256sum -c --quiet MANIFEST_A86.sha256 || { echo "SETUP FAILED: a86 manifest"; exit 1; }
[ -e a75 ] && { echo "SETUP FAILED: $B/a75 exists"; exit 1; }
[ -e a75_search_box.tgz ] || cp /root/autodl-tmp/a75_search_box.tgz $B/a75_search_box.tgz
[ "$(sha256sum a75_search_box.tgz | cut -c1-64)" = $BUNDLE_SHA ] || { echo "SETUP FAILED: bundle sha"; exit 1; }
nice -n 19 ionice -c3 tar xzf a75_search_box.tgz -C $B 2>&1 | grep -v 'in the future' || true
( cd a75 && sha256sum -c --quiet MANIFEST.sha256 ) || { echo "SETUP FAILED: bundle manifest"; exit 1; }
[ "$(sha256sum a75/ckpt/ckpt/wm_s.pt | cut -c1-64)" = 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 ] \
  || { echo "SETUP FAILED: wm_s.pt"; exit 1; }
for f in ship/a75ss.py ship/a75_wmforms.py ship/wm_fix_crn.py; do
  cmp -s $f a75/a75ship/$(basename $f) || { echo "SETUP FAILED: $f differs from the bundle"; exit 1; }; done
cd a75/code_split/code_v12r7b
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.:$B/ship nice -n 19 /root/miniconda3/bin/python -B $B/check_s5_gen0.py \
  $B/ref/S2B_GEN0_KEYS.json $B/cfg/S2B_REF/A75S2B_SEARCH_CFG.json $B/cfg/S4_REF/A85S4_SEARCH_CFG.json \
  $B/cfg/A86S5_SEARCH_CFG.json $S5_GEN0_SHA || { echo "SETUP FAILED: gen0 check"; exit 1; }
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.:$B/ship nice -n 19 /root/miniconda3/bin/python -B -c \
  "import a86s5, a75_wmforms, wm_fix_crn; from tools.wzprop import wm_backend; print('IMPORT_OK')" \
  || { echo "SETUP FAILED: import"; exit 1; }
date > $B/SETUP_OK
echo SETUP_OK
