#!/usr/bin/env bash
# CLEAN5C4 situational admission re-run (privileged teacher bins) for new WMVERIFY verifications (r3 p20).
# Read-only on the hosts; zero SUMO. Usage (from the repo root, local Windows/Git Bash):
#   ops/c4sit_rerun.sh <host> <host_wmv_root> <local_analysis_dir> <act_file> <bank_file> <tag>
# e.g. ops/c4sit_rerun.sh cpuZ /home/ubuntu/wmverify_r3/wmv_r3 $R/wmverify_r3/ANALYSIS_R3 \
#        $R/wmverify_r3/ANALYSIS_R3/act_wmv_r3.jsonl $R/wmverify_r3/ANALYSIS_R3/BANK_R3.jsonl r3
# Extra ARMS files with 'structured' arms: STRUCTURED="a.json b.json" (env). Writes
# clean5c/c4sit/GUIDANCE_C4SIT_priv_<tag>.json (or .NO_ADMISSION.json); never overwrites.
set -euo pipefail
host=$1; wmv=$2; analysis=$3; act=$4; bank=$5; tag=$6
R=${R:-E:/research-dense/runs/active/C4-V7FT-20261017}
PY=${PY:-C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe}
sha=$(sha256sum tools/coopmwm/fullpol_situational.py | cut -c1-12)
tmp=$(mktemp -d); mkdir -p "$tmp/tools/coopmwm"
cp tools/coopmwm/fullpol_situational.py "$tmp/tools/coopmwm/"; touch "$tmp/tools/__init__.py" "$tmp/tools/coopmwm/__init__.py"
cp ops/sit_reduce.py "$tmp/"; tar -C "$tmp" -czf "$tmp.tgz" .
scp -q -o BatchMode=yes "$tmp.tgz" "$host:/tmp/sr_$sha.tgz"
ssh -o BatchMode=yes "$host" "D=/home/ubuntu/sitreduce_$sha; mkdir -p \$D && tar -C \$D -xzf /tmp/sr_$sha.tgz && cd \$D && \
  nice -n 19 python3 sit_reduce.py \$D privileged $wmv > PONSET_$tag.jsonl && wc -l PONSET_$tag.jsonl"
scp -q -o BatchMode=yes "$host:/home/ubuntu/sitreduce_$sha/PONSET_$tag.jsonl" clean5c/c4sit/onset_priv/
O=clean5c/c4sit/onset_priv
"$PY" -m tools.coopmwm.fullpol_situational --mode privileged --allowed-scope p20 \
  --act "$R/wmverify_mc/ANALYSIS_MC/act_wmv_mc.jsonl" "$R/wmverify_p20/ANALYSIS_P20/act_wmv_p20.jsonl" \
        "$R/wmverify_rep/ANALYSIS_REP/act_wmv_rep.jsonl.gz" "$act" \
  --onset $O/PONSET_wmv_mc.jsonl $O/PONSET_wmv_p20.jsonl $O/PONSET_wmv_rep.jsonl "$O/PONSET_$tag.jsonl" \
  --analysis "$R/wmverify_mc/ANALYSIS_MC" "$R/wmverify_p20/ANALYSIS_P20" "$R/wmverify_rep/ANALYSIS_REP" "$analysis" \
  --bank "$R/wmverify_mc/ANALYSIS_MC/BANK_MC.jsonl" "$R/wmverify_p20/ANALYSIS_P20/BANK_P20.jsonl" "$bank" \
  --structured "$R/wmc_strategy_v3/ARMS_D3200_p60.json" "$R/wmc_strategy_v3/ARMS_D3200_p40_search.json" \
               "$R/wmc_strategy_v3/ARMS_D2800_D2400.json" ${STRUCTURED:-} \
  --provenance "CLEAN5C4 situational admission v3 (privileged teacher bins, ALLOWED p20) + $tag" \
  --out "clean5c/c4sit/GUIDANCE_C4SIT_priv_$tag.json"
