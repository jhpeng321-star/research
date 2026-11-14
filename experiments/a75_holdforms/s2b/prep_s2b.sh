#!/bin/bash
# A79 prep: build reduced cfgs + MPS launchers + chain script. Touches no running process.
set -eu
A=/root/autodl-tmp/a75; P=/root/autodl-tmp/a75p20
for f in $A/cfg/A75S2B_SEARCH_CFG.json $P/cfg/A75S3B_SEARCH_CFG_p20.json $A/run_search_s2b.sh $P/run_search_s3b.sh $A/chain_s2b.sh; do
  [ -e "$f" ] && { echo "REFUSE: $f exists"; exit 3; }; done
/root/miniconda3/bin/python - <<'PY'
import json
for src, dst in (('/root/autodl-tmp/a75/cfg/A75B_SEARCH_CFG.json', '/root/autodl-tmp/a75/cfg/A75S2B_SEARCH_CFG.json'),
                 ('/root/autodl-tmp/a75p20/cfg/A75S3_SEARCH_CFG_p20.json', '/root/autodl-tmp/a75p20/cfg/A75S3B_SEARCH_CFG_p20.json')):
    d = json.load(open(src)); o = json.loads(json.dumps(d))
    assert d['generations'] == 8 and d['finalists'] == 36
    d['generations'] = 4; d['finalists'] = 18
    json.dump(d, open(dst, 'w'), indent=1)
    n = json.load(open(dst))
    print(dst, 'seed', n['seed'], 'diff keys', sorted(k for k in n if n[k] != o[k]), 'same keyset', set(n) == set(o))
PY
sed -e 's/--workers 10/--workers 14/' -e 's/^# A75 hold-forms WM search launcher/# A79 S2b (MPS, 14 workers) copy of: A75 hold-forms WM search launcher/' $A/run_search_a75.sh > $A/run_search_s2b.sh
sed -e 's/--workers 10/--workers 14/' -e 's/^# A75-S3 (new hold forms/# A79 S3b (MPS, 14 workers) copy of: A75-S3 (new hold forms/' $P/run_search_s3.sh > $P/run_search_s3b.sh
cat > $A/chain_s2b.sh <<'CH'
#!/bin/bash
# A79 chain: S2b (p60) then S3b (p20), each once, under the MPS daemon started beforehand. Stops if S2b fails.
set -u
A=/root/autodl-tmp/a75; P=/root/autodl-tmp/a75p20
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_a75_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_a75_log
exec >> $A/chain_s2b.log 2>&1
echo "chain pid $$ S2b start $(date)"
bash $A/run_search_s2b.sh $A/cfg/A75S2B_SEARCH_CFG.json $A/out_s2b; rc=$?
echo "S2b rc $rc $(date)"
if [ "$rc" -ne 0 ]; then echo "CHAIN_STOPPED after S2b failure (no retry)"; echo QUEUE_DONE; exit 1; fi
echo "S3b start $(date)"
bash $P/run_search_s3b.sh $P/cfg/A75S3B_SEARCH_CFG_p20.json $P/out_s3b
echo "S3b rc $? $(date)"
echo QUEUE_DONE
CH
chmod +x $A/run_search_s2b.sh $P/run_search_s3b.sh $A/chain_s2b.sh
diff $A/run_search_a75.sh $A/run_search_s2b.sh || true; diff $P/run_search_s3.sh $P/run_search_s3b.sh || true
sha256sum $A/cfg/A75S2B_SEARCH_CFG.json $P/cfg/A75S3B_SEARCH_CFG_p20.json $A/run_search_s2b.sh $P/run_search_s3b.sh $A/chain_s2b.sh
