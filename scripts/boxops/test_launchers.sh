#!/usr/bin/env bash
# Local fixtures only: never SUMO, apt, ssh, rsync, or cloud metadata.
set -euo pipefail
usage() { echo 'Usage: test_launchers.sh [--help] (Linux stdlib Python + Bash fixture checks; no network/SUMO)'; }
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# == 0 ]] || { usage >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TMP=$(mktemp -d "$HERE/.fixture-XXXXXX")
cleanup() {
    local rc=$? log
    if ((rc != 0)); then
        [[ ! -f $TMP/events ]] || cat "$TMP/events" >&2
        for log in "$HOME"/c4/*/*/logs/*.log; do
            [[ ! -f $log ]] || { echo "fixture log: $log" >&2; cat "$log" >&2; }
        done
    fi
    rm -rf -- "$TMP"
}
trap cleanup EXIT
export HOME=$TMP/home REPO=$TMP/repo VENV=$TMP/venv
export PYTHONDONTWRITEBYTECODE=1 FAKE_EVENTS=$TMP/events
export REAL_PYTHON
REAL_PYTHON=$(command -v python3)
mkdir -p "$HOME/c4" "$REPO/traffic_shepherd/c4" "$REPO/src/wms/control/mappo" \
    "$REPO/src/wms/core" "$REPO/src/wms/data" "$REPO/configs" "$REPO/scripts/genrand" \
    "$TMP/bin" "$VENV/bin" "$TMP/sumo/bin"
cp "$ROOT/scripts/genrand/collect_genrand.sh" "$REPO/scripts/genrand/collect_genrand.sh"
"$REAL_PYTHON" - "$REPO/scripts/genrand/collect_genrand.sh" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
data = p.read_bytes()
p.write_bytes(data.replace(b'\r\n', b'\n').replace(b'\r', b'\n'))
PY
cp "$ROOT/configs/c4_mappo.toml" "$REPO/configs/c4_mappo.toml"
cp "$ROOT/src/wms/core/toml.py" "$ROOT/src/wms/core/toml_subset.py" "$REPO/src/wms/core/"
printf '%040d\n' 1 >"$REPO/BOXOPS_CODE_ID"
printf 'fake-initial-weights\n' >"$TMP/init.pt"
export INIT_FROM=$TMP/init.pt POPULATION=textbook RETRY_DELAY=0 TRANSFER_RETRIES=3
export WORKERS=2 NATIVE_SEEDS=961000-961001 GENRAND_SEEDS=961480-961481 BRANCH_SEEDS=961960
export SPOT_METADATA_URL='' SPOT_STOP_AT_UTC='' SPOT_NOTICE_FILE=$TMP/notice
export PATH="$TMP/bin:$PATH"
cat >"$VENV/bin/python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == -c ]]; then printf '%s\n' "$FAKE_SUMO_HOME"; exit 0; fi
exec "$REAL_PYTHON" "$@"
SH
export FAKE_SUMO_HOME=$TMP/sumo
cat >"$TMP/sumo/bin/sumo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
echo 'SUMO fixture only'; exit 0
SH
cat >"$TMP/bin/ssh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'ssh mocked\n' >>"$FAKE_EVENTS"
SH
cat >"$TMP/bin/rsync" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'rsync %s\n' "$*" >>"$FAKE_EVENTS"
if [[ -f $FAKE_FAIL_RSYNC ]]; then mv "$FAKE_FAIL_RSYNC" "$FAKE_FAIL_RSYNC.used"; exit 12; fi
SH
export FAKE_FAIL_RSYNC=$TMP/fail-rsync FAKE_FAIL_CONVERSION=$TMP/fail-conversion
chmod +x "$VENV/bin/python" "$TMP/bin/ssh" "$TMP/bin/rsync" "$TMP/sumo/bin/sumo"
cat >"$REPO/traffic_shepherd/c4/env.py" <<'PY'
class CavSumoEnv:
    def __init__(self, *, population='fixture'):
        raise AssertionError('must never instantiate a simulation env')
PY
cat >"$REPO/traffic_shepherd/c4/collect.py" <<'PY'
import argparse, json, os
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--population', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--policy', required=True)
a, other = ap.parse_known_args()
assert a.population == 'textbook'
with open(os.environ['FAKE_EVENTS'], 'a') as log:
    log.write('collect ' + a.policy + ' ' + ' '.join(other) + '\n')
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out/'COLLECT_SUMMARY.json').write_text(json.dumps({'episodes_complete': 1, 'episodes_failed': 0}))
(out/'INDEX.json').write_text(json.dumps({'already_complete': []}))
ep = out/'episodes'/'fixture'; ep.mkdir(parents=True)
(ep/'summary.json').write_text('{"fixture": true}')
(ep/'manifest.json').write_text('{"status": "complete"}')
(ep/'episode.npz').write_bytes(b'fixture, not a SUMO episode')
PY
cat >"$REPO/traffic_shepherd/c4/genrand_driver.py" <<'PY'
import argparse, os, subprocess, sys
from pathlib import Path
def collect_cmd(args, seeds, out, workers, branch):
    return [sys.executable, '-m', 'traffic_shepherd.c4.collect', '--policy', 'generic_random',
            '--seeds', seeds, '--out', str(out)]
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--branch-seeds', required=True)
    ap.add_argument('--plain-seeds', default=''); ap.add_argument('--estimate-only', action='store_true')
    a, rest = ap.parse_known_args(argv)
    assert not a.plain_seeds
    if a.estimate_only: return 0
    assert '--no-convert' in rest
    with open(os.environ['FAKE_EVENTS'], 'a') as log: log.write('paired\n')
    return subprocess.call(collect_cmd(a, a.branch_seeds, Path(a.out)/'branch', 1, True))
PY
cat >"$REPO/src/wms/data/c4_episodes.py" <<'PY'
import argparse, os
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--episodes', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
assert not any(out.iterdir()), 'conversion must get fresh destination'
(out/'fixture.txt').write_text('converted')
with open(os.environ['FAKE_EVENTS'], 'a') as log: log.write('convert ' + a.episodes + '\n')
fail=Path(os.environ['FAKE_FAIL_CONVERSION'])
if fail.exists() and 'generic_random' in a.episodes:
    fail.rename(str(fail)+'.used'); raise SystemExit(1)
PY
cat >"$REPO/src/wms/control/mappo/train.py" <<'PY'
import argparse, json, os, time
from pathlib import Path
from wms.core.toml import loads
ap=argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--init-from'); ap.add_argument('--resume', action='store_true')
a, rest=ap.parse_known_args(); out=Path(a.out)
assert loads(Path(a.config).read_text())['env']['kwargs']['population'] == 'textbook'
assert (a.resume and not a.init_from) or (not a.resume and a.init_from)
with open(os.environ['FAKE_EVENTS'], 'a') as log: log.write('train ' + ('resume' if a.resume else 'init') + '\n')
check=out/'checkpoints'; check.mkdir(exist_ok=True); (check/'ckpt_00001.pt').write_bytes(b'fixture')
if Path(os.environ['SPOT_NOTICE_FILE']).exists():
    for _ in range(100):
        if (out/'STOP_REQUESTED').exists(): break
        time.sleep(.05)
    else: raise AssertionError('watcher did not request stop')
rc=3 if (out/'STOP_REQUESTED').exists() else 0
(out/'RUN_SUMMARY.json').write_text(json.dumps({'status': 'stopped_by_request' if rc else 'completed'}))
raise SystemExit(rc)
PY
# Local ops adapter must be directly executable when used as PY by genrand.
# /mnt Windows mounts often expose execute bits; copy a fixture-owned bridge anyway.
cp "$HERE/python_bridge.sh" "$HERE/ops.py" "$HERE/common.sh" "$HERE/collect_v5.sh" \
    "$HERE/mappo_ft_v5.sh" "$HERE/eval_v5.sh" "$TMP/"
chmod +x "$TMP/python_bridge.sh"
touch "$FAKE_FAIL_CONVERSION" "$FAKE_FAIL_RSYNC"
if bash "$TMP/collect_v5.sh" fixture-collect; then echo 'expected interrupted conversion' >&2; exit 1; fi
bash "$TMP/collect_v5.sh" fixture-collect
test -f "$HOME/c4/data_v5/fixture-collect/PUSHED"
test -f "$HOME/c4/data_v5/fixture-collect/generic_random.CONVERTED"
grep -q 'c4_episodes.attempt-' "$HOME/c4/data_v5/fixture-collect/generic_random.CONVERTED"
bash "$TMP/collect_v5.sh" fixture-collect
test "$(grep -c '^collect ' "$FAKE_EVENTS")" -eq 3
test "$(grep -c '^paired$' "$FAKE_EVENTS")" -eq 1
test "$(grep -c '^convert ' "$FAKE_EVENTS")" -eq 4
touch "$SPOT_NOTICE_FILE"
export SPOT_POLL_SECONDS=0.05
rc=0; bash "$TMP/mappo_ft_v5.sh" fixture-train || rc=$?
test "$rc" -eq 3
test -f "$HOME/c4/mappo_v5/fixture-train/STOP_REQUESTED"
test -f "$HOME/c4/mappo_v5/fixture-train/PUSHED"
mv "$SPOT_NOTICE_FILE" "$SPOT_NOTICE_FILE.used"
mv "$HOME/c4/mappo_v5/fixture-train/STOP_REQUESTED" "$TMP/stop.previous"
bash "$TMP/mappo_ft_v5.sh" fixture-train
test -f "$HOME/c4/mappo_v5/fixture-train/TRAIN_DONE"
test "$(grep -c '^train init$' "$FAKE_EVENTS")" -eq 1
test "$(grep -c '^train resume$' "$FAKE_EVENTS")" -eq 1
export CONTROLLERS=native,maxaccel_lc513 SEEDS=4107
bash "$TMP/eval_v5.sh" fixture-eval
bash "$TMP/eval_v5.sh" fixture-eval
test "$(grep -c '^collect ' "$FAKE_EVENTS")" -eq 5
grep 'rsync .*fixture-eval/' "$FAKE_EVENTS" | grep -q -- "--exclude=\*"
grep '^collect maxaccel_lc513 ' "$FAKE_EVENTS" | grep -q -- '--demands 1900,2100,2400,2700'
grep '^collect maxaccel_lc513 ' "$FAKE_EVENTS" | grep -q -- '--penetrations 0.05,0.1,0.15,0.2,0.3,0.4,0.6'
# Lock contention fails without launching another worker.
OUT=$HOME/c4/data_v5/fixture-collect
exec 7>"$OUT/.boxops.lock"
flock -n 7
if bash "$TMP/collect_v5.sh" fixture-collect; then echo 'duplicate lock was accepted' >&2; exit 1; fi
flock -u 7
echo 'PASS: launcher fixtures (conversion recovery, transfer retry, phase resume, population forwarding, spot stop, MAPPO resume, eval whitelist, lock contention)'
