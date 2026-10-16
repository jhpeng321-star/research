# C4 v5: cloud Ubuntu 22.04 / 48-core box operations

All commands below run on the indicated Linux host. The launchers are Bash,
use `set -euo pipefail`, and need no laptop connection after `nohup setsid`.
They use 46 SUMO workers and single-threaded BLAS per worker. No launcher retries
collection or training processes: only completed, synchronous transfers are retried.

## 1. Integrator: publish an immutable runtime code archive to the relay

First integrate `c4/boxops` and `c4/popfix` into the intended **runtime** commit.
The collector must expose `--population`, propagate it to its environment, and
`CavSumoEnv.__init__` must accept `population` for textbook MAPPO fine-tuning.
No population is silently dropped. The GENRAND adapter forwards it through
`scripts/genrand/collect_genrand.sh`, even when that driver has no population CLI.
The paired phase uses an empty plain-seed argument: the 480 random episodes
have already been collected separately.

From that integrated checkout, on the relay (the GPU host), or a Linux integrator host
with its own relay access:

```bash
cd /path/to/integrated/research-dense
export CODE_ID="$(git rev-parse HEAD)"
export ARCHIVE_ID="c4-code-$CODE_ID"
mkdir -p scripts/boxops/dist
python3 -B scripts/boxops/ops.py bundle "$PWD" "scripts/boxops/dist/$ARCHIVE_ID.tar.gz"

# Omit SSH/rsync only if you generated the archive directly in the relay directory.
ssh -p 22 user@relay.example.com 'mkdir -p /workspace/c4/relay'
rsync -a --partial -e 'ssh -p 22' \
  "scripts/boxops/dist/$ARCHIVE_ID.tar.gz" \
  "scripts/boxops/dist/$ARCHIVE_ID.tar.gz.sha256" \
  user@relay.example.com:/workspace/c4/relay/
```

`bundle` uses `git archive HEAD` with a permitted tracked-runtime file list.
It excludes restricted paths before archiving; it never opens them. It includes
code, configurations, the preserved network, scenario templates, and baseline
assets, without historical runs/data. SHA256 binds the archive to its sidecar;
the git-archive PAX commit is retained in `BOXOPS_CODE_ID` after extraction.
Archive names are immutable: use a new name for a new commit. Keep the bootstrap
files below from the same commit.

## 2. Fresh box: bootstrap relay access (one interactive session)

Paste these commands in the cloud box's provider console. This bootstrap must
precede `setup_box.sh`: a genuinely fresh box may lack rsync. Use `sudo` only
when logged in as a non-root user; passwordless sudo is required by setup.

```bash
sudo apt-get update
sudo apt-get install -y python3 rsync openssh-client util-linux ca-certificates
mkdir -p ~/c4/bootstrap ~/.ssh
chmod 700 ~/.ssh
test -f ~/.ssh/relay_key || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/relay_key -C c4-box-relay
chmod 600 ~/.ssh/relay_key
cat ~/.ssh/relay_key.pub
```

On a root-only box, omit `sudo` in the two apt commands. Authorize **only the
printed public key** in the relay user's `~/.ssh/authorized_keys` through your
existing relay session; do not copy a private key between machines. Check the
relay host-key fingerprint against the provider/relay console on first contact.
The scripts use `accept-new` (existing host-key mismatches still fail).

Copy `common.sh`, `ops.py`, and `fetch_code.sh` from the integrated commit to
`~/c4/bootstrap/` using the provider console or the integrator's SSH connection
to the box. These small files are the only laptop bootstrap; large code/data
transfers are box-to-relay. Then, on the box:

```bash
sed -i 's/\r$//' ~/c4/bootstrap/*.sh
chmod +x ~/c4/bootstrap/*.sh
export ARCHIVE_ID=c4-code-REPLACE_WITH_INTEGRATED_COMMIT
bash ~/c4/bootstrap/fetch_code.sh "$ARCHIVE_ID"
bash ~/c4/repo/scripts/boxops/setup_box.sh
test -f ~/c4/SETUP_DONE
tail -n 20 ~/setup.log
```

Setup is idempotent and locked. It installs Ubuntu packages including venv/pip,
rsync, and all listed SUMO GUI libraries (including `libxrender1`), then creates
`~/c4env`. Torch is explicitly pinned to **2.14.1** from the PyTorch CPU index;
the remaining pins are in `requirements-box.txt`. An unavailable pinned wheel
is a hard install failure; obtain the approved wheel/pin before proceeding,
rather than silently substituting a version. Transitive installed versions are
saved in `~/c4/INSTALLED_FREEZE.txt`; `pip check` must pass. Setup obtains
`SUMO_HOME` from `sumo.SUMO_HOME`, checks `sumo --version`, then runs the native
collector for **30 simulated seconds** (no warmup), confirming one complete
TraCI episode before writing `~/c4/SETUP_DONE`. All output appends to
`~/setup.log`. Repeated setup preserves prior markers as `SETUP_DONE.previous-*`.

`fetch_code.sh` verifies SHA before inspecting/unpacking tar members, rejects
links/traversal/restricted names, extracts into a new staging directory, strips
shell-script CRLF, and installs atomically at `~/c4/repo`. Re-fetching the same
SHA preserves the repo. A different SHA refuses to overwrite it: fetch into a
fresh path via `REPO=$HOME/c4/repo-$CODE_ID`, then use the same `REPO` override
for all launchers. Never swap code underneath an active run.

## 3. Collection: 480 native + 480 random + 36 paired base episodes

On the box:

```bash
cd ~/c4/repo
export RUN_ID=v5-box01-20261014
export POPULATION=textbook
# Registered training block 961000-961999; the three defaults are disjoint.
export NATIVE_SEEDS=961000-961479
export GENRAND_SEEDS=961480-961959
export BRANCH_SEEDS=961960-961995
export PENETRATIONS=0.05,0.1,0.2,0.4
export PEN_WEIGHTS=1,1,2,2
export DEMANDS=2400,2700
export WORKERS=46
nohup setsid bash scripts/boxops/collect_v5.sh "$RUN_ID" \
  >"$HOME/c4/collect-$RUN_ID.launch.log" 2>&1 < /dev/null &
echo "launcher PID=$!"
```

`weighted-cycle` makes **one episode per seed**, not a Cartesian product. The
defaults produce exactly 480/480/36 base episodes. Paired branches use 900 s
horizon, 300 s warmup, 60 s cadence/horizon, four random arms plus a native arm;
the driver's determinism repeat check remains enabled. Each phase converts
to its own `c4_episodes` source. Branch conversion includes base episodes and
branch arms; the adapter selects only complete episode roots and excludes
preserved `.incomplete-*` ancestors before invoking the converter. Do not
substitute unregistered seeds; the repo guard remains in
force. For another box choose disjoint **registered** ranges, or explicitly
document intentional repeats in separate RUN_IDs.

Local output: `~/c4/data_v5/$RUN_ID`; relay output:
`/workspace/c4/data_v5/$RUN_ID/`. Every run contains `CODE_ID.txt`
(git commit, flags, each start/end UTC and exit status), immutable
`RUN_SPEC.json`, root `COLLECT_SUMMARY.json`, phase summaries, and `logs/`.
Successful phases have `*.DONE`; conversions have `*.CONVERTED` (destination
path). Interrupted conversions are preserved and retried into a fresh
`c4_episodes.attempt-*` directory. Transfer uses `rsync --partial` with bounded
retries and no deletion. `PUSHED` is written only after the data upload and
sent to the relay as the last transfer. A new invocation archives the old
marker and uploads updated metadata/results again.

To resume collection, rerun the **same exports and nohup command**. A per-run
`flock` prevents duplicates. The collector skips complete episodes, and phase
markers skip completed phases/conversions. Code/flag/input-hash drift refuses
resume instead of mixing datasets. A failed upload reruns only transfer work.

## 4. MAPPO fine-tune, spot stop, and resume

Place the approved starting checkpoint on the relay (integrator session):

```bash
rsync -a --partial -e 'ssh -p 22' /path/to/approved/policy.pt \
  user@relay.example.com:/workspace/c4/relay/mappo-init.pt
```

On the box:

```bash
cd ~/c4/repo
mkdir -p ~/c4/inputs
rsync -a --partial --timeout=120 \
  -e "ssh -p 22 -i $HOME/.ssh/relay_key -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  user@relay.example.com:/workspace/c4/relay/mappo-init.pt ~/c4/inputs/
export RUN_ID=v5-mappo-box01-20261014
export POPULATION=textbook
export INIT_FROM=$HOME/c4/inputs/mappo-init.pt
export SPOT_NOTICE_FILE=$HOME/c4/mappo_v5/$RUN_ID/SPOT_NOTICE
# Have your provider's termination hook touch SPOT_NOTICE_FILE.
# Optional: export SPOT_METADATA_URL to the provider's documented local endpoint.
# AWS example (if IMDSv1 is enabled):
# export SPOT_METADATA_URL=http://169.254.169.254/latest/meta-data/spot/instance-action
# Optional scheduled stop: export SPOT_STOP_AT_UTC=2026-10-15T02:00:00Z
nohup setsid bash scripts/boxops/mappo_ft_v5.sh "$RUN_ID" \
  >"$HOME/c4/train-$RUN_ID.launch.log" 2>&1 < /dev/null &
echo "launcher PID=$!"
```

Launcher uses `ulimit -n 65536`, 46 workers/episodes per iteration, warmup 300 s,
horizon 600 s, 44 update threads, and a 100,000,000 SUMO-second budget. A runtime
config copy preserves `[env].kwargs` and adds `population` when supported.
Default `REQUIRE_POPULATION=1` refuses missing support; an explicit
`REQUIRE_POPULATION=0` permits a legacy exploratory run and records this choice.
Fresh output requires `INIT_FROM`, with acceleration re-decoded to 2.6. With
existing checkpoints the launcher passes `--resume` and omits `--init-from`.
Keep the same original `INIT_FROM` path/file when resuming: it is part of the
immutable run specification.

The watcher polls every 5 s. A local notice, scheduled stop, or optional cloud
metadata notice writes `STOP_REQUESTED`; TERM/INT/HUP to the **launcher PID**
also requests graceful stopping. Metadata polling is opt-in because provider
endpoints/authentication differ; configure a provider hook or endpoint before
using spot capacity. The trainer checkpoints at its next safe boundary and
returns **3** for a requested stop. The launcher waits for it, terminates only
its own watcher PID, then uploads checkpoints, summaries and logs to
`data_v5/$RUN_ID/mappo/` (worker scenarios excluded). It never restarts training
in a retry loop. Allow time for an in-flight episode and checkpoint upload;
the watcher cannot guarantee recovery after an immediate hard power loss.

For a manual stop:

```bash
touch "$HOME/c4/mappo_v5/$RUN_ID/STOP_REQUESTED"
# Wait for launcher exit and checkpoint upload before restarting.
```

Once the termination condition is cleared, archive the request and notice,
then rerun the same exports and launch command:

```bash
OUT=$HOME/c4/mappo_v5/$RUN_ID
test ! -f "$OUT/STOP_REQUESTED" || mv "$OUT/STOP_REQUESTED" "$OUT/STOP_REQUESTED.previous-$(date -u +%s)"
test ! -f "$SPOT_NOTICE_FILE" || mv "$SPOT_NOTICE_FILE" "$SPOT_NOTICE_FILE.previous-$(date -u +%s)"
unset SPOT_STOP_AT_UTC
nohup setsid bash scripts/boxops/mappo_ft_v5.sh "$RUN_ID" \
  >"$HOME/c4/train-$RUN_ID.resume.log" 2>&1 < /dev/null &
```

For a replacement box, repeat bootstrap/setup with the **same code archive**
and copy the checkpointed run back from the relay before launching:

```bash
mkdir -p "$HOME/c4/mappo_v5/$RUN_ID"
rsync -a --partial --timeout=120 \
  -e "ssh -p 22 -i $HOME/.ssh/relay_key -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "user@relay.example.com:/workspace/c4/data_v5/$RUN_ID/mappo/" \
  "$HOME/c4/mappo_v5/$RUN_ID/"
# Restore mappo-init.pt at the same INIT_FROM path; clear old notices as above.
```

The same restore pattern works for collection, using the run root rather than
its `mappo/` subdirectory. Full checkpoint corruption is handled by the
trainer's SHA-checked checkpoint loader; do not modify its `LATEST` pointer.

## 5. Evaluation (summaries only)

On the box after checkpoints are available:

```bash
cd ~/c4/repo
export RUN_ID=v5-eval-box01-20261014
export POPULATION=textbook
export CONTROLLERS="native,damping,pi,maxaccel_lc513,$HOME/c4/mappo_v5/v5-mappo-box01-20261014/policy_00010.pt"
export DEMANDS=1900,2100,2400,2700
export PENETRATIONS=0.05,0.1,0.15,0.2,0.3,0.4,0.6
export SEEDS=4107-4148
nohup setsid bash scripts/boxops/eval_v5.sh "$RUN_ID" \
  >"$HOME/c4/eval-$RUN_ID.launch.log" 2>&1 < /dev/null &
echo "launcher PID=$!"
```

Replace the checkpoint path with an actual exported actor; controller entries
are comma-separated and cannot contain commas. Evaluation uses the full
controller × demand × penetration × seed product and 300/600 s warmup/horizon.
Controller directories have numeric IDs to avoid checkpoint-basename collisions.
Rerun with identical settings to resume. Local raw episodes stay under
`~/c4/eval_v5/$RUN_ID`. Relay `data_v5/$RUN_ID/eval/` receives a strict whitelist:
episode `summary.json`, `COLLECT_SUMMARY*`, `INDEX.json`, `CODE_ID.txt`,
`RUN_SPEC.json`, and done markers, followed by `PUSHED`. **No `episode.npz`
or checkpoint payload is uploaded by evaluation.**

## Checks, help, and overrides

```bash
for script in scripts/boxops/*.sh; do bash -n "$script" || exit; echo "bash -n PASS: $script"; done
python3 -B scripts/boxops/ops.py --help
python3 -B scripts/boxops/test_ops.py       # isolated stdlib tests, Python >=3.11
bash scripts/boxops/test_launchers.sh      # fake collectors/relay only; no network or SUMO
```

Every shell file supports `--help` (common.sh describes sourcing). Optional
overrides include `REPO`, `VENV`, `PYTHON`, `RUN_ROOT`, `RELAY_HOST`,
`RELAY_PORT`, `RELAY_DIR`, `RELAY_DATA`, `RELAY_KEY`, `TRANSFER_RETRIES` (8),
and `RETRY_DELAY` (15 s). Use distinct RUN_IDs across campaign modes/boxes.
Resume refuses different code or scientific parameters; worker-count changes
also require a new RUN_ID. No automatic remote setup or launching is performed
by the integrator archive command.
