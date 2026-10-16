# C4 MAPPO baseline launcher for a Windows spot instance (48 cores).
# Linux equivalent: scripts/c4/run_mappo.sh.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\c4\run_mappo.ps1 -Mode check
#   ... -Mode smoke                     # CPU fake env (no SUMO), foreground
#   ... -Mode sumo-smoke                # real SUMO: 2 workers, 2 iterations, warmup 300 s, horizon 60 s, foreground
#   ... -Mode run [-Workers 46]         # DETACHED training run (survives the SSH session), new run dir
#   ... -Mode run -RunId <id> -Resume   # continue that run in place (trainer resumes the newest valid checkpoint)
#   ... -Mode status -RunId <id>        # STATE, pid, last iteration row
#   ... -Mode stop -RunId <id>          # graceful stop (STOP_REQUESTED), then THIS run's own pid tree after 120 s
#
# Detached runs (Windows OpenSSH kills the session's process tree on disconnect) are started by the
# first launcher in -Launcher that works (default "cim,task,startprocess"; a launcher counts as
# working when the wrapper writes <run>\STATE within 30 s): cim = Win32_Process.Create (outside the
# session's job object; Defender ASR "block process creations from WMI" blocks it - observed on the
# laptop), task = one-shot scheduled task as the current user (S4U logon, no stored password, no time
# limit; the wrapper unregisters it at the end), startprocess = last resort (may die with SSH).  The detached wrapper (-Mode _job) re-applies the environment from
# <run>\job.json (a CIM process does not inherit this session's variables), runs Python with
# stdout/stderr logs (one pair per session), and writes STATE = running | done | stopped_spot |
# stopped | failed:<rc>.  No retry loops.
#
# Spot reclaim (spot instances can be reclaimed at any time, ~2 min notice): the wrapper
# polls the termination notice every 5 s (SPOT_NOTICE_URL, default
# http://metadata.tencentyun.com/latest/meta-data/spot/termination-time; 404 = none) and on a
# notice writes <run>\out\STOP_REQUESTED; the trainer checks it between episodes, while waiting
# for workers and between PPO minibatches, abandons the in-flight iteration, stops its workers and
# exits with code 3.  Every finished iteration is already checkpointed (ckpt_XXXXX.pt + LATEST
# pointer + status.json), so at most one iteration is lost.  On a new instance copy the run dir
# back and use -Mode run -RunId <id> -Resume.  SPOT_WATCH=0 disables the watcher;
# SPOT_NOTICE_FILE=<path> simulates a notice when that file exists (drill).
#
# Env (optional): PY (python.exe), C4_REPO (repo root; default = two levels above this script),
# C4_RUNS (default <repo>\runs\active\c4_mappo), CONFIG (default configs\c4_mappo.toml),
# MAX_SUMO_SECONDS, UPDATE_THREADS (default = workers), EXTRA_ARGS (extra CLI arguments, space separated),
# SUMO_HOME (sumo.exe must be on PATH or in %SUMO_HOME%\bin).
[CmdletBinding()]
param(
    [ValidateSet("check", "smoke", "sumo-smoke", "run", "status", "stop", "_job")]
    [string]$Mode = "run",
    [string]$RunId = "",
    [int]$Workers = 0,
    [switch]$Resume,
    [switch]$Foreground,
    [string]$Launcher = "cim,task,startprocess",
    [string]$RunDir = "",
    [string]$TaskName = ""
)
$ErrorActionPreference = "Stop"

function Log([string]$msg) { Write-Host ("[{0}] {1}" -f (Get-Date).ToUniversalTime().ToString("HH:mm:ssZ"), $msg) }
function Die([string]$msg) { Write-Host "REFUSED: $msg" -ForegroundColor Red; exit 2 }
function EnvOr([string]$name, [string]$default) {
    $v = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($v)) { return $default } else { return $v }
}

# ------------------------------------------------------------------ spot termination notice
# Cloud spot metadata: GET <metadata>/spot/termination-time -> 404 = no notice, 200 + time = reclaim
# scheduled (about 2 minutes ahead).  SPOT_NOTICE_FILE (a local file that exists) simulates a notice.
function Get-SpotNotice([string]$url, [string]$file) {
    if ($file -and (Test-Path -LiteralPath $file)) { return "file:" + (Get-Content -Raw -LiteralPath $file) }
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200 -and "$($r.Content)".Trim()) { return "$($r.Content)".Trim() }
        return $null
    } catch {
        $resp = $_.Exception.Response
        if ($resp -and [int]$resp.StatusCode -eq 404) { return $null }
        throw
    }
}

# ------------------------------------------------------------------ detached job wrapper
if ($Mode -eq "_job") {
    $job = Get-Content -Raw -LiteralPath (Join-Path $RunDir "job.json") | ConvertFrom-Json
    foreach ($p in $job.env.PSObject.Properties) { [Environment]::SetEnvironmentVariable($p.Name, [string]$p.Value, "Process") }
    $session = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $watchLog = Join-Path $RunDir "spot_watch.log"
    Set-Content -LiteralPath (Join-Path $RunDir "STATE") -Value "running" -Encoding ascii
    $proc = Start-Process -FilePath $job.python -ArgumentList $job.args -WorkingDirectory $job.repo `
        -RedirectStandardOutput (Join-Path $RunDir "stdout_$session.log") `
        -RedirectStandardError (Join-Path $RunDir "stderr_$session.log") -WindowStyle Hidden -PassThru
    $null = $proc.Handle
    Set-Content -LiteralPath (Join-Path $RunDir "pid") -Value $proc.Id -Encoding ascii
    # Spot watcher: poll every 5 s while the trainer runs; on a notice create <out>\STOP_REQUESTED,
    # which the trainer checks between episodes and PPO minibatches (exit code 3 after a clean stop).
    $noticed = $false
    $unreachable = $false
    while (-not $proc.WaitForExit(5000)) {
        if (-not $job.spot_watch -or $noticed) { continue }
        try {
            $notice = Get-SpotNotice $job.spot_url $job.spot_notice_file
        } catch {
            if (-not $unreachable) {
                Add-Content -LiteralPath $watchLog -Value ("{0} metadata unreachable ({1}); still polling" -f (Get-Date).ToUniversalTime().ToString("o"), $_.Exception.Message)
                $unreachable = $true
            }
            continue
        }
        if ($notice) {
            $noticed = $true
            Set-Content -LiteralPath (Join-Path $job.out_dir "STOP_REQUESTED") -Value ("spot termination notice: {0}" -f $notice) -Encoding ascii
            Add-Content -LiteralPath $watchLog -Value ("{0} NOTICE {1}: STOP_REQUESTED written" -f (Get-Date).ToUniversalTime().ToString("o"), $notice)
        }
    }
    $rc = $proc.ExitCode
    $state = if ($rc -eq 0) { "done" } elseif ($rc -eq 3) { if ($noticed) { "stopped_spot" } else { "stopped" } } else { "failed:$rc" }
    Set-Content -LiteralPath (Join-Path $RunDir "STATE") -Value $state -Encoding ascii
    if ($TaskName) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue }
    exit $rc
}

# ------------------------------------------------------------------ settings
$Repo = (Resolve-Path (EnvOr "C4_REPO" (Join-Path $PSScriptRoot "..\.."))).Path
$Runs = EnvOr "C4_RUNS" (Join-Path $Repo "runs\active\c4_mappo")
$Py = EnvOr "PY" "python"
$Config = EnvOr "CONFIG" "configs\c4_mappo.toml"
if ($Workers -le 0) { $Workers = [Math]::Max(1, [Environment]::ProcessorCount - 2) }
$UpdateThreads = [int](EnvOr "UPDATE_THREADS" "$Workers")
if (-not (Test-Path -LiteralPath (Join-Path $Repo $Config))) { Die "config $Config not found under $Repo" }
$pyCmd = Get-Command $Py -ErrorAction SilentlyContinue
if (-not $pyCmd) { Die "python '$Py' not found (set PY to the rl_driving python.exe)" }
$Py = $pyCmd.Source
$envVars = [ordered]@{
    PYTHONPATH        = "$(Join-Path $Repo 'src');$Repo"
    PYTHONUTF8        = "1"
    PYTHONUNBUFFERED  = "1"
    OMP_NUM_THREADS   = "$UpdateThreads"
    MKL_NUM_THREADS   = "$UpdateThreads"
    KMP_DUPLICATE_LIB_OK = "TRUE"
    PATH              = $env:PATH
}
$sumoHome = EnvOr "SUMO_HOME" ""
if ($sumoHome) { $envVars["SUMO_HOME"] = $sumoHome; $envVars["PATH"] = "$(Join-Path $sumoHome 'bin');$($env:PATH)" }
foreach ($k in $envVars.Keys) { [Environment]::SetEnvironmentVariable($k, $envVars[$k], "Process") }

function RunDirFor([string]$name) {
    if ($RunId) { return (Join-Path $Runs $RunId) }
    return (Join-Path $Runs ("{0}_{1}" -f $name, (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")))
}

function Quote([string]$a) { if ($a -match '[\s"]') { return '"' + ($a -replace '"', '\"') + '"' } else { return $a } }

function TrainArgs([string]$outDir) {
    $a = @("-m", "wms.control.mappo.train", "--config", $Config, "--out", $outDir, "--update-threads", "$UpdateThreads")
    $mss = EnvOr "MAX_SUMO_SECONDS" ""
    if ($mss) { $a += @("--max-sumo-seconds", $mss) }
    return $a
}

function Foreground([string]$runDir, [string[]]$arguments) {
    Set-Content -LiteralPath (Join-Path $runDir "COMMAND.txt") -Value ("$Py " + (($arguments | ForEach-Object { Quote $_ }) -join " "))
    Log "run dir: $runDir"
    $proc = Start-Process -FilePath $Py -ArgumentList (($arguments | ForEach-Object { Quote $_ }) -join " ") `
        -WorkingDirectory $Repo -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $runDir "stdout.log") -RedirectStandardError (Join-Path $runDir "stderr.log")
    $null = $proc.Handle
    $proc.WaitForExit()
    $rc = $proc.ExitCode
    Set-Content -LiteralPath (Join-Path $runDir "STATE") -Value $(if ($rc -eq 0) { "done" } else { "failed:$rc" }) -Encoding ascii
    Get-Content -LiteralPath (Join-Path $runDir "stdout.log") -Tail 5
    if ($rc -ne 0) { Get-Content -LiteralPath (Join-Path $runDir "stderr.log") -Tail 30; Die "$Mode failed (rc $rc)" }
    Log "$Mode ok: $runDir"
}

function NewRunDir([string]$name) {
    $dir = RunDirFor $name
    if (Test-Path -LiteralPath $dir) {
        if (-not $Resume) { Die "$dir exists; it is never overwritten. Use -Resume to continue it in place, or a new -RunId." }
        $state = (Get-Content -LiteralPath (Join-Path $dir "STATE") -ErrorAction SilentlyContinue)
        $pidFile = Join-Path $dir "pid"
        if ($state -eq "running" -and (Test-Path $pidFile)) {
            $old = [int](Get-Content -LiteralPath $pidFile)
            if (Get-Process -Id $old -ErrorAction SilentlyContinue) { Die "$dir is still running (pid $old)" }
        }
        Log "resuming $dir (previous STATE=$state)"
    } else {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    return (Resolve-Path -LiteralPath $dir).Path
}

switch ($Mode) {
    "check" {
        & $Py -c "import wms.control.mappo.train as m, wms.control.mappo.config_helpers as h; h.config_from_dict(m.load_config(r'$Config')); import traffic_shepherd.c4; print('import+config ok')"
        if ($LASTEXITCODE -ne 0) { Die "check failed" }
        if (-not (Get-Command sumo -ErrorAction SilentlyContinue)) { Die "sumo.exe not on PATH (set SUMO_HOME)" }
        Log "check ok (python $Py, workers $Workers, update threads $UpdateThreads)"
    }
    "smoke" {
        $dir = NewRunDir "mappo_smoke"
        Foreground $dir ((TrainArgs (Join-Path $dir "out")) + @("--workers", "1", "--smoke"))
    }
    "sumo-smoke" {
        $dir = NewRunDir "mappo_sumo_smoke"
        Foreground $dir ((TrainArgs (Join-Path $dir "out")) + @("--workers", "2", "--iterations", "2",
            "--episodes-per-iteration", "2", "--warmup-s", "300", "--horizon-s", "60", "--set", "minibatch_frames=32"))
    }
    "run" {
        $dir = NewRunDir "mappo"
        $arguments = (TrainArgs (Join-Path $dir "out")) + @("--workers", "$Workers")
        if ($Resume -and (Test-Path -LiteralPath (Join-Path $dir "out\checkpoints"))) { $arguments += @("--resume") }
        $extra = EnvOr "EXTRA_ARGS" ""
        if ($extra) { $arguments += ($extra -split '\s+' | Where-Object { $_ }) }
        $argLine = ($arguments | ForEach-Object { Quote $_ }) -join " "
        Set-Content -LiteralPath (Join-Path $dir "COMMAND.txt") -Value "$Py $argLine"
        if ($Foreground) { Foreground $dir $arguments; break }
        $outDir = Join-Path $dir "out"
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        $job = [ordered]@{ python = $Py; args = $argLine; repo = $Repo; env = $envVars; workers = $Workers;
                           out_dir = $outDir; spot_watch = ((EnvOr "SPOT_WATCH" "1") -ne "0");
                           spot_url = (EnvOr "SPOT_NOTICE_URL" "http://metadata.tencentyun.com/latest/meta-data/spot/termination-time");
                           spot_notice_file = (EnvOr "SPOT_NOTICE_FILE" "");
                           started_utc = (Get-Date).ToUniversalTime().ToString("o") }
        $job | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $dir "job.json") -Encoding utf8
        $ps = (Get-Command powershell.exe).Source
        $wrapperArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`" -Mode _job -RunDir `"$dir`""
        $stateFile = Join-Path $dir "STATE"
        if (Test-Path -LiteralPath $stateFile) { Remove-Item -LiteralPath $stateFile }   # our own marker; the new wrapper rewrites it
        $launched = $null
        foreach ($method in ($Launcher -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
            try {
                switch ($method) {
                    "cim" {   # outside the SSH session's job object; may be blocked by a WMI-child ASR rule
                        $res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
                            -Arguments @{ CommandLine = "`"$ps`" $wrapperArgs"; CurrentDirectory = $Repo }
                        if ($res.ReturnValue -ne 0) { throw "Win32_Process.Create returned $($res.ReturnValue)" }
                        Set-Content -LiteralPath (Join-Path $dir "launcher.pid") -Value $res.ProcessId -Encoding ascii
                    }
                    "task" {  # one-shot scheduled task, current user, S4U (no stored password), no time limit
                        $name = "c4_mappo_" + (Split-Path $dir -Leaf)
                        $action = New-ScheduledTaskAction -Execute $ps -Argument "$wrapperArgs -TaskName $name" -WorkingDirectory $Repo
                        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U
                        $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
                            -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
                        Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings -Force | Out-Null
                        Start-ScheduledTask -TaskName $name
                        Set-Content -LiteralPath (Join-Path $dir "launcher.task") -Value $name -Encoding ascii
                    }
                    "startprocess" {   # last resort: may be killed when an SSH session closes
                        $p = Start-Process -FilePath $ps -ArgumentList $wrapperArgs -WorkingDirectory $Repo -WindowStyle Hidden -PassThru
                        Set-Content -LiteralPath (Join-Path $dir "launcher.pid") -Value $p.Id -Encoding ascii
                        Log "WARNING: Start-Process launcher; over SSH this run may die with the session"
                    }
                    default { throw "unknown launcher '$method'" }
                }
            } catch {
                Log "launcher $method failed: $($_.Exception.Message)"
                continue
            }
            $deadline = (Get-Date).AddSeconds(30)
            while ((Get-Date) -lt $deadline -and -not (Test-Path -LiteralPath $stateFile)) { Start-Sleep -Milliseconds 500 }
            if (Test-Path -LiteralPath $stateFile) { $launched = $method; break }
            Log "launcher $method started nothing within 30 s (blocked?); trying the next one"
            if ($method -eq "task") { Unregister-ScheduledTask -TaskName ("c4_mappo_" + (Split-Path $dir -Leaf)) -Confirm:$false -ErrorAction SilentlyContinue }
        }
        if (-not $launched) { Die "no launcher could start the detached wrapper ($Launcher)" }
        Log "launched detached MAPPO run via $launched, $Workers workers"
        Log "follow: Get-Content -Wait '$dir\out\log.jsonl'   state: Get-Content '$dir\STATE'   status: -Mode status -RunId $(Split-Path $dir -Leaf)"
    }
    "status" {
        if (-not $RunId) { Die "-RunId is required" }
        $dir = Join-Path $Runs $RunId
        $state = Get-Content -LiteralPath (Join-Path $dir "STATE") -ErrorAction SilentlyContinue
        $procId = Get-Content -LiteralPath (Join-Path $dir "pid") -ErrorAction SilentlyContinue
        $alive = if ($procId) { [bool](Get-Process -Id ([int]$procId) -ErrorAction SilentlyContinue) } else { $false }
        Log "STATE=$state pid=$procId alive=$alive"
        $log = Join-Path $dir "out\log.jsonl"
        if (Test-Path -LiteralPath $log) {
            $last = Get-Content -LiteralPath $log -Tail 400 | Where-Object { $_ -match '"type": "iter"' } | Select-Object -Last 1
            if ($last) {
                $row = $last | ConvertFrom-Json
                Log ("iteration {0}: charged SUMO-s {1:N0}, mean return {2:N3}, iteration {3:N0} s, collisions {4}, failures {5}" -f `
                    $row.iteration, $row.sumo_seconds_used, $row.mean_episode_return, $row.iteration_wall_s, $row.collisions, $row.failed_episodes)
            }
        }
        $status = Join-Path $dir "out\status.json"
        if (Test-Path -LiteralPath $status) { Get-Content -Raw -LiteralPath $status }
        $err = Get-ChildItem -LiteralPath $dir -Filter "stderr*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
        if ($err) { Get-Content -LiteralPath $err.FullName -Tail 5 }
        $watch = Join-Path $dir "spot_watch.log"
        if (Test-Path -LiteralPath $watch) { Get-Content -LiteralPath $watch -Tail 3 }
    }
    "stop" {
        if (-not $RunId) { Die "-RunId is required" }
        $dir = Join-Path $Runs $RunId
        $procId = Get-Content -LiteralPath (Join-Path $dir "pid") -ErrorAction SilentlyContinue
        if (-not $procId) { Die "no pid file in $dir" }
        $proc = Get-Process -Id ([int]$procId) -ErrorAction SilentlyContinue
        if (-not $proc) { Log "pid $procId is not running"; break }
        # Graceful: the trainer abandons the in-flight iteration and exits (code 3) within seconds.
        Set-Content -LiteralPath (Join-Path $dir "out\STOP_REQUESTED") -Value "manual stop" -Encoding ascii
        if (-not $proc.WaitForExit(120000)) {
            # Own process tree only (trainer + its spawn workers + their SUMO children).
            & taskkill.exe /PID ([int]$procId) /T /F
            Log "graceful stop timed out; killed pid $procId (tree)"
        }
        Log "stopped pid $procId; resume later with -Mode run -RunId $RunId -Resume"
    }
}
