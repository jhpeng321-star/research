#Requires -Version 5.1
<#
.SYNOPSIS
  MORNING_CLEANUP.ps1 - owner-run clean-up / switch-over helper.   Generated @@GENERATED@@ by gen_cleanup.py.
  DO NOT EDIT BY HAND: re-run  python -I E:\research-dense\tools\dataops\cleanup\gen_cleanup.py  to regenerate it
  from the Drive-backup receipts (data_ops\receipts).  Output folder: E:\research-dense\data_ops\cleanup_out.

.DESCRIPTION
  REPORT-ONLY BY DEFAULT.  Nothing is deleted or copied unless -Execute is given.
  Every action (also report-only decisions) is appended to  E:\research-dense\data_ops\cleanup_out\journal.log.

  -Stage 1   Remove local items whose Drive copy is verified (see "Eligibility").  Also: ALL finished git worktrees
             (candidates = every E:\rd-wt\<name> with an ok wtclean receipt; the "protected" list is empty).  A worktree is
             removed only after: branch + untracked-outputs + tracked-dirty-state checks; every junction/symlink inside
             it is unlinked first (link only, never followed, target logged and verified to still exist, dependents
             before the worktrees they point into); then  git worktree remove --force.  Sensitive-named untracked
             files must already be in sealed\ (MOVE_TO_SEALED.ps1) - except for worktrees whose own root is
             sensitive-named (j4bfix), whose files are moved to sealed\relocated_20261010\ by this script (opaque
             move, SHA-256 checked) just before the removal.  E:\rd-wt\reorg3 (it holds the upload lists/manifests this
             script reads) is removed LAST, only with -RetireReorg and only if no other worktree remains.
  -Stage 2   Copy each keep-local hot-data item into E:\research-dense\data_cache\<name> with per-file SHA-256
             verification (copy, never move).  With -RemoveOld (+ -Execute, typed confirmation) the OLD copy is
             removed only if the cache copy re-verifies AND a Drive copy verifies AND no active code references it.
  -Stage 3   Obsolete (the single-work-tree switch was completed on 2026-10-10): prints a notice only.

  Stage 1 eligibility (ALL must hold; the first failing check is the reason printed):
    * the item has a passing Drive receipt AND the Drive copy is re-checked LIVE with rclone (size + MD5 at the
      CURRENT Drive path) against the same member manifest the local files are compared with;
    * local files == manifest members (no extra file, same size, not modified after the backup; with -Deep or
      -Execute also MD5 re-hashed);
    * no junction/symlink inside the item or among its ancestors; no protected name (sealed/CAL/HOLDOUT/chunk-5/
      INC1) inside; not in the keep-local set; not referenced by active code/config (rg literal scan);
    * inside the git repo: no tracked file;  worktrees: branch exists, no uncommitted tracked change, every
      untracked/ignored file is on Drive with equal MD5, no sensitive-named file left behind.
  Literal paths only (generated from receipts); deletion uses a no-follow .NET walker; nothing is wildcard-matched;
  Drive objects are never deleted.

.PARAMETER Stage          1, 2 or 3 (default 1).
.PARAMETER Execute        actually act (default: report only).  Stage 1/2 prompt for a typed confirmation unless -Yes.
.PARAMETER Deep           re-hash local files (MD5) in report mode (always on with -Execute unless -NoDeep).
.PARAMETER Only           restrict to item ids / paths / worktree names containing any of these substrings.
.PARAMETER Skip           skip items whose id/path/name contains any of these substrings.
.PARAMETER SkipWorktrees  Stage 1: do not evaluate worktrees.   -SkipItems: only worktrees.
.PARAMETER AllowSensitiveDup  Stage 1: worktrees whose only unbacked files are sensitive-NAMED code/protocol files
                          (e.g. tools\holdout_guard.py) are eligible when the same relative path exists in
                          E:\research-dense with identical size (metadata only, contents are never opened).
.PARAMETER ReleaseProtectedNames  Stage 1: items containing protected-NAME entries (e.g. protocols\chunk5_test_seal.json inside a mirrored bundle)
                          are normally held; with this flag they are eligible only if EVERY such entry is verified on Drive too (same
                          size+MD5 rule as every other file). Never applies to the item path itself.
.PARAMETER RemoveAroundProtected  Stage 1: for items that contain protected-name entries (sealed/CAL/HOLDOUT/chunk-5/INC1) that were never uploaded,
                          remove only the verified files and leave the protected entries (and their parent folders) untouched.
.PARAMETER AllowProtectedWorktrees  Stage 1: also consider the worktrees named in the generator's protected list (empty since 2026-10-10:
                          the owner decided that ALL worktrees retire; the mechanism is kept).
.PARAMETER IncludeBackupOnly  Stage 1: include receipts of class KEEP-ACTIVE-backup-only (label sidecars).
.PARAMETER ScanTrees      reference scan trees: main (default; the single work tree) | clean | both.
.PARAMETER RetireReorg    Stage 1: allow removing E:\rd-wt\reorg3 (upload lists / manifests), LAST and only if no other worktree remains.
.PARAMETER AllowNestedGit Stage 1: plain directories (not git worktrees) that contain a nested .git repository (e.g. bundle3/rented_bundle)
                          are normally held because the nested history is not backed up; this releases them.
.PARAMETER RelocateSensitive  Stage 1: a worktree that still holds sensitive-named UNTRACKED files after MOVE_TO_SEALED (e.g. names the move plan did not
                          cover, such as *HELDOUT*) is released by moving those files, unopened, into sealed\relocated_20261010\<drive>\...
                          (same rules as MOVE_TO_SEALED: SHA-256 before/after, no overwrite) right before its removal.  Nothing is deleted.
.PARAMETER ScanSensitiveNamedFiles  the reference scan skips sensitive-named files (e.g. tools\holdout_guard.py, tests\...\test_holdout_guard.py) by default so that
                          agents never open them; the owner adds this switch in the final run to include them.
.PARAMETER AssumeStage2bDone  REPORT-ONLY simulation: keep-local sources with old_policy 'remove' are treated as already removed by Stage 2 -RemoveOld (refused with -Execute).
.PARAMETER ExplainRefs    diagnostic: print every code reference / record citation that names this path, then exit.
.PARAMETER AssumeSealedMoved  REPORT-ONLY simulation: path of a SENSITIVE_MOVE_PLAN.csv; sensitive-named untracked files that the plan covers
                          are treated as already moved to sealed\ (refused together with -Execute).
.PARAMETER RemoveOld      Stage 2: remove old copies (separately confirmed).
.PARAMETER SelfTest       exercise the walker/copy/delete/junction-unlink/worktree-remove primitives on a scratch folder under the output folder (touches no research file).
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File E:\research-dense\data_ops\cleanup_out\MORNING_CLEANUP.ps1 -Stage 1
  powershell -NoProfile -ExecutionPolicy Bypass -File E:\research-dense\data_ops\cleanup_out\MORNING_CLEANUP.ps1 -Stage 1 -Execute
#>
[CmdletBinding()]
param(
    [ValidateSet(1, 2, 3)][int]$Stage = 1,
    [switch]$Execute,
    [switch]$Yes,
    [switch]$Deep,
    [switch]$NoDeep,
    [string[]]$Only = @(),
    [string[]]$Skip = @(),
    [switch]$SkipWorktrees,
    [switch]$SkipItems,
    [switch]$SkipReferenceScan,
    [switch]$NoTrimRecordRefs,
    [switch]$AllowSensitiveDup,
    [switch]$ReleaseProtectedNames,
    [switch]$RemoveAroundProtected,
    [switch]$AllowProtectedWorktrees,
    [switch]$IncludeBackupOnly,
    [ValidateSet('both', 'clean', 'main')][string]$ScanTrees = 'main',
    [string]$IgnoreReferenceFiles = '',
    [switch]$RetireReorg,
    [switch]$AllowNestedGit,
    [switch]$RelocateSensitive,
    [switch]$ScanSensitiveNamedFiles,
    [string]$ExplainRefs = '',
    [switch]$AssumeStage2bDone,
    [string]$AssumeSealedMoved = '',
    [switch]$RemoveOld,
    [int]$MarginGiB = 4,
    [switch]$ShowHolds,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# -File passes "a,b,c" as ONE string: split on commas
$Only = @($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Skip = @($Skip | ForEach-Object { $_ -split ',' } | Where-Object { $_ })

$ManifestJson = @'
@@MANIFEST_JSON@@
'@

# =====================================================================================================
#  C# helpers: no-follow walker, hashing, copy with inline hashes, no-follow delete (long-path safe)
# =====================================================================================================
$CSharp = @'
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;

public sealed class WalkResult {
    public List<string> Rel = new List<string>();
    public List<long> Size = new List<long>();
    public List<long> MTicks = new List<long>();
    public List<string> Reparse = new List<string>();
    public List<string> Sensitive = new List<string>();
    public List<string> Errors = new List<string>();
    public long Bytes;
    public int Dirs;
}

public static class MC {
    static string[] exact = new string[0];
    static string[] contains = new string[0];
    public static void SetProtected(string[] ex, string[] co) { exact = ex; contains = co; }

    public static string LP(string p) {
        if (p == null) return p;
        if (p.StartsWith(@"\\?\")) return p;
        if (p.StartsWith(@"\\")) return @"\\?\UNC\" + p.Substring(2);
        return @"\\?\" + p;
    }
    public static bool IsSensitiveName(string name) {
        string n = name.ToLowerInvariant();
        foreach (string e in exact) { if (n == e) return true; }
        foreach (string c in contains) { if (n.IndexOf(c, StringComparison.Ordinal) >= 0) return true; }
        return false;
    }
    static bool IsReparse(FileSystemInfo fi) { return (fi.Attributes & FileAttributes.ReparsePoint) != 0; }
    public static bool PathIsReparse(string p) {
        string l = LP(p);
        if (File.Exists(l)) return IsReparse(new FileInfo(l));
        if (Directory.Exists(l)) return IsReparse(new DirectoryInfo(l));
        return false;
    }

    // Walk a file or directory without following reparse points. Sensitive-named entries are listed, not descended.
    public static WalkResult Walk(string root, bool skipSensitive) {
        WalkResult r = new WalkResult();
        string rl = LP(root);
        if (File.Exists(rl)) {
            FileInfo f = new FileInfo(rl);
            if (IsReparse(f)) { r.Reparse.Add(f.Name); return r; }
            r.Rel.Add(f.Name); r.Size.Add(f.Length); r.MTicks.Add(f.LastWriteTimeUtc.Ticks); r.Bytes += f.Length;
            return r;
        }
        DirectoryInfo rd = new DirectoryInfo(rl);
        if (!rd.Exists) return r;
        if (IsReparse(rd)) { r.Reparse.Add("."); return r; }
        Stack<DirectoryInfo> dirs = new Stack<DirectoryInfo>();
        Stack<string> prefixes = new Stack<string>();
        dirs.Push(rd); prefixes.Push("");
        while (dirs.Count > 0) {
            DirectoryInfo cur = dirs.Pop();
            string pre = prefixes.Pop();
            r.Dirs++;
            List<FileSystemInfo> list = new List<FileSystemInfo>();
            try { list.AddRange(cur.EnumerateFileSystemInfos()); }
            catch (Exception ex) { r.Errors.Add(pre + ": " + ex.Message); continue; }
            foreach (FileSystemInfo fi in list) {
                string rel = pre.Length == 0 ? fi.Name : pre + "/" + fi.Name;
                try {
                    if (IsReparse(fi)) { r.Reparse.Add(rel); continue; }
                    if (skipSensitive && IsSensitiveName(fi.Name)) { r.Sensitive.Add(rel); continue; }
                    DirectoryInfo di = fi as DirectoryInfo;
                    if (di != null) { dirs.Push(di); prefixes.Push(rel); }
                    else {
                        FileInfo f = (FileInfo)fi;
                        r.Rel.Add(rel); r.Size.Add(f.Length); r.MTicks.Add(f.LastWriteTimeUtc.Ticks); r.Bytes += f.Length;
                    }
                } catch (Exception ex) { r.Errors.Add(rel + ": " + ex.Message); }
            }
        }
        return r;
    }

    static string Hex(byte[] b) { StringBuilder sb = new StringBuilder(b.Length * 2); foreach (byte x in b) sb.Append(x.ToString("x2")); return sb.ToString(); }

    public static string Md5File(string path) {
        using (FileStream fs = new FileStream(LP(path), FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 20, FileOptions.SequentialScan))
        using (MD5 h = MD5.Create()) { return Hex(h.ComputeHash(fs)); }
    }
    public static string Sha256File(string path) {
        using (FileStream fs = new FileStream(LP(path), FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 20, FileOptions.SequentialScan))
        using (SHA256 h = SHA256.Create()) { return Hex(h.ComputeHash(fs)); }
    }
    // Returns rel paths whose hash differs (max 25). root may be a file (rel ignored).
    public static string[] VerifyHashes(string root, string[] rels, string[] expected, bool sha256) {
        List<string> bad = new List<string>();
        bool isFile = File.Exists(LP(root));
        for (int i = 0; i < rels.Length; i++) {
            string p = isFile ? root : root + "\\" + rels[i].Replace('/', '\\');
            string got;
            try { got = sha256 ? Sha256File(p) : Md5File(p); } catch (Exception ex) { got = "ERR:" + ex.Message; }
            if (!string.Equals(got, expected[i], StringComparison.OrdinalIgnoreCase)) { bad.Add(rels[i]); if (bad.Count >= 25) break; }
        }
        return bad.ToArray();
    }
    // Copy src to dst (created/overwritten) hashing the stream once: returns {md5, sha256, bytes}
    public static string[] CopyHash(string src, string dst) {
        string dl = LP(dst);
        Directory.CreateDirectory(Path.GetDirectoryName(dl));
        using (FileStream fi = new FileStream(LP(src), FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 20, FileOptions.SequentialScan))
        using (FileStream fo = new FileStream(dl, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan))
        using (MD5 m = MD5.Create())
        using (SHA256 s = SHA256.Create()) {
            byte[] buf = new byte[8 << 20];
            long total = 0; int n;
            while ((n = fi.Read(buf, 0, buf.Length)) > 0) {
                m.TransformBlock(buf, 0, n, null, 0); s.TransformBlock(buf, 0, n, null, 0);
                fo.Write(buf, 0, n); total += n;
            }
            m.TransformFinalBlock(buf, 0, 0); s.TransformFinalBlock(buf, 0, 0);
            fo.Flush(true);
            return new string[] { Hex(m.Hash), Hex(s.Hash), total.ToString() };
        }
    }
    public static void MoveFile(string a, string b) {
        string bl = LP(b);
        if (File.Exists(bl)) File.Delete(bl);
        File.Move(LP(a), bl);
    }
    public static bool IsEmptyDir(string p) {
        string l = LP(p);
        if (!Directory.Exists(l)) return false;
        foreach (string e in Directory.EnumerateFileSystemEntries(l)) { return false; }
        return true;
    }
    public static void DeleteEmptyDir(string p) { Directory.Delete(LP(p), false); }
    public static bool Exists(string p) { string l = LP(p); return File.Exists(l) || Directory.Exists(l); }
    // Remove ONE reparse point (link only).  A directory junction/symlink is removed with RemoveDirectory (non-recursive:
    // nothing below the link is touched), a file symlink with DeleteFile.  Refuses anything that is not a reparse point.
    public static void UnlinkLink(string p) {
        string l = LP(p);
        FileAttributes a = File.GetAttributes(l);
        if ((a & FileAttributes.ReparsePoint) == 0) throw new InvalidOperationException("not a reparse point: " + p);
        if ((a & FileAttributes.Directory) != 0) Directory.Delete(l, false);
        else File.Delete(l);
    }

    // Delete exactly the listed files (relative to root), then remove directories that became empty.
    public static long DeleteListed(string root, string[] rels, List<string> errors) {
        long bytes = 0;
        HashSet<string> dirs = new HashSet<string>();
        bool isFile = File.Exists(LP(root));
        string rt = root.TrimEnd('\\');
        foreach (string rel in rels) {
            string full = isFile ? rt : rt + "\\" + rel.Replace('/', '\\');
            try {
                FileInfo f = new FileInfo(LP(full));
                if (IsReparse(f)) throw new InvalidOperationException("reparse point: " + full);
                long n = f.Length; f.Attributes = FileAttributes.Normal; f.Delete(); bytes += n;
                string d = Path.GetDirectoryName(full);
                while (d != null && d.Length > rt.Length) { dirs.Add(d); d = Path.GetDirectoryName(d); }
            } catch (Exception ex) { errors.Add(full + ": " + ex.Message); }
        }
        List<string> dl = new List<string>(dirs);
        dl.Sort(delegate(string a, string b) { return b.Length.CompareTo(a.Length); });
        foreach (string d in dl) {
            try { if (Directory.Exists(LP(d)) && IsEmptyDir(d)) Directory.Delete(LP(d), false); } catch (Exception) { }
        }
        return bytes;
    }

    // No-follow recursive delete. Throws if a reparse point is met (nothing below it is touched).
    public static long DeleteTree(string root, List<string> errors) {
        string rl = LP(root);
        if (File.Exists(rl)) {
            FileInfo f = new FileInfo(rl);
            if (IsReparse(f)) throw new InvalidOperationException("reparse point: " + root);
            long n = f.Length;
            try { f.Attributes = FileAttributes.Normal; f.Delete(); } catch (Exception ex) { errors.Add(root + ": " + ex.Message); return 0; }
            return n;
        }
        DirectoryInfo d = new DirectoryInfo(rl);
        if (!d.Exists) return 0;
        if (IsReparse(d)) throw new InvalidOperationException("reparse point: " + root);
        WalkResult pre = Walk(root, false);
        if (pre.Reparse.Count > 0) throw new InvalidOperationException("reparse point inside " + root + ": " + pre.Reparse[0]);
        return DeleteDir(d, errors);
    }
    static long DeleteDir(DirectoryInfo d, List<string> errors) {
        long bytes = 0;
        List<FileSystemInfo> list = new List<FileSystemInfo>(d.EnumerateFileSystemInfos());
        foreach (FileSystemInfo fi in list) {
            if (IsReparse(fi)) throw new InvalidOperationException("reparse point appeared: " + fi.FullName);
        }
        foreach (FileSystemInfo fi in list) {
            DirectoryInfo di = fi as DirectoryInfo;
            if (di != null) { bytes += DeleteDir(di, errors); continue; }
            FileInfo f = (FileInfo)fi;
            try { long n = f.Length; f.Attributes = FileAttributes.Normal; f.Delete(); bytes += n; }
            catch (Exception ex) { errors.Add(f.FullName + ": " + ex.Message); }
        }
        try { d.Attributes = FileAttributes.Directory; d.Delete(false); } catch (Exception ex) { errors.Add(d.FullName + ": " + ex.Message); }
        return bytes;
    }
}
'@
if (-not ('MC' -as [type])) { Add-Type -TypeDefinition $CSharp -Language CSharp }

# =====================================================================================================
#  setup
# =====================================================================================================
$M = $ManifestJson | ConvertFrom-Json
$P = $M.paths
$EXEC = [bool]$Execute
$DEEP = if ($Deep) { $true } elseif ($NoDeep) { $false } else { $EXEC }
$script:DeepNow = $DEEP
[MC]::SetProtected([string[]]@($M.protected.exact), [string[]]@($M.protected.contains))
$Rclone = [string]$P.rclone
$MainRepo = [string]$P.main
$CleanRepo = [string]$P.clean
$CacheRoot = [string]$P.cache
$OutDir = [string]$P.out
$JournalPath = [string]$P.journal
$RemovedMap = [string]$P.removed_map
$SealedRoot = [string]$P.sealed_root
$SealedManifestPath = [string]$P.sealed_manifest
$BroadSens = [regex]::new([string]$M.protected.broad_regex)   # MOVE_TO_SEALED's sensitive-name rule (case-insensitive)
if (($AssumeSealedMoved -or $AssumeStage2bDone) -and $EXEC) { throw '-AssumeSealedMoved / -AssumeStage2bDone are report-only simulations and cannot be combined with -Execute' }
$RunStamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$script:Res = New-Object System.Collections.ArrayList
$script:Freed = @{ C = [long]0; D = [long]0; E = [long]0; G = [long]0 }
$script:Errs = New-Object System.Collections.ArrayList
$script:PartialOk = $false
$script:S2 = $null
$script:NoReport = $false
$script:NoOffloadMap = $false

if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }
$script:JW = New-Object System.IO.StreamWriter($JournalPath, $true, (New-Object System.Text.UTF8Encoding($false)))
$script:JW.AutoFlush = $true

function Jrn([string]$Act, [string]$Target, $Bytes, [string]$Detail) {
    $mode = if ($EXEC) { 'EXEC' } else { 'REPORT' }
    $cols = @((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), ('S' + $Stage), $mode, $Act, $Target, [string]$Bytes, ($Detail -replace "[`r`n`t]+", ' '))
    $script:JW.WriteLine(($cols -join "`t"))
}
function Say([string]$Msg, [string]$Color = 'Gray') { Write-Host $Msg -ForegroundColor $Color }
function GB($n) { '{0:N2} GiB' -f ([double]$n / 1GB) }
function NP([string]$p) { $p = $p -replace '/', '\'; if ($p.Length -gt 3) { $p = $p.TrimEnd('\') }; return $p }
function K([string]$p) { return (NP $p).ToLowerInvariant() }
function UnderP([string]$p, [string]$root) {
    $a = K $p; $b = (K $root).TrimEnd('\')
    return ($a -eq $b) -or $a.StartsWith($b + '\')
}
function DriveOf([string]$p) { if ($p.Length -ge 1) { return $p.Substring(0, 1).ToUpperInvariant() } else { return '?' } }
function FreeBytes([string]$letter) { try { return ([System.IO.DriveInfo]($letter + ':\')).AvailableFreeSpace } catch { return [long]-1 } }
function Add-Res($Id, $Type, $Path, $Status, $Reason, $Bytes, $Files, $Drive) {
    $o = [pscustomobject]@{ Id = $Id; Type = $Type; Path = $Path; Drv = (DriveOf $Path); Status = $Status; Reason = $Reason; Bytes = [long]$Bytes; Files = [int]$Files; Drive = $Drive }
    [void]$script:Res.Add($o)
    Jrn $Status $Path $Bytes $Reason
    return $o
}

# ---------------- rclone / git process wrappers ----------------
function Invoke-Proc([string]$Exe, [string[]]$A, [int]$TimeoutSec = 1800, [string]$Cwd = '') {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $q = foreach ($a in $A) { if ($a -match '[\s"]' -or $a -eq '') { '"' + ($a -replace '"', '\"') + '"' } else { $a } }
    $psi.Arguments = ($q -join ' ')
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    if ($Cwd) { $psi.WorkingDirectory = $Cwd }
    $p = [System.Diagnostics.Process]::Start($psi)
    $ot = $p.StandardOutput.ReadToEndAsync()
    $et = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill() } catch { }
        return @{ rc = 124; out = ''; err = "timeout after $TimeoutSec s" }
    }
    return @{ rc = $p.ExitCode; out = $ot.Result; err = $et.Result }
}
function Rcl([string[]]$A, [int]$TimeoutSec = 1800) { return Invoke-Proc $Rclone $A $TimeoutSec }
function GitX([string]$Dir, [string[]]$A, [int]$TimeoutSec = 600) {
    return Invoke-Proc 'git' (@('-c', 'core.quotepath=false', '-C', $Dir) + $A) $TimeoutSec
}

# ---------------- remote listing cache (rclone lsf, MD5 from Drive metadata) ----------------
$script:RK = @{}      # key: path below gdrive:research-archive/ (lowercase) -> @{ size; md5 }
$script:RRoots = @{}  # list roots already loaded
$script:RFail = @{}
function Remote-Rel([string]$full) { return ($full -replace '^gdrive:research-archive/?', '').Trim('/') }
function List-RootFor([string]$rel) {
    $parts = $rel.Split('/')
    if ($parts[0] -eq 'worktrees') { return 'worktrees' }
    if ($parts.Count -ge 2 -and $parts[0] -eq 'data' -and $parts[1] -eq 'staging') { return 'data/staging' }
    if ($parts.Count -ge 2) { return ($parts[0] + '/' + $parts[1]) }
    return $parts[0]
}
function Load-Remote([string]$root, [string]$hash = 'md5') {
    $ck = $hash + '|' + $root
    if ($script:RRoots.ContainsKey($ck)) { return $true }
    $r = Rcl @('lsf', '-R', '--files-only', '--format', 'psh', '--hash', $hash, '--separator', '|', ('gdrive:research-archive/' + $root)) 3600
    if ($r.rc -ne 0) { $script:RFail[$ck] = ($r.err -replace '\s+', ' ').Trim(); return $false }
    $n = 0
    foreach ($line in ($r.out -split "`n")) {
        $line = $line.TrimEnd("`r")
        if (-not $line) { continue }
        $f = $line.Split('|')
        if ($f.Count -lt 3) { continue }
        $h = $f[$f.Count - 1]; $s = $f[$f.Count - 2]
        $path = ($f[0..($f.Count - 3)] -join '|')
        $key = ($root + '/' + $path).ToLowerInvariant()
        $ent = $script:RK[$key]
        if ($null -eq $ent) { $ent = @{ size = [long]$s } ; $script:RK[$key] = $ent }
        $ent[$hash] = $h.ToLowerInvariant()
        $n++
    }
    $script:RRoots[$ck] = $n
    return $true
}
function Get-Remote([string]$fullPath, [string]$hash = 'md5') {
    $rel = Remote-Rel $fullPath
    $root = List-RootFor $rel
    if (-not (Load-Remote $root $hash)) { return $null }
    return $script:RK[$rel.ToLowerInvariant()]
}
function Reset-Remote { $script:RK = @{}; $script:RRoots = @{}; $script:RFail = @{} }
function Remote-Err { if ($script:RFail.Count -gt 0) { return ' [rclone listing failed: ' + (@($script:RFail.Values)[0]) + ']' } return '' }

# ---------------- manifests ----------------
$script:ManCache = @{}
function Read-Manifest([string]$path, [string]$expectSha, [string]$fallbackTs) {
    $ck = (K $path)
    if ($script:ManCache.ContainsKey($ck)) { return $script:ManCache[$ck] }
    $res = @{ ok = $false; why = ''; map = @{}; ticks = [long]0; n = 0 }
    if (-not (Test-Path -LiteralPath $path)) { $res.why = "manifest missing: $path"; $script:ManCache[$ck] = $res; return $res }
    if ($expectSha) {
        $act = [MC]::Sha256File($path)
        if ($act -ne $expectSha) { $res.why = "manifest changed since generation (regenerate): $path"; $script:ManCache[$ck] = $res; return $res }
    }
    $gen = $null
    foreach ($line in [System.IO.File]::ReadLines($path)) {
        if ($line.StartsWith('#')) {
            if ($line -match '^#\s*generated=(.+)$') { try { $gen = [DateTimeOffset]::Parse($Matches[1].Trim()).UtcDateTime } catch { } }
            continue
        }
        if (-not $line.Trim()) { continue }
        $c = $line.TrimEnd("`r").Split("`t")
        if ($c.Count -ge 3) { $md5 = $c[0]; $sz = $c[1]; $rel = ($c[2..($c.Count - 1)] -join "`t") }
        else {
            $mf = [regex]::Match($line, '^([0-9a-fA-F]{32})\s+(\d+)\s+(.*)$')
            if (-not $mf.Success) { $res.why = "unparseable manifest line in $path"; $script:ManCache[$ck] = $res; return $res }
            $md5 = $mf.Groups[1].Value; $sz = $mf.Groups[2].Value; $rel = $mf.Groups[3].Value.Trim()
        }
        $res.map[$rel.Replace('\', '/')] = @{ md5 = $md5.ToLowerInvariant(); size = [long]$sz }
        $res.n++
    }
    if ($null -eq $gen -and $fallbackTs) { try { $gen = ([datetime]::Parse($fallbackTs)).ToUniversalTime() } catch { } }
    if ($null -ne $gen) { $res.ticks = $gen.Ticks }
    $res.ok = $true
    $script:ManCache[$ck] = $res
    return $res
}

# ---------------- protection / overlap / reference / process checks ----------------
$ActiveWt = @{}
foreach ($n in @($M.active_worktrees)) { $ActiveWt[(K ('E:\rd-wt\' + $n))] = ($n + ', excluded by wtclean') }
if (-not $AllowProtectedWorktrees) { foreach ($n in @($M.protected_worktrees)) { $ActiveWt[(K ('E:\rd-wt\' + $n))] = ($n + ', live/integration tree') } }
function Get-ProtectedReason([string]$path, [bool]$skipWt = $false) {
    $p = NP $path
    foreach ($c in ($p -split '\\')) { if ($c -and [MC]::IsSensitiveName($c)) { return "protected name component '$c' (sealed/CAL/HOLDOUT/chunk-5/INC1)" } }
    foreach ($r in @($M.protected.roots)) { if (UnderP $p $r) { return "protected root $r" } }
    $ok = $false
    foreach ($r in @($M.allowed_roots)) { if (UnderP $p $r) { $ok = $true; break } }
    if (-not $ok) { return 'outside the deletion allow-list' }
    foreach ($r in @($M.container_roots)) { if ((K $p) -eq (K $r)) { return "container root $r" } }
    if (-not $skipWt) { foreach ($k in $ActiveWt.Keys) { if (UnderP $p $k) { return ("protected worktree " + $ActiveWt[$k]) } } }
    foreach ($h in @($M.hold_until_flag)) {
        if ((UnderP $p ([string]$h.path)) -and -not (Get-Variable -Name ([string]$h.flag) -Scope Script -ValueOnly -ErrorAction SilentlyContinue)) {
            return ("held until -" + $h.flag + ": " + $h.why)
        }
    }
    return $null
}
function Get-KeepOverlap([string]$path) {
    foreach ($k in @($M.keep_local)) {
        $s = [string]$k.source
        if ($AssumeStage2bDone -and [string]$k.old_policy -eq 'remove') { continue }   # report-only simulation of "Stage 2 -RemoveOld -Execute done"
        if ((UnderP $path $s) -or (UnderP $s $path)) {
            if (Test-Path -LiteralPath $s) { return [string]$k.name }
        }
    }
    return $null
}
$script:ProcList = $null
function Find-ProcUsing([string]$path) {
    if ($null -eq $script:ProcList) {
        $script:ProcList = @()
        try { $script:ProcList = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | ForEach-Object { [pscustomobject]@{ Pid = $_.ProcessId; Name = $_.Name; Cmd = $_.CommandLine } }) } catch { }
    }
    $rx = [regex]::new([regex]::Escape((NP $path)) + '(?![A-Za-z0-9_\-])', 'IgnoreCase')
    $rx2 = [regex]::new([regex]::Escape((NP $path).Replace('\', '/')) + '(?![A-Za-z0-9_\-])', 'IgnoreCase')
    foreach ($p in $script:ProcList) {
        if ($p.Pid -eq $PID) { continue }
        if ($rx.IsMatch($p.Cmd) -or $rx2.IsMatch($p.Cmd)) { return ("pid {0} {1}" -f $p.Pid, $p.Name) }
    }
    return $null
}

$script:Lit = $null   # list of @{ lit; file; tree }
function Get-RgExe {
    $c = Get-Command rg -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in @('C:\Users\jpeng\.kimi-code\bin\rg.exe', 'C:\Program Files\ripgrep\rg.exe', 'C:\Users\jpeng\miniconda3\Library\bin\rg.exe')) { if (Test-Path -LiteralPath $p) { return $p } }
    return $null
}
function Scan-Literals {
    $script:Lit = New-Object System.Collections.ArrayList
    if ($SkipReferenceScan) { return }
    $rg = Get-RgExe
    # files whose literals do not count as "active code" (archival copies of the data-ops tooling)
    $IgnoreRef = if ($IgnoreReferenceFiles) { $IgnoreReferenceFiles } else { [string]$M.ref_scan.ignore_files }
    # sensitive-named files (holdout/heldout/cal_split/chunk5/chunk-5/j4b/inc1/sealed) are not opened by the scan unless the owner asks for it
    $sensGlobs = @()
    if (-not $ScanSensitiveNamedFiles) { foreach ($t0 in 'holdout', 'heldout', 'cal_split', 'chunk5', 'chunk-5', 'j4b', 'inc1', 'sealed') { $sensGlobs += @('--iglob', ('!*' + $t0 + '*')) } }
    $trees = @()
    if ($ScanTrees -in @('both', 'main')) { $trees += , @('main', $M.paths.main) }
    if ($ScanTrees -in @('both', 'clean') -and ((K $M.paths.clean) -ne (K $M.paths.main) -or $ScanTrees -eq 'clean')) { $trees += , @('clean', $M.paths.clean) }
    $re = '[A-Za-z]:[\\/][^\s"''<>|*?(){}\[\],;`]+'
    foreach ($t in $trees) {
        $tag = $t[0]; $root = $t[1]
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $dirs = @($M.ref_scan.dirs.PSObject.Properties | Where-Object { (K $_.Name) -eq (K $root) } | ForEach-Object { $_.Value })
        $targets = @()
        foreach ($d in $dirs) { $pp = Join-Path $root $d; if (Test-Path -LiteralPath $pp) { $targets += $pp } }
        foreach ($g in @($M.ref_scan.root_files)) {
            foreach ($f in (Get-ChildItem -LiteralPath $root -Filter $g -File -ErrorAction SilentlyContinue)) { $targets += $f.FullName }
        }
        if ($targets.Count -eq 0) { continue }
        $lines = @()
        if ($rg) {
            $a = @('-o', '-n', '--no-heading', '--with-filename', '--no-ignore', '--max-filesize', '5M')
            foreach ($g in @('*.py', '*.ps1', '*.sh', '*.toml', '*.json', '*.yaml', '*.yml', '*.cfg', '*.ini', '*.bat', '*.cmd')) { $a += @('-g', $g) }
            $a += $sensGlobs + @('-g', '!**/__pycache__/**', '-g', '!**/.git/**', '-e', $re) + $targets
            $r = Invoke-Proc $rg $a 900
            if ($r.rc -le 1) { $lines = @($r.out -split "`n") } else { Say "  (rg failed on $tag tree: $($r.err.Trim()))" 'Yellow' }
        } else {
            Say "  (rg not found: slow PowerShell reference scan of the $tag tree)" 'Yellow'
            foreach ($t2 in $targets) {
                $files = if ((Get-Item -LiteralPath $t2).PSIsContainer) { Get-ChildItem -LiteralPath $t2 -Recurse -File -Include *.py, *.ps1, *.sh, *.toml, *.json, *.yaml, *.yml, *.cfg, *.ini -ErrorAction SilentlyContinue } else { Get-Item -LiteralPath $t2 }
                foreach ($f in $files) {
                    foreach ($mf in (Select-String -LiteralPath $f.FullName -Pattern $re -AllMatches -ErrorAction SilentlyContinue)) {
                        foreach ($mm in $mf.Matches) { $lines += ('{0}:{1}:{2}' -f $f.FullName, $mf.LineNumber, $mm.Value) }
                    }
                }
            }
        }
        foreach ($ln in $lines) {
            $ln = $ln.TrimEnd("`r")
            $mf = [regex]::Match($ln, '^(.+?):(\d+):(.+)$')
            if (-not $mf.Success) { continue }
            $file = $mf.Groups[1].Value
            if ($IgnoreRef -and $file -match $IgnoreRef) { continue }
            $lit = ($mf.Groups[3].Value -replace '\\\\+', '\' -replace '/', '\').TrimEnd('.', '\', ':')
            if ($lit.Length -lt 4) { continue }
            [void]$script:Lit.Add(@{ lit = (K $lit); file = $file; tree = $tag })
        }
        # relative repo paths ("output/x", "runs\x", Path(...) / "output/x/y.py") and dotted imports
        # ("from output.x.y import"), resolved against the tree root.  Added 2026-10-10 after Stage 1 removed
        # untracked output/ code dirs that tests and traffic_shepherd.mechanics import by relative name.
        $relRe = '(^|[^A-Za-z0-9_\\/.:-])(output|runs|tmp)([\\/]+|\.)[A-Za-z0-9_][A-Za-z0-9_.\-]*([\\/]+[A-Za-z0-9_.\-]+)*'
        $rlines = @()
        if ($rg) {
            $a = @('-o', '-n', '--no-heading', '--with-filename', '--no-ignore', '--max-filesize', '5M')
            foreach ($g in @('*.py', '*.ps1', '*.sh', '*.toml', '*.json', '*.yaml', '*.yml', '*.cfg', '*.ini', '*.bat', '*.cmd')) { $a += @('-g', $g) }
            $a += $sensGlobs + @('-g', '!**/__pycache__/**', '-g', '!**/.git/**', '-e', $relRe) + $targets
            $r = Invoke-Proc $rg $a 900
            if ($r.rc -le 1) { $rlines = @($r.out -split "`n") } else { Say "  (rg relative scan failed on $tag tree: $($r.err.Trim()))" 'Yellow' }
        } else {
            Say "  (rg not found: relative-path references of the $tag tree NOT scanned)" 'Yellow'
        }
        foreach ($ln in $rlines) {
            $ln = $ln.TrimEnd("`r")
            $mf = [regex]::Match($ln, '^(.+?):(\d+):(.+)$')
            if (-not $mf.Success) { continue }
            $file = $mf.Groups[1].Value
            if ($IgnoreRef -and $file -match $IgnoreRef) { continue }
            $v = $mf.Groups[3].Value
            if ($v -notmatch '^(output|runs|tmp)') { $v = $v.Substring(1) }
            if ($v -notmatch '[\\/]') { $v = $v -replace '\.', '\' }   # dotted import form
            $rel = ($v -replace '[\\/]+', '\').TrimEnd('.', '\', ':')
            if ($rel -notmatch '\\') { continue }
            [void]$script:Lit.Add(@{ lit = (K (Join-Path $root $rel)); file = $file; tree = $tag })
        }
    }
    # de-duplicate
    $seen = @{}; $uniq = New-Object System.Collections.ArrayList
    foreach ($e in $script:Lit) { $k = $e.lit + '|' + $e.file; if (-not $seen.ContainsKey($k)) { $seen[$k] = 1; [void]$uniq.Add($e) } }
    $script:Lit = $uniq
}
$GenericLit = @{}
foreach ($g in @($M.generic_literals)) { $GenericLit[(K $g)] = 1 }
function Get-Refs([string]$path, [bool]$cleanOnly = $false) {
    $c = K $path
    $hits = New-Object System.Collections.ArrayList
    foreach ($e in $script:Lit) {
        if ($cleanOnly -and $e.tree -ne 'clean') { continue }
        $l = $e.lit
        $isHit = $false
        if ($l -eq $c -or $l.StartsWith($c + '\')) { $isHit = $true }
        elseif ($c.StartsWith($l + '\') -and -not $GenericLit.ContainsKey($l)) { $isHit = $true }
        if ($isHit) { [void]$hits.Add(("{0}:{1} -> {2}" -f $e.tree, ($e.file -replace '^.*\\', ''), $e.lit)); if ($hits.Count -ge 4) { break } }
    }
    return @($hits)
}
# Frozen records (the experiment ledger and other protocols/*.json, the root smoke verdicts, docs/records) only
# CITE evidence; since 2026-10-10 the ledger accepts Drive-offloaded paths (protocols/offloaded_paths.json).
# An item cited only by records is trimmed: its cited files (<= 1 GiB each) stay, the verified rest goes.
function Test-RecordFile([string]$file) {
    $f = $file.ToLowerInvariant()
    # tools\data_registry.json is the offload registry (original path -> Drive object + md5 manifest, fetched on demand)
    return ($f -like '*\protocols\*.json' -or $f -like '*\docs\records\*' -or $f -like '*\tools\data_registry.json' -or
            $f -like '*\smoke_v22_r2_verdict.json' -or $f -like '*\v22_smoke_pass.json')
}
function Get-RefSplit([string]$path) {
    # all hits (no cap), split into code references and record citations
    $c = K $path
    $code = New-Object System.Collections.ArrayList; $rec = New-Object System.Collections.ArrayList
    foreach ($e in $script:Lit) {
        $l = $e.lit
        $isHit = ($l -eq $c -or $l.StartsWith($c + '\')) -or ($c.StartsWith($l + '\') -and -not $GenericLit.ContainsKey($l))
        if (-not $isHit) { continue }
        if (Test-RecordFile $e.file) { [void]$rec.Add($e) } else { [void]$code.Add($e) }
    }
    return @{ code = @($code); rec = @($rec) }
}
function Get-CitedRels([string]$path, $recHits, $w) {
    # walked relative paths of the files inside $path that records cite (a cited sub-directory keeps all its files)
    $byKey = @{}; for ($i = 0; $i -lt $w.Rel.Count; $i++) { $byKey[(K $w.Rel[$i])] = $i }
    $keep = New-Object 'System.Collections.Generic.HashSet[string]'
    $big = New-Object System.Collections.ArrayList
    $c = K $path
    foreach ($e in $recHits) {
        $l = $e.lit
        if (-not $l.StartsWith($c + '\')) { continue }   # the item itself or an ancestor: covered by the offload map
        $rk = $l.Substring($c.Length + 1)
        $idx = @()
        if ($byKey.ContainsKey($rk)) { $idx = @($byKey[$rk]) }
        else { foreach ($k in $byKey.Keys) { if ($k.StartsWith($rk + '\')) { $idx += $byKey[$k] } } }
        foreach ($i in $idx) {
            if ($w.Size[$i] -gt 1GB) { [void]$big.Add($w.Rel[$i]); continue }
            [void]$keep.Add($w.Rel[$i])
        }
    }
    return @{ keep = $keep; big = @($big) }
}

# ---------------- git-tracked check inside the main repo ----------------
function Test-TrackedInMain([string]$path) {
    if (-not (UnderP $path $MainRepo)) { return $null }
    $rel = (NP $path).Substring((NP $MainRepo).Length).TrimStart('\') -replace '\\', '/'
    if (-not $rel) { return 'is the repo root' }
    $r = GitX $MainRepo @('ls-files', '-z', '--', $rel)
    if ($r.rc -ne 0) { return "git ls-files failed: $($r.err.Trim())" }
    if ($r.out.Trim("`0", "`n", "`r", ' ')) { $n = @($r.out -split "`0" | Where-Object { $_ }).Count; return "$n tracked file(s) (history keeps them; delete via git, not here)" }
    return $null
}

# ---------------- local vs manifest comparison ----------------
function Check-Local($w, $exp, [bool]$deep, [string]$root, [bool]$sha, [bool]$allowSens = $false) {
    # $exp: hashtable rel -> @{ md5/sha; size; t }
    if ($w.Reparse.Count -gt 0) { return @{ ok = $false; status = 'LINKS'; reason = ("junction/symlink inside: " + ($w.Reparse[0..([Math]::Min(2, $w.Reparse.Count - 1))] -join ', ')) } }
    if ($w.Sensitive.Count -gt 0 -and -not $allowSens) { return @{ ok = $false; status = 'SENSITIVE_INSIDE'; reason = ("{0} protected-name entries inside (never uploaded): {1}" -f $w.Sensitive.Count, ($w.Sensitive[0..([Math]::Min(1, $w.Sensitive.Count - 1))] -join ', ')) } }
    if ($w.Errors.Count -gt 0) { return @{ ok = $false; status = 'WALK_ERROR'; reason = $w.Errors[0] } }
    $extra = New-Object System.Collections.ArrayList; $bad = New-Object System.Collections.ArrayList; $mod = New-Object System.Collections.ArrayList
    $hr = New-Object System.Collections.ArrayList; $he = New-Object System.Collections.ArrayList
    for ($i = 0; $i -lt $w.Rel.Count; $i++) {
        $rel = $w.Rel[$i]
        $e = $exp[$rel]
        if ($null -eq $e) { [void]$extra.Add($rel); continue }
        if ($e.size -ne $w.Size[$i]) { [void]$bad.Add($rel); continue }
        if ($e.t -gt 0 -and $w.MTicks[$i] -gt ($e.t + 20000000)) { [void]$mod.Add($rel); continue }
        if ($deep) { [void]$hr.Add($rel); [void]$he.Add([string]$e.h) }
    }
    if ($extra.Count) { return @{ ok = $false; status = 'LOCAL_NOT_IN_BACKUP'; reason = ("{0} local file(s) not in the backup manifest, e.g. {1}" -f $extra.Count, $extra[0]) } }
    if ($bad.Count) { return @{ ok = $false; status = 'LOCAL_CHANGED'; reason = ("{0} file(s) differ in size from the backup, e.g. {1}" -f $bad.Count, $bad[0]) } }
    if ($mod.Count) { return @{ ok = $false; status = 'LOCAL_CHANGED'; reason = ("{0} file(s) modified after the backup was taken, e.g. {1}" -f $mod.Count, $mod[0]) } }
    if ($deep -and $hr.Count -gt 0) {
        $mm = [MC]::VerifyHashes($root, [string[]]$hr.ToArray(), [string[]]$he.ToArray(), $sha)
        if ($mm.Count -gt 0) { return @{ ok = $false; status = 'LOCAL_CHANGED'; reason = ("{0}+ file(s) fail {1} re-hash, e.g. {2}" -f $mm.Count, $(if ($sha) { 'SHA-256' } else { 'MD5' }), $mm[0]) } }
    }
    return @{ ok = $true; status = 'OK'; reason = '' }
}

# ---------------- Stage 1: item evaluation ----------------
function Eval-Proof($it, $pr, $w, [bool]$isFile) {
    $path = [string]$it.path
    switch ([string]$pr.t) {
        'mirror' {
            $mf = Read-Manifest ([string]$pr.manifest) ([string]$pr.manifest_sha256) ([string]$pr.ts)
            if (-not $mf.ok) { return @{ ok = $false; status = 'RECEIPT_STALE'; reason = $mf.why } }
            $prefix = [string]$pr.prefix
            $exp = @{}
            foreach ($k in $mf.map.Keys) {
                if ($prefix -and -not $k.StartsWith($prefix)) { continue }
                $rel = $k.Substring($prefix.Length)
                $e = $mf.map[$k]
                $exp[$rel] = @{ size = $e.size; h = $e.md5; t = $mf.ticks }
            }
            $lc = Check-Local $w $exp $script:DeepNow $path $false $script:PartialOk
            if (-not $lc.ok) { return $lc }
            # live Drive re-check: every local file must exist at <remote>/<prefix+rel> with equal size+MD5
            $remote = ([string]$pr.remote).TrimEnd('/')
            $bad = 0; $first = ''
            for ($i = 0; $i -lt $w.Rel.Count; $i++) {
                $rel = $w.Rel[$i]
                $ent = Get-Remote ($remote + '/' + $prefix + $rel)
                $e = $exp[$rel]
                if ($null -eq $ent -or $ent.size -ne $e.size -or $ent.md5 -ne $e.h) { $bad++; if (-not $first) { $first = $rel } }
            }
            if ($bad -gt 0) { return @{ ok = $false; status = 'DRIVE_UNVERIFIED'; reason = ("{0} file(s) missing or different on Drive at {1}, e.g. {2}{3}" -f $bad, $remote, $first, (Remote-Err)) } }
            return @{ ok = $true; status = 'OK'; reason = ''; drive = @($remote + '  (mirror, ' + $w.Rel.Count + ' files size+MD5 re-checked)') }
        }
        'archive' {
            $exp = @{}
            $drv = @()
            foreach ($part in @($pr.parts)) {
                $mf = Read-Manifest ([string]$part.manifest) ([string]$part.manifest_sha256) ([string]$part.ts)
                if (-not $mf.ok) { return @{ ok = $false; status = 'RECEIPT_STALE'; reason = $mf.why } }
                foreach ($k in $mf.map.Keys) { $e = $mf.map[$k]; $exp[$k] = @{ size = $e.size; h = $e.md5; t = $mf.ticks } }
            }
            $lc = Check-Local $w $exp $script:DeepNow $path $false $script:PartialOk
            if (-not $lc.ok) { return $lc }
            foreach ($part in @($pr.parts)) {
                $ent = Get-Remote ([string]$part.remote)
                if ($null -eq $ent) { return @{ ok = $false; status = 'DRIVE_UNVERIFIED'; reason = ("archive not found on Drive: " + $part.remote + (Remote-Err)) } }
                if ([long]$ent.size -ne [long]$part.bytes -or $ent.md5 -ne ([string]$part.md5).ToLowerInvariant()) {
                    return @{ ok = $false; status = 'DRIVE_UNVERIFIED'; reason = ("archive size/MD5 differs on Drive: " + $part.remote) }
                }
                $drv += ([string]$part.remote)
            }
            return @{ ok = $true; status = 'OK'; reason = ''; drive = $drv }
        }
        'legacy' {
            if (-not (Test-Path -LiteralPath ([string]$pr.receipt))) { return @{ ok = $false; status = 'RECEIPT_STALE'; reason = 'legacy receipt missing' } }
            $rj = Get-Content -LiteralPath ([string]$pr.receipt) -Raw | ConvertFrom-Json
            $strip = [string]$pr.strip
            $exp = @{}
            foreach ($pp in $rj.files_sha256.PSObject.Properties) {
                $rel = $pp.Name.Replace('\', '/'); if ($rel.StartsWith($strip)) { $rel = $rel.Substring($strip.Length) }
                $exp[$rel] = @{ size = -1; h = ([string]$pp.Value).ToLowerInvariant(); t = [long]0 }
            }
            # sizes are not recorded: size check is skipped, content is verified by SHA-256 (deep) and by file-set equality
            if ($w.Reparse.Count -gt 0) { return @{ ok = $false; status = 'LINKS'; reason = 'junction inside' } }
            if ($w.Sensitive.Count -gt 0 -and -not $script:PartialOk) { return @{ ok = $false; status = 'SENSITIVE_INSIDE'; reason = 'protected-name entries inside' } }
            $extra = @(); $rels = New-Object System.Collections.ArrayList; $hs = New-Object System.Collections.ArrayList
            foreach ($rel in $w.Rel) { if (-not $exp.ContainsKey($rel)) { $extra += $rel } else { [void]$rels.Add($rel); [void]$hs.Add($exp[$rel].h) } }
            if ($extra.Count) { return @{ ok = $false; status = 'LOCAL_NOT_IN_BACKUP'; reason = ("{0} local file(s) not in the legacy receipt, e.g. {1}" -f $extra.Count, $extra[0]) } }
            if ($script:DeepNow) {
                $mm = [MC]::VerifyHashes($path, [string[]]$rels.ToArray(), [string[]]$hs.ToArray(), $true)
                if ($mm.Count) { return @{ ok = $false; status = 'LOCAL_CHANGED'; reason = ("{0}+ file(s) fail SHA-256 vs legacy receipt, e.g. {1}" -f $mm.Count, $mm[0]) } }
            }
            $ent = Get-Remote ([string]$pr.remote)
            if ($null -eq $ent -or [long]$ent.size -ne [long]$pr.bytes -or $ent.md5 -ne ([string]$pr.md5).ToLowerInvariant()) {
                return @{ ok = $false; status = 'DRIVE_UNVERIFIED'; reason = ("legacy archive not matching on Drive: " + $pr.remote) }
            }
            $note = if ($script:DeepNow) { 'SHA-256 verified' } else { 'file-set equal; SHA-256 re-hash deferred to -Deep/-Execute' }
            return @{ ok = $true; status = 'OK'; reason = ''; drive = @([string]$pr.remote + '  (legacy archive; ' + $note + ')') }
        }
    }
    return @{ ok = $false; status = 'NO_PROOF'; reason = 'unknown proof type' }
}

function Eval-Special($it, $w) {
    $sp = $it.special
    $path = [string]$it.path
    switch ([string]$sp.kind) {
        'redundant_of_keeplocal' {
            $k = @($M.keep_local) | Where-Object { $_.name -eq $sp.keep } | Select-Object -First 1
            $cacheOk = Test-Path -LiteralPath (Join-Path (Join-Path $CacheRoot $k.cache) 'CACHE_RECEIPT.json')
            $srcOk = $false
            if (Test-Path -LiteralPath $k.source) {
                $mf = Read-Manifest ([string]$k.manifest) ([string]$k.manifest_sha256) ''
                if ($mf.ok) {
                    $ws = [MC]::Walk($k.source, $true); $same = ($ws.Rel.Count -eq $mf.n)
                    if ($same) { for ($i = 0; $i -lt $ws.Rel.Count; $i++) { $e = $mf.map[$ws.Rel[$i]]; if ($null -eq $e -or $e.size -ne $ws.Size[$i]) { $same = $false; break } } }
                    $srcOk = $same
                }
            }
            if (-not ($cacheOk -or $srcOk)) { return @{ ok = $false; status = 'SPECIAL_FAIL'; reason = ("superset '" + $sp.keep + "' not verified present (neither source nor verified cache copy)") } }
            return @{ ok = $true; status = 'OK'; reason = ''; drive = @('superset ' + $sp.keep + ' present (' + $(if ($cacheOk) { 'cache receipt' } else { 'source sizes match manifest' }) + ')') }
        }
        'fixed_size' {
            if ($w.Bytes -ne [long]$sp.size) { return @{ ok = $false; status = 'LOCAL_CHANGED'; reason = ("size is now {0}, expected {1} (file is still changing?)" -f $w.Bytes, $sp.size) } }
            $age = ((Get-Date).ToUniversalTime().Ticks - $w.MTicks[0]) / 1e7 / 60
            if ($age -lt 15) { return @{ ok = $false; status = 'LOCAL_CHANGED'; reason = ("modified {0:N0} min ago - wait" -f $age) } }
            return @{ ok = $true; status = 'OK'; reason = ''; drive = @('owner-designated partial/dropped copy') }
        }
        'git_ignored_scratch' {
            $rel = [string]$sp.rel
            $c = GitX ([string]$sp.repo) @('check-ignore', '-q', '--', $rel)
            if ($c.rc -ne 0) { return @{ ok = $false; status = 'SPECIAL_FAIL'; reason = "$rel is not git-ignored in $($sp.repo)" } }
            $l = GitX ([string]$sp.repo) @('ls-files', '-z', '--', $rel)
            if ($l.out.Trim("`0", "`n", ' ')) { return @{ ok = $false; status = 'SPECIAL_FAIL'; reason = 'contains tracked files' } }
            return @{ ok = $true; status = 'OK'; reason = ''; drive = @('git-ignored regenerable scratch') }
        }
    }
    return @{ ok = $false; status = 'NO_PROOF'; reason = 'unknown special' }
}

function Eval-Item($it) {
    $path = NP ([string]$it.path)
    $id = [string]$it.id
    if (-not (Test-Path -LiteralPath $path)) { return Add-Res $id 'item' $path 'GONE' 'not present (already removed)' 0 0 @() }
    $skipWt = [bool]($it.special -and [string]$it.special.kind -eq 'git_ignored_scratch')   # explicit owner request: scratch inside the integration tree
    $why = Get-ProtectedReason $path $skipWt
    if ($why) { return Add-Res $id 'item' $path 'PROTECTED' $why 0 0 @() }
    if ($it.backup_only -and -not $IncludeBackupOnly) { return Add-Res $id 'item' $path 'BACKUP_ONLY' 'class KEEP-ACTIVE-backup-only (live-referenced sidecar): backed up, not offered (-IncludeBackupOnly)' 0 0 @() }
    $ko = Get-KeepOverlap $path
    if ($ko) { return Add-Res $id 'item' $path 'KEEP_LOCAL' ("is/contains keep-local item '$ko' (Stage 2 first)") 0 0 @() }
    if ([MC]::PathIsReparse($path)) { return Add-Res $id 'item' $path 'LINKS' 'item itself is a junction/symlink' 0 0 @() }
    $anc = $path; $linkAnc = $null
    while ($true) {
        $par = Split-Path -Parent $anc
        if (-not $par -or $par -eq $anc) { break }
        if ([MC]::PathIsReparse($par)) { $linkAnc = $par; break }
        $anc = $par
    }
    if ($linkAnc) { return Add-Res $id 'item' $path 'LINKS' "reparse ancestor $linkAnc" 0 0 @() }
    $isFile = Test-Path -LiteralPath $path -PathType Leaf
    $pr = Find-ProcUsing $path
    if ($pr) { return Add-Res $id 'item' $path 'IN_USE' "referenced on a running process command line: $pr" 0 0 @() }
    $w = [MC]::Walk($path, (-not $ReleaseProtectedNames))
    if ($AssumeSealedMoved -and $w.Sensitive.Count -gt 0) {
        # simulation: protected-name entries that the MOVE_TO_SEALED plan covers count as already moved out
        $unc = New-Object 'System.Collections.Generic.List[string]'
        foreach ($sn in $w.Sensitive) { if (-not (Test-SealedCovered (Join-Path $path ($sn -replace '/', '\')))) { $unc.Add($sn) } }
        $w.Sensitive = $unc
    }
    $bytes = $w.Bytes; $files = $w.Rel.Count
    $links = @()
    if ($w.Reparse.Count -gt 0) {
        $inside = $false
        if ($it.special -and $it.special.unlink_links) {
            $inside = $true
            foreach ($rl in $w.Reparse) {
                $lpth = Join-Path $path ($rl -replace '/', '\')
                $tg = $null; try { $tg = (Get-Item -LiteralPath $lpth -Force).Target } catch { }
                if ($tg -is [array]) { $tg = $tg[0] }
                $tg = ([string]$tg) -replace '^\\\\\?\\', ''
                if (-not $tg -or -not (UnderP $tg $path)) { $inside = $false; break }
            }
        }
        if (-not $inside) { return Add-Res $id 'item' $path 'LINKS' ("junction/symlink inside: " + $w.Reparse[0]) $bytes $files @() }
        $links = @($w.Reparse)
    }
    $script:PartialOk = $false; $left = 0
    if ($w.Sensitive.Count -gt 0) {
        if ($RemoveAroundProtected -and $isFile -eq $false) { $script:PartialOk = $true; $left = $w.Sensitive.Count }
        else { return Add-Res $id 'item' $path 'SENSITIVE_INSIDE' ("{0} protected-name entries inside (never uploaded; whole-folder removal refused; -RemoveAroundProtected removes only the verified rest, -ReleaseProtectedNames if a mirror receipt covers them too), e.g. {1}" -f $w.Sensitive.Count, $w.Sensitive[0]) $bytes $files @() }
    }
    $keepRels = $null
    if (-not $SkipReferenceScan) {
        $refs = Get-Refs $path
        if ($refs.Count -gt 0) {
            $split = Get-RefSplit $path
            if ($split.code.Count -gt 0 -or $NoTrimRecordRefs -or $isFile -or $it.special) {
                return Add-Res $id 'item' $path 'REFERENCED' ("active code/config reference: " + ($refs -join ' | ')) $bytes $files @()
            }
            $cr = Get-CitedRels $path $split.rec $w
            $keepRels = $cr.keep
            if ($keepRels.Count -ge $files) { return Add-Res $id 'item' $path 'REFERENCED' ("every file is cited by a frozen record: " + ($refs -join ' | ')) $bytes $files @() }
        }
    }
    $tr = Test-TrackedInMain $path
    if ($tr) { return Add-Res $id 'item' $path 'TRACKED' $tr $bytes $files @() }
    if ($it.special) {
        $sr = Eval-Special $it $w
        if (-not $sr.ok) { return Add-Res $id 'item' $path $sr.status $sr.reason $bytes $files @() }
        $o = Add-Res $id 'item' $path 'ELIGIBLE' 'redundant' $bytes $files $sr.drive
        $maxm0 = 0; if ($w.MTicks.Count) { $maxm0 = ($w.MTicks | Measure-Object -Maximum).Maximum }
        $o | Add-Member -NotePropertyName Snap -NotePropertyValue @{ bytes = $bytes; files = $files; maxm = $maxm0; links = $links } -Force
        $o | Add-Member -NotePropertyName Item -NotePropertyValue $it -Force
        return $o
    }
    $fails = @()
    $okp = $null
    foreach ($pf in @($it.proofs)) {
        $pe = Eval-Proof $it $pf $w $isFile
        if ($pe.ok) { $okp = $pe; break } else { $fails += ("[" + $pf.t + "] " + $pe.status + ": " + $pe.reason) }
    }
    if ($null -eq $okp) {
        $first = ($fails | Select-Object -First 1)
        $st = if ($first -match '\] (\w+):') { $Matches[1] } else { 'NO_PROOF' }
        return Add-Res $id 'item' $path $st ($fails -join ' || ') $bytes $files @()
    }
    $note = ''
    if ($left -gt 0) { $note = ("PARTIAL: removes the {0} verified files; {1} protected-name entr{2} stay in place (e.g. {3})" -f $files, $left, $(if ($left -eq 1) { 'y' } else { 'ies' }), $w.Sensitive[0]) }
    $rels = @($w.Rel); $trim = $false
    if ($keepRels -and $keepRels.Count -gt 0) {
        $rels = @($w.Rel | Where-Object { -not $keepRels.Contains($_) }); $trim = $true
        $kb = [long]0; for ($i = 0; $i -lt $w.Rel.Count; $i++) { if ($keepRels.Contains($w.Rel[$i])) { $kb += $w.Size[$i] } }
        $note = (("TRIM: keeps the {0} file(s) cited by frozen records ({1}), removes the other {2} verified files; " -f $keepRels.Count, (GB $kb), $rels.Count) + $note).Trim()
    } elseif ($null -ne $keepRels) { $note = ("cited only as a whole by frozen records: offloaded to Drive (protocols/offloaded_paths.json) " + $note).Trim() }
    $o = Add-Res $id 'item' $path 'ELIGIBLE' $note $bytes $files $okp.drive
    $maxm = 0; if ($w.MTicks.Count) { $maxm = ($w.MTicks | Measure-Object -Maximum).Maximum }
    $o | Add-Member -NotePropertyName Snap -NotePropertyValue @{ bytes = $bytes; files = $files; maxm = $maxm; links = $links; partial = (($left -gt 0) -or $trim); left = $left; rels = $rels } -Force
    $o | Add-Member -NotePropertyName Item -NotePropertyValue $it -Force
    return $o
}

# ---------------- Stage 1: worktrees ----------------
$script:WtRegistered = $null
$script:WtLinkCache = @{}
$script:SealedRows = $null
$script:SealedIndex = @{}
$script:TarMan = @{}
function Get-RegisteredWorktrees {
    if ($null -ne $script:WtRegistered) { return $script:WtRegistered }
    $script:WtRegistered = @{}
    $r = GitX $MainRepo @('worktree', 'list', '--porcelain')
    foreach ($line in ($r.out -split "`n")) { if ($line -match '^worktree (.+)$') { $script:WtRegistered[(K $Matches[1].Trim())] = 1 } }
    return $script:WtRegistered
}
function Read-ListFile([string]$path) {
    $set = @{}
    if (-not $path -or -not (Test-Path -LiteralPath $path)) { return $set }
    foreach ($l in [System.IO.File]::ReadLines($path)) { $l = $l.TrimEnd("`r"); if ($l) { $set[$l] = 1 } }
    return $set
}

# One object per reparse point found by the no-follow walker (rel, abs, target, exists, isDir, kind).
# kind: 'self' (target inside the worktree), 'worktree:<name>' (target inside another E:\rd-wt tree), 'external'.
function Get-LinkInfo([string]$root, $reparseRels) {
    $out = New-Object System.Collections.ArrayList
    $rdwt = [string]$P.rdwt
    foreach ($rl in @($reparseRels)) {
        if (-not $rl -or $rl -eq '.') { continue }
        $abs = Join-Path $root (([string]$rl) -replace '/', '\')
        $tg = $null
        try { $tg = (Get-Item -LiteralPath $abs -Force -ErrorAction Stop).Target } catch { }
        if (-not $tg) { try { $tg = (Get-Item -LiteralPath ([MC]::LP($abs)) -Force -ErrorAction Stop).Target } catch { } }
        if ($tg -is [array]) { $tg = $tg[0] }
        $tg = ([string]$tg) -replace '^\\\\\?\\', ''
        if ($tg -and -not [System.IO.Path]::IsPathRooted($tg)) { $tg = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $abs) $tg)) }
        $isDir = $false
        try { $isDir = (([System.IO.File]::GetAttributes([MC]::LP($abs)) -band [System.IO.FileAttributes]::Directory) -ne 0) } catch { }
        $kind = 'external'
        if ($tg) {
            if (UnderP $tg $root) { $kind = 'self' }
            elseif (UnderP $tg $rdwt) {
                $rest = (NP $tg).Substring((NP $rdwt).Length).TrimStart('\')
                $kind = 'worktree:' + ($rest.Split('\')[0])
            }
        }
        $ex = $false; if ($tg) { $ex = [MC]::Exists($tg) }
        [void]$out.Add([pscustomobject]@{ rel = [string]$rl; abs = $abs; target = $tg; exists = $ex; isDir = $isDir; kind = $kind })
    }
    return @($out)
}

# Unlink EVERY reparse point inside $root (link only: a directory junction/symlink is removed with RemoveDirectory, a file
# symlink with DeleteFile; nothing below a link is ever touched).  Each link is journaled with its target; a target that
# existed right before the unlink must still exist right after it.  A final no-follow walk must find no reparse point.
function Unlink-AllLinks([string]$root) {
    $w = [MC]::Walk($root, $false)
    if (@($w.Reparse) -contains '.') { throw "the root $root is itself a reparse point" }
    $infos = @(Get-LinkInfo $root $w.Reparse)
    foreach ($li in $infos) {
        if (-not $li.target) { throw "unreadable link target for $($li.abs): refusing" }
        $before = [MC]::Exists($li.target)
        [MC]::UnlinkLink($li.abs)
        if ([MC]::PathIsReparse($li.abs) -or [MC]::Exists($li.abs)) { throw "could not unlink $($li.abs)" }
        $after = [MC]::Exists($li.target)
        if ($before -and -not $after) { throw "target $($li.target) of $($li.abs) vanished while unlinking (link-only unlink expected)" }
        Jrn 'UNLINKED' $li.abs 0 ("junction/symlink inside the worktree removed (link only); target " + $li.target + $(if ($before) { '; target verified present afterwards' } else { '; target was already absent (dangling link)' }))
    }
    $w2 = [MC]::Walk($root, $false)
    if (@($w2.Reparse).Count -gt 0) { throw ("reparse point(s) remain after unlinking: " + (@($w2.Reparse)[0])) }
    return $infos.Count
}

# ---- tracked-dirty worktrees: accepted only if the dirty state is preserved ----
function Get-TarMembers([string]$manifestPath) {
    $k = K $manifestPath
    if ($script:TarMan.ContainsKey($k)) { return $script:TarMan[$k] }
    $res = $null
    if ($manifestPath -and (Test-Path -LiteralPath $manifestPath)) {
        try {
            Add-Type -AssemblyName System.Web.Extensions -ErrorAction Stop
            $js = New-Object System.Web.Script.Serialization.JavaScriptSerializer
            $js.MaxJsonLength = [int]::MaxValue; $js.RecursionLimit = 100
            $o = $js.DeserializeObject([System.IO.File]::ReadAllText($manifestPath, [System.Text.Encoding]::UTF8))
            $res = @{}
            foreach ($mf in $o['members']) { $res[[string]$mf['path']] = $mf }
        } catch { $res = $null }
    }
    $script:TarMan[$k] = $res
    return $res
}
function Test-DirtyPreserved($w, [string]$path, [string[]]$lines) {
    # (a) the receipt's preserve ref / wip commit contains exactly the current tracked state
    $ref = ''
    if ($w.preserve_ref) { $ref = [string]$w.preserve_ref } elseif ($w.commit) { $ref = [string]$w.commit }
    if ($ref) {
        $rv = GitX $MainRepo @('rev-parse', '--verify', '--quiet', ($ref + '^{commit}'))
        if ($rv.rc -eq 0) {
            $d = GitX $path @('diff', '--quiet', $rv.out.Trim(), '--')
            if ($d.rc -eq 0) { return ("tracked state equals the preserved commit " + $ref + " (" + $rv.out.Trim().Substring(0, 8) + ")") }
        }
    }
    # (b) tar.zst receipt: every dirty file was archived verbatim (+ patch.diff) and is unchanged since
    if ([string]$w.method -eq 'tar.zst' -and $w.archive -and @($w.dirty_paths).Count -gt 0 -and @($w.dirty_sens_skipped).Count -eq 0) {
        $mem = Get-TarMembers ([string]$w.archive.manifest_local)
        if ($null -eq $mem -or -not $mem.ContainsKey('__wtclean_tracked_dirty__/patch.diff')) { return $null }
        $dp = @{}; foreach ($p in @($w.dirty_paths)) { $dp[[string]$p] = 1 }
        $n = 0
        foreach ($l in $lines) {
            if ($l.Length -lt 4) { continue }
            $xy = $l.Substring(0, 2).Trim(); $p = $l.Substring(3).Trim()
            if ($p.Contains(' -> ') -or $p.StartsWith('"') -or -not $dp.ContainsKey($p)) { return $null }
            if ($xy -eq 'D') { $n++; continue }                         # deletion: described by patch.diff
            if ($xy -notin @('M', 'MM', 'A', 'AM')) { return $null }
            $key = '__wtclean_tracked_dirty__/files/' + $p
            $loc = Join-Path $path ($p -replace '/', '\')
            if (-not $mem.ContainsKey($key) -or -not (Test-Path -LiteralPath $loc -PathType Leaf)) { return $null }
            if (([MC]::Sha256File($loc)) -ne ([string]$mem[$key]['sha256']).ToLowerInvariant()) { return $null }
            $n++
        }
        return ("{0} dirty tracked file(s) are archived verbatim (+patch.diff) in the receipt's tar.zst and unchanged" -f $n)
    }
    return $null
}

# ---- sealed\ move records (MOVE_TO_SEALED.ps1 writes MOVE_MANIFEST.csv: run,source,dest,kind,bytes,files,sha256,status) ----
function Get-SealedMoveRows {
    if ($null -ne $script:SealedRows) { return $script:SealedRows }
    $rows = New-Object System.Collections.ArrayList
    $script:SealedIndex = @{}
    $file = $SealedManifestPath; $col = 'source'
    if ($AssumeSealedMoved) { $file = $AssumeSealedMoved; $col = 'source_path' }
    if ($file -and (Test-Path -LiteralPath $file)) {
        foreach ($r in (Import-Csv -LiteralPath $file)) {
            if (-not $AssumeSealedMoved -and $r.status -ne 'MOVED') { continue }
            $s = [string]$r.$col
            if (-not $s) { continue }
            $row = [pscustomobject]@{ src = (K $s); dst = [string]$r.dest }
            [void]$rows.Add($row)
            $script:SealedIndex[$row.src] = $row
        }
    }
    $script:SealedRows = $rows
    return $rows
}
function Test-SealedCovered([string]$full) {
    # the file itself or one of its parent folders is the source of a MOVED row (plan row in simulation mode)
    [void](Get-SealedMoveRows)
    $acc = ''
    foreach ($seg in ((K $full) -split '\\')) {
        $acc = if ($acc) { $acc + '\' + $seg } else { $seg }
        $r = $script:SealedIndex[$acc]
        if ($r -and ($AssumeSealedMoved -or (Test-Path -LiteralPath $r.dst))) { return $r }
    }
    return $null
}


function Move-ToSealed($r) {
    # opaque move (never opened; bytes only streamed through SHA-256) of this worktree's untracked sensitive-named files
    $moved = 0
    foreach ($rel in @($r.Snap.sealMoves)) {
        $src = Join-Path $r.Path (([string]$rel) -replace '/', '\')
        $dst = Join-Path $SealedRoot ((DriveOf $src) + '\' + (NP $src).Substring(3))
        if (-not (UnderP $dst $SealedRoot)) { throw "sealed destination outside the sealed root: $dst" }
        if ((DriveOf $src) -ne (DriveOf $SealedRoot)) { throw "sealed move across drives is not supported: $src" }
        if ([MC]::PathIsReparse($src)) { throw "reparse point: $src" }
        if (-not [System.IO.File]::Exists([MC]::LP($src))) { throw "file vanished before the sealed move: $src" }
        if ([MC]::Exists($dst)) { throw "sealed destination already exists: $dst" }
        $len = (New-Object System.IO.FileInfo ([MC]::LP($src))).Length
        $h = [MC]::Sha256File($src)
        [void][System.IO.Directory]::CreateDirectory([MC]::LP((Split-Path -Parent $dst)))
        [System.IO.File]::Move([MC]::LP($src), [MC]::LP($dst))
        if (([MC]::Sha256File($dst)) -ne $h) { throw "sealed copy hash differs from the source hash after the move: $dst" }
        if (-not (Test-Path -LiteralPath $SealedManifestPath)) {
            [void][System.IO.Directory]::CreateDirectory([MC]::LP((Split-Path -Parent $SealedManifestPath)))
            [System.IO.File]::WriteAllText($SealedManifestPath, "run,source,dest,kind,bytes,files,sha256,status`n", (New-Object System.Text.UTF8Encoding($true)))
        }
        [System.IO.File]::AppendAllText($SealedManifestPath, ('{0},"{1}","{2}",file,{3},1,{4},MOVED' -f $RunStamp, $src, $dst, $len, $h) + "`n", (New-Object System.Text.UTF8Encoding($false)))
        Jrn 'SEALED_MOVE' $src $len ('-> ' + $dst + '; sha256=' + $h + '; opaque move by the clean-up script')
        $moved++
    }
    return $moved
}

function Eval-Worktree($w) {
    $name = [string]$w.name; $path = NP ([string]$w.path); $id = 'wt:' + $name
    if (-not (Test-Path -LiteralPath $path)) { return Add-Res $id 'worktree' $path 'GONE' 'not present (already removed)' 0 0 @() }
    $why = Get-ProtectedReason $path
    if ($why) { return Add-Res $id 'worktree' $path 'PROTECTED' $why 0 0 @() }
    $st = [string]$w.status
    $okRcpt = ($st -eq 'ok') -or ($st -eq 'commit_failed' -and "$($w.upload_exit)" -in @('', '0') -and "$($w.check_exit)" -in @('', '0'))
    if (-not $okRcpt) { return Add-Res $id 'worktree' $path 'RECEIPT_NOT_PASSED' "wtclean receipt status '$st'" 0 0 @() }
    if ([MC]::PathIsReparse($path)) { return Add-Res $id 'worktree' $path 'LINKS' 'the worktree root itself is a junction/symlink: manual handling' 0 0 @() }
    $ko = Get-KeepOverlap $path
    if ($ko) { return Add-Res $id 'worktree' $path 'KEEP_LOCAL' ("contains keep-local item '$ko' (Stage 2 -RemoveOld + registry switch first)") 0 0 @() }
    $pr = Find-ProcUsing $path
    if ($pr) { return Add-Res $id 'worktree' $path 'IN_USE' "running process: $pr" 0 0 @() }
    if (-not $SkipReferenceScan) {
        $sp = Get-RefSplit $path
        if ($sp.code.Count -gt 0) {
            $rh = @($sp.code | Select-Object -First 4 | ForEach-Object { "{0}:{1} -> {2}" -f $_.tree, ($_.file -replace '^.*\\', ''), $_.lit })
            return Add-Res $id 'worktree' $path 'REFERENCED' ("active code/config reference: " + ($rh -join ' | ')) 0 0 @()
        }
    }
    $cls = [string]$w.cls
    $isWt = ($cls -eq 'worktree')
    if ($isWt -and -not (Get-RegisteredWorktrees).ContainsKey((K $path))) { return Add-Res $id 'worktree' $path 'NOT_REGISTERED' 'not in git worktree list of the main repo' 0 0 @() }
    # --- no-follow walk: junctions / symlinks (never followed) ---
    $wlk = [MC]::Walk($path, $false)
    $links = @(Get-LinkInfo $path $wlk.Reparse)
    $script:WtLinkCache[(K $path)] = $links
    $linkSet = @{}; foreach ($li in $links) { $linkSet[([string]$li.rel).ToLowerInvariant()] = 1 }
    $bad = @($links | Where-Object { -not $_.target })
    if ($bad.Count -gt 0) { return Add-Res $id 'worktree' $path 'LINKS' ("reparse point with an unreadable target (not a plain junction/symlink): " + $bad[0].rel) $wlk.Bytes $wlk.Rel.Count @() }
    # --- git state ---
    $head = ''; $dirtyNote = ''
    if ($isWt) {
        $s = GitX $path @('status', '--porcelain=v1', '--untracked-files=no')
        if ($s.rc -ne 0) { return Add-Res $id 'worktree' $path 'GIT_ERROR' ("git status failed: " + $s.err.Trim()) $wlk.Bytes $wlk.Rel.Count @() }
        if ($s.out.Trim()) {
            $dl = @($s.out -split "`n" | ForEach-Object { $_.TrimEnd("`r") } | Where-Object { $_.Trim() })
            $pres = Test-DirtyPreserved $w $path $dl
            if (-not $pres) { return Add-Res $id 'worktree' $path 'DIRTY_TRACKED' ("{0} uncommitted tracked change(s), not preserved by a ref or by the receipt's archive" -f $dl.Count) $wlk.Bytes $wlk.Rel.Count @() }
            $dirtyNote = ("{0} uncommitted tracked change(s) accepted: {1}" -f $dl.Count, $pres)
        }
        $h = GitX $path @('rev-parse', 'HEAD'); $head = $h.out.Trim()
        $br = [string]$w.branch
        if ($br -and $br -ne 'HEAD' -and $br -notmatch '^refs/') { $br = 'refs/heads/' + $br }
        if ($br -and $br -ne 'HEAD') {
            $b = GitX $MainRepo @('rev-parse', '--verify', '--quiet', $br)
            if ($b.rc -ne 0) { return Add-Res $id 'worktree' $path 'BRANCH_MISSING' "branch $br does not exist in the main repo" $wlk.Bytes $wlk.Rel.Count @() }
            if ($b.out.Trim() -ne $head) {
                $anc = GitX $MainRepo @('merge-base', '--is-ancestor', $head, $br)
                if ($anc.rc -ne 0) { return Add-Res $id 'worktree' $path 'BRANCH_MISMATCH' "HEAD $($head.Substring(0,8)) is not contained in $br" $wlk.Bytes $wlk.Rel.Count @() }
            }
        } else {
            $c = GitX $MainRepo @('for-each-ref', '--contains', $head, '--count=1', '--format=%(refname)', 'refs/heads', 'refs/remotes', 'refs/tags')
            if (-not $c.out.Trim()) { return Add-Res $id 'worktree' $path 'COMMIT_UNREACHABLE' "detached HEAD $($head.Substring(0,8)) is contained in no branch/tag" $wlk.Bytes $wlk.Rel.Count @() }
        }
        $o1 = GitX $path @('ls-files', '--others', '--exclude-standard', '-z')
        $o2 = GitX $path @('ls-files', '--others', '--ignored', '--exclude-standard', '-z')
        $cur = New-Object System.Collections.ArrayList
        foreach ($o in @($o1.out, $o2.out)) { foreach ($f in ($o -split "`0")) { if ($f) { [void]$cur.Add($f) } } }
    } else {
        $cur = New-Object System.Collections.ArrayList
        $nested = New-Object System.Collections.ArrayList
        foreach ($r in $wlk.Rel) {
            if ($r -match '(^|/)\.git/') { [void]$nested.Add($r); continue }
            [void]$cur.Add($r)
        }
        if ($nested.Count -gt 0 -and -not $AllowNestedGit) {
            return Add-Res $id 'worktree' $path 'NESTED_GIT' ("{0} file(s) of a nested .git repository (e.g. {1}): its history is not backed up; -AllowNestedGit releases it (regenerable bundles)" -f $nested.Count, $nested[0]) $wlk.Bytes $wlk.Rel.Count @()
        }
        if (Test-Path -LiteralPath (Join-Path $path '.git')) {
            $h = GitX $path @('rev-parse', '--verify', '--quiet', 'HEAD')
            if ($h.rc -eq 0 -and -not $AllowNestedGit) { return Add-Res $id 'worktree' $path 'HAS_COMMITS' 'standalone git dir with its own commits (history not backed up)' $wlk.Bytes $wlk.Rel.Count @() }
        }
    }
    # git (core.symlinks=false) descends into junctions: drop everything below a link root from the file list
    if ($linkSet.Count -gt 0) {
        $keepCur = New-Object System.Collections.ArrayList
        foreach ($f in $cur) {
            $under = $false; $acc = ''
            foreach ($seg in ($f -split '/')) { $acc = if ($acc) { $acc + '/' + $seg } else { $seg }; if ($linkSet.ContainsKey($acc.ToLowerInvariant())) { $under = $true; break } }
            if (-not $under) { [void]$keepCur.Add($f) }
        }
        $cur = $keepCur
    }
    $list = Read-ListFile ([string]$w.list)
    if ($w.list -and $w.list_sha256) { if (([MC]::Sha256File([string]$w.list)) -ne $w.list_sha256) { return Add-Res $id 'worktree' $path 'RECEIPT_STALE' 'upload list changed since generation (regenerate)' $wlk.Bytes $wlk.Rel.Count @() } }
    # --- classify every untracked/ignored file that is not in the uploaded list ---
    $missing = New-Object System.Collections.ArrayList; $sensLeft = New-Object System.Collections.ArrayList
    foreach ($f in $cur) {
        if ($list.ContainsKey($f)) { continue }
        if ($f.StartsWith('.git/')) { continue }
        if ($BroadSens.IsMatch($f)) { [void]$sensLeft.Add($f) } else { [void]$missing.Add($f) }
    }
    if ($missing.Count -gt 0) { return Add-Res $id 'worktree' $path 'NEW_UNBACKED' ("{0} untracked/ignored file(s) not in the uploaded list, e.g. {1}" -f $missing.Count, $missing[0]) $wlk.Bytes $wlk.Rel.Count @() }
    $sealMoves = @(); $assumedN = 0
    if ($sensLeft.Count -gt 0 -and $AssumeSealedMoved) {
        $unc = @($sensLeft | Where-Object { -not (Test-SealedCovered (Join-Path $path ($_ -replace '/', '\'))) })
        $assumedN = $sensLeft.Count - $unc.Count
        $sensLeft = New-Object System.Collections.ArrayList; foreach ($u in $unc) { [void]$sensLeft.Add($u) }
    }
    if ($sensLeft.Count -gt 0 -and ($w.sealed_relocate -or $RelocateSensitive)) {
        # this worktree's own root is sensitive-named (MOVE_TO_SEALED holds it) or -RelocateSensitive was given: its untracked
        # sensitive-named files are moved unopened into sealed\ by this script right before the removal
        $sealMoves = @($sensLeft); $sensLeft = New-Object System.Collections.ArrayList
    }
    $relNote = ''
    if ($sensLeft.Count -gt 0) {
        $dupOk = $false
        if ($AllowSensitiveDup) {
            $dupOk = $true
            foreach ($f in $sensLeft) {
                $loc = Join-Path $path ($f -replace '/', '\'); $mn = Join-Path $MainRepo ($f -replace '/', '\')
                $ext = [System.IO.Path]::GetExtension($f).ToLowerInvariant()
                if (-not (Test-Path -LiteralPath $loc)) { continue }
                if ($ext -eq '.pyc' -and $f -match '(^|/)__pycache__/') { continue }   # regenerable byte-code cache
                if ($ext -notin @('.py', '.json', '.md', '.txt', '.pyc', '.ps1', '.sh', '.toml') -or -not (Test-Path -LiteralPath $mn -PathType Leaf)) { $dupOk = $false; break }
                if ((Get-Item -LiteralPath $loc -Force).Length -ne (Get-Item -LiteralPath $mn -Force).Length -or (Get-Item -LiteralPath $loc -Force).Length -gt 2MB) { $dupOk = $false; break }
            }
        }
        if (-not $dupOk) {
            return Add-Res $id 'worktree' $path 'SENSITIVE_SKIPPED' ("{0} sensitive-named untracked file(s) are still here and not on Drive (e.g. {1}): run MOVE_TO_SEALED.ps1 -Execute first (-AllowSensitiveDup releases pure code/protocol duplicates of the work tree)" -f $sensLeft.Count, $sensLeft[0]) $wlk.Bytes $wlk.Rel.Count @()
        }
        $relNote = ("released by -AllowSensitiveDup: {0} sensitive-named untracked file(s) are regenerable caches or size-identical copies in the work tree" -f $sensLeft.Count)
    } elseif (([int]$w.sens_count -gt 0) -and $sealMoves.Count -eq 0 -and -not $AssumeSealedMoved) {
        # the receipt says sensitive-named files were skipped and none is left: they must have been moved to sealed\ (record check)
        $unacc = New-Object System.Collections.ArrayList
        $names = @($w.sens)
        if ($names.Count -gt 0) {
            foreach ($n in $names) {
                $full = Join-Path $path (([string]$n) -replace '/', '\')
                if (Test-Path -LiteralPath $full) { continue }
                if ($list.ContainsKey([string]$n)) { continue }
                $underLink = $false; $acc = ''
                foreach ($seg in (([string]$n) -split '/')) { $acc = if ($acc) { $acc + '/' + $seg } else { $seg }; if ($linkSet.ContainsKey($acc.ToLowerInvariant())) { $underLink = $true; break } }
                if ($underLink) { continue }   # lived below a junction: not part of this worktree
                if (-not (Test-SealedCovered $full)) { [void]$unacc.Add([string]$n) }
            }
        } else {
            $any = $false; foreach ($sr in (Get-SealedMoveRows)) { if (UnderP $sr.src $path) { $any = $true; break } }
            if (-not $any) { [void]$unacc.Add('(no recorded names)') }
        }
        if ($unacc.Count -gt 0) {
            return Add-Res $id 'worktree' $path 'SENSITIVE_UNACCOUNTED' ("{0} recorded sensitive-named file(s) are gone from the worktree but have no 'MOVED' record in {1}, e.g. {2}: run MOVE_TO_SEALED.ps1 -Execute or account for them manually" -f $unacc.Count, $SealedManifestPath, $unacc[0]) $wlk.Bytes $wlk.Rel.Count @()
        }
    }
    # --- Drive: every present list file must exist under worktrees/<name>/ with equal size + MD5 ---
    $dest = ([string]$w.dest).TrimEnd('/')
    $present = New-Object System.Collections.ArrayList
    foreach ($f in $cur) { if ($list.ContainsKey($f)) { [void]$present.Add($f) } }
    if ([string]$w.method -eq 'tar.zst' -and $w.archive) {
        # tar.zst backup (wtclean pack_wt.py): the Drive proof is the archive object (size + md5 recorded in the
        # receipt and re-verified on Drive) plus its manifest; there are no per-file objects under $dest.
        $ar = $w.archive
        $e1 = Get-Remote ([string]$ar.remote)
        $e2 = Get-Remote ([string]$ar.manifest_remote)
        if ($null -eq $e1 -or [long]$e1.size -ne [long]$ar.size -or ([string]$e1.md5).ToLowerInvariant() -ne ([string]$ar.md5).ToLowerInvariant()) {
            return Add-Res $id 'worktree' $path 'DRIVE_UNVERIFIED' ("tar.zst archive absent/different on Drive: " + [string]$ar.remote) $wlk.Bytes $wlk.Rel.Count @()
        }
        if ($null -eq $e2 -or [long]$e2.size -ne [long]$ar.manifest_size -or ([string]$e2.md5).ToLowerInvariant() -ne ([string]$ar.manifest_md5).ToLowerInvariant()) {
            return Add-Res $id 'worktree' $path 'DRIVE_UNVERIFIED' ("tar.zst manifest absent/different on Drive: " + [string]$ar.manifest_remote) $wlk.Bytes $wlk.Rel.Count @()
        }
    } elseif ($present.Count -gt 0) {
        if (-not $dest) { return Add-Res $id 'worktree' $path 'DRIVE_UNVERIFIED' 'no Drive destination in receipt' $wlk.Bytes $wlk.Rel.Count @() }
        $badR = 0; $first = ''
        foreach ($f in $present) {
            $lp = Join-Path $path ($f -replace '/', '\')
            $ent = Get-Remote ($dest + '/' + $f)
            if ($null -eq $ent) { $badR++; if (-not $first) { $first = "$f (absent)" }; continue }
            $fi = New-Object System.IO.FileInfo ([MC]::LP($lp))
            if ($fi.Length -ne $ent.size) { $badR++; if (-not $first) { $first = "$f (size)" }; continue }
            if ((([MC]::Md5File($lp))) -ne $ent.md5) { $badR++; if (-not $first) { $first = "$f (md5)" } }
        }
        if ($badR -gt 0) { return Add-Res $id 'worktree' $path 'DRIVE_UNVERIFIED' ("{0} file(s) absent/different on Drive under {1}, e.g. {2}" -f $badR, $dest, $first) $wlk.Bytes $wlk.Rel.Count @() }
    }
    $notes = @()
    if ($relNote) { $notes += $relNote }
    if ($dirtyNote) { $notes += $dirtyNote }
    if ($assumedN -gt 0) { $notes += ("SIMULATION: {0} sensitive-named file(s) assumed already moved to sealed\ (plan {1})" -f $assumedN, $AssumeSealedMoved) }
    if ($sealMoves.Count -gt 0) { $notes += ("SEALED-RELOCATE: {0} sensitive-named untracked file(s) will be moved unopened to {1} before the removal" -f $sealMoves.Count, (Join-Path $SealedRoot ((DriveOf $path) + '\' + (NP $path).Substring(3)))) }
    if ($links.Count -gt 0) {
        $kinds = @($links | Group-Object kind | ForEach-Object { "{0} {1}" -f $_.Count, $_.Name })
        $notes += ("UNLINK-FIRST: {0} junction/symlink(s) are unlinked (link only) before the removal [{1}]" -f $links.Count, ($kinds -join ', '))
    }
    $o = Add-Res $id 'worktree' $path 'ELIGIBLE' ($notes -join '; ') $wlk.Bytes $wlk.Rel.Count @(($(if ($dest) { $dest } else { '(no untracked content)' }) + "; branch " + $w.branch + " @ " + $(if ($head) { $head.Substring(0, 8) } else { '-' })))
    $o | Add-Member -NotePropertyName Snap -NotePropertyValue @{ bytes = $wlk.Bytes; files = $wlk.Rel.Count; maxm = 0; links = $links; sealMoves = $sealMoves } -Force
    $o | Add-Member -NotePropertyName Wt -NotePropertyValue $w -Force
    return $o
}

# ---------------- Stage 1: deletion ----------------
function Prune-EmptyParents([string]$path) {
    $cur = Split-Path -Parent (NP $path)
    while ($cur) {
        $isC = $false
        foreach ($r in @($M.container_roots)) { if ((K $cur) -eq (K $r)) { $isC = $true; break } }
        if ($isC) { break }
        $ok = $false; foreach ($r in @($M.allowed_roots)) { if (UnderP $cur $r) { $ok = $true; break } }
        if (-not $ok) { break }
        if ([MC]::PathIsReparse($cur)) { break }
        if (-not [MC]::IsEmptyDir($cur)) { break }
        try { [MC]::DeleteEmptyDir($cur); Jrn 'PRUNE_EMPTY_DIR' $cur 0 'now-empty parent'; } catch { break }
        $cur = Split-Path -Parent $cur
    }
}
function Record-Removed($r, [long]$bytes) {
    $obj = [ordered]@{ ts = (Get-Date).ToUniversalTime().ToString('o'); path = $r.Path; type = $r.Type; bytes = $bytes; files = $r.Files; drive = @($r.Drive) }
    [System.IO.File]::AppendAllText($RemovedMap, (($obj | ConvertTo-Json -Compress -Depth 5) + "`n"), (New-Object System.Text.UTF8Encoding($false)))
}
function Revalidate($r) {
    # immediately before deletion: unchanged since evaluation (count, bytes, newest mtime)
    if (-not (Test-Path -LiteralPath $r.Path)) { return 'vanished' }
    if ($r.Type -eq 'worktree') { return $null }   # worktrees are re-evaluated from scratch right before this call; links are re-walked in Unlink-AllLinks
    $w = [MC]::Walk($r.Path, (-not $ReleaseProtectedNames))
    $nl = 0; if ($r.Snap.links) { $nl = @($r.Snap.links).Count }
    $lf = 0; if ($r.Snap.left) { $lf = [int]$r.Snap.left }
    if ($w.Reparse.Count -gt $nl -or $w.Sensitive.Count -ne $lf) { return 'links/protected entries appeared' }
    $maxm = 0; if ($w.MTicks.Count) { $maxm = ($w.MTicks | Measure-Object -Maximum).Maximum }
    if ($w.Bytes -ne $r.Snap.bytes -or $w.Rel.Count -ne $r.Snap.files -or $maxm -ne $r.Snap.maxm) { return 'changed since evaluation' }
    $skipWt2 = [bool]($r.Item -and $r.Item.special -and [string]$r.Item.special.kind -eq 'git_ignored_scratch')
    $why = Get-ProtectedReason $r.Path $skipWt2
    if ($why) { return $why }
    return $null
}
function Remove-Eligible($r) {
    $bad = Revalidate $r
    if ($bad) { Say ("  SKIP   {0}  ({1})" -f $r.Path, $bad) 'Yellow'; Jrn 'SKIP_AT_DELETE' $r.Path 0 $bad; $r.Status = 'CHANGED_AT_DELETE'; $r.Reason = $bad; return }
    try {
        if ($r.Type -eq 'worktree') {
            # 1. sensitive-named untracked files of a sensitive-named worktree root: opaque move into sealed\ (SHA-256 checked)
            if ($r.Snap.sealMoves -and @($r.Snap.sealMoves).Count -gt 0) {
                $nm = Move-ToSealed $r
                Say ("  SEALED  {0}: {1} sensitive-named file(s) moved unopened into {2}" -f $r.Path, $nm, $SealedRoot) 'Cyan'
            }
            # 2. unlink EVERY junction/symlink inside (link only; targets must survive).  git (core.symlinks=false) would
            #    otherwise descend into a junction and delete the target's contents.
            $nl = Unlink-AllLinks $r.Path
            if ($nl -gt 0) { Say ("  UNLINK  {0}: {1} junction/symlink(s) removed (link only), targets verified" -f $r.Path, $nl) 'DarkGray' }
            # 3. remove the tree
            if ([string]$r.Wt.cls -eq 'worktree') {
                $g = GitX $MainRepo @('worktree', 'remove', '--force', $r.Path)
                $gone = -not (Test-Path -LiteralPath $r.Path)
                if ($g.rc -ne 0 -or -not $gone) { throw ("git worktree remove failed (rc {0}): {1}" -f $g.rc, ($g.err -replace '\s+', ' ').Trim()) }
                $freed = [long]$r.Snap.bytes
            } else {
                $errs = New-Object 'System.Collections.Generic.List[string]'
                $freed = [Math]::Max([MC]::DeleteTree($r.Path, $errs), 0)
                if ($errs.Count -gt 0) { foreach ($e in $errs) { [void]$script:Errs.Add($e) }; Jrn 'DELETE_ERRORS' $r.Path $freed ($errs[0]) }
            }
        } else {
            $errs = New-Object 'System.Collections.Generic.List[string]'
            if ($r.Snap.links) {
                foreach ($rl in @($r.Snap.links)) {
                    $lpth = Join-Path $r.Path ($rl -replace '/', '\')
                    $tg = $null; try { $tg = (Get-Item -LiteralPath $lpth -Force).Target } catch { }
                    if ($tg -is [array]) { $tg = $tg[0] }
                    $tg = ([string]$tg) -replace '^\\\\\?\\', ''
                    if (-not $tg -or -not (UnderP $tg $r.Path)) { throw "link $lpth points outside the item ($tg): refusing" }
                    [MC]::UnlinkLink($lpth)
                    if (Test-Path -LiteralPath $lpth) { throw "could not unlink $lpth" }
                    Jrn 'UNLINKED' $lpth 0 ("junction inside the item removed (link only); target " + $tg)
                }
            }
            if ($r.Snap.partial) { $freed = [MC]::DeleteListed($r.Path, [string[]]@($r.Snap.rels), $errs) }
            else { $freed = [MC]::DeleteTree($r.Path, $errs) }
            if ($errs.Count -gt 0) { foreach ($e in $errs) { [void]$script:Errs.Add($e) }; Jrn 'DELETE_ERRORS' $r.Path $freed ($errs[0]) }
        }
        $script:Freed[$r.Drv] += $freed
        $r.Status = 'DELETED'
        Say ("  DELETED {0}  ({1})" -f $r.Path, (GB $freed)) 'Green'
        Jrn 'DELETED' $r.Path $freed ((@($r.Drive) -join ' ; '))
        Record-Removed $r $freed
        Prune-EmptyParents $r.Path
    } catch {
        Say ("  ERROR  {0}  {1}" -f $r.Path, $_.Exception.Message) 'Red'
        Jrn 'ERROR' $r.Path 0 $_.Exception.Message
        $r.Status = 'ERROR'; $r.Reason = $_.Exception.Message
        [void]$script:Errs.Add($r.Path + ': ' + $_.Exception.Message)
    }
}

function Test-Match($s, [string[]]$pats) { foreach ($p in $pats) { if ($p -and $s.IndexOf($p, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true } }; return $false }

function Print-Pending {
    $pi = $M.pending_info
    if ($null -eq $pi -or @($pi.lines).Count -eq 0) { return }
    Say '' ; Say '--- NOT YET SAFE: upload pending / failed (never deleted) ---' 'Cyan'
    foreach ($l in @($pi.lines)) { Say ('  ' + $l) 'DarkYellow' }
}

function Print-Holds($rows) {
    $groups = $rows | Group-Object Status | Sort-Object { - ($_.Group | Measure-Object Bytes -Sum).Sum }
    foreach ($g in $groups) {
        if ($g.Name -in @('ELIGIBLE', 'DELETED', 'GONE')) { continue }
        $sum = ($g.Group | Measure-Object Bytes -Sum).Sum
        Say ("  {0,-22} {1,4} item(s)  {2,12}" -f $g.Name, $g.Count, (GB $sum)) 'Yellow'
        $show = if ($ShowHolds) { $g.Group } else { $g.Group | Sort-Object Bytes -Descending | Select-Object -First 3 }
        foreach ($x in $show) { Say ("      {0}`n          {1}" -f $x.Path, ($x.Reason.Substring(0, [Math]::Min(230, $x.Reason.Length)))) 'DarkGray' }
    }
}

# Order of the worktree removals.  A worktree W whose junctions point INTO another worktree T is removed BEFORE T (so
# T's data is still there when W's links are unlinked and their targets verified).  A worktree that stays (held) and
# links into an eligible tree holds that tree back.  reorg3 (upload lists / manifests) is always last.
function Resolve-WorktreeOrder {
    $all = @($script:Res | Where-Object { $_.Type -eq 'worktree' -and $_.Status -notin @('GONE', 'DELETED') })
    $byPath = @{}; foreach ($w in @($M.worktrees)) { $byPath[(K ([string]$w.path))] = $w }
    foreach ($x in $all) {
        if ($null -eq $x.PSObject.Properties['Deps']) { $x | Add-Member -NotePropertyName Deps -NotePropertyValue (New-Object System.Collections.ArrayList) -Force }
        $k = K $x.Path
        if (-not $script:WtLinkCache.ContainsKey($k) -and $byPath.ContainsKey($k) -and [int]$byPath[$k].junctions -gt 0 -and (Test-Path -LiteralPath $x.Path)) {
            $wk = [MC]::Walk($x.Path, $false); $script:WtLinkCache[$k] = @(Get-LinkInfo $x.Path $wk.Reparse)
        }
    }
    function Links-Into($x, $t) {
        $lk = $script:WtLinkCache[(K $x.Path)]
        if (-not $lk) { return @() }
        return @($lk | Where-Object { $_.target -and (UnderP $_.target $t.Path) })
    }
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($t in @($all | Where-Object { $_.Status -eq 'ELIGIBLE' })) {
            foreach ($x in $all) {
                if ($x -eq $t -or $x.Status -eq 'ELIGIBLE' -or $x.Status -in @('GONE', 'DELETED')) { continue }
                $h = @(Links-Into $x $t)
                if ($h.Count -gt 0) {
                    $t.Status = 'LINKED_FROM_HELD'
                    $t.Reason = ("{0} (status {1}, stays) has junction(s) into this tree, e.g. {2} -> {3}: it must be retired first" -f $x.Id, $x.Status, $h[0].rel, $h[0].target)
                    Jrn 'LINKED_FROM_HELD' $t.Path $t.Bytes $t.Reason
                    $changed = $true; break
                }
            }
            if ($changed) { break }
        }
    }
    $elig = @($all | Where-Object { $_.Status -eq 'ELIGIBLE' })
    foreach ($t in $elig) {
        $t.Deps.Clear()
        foreach ($x in $elig) { if ($x -ne $t -and (@(Links-Into $x $t)).Count -gt 0) { [void]$t.Deps.Add($x) } }
    }
    $order = New-Object System.Collections.ArrayList; $done = @{}
    $pending = New-Object System.Collections.ArrayList; foreach ($t in $elig) { [void]$pending.Add($t) }
    while ($pending.Count -gt 0) {
        $ready = @($pending | Where-Object { (@($_.Deps | Where-Object { -not $done.ContainsKey($_.Id) })).Count -eq 0 })
        $cand = @($ready | Where-Object { -not $_.Wt.last })
        $anyNormal = (@($pending | Where-Object { -not $_.Wt.last })).Count -gt 0
        if ($cand.Count -eq 0 -and -not $anyNormal) { $cand = @($ready) }
        if ($cand.Count -eq 0) {
            foreach ($t in $pending) { $t.Status = 'LINK_CYCLE'; $t.Reason = 'junctions between eligible worktrees form a cycle: manual handling'; Jrn 'LINK_CYCLE' $t.Path $t.Bytes $t.Reason }
            break
        }
        $pick = $cand | Sort-Object Bytes | Select-Object -First 1
        [void]$order.Add($pick); $done[$pick.Id] = 1; [void]$pending.Remove($pick)
    }
    $script:WtOrder = @($order)
}
$script:WtOrder = @()

function Test-OthersRemain {
    # reorg3 holds the lists/manifests every worktree check reads: it is removed only when it is the last entry in E:\rd-wt
    $rd = [string]$P.rdwt
    $left = @(Get-ChildItem -LiteralPath $rd -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne (Split-Path -Leaf ([string]$P.reorg3)) })
    return $left
}

function Invoke-Stage1 {
    Say '=== STAGE 1: remove items and worktrees verified on Drive ===' 'Cyan'
    Say ("mode={0}  deep-hash={1}  reference-scan={2}  generated={3}" -f $(if ($EXEC) { 'EXECUTE' } else { 'REPORT-ONLY' }), $DEEP, $(if ($SkipReferenceScan) { 'OFF' } else { $ScanTrees }), $M.generated)
    if ($AssumeSealedMoved) { Say ("SIMULATION: sensitive-named files covered by {0} are treated as already moved to sealed\ (report only)" -f $AssumeSealedMoved) 'Yellow' }
    $freeBefore = @{}; foreach ($d in 'C', 'D', 'E', 'G') { $freeBefore[$d] = FreeBytes $d }
    $t0 = Get-Date
    if (-not $SkipReferenceScan) { Say 'scanning active code/config for path literals ...'; Scan-Literals; Say ("  {0} literal references found" -f $script:Lit.Count) }
    $items = @($M.items)
    if ($Only.Count) { $items = @($items | Where-Object { Test-Match ($_.id + ' ' + $_.path) $Only }) }
    if ($Skip.Count) { $items = @($items | Where-Object { -not (Test-Match ($_.id + ' ' + $_.path) $Skip) }) }
    $n = 0
    if (-not $SkipItems) {
        Say ("evaluating {0} backup-verified items (Drive listing is re-read live) ..." -f $items.Count)
        foreach ($it in ($items | Sort-Object { $_.path })) {
            $n++
            try { [void](Eval-Item $it) } catch { [void](Add-Res ([string]$it.id) 'item' ([string]$it.path) 'ERROR' $_.Exception.Message 0 0 @()) ; [void]$script:Errs.Add([string]$it.path + ': ' + $_.Exception.Message) }
            if ($n % 50 -eq 0) { Say ("  ... {0}/{1}  ({2:N0} s)" -f $n, $items.Count, ((Get-Date) - $t0).TotalSeconds) 'DarkGray' }
        }
    }
    # an eligible item inside another eligible item is covered by the parent: count/delete it once
    $chosenPaths = New-Object System.Collections.ArrayList
    foreach ($r0 in (@($script:Res | Where-Object { $_.Status -eq 'ELIGIBLE' -and $_.Type -eq 'item' }) | Sort-Object { $_.Path.Length })) {
        $par = $null
        foreach ($c0 in $chosenPaths) { if (UnderP $r0.Path $c0) { $par = $c0; break } }
        if ($par) { $r0.Status = 'NESTED'; $r0.Reason = "covered by eligible parent $par"; Jrn 'NESTED' $r0.Path $r0.Bytes $r0.Reason }
        else { [void]$chosenPaths.Add($r0.Path) }
    }
    if (-not $SkipWorktrees) {
        $wts = @($M.worktrees)
        if ($Only.Count) { $wts = @($wts | Where-Object { Test-Match ($_.name + ' ' + $_.path) $Only }) }
        if ($Skip.Count) { $wts = @($wts | Where-Object { -not (Test-Match ($_.name + ' ' + $_.path) $Skip) }) }
        Say ("evaluating {0} worktrees with wtclean receipts ..." -f $wts.Count)
        $n = 0
        foreach ($w in ($wts | Sort-Object { $_.name })) {
            $n++
            try { [void](Eval-Worktree $w) } catch { [void](Add-Res ('wt:' + $w.name) 'worktree' ([string]$w.path) 'ERROR' $_.Exception.Message 0 0 @()); [void]$script:Errs.Add([string]$w.path + ': ' + $_.Exception.Message) }
            if ($n % 25 -eq 0) { Say ("  ... {0}/{1}  ({2:N0} s)" -f $n, $wts.Count, ((Get-Date) - $t0).TotalSeconds) 'DarkGray' }
        }
        foreach ($u in @($M.unreceipted_dirs)) {
            if ($Only.Count -and -not (Test-Match ($u.name + ' ' + $u.path) $Only)) { continue }
            if ($Skip.Count -and (Test-Match ($u.name + ' ' + $u.path) $Skip)) { continue }
            if (Test-Path -LiteralPath ([string]$u.path)) { [void](Add-Res ('wt:' + $u.name) 'worktree' ([string]$u.path) 'NO_RECEIPT' 'no ok wtclean receipt for this directory (not backed up): never removed by this script' 0 0 @()) }
        }
        Resolve-WorktreeOrder
        $ordAll = @($script:WtOrder | ForEach-Object { $_.Id -replace '^wt:', '' })
        Jrn 'ORDER' 'worktrees' $ordAll.Count ($ordAll -join ' ')
        foreach ($t in @($script:WtOrder | Where-Object { @($_.Deps).Count -gt 0 })) { Jrn 'ORDER_DEPENDS' $t.Path 0 ('removed after: ' + ((@($t.Deps) | ForEach-Object { $_.Id -replace '^wt:', '' }) -join ', ')) }
        $lastR = @($script:Res | Where-Object { $_.Type -eq 'worktree' -and $_.Status -eq 'ELIGIBLE' -and $_.Wt.last })
        foreach ($lr in $lastR) { $lr.Reason = ("removed LAST (and only if E:\rd-wt holds nothing else); {0} entr(y/ies) there now. " -f (@(Test-OthersRemain)).Count) + $lr.Reason }
    }
    $elig = @($script:Res | Where-Object { $_.Status -eq 'ELIGIBLE' })
    Say ''
    Say '--- ELIGIBLE (Drive copy verified live) ---' 'Green'
    foreach ($r in ($elig | Sort-Object Drv, @{ Expression = 'Bytes'; Descending = $true })) {
        if ($elig.Count -gt 60 -and $r.Bytes -lt 50MB -and -not $ShowHolds) { continue }
        Say ("  {0} {1,-9} {2,11}  {3}" -f $r.Drv, $r.Type, (GB $r.Bytes), $r.Path) 'Green'
    }
    if ($elig.Count -gt 60 -and -not $ShowHolds) { Say ("  (items < 50 MB not listed individually; {0} eligible items in total; -ShowHolds lists everything)" -f $elig.Count) 'DarkGray' }
    $withLinks = @($elig | Where-Object { $_.Type -eq 'worktree' -and $_.Snap.links -and @($_.Snap.links).Count -gt 0 })
    $withSeal = @($elig | Where-Object { $_.Type -eq 'worktree' -and $_.Snap.sealMoves -and @($_.Snap.sealMoves).Count -gt 0 })
    if ($withLinks.Count -gt 0) {
        Say ''
        Say ("--- junction/symlink handling: {0} eligible worktree(s) are unlinked first (link only), {1} link(s) in total ---" -f $withLinks.Count, (($withLinks | ForEach-Object { @($_.Snap.links).Count } | Measure-Object -Sum).Sum)) 'Cyan'
        foreach ($r in $withLinks) {
            $kinds = @($r.Snap.links | Group-Object kind | ForEach-Object { "{0} {1}" -f $_.Count, $_.Name })
            Say ("  {0,-34} {1}" -f $r.Id, ($kinds -join ', ')) 'DarkGray'
        }
        foreach ($t in @($script:WtOrder | Where-Object { @($_.Deps).Count -gt 0 })) { Say ("  {0,-30} is removed AFTER: {1}   (their junctions point into it)" -f $t.Id, ((@($t.Deps) | ForEach-Object { $_.Id }) -join ', ')) 'DarkGray' }
    }
    if ($withSeal.Count -gt 0) {
        Say ''
        foreach ($r in $withSeal) { Say ("--- sealed relocation: {0} moves {1} sensitive-named untracked file(s) into {2} before the removal" -f $r.Id, @($r.Snap.sealMoves).Count, $SealedRoot) 'Cyan' }
    }
    Say ''
    Say '--- HELD (not removed) ---' 'Yellow'
    Print-Holds $script:Res
    Print-Pending
    Say ''
    Say '--- WOULD FREE (eligible now) ---' 'Cyan'
    foreach ($d in 'C', 'D', 'E', 'G') {
        $sel = @($elig | Where-Object { $_.Drv -eq $d })
        Say ("  {0}: {1,12}  in {2} item(s)   (free now {3})" -f $d, (GB (($sel | Measure-Object Bytes -Sum).Sum)), $sel.Count, (GB $freeBefore[$d])) 'Cyan'
    }
    if ($EXEC -and $elig.Count -gt 0) {
        $tot = ($elig | Measure-Object Bytes -Sum).Sum
        Say ''
        Say ("About to DELETE {0} items/worktrees ({1}) after one more live Drive re-check; {2} junction/symlink(s) will be unlinked (link only) first. Journal: {3}" -f $elig.Count, (GB $tot), (($withLinks | ForEach-Object { @($_.Snap.links).Count } | Measure-Object -Sum).Sum), $JournalPath) 'Red'
        if (-not $Yes) { $ans = Read-Host 'Type DELETE to proceed'; if ($ans -cne 'DELETE') { Say 'aborted - nothing deleted.' 'Yellow'; Finish-Stage 1 $freeBefore; return } }
        Say 're-reading the Drive listings (fresh) and re-verifying each item immediately before deleting ...' 'Cyan'
        Reset-Remote
        $script:ManCache = @{}
        $script:DeepNow = $false   # content was hashed minutes ago in this run; the re-check re-reads Drive + size/mtime snapshot
        # 1. items (smallest first per drive)
        foreach ($r in (@($elig | Where-Object { $_.Type -eq 'item' }) | Sort-Object Drv, Bytes)) {
            $fresh = Eval-Item $r.Item
            [void]$script:Res.Remove($fresh)
            if ($fresh.Status -ne 'ELIGIBLE') { Say ("  SKIP   {0}  (re-check: {1} {2})" -f $r.Path, $fresh.Status, $fresh.Reason) 'Yellow'; $r.Status = 'CHANGED_AT_DELETE'; $r.Reason = $fresh.Status + ': ' + $fresh.Reason; continue }
            $r.Snap = $fresh.Snap; $r.Drive = $fresh.Drive
            Remove-Eligible $r
        }
        # 2. worktrees in dependency order (dependents first, reorg3 last)
        foreach ($r in @($script:WtOrder)) {
            if ($r.Status -ne 'ELIGIBLE') { continue }
            if ($r.Wt.last) {
                $left = @(Test-OthersRemain)
                if ($left.Count -gt 0) { Say ("  HOLD   {0}  ({1} other entr(y/ies) remain in E:\rd-wt, e.g. {2}: reorg3 is removed last)" -f $r.Path, $left.Count, $left[0].Name) 'Yellow'; $r.Status = 'HELD_LAST'; $r.Reason = "$($left.Count) other entries remain in E:\rd-wt (e.g. $($left[0].Name)); reorg3 holds the lists their checks read"; Jrn 'HELD_LAST' $r.Path 0 $r.Reason; continue }
            }
            $depLeft = @($r.Deps | Where-Object { Test-Path -LiteralPath $_.Path })
            if ($depLeft.Count -gt 0) { Say ("  HOLD   {0}  (worktree {1} still exists and has junctions into it)" -f $r.Path, $depLeft[0].Path) 'Yellow'; $r.Status = 'LINKED_FROM_PRESENT'; $r.Reason = "dependent worktree $($depLeft[0].Id) still exists (not retired in this run)"; Jrn 'LINKED_FROM_PRESENT' $r.Path 0 $r.Reason; continue }
            $fresh = Eval-Worktree $r.Wt
            [void]$script:Res.Remove($fresh)
            if ($fresh.Status -ne 'ELIGIBLE') { Say ("  SKIP   {0}  (re-check: {1} {2})" -f $r.Path, $fresh.Status, $fresh.Reason) 'Yellow'; $r.Status = 'CHANGED_AT_DELETE'; $r.Reason = $fresh.Status + ': ' + $fresh.Reason; continue }
            $r.Snap = $fresh.Snap
            Remove-Eligible $r
        }
    }
    if (-not $EXEC) {
        Say ''
        Say 'To act on the ELIGIBLE list (prompts for a typed DELETE; re-verifies Drive live right before each removal):' 'Cyan'
        Say ('  powershell -NoProfile -ExecutionPolicy Bypass -File ' + (Join-Path $OutDir 'MORNING_CLEANUP.ps1') + ' -Stage 1 -Execute') 'White'
        Say '  (flags: -RetireReorg removes reorg3 last; -RemoveAroundProtected / -AllowSensitiveDup / -AllowNestedGit release the partial cases described in the header)' 'DarkGray'
    }
    Finish-Stage 1 $freeBefore
}

function Finish-Stage([int]$st, $freeBefore) {
    $stamp = $RunStamp
    $rep = [ordered]@{ stage = $st; mode = $(if ($EXEC) { 'execute' } else { 'report' }); generated = $M.generated; run = $stamp; deep = $DEEP; results = @($script:Res | ForEach-Object { [ordered]@{ id = $_.Id; type = $_.Type; path = $_.Path; drive = $_.Drv; status = $_.Status; reason = $_.Reason; bytes = $_.Bytes; files = $_.Files; drive_ref = @($_.Drive) } }); errors = @($script:Errs) }
    $f1 = Join-Path $OutDir ("report_stage{0}_{1}.json" -f $st, $stamp)
    if ($script:S2) { $rep['stage2'] = @($script:S2) }
    if (-not $script:NoReport) {
        [System.IO.File]::WriteAllText($f1, ($rep | ConvertTo-Json -Depth 6), (New-Object System.Text.UTF8Encoding($false)))
        Copy-Item -LiteralPath $f1 -Destination (Join-Path $OutDir ("report_stage{0}_latest.json" -f $st)) -Force
    }
    if ($EXEC -and -not $script:NoOffloadMap -and $st -eq 1 -and @($script:Res | Where-Object { $_.Status -eq 'DELETED' }).Count -gt 0) {
        # record every removed path with its Drive copy so the experiment ledger reports "offloaded", not "missing"
        $py = 'C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe'; $om = Join-Path $MainRepo 'tools\offload_map.py'
        if ((Test-Path -LiteralPath $py) -and (Test-Path -LiteralPath $om)) {
            $u = Invoke-Proc $py @($om, 'update', $JournalPath) 300 $MainRepo
            Say ("offload map: " + ($u.out + $u.err).Trim()) $(if ($u.rc -eq 0) { 'DarkGray' } else { 'Red' })
        } else { Say 'offload map NOT updated: run  python tools/offload_map.py update <journal>  in E:\research-dense' 'Yellow' }
    }
    Say ''
    Say ("=== SUMMARY stage {0} ({1}) ===" -f $st, $(if ($EXEC) { 'EXECUTE' } else { 'REPORT-ONLY - nothing was changed' })) 'Cyan'
    $c = $script:Res | Group-Object Status | Sort-Object Name
    Say ("status counts: " + (($c | ForEach-Object { "{0}={1}" -f $_.Name, $_.Count }) -join '  '))
    Say ("report: $f1") 'DarkGray'
    Say ("journal: $JournalPath") 'DarkGray'
    if ($script:Errs.Count) { Say ("{0} error(s); first: {1}" -f $script:Errs.Count, $script:Errs[0]) 'Red' }
    $line = 'BYTES ACTUALLY FREED (measured file sizes removed by this run):'
    foreach ($d in 'C', 'D', 'E', 'G') {
        $now = FreeBytes $d
        $delta = if ($freeBefore[$d] -ge 0 -and $now -ge 0) { $now - $freeBefore[$d] } else { 0 }
        $line += ("`n   {0}: removed {1,15:N0} bytes ({2})   free-space delta {3,15:N0} bytes   free now {4}" -f $d, $script:Freed[$d], (GB $script:Freed[$d]), $delta, (GB $now))
    }
    Say $line 'Magenta'
    Jrn 'RUN_END' ("stage$st") (($script:Freed.Values | Measure-Object -Sum).Sum) ("freed C={0} D={1} E={2} G={3}" -f $script:Freed.C, $script:Freed.D, $script:Freed.E, $script:Freed.G)
}

# =====================================================================================================
#  Stage 2: keep-local -> data_cache (copy + SHA-256), then optional removal of the old copy
# =====================================================================================================
function Read-Sha256Sums([string]$file) {
    $map = @{}
    if (-not (Test-Path -LiteralPath $file)) { return $map }
    foreach ($l in [System.IO.File]::ReadLines($file)) {
        $mf = [regex]::Match($l.TrimEnd("`r"), '^([0-9a-fA-F]{64})\s+\*?(.+)$')
        if ($mf.Success) { $map[($mf.Groups[2].Value.Trim() -replace '^\./', '' -replace '\\', '/')] = $mf.Groups[1].Value.ToLowerInvariant() }
    }
    return $map
}
function Get-CacheReceipt([string]$dest) {
    $p = Join-Path $dest 'CACHE_RECEIPT.json'
    if (Test-Path -LiteralPath $p) { try { return (Get-Content -LiteralPath $p -Raw | ConvertFrom-Json) } catch { return $null } }
    return $null
}
function Test-DrivePackProof($k, $proof) {
    # returns @{ ok; why; desc }
    switch ([string]$proof.t) {
        'dir_md5' {
            if (-not $k.manifest) { return @{ ok = $false; why = 'no md5 manifest to compare'; desc = $proof.remote } }
            $mf = Read-Manifest ([string]$k.manifest) ([string]$k.manifest_sha256) ''
            if (-not $mf.ok) { return @{ ok = $false; why = $mf.why; desc = $proof.remote } }
            $r = Rcl @('lsf', '-R', '--files-only', '--format', 'psh', '--hash', 'md5', '--separator', '|', [string]$proof.remote) 1800
            if ($r.rc -ne 0) { return @{ ok = $false; why = 'remote dir not found'; desc = $proof.remote } }
            $rem = @{}
            foreach ($line in ($r.out -split "`n")) { $f = $line.TrimEnd("`r").Split('|'); if ($f.Count -ge 3) { $rem[($f[0..($f.Count - 3)] -join '|')] = @{ size = [long]$f[$f.Count - 2]; md5 = $f[$f.Count - 1].ToLowerInvariant() } } }
            $bad = 0; $first = ''
            foreach ($rel in $mf.map.Keys) { $e = $mf.map[$rel]; $x = $rem[$rel]; if ($null -eq $x -or $x.size -ne $e.size -or $x.md5 -ne $e.md5) { $bad++; if (-not $first) { $first = $rel } } }
            if ($bad) { return @{ ok = $false; why = "$bad of $($mf.n) member(s) absent/different (e.g. $first)"; desc = $proof.remote } }
            return @{ ok = $true; why = ''; desc = ("{0} ({1} files size+MD5 match)" -f $proof.remote, $mf.n) }
        }
        'tar_parts' {
            $base = ([string]$proof.remote).TrimEnd('/')
            $sums = Rcl @('cat', ($base + '/' + $proof.parts_sums))
            if ($sums.rc -ne 0) { return @{ ok = $false; why = 'SHA256SUMS not found on Drive'; desc = $base } }
            $want = @{}
            foreach ($l in ($sums.out -split "`n")) { $mm = [regex]::Match($l.TrimEnd("`r"), '^([0-9a-fA-F]{64})\s+\*?(.+)$'); if ($mm.Success) { $want[$mm.Groups[2].Value.Trim()] = $mm.Groups[1].Value.ToLowerInvariant() } }
            $ls = Rcl @('lsf', '--files-only', '--format', 'psh', '--hash', 'sha256', '--separator', '|', $base)
            if ($ls.rc -ne 0) { return @{ ok = $false; why = 'cannot list Drive parts'; desc = $base } }
            $have = @{}; foreach ($line in ($ls.out -split "`n")) { $f = $line.TrimEnd("`r").Split('|'); if ($f.Count -ge 3) { $have[$f[0]] = $f[2].ToLowerInvariant() } }
            foreach ($n in $want.Keys) { if ($have[$n] -ne $want[$n]) { return @{ ok = $false; why = "part $n missing or sha256 differs on Drive"; desc = $base } } }
            $cm = Rcl @('cat', ($base + '/' + $proof.content_manifest))
            if ($cm.rc -ne 0) { return @{ ok = $false; why = 'content manifest not found'; desc = $base } }
            $cj = $cm.out | ConvertFrom-Json
            $cache = Join-Path $CacheRoot $k.cache
            $rc = Get-CacheReceipt $cache
            $cnt = 0; $cmp = 0; $bad = 0; $first = ''
            foreach ($pp in $cj.sha256.PSObject.Properties) {
                $cnt++
                $got = $null
                if ($rc) { $e = @($rc.files) | Where-Object { $_.rel -eq $pp.Name } | Select-Object -First 1; if ($e) { $got = $e.sha256 } }
                if (-not $got) { continue }
                $cmp++
                if ($got -ne ([string]$pp.Value).ToLowerInvariant()) { $bad++; if (-not $first) { $first = $pp.Name } }
            }
            if ($bad) { return @{ ok = $false; why = "$bad file(s) differ from the archived content (e.g. $first)"; desc = $base } }
            if ($cmp -lt $cnt) { return @{ ok = $false; why = ("Drive parts verified (sha256 = SHA256SUMS) but the archived content ({0} files) is compared with the cache only after Stage 2 copied it ({1}/{0} compared)" -f $cnt, $cmp); desc = $base } }
            return @{ ok = $true; why = ''; desc = ("{0} (tar.zst parts sha256 match SHA256SUMS; all {1} archived members' sha256 equal the cache copy)" -f $base, $cnt) }
        }
        'archive_cover' {
            $unit = (NP ($MainRepo + '\' + ([string]$proof.unit -replace '/', '\')))
            $it = @($M.items) | Where-Object { (K $_.path) -eq (K $unit) } | Select-Object -First 1
            if ($null -eq $it) { return @{ ok = $false; why = "no verified Drive archive covers $unit yet (upload pending)"; desc = [string]$proof.unit } }
            $pf = @($it.proofs) | Where-Object { $_.t -eq 'archive' } | Select-Object -First 1
            $exp = @{}
            foreach ($part in @($pf.parts)) { $mf = Read-Manifest ([string]$part.manifest) ([string]$part.manifest_sha256) ''; if ($mf.ok) { foreach ($kk in $mf.map.Keys) { $exp[$kk] = $mf.map[$kk] } }
                $ent = Get-Remote ([string]$part.remote); if ($null -eq $ent -or $ent.md5 -ne ([string]$part.md5).ToLowerInvariant()) { return @{ ok = $false; why = 'archive not matching on Drive'; desc = [string]$part.remote } } }
            $w = [MC]::Walk($k.source, $true)
            $relBase = ((NP $k.source).Substring((NP $unit).Length).TrimStart('\') -replace '\\', '/')
            $bad = 0; $first = ''
            $rcv = Get-CacheReceipt (Join-Path $CacheRoot $k.cache)
            $cm = @{}; if ($rcv) { foreach ($ce in @($rcv.files)) { $cm[[string]$ce.rel] = $ce } }
            for ($i = 0; $i -lt $w.Rel.Count; $i++) {
                $key = $relBase + '/' + $w.Rel[$i]
                $e = $exp[$key]
                if ($null -eq $e -or $e.size -ne $w.Size[$i]) { $bad++; if (-not $first) { $first = $w.Rel[$i] }; continue }
                $c = $cm[$w.Rel[$i]]
                if ($c -and $c.md5 -ne $e.md5) { $bad++; if (-not $first) { $first = $w.Rel[$i] } }
            }
            if ($bad) { return @{ ok = $false; why = "$bad file(s) not covered by the verified archive (e.g. $first)"; desc = [string]$proof.unit } }
            return @{ ok = $true; why = ''; desc = ("covered by Drive archive of {0} ({1} files size match)" -f $proof.unit, $w.Rel.Count) }
        }
        'dir_listing' {
            $r = Rcl @('lsf', '-R', '--files-only', '--format', 'psh', '--hash', 'md5', '--separator', '|', [string]$proof.remote) 900
            if ($r.rc -ne 0) { return @{ ok = $false; why = 'remote dir not found'; desc = $proof.remote } }
            $rem = @{}; foreach ($line in ($r.out -split "`n")) { $f = $line.TrimEnd("`r").Split('|'); if ($f.Count -ge 3) { $rem[($f[0..($f.Count - 3)] -join '|')] = @{ size = [long]$f[$f.Count - 2]; md5 = $f[$f.Count - 1].ToLowerInvariant() } } }
            $w = [MC]::Walk($k.source, $true); $bad = 0; $first = ''
            for ($i = 0; $i -lt $w.Rel.Count; $i++) {
                $x = $rem[$w.Rel[$i]]
                if ($null -eq $x -or $x.size -ne $w.Size[$i]) { $bad++; if (-not $first) { $first = $w.Rel[$i] }; continue }
                if ($EXEC -and ([MC]::Md5File((Join-Path $k.source $w.Rel[$i]))) -ne $x.md5) { $bad++; if (-not $first) { $first = $w.Rel[$i] } }
            }
            if ($bad) { return @{ ok = $false; why = "$bad of $($w.Rel.Count) file(s) absent/different on Drive (e.g. $first)"; desc = $proof.remote } }
            return @{ ok = $true; why = ''; desc = ("{0} ({1} files name+size{2} match)" -f $proof.remote, $w.Rel.Count, $(if ($EXEC) { '+MD5' } else { '' })) }
        }
        'git_tracked' {
            $l = GitX ([string]$proof.repo) @('ls-files', '-z', '--', [string]$proof.rel)
            if ($l.out.Trim("`0", "`n", ' ')) { return @{ ok = $true; why = ''; desc = ("tracked in git: " + $proof.repo + ' ' + $proof.rel) } }
            return @{ ok = $false; why = 'not tracked in git'; desc = $proof.rel }
        }
        'wt_tar' {
            # the keep-local source lives inside a worktree that has a verified tar.zst on Drive: every source file must be a
            # member of that archive (same size) and the cache copy's SHA-256 must equal the archived member's SHA-256
            $wt = @($M.worktrees) | Where-Object { $_.name -eq [string]$proof.worktree } | Select-Object -First 1
            if ($null -eq $wt -or [string]$wt.method -ne 'tar.zst' -or -not $wt.archive) { return @{ ok = $false; why = ("no tar.zst receipt for worktree " + $proof.worktree); desc = [string]$proof.worktree } }
            $ar = $wt.archive
            $e1 = Get-Remote ([string]$ar.remote); $e2 = Get-Remote ([string]$ar.manifest_remote)
            if ($null -eq $e1 -or [long]$e1.size -ne [long]$ar.size -or ([string]$e1.md5).ToLowerInvariant() -ne ([string]$ar.md5).ToLowerInvariant()) { return @{ ok = $false; why = 'worktree tar.zst absent/different on Drive'; desc = [string]$ar.remote } }
            if ($null -eq $e2 -or [long]$e2.size -ne [long]$ar.manifest_size -or ([string]$e2.md5).ToLowerInvariant() -ne ([string]$ar.manifest_md5).ToLowerInvariant()) { return @{ ok = $false; why = 'worktree tar.zst manifest absent/different on Drive'; desc = [string]$ar.manifest_remote } }
            $mem = Get-TarMembers ([string]$ar.manifest_local)
            if ($null -eq $mem) { return @{ ok = $false; why = ('archive member manifest unreadable: ' + $ar.manifest_local); desc = [string]$ar.remote } }
            $w = [MC]::Walk($k.source, $true)
            $relBase = ([string]$proof.rel).TrimEnd('/') + '/'
            $rcv = Get-CacheReceipt (Join-Path $CacheRoot $k.cache); $cm = @{}
            if ($rcv) { foreach ($ce in @($rcv.files)) { $cm[[string]$ce.rel] = $ce } }
            $bad = 0; $first = ''; $cmp = 0
            for ($i = 0; $i -lt $w.Rel.Count; $i++) {
                $mf = $mem[$relBase + $w.Rel[$i]]
                if ($null -eq $mf -or [long]$mf['size'] -ne $w.Size[$i]) { $bad++; if (-not $first) { $first = $w.Rel[$i] }; continue }
                $c = $cm[$w.Rel[$i]]
                if ($c) { $cmp++; if (([string]$c.sha256).ToLowerInvariant() -ne ([string]$mf['sha256']).ToLowerInvariant()) { $bad++; if (-not $first) { $first = $w.Rel[$i] } } }
            }
            if ($bad) { return @{ ok = $false; why = "$bad file(s) not in the archive / different (e.g. $first)"; desc = [string]$ar.remote } }
            if ($cmp -lt $w.Rel.Count) { return @{ ok = $false; why = ("cache copy not receipted for all files yet ({0}/{1} compared)" -f $cmp, $w.Rel.Count); desc = [string]$ar.remote } }
            return @{ ok = $true; why = ''; desc = ("{0} (worktree {1}: all {2} files are members of the verified tar.zst; cache SHA-256 equals the archived SHA-256)" -f $ar.remote, $proof.worktree, $w.Rel.Count) }
        }
    }
    return @{ ok = $false; why = 'unknown proof'; desc = '' }
}

function Invoke-Stage2 {
    Say '=== STAGE 2: keep-local hot data -> data_cache (copy + SHA-256; never move) ===' 'Cyan'
    Say ("mode={0}  cache={1}  margin={2} GiB" -f $(if ($EXEC) { 'EXECUTE' } else { 'REPORT-ONLY' }), $CacheRoot, $MarginGiB)
    $freeBefore = @{}; foreach ($d in 'C', 'D', 'E', 'G') { $freeBefore[$d] = FreeBytes $d }
    $keeps = @($M.keep_local)
    if ($Only.Count) { $keeps = @($keeps | Where-Object { Test-Match ($_.name + ' ' + $_.cache) $Only }) }
    $order = @($keeps | Sort-Object { if (Test-Path -LiteralPath $_.source) { (Get-ChildItem -LiteralPath $_.source -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum } else { 0 } })
    if (-not $SkipReferenceScan -and $RemoveOld) { Scan-Literals }
    $cacheDrive = DriveOf $CacheRoot
    $margin = [long]$MarginGiB * 1GB
    $summary = New-Object System.Collections.ArrayList
    $planned = [long]0
    if ($EXEC -and -not $Yes) {
        $ans = Read-Host 'Stage 2 COPIES keep-local items into data_cache (never moves/deletes anything). Type COPY to proceed'
        if ($ans -cne 'COPY') { Say 'aborted - nothing copied.' 'Yellow'; Finish-Stage 2 $freeBefore; return }
    }
    foreach ($k in $order) {
        $src = NP ([string]$k.source); $dest = NP (Join-Path $CacheRoot ([string]$k.cache))
        Say ''
        Say ("[{0}]  {1}  ->  {2}" -f $k.name, $src, $dest) 'White'
        if (-not (Test-Path -LiteralPath $src)) {
            $rc = Get-CacheReceipt $dest
            Say ("  source gone; cache receipt: {0}" -f $(if ($rc) { 'present (already migrated)' } else { 'MISSING - nothing to copy' })) 'Yellow'
            [void]$summary.Add([pscustomobject]@{ Name = $k.name; State = $(if ($rc) { 'ALREADY_MIGRATED' } else { 'SOURCE_MISSING' }); Bytes = 0; Note = '' })
            continue
        }
        $w = [MC]::Walk($src, $true)
        if ($w.Reparse.Count) { Say "  junction inside source: manual handling" 'Red'; [void]$summary.Add([pscustomobject]@{ Name = $k.name; State = 'LINKS'; Bytes = $w.Bytes; Note = '' }); continue }
        $skipRel = @($w.Rel | Where-Object { $_ -like '*.part' -or $_ -like '*.tmp' })
        $rels = @($w.Rel | Where-Object { $_ -notlike '*.part' -and $_ -notlike '*.tmp' })
        # reference hashes
        $refMd5 = @{}; $refSha = @{}
        if ($k.manifest) { $mf = Read-Manifest ([string]$k.manifest) ([string]$k.manifest_sha256) ''; if ($mf.ok) { foreach ($kk in $mf.map.Keys) { $refMd5[$kk] = $mf.map[$kk] } } else { Say ("  WARNING: " + $mf.why) 'Yellow' } }
        if ($k.ref_sha256sums) {
            foreach ($cand in @((Join-Path $src $k.ref_sha256sums), (Join-Path (Split-Path -Parent $src) $k.ref_sha256sums))) { if (Test-Path -LiteralPath $cand) { $refSha = Read-Sha256Sums $cand; break } }
        }
        $incomplete = $false; $notes = @()
        if ($refMd5.Count) {
            $missingSrc = @($refMd5.Keys | Where-Object { $rels -notcontains $_ })
            if ($missingSrc.Count) { $incomplete = $true; $notes += ("{0} manifest member(s) missing in source" -f $missingSrc.Count) }
            $szBad = 0; for ($i = 0; $i -lt $w.Rel.Count; $i++) { $e = $refMd5[$w.Rel[$i]]; if ($e -and $e.size -ne $w.Size[$i]) { $szBad++ } }
            if ($szBad) { $incomplete = $true; $notes += ("{0} file(s) differ in size from the md5 manifest" -f $szBad) }
        }
        if ($refSha.Count -and -not $refMd5.Count) {
            $missingSrc = @($refSha.Keys | Where-Object { $rels -notcontains $_ })
            if ($missingSrc.Count) { $incomplete = $true; $notes += ("{0}/{1} member(s) of {2} not present yet" -f $missingSrc.Count, $refSha.Count, $k.ref_sha256sums) }
        }
        if ($skipRel.Count) { $incomplete = $true; $notes += ("{0} in-progress file(s) (*.part)" -f $skipRel.Count) }
        if (-not $k.complete) { $notes += 'owner note: copy still in progress' }
        # what is already in the cache?
        $rc = Get-CacheReceipt $dest
        $have = @{}; if ($rc) { foreach ($e in @($rc.files)) { $have[[string]$e.rel] = $e } }
        $todo = New-Object System.Collections.ArrayList; $need = [long]0
        for ($i = 0; $i -lt $w.Rel.Count; $i++) {
            $rel = $w.Rel[$i]; if ($rel -like '*.part' -or $rel -like '*.tmp') { continue }
            $e = $have[$rel]; $dp = Join-Path $dest ($rel -replace '/', '\')
            if ($e -and (Test-Path -LiteralPath $dp -PathType Leaf) -and ((Get-Item -LiteralPath $dp -Force).Length -eq $w.Size[$i]) -and [long]$e.size -eq $w.Size[$i]) { continue }
            [void]$todo.Add($i); $need += $w.Size[$i]
        }
        $free = (FreeBytes $cacheDrive) - $(if ($EXEC) { 0 } else { $planned })
        Say ("  source {0} in {1} files;  to copy {2} in {3} files;  cache drive free {4} (margin {5} GiB)" -f (GB $w.Bytes), $rels.Count, (GB $need), $todo.Count, (GB $free), $MarginGiB)
        foreach ($n in $notes) { Say ("  note: " + $n) 'Yellow' }
        $state = 'READY'
        if ($incomplete -and -not $k.complete) { $state = 'INCOMPLETE_SOURCE' }
        elseif ($incomplete) { $state = 'INCOMPLETE_SOURCE' }
        if ($todo.Count -gt 0 -and $free -lt ($need + $margin)) { $state = 'DEFER_SPACE'; Say ("  DEFER: need {0} + {1} GiB margin, have {2}{3}. Free space first (Stage 1) and re-run." -f (GB $need), $MarginGiB, (GB $free), $(if ($EXEC) { '' } else { ' (after the earlier planned copies)' })) 'Yellow' }
        if ($state -eq 'READY') { $planned += $need }
        if ($todo.Count -eq 0 -and $rc) { $state = 'CACHED' }
        # Drive readiness for -RemoveOld (report: always)
        $proofs = @(); $driveOk = $false
        foreach ($pf in @($k.drive)) { $pe = Test-DrivePackProof $k $pf; $proofs += ("{0}: {1}" -f $(if ($pe.ok) { 'OK ' } else { 'no ' }), $(if ($pe.ok) { $pe.desc } else { "$($pe.desc) - $($pe.why)" })); if ($pe.ok) { $driveOk = $true } }
        foreach ($l in $proofs) { Say ("  Drive: " + $l) $(if ($l.StartsWith('OK')) { 'Green' } else { 'DarkYellow' }) }
        if ($state -in @('READY') -and $EXEC) {
            $new = New-Object System.Collections.ArrayList
            foreach ($e in $have.Values) { [void]$new.Add($e) }
            $failed = $false
            foreach ($i in $todo) {
                $rel = $w.Rel[$i]; $sp = Join-Path $src ($rel -replace '/', '\'); $dp = Join-Path $dest ($rel -replace '/', '\'); $tp = $dp + '.part'
                $t1 = Get-Date
                $h = [MC]::CopyHash($sp, $tp)     # md5, sha256, bytes of the SOURCE stream
                $badWhy = $null
                if ([long]$h[2] -ne $w.Size[$i]) { $badWhy = "size changed while copying ($($h[2]) vs $($w.Size[$i]))" }
                elseif ($refMd5.ContainsKey($rel) -and $refMd5[$rel].md5 -ne $h[0]) { $badWhy = 'source MD5 differs from the dataops md5 manifest' }
                elseif ($refSha.ContainsKey($rel) -and $refSha[$rel] -ne $h[1]) { $badWhy = 'source SHA-256 differs from the published SHA256SUMS' }
                if (-not $badWhy) { $dh = [MC]::Sha256File($tp); if ($dh -ne $h[1]) { $badWhy = 'copy SHA-256 differs from source stream SHA-256 (disk error?)' } }
                if ($badWhy) { Say ("  FAIL {0}: {1}" -f $rel, $badWhy) 'Red'; Jrn 'COPY_FAIL' $sp 0 $badWhy; [void]$script:Errs.Add("$sp : $badWhy"); try { Remove-Item -LiteralPath $tp -Force } catch { }; $failed = $true; break }
                [MC]::MoveFile($tp, $dp)
                [void]$new.Add([pscustomobject]@{ rel = $rel; size = [long]$h[2]; sha256 = $h[1]; md5 = $h[0]; ref = $(if ($refSha.ContainsKey($rel)) { 'sha256sums' } elseif ($refMd5.ContainsKey($rel)) { 'md5-manifest' } else { 'copy-only' }) })
                Jrn 'COPIED' $dp ([long]$h[2]) ("sha256=" + $h[1])
                Say ("  ok  {0}  {1}  {2:N0}s  sha256 {3}" -f $rel, (GB ([long]$h[2])), ((Get-Date) - $t1).TotalSeconds, $h[1].Substring(0, 12)) 'Green'
            }
            if (-not $failed) {
                $receipt = [ordered]@{ schema = 'cache_receipt/1'; name = $k.name; source = $src; cache = $dest; created = (Get-Date).ToUniversalTime().ToString('o'); complete = (-not $incomplete); incomplete_notes = $notes; files = @($new | Sort-Object rel); bytes = [long](($new | Measure-Object size -Sum).Sum) }
                [System.IO.File]::WriteAllText((Join-Path $dest 'CACHE_RECEIPT.json'), ($receipt | ConvertTo-Json -Depth 6), (New-Object System.Text.UTF8Encoding($false)))
                $state = if ($incomplete) { 'COPIED_PARTIAL' } else { 'COPIED' }
            } else { $state = 'COPY_FAILED' }
        } elseif ($state -eq 'READY') { Say ("  WOULD copy {0} files ({1}) with inline SHA-256 + re-read verification" -f $todo.Count, (GB $need)) 'Cyan' }
        [void]$summary.Add([pscustomobject]@{ Name = $k.name; State = $state; Bytes = $need; Note = (@($notes) -join '; ') ; Old = $k.old_policy; DriveOk = $driveOk; Src = $src; K = $k })
        Jrn ('STAGE2_' + $state) $src $need ($notes -join '; ')
    }
    Say ''
    Say '--- STAGE 2 SUMMARY ---' 'Cyan'
    foreach ($s in $summary) { Say ("  {0,-24} {1,-18} {2,12}  {3}" -f $s.Name, $s.State, (GB $s.Bytes), $s.Note) 'White' }
    $script:S2 = @($summary | ForEach-Object { [ordered]@{ name = $_.Name; state = $_.State; bytes = $_.Bytes; note = $_.Note; old_policy = $_.Old; drive_ok = $_.DriveOk; source = $_.Src } })
    Say ('registry snippet to apply after the copies verify: ' + (Join-Path $OutDir 'data_registry_cache.toml')) 'Cyan'
    # ---- 2b: remove old copies (separately confirmed) ----
    if ($RemoveOld) {
        Say ''
        Say '=== STAGE 2b: remove OLD copies (cache verified + Drive verified + not referenced) ===' 'Cyan'
        foreach ($s in $summary) {
            if ($null -eq $s.K) { continue }
            $k = $s.K; $src = $s.Src
            if ($k.old_policy -ne 'remove') { Say ("  [{0}] keep: old copy lives in the clean tree / is frozen evidence (old_policy={1})" -f $k.name, $k.old_policy) 'DarkGray'; continue }
            if (-not (Test-Path -LiteralPath $src)) { Say ("  [{0}] source already gone" -f $k.name) 'DarkGray'; continue }
            $dest = NP (Join-Path $CacheRoot ([string]$k.cache))
            $rc = Get-CacheReceipt $dest
            if (-not $rc -or -not $rc.complete) { Say ("  [{0}] HOLD: no complete cache receipt" -f $k.name) 'Yellow'; continue }
            $driveOk2 = $false; $driveWhy = @()
            foreach ($pf in @($k.drive)) { $pe = Test-DrivePackProof $k $pf; if ($pe.ok) { $driveOk2 = $true; $driveWhy += $pe.desc; break } else { $driveWhy += ("{0}: {1}" -f $pe.desc, $pe.why) } }
            if (-not $driveOk2) { Say ("  [{0}] HOLD: no Drive copy verified yet ({1})" -f $k.name, ($driveWhy -join ' | ')) 'Yellow'; continue }
            $refs = @()
            if (-not $SkipReferenceScan) { $refs = @((Get-RefSplit $src).code | Select-Object -First 4 | ForEach-Object { "{0}:{1} -> {2}" -f $_.tree, ($_.file -replace '^.*\\', ''), $_.lit }) }
            if ($refs.Count) { Say ("  [{0}] HOLD: still referenced by active code/config: {1}  (point the registry at data_cache first)" -f $k.name, ($refs -join ' | ')) 'Yellow'; continue }
            $pr = Find-ProcUsing $src; if ($pr) { Say ("  [{0}] HOLD: in use by {1}" -f $k.name, $pr) 'Yellow'; continue }
            # cache copy must re-verify against the receipt (size always; SHA-256 re-hash with -Execute)
            $bad = 0
            foreach ($e in @($rc.files)) {
                $dp = Join-Path $dest ($e.rel -replace '/', '\'); $sp = Join-Path $src ($e.rel -replace '/', '\')
                if (-not (Test-Path -LiteralPath $dp -PathType Leaf) -or (Get-Item -LiteralPath $dp -Force).Length -ne [long]$e.size) { $bad++; continue }
                if (Test-Path -LiteralPath $sp -PathType Leaf) { if ((Get-Item -LiteralPath $sp -Force).Length -ne [long]$e.size) { $bad++; continue } }
                if ($EXEC -and ([MC]::Sha256File($dp)) -ne $e.sha256) { $bad++ }
            }
            $wsrc = [MC]::Walk($src, $true)
            $extra = @($wsrc.Rel | Where-Object { $rel = $_; -not (@($rc.files) | Where-Object { $_.rel -eq $rel }) })
            if ($bad -gt 0) { Say ("  [{0}] HOLD: cache copy fails verification ({1} file(s))" -f $k.name, $bad) 'Red'; continue }
            if ($extra.Count) { Say ("  [{0}] HOLD: {1} file(s) in the old copy are not in the cache (e.g. {2})" -f $k.name, $extra.Count, $extra[0]) 'Yellow'; continue }
            if ($wsrc.Reparse.Count) { Say ("  [{0}] HOLD: junction inside" -f $k.name) 'Yellow'; continue }
            $sz = $wsrc.Bytes
            if (-not $EXEC) { Say ("  [{0}] WOULD remove old copy {1}  ({2})" -f $k.name, $src, (GB $sz)) 'Cyan'; Jrn 'WOULD_REMOVE_OLD' $src $sz ''; continue }
            if (-not $Yes) { $ans = Read-Host ("  remove OLD copy {0} ({1})? type REMOVE-OLD" -f $src, (GB $sz)); if ($ans -cne 'REMOVE-OLD') { Say '  skipped.' 'Yellow'; continue } }
            $errs = New-Object 'System.Collections.Generic.List[string]'
            $freed = [MC]::DeleteTree($src, $errs)
            $script:Freed[(DriveOf $src)] += $freed
            Jrn 'REMOVED_OLD' $src $freed ($k.name)
            Record-Removed ([pscustomobject]@{ Path = $src; Type = 'keep-local-old'; Files = $wsrc.Rel.Count; Drive = @("cache:$dest") }) $freed
            Say ("  [{0}] removed old copy ({1})" -f $k.name, (GB $freed)) 'Green'
            Prune-EmptyParents $src
        }
    }
    Finish-Stage 2 $freeBefore
}

# =====================================================================================================
#  Stage 3: obsolete (the single-work-tree switch was completed on 2026-10-10)
# =====================================================================================================
function Invoke-Stage3 {
    Say '=== STAGE 3 is obsolete: the single-work-tree switch was completed on 2026-10-10 (E:\research-dense is the clean tree). Nothing to do. ===' 'Yellow'
    $freeBefore = @{}; foreach ($d in 'C', 'D', 'E', 'G') { $freeBefore[$d] = FreeBytes $d }
    $script:NoReport = $true
    Finish-Stage 3 $freeBefore
}


# =====================================================================================================
#  -SelfTest: primitives on a scratch folder (no research file is touched)
# =====================================================================================================
function Invoke-SelfTest {
    $base = Join-Path $OutDir ('_selftest_' + $RunStamp)
    if (-not $base.StartsWith($OutDir, [StringComparison]::OrdinalIgnoreCase)) { throw 'selftest path guard' }
    $script:pass = 0; $script:fail = 0
    function T([string]$name, [bool]$ok) { if ($ok) { $script:pass++ ; Say ("  PASS  $name") 'Green' } else { $script:fail++; Say ("  FAIL  $name") 'Red' } }
    Say "=== SELF TEST in $base ===" 'Cyan'
    New-Item -ItemType Directory -Path $base -Force | Out-Null
    # the self-test journals into its own scratch file (the real journal feeds tools/offload_map.py)
    $saveJW = $script:JW; $saveJP = $script:JournalPath
    $script:JournalPath = Join-Path $base 'journal_selftest.log'
    $script:JW = New-Object System.IO.StreamWriter($script:JournalPath, $true, (New-Object System.Text.UTF8Encoding($false)))
    $script:JW.AutoFlush = $true
    $tree = Join-Path $base 'tree'; $target = Join-Path $base 'outside_target'
    New-Item -ItemType Directory -Path (Join-Path $tree 'a\b') -Force | Out-Null
    New-Item -ItemType Directory -Path $target -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $tree 'a\f1.txt') -Value 'one'
    Set-ItemProperty -LiteralPath (Join-Path $tree 'a\f1.txt') -Name IsReadOnly -Value $true
    Set-Content -LiteralPath (Join-Path $tree 'a\b\f2.txt') -Value 'two'
    Set-Content -LiteralPath (Join-Path $target 'keepme.txt') -Value 'must survive'
    New-Item -ItemType Junction -Path (Join-Path $tree 'a\b\jl') -Target $target | Out-Null
    $w = [MC]::Walk($tree, $true)
    T 'walker lists regular files and reports the junction without following it' (($w.Rel.Count -eq 2) -and ($w.Reparse.Count -eq 1) -and ($w.Reparse[0] -eq 'a/b/jl'))
    $threw = $false
    try { $e = New-Object 'System.Collections.Generic.List[string]'; [void][MC]::DeleteTree($tree, $e) } catch { $threw = $true }
    T 'DeleteTree refuses a tree containing a junction and deletes NOTHING' ($threw -and (Test-Path -LiteralPath (Join-Path $tree 'a\f1.txt')) -and (Test-Path -LiteralPath (Join-Path $target 'keepme.txt')))
    [MC]::SetProtected([string[]]@('sealed'), [string[]]@('chunk5'))
    New-Item -ItemType Directory -Path (Join-Path $tree 'sealed') -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $tree 'sealed\x.txt') -Value 'x'
    Set-Content -LiteralPath (Join-Path $tree 'my_chunk5_file.txt') -Value 'x'
    $w2 = [MC]::Walk($tree, $true)
    T 'protected names are reported and not descended' (($w2.Sensitive.Count -eq 2) -and (-not ($w2.Rel -contains 'sealed/x.txt')))
    [MC]::SetProtected([string[]]@($M.protected.exact), [string[]]@($M.protected.contains))
    cmd /c rmdir (Join-Path $tree 'a\b\jl') | Out-Null
    Remove-Item -LiteralPath (Join-Path $tree 'sealed') -Recurse -Force
    Remove-Item -LiteralPath (Join-Path $tree 'my_chunk5_file.txt') -Force
    T 'junction removed without touching its target' (-not (Test-Path -LiteralPath (Join-Path $tree 'a\b\jl')) -and (Test-Path -LiteralPath (Join-Path $target 'keepme.txt')))
    $e = New-Object 'System.Collections.Generic.List[string]'
    $n = [MC]::DeleteTree($tree, $e)
    T 'DeleteTree removes a clean tree incl. a read-only file and returns the byte count' ((-not (Test-Path -LiteralPath $tree)) -and $n -gt 0 -and $e.Count -eq 0)
    # long path
    $lp = $base; for ($i = 0; $i -lt 12; $i++) { $lp = Join-Path $lp ('segment_number_{0:00}_abcdefghij' -f $i) }
    [System.IO.Directory]::CreateDirectory([MC]::LP($lp)) | Out-Null
    [System.IO.File]::WriteAllText([MC]::LP((Join-Path $lp 'deep.txt')), 'deep')
    $e2 = New-Object 'System.Collections.Generic.List[string]'
    $n2 = [MC]::DeleteTree((Join-Path $base 'segment_number_00_abcdefghij'), $e2)
    T ('long path (>260) walk + delete ({0} chars)' -f $lp.Length) (($lp.Length -gt 260) -and ($n2 -eq 4) -and -not (Test-Path -LiteralPath (Join-Path $base 'segment_number_00_abcdefghij')))
    # copy + hash
    $src = Join-Path $base 'src.bin'; $dst = Join-Path $base 'sub dir\dst.bin'
    $buf = New-Object byte[] (20MB); (New-Object System.Random 7).NextBytes($buf); [System.IO.File]::WriteAllBytes($src, $buf)
    $h = [MC]::CopyHash($src, $dst)
    $fh = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash.ToLowerInvariant(); $fm = (Get-FileHash -LiteralPath $src -Algorithm MD5).Hash.ToLowerInvariant()
    T 'CopyHash md5+sha256 of the stream equal Get-FileHash and the bytes are identical' (($h[0] -eq $fm) -and ($h[1] -eq $fh) -and ([long]$h[2] -eq 20MB) -and (([MC]::Sha256File($dst)) -eq $fh))
    $bad = [MC]::VerifyHashes($base, [string[]]@('src.bin', 'sub dir/dst.bin'), [string[]]@($fm, ('0' * 32)), $false)
    T 'VerifyHashes flags only the wrong file' (($bad.Count -eq 1) -and ($bad[0] -eq 'sub dir/dst.bin'))
    # git worktree remove with a space in the path (scratch repo)
    $repo = Join-Path $base 'scratch_repo'; New-Item -ItemType Directory -Path $repo -Force | Out-Null
    [void](GitX $repo @('init', '-q')); [void](GitX $repo @('config', 'user.email', 'selftest@example.invalid')); [void](GitX $repo @('config', 'user.name', 'selftest'))
    Set-Content -LiteralPath (Join-Path $repo 'a.txt') -Value 'a'
    [void](GitX $repo @('add', 'a.txt')); [void](GitX $repo @('commit', '-q', '-m', 'init'))
    $wt = Join-Path $base 'wt with space'
    [void](GitX $repo @('worktree', 'add', '-q', '-b', 'selftest-branch', $wt))
    Set-Content -LiteralPath (Join-Path $wt 'untracked.txt') -Value 'u'
    $old = $script:MainRepo; $script:MainRepo = $repo
    $fake = [pscustomobject]@{ Path = $wt; Type = 'worktree'; Drv = 'E'; Drive = @('selftest'); Files = 2; Status = 'ELIGIBLE'; Reason = ''; Snap = @{ bytes = 10; files = 2; maxm = 0 }; Wt = [pscustomobject]@{ cls = 'worktree' } }
    $script:RemovedMap = Join-Path $base 'removed_selftest.jsonl'
    $before = $script:Freed['E']
    Remove-Eligible $fake
    $script:MainRepo = $old
    T 'Remove-Eligible removes a registered worktree via git worktree remove --force (path with space)' ((-not (Test-Path -LiteralPath $wt)) -and ($fake.Status -eq 'DELETED') -and (Test-Path -LiteralPath (Join-Path $repo 'a.txt')))
    $script:Freed['E'] = $before
    # partial removal around a protected entry (DeleteListed) + unlinking inside-tree junctions via Remove-Eligible
    $pt = Join-Path $base 'partial'
    New-Item -ItemType Directory -Path (Join-Path $pt 'keepdir\sealed') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $pt 'gone\sub') -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $pt 'keepdir\sealed\secret.txt') -Value 'sealed data'
    Set-Content -LiteralPath (Join-Path $pt 'keepdir\ok.txt') -Value 'ok'
    Set-Content -LiteralPath (Join-Path $pt 'gone\sub\a.txt') -Value 'a'
    Set-Content -LiteralPath (Join-Path $pt 'top.txt') -Value 't'
    [MC]::SetProtected([string[]]@('sealed'), [string[]]@('chunk5'))
    $wp = [MC]::Walk($pt, $true)
    $e4 = New-Object 'System.Collections.Generic.List[string]'
    $n4 = [MC]::DeleteListed($pt, [string[]]@($wp.Rel), $e4)
    [MC]::SetProtected([string[]]@($M.protected.exact), [string[]]@($M.protected.contains))
    T 'DeleteListed removes only the listed files, prunes emptied dirs, leaves the protected folder untouched' ((Test-Path -LiteralPath (Join-Path $pt 'keepdir\sealed\secret.txt')) -and -not (Test-Path -LiteralPath (Join-Path $pt 'gone')) -and -not (Test-Path -LiteralPath (Join-Path $pt 'top.txt')) -and -not (Test-Path -LiteralPath (Join-Path $pt 'keepdir\ok.txt')) -and $e4.Count -eq 0 -and $n4 -gt 0)
    $sc = Join-Path $base 'scratch_item'
    New-Item -ItemType Directory -Path (Join-Path $sc 'real') -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $sc 'real\f.txt') -Value 'f'
    New-Item -ItemType Junction -Path (Join-Path $sc 'alias') -Target (Join-Path $sc 'real') | Out-Null
    $saveRoots = $M.protected.roots; $saveAllowed = $M.allowed_roots
    $M.protected.roots = @(); $M.allowed_roots = @($base)
    $fake2 = [pscustomobject]@{ Path = $sc; Type = 'item'; Drv = 'E'; Drive = @('selftest'); Files = 1; Status = 'ELIGIBLE'; Reason = ''; Snap = @{ bytes = 4; files = 1; maxm = 0; links = @('alias') }; Wt = $null }
    $w5 = [MC]::Walk($sc, $true)
    $fake2.Snap.bytes = $w5.Bytes; $fake2.Snap.files = $w5.Rel.Count; $fake2.Snap.maxm = ($w5.MTicks | Measure-Object -Maximum).Maximum
    Remove-Eligible $fake2
    $M.protected.roots = $saveRoots; $M.allowed_roots = $saveAllowed
    T 'Remove-Eligible unlinks an inside-tree junction (link only) and then removes the tree' ((-not (Test-Path -LiteralPath $sc)) -and ($fake2.Status -eq 'DELETED'))
    $script:Freed['E'] = $before
    # --- synthetic end-to-end evaluation of a mirror-backed item (the Drive listing is FAKED, rclone is not called) ---
    $saveRoots2 = $M.protected.roots; $saveAllowed2 = $M.allowed_roots
    $M.protected.roots = @(); $M.allowed_roots = @($base)
    $saveSkipRef = $SkipReferenceScan; Set-Variable -Name SkipReferenceScan -Value $true -Scope Script
    $saveDeep = $script:DeepNow; $script:DeepNow = $true
    $saveRRoots = $script:RRoots; $saveRK = $script:RK; $script:RRoots = @{}; $script:RK = @{}
    $script:RRoots['md5|selftest/evalitem'] = 2
    $ev = Join-Path $base 'evalitem'
    New-Item -ItemType Directory -Path (Join-Path $ev 'sub') -Force | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $ev 'a.txt'), 'alpha'); [System.IO.File]::WriteAllText((Join-Path $ev 'sub\b.txt'), 'bravo')
    $ma = (Get-FileHash -LiteralPath (Join-Path $ev 'a.txt') -Algorithm MD5).Hash.ToLowerInvariant()
    $mb2 = (Get-FileHash -LiteralPath (Join-Path $ev 'sub\b.txt') -Algorithm MD5).Hash.ToLowerInvariant()
    $man = Join-Path $base 'evalitem.md5'
    [System.IO.File]::WriteAllText($man, ("{0}  5  a.txt`n{1}  5  sub/b.txt`n" -f $ma, $mb2))
    $script:RK['selftest/evalitem/a.txt'] = @{ size = 5; md5 = $ma }
    $script:RK['selftest/evalitem/sub/b.txt'] = @{ size = 5; md5 = $mb2 }
    function MkItem($ts) { [pscustomobject]@{ id = 'selftest'; path = $ev; backup_only = $false; special = $null; proofs = @([pscustomobject]@{ t = 'mirror'; receipt = 'selftest'; remote = 'gdrive:research-archive/selftest/evalitem'; manifest = $man; manifest_sha256 = $null; prefix = ''; ts = $ts }) } }
    $future = (Get-Date).AddHours(1).ToString('o'); $past = (Get-Date).AddHours(-1).ToString('o')
    $script:ManCache = @{}
    $r1 = Eval-Item (MkItem $future)
    T 'eval: intact local copy + matching (faked) Drive listing -> ELIGIBLE' ($r1.Status -eq 'ELIGIBLE')
    $script:ManCache = @{}
    $r2 = Eval-Item (MkItem $past)
    T 'eval: files newer than the backup timestamp -> LOCAL_CHANGED (mtime guard)' ($r2.Status -eq 'LOCAL_CHANGED')
    $script:ManCache = @{}
    [System.IO.File]::WriteAllText((Join-Path $ev 'a.txt'), 'alpHa')
    $r3 = Eval-Item (MkItem $future)
    T 'eval: same-size content change is caught by the MD5 re-hash -> LOCAL_CHANGED' ($r3.Status -eq 'LOCAL_CHANGED' -and $r3.Reason -match 'MD5')
    [System.IO.File]::WriteAllText((Join-Path $ev 'a.txt'), 'alpha')
    $script:ManCache = @{}
    $script:RK['selftest/evalitem/sub/b.txt'] = @{ size = 5; md5 = ('0' * 32) }
    $r4 = Eval-Item (MkItem $future)
    T 'eval: Drive copy with a different MD5 -> DRIVE_UNVERIFIED' ($r4.Status -eq 'DRIVE_UNVERIFIED')
    $script:RK['selftest/evalitem/sub/b.txt'] = @{ size = 5; md5 = $mb2 }
    $script:RK.Remove('selftest/evalitem/a.txt')
    $script:ManCache = @{}
    $r5 = Eval-Item (MkItem $future)
    T 'eval: file absent on Drive -> DRIVE_UNVERIFIED' ($r5.Status -eq 'DRIVE_UNVERIFIED')
    $script:RK['selftest/evalitem/a.txt'] = @{ size = 5; md5 = $ma }
    [System.IO.File]::WriteAllText((Join-Path $ev 'extra.txt'), 'not in backup')
    $script:ManCache = @{}
    $r6 = Eval-Item (MkItem $future)
    T 'eval: extra local file not in the backup manifest -> LOCAL_NOT_IN_BACKUP' ($r6.Status -eq 'LOCAL_NOT_IN_BACKUP')
    Remove-Item -LiteralPath (Join-Path $ev 'extra.txt') -Force
    New-Item -ItemType Directory -Path (Join-Path $ev 'sealed') -Force | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $ev 'sealed\x.txt'), 'x')
    $script:ManCache = @{}
    $r7 = Eval-Item (MkItem $future)
    T 'eval: protected-name folder inside -> SENSITIVE_INSIDE (whole-folder removal refused)' ($r7.Status -eq 'SENSITIVE_INSIDE')
    Set-Variable -Name RemoveAroundProtected -Value $true -Scope Script
    $script:ManCache = @{}
    $r8 = Eval-Item (MkItem $future)
    T 'eval: -RemoveAroundProtected -> ELIGIBLE as PARTIAL (protected folder left alone)' ($r8.Status -eq 'ELIGIBLE' -and $r8.Snap.partial -and $r8.Snap.left -eq 1)
    $fake3 = $r8; Remove-Eligible $fake3
    T 'partial removal deleted only the verified files and left sealed\x.txt' ((Test-Path -LiteralPath (Join-Path $ev 'sealed\x.txt')) -and -not (Test-Path -LiteralPath (Join-Path $ev 'a.txt')) -and -not (Test-Path -LiteralPath (Join-Path $ev 'sub')))
    Set-Variable -Name RemoveAroundProtected -Value $false -Scope Script
    # --- items cited only by frozen records are trimmed to their cited files; code references still hold ---
    $et = Join-Path $base 'evaltrim'
    New-Item -ItemType Directory -Path (Join-Path $et 'sub') -Force | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $et 'a.txt'), 'alpha'); [System.IO.File]::WriteAllText((Join-Path $et 'sub\b.txt'), 'bravo')
    $script:RRoots['md5|selftest/evaltrim'] = 2
    $script:RK['selftest/evaltrim/a.txt'] = @{ size = 5; md5 = $ma }
    $script:RK['selftest/evaltrim/sub/b.txt'] = @{ size = 5; md5 = $mb2 }
    $mant = Join-Path $base 'evaltrim.md5'
    [System.IO.File]::WriteAllText($mant, ("{0}  5  a.txt`n{1}  5  sub/b.txt`n" -f $ma, $mb2))
    function MkTrim { [pscustomobject]@{ id = 'selftest-trim'; path = $et; backup_only = $false; special = $null; proofs = @([pscustomobject]@{ t = 'mirror'; receipt = 'selftest'; remote = 'gdrive:research-archive/selftest/evaltrim'; manifest = $mant; manifest_sha256 = $null; prefix = ''; ts = $future }) } }
    Set-Variable -Name SkipReferenceScan -Value $false -Scope Script
    $saveLit = $script:Lit
    $script:Lit = New-Object System.Collections.ArrayList
    [void]$script:Lit.Add(@{ lit = (K (Join-Path $et 'sub\b.txt')); file = 'E:\x\protocols\experiment_history.json'; tree = 'main' })
    $script:ManCache = @{}
    $r9 = Eval-Item (MkTrim)
    T 'eval: cited only by the ledger -> ELIGIBLE as TRIM keeping the cited file' ($r9.Status -eq 'ELIGIBLE' -and $r9.Snap.partial -and (@($r9.Snap.rels) -join ',') -eq 'a.txt' -and $r9.Reason -match '^TRIM')
    [void]$script:Lit.Add(@{ lit = (K $et); file = 'E:\x\tools\uses_it.py'; tree = 'main' })
    $script:ManCache = @{}
    $r10 = Eval-Item (MkTrim)
    T 'eval: an additional code reference -> REFERENCED (held)' ($r10.Status -eq 'REFERENCED')
    $script:Lit.RemoveAt(1)
    Remove-Eligible $r9
    T 'trim removal deleted the uncited file and kept the cited one' ((Test-Path -LiteralPath (Join-Path $et 'sub\b.txt')) -and -not (Test-Path -LiteralPath (Join-Path $et 'a.txt')))
    $script:Lit = $saveLit
    $script:RRoots = $saveRRoots; $script:RK = $saveRK; $script:DeepNow = $saveDeep
    Set-Variable -Name SkipReferenceScan -Value $saveSkipRef -Scope Script
    $M.protected.roots = $saveRoots2; $M.allowed_roots = $saveAllowed2
    $script:Freed['E'] = $before
    $script:Res.Clear()
    # --- synthetic Stage 2: copy + SHA-256 receipt + corrupted-source detection + guarded removal of the old copy ---
    $s2 = Join-Path $base 'stage2'
    $srcDir = Join-Path $s2 'repo\data\pack_x'; $cacheDir = Join-Path $s2 'cache'
    New-Item -ItemType Directory -Path $srcDir -Force | Out-Null
    $b1 = New-Object byte[] (3MB); (New-Object System.Random 11).NextBytes($b1); [System.IO.File]::WriteAllBytes((Join-Path $srcDir 'one.npy'), $b1)
    [System.IO.File]::WriteAllText((Join-Path $srcDir 'two.json'), '{"k": 2}')
    $m1 = (Get-FileHash -LiteralPath (Join-Path $srcDir 'one.npy') -Algorithm MD5).Hash.ToLowerInvariant()
    $m2 = (Get-FileHash -LiteralPath (Join-Path $srcDir 'two.json') -Algorithm MD5).Hash.ToLowerInvariant()
    $h1 = (Get-FileHash -LiteralPath (Join-Path $srcDir 'one.npy') -Algorithm SHA256).Hash.ToLowerInvariant()
    $s2man = Join-Path $s2 'pack_x.md5'
    $len2 = (Get-Item -LiteralPath (Join-Path $srcDir 'two.json')).Length
    [System.IO.File]::WriteAllText($s2man, ("# pack_manifest/1`n{0}`t{1}`tone.npy`n{2}`t{3}`ttwo.json`n" -f $m1, 3MB, $m2, $len2))
    [System.IO.File]::WriteAllText((Join-Path $srcDir 'SHA256SUMS'), ("{0}  one.npy`n" -f $h1))
    $repo2 = Join-Path $s2 'repo'
    [void](GitX $repo2 @('init', '-q')); [void](GitX $repo2 @('config', 'user.email', 'selftest@example.invalid')); [void](GitX $repo2 @('config', 'user.name', 'selftest'))
    [void](GitX $repo2 @('add', 'data')); [void](GitX $repo2 @('commit', '-q', '-m', 'x'))
    $kfake = [pscustomobject]@{ name = 'pack_x'; source = $srcDir; cache = 'pack_x'; old_policy = 'remove'; note = ''; complete = $true; registry = 'pack_x'
        manifest = $s2man; manifest_sha256 = $null; ref_sha256sums = 'SHA256SUMS'
        drive = @([pscustomobject]@{ t = 'git_tracked'; repo = $repo2; rel = 'data/pack_x' }) }
    $saveKeep = $M.keep_local; $M.keep_local = @($kfake)
    $saveCache = $script:CacheRoot; $script:CacheRoot = $cacheDir
    $saveExec = $script:EXEC; $script:EXEC = $true
    $saveYes = $Yes; Set-Variable -Name Yes -Value $true -Scope Script
    $saveMargin = $MarginGiB; Set-Variable -Name MarginGiB -Value 0 -Scope Script
    $saveRef2 = $SkipReferenceScan; Set-Variable -Name SkipReferenceScan -Value $true -Scope Script
    $saveOnly = $Only; Set-Variable -Name Only -Value @() -Scope Script
    $saveSummary = $script:S2; $script:NoReport = $true
    # (a) corrupt the source AFTER the manifest was taken: same size, different bytes -> must not be copied
    $orig = [System.IO.File]::ReadAllBytes((Join-Path $srcDir 'one.npy')); $bad1 = $orig.Clone(); $bad1[100] = ($bad1[100] -bxor 0xFF)
    [System.IO.File]::WriteAllBytes((Join-Path $srcDir 'one.npy'), $bad1)
    $script:ManCache = @{}
    $freeSave = @{}; foreach ($d in 'C', 'D', 'E', 'G') { $freeSave[$d] = 0 }
    Invoke-Stage2 *> $null
    T 'stage 2: source that disagrees with the md5 manifest / SHA256SUMS is refused (no final file, no receipt)' ((-not (Test-Path -LiteralPath (Join-Path $cacheDir 'pack_x\one.npy'))) -and (-not (Test-Path -LiteralPath (Join-Path $cacheDir 'pack_x\CACHE_RECEIPT.json'))) -and ($script:S2[0].state -eq 'COPY_FAILED'))
    # (b) restore the source: copy succeeds, receipt has the sha256 of every file
    [System.IO.File]::WriteAllBytes((Join-Path $srcDir 'one.npy'), $orig)
    $script:ManCache = @{}
    Invoke-Stage2 *> $null
    $rcp = Get-Content -LiteralPath (Join-Path $cacheDir 'pack_x\CACHE_RECEIPT.json') -Raw | ConvertFrom-Json
    $e1 = @($rcp.files) | Where-Object { $_.rel -eq 'one.npy' }
    T 'stage 2: verified copy lands in data_cache with a receipt holding the per-file SHA-256 (source untouched)' (($script:S2[0].state -eq 'COPIED') -and ($e1.sha256 -eq $h1) -and (Test-Path -LiteralPath (Join-Path $srcDir 'one.npy')) -and (([MC]::Sha256File((Join-Path $cacheDir 'pack_x\one.npy'))) -eq $h1))
    # (c) old-copy removal needs cache receipt + Drive proof; the git_tracked proof is satisfied in the scratch repo
    Set-Variable -Name RemoveOld -Value $true -Scope Script
    $script:ManCache = @{}
    Invoke-Stage2 *> $null
    T 'stage 2b: old copy removed only after cache + Drive proof verify (cache copy remains)' ((-not (Test-Path -LiteralPath $srcDir)) -and (Test-Path -LiteralPath (Join-Path $cacheDir 'pack_x\one.npy')))
    Set-Variable -Name RemoveOld -Value $false -Scope Script
    $M.keep_local = $saveKeep; $script:CacheRoot = $saveCache; $script:EXEC = $saveExec
    Set-Variable -Name Yes -Value $saveYes -Scope Script; Set-Variable -Name MarginGiB -Value $saveMargin -Scope Script
    Set-Variable -Name SkipReferenceScan -Value $saveRef2 -Scope Script; Set-Variable -Name Only -Value $saveOnly -Scope Script
    $script:S2 = $saveSummary; $script:NoReport = $false
    $script:Freed['E'] = $before
    # --- worktree retirement primitives: junction unlink-first, dependency order, evaluation, sealed relocation, tracked-dirty rule ---
    $saveMain = $script:MainRepo; $script:MainRepo = $repo
    $saveRoots3 = $M.protected.roots; $saveAllowed3 = $M.allowed_roots; $saveKeep3 = $M.keep_local
    $M.protected.roots = @(); $M.allowed_roots = @($base); $M.keep_local = @()
    $saveSkip3 = $SkipReferenceScan; Set-Variable -Name SkipReferenceScan -Value $true -Scope Script
    $saveSealRoot = $script:SealedRoot; $saveSealMan = $script:SealedManifestPath
    $script:SealedRoot = Join-Path $base 'sealed_scratch'; $script:SealedManifestPath = Join-Path $script:SealedRoot 'MOVE_MANIFEST.csv'
    $saveRes3 = $script:Res; $script:Res = New-Object System.Collections.ArrayList
    $saveLinkCache = $script:WtLinkCache; $script:WtLinkCache = @{}
    function MkWt([string]$nm) { $p = Join-Path $base $nm; [void](GitX $repo @('worktree', 'add', '-q', '-b', ('st-' + $nm), $p)); return $p }
    function MkFake($p, $snap) { [pscustomobject]@{ Id = ('wt:' + (Split-Path -Leaf $p)); Path = $p; Type = 'worktree'; Drv = 'E'; Drive = @('selftest'); Files = 1; Status = 'ELIGIBLE'; Reason = ''; Bytes = 10; Snap = $snap; Wt = [pscustomobject]@{ cls = 'worktree'; last = $false } } }
    function MkWd([string]$nm, [string]$p) { [pscustomobject]@{ name = $nm; path = $p; cls = 'worktree'; branch = ('st-' + $nm); status = 'ok'; upload_exit = 0; check_exit = 0; dest = ''; list = ''; list_sha256 = $null; method = $null; archive = $null; sens_count = 0; sens = @(); junctions = 0; committed = $false; commit = $null; preserve_ref = $null; dirty_paths = @(); dirty_sens_skipped = @(); sealed_relocate = $false; last = $false } }
    function JrnText { $fs = [System.IO.File]::Open($JournalPath, 'Open', 'Read', 'ReadWrite'); try { $sr = New-Object System.IO.StreamReader($fs); return $sr.ReadToEnd() } finally { $fs.Dispose() } }
    # T1: junctions to an outside tree and to the worktree itself are unlinked first; the outside target survives
    $out1 = Join-Path $base 'out1'; New-Item -ItemType Directory -Path (Join-Path $out1 'deep\sub') -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $out1 'deep\sub\keep.txt') -Value 'target data'
    $w1 = MkWt 'wt_links'
    Set-Content -LiteralPath (Join-Path $w1 'junk.txt') -Value 'u'
    New-Item -ItemType Directory -Path (Join-Path $w1 'real') -Force | Out-Null; Set-Content -LiteralPath (Join-Path $w1 'real\f.txt') -Value 'f'
    New-Item -ItemType Junction -Path (Join-Path $w1 'ext_out') -Target $out1 | Out-Null
    New-Item -ItemType Junction -Path (Join-Path $w1 'real\alias') -Target (Join-Path $w1 'real') | Out-Null
    $f1 = MkFake $w1 @{ bytes = 10; files = 2; maxm = 0 }
    Remove-Eligible $f1
    $jt = JrnText
    T 'worktree with junctions (outside + self): unlinked first, git worktree remove succeeds, outside target and its files survive' ((-not (Test-Path -LiteralPath $w1)) -and ($f1.Status -eq 'DELETED') -and (Test-Path -LiteralPath (Join-Path $out1 'deep\sub\keep.txt')) -and ($jt -match 'UNLINKED\t[^\t]*ext_out\t') -and ($jt -match 'target verified present afterwards'))
    # T2: a dangling junction (target already removed, e.g. by Stage 2b) is unlinked without complaint
    $out2 = Join-Path $base 'out2'; New-Item -ItemType Directory -Path $out2 -Force | Out-Null
    $w2 = MkWt 'wt_dangling'
    New-Item -ItemType Junction -Path (Join-Path $w2 'gone_target') -Target $out2 | Out-Null
    Remove-Item -LiteralPath $out2 -Recurse -Force
    $f2 = MkFake $w2 @{ bytes = 10; files = 1; maxm = 0 }
    Remove-Eligible $f2
    T 'dangling junction (target already gone) is unlinked and journaled as dangling; worktree removed' ((-not (Test-Path -LiteralPath $w2)) -and ($f2.Status -eq 'DELETED') -and ((JrnText) -match 'target was already absent'))
    $thrown = $false; try { [MC]::UnlinkLink($base) } catch { $thrown = $true }
    T 'UnlinkLink refuses a directory that is not a reparse point' ($thrown -and (Test-Path -LiteralPath $base))
    # T3: dependency order and the held-dependent rule
    $wA = MkWt 'wt_dep_a'; $wB = MkWt 'wt_dep_b'
    New-Item -ItemType Directory -Path (Join-Path $wB 'payload') -Force | Out-Null; Set-Content -LiteralPath (Join-Path $wB 'payload\p.txt') -Value 'p'
    New-Item -ItemType Junction -Path (Join-Path $wA 'to_b') -Target (Join-Path $wB 'payload') | Out-Null
    $script:WtLinkCache[(K $wA)] = @(Get-LinkInfo $wA ([MC]::Walk($wA, $false).Reparse))
    $rA = MkFake $wA @{ bytes = 5; files = 1; maxm = 0 }; $rB = MkFake $wB @{ bytes = 1; files = 1; maxm = 0 }
    $rB.Bytes = 1; [void]$script:Res.Add($rB); [void]$script:Res.Add($rA)
    Resolve-WorktreeOrder
    T 'order: the worktree with a junction into B is removed before B (even though B is smaller)' ((@($script:WtOrder).Count -eq 2) -and ($script:WtOrder[0].Id -eq $rA.Id) -and ($script:WtOrder[1].Id -eq $rB.Id) -and ($rA.Status -eq 'ELIGIBLE') -and ($rB.Status -eq 'ELIGIBLE'))
    $rA.Status = 'DIRTY_TRACKED'
    Resolve-WorktreeOrder
    T 'a worktree that stays (held) and links into B holds B back (LINKED_FROM_HELD)' ($rB.Status -eq 'LINKED_FROM_HELD')
    $rA.Status = 'ELIGIBLE'; $rB.Status = 'ELIGIBLE'
    # T4: evaluation: a worktree whose only content is a junction (git descends into it: everything below is ignored)
    $out4 = Join-Path $base 'out4'; New-Item -ItemType Directory -Path (Join-Path $out4 'inner') -Force | Out-Null; Set-Content -LiteralPath (Join-Path $out4 'inner\big.bin') -Value 'x'
    $w4 = MkWt 'wt_eval_links'
    New-Item -ItemType Junction -Path (Join-Path $w4 'ext') -Target $out4 | Out-Null
    $script:WtRegistered = $null
    $r4 = Eval-Worktree (MkWd 'wt_eval_links' $w4)
    T 'eval: junction-only worktree -> ELIGIBLE, link recorded (external target), files below the link not counted as unbacked' (($r4.Status -eq 'ELIGIBLE') -and (@($r4.Snap.links).Count -eq 1) -and ($r4.Snap.links[0].kind -eq 'external') -and ($r4.Snap.links[0].exists))
    Set-Content -LiteralPath (Join-Path $w4 'stray.txt') -Value 'not in the list'
    $script:WtRegistered = $null
    $r4b = Eval-Worktree (MkWd 'wt_eval_links' $w4)
    T 'eval: an untracked file that is not in the uploaded list -> NEW_UNBACKED' ($r4b.Status -eq 'NEW_UNBACKED')
    # T5: sensitive-named untracked file: held; with sealed_relocate it is moved unopened into sealed\ before the removal
    $w5 = MkWt 'wt_eval_sens'
    $sf = Join-Path $w5 'my_chunk5_note.txt'; Set-Content -LiteralPath $sf -Value 'opaque test bytes'
    $shaSrc = [MC]::Sha256File($sf)
    $script:WtRegistered = $null
    $r5 = Eval-Worktree (MkWd 'wt_eval_sens' $w5)
    T 'eval: sensitive-named untracked file still in place -> SENSITIVE_SKIPPED (run MOVE_TO_SEALED first)' (($r5.Status -eq 'SENSITIVE_SKIPPED') -and ($r5.Reason -match 'MOVE_TO_SEALED'))
    $wd5 = MkWd 'wt_eval_sens' $w5; $wd5.sealed_relocate = $true
    $script:WtRegistered = $null
    $r5b = Eval-Worktree $wd5
    Remove-Eligible $r5b
    $sd = Join-Path $script:SealedRoot ('E\' + (NP $sf).Substring(3))
    $manTxt = ''; if (Test-Path -LiteralPath $script:SealedManifestPath) { $manTxt = [System.IO.File]::ReadAllText($script:SealedManifestPath) }
    T 'sealed relocation: file moved unopened to sealed\ (SHA-256 equal), MOVE_MANIFEST row written, worktree then removed' (($r5b.Status -eq 'DELETED') -and (-not (Test-Path -LiteralPath $w5)) -and (Test-Path -LiteralPath $sd) -and (([MC]::Sha256File($sd)) -eq $shaSrc) -and ($manTxt -match 'MOVED') -and ($manTxt -match 'my_chunk5_note'))
    # T6: sealed relocation refuses an existing destination (nothing is removed)
    $w6 = MkWt 'wt_seal_exists'
    $sf6 = Join-Path $w6 'x_holdout_y.txt'; Set-Content -LiteralPath $sf6 -Value 'abc'
    $sd6 = Join-Path $script:SealedRoot ('E\' + (NP $sf6).Substring(3)); [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $sd6)); Set-Content -LiteralPath $sd6 -Value 'already there'
    $f6 = MkFake $w6 @{ bytes = 3; files = 1; maxm = 0; sealMoves = @('x_holdout_y.txt') }
    Remove-Eligible $f6
    T 'sealed relocation refuses an existing destination: ERROR, source file and worktree untouched' (($f6.Status -eq 'ERROR') -and (Test-Path -LiteralPath $sf6) -and (Test-Path -LiteralPath $w6))
    # T7: tracked-dirty worktree: accepted only if a preserve ref holds exactly the current tracked state
    $w7 = MkWt 'wt_dirty'; $w7wip = MkWt 'wt_dirty_wip'
    Set-Content -LiteralPath (Join-Path $w7wip 'a.txt') -Value 'changed'
    [void](GitX $w7wip @('add', 'a.txt')); [void](GitX $w7wip @('-c', 'user.email=selftest@example.invalid', '-c', 'user.name=selftest', 'commit', '-q', '-m', 'wip'))
    Set-Content -LiteralPath (Join-Path $w7 'a.txt') -Value 'changed'
    $wd7 = MkWd 'wt_dirty' $w7; $wd7.preserve_ref = 'st-wt_dirty_wip'
    $script:WtRegistered = $null
    $r7 = Eval-Worktree $wd7
    T 'eval: dirty tracked state equal to the preserve ref -> ELIGIBLE with an "accepted" note' (($r7.Status -eq 'ELIGIBLE') -and ($r7.Reason -match 'accepted'))
    $wd7b = MkWd 'wt_dirty' $w7
    $script:WtRegistered = $null
    $r7b = Eval-Worktree $wd7b
    T 'eval: dirty tracked state without a preserving ref -> DIRTY_TRACKED' ($r7b.Status -eq 'DIRTY_TRACKED')
    # T8: the receipt says a sensitive-named file was skipped and none is left: it needs a MOVED record in MOVE_MANIFEST.csv
    $w8 = MkWt 'wt_sens_record'
    $wd8 = MkWd 'wt_sens_record' $w8; $wd8.sens_count = 1; $wd8.sens = @('notes_holdout_a.txt')
    $script:WtRegistered = $null; $script:SealedRows = $null
    $r8 = Eval-Worktree $wd8
    T 'eval: recorded sensitive-named file gone from the worktree but no sealed-move record -> SENSITIVE_UNACCOUNTED' ($r8.Status -eq 'SENSITIVE_UNACCOUNTED')
    $d8 = Join-Path $script:SealedRoot 'E\moved_here\notes_holdout_a.txt'; [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $d8)); Set-Content -LiteralPath $d8 -Value 'moved'
    [System.IO.File]::WriteAllText($script:SealedManifestPath, ("run,source,dest,kind,bytes,files,sha256,status`n{0},`"{1}`",`"{2}`",file,5,1,x,MOVED`n" -f 'selftest', (Join-Path $w8 'notes_holdout_a.txt'), $d8))
    $script:WtRegistered = $null; $script:SealedRows = $null
    $r8b = Eval-Worktree $wd8
    T 'eval: the same file with a MOVED row (destination present) -> ELIGIBLE' ($r8b.Status -eq 'ELIGIBLE')
    $script:SealedRows = $null
    # T9: Invoke-Stage1 -Execute end to end on scratch worktrees: dependency order, junction targets survive, a held worktree keeps
    #     the "last" directory (reorg3 stand-in) in place; then, once nothing else remains, the last directory goes
    $rd9 = Join-Path $base 'rdwt9'; New-Item -ItemType Directory -Path $rd9 -Force | Out-Null
    function MkWt9([string]$nm) { $p = Join-Path $rd9 $nm; [void](GitX $repo @('worktree', 'add', '-q', '-b', ('st9-' + $nm), $p)); return $p }
    function MkWd9([string]$nm, [string]$p) { $o = MkWd $nm $p; $o.branch = ('st9-' + $nm); return $o }
    $out9 = Join-Path $base 'out9'; New-Item -ItemType Directory -Path (Join-Path $out9 'd') -Force | Out-Null; Set-Content -LiteralPath (Join-Path $out9 'd\keep9.txt') -Value 'k'
    $x9b = MkWt9 'ex_b'; $x9a = MkWt9 'ex_a'; $x9c = MkWt9 'ex_c'; $x9d = MkWt9 'ex_dirty'
    New-Item -ItemType Directory -Path (Join-Path $x9b 'payload') -Force | Out-Null
    New-Item -ItemType Junction -Path (Join-Path $x9a 'into_b') -Target (Join-Path $x9b 'payload') | Out-Null
    New-Item -ItemType Junction -Path (Join-Path $x9a 'outside') -Target $out9 | Out-Null
    Set-Content -LiteralPath (Join-Path $x9d 'a.txt') -Value 'dirty and not preserved'
    $last9 = Join-Path $rd9 'reorg9'; New-Item -ItemType Directory -Path (Join-Path $last9 'lists') -Force | Out-Null
    $wl9 = MkWd9 'reorg9' $last9; $wl9.cls = 'plaindir'; $wl9.last = $true
    $wlist9 = @((MkWd9 'ex_b' $x9b), (MkWd9 'ex_a' $x9a), (MkWd9 'ex_c' $x9c), (MkWd9 'ex_dirty' $x9d), $wl9)
    $saveWts = $M.worktrees; $saveItems = $M.items; $saveUnrc = $M.unreceipted_dirs; $saveHold = $M.hold_until_flag
    $saveRdwt = $P.rdwt; $saveReorg = $P.reorg3
    $M.worktrees = $wlist9; $M.items = @(); $M.unreceipted_dirs = @(); $M.hold_until_flag = @()
    $P.rdwt = $rd9; $P.reorg3 = $last9
    $saveExec9 = $script:EXEC; $script:EXEC = $true
    $saveYes9 = $Yes; Set-Variable -Name Yes -Value $true -Scope Script
    $saveNR = $script:NoReport; $script:NoReport = $true; $script:NoOffloadMap = $true
    $script:Res = New-Object System.Collections.ArrayList; $script:WtLinkCache = @{}; $script:WtRegistered = $null; $script:ProcList = @()
    $script:ManCache = @{}
    Invoke-Stage1 *> $null
    $st9 = @{}; foreach ($r9 in $script:Res) { $st9[$r9.Id] = $r9.Status }
    Say ('  (scratch stage 1 statuses: ' + (($script:Res | ForEach-Object { $_.Id + '=' + $_.Status + $(if ($_.Status -ne 'DELETED' -and $_.Status -ne 'GONE') { ' [' + $_.Reason + ']' }) }) -join ' ; ') + ')') 'DarkGray'
    T 'stage 1 -Execute (scratch): eligible worktrees removed in dependency order, held worktree stays, "last" dir held while it is not alone' (($st9['wt:ex_a'] -eq 'DELETED') -and ($st9['wt:ex_b'] -eq 'DELETED') -and ($st9['wt:ex_c'] -eq 'DELETED') -and ($st9['wt:ex_dirty'] -eq 'DIRTY_TRACKED') -and ($st9['wt:reorg9'] -eq 'HELD_LAST') -and (Test-Path -LiteralPath $x9d) -and (Test-Path -LiteralPath $last9) -and -not (Test-Path -LiteralPath $x9a) -and -not (Test-Path -LiteralPath $x9b))
    T 'stage 1 -Execute (scratch): the junction target outside survived' (Test-Path -LiteralPath (Join-Path $out9 'd\keep9.txt'))
    $jt9 = JrnText
    $ia = $jt9.LastIndexOf('UNLINKED' + "`t" + $x9a); $ib = $jt9.LastIndexOf('DELETED' + "`t" + $x9b)
    T 'journal: ex_a was unlinked (before ex_b, the tree its junction points into, was deleted)' (($ia -ge 0) -and ($ib -gt $ia))
    [void](GitX $repo @('worktree', 'remove', '--force', $x9d))
    $M.worktrees = @($wl9)
    $script:Res = New-Object System.Collections.ArrayList; $script:WtLinkCache = @{}; $script:WtRegistered = $null; $script:ManCache = @{}
    Invoke-Stage1 *> $null
    $st9b = @{}; foreach ($r9 in $script:Res) { $st9b[$r9.Id] = $r9.Status }
    T 'stage 1 -Execute (scratch): when nothing else remains the "last" directory is removed' (($st9b['wt:reorg9'] -eq 'DELETED') -and -not (Test-Path -LiteralPath $last9))
    $M.worktrees = $saveWts; $M.items = $saveItems; $M.unreceipted_dirs = $saveUnrc; $M.hold_until_flag = $saveHold
    $P.rdwt = $saveRdwt; $P.reorg3 = $saveReorg
    $script:EXEC = $saveExec9; Set-Variable -Name Yes -Value $saveYes9 -Scope Script
    $script:NoReport = $saveNR; $script:NoOffloadMap = $false; $script:ProcList = $null
    # restore the globals touched above
    $script:Res = $saveRes3; $script:WtLinkCache = $saveLinkCache; $script:WtRegistered = $null
    $script:SealedRoot = $saveSealRoot; $script:SealedManifestPath = $saveSealMan
    Set-Variable -Name SkipReferenceScan -Value $saveSkip3 -Scope Script
    $M.protected.roots = $saveRoots3; $M.allowed_roots = $saveAllowed3; $M.keep_local = $saveKeep3
    $script:MainRepo = $saveMain
    $script:Freed['E'] = $before
    $script:JW.Dispose(); $script:JW = $saveJW; $script:JournalPath = $saveJP
    # cleanup
    $e3 = New-Object 'System.Collections.Generic.List[string]'
    try { $wl = [MC]::Walk($base, $false); foreach ($rl in $wl.Reparse) { [void](Invoke-Proc 'cmd.exe' @('/c', 'rmdir', (Join-Path $base ($rl -replace '/', '\'))) 60) } } catch { }
    try { [void][MC]::DeleteTree($base, $e3) } catch { Say ("cleanup problem: " + $_.Exception.Message) 'Red' }
    T 'scratch folder removed' (-not (Test-Path -LiteralPath $base))
    Say ("SELF TEST: {0} passed, {1} failed" -f $script:pass, $script:fail) $(if ($script:fail) { 'Red' } else { 'Green' })
}

# =====================================================================================================
#  main
# =====================================================================================================
try {
    if (-not (Test-Path -LiteralPath $Rclone)) { throw "rclone not found: $Rclone" }
    Jrn 'RUN_START' ("stage$Stage") 0 ("execute=$EXEC deep=$DEEP only=" + ($Only -join ',') + " skip=" + ($Skip -join ','))
    Say ("MORNING_CLEANUP  generated {0}   {1}" -f $M.generated, $(if ($EXEC) { 'EXECUTE MODE' } else { 'REPORT-ONLY (no -Execute): nothing will be changed' })) $(if ($EXEC) { 'Red' } else { 'Green' })
    if (-not $SelfTest -and ($Stage -eq 1 -or $RemoveOld)) {
        $a = Rcl @('lsf', '--max-depth', '1', 'gdrive:research-archive/') 120
        if ($a.rc -ne 0) { throw ("rclone cannot reach gdrive:research-archive/ : " + ($a.err -replace '\s+', ' ')) }
    }
    if ($SelfTest) { Invoke-SelfTest; return }
    if ($ExplainRefs) {
        # diagnostic: every literal in the work tree that names this path (or a parent/child of it), split into code and records
        Scan-Literals
        Say ("reference scan: {0} literals; files matching '{1}' are ignored" -f $script:Lit.Count, $(if ($IgnoreReferenceFiles) { $IgnoreReferenceFiles } else { [string]$M.ref_scan.ignore_files })) 'Cyan'
        $sp = Get-RefSplit $ExplainRefs
        Say ("{0}: {1} code reference(s) (these HOLD it), {2} record citation(s) (these do not)" -f $ExplainRefs, @($sp.code).Count, @($sp.rec).Count) 'Cyan'
        foreach ($e in @($sp.code | Select-Object -First 40)) { Say ("  CODE  {0}  ->  {1}" -f $e.file, $e.lit) 'Yellow' }
        foreach ($e in @($sp.rec | Select-Object -First 10)) { Say ("  RECORD {0}  ->  {1}" -f $e.file, $e.lit) 'DarkGray' }
        return
    }
    switch ($Stage) {
        1 { Invoke-Stage1 }
        2 { Invoke-Stage2 }
        3 { Invoke-Stage3 }
    }
} finally {
    if ($script:JW) { $script:JW.Dispose() }
}
