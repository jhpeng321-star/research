# dirsize.ps1 <path> -> prints "files<TAB>bytes" for a tree, never following junctions/symlinks.
# Also prints any reparse points encountered (prefix REP>). Optional: -Filter regex of relative paths to EXCLUDE.
param(
  [Parameter(Mandatory=$true)][string]$Path,
  [string]$Exclude = ''
)
$ErrorActionPreference = 'SilentlyContinue'
$root = (Resolve-Path -LiteralPath $Path).Path
$stack = New-Object System.Collections.Stack
$stack.Push($root)
$files = 0L; $bytes = 0L
$ex = $null
if ($Exclude) { $ex = [regex]::new($Exclude, 'IgnoreCase') }
while ($stack.Count -gt 0) {
  $dir = $stack.Pop()
  foreach ($item in (Get-ChildItem -LiteralPath $dir -Force)) {
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
      Write-Output ("REP>`t" + $item.FullName + "`t" + $item.LinkType)
      continue
    }
    if ($item.PSIsContainer) { $stack.Push($item.FullName) }
    else {
      if ($ex -and $ex.IsMatch($item.FullName.Substring($root.Length+1))) { continue }
      $files++; $bytes += $item.Length
    }
  }
}
Write-Output ("TOTAL`tfiles=$files`tbytes=$bytes")
