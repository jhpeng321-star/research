# gen_manifests_wrapper.ps1 -- detached driver for pack md5 manifests; writes logs/manifests.log and logs/manifests.done
$ErrorActionPreference = 'Continue'
$log = 'E:\rd-wt\reorg3\dataops\logs\manifests.log'
$done = 'E:\rd-wt\reorg3\dataops\logs\manifests.done'
Remove-Item -LiteralPath $done -ErrorAction SilentlyContinue
"[$(Get-Date -Format o)] START" | Out-File -LiteralPath $log -Encoding utf8
& 'E:\rd-wt\reorg3\dataops\scripts\run_pack_manifests.ps1' *>> $log
$rc = $LASTEXITCODE
"[$(Get-Date -Format o)] END rc=$rc" | Out-File -LiteralPath $log -Encoding utf8 -Append
"$rc" | Out-File -LiteralPath $done -Encoding ascii
