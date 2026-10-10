# upload_driver_wrapper.ps1 -- detached driver runner. Writes logs/upload.log and logs/upload.done
$ErrorActionPreference = 'Continue'
$log = 'E:\rd-wt\reorg3\dataops\logs\upload.log'
$done = 'E:\rd-wt\reorg3\dataops\logs\upload.done'
Remove-Item -LiteralPath $done -ErrorAction SilentlyContinue
"[$(Get-Date -Format o)] START upload driver" | Out-File -LiteralPath $log -Encoding utf8
& 'C:\Python314\python.exe' 'E:\rd-wt\reorg3\dataops\scripts\upload_driver.py' --queue 'E:\rd-wt\reorg3\dataops\_out\queue.json' --max-seconds 7200 *>> $log
$rc = $LASTEXITCODE
# refresh the aggregate INDEX from receipts and push it to Drive
& 'C:\Python314\python.exe' 'E:\rd-wt\reorg3\dataops\scripts\build_index.py' *>> $log
& 'C:\Python314\python.exe' 'E:\rd-wt\reorg3\dataops\scripts\push_index.py' *>> $log
& 'C:\Python314\python.exe' 'E:\rd-wt\reorg3\dataops\scripts\make_summary.py' *>> $log
"[$(Get-Date -Format o)] END rc=$rc" | Out-File -LiteralPath $log -Encoding utf8 -Append
"$rc" | Out-File -LiteralPath $done -Encoding ascii
