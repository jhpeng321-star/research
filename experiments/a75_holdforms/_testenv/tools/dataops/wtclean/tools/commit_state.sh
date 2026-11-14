#!/bin/bash
# commit the wtclean state (excluding _tmp/ and locks/ via .git/info/exclude)
cd /e/rd-wt/reorg3/wtclean || exit 1
for p in inventory lists receipts tools manifests receipts_prev receipts_failed wt_progress.log WT_SUMMARY.md; do
  [ -e "$p" ] && git add "$p" 2>/dev/null
done
git commit -q -m "wtclean: ${1:-state snapshot}

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" 2>&1 | tail -1
git log --oneline | head -1
