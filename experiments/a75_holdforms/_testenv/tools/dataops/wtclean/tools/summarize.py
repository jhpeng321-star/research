#!/usr/bin/env python3
"""summarize.py - regenerate WT_SUMMARY.md from inventory + receipts (read-only; touches only WT_SUMMARY.md)."""
import datetime
import glob
import json
import os
import re
import sys

sys.dont_write_bytecode = True
BASE = r"E:\rd-wt\reorg3\wtclean"


def rj(p):
    return json.load(open(p, encoding="utf-8-sig"))


def gib(n):
    return n / 2 ** 30


def main():
    inv = rj(BASE + r"\inventory\inventory.json")
    byname = {t["name"]: t for t in inv}
    rec = {}
    for f in glob.glob(BASE + r"\receipts\*.json"):
        try:
            r = rj(f)
            rec[r["name"]] = r
        except Exception:
            pass
    active = [t["name"] for t in inv if t.get("active")]
    elig = [t for t in inv if not t.get("active")]
    on_disk = set(os.listdir(r"E:\rd-wt"))
    new_unlisted = sorted(on_disk - set(byname))

    tar_ok, legacy_ok, legacy_cf, failed, pending, noupload = [], [], [], [], [], []
    for t in elig:
        n = t["name"]
        r = rec.get(n)
        if r is None:
            pending.append(t)
            continue
        st = r.get("status")
        if r.get("method") == "tar.zst" and st == "ok":
            tar_ok.append((t, r))
        elif r.get("method") == "none" and st == "ok":
            noupload.append((t, r))
        elif st == "ok":
            legacy_ok.append((t, r))
        elif st == "commit_failed" and (r.get("upload") or {}).get("exit") == 0 and (r.get("check") or {}).get("exit") == 0:
            legacy_cf.append((t, r))
        else:
            failed.append((t, r))
    # parked failures from the tar packer
    parked = {}
    for f in glob.glob(BASE + r"\receipts_failed\*.json"):
        try:
            r = rj(f)
            parked.setdefault(r["name"], []).append(r)
        except Exception:
            pass

    arch_bytes = sum(r["archive"]["size"] for _, r in tar_ok)
    mem_bytes = sum(r["archive"]["member_bytes"] for _, r in tar_ok)
    mem_count = sum(r["archive"]["member_count"] for _, r in tar_ok)
    up_secs = sum(r["archive"]["upload_seconds"] for _, r in tar_ok)
    # wall clock window of tar receipts
    ts = sorted(r["timestamp"] for _, r in tar_ok)
    agg = ""
    if len(ts) >= 2:
        t0 = datetime.datetime.fromisoformat(ts[0])
        t1 = datetime.datetime.fromisoformat(ts[-1])
        el = max((t1 - t0).total_seconds(), 1)
        agg = "%.2f MB/s aggregate archive bytes over the %.0f min between first and last tar receipt (%d archives)" % (
            arch_bytes / el / 1e6, el / 60, len(tar_ok))
    pend_all = pending + [t for t, _ in failed]
    pf = sum(t["uploadCount"] for t in pend_all)
    pb = sum(t["uploadBytes"] for t in pend_all)

    L = []
    L.append("# Worktree backup summary (wtclean)\n")
    L.append("Generated %s by tools/summarize.py (read-only over inventory/receipts).\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    L.append("## Counts (inventory snapshot of %d worktree-like dirs; %d active excluded)\n" % (len(inv), len(active)))
    L.append("| state | worktrees |")
    L.append("|---|---|")
    L.append("| verified, tar.zst (this takeover) | %d |" % len(tar_ok))
    L.append("| verified, no untracked content to back up | %d |" % len(noupload))
    L.append("| verified, per-file mirror (earlier uploader, status ok) | %d |" % len(legacy_ok))
    L.append("| per-file mirror ok but commit step reported failure (upload+check exit 0; accepted by template) | %d |" % len(legacy_cf))
    L.append("| failed / not passed | %d |" % len(failed))
    L.append("| pending (no receipt) | %d |" % len(pending))
    L.append("| eligible total | %d |" % len(elig))
    L.append("")
    L.append("tar.zst volume: %d members, %.2f GiB source -> %.2f GiB archives (%.1fx); upload %.2f MB/s per archive (sum of upload seconds %.0f s). %s\n"
             % (mem_count, gib(mem_bytes), gib(arch_bytes), (mem_bytes / arch_bytes) if arch_bytes else 0,
                (arch_bytes / up_secs / 1e6) if up_secs else 0, up_secs, agg))
    L.append("Pending/failed workload (inventory sizes): %d worktrees, %d files, %.2f GiB.\n" % (len(pend_all), pf, gib(pb)))
    if failed:
        L.append("## Failed / not passed\n")
        for t, r in failed:
            L.append("- %s: status=%s error=%s" % (t["name"], r.get("status"), (r.get("error") or "")[:200]))
        L.append("")
    parked = {n: rs for n, rs in parked.items()
              if (rec.get(n) or {}).get("status") != "ok" or (rec.get(n) or {}).get("method") != "tar.zst"}
    if parked:
        L.append("## Parked failed tar attempts (earlier receipt kept)\n")
        for n, rs in sorted(parked.items()):
            L.append("- %s: %d attempt(s); last: %s %s" % (n, len(rs), rs[-1].get("status"), (rs[-1].get("error") or "")[:160]))
        L.append("")
    if pending:
        L.append("## Pending\n")
        for t in sorted(pending, key=lambda t: t["uploadBytes"]):
            L.append("- %s: %d files, %.1f MiB" % (t["name"], t["uploadCount"], t["uploadBytes"] / 2 ** 20))
        L.append("")
    sens = [(t["name"], r.get("skippedSensitiveCount")) for t, r in tar_ok if r.get("skippedSensitiveCount")]
    junc = [(t["name"], len(r.get("junctions") or [])) for t, r in tar_ok if r.get("junctions")]
    dirty_sens = [(t["name"], r.get("trackedDirtySensitiveSkipped")) for t, r in tar_ok if r.get("trackedDirtySensitiveSkipped")]
    L.append("## Skips recorded in tar.zst receipts\n")
    L.append("- sensitive-named files skipped (names only, never opened/uploaded): %d worktrees, %d files total"
             % (len(sens), sum(c for _, c in sens)))
    L.append("- worktrees with junctions/symlinks (not followed): %s" % (", ".join("%s(%d)" % x for x in junc) or "none"))
    L.append("- tracked-dirty sensitive-named paths left uncommitted: %s" % (", ".join("%s(%d)" % (n, len(v)) for n, v in dirty_sens) or "none"))
    L.append("")
    L.append("## Active / not touched\n")
    L.append("- inventory active: " + ", ".join(active))
    L.append("- not in inventory (created later, treated ACTIVE): " + ", ".join(new_unlisted))
    L.append("")
    L.append("## Schema notes for E:/research-dense/tmp/morning/gen_cleanup.py\n")
    L.append("- tar receipts keep every legacy field (name,path,class,manual,branch,headBefore,headType,dirtyBefore,committed,commitSha,preserveRef,dest,listFile,uploadCount,uploadBytes,upload{ran,exit,log},check{ran,exit,result,log},junctions,skippedSensitiveCount/Sample,skippedJunctionCount,status,error,timestamp).")
    L.append("- added: method='tar.zst', archive{remote,size,md5,sha256,manifest_*,member_count,member_bytes,verified{..}}, memberCount, memberBytes, skippedSensitive (full list), junctionsFound, nestedRepoDirs.")
    L.append("- dest = gdrive:research-archive/worktrees/<name>; files on Drive: <name>.tar.zst and <name>.manifest.json (member path,size,sha256). listFile = lists/<name>.tar.txt (archived member paths).")
    L.append("- The template's per-file Drive proof (dest/<relpath> size+md5) does NOT apply to method='tar.zst' receipts: it must check the archive (size+md5 in receipt.archive) instead; see tools/patch_proposal/*.patch (unapplied proposal).")
    L.append("")
    L.append("## Takeover notes (static, written by the 06:15-09:30 takeover)\n")
    L.append("- Method: tools/pack_wt.py = commit tracked changes (non-sensitive paths) -> git ls-files --others [+ --ignored] (junction/symlink/sensitive filtered) -> tar(PAX)|zstd -T0 -10 with per-member sha256 -> local verify (zstd -dc | tar -t == manifest; streamed sha256 == manifest) -> rclone copy (archive + manifest) -> rclone md5sum + lsjson size check -> receipt. tools/run_queue.py runs <=3 in parallel; tools/verify_remote.py is an independent Drive round-trip (rclone cat | zstd -dc | sha256 per member vs manifest).")
    L.append("- Upload chunk size: 256M for archives <= 64 MB, 32M above (accurate progress for the stall watchdog); a Clash upstream node sometimes pins a connection to ~0.03-0.4 MB/s, so a watchdog kills OUR rclone and retries on a new connection.")
    L.append("- The earlier per-file uploader (powershell tools/process.ps1, PID 40236, started 23:00) was NOT killed; it keeps walking the same list. It collided on a few items (wm53, pl21, rc05, ws-review2, ws-swgen, ws-ranking, ws-eval, j4bfix...): when it finishes after a tar.zst receipt it overwrites it with a legacy per-file receipt (both are valid backups on Drive). Prior receipts are copied to receipts_prev/, failed tar attempts are parked in receipts_failed/.")
    L.append("- Keep-local per-file mirrors (generator dir_md5 proofs): done for sidecars_packv3_v2, identities_dspeed, identities_damping, labels_overlay_dspeed, labels_overlay_damping (tools/mirror_keeplocal.py). The two big packs (pack_dspeed 12 GB, pack_damping 5 GB) were started with tools/mirror_pack.py but STOPPED at ~9%/14% to give bandwidth to gpu1; they are inside the verified dspeed-data/damping-snap tar.zst archives. Re-run: python -I tools/mirror_pack.py dspeed-data _out/pack_dspeed 7 ; python -I tools/mirror_pack.py damping-snap _out/pack_damping 5.")
    L.append("- tools/git_precheck.py (read-only) found, among OLD per-file receipts: pl53 and pl57 have tracked changes again after their backup (still in use?); mcpclaude/mcpdrive/mcpopencode branches do not exist in the main repo (other repo).")
    L.append("")
    open(BASE + r"\WT_SUMMARY.md", "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
