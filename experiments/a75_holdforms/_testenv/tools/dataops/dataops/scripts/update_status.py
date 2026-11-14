"""Measure aggregate rate (archive bytes of receipts since SINCE / elapsed), refresh
_out/throughput.json, verified_for_deletion.json and DATA_SUMMARY.md."""
import datetime, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
SINCE = os.environ.get("DATAOPS_SINCE", "2026-10-10T01:13:15")
rd = os.path.join(WORK, "receipts")
tot = n = 0
for f in os.listdir(rd):
    if f.endswith(".json") and f != "COVERAGE.json":
        try:
            r = json.load(open(os.path.join(rd, f), encoding="utf-8"))
        except Exception:
            continue
        if r.get("check") == "passed" and r.get("timestamp", "") >= SINCE:
            tot += r.get("bytes", 0); n += 1
el = (datetime.datetime.now() - datetime.datetime.fromisoformat(SINCE)).total_seconds()
mbps = tot / 1e6 / max(el, 1)
json.dump({"mbps": round(mbps, 3), "source_bytes": tot, "units": n, "since": SINCE,
           "elapsed_s": round(el), "note": "source bytes verified / wall time (includes compress+verify, small-file latency)"},
          open(os.path.join(WORK, "_out", "throughput.json"), "w"), indent=1)
print("rate %.3f MB/s units=%d bytes=%.2f GB" % (mbps, n, tot / 1e9))
py = sys.executable
subprocess.run([py, os.path.join(HERE, "make_deletion_list.py")])
subprocess.run([py, os.path.join(HERE, "make_summary.py")])
