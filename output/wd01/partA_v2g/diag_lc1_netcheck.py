"""One-off read-only inspection of the frozen network topology for the
WD-01-PA2G-LC1 counting audit (independent of partA_data/pdl.py)."""
import gzip
import re
from pathlib import Path

NET = Path(__file__).resolve().parents[3] / "assets" / "research" / \
    "highway_merge_shepherd_long" / "osm.net.xml.gz"
txt = gzip.open(NET, "rt", encoding="utf-8").read()

print("--- all connections FROM edge 1487606179:")
for m in re.finditer(r'<connection from="1487606179"[^>]*/>', txt):
    print(m.group(0))
print("--- all connections INTO edge 1487606178:")
for m in re.finditer(r'<connection[^>]*to="1487606178"[^>]*/>', txt):
    print(m.group(0))
print("--- edges (id, lanes, length):")
for m in re.finditer(r'<edge id="([^"]+)"([^>]*)>(.*?)</edge>', txt, re.S):
    eid, attrs, body = m.group(1), m.group(2), m.group(3)
    if eid not in ("1191554636", "1487606179", "1487606178", "11553048#1"):
        continue
    lanes = re.findall(r'<lane id="([^"]+)"[^>]*length="([^"]+)"', body)
    print(eid, re.sub(r"\s+", " ", attrs)[:120], lanes)
print("--- junctions (incLanes mention our edges):")
for m in re.finditer(r'<junction id="([^"]+)"[^>]*incLanes="([^"]*)"', txt):
    jid, inc = m.group(1), m.group(2)
    if "1487606179" in inc or "11553048" in inc:
        print(jid, "| inc:", inc)
