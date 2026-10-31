"""Read only corridor geometry; never scenario routes or driver assignments.

Cross-check the obsstate helper's completed netframe facts against the local
network and publish the exact geometric inputs used by the lane port.
"""
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / 'assets/research/highway_merge_shepherd_long/osm.net.xml.gz'
    with gzip.open(path, 'rb') as handle:
        net = ET.parse(handle).getroot()
    allowed = {'1191554636', '1487606179', '1487606178', '11553048#1'}
    lanes = {edge.attrib['id']: [dict(lane.attrib) for lane in edge.findall('lane')]
             for edge in net.findall('edge') if edge.attrib['id'] in allowed}
    # Shapes are irrelevant to the 1-D lane model.
    for rows in lanes.values():
        for lane in rows:
            lane.pop('shape', None)
    connections = [dict(c.attrib) for c in net.findall('connection')
                   if c.attrib['from'] in allowed]
    report = dict(net_path=str(path.relative_to(root)),
                  net_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  lanes=lanes, connections=connections,
                  helper='rb-shared/obsstate: netframe.py documented and verified geometry')
    out = root / 'tools/rblane/receipts/GEOMETRY_AUDIT.json'
    assert out.parent.is_dir() and not out.exists()
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
