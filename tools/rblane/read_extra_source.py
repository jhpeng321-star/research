"""Download two public headers for READING ONLY into a NEW empty directory."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle-implementation', action='store_true')
    args = parser.parse_args()
    parent = Path('E:/Temp/opencode')
    assert parent.is_dir()
    out = parent / ('sumo-1.27.1-lane-vehicle' if args.vehicle_implementation
                    else 'sumo-1.27.1-lane-extra')
    out.mkdir(exist_ok=False)
    assert not any(out.iterdir())
    receipts = []
    paths = ('src/microsim/MSVehicle.cpp',) if args.vehicle_implementation else (
        'src/microsim/MSVehicle.h', 'src/utils/common/StdDefs.h')
    for rel in paths:
        url = 'https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_27_1/' + rel
        with urlopen(url, timeout=30) as response:
            source = response.read()
        target = out / Path(rel).name
        target.write_bytes(source)
        receipts.append(dict(url=url, path=str(target), sha256=hashlib.sha256(source).hexdigest()))
    print(json.dumps(receipts, indent=2))


if __name__ == '__main__':
    main()
