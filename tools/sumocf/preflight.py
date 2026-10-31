"""No-SUMO runtime verification and seed guard receipt on cpuQ."""
import hashlib
import json
import os
from pathlib import Path
import sys

R = Path('/home/ubuntu/e3final')
sys.path[:0] = [str(R / 'runtime/src'), str(R / 'runtime')]


def main():
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    from wms.control.policy import load_policy_fn
    import torch
    torch.set_num_threads(1)
    seeds = list(range(969200, 969392)) + list(range(969600, 969625))
    for s in seeds:
        for d in (1900, 2400, 2700):
            require_development_case(s, d, 'exposed')
            check_case(s, d)
    path = Path('/home/ubuntu/mappospeed/production2/policy_latest.pt')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
    executor = load_policy_fn(path)
    assert executor.native_lanes and not executor.residual_on_native
    payload = torch.load(path, weights_only=True, map_location='cpu')
    m = R / 'runtime/SOURCE_MANIFEST.json'
    assert hashlib.sha256(m.read_bytes()).hexdigest() == 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'
    entries = json.loads(m.read_text())
    assert all(hashlib.sha256((R / 'runtime' / n).read_bytes()).hexdigest() == h
               for n, h in entries.items())
    print(json.dumps(dict(guards=len(seeds), cells_checked=len(seeds)*3,
                         source_files=len(entries), mappo_sha256=sha,
                         bounds=[executor.accel_min, executor.accel_max],
                         payload_metadata={k:v for k,v in payload.items()
                                           if isinstance(v, (str, int, float, bool))},
                         cores=os.cpu_count(), load=os.getloadavg()), indent=1))


if __name__ == '__main__':
    main()
