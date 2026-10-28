"""Zero-spawn backend capability check. Never connects to or starts SUMO."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traci


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    args = ap.parse_args()
    names = ("getNativeResidualContract","setNativeAccelerationResidual")
    available = {name:callable(getattr(traci.vehicle,name,None)) for name in names}
    source = inspect.getsource(traci.vehicle.getSpeedWithoutTraCI)
    result = {"schema":"execparity_backend_preflight/1", "required_contract":"native_relative_same_step/1",
        "extension_methods_available":available,
        "stock_cached_getter_source_sha256":hashlib.sha256(source.encode()).hexdigest(),
        "simulator_started":False,"connected_to_simulator":False,
        "passed":all(available.values()),
        "status":"EXTENSION_AVAILABLE_REQUIRES_RUNTIME_CONTRACT_CHECK" if all(available.values()) else
                 "BLOCKED_SAME_STEP_NATIVE_RESIDUAL_BACKEND_MISSING"}
    assert args.out.parent.is_dir()
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
