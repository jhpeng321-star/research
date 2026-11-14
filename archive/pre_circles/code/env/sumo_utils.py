"""TraCI helpers for starting/closing SUMO and querying vehicles."""

import socket
from pathlib import Path
from typing import List, Optional

import traci


def find_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def start_sumo(
    sumo_binary: str,
    sumo_config: str,
    project_root: Path,
    port: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
    max_retries: int = 8,
    seed: Optional[int] = None,
) -> int:
    """Start a SUMO process with TraCI and return the TraCI connection port."""
    # traci.start will append the --remote-port option, so do not include it here.
    sumo_cmd = [sumo_binary, "-c", str(project_root / sumo_config)]
    if seed is not None:
        # Pass the seed through to SUMO so TraCI runs are reproducible and can
        # be paired with the seeded pure-SUMO baseline runs.
        sumo_cmd += ["--seed", str(seed)]
    if extra_args:
        sumo_cmd.extend(extra_args)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        close_sumo()
        use_port = port if (port is not None and attempt == 0) else find_free_port()
        try:
            traci.start(sumo_cmd, port=use_port)
            return use_port
        except Exception as exc:  # noqa: BLE001 - retry on TraCI/port races
            last_err = exc
            close_sumo()
            port = None
    raise RuntimeError(f"Failed to start SUMO after {max_retries} tries") from last_err


def close_sumo() -> None:
    """Close the active TraCI connection, if any."""
    try:
        if traci.isLoaded():
            traci.close()
    except Exception:
        pass


def get_active_vehicle_ids() -> List[str]:
    """Return IDs of all vehicles currently in the simulation."""
    return list(traci.vehicle.getIDList())


def vehicle_in_network(veh_id: str) -> bool:
    """Return True if the vehicle is still in the simulation."""
    return veh_id in traci.vehicle.getIDList()


def get_departed_ids() -> List[str]:
    """Return vehicles that entered the network during the last step."""
    return list(traci.simulation.getDepartedIDList())


def get_arrived_ids() -> List[str]:
    """Return vehicles that left the network during the last step."""
    return list(traci.simulation.getArrivedIDList())
