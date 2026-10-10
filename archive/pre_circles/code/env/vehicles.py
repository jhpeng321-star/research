"""Vehicle population management: seeded CAV selection and HDV driver types."""

from typing import Dict, List, Optional, Set

import numpy as np
import traci


class CAVSelector:
    """Select vehicles as CAVs using a seeded RNG at departure time.

    ``placement="scatter"`` (default): each vehicle is a CAV with probability
    ``penetration`` (the probe baseline). ``placement="block"``: CAVs arrive
    as contiguous runs of ``block_size`` consecutive vehicles, repeated every
    ``block_cycle`` departures — a moving gate whose formation (not just
    fraction) tests whether the 20% throttle is leaky (F1 probe). The cycle
    is scaled so the long-run CAV rate stays ~= penetration.
    """

    def __init__(
        self,
        penetration: float,
        seed: int = 42,
        placement: str = "scatter",
        block_size: int = 3,
        block_cycle: Optional[int] = None,
    ):
        if not 0.0 <= penetration <= 1.0:
            raise ValueError("penetration must be in [0, 1]")
        if placement not in ("scatter", "block"):
            raise ValueError(f"placement must be scatter|block, got {placement}")
        self.penetration = penetration
        self.placement = placement
        self.block_size = max(1, int(block_size))
        # Long-run CAV rate = block_size / block_cycle ~= penetration.
        self.block_cycle = block_cycle or max(
            int(round(self.block_size / max(penetration, 1e-6))), self.block_size
        )
        self._rng = np.random.default_rng(seed)
        self.cav_ids: Set[str] = set()
        self._departure_count = 0

    def reset(self, seed: int) -> None:
        """Reset the selector for a new episode."""
        self._rng = np.random.default_rng(seed)
        self.cav_ids.clear()
        self._departure_count = 0

    def consider(self, veh_id: str) -> bool:
        """Return True if the vehicle should be a CAV."""
        if self.placement == "block":
            # Contiguous runs: the first block_size vehicles of every cycle.
            pos = self._departure_count % self.block_cycle
            self._departure_count += 1
            is_cav = pos < self.block_size
        else:
            is_cav = self._rng.random() < self.penetration
        if is_cav:
            self.cav_ids.add(veh_id)
            return True
        return False

    def add_departed(self, departed_ids: List[str]) -> List[str]:
        """Process a list of newly departed vehicles and return newly selected CAVs."""
        new_cavs = []
        for veh_id in departed_ids:
            if veh_id in self.cav_ids:
                continue
            if self.consider(veh_id):
                new_cavs.append(veh_id)
        return new_cavs

    def add_departed_aggressive_first(
        self, departed_ids: List[str], type_lookup: Dict[str, str]
    ) -> List[str]:
        """Aggressive-first CAV selection (autopsy-driven, 2026-08-06).

        TTC events are amplified 5.5x by aggressive FOLLOWERs. This mode makes
        every aggressive-type vehicle a CAV (removing the amplifier from the
        human pool), and fills the remaining penetration quota from non-
        aggressive vehicles at p_nonagg = (pen - p_agg) / (1 - p_agg) so the
        overall CAV rate stays ~= penetration (p_agg = 0.10).
        """
        p_agg = 0.10
        p_nonagg = max(0.0, min(1.0, (self.penetration - p_agg) / (1.0 - p_agg)))
        new_cavs = []
        for veh_id in departed_ids:
            if veh_id in self.cav_ids:
                continue
            kind = type_lookup.get(veh_id, "")
            if kind == "aggressive_car":
                self.cav_ids.add(veh_id)
                new_cavs.append(veh_id)
            elif self._rng.random() < p_nonagg:
                self.cav_ids.add(veh_id)
                new_cavs.append(veh_id)
        return new_cavs

    def remove_arrived(self, arrived_ids: List[str]) -> None:
        """Remove arrived vehicles from the CAV set."""
        for veh_id in arrived_ids:
            self.cav_ids.discard(veh_id)

    def get_active_cavs(self, current_vehicles: List[str]) -> Set[str]:
        """Return the CAVs that are still in the network."""
        return self.cav_ids.intersection(current_vehicles)


class HDVTypeAssigner:
    """Assigns one of three driver types to each vehicle when it departs.

    Driver types (must match the vType IDs in hdv_types.add.xml):
      - aggressive_car
      - normal_car
      - cautious_car

    Proportions are passed as a list of three floats summing to 1.0, ordered as
    [aggressive, normal, cautious]. A seeded RNG keeps assignment reproducible.
    """

    DRIVER_TYPES = ["aggressive_car", "normal_car", "cautious_car"]

    def __init__(self, proportions: List[float], seed: int = 42):
        if len(proportions) != len(self.DRIVER_TYPES):
            raise ValueError(f"proportions must have {len(self.DRIVER_TYPES)} entries")
        total = float(sum(proportions))
        if not np.isclose(total, 1.0):
            raise ValueError(f"proportions must sum to 1.0, got {total}")
        self.proportions = [float(p) for p in proportions]
        self._rng = np.random.default_rng(seed)
        self._assigned: Dict[str, str] = {}

    def reset(self, seed: int) -> None:
        """Reset the assigner for a new episode."""
        self._rng = np.random.default_rng(seed)
        self._assigned.clear()

    def assign_type(self, veh_id: str) -> str:
        """Assign and apply a driver type to one vehicle. Returns the type name."""
        type_idx = int(self._rng.choice(len(self.DRIVER_TYPES), p=self.proportions))
        vtype = self.DRIVER_TYPES[type_idx]
        traci.vehicle.setType(veh_id, vtype)
        self._assigned[veh_id] = vtype
        return vtype

    def assign_all_departed(
        self,
        departed_ids: List[str],
        skip: Optional[Set[str]] = None,
    ) -> Dict[str, str]:
        """Assign types to all newly departed vehicles.

        Args:
            departed_ids: vehicles that entered the network in the last step.
            skip: vehicle IDs that should not be re-typed (e.g., already assigned).

        Returns:
            Dict mapping vehicle_id -> assigned vType name.
        """
        if skip is None:
            skip = set()
        assignments = {}
        for veh_id in departed_ids:
            if veh_id in self._assigned or veh_id in skip:
                continue
            vtype = self.assign_type(veh_id)
            assignments[veh_id] = vtype
        return assignments

    def get_assigned_types(self) -> Dict[str, str]:
        """Return a copy of all vehicle -> type assignments made so far."""
        return dict(self._assigned)

    def count_by_type(self) -> Dict[str, int]:
        """Return counts of each driver type assigned so far."""
        counts = {t: 0 for t in self.DRIVER_TYPES}
        for vtype in self._assigned.values():
            counts[vtype] = counts.get(vtype, 0) + 1
        return counts
