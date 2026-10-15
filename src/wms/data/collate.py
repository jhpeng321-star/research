"""Batch collation of dataset records (keys, dtypes and shapes are a frozen contract)."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

ACTION_TIMELINE_KEYS = ("controller", "acceleration_mps2", "target_speed_mps", "control_dt_s")


def collate(records: list[dict]) -> dict[str, Tensor | list]:
    """Stack per-window records; optional keys appear only if every record carries them."""
    keys = ("states", "present", "slot_index")
    out = {key: torch.as_tensor(np.stack([r[key] for r in records])) for key in keys}
    out["states"] = out["states"].float()
    out["present"] = out["present"].bool()
    out["slot_index"] = out["slot_index"].long()
    if all("loss_valid" in r for r in records):
        out["loss_valid"] = torch.as_tensor(
            np.stack([r["loss_valid"] for r in records]), dtype=torch.bool)
    if all("vehicle_ids" in r for r in records):
        out["vehicle_ids"] = torch.as_tensor(
            np.stack([r["vehicle_ids"] for r in records]), dtype=torch.int64)
    if all("cav_teacher" in r for r in records):
        out["cav_teacher"] = torch.as_tensor(
            np.stack([r["cav_teacher"] for r in records]), dtype=torch.float32)
    if all("is_cav" in r for r in records):
        out["is_cav"] = torch.as_tensor(
            np.stack([r["is_cav"] for r in records]), dtype=torch.bool)
    if all("hist" in r for r in records):
        out["hist"] = torch.as_tensor(np.stack([r["hist"] for r in records])).float()
        out["hist_mask"] = torch.as_tensor(np.stack([r["hist_mask"] for r in records])).bool()
    if any("driver_hist" in r for r in records):
        if not all("driver_hist" in r and "driver_hist_mask" in r for r in records):
            raise ValueError("all mixed records must supply configured driver history")
        out["driver_hist"] = torch.as_tensor(
            np.stack([r["driver_hist"] for r in records])).float()
        out["driver_hist_mask"] = torch.as_tensor(
            np.stack([r["driver_hist_mask"] for r in records])).bool()
    if all("entry_feat" in r for r in records):
        out["entry_feat"] = torch.as_tensor(
            np.stack([r["entry_feat"] for r in records]), dtype=torch.float32)
    if all("controller_id" in r for r in records):
        out["controller_id"] = torch.as_tensor(
            [int(r["controller_id"]) for r in records], dtype=torch.long)
    if all("action_timeline" in r for r in records):
        out["action_timeline"] = {
            key: (torch.as_tensor(np.stack([r["action_timeline"][key] for r in records]))
                  if key != "control_dt_s" else float(records[0]["action_timeline"][key]))
            for key in ACTION_TIMELINE_KEYS}
        if all("command_known" in r["action_timeline"] for r in records):
            # DATAHYG strict: the issued command channels are recorded only where True.
            out["action_timeline"]["command_known"] = torch.as_tensor(
                np.stack([r["action_timeline"]["command_known"] for r in records]),
                dtype=torch.bool)
    for key in ("road_present", "road_continue", "road_valid"):
        # entries-v2 windows: recorded on-road presence for the C4 aggregate loss
        if all(key in r for r in records):
            out[key] = torch.as_tensor(np.stack([r[key] for r in records]), dtype=torch.bool)
    if all("entrant" in r for r in records):
        out["entrant"] = torch.as_tensor(
            np.stack([r["entrant"] for r in records]), dtype=torch.bool)
    if all("pair_group" in r for r in records):
        # PAIRLOSS paired batch: pair group id and the batch row of the group's reference arm
        out["pair_group"] = torch.as_tensor([int(r["pair_group"]) for r in records],
                                            dtype=torch.long)
        out["pair_ref"] = torch.as_tensor([int(r["pair_ref"]) for r in records],
                                          dtype=torch.long)
        if all(r.get("meta", {}).get("world") is not None for r in records):
            out["pair_world"] = [r["meta"]["world"] for r in records]
        if all(r.get("meta", {}).get("pair_key") is not None for r in records):
            out["pair_key"] = [r["meta"]["pair_key"] for r in records]
    out["identities"] = [r.get("identities") for r in records]
    return out
