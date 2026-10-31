"""File hand-off for the WM 60 s direction screen (WM runs on another host, e.g. gpuA).

1. cpuQ writes ``screens/request_uXX.json`` (update, theta, sigma, sensitivity, the 8
   private-generator directions) from the resumable driver state - no SUMO.
2. The WM screen producer (frozen WM2 b8797242 + learned mix) reads it and writes
   ``screens/screen_uXX.json`` = {update, request_sha256, kept, predicted_gain,
   rejected, all_gains, rule, per_start}.
3. ``FileScreener.screen`` refuses unless the request bytes, update, theta, sigma and
   directions match exactly what the driver is about to evaluate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def request_payload(update: int, theta: torch.Tensor, sigma: float, sensitivity: float,
                    directions: torch.Tensor) -> dict:
    return dict(schema="sumoes_screen_request/1", update=int(update),
                theta=[float(v) for v in theta.double().tolist()], sigma=float(sigma),
                sensitivity=float(sensitivity),
                directions=[[float(v) for v in row] for row in directions.double().tolist()])


def write_request(folder: Path, payload: dict) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, allow_nan=False) + "\n"
    path = folder / f"request_u{payload['update']:02d}.json"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"{path} exists with different content; never overwrite")
    else:
        path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


class FileScreener:
    def __init__(self, folder: str | Path, keep: int = 3) -> None:
        self.folder, self.keep = Path(folder), keep

    def screen(self, theta, sigma, directions, update):
        request_path = self.folder / f"request_u{update:02d}.json"
        text = request_path.read_text()
        request = json.loads(text)
        expected = request_payload(update, theta, sigma, request["sensitivity"], directions)
        if request != expected:
            raise ValueError(f"screen request u{update} does not match the driver state")
        result = json.loads((self.folder / f"screen_u{update:02d}.json").read_text())
        if update >= 1 and (result.get('schema') != 'sumoes_screen/2'
                            or result.get('amendment') != 'MAIN1440'):
            raise ValueError('updates 1+ require the MAIN1440 cap-relative braking screen')
        if (result.get("update") != update
                or result.get("request_sha256") != hashlib.sha256(text.encode()).hexdigest()):
            raise ValueError(f"screen result u{update} is not bound to its request")
        kept = [int(i) for i in result["kept"]]
        if len(kept) > self.keep or len(set(kept)) != len(kept) or any(
                not 0 <= i < directions.shape[0] for i in kept):
            raise ValueError("invalid kept direction list")
        return dict(kept=kept, predicted_gain=result.get("predicted_gain"),
                    rejected=result.get("rejected", []), all_gains=result.get("all_gains"),
                    rule=result.get("rule"))
