"""JSON-in / JSON-out CLI: ``python -m wm_api score --ckpt DIR --request R``.

Request JSON (see WM_API.md for the full schema)::

    {"state": {...}, "candidate_plans": [...], "horizon_s": 300,
     "members": null, "n_samples": 1, "seed": 0}

Response: the ``ScoreResult`` JSON on stdout (``--out`` writes it to a
file instead).  Refusals exit 2 with ``{"error": {"type", "message"}}``;
unexpected errors exit 1.  The CLI runs CPU-only, 2 torch threads, and
sets BelowNormal process priority on Windows best-effort.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wm_api.errors import WmApiError


def _lower_priority_windows() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)
    except Exception:
        pass


def _example_request() -> dict:
    return {
        "state": {
            "ids": ["cav0", "cav1", "hdv0", "hdv1"],
            "lane_phys": [1, 0, 1, 2],
            "s_down_m": [-120.0, -80.0, -60.0, 40.0],
            "v_mps": [20.0, 19.0, 21.0, 18.0],
            "a_mps2": [0.0, 0.0, 0.0, 0.0],
            "length_m": [5.0, 5.0, 5.0, 5.0],
            "is_cav": [True, True, False, False],
            "t0_s": 540.0,
            "entrance_counters": None,
            "obstacles": [],
        },
        "candidate_plans": [
            {"actions": {}},
            {"actions": {"cav0": [{"time_s": 0.0, "action": "damping"},
                                  {"time_s": 60.0, "action": "native"}],
                         "cav1": "damping"}},
            {"actions": {},
             "wall": {"wall_id": 1, "speed_mps": 6.5, "hold_s": 120.0,
                      "members": ["cav0", "cav1"]}},
        ],
        "horizon_s": 300,
        "members": None,
        "n_samples": 1,
        "seed": 0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wm_api",
        description="WM scoring API (api_version 1.0): score candidate "
                    "per-CAV plans from a t0 snapshot with the Idea A "
                    "bundle. Zero SUMO; deterministic CPU mode.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_score = sub.add_parser(
        "score", help="score one request (JSON in, ScoreResult JSON out)")
    p_score.add_argument("--ckpt", default=None,
                         help="member-checkpoint directory (member_*.pt + "
                              "optional SHA256SUMS)")
    p_score.add_argument("--request", default=None, help="request JSON")
    p_score.add_argument("--out", default=None,
                         help="write the response JSON here (default stdout)")
    p_score.add_argument("--threads", type=int, default=2, choices=(1, 2),
                         help="torch CPU threads (default 2)")
    p_score.add_argument("--no-sha-verify", action="store_true",
                         help="skip SHA256SUMS verification (hashes are "
                              "still computed and reported)")
    p_score.add_argument("--allow-other-horizon", action="store_true",
                         help="permit horizon_s != 300 (nearest tau "
                              "reported)")
    p_score.add_argument("--example", action="store_true",
                         help="print an example request JSON and exit")
    args = parser.parse_args(argv)
    if args.example:
        print(json.dumps(_example_request(), indent=2))
        return 0
    if not args.ckpt or not args.request:
        parser.error("score needs --ckpt and --request (or --example)")
    _lower_priority_windows()
    try:
        import torch

        torch.set_num_threads(int(args.threads))
    except Exception:
        pass
    try:
        from wm_api import API_VERSION, WmScorer

        request = json.loads(
            Path(args.request).read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        scorer = WmScorer(args.ckpt, threads=args.threads,
                          verify_sha=not args.no_sha_verify)
        kwargs = {}
        if args.allow_other_horizon:
            kwargs["allow_other_horizon"] = True
        result = scorer.score(
            request["state"],
            request["candidate_plans"],
            float(request.get("horizon_s", 300)),
            members=request.get("members"),
            n_samples=request.get("n_samples"),
            seed=request.get("seed"),
            **kwargs)
        text = result.to_json_text()
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(json.dumps({"ok": True, "api_version": API_VERSION,
                              "out": str(args.out)}))
        else:
            print(text)
        return 0
    except Exception as exc:
        code = 2 if isinstance(exc, WmApiError) else 1
        print(json.dumps({"error": {"type": type(exc).__name__,
                                    "message": str(exc),
                                    "api_version": API_VERSION}},
                         default=str, indent=2))
        return code


if __name__ == "__main__":
    sys.exit(main())
