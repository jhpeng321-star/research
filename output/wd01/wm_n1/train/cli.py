"""WM-N1 training CLI.

    python -m train --member 1 --seed 101 --config cfg.json --corpus PACK \
        --out RUN_DIR [--resume CKPT] [--device cuda]

The member seed is recorded in ``TRAIN_RECEIPT.json`` before any step; member 1
trains alone first, then members 2-5 (REGISTRATION_WM_N1_AMENDMENT_1.md A1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MEMBER_ORDER, TrainConfig
from .dataset import MemmapCorpus
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train")
    p.add_argument("--member", type=int, required=True, choices=(1, 2, 3, 4, 5))
    p.add_argument("--seed", type=int, required=True,
                   help="recorded before training (amendment A1)")
    p.add_argument("--config", default=None)
    p.add_argument("--corpus", required=True, help="packed memmap dir")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    return p


def record_seed(out_dir, member: int, seed: int) -> str:
    """Write ``SEEDS.json`` (member -> seed) before the first step."""
    path = Path(out_dir) / "SEEDS.json"
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    data[str(int(member))] = int(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return str(path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig.load(args.config) if args.config else TrainConfig()
    config.out_dir = args.out
    if args.device:
        config.device = args.device
    if args.resume:
        config.resume = args.resume

    # Seed recorded BEFORE any training step (amendment A1).
    record_seed(config.out_dir, args.member, args.seed)
    config.save(Path(config.out_dir) / "TRAIN_CONFIG.json")

    corpus = MemmapCorpus(args.corpus, split="TRAIN", pair_sampling=True)
    trainer = Trainer(config, corpus, member=args.member, seed=args.seed,
                      device=args.device)
    if args.resume:
        trainer.resume(args.resume)
    result = trainer.fit()
    summary = {"status": "ok", "member": result["member"],
               "seed": result["seed"], "steps": result["steps"],
               "member_order": list(MEMBER_ORDER),
               "checkpoint": result["checkpoint"]}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
