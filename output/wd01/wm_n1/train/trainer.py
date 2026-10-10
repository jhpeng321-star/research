"""WM-N1 Trainer: one ensemble member with checkpoint/resume.

The aggregation of the losses is ``model.losses.total_loss`` (which already
adds the paired contrast term when ``targets["contrast"]`` is supplied).  Early
stopping uses the world-grouped CAL loss (WM_N1_DESIGN.md:121-122); CAL never
contributes gradients.

Checkpoints are plain ``state_dict``s plus a JSON config sidecar; every
``torch.load`` sets ``weights_only`` explicitly so torch 2.5.1 (default False)
and newer torch (default True) agree.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from model import losses as L
from model.net import build_member, parameter_count

from . import config as config_mod
from .dataset import MemmapCorpus, collate, contrast_batch_indices

CHECKPOINT_FORMAT = "wm_n1_member/1"


def _resolve_device(device) -> torch.device:
    dev = torch.device(device or "cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


class Trainer:
    """Train a single :class:`model.net.WmN1Net` member."""

    def __init__(self, config, corpus, *, member: int, seed: int,
                 device: str | None = None):
        """``member`` is 1..5; ``seed`` is recorded before training (A1)."""
        self.config = config
        self.corpus = corpus
        self.member = int(member)
        if not 1 <= self.member <= int(config.n_members):
            raise ValueError(
                f"member {self.member} outside 1..{config.n_members}")
        self.seed = int(seed)
        self.device = _resolve_device(device or config.device)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model = build_member(self.seed, config.model_config())
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay)
        self.epoch = 0
        self.step = 0
        self.losses: list = []
        self.history: list = []
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(self.seed)

    # ------------------------------------------------------------ batching --
    def _iter_batches(self):
        """Yield lists of dataset items, forever (one epoch per outer loop)."""
        n = len(self.corpus)
        if n == 0:
            return
        bs = max(1, int(self.config.batch_size))
        while True:
            if getattr(self.corpus, "pair_sampling", False):
                order = list(range(n))
            else:
                order = torch.randperm(n, generator=self.gen).tolist()
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                if not idx:
                    continue
                yield [self.corpus[j] for j in idx]

    def _targets(self, batch):
        targets = {
            "accel": batch["accel"],
            "target_mask": batch["target_mask"],
            "lane": batch["lane"],
            "arrivals": batch["arrivals"],
            "outcome_quantiles": batch["outcome_quantiles"],
            "outcome_mask": batch["outcome_mask"],
            "collision": batch["collision"],
        }
        contrast = contrast_batch_indices(batch["meta"])
        if int(contrast["a"].numel()) > 0:
            targets["contrast"] = contrast
        return targets

    def _forward_loss(self, items, *, train: bool):
        batch = collate(items)
        args = {k: batch[k].to(self.device)
                for k in ("veh_step", "veh_mask", "plan", "wall", "wall_mask",
                          "counter")}
        targets = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                   for k, v in self._targets(batch).items()}
        was_training = self.model.training
        self.model.train(train)
        try:
            pred = self.model(**args)
            return L.total_loss(pred, targets, self.config.loss_weights())
        finally:
            self.model.train(was_training)

    # -------------------------------------------------------------- train ---
    def fit(self) -> dict:
        """Run the schedule; return ``{"member", "seed", "steps", "losses"}``.

        Writes ``member_<k>.pt`` (state_dict + config + seed) under
        ``config.out_dir`` and a ``TRAIN_RECEIPT.json`` (seeds recorded).
        """
        self.model.train()
        batches = self._iter_batches()
        best_cal = None
        stale = 0
        for epoch in range(self.epoch, int(self.config.epochs)):
            epoch_losses = []
            for _ in range(int(self.config.steps_per_epoch)):
                items = next(batches, None)
                if items is None:
                    break
                loss = self._train_step(items)
                epoch_losses.append(loss)
                self.losses.append(loss)
                self.step += 1
            self.epoch = epoch + 1
            mean = (sum(epoch_losses) / len(epoch_losses)
                    if epoch_losses else float("nan"))
            self.history.append({"epoch": self.epoch, "mean_loss": mean})
            if self.config.calibrate_on_cal:
                cal = self.evaluate_cal()
                value = cal.get("loss")
                if value is not None:
                    if best_cal is None or value < best_cal:
                        best_cal, stale = value, 0
                    else:
                        stale += 1
                        if stale >= int(self.config.early_stop_patience):
                            break
        ckpt = self.save_checkpoint(
            Path(self.config.out_dir) / f"member_{self.member}.pt")
        self._write_receipt(ckpt)
        return {"member": self.member, "seed": self.seed, "steps": self.step,
                "losses": list(self.losses), "history": list(self.history),
                "checkpoint": str(ckpt), "parameters": parameter_count(self.model)}

    def _train_step(self, items) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._forward_loss(items, train=True)
        loss.backward()
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           float(self.config.grad_clip))
        self.optimizer.step()
        return float(loss.detach())

    # --------------------------------------------------------- checkpoint ---
    def save_checkpoint(self, path) -> Path:
        """Persist model/optimizer/scheduler/epoch and the recorded seed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": int(self.epoch),
            "step": int(self.step),
            "member": self.member,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "gen_state": self.gen.get_state(),
        }
        torch.save(payload, path)
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps({
            "format": CHECKPOINT_FORMAT, "member": self.member,
            "seed": self.seed, "epoch": int(self.epoch), "step": int(self.step),
            "config": self.config.to_dict()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return path

    def resume(self, path) -> "Trainer":
        """Restore from ``path`` and continue the same member/seed."""
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"checkpoint {path} format {ckpt.get('format')!r} != "
                f"{CHECKPOINT_FORMAT!r}")
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        self.epoch = int(ckpt.get("epoch", 0))
        self.step = int(ckpt.get("step", 0))
        self.member = int(ckpt.get("member", self.member))
        self.seed = int(ckpt.get("seed", self.seed))
        if "gen_state" in ckpt:
            self.gen.set_state(ckpt["gen_state"])
        return self

    # ------------------------------------------------------------ cal eval --
    def evaluate_cal(self) -> dict:
        """Diagnostic CAL metrics only; must not change hyperparameters."""
        cal = self.corpus
        if getattr(cal, "split", "TRAIN") != "CALIBRATION":
            pack_dir = getattr(cal, "pack_dir", None)
            if pack_dir is None:
                return {"status": "unavailable", "loss": None}
            try:
                cal = MemmapCorpus(pack_dir, split="CALIBRATION")
            except Exception as exc:                       # noqa: BLE001
                return {"status": "unavailable", "loss": None, "error": str(exc)}
        if len(cal) == 0:
            return {"status": "empty", "loss": None, "n": 0}
        was_training = self.model.training
        self.model.eval()
        total, n = 0.0, 0
        try:
            with torch.no_grad():
                for i in range(len(cal)):
                    loss = self._forward_loss([cal[i]], train=False)
                    total += float(loss.detach())
                    n += 1
        finally:
            self.model.train(was_training)
        return {"status": "ok", "member": self.member, "seed": self.seed,
                "n": n, "loss": total / max(1, n)}

    # ------------------------------------------------------------- receipt --
    def _write_receipt(self, checkpoint: Path) -> Path:
        out = Path(self.config.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt = {
            "member": self.member, "seed": self.seed,
            "member_order": list(config_mod.MEMBER_ORDER),
            "steps": self.step, "epochs": self.epoch,
            "checkpoint": str(checkpoint),
            "parameters": parameter_count(self.model),
            "config": self.config.to_dict(),
        }
        path = out / "TRAIN_RECEIPT.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        return path
