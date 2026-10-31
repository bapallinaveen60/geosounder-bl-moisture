"""PyTorch Training Engine for ThermoBL-Net.

Supports DistributedDataParallel (DDP), Automatic Mixed Precision (AMP),
gradient clipping, and composite physics loss tracking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.data.dataset import ThermoBLDataset, get_dataloader
from src.losses.composite_loss import CompositePhysicsLoss
from src.models.thermobonet import ThermoBLNet
from src.training.lr_scheduler import WarmupCosineScheduler


class ThermoBLTrainer:
    """Trainer orchestrator for ThermoBL-Net."""

    def __init__(
        self, config: Dict[str, Any], rank: int = 0, world_size: int = 1
    ) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

        # Setup compute device
        if torch.cuda.is_available():
            self.device = torch.device(
                f"cuda:{rank}" if self.is_distributed else "cuda"
            )
        else:
            self.device = torch.device("cpu")

        # Directories
        self.output_dir = Path(config["training"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Instantiate Model
        model_cfg = config.get("model", {})
        self.model = ThermoBLNet(
            in_channels=model_cfg.get("input_channels", 26),
            out_levels=model_cfg.get("output_levels", 5),
        ).to(self.device)

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        # 2. Instantiate Optimizer
        opt_cfg = config["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            betas=tuple(opt_cfg["betas"]),
        )

        # 3. Instantiate Scheduler
        sched_cfg = config["scheduler"]
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=sched_cfg["warmup_epochs"],
            total_epochs=sched_cfg["total_epochs"],
            min_lr=float(opt_cfg["min_lr"]),
        )

        # 4. Instantiate Physics Loss
        loss_w = config["loss_weights"]
        loss_p = config.get("loss_parameters", {})
        self.criterion = CompositePhysicsLoss(
            lambda_profile=float(loss_w["lambda_profile"]),
            lambda_column=float(loss_w["lambda_column"]),
            lambda_monotonic=float(loss_w["lambda_monotonic"]),
            lambda_conv=float(loss_w["lambda_conv"]),
            huber_delta=float(loss_p.get("huber_delta", 0.5)),
            monotonic_delta_q=float(loss_p.get("monotonic_delta_q", 0.2)),
            rho_w=float(loss_p.get("density_liquid_water", 1000.0)),
            g=float(loss_p.get("gravity_constant", 9.80665)),
        ).to(self.device)

        # Mixed Precision Scaler
        use_amp = (
            config["training"].get("mixed_precision", True)
            and torch.cuda.is_available()
        )
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.scaler = torch.amp.GradScaler(self.device_type, enabled=use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.grad_clip = float(config["training"].get("gradient_clipping", 1.0))
        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss_accum = 0.0
        prof_loss_accum = 0.0
        col_loss_accum = 0.0
        mono_loss_accum = 0.0
        conv_loss_accum = 0.0
        num_batches = 0

        for batch in dataloader:
            x_seq = batch["input_sequence"].to(self.device)
            target_q = batch["target_q"].to(self.device)
            sonde_mask = batch["target_mask"].to(self.device)
            gnss_pwv = batch["gnss_pwv"].to(self.device)
            pwv_mask = batch["pwv_mask"].to(self.device)
            winds_uv = batch["winds_uv"].to(self.device)
            upper_pwv = batch["upper_pwv"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device_type, enabled=self.scaler.is_enabled()
            ):
                q_pred = self.model(x_seq)
                losses = self.criterion(
                    q_pred=q_pred,
                    q_true=target_q,
                    sonde_mask=sonde_mask,
                    gnss_pwv_true=gnss_pwv,
                    upper_pwv_residual=upper_pwv,
                    gnss_mask=pwv_mask,
                    winds_uv=winds_uv,
                )
                loss = losses["total_loss"]

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_accum += loss.item()
            prof_loss_accum += losses["loss_profile"].item()
            col_loss_accum += losses["loss_column"].item()
            mono_loss_accum += losses["loss_monotonic"].item()
            conv_loss_accum += losses["loss_conv"].item()
            num_batches += 1

        self.scheduler.step()

        return {
            "total_loss": total_loss_accum / max(1, num_batches),
            "loss_profile": prof_loss_accum / max(1, num_batches),
            "loss_column": col_loss_accum / max(1, num_batches),
            "loss_monotonic": mono_loss_accum / max(1, num_batches),
            "loss_conv": conv_loss_accum / max(1, num_batches),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation hold-out set."""
        self.model.eval()
        val_loss_accum = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x_seq = batch["input_sequence"].to(self.device)
                target_q = batch["target_q"].to(self.device)
                sonde_mask = batch["target_mask"].to(self.device)
                gnss_pwv = batch["gnss_pwv"].to(self.device)
                pwv_mask = batch["pwv_mask"].to(self.device)
                winds_uv = batch["winds_uv"].to(self.device)
                upper_pwv = batch["upper_pwv"].to(self.device)

                with torch.amp.autocast(
                    device_type=self.device_type, enabled=self.scaler.is_enabled()
                ):
                    q_pred = self.model(x_seq)
                    losses = self.criterion(
                        q_pred=q_pred,
                        q_true=target_q,
                        sonde_mask=sonde_mask,
                        gnss_pwv_true=gnss_pwv,
                        upper_pwv_residual=upper_pwv,
                        gnss_mask=pwv_mask,
                        winds_uv=winds_uv,
                    )
                    val_loss_accum += losses["total_loss"].item()
                    num_batches += 1

        return {"val_total_loss": val_loss_accum / max(1, num_batches)}

    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False) -> None:
        """Save model weights and training metadata."""
        if self.rank != 0:
            return

        state = {
            "epoch": epoch,
            "model_state_dict": (
                self.model.module.state_dict()
                if self.is_distributed
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        torch.save(state, path)
        if is_best:
            best_path = path.parent / "best_model.pt"
            torch.save(state, best_path)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> None:
        """Run full multi-epoch optimization loop."""
        if self.rank == 0:
            print(
                f"Starting ThermoBL-Net Training for {epochs} epochs on device: {self.device}"
            )

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            val_loss = val_metrics["val_total_loss"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if self.rank == 0:
                print(
                    f"Epoch [{epoch:03d}/{epochs:03d}] "
                    f"Train Loss: {train_metrics['total_loss']:.4f} "
                    f"(Prof: {train_metrics['loss_profile']:.3f}, Col: {train_metrics['loss_column']:.3f}, Mono: {train_metrics['loss_monotonic']:.3f}, Conv: {train_metrics['loss_conv']:.3f}) | "
                    f"Val Loss: {val_loss:.4f} | LR: {train_metrics['lr']:.2e} "
                    f"{'[*] BEST' if is_best else ''}"
                )
                self.save_checkpoint(
                    self.output_dir / "latest_checkpoint.pt",
                    epoch=epoch,
                    is_best=is_best,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ThermoBL-Net")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    # Instantiate datasets for quick local test / dry-run
    train_dataset = ThermoBLDataset(
        num_samples=8, grid_shape=(32, 32), is_synthetic=True
    )
    val_dataset = ThermoBLDataset(num_samples=4, grid_shape=(32, 32), is_synthetic=True)

    train_loader = get_dataloader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0
    )
    val_loader = get_dataloader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    trainer = ThermoBLTrainer(config=train_cfg)
    trainer.fit(train_loader, val_loader, epochs=2)


if __name__ == "__main__":
    main()
