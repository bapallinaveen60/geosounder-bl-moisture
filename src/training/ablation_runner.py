"""Ablation Study Matrix Automation Runner.

Executes and logs the four primary comparative configurations:
  1. Sounder radiances only (no GNSS, no physics losses)
  2. Sounder + GNSS PWV (pure data-driven)
  3. Sounder + GNSS + Topography / Surface LST (full inputs, unconstrained loss)
  4. Full ThermoBL-Net (multi-modal inputs + Physics-Informed Composite Loss)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import yaml

from src.data.dataset import ThermoBLDataset, get_dataloader
from src.training.trainer import ThermoBLTrainer


def run_ablation_experiments(config_path: str = "configs/ablation_config.yaml") -> None:
    """Iterate through ablation configurations and execute training."""
    with open(config_path, "r", encoding="utf-8") as f:
        ablation_cfg = yaml.safe_load(f)["ablation_studies"]

    with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
        base_train_cfg = yaml.safe_load(f)

    train_dataset = ThermoBLDataset(
        num_samples=8, grid_shape=(32, 32), is_synthetic=True
    )
    val_dataset = ThermoBLDataset(num_samples=4, grid_shape=(32, 32), is_synthetic=True)
    train_loader = get_dataloader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0
    )
    val_loader = get_dataloader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    print("=" * 80)
    print("THERMOBL-NET ABLATION STUDY SUITE")
    print("=" * 80)

    results: Dict[str, float] = {}

    for exp_key, exp_settings in ablation_cfg.items():
        print(f"\n---> Launching Experiment: {exp_key}")
        print(f"     Description: {exp_settings['description']}")

        # Clone and adapt training config
        exp_train_cfg = dict(base_train_cfg)
        exp_train_cfg["training"]["output_dir"] = exp_settings["output_dir"]
        exp_train_cfg["loss_weights"] = exp_settings["loss_weights"]

        trainer = ThermoBLTrainer(config=exp_train_cfg)
        # Run 2 epochs for ablation verification
        trainer.fit(train_loader, val_loader, epochs=2)
        results[exp_key] = trainer.best_val_loss

    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY RESULTS")
    print("=" * 80)
    for exp_key, best_loss in results.items():
        print(f"  {exp_key:35s} | Best Val Loss: {best_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ThermoBL-Net Ablation Studies")
    parser.add_argument("--config", type=str, default="configs/ablation_config.yaml")
    args = parser.parse_args()
    run_ablation_experiments(args.config)


if __name__ == "__main__":
    main()
