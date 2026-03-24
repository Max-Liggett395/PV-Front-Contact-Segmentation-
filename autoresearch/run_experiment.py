"""AutoResearch experiment runner.

Trains a model from an experiment config and writes standardized results JSON.
Designed to run inside a RunPod pod or any GPU environment.

Usage:
    python -m autoresearch.run_experiment \
        --experiment configs/experiment/baseline.yaml \
        --run-name exp-001-dice-loss \
        --max-epochs 200 \
        --patience 50 \
        --output-dir /results
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import SEMDataModule
from src.models import create_model
from src.models.losses import create_loss
from src.training import Trainer
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="AutoResearch experiment runner")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="autoresearch/results")
    args = parser.parse_args()

    start_time = time.time()

    # Load configs
    exp_cfg = load_config(args.experiment)
    data_cfg = load_config(os.path.join("configs", exp_cfg["data"] + ".yaml"))
    model_cfg = load_config(os.path.join("configs", exp_cfg["model"] + ".yaml"))

    # Create run directory
    run_dir = os.path.join("logs", "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Training config with overrides
    training_cfg = exp_cfg.get("training", {})
    training_cfg["checkpoint_dir"] = os.path.join(run_dir, "checkpoints")
    training_cfg["log_dir"] = os.path.join(run_dir, "tensorboard")
    training_cfg["num_classes"] = model_cfg.get("num_classes", 6)
    training_cfg["patience"] = args.patience
    training_cfg["max_epochs"] = args.max_epochs

    # Pass in_channels from model config to data config
    data_cfg["in_channels"] = model_cfg.get("in_channels", 1)
    data_overrides = exp_cfg.get("data_overrides", {})
    data_cfg.update(data_overrides)

    # Data
    dm = SEMDataModule(data_cfg)
    dm.setup()

    # Model
    model = create_model(model_cfg)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_cfg['architecture']} | Params: {param_count:,}")

    # Loss — use the factory from src/models/losses.py
    loss_cfg = exp_cfg.get("loss", {})
    loss_fn = create_loss(loss_cfg)

    # Optimizer
    opt_cfg = exp_cfg.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.get("lr", 5e-5),
            weight_decay=opt_cfg.get("weight_decay", 1e-2),
        )
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-2),
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.get("lr", 5e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )

    # Scheduler
    sched_cfg = exp_cfg.get("scheduler", None)
    scheduler = None
    if sched_cfg:
        sched_type = sched_cfg.get("type", "").lower()
        if sched_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 10),
                min_lr=sched_cfg.get("min_lr", 1e-7),
            )
        elif sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.max_epochs,
                eta_min=sched_cfg.get("min_lr", 1e-7),
            )
        elif sched_type == "cosine_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=sched_cfg.get("T_0", 50),
                T_mult=sched_cfg.get("T_mult", 2),
                eta_min=sched_cfg.get("min_lr", 1e-7),
            )
        elif sched_type == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt_cfg.get("lr", 5e-5) * 10,
                epochs=args.max_epochs,
                steps_per_epoch=len(dm.train_dataloader()),
            )

    # Train
    trainer = Trainer(model, loss_fn, optimizer, scheduler, training_cfg)
    best_metric = trainer.fit(
        dm.train_dataloader(),
        dm.val_dataloader(),
        max_epochs=args.max_epochs,
    )

    elapsed = time.time() - start_time

    # Load best checkpoint and extract final metrics
    best_ckpt_path = os.path.join(training_cfg["checkpoint_dir"], "best.pt")
    results = {
        "run_name": args.run_name,
        "experiment_config": args.experiment,
        "model": model_cfg["architecture"],
        "param_count": param_count,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "best_val_loss": float(best_metric),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    # Extract best metrics from checkpoint
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        metrics = ckpt.get("metrics", {})
        results["best_epoch"] = ckpt.get("epoch", -1)
        results["val_miou"] = float(metrics.get("miou", 0))
        results["val_f1_macro"] = float(metrics.get("f1_macro", 0))
        results["pixel_accuracy"] = float(metrics.get("pixel_accuracy", 0))
        results["per_class_iou"] = [float(x) for x in metrics.get("per_class_iou", [])]
        results["val_loss"] = float(metrics.get("val_loss", 0))

    # Write results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.run_name}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULT: {args.run_name}")
    print(f"  val_miou:         {results.get('val_miou', 'N/A'):.4f}")
    print(f"  val_f1_macro:     {results.get('val_f1_macro', 'N/A'):.4f}")
    print(f"  pixel_accuracy:   {results.get('pixel_accuracy', 'N/A'):.4f}")
    print(f"  best_epoch:       {results.get('best_epoch', 'N/A')}")
    print(f"  elapsed:          {elapsed/60:.1f} min")
    print(f"  results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
