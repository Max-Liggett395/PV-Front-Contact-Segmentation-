"""Training entrypoint."""

import argparse
import os
from datetime import datetime

import torch

from src.data import SEMDataModule
from src.models import create_model
from src.models.losses import create_loss
from src.training import Trainer
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train SEM segmentation model")
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run")
    args = parser.parse_args()

    # Load experiment config (references data + model configs)
    exp_cfg = load_config(args.experiment)
    data_cfg = load_config(os.path.join("configs", exp_cfg["data"] + ".yaml"))
    model_cfg = load_config(os.path.join("configs", exp_cfg["model"] + ".yaml"))

    # Create run directory
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("logs", "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    training_cfg = exp_cfg.get("training", {})
    training_cfg["checkpoint_dir"] = os.path.join(run_dir, "checkpoints")
    training_cfg["log_dir"] = os.path.join(run_dir, "tensorboard")
    training_cfg["num_classes"] = model_cfg.get("num_classes", 6)

    # Pass in_channels from model config to data config
    data_cfg["in_channels"] = model_cfg.get("in_channels", 1)

    # Allow experiment config to override data settings (e.g. batch_size)
    data_overrides = exp_cfg.get("data_overrides", {})
    data_cfg.update(data_overrides)

    # Data
    dm = SEMDataModule(data_cfg)
    dm.setup()

    # Model
    model = create_model(model_cfg)
    print(f"Model: {model_cfg['architecture']}")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}")

    # Loss — use factory for flexibility (supports ce, dice, dice_ce)
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

    max_epochs = training_cfg.get("max_epochs", 1000)

    # Scheduler (optional)
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
        elif sched_type in ("cosine", "cosine_annealing"):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg.get("T_max", max_epochs),
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
                epochs=max_epochs,
                steps_per_epoch=len(dm.train_dataloader()),
            )

    # Train
    trainer = Trainer(model, loss_fn, optimizer, scheduler, training_cfg)
    best = trainer.fit(
        dm.train_dataloader(),
        dm.val_dataloader(),
        max_epochs=max_epochs,
    )
    print(f"Training complete. Best {training_cfg.get('monitor', 'val_loss')}: {best:.4f}")


if __name__ == "__main__":
    main()
