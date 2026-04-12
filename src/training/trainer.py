"""Training loop for semantic segmentation models."""

import os
import time

import torch
import yaml
try:
    from torch.amp import GradScaler, autocast
    _autocast_kwargs = lambda dev: {"device_type": dev}
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _autocast_kwargs = lambda dev: {}
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.utils.plotting import plot_training_curves, plot_sample_predictions


class Trainer:
    """Handles the training loop, validation, checkpointing, and early stopping."""

    def __init__(self, model, loss_fn, optimizer, scheduler, cfg):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg

        # Device
        self.device = self._get_device(cfg.get("device", "auto"))
        self.model.to(self.device)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn.to(self.device)

        # Mixed precision
        self.use_amp = cfg.get("mixed_precision", False) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = cfg.get("checkpoint_dir", "logs/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Early stopping
        self.patience = cfg.get("patience", 50)
        self.min_delta = cfg.get("min_delta", 1e-4)

        # Monitoring
        self.monitor = cfg.get("monitor", "val_loss")
        self.monitor_mode = cfg.get("monitor_mode", "min")

        # Logging
        log_dir = cfg.get("log_dir", "logs/runs")
        self.writer = SummaryWriter(log_dir=log_dir)

        self.num_classes = cfg.get("num_classes", 6)

        # Metric history for plotting
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "miou": [],
            "f1_macro": [],
            "pixel_accuracy": [],
            "lr": [],
        }

    def fit(self, train_loader, val_loader, max_epochs):
        """Run the full training loop."""
        best_metric = float("inf") if self.monitor_mode == "min" else float("-inf")
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate_epoch(val_loader, epoch)
            val_loss = val_metrics["val_loss"]

            # Track history for plotting
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["miou"].append(val_metrics.get("miou", 0.0))
            self.history["f1_macro"].append(val_metrics.get("f1_macro", 0.0))
            self.history["pixel_accuracy"].append(val_metrics.get("pixel_accuracy", 0.0))
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Logging
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            for k, v in val_metrics.items():
                if k != "val_loss" and not isinstance(v, list):
                    self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            # LR scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.monitor, val_loss))
                else:
                    self.scheduler.step()

            # Current metric
            current = val_metrics.get(self.monitor, val_loss)
            improved = (
                (current < best_metric - self.min_delta) if self.monitor_mode == "min"
                else (current > best_metric + self.min_delta)
            )

            if improved:
                best_metric = current
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1

            self._save_checkpoint(epoch, val_metrics, is_best=False)

            # Print summary
            metric_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in val_metrics.items() if not isinstance(v, list)
            )
            print(f"Epoch {epoch}/{max_epochs} | train_loss: {train_loss:.4f} | {metric_str}")

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break

        self.writer.close()

        # Auto-generate plots
        plot_dir = os.path.join(os.path.dirname(self.checkpoint_dir), "plots")
        plot_training_curves(self.history, plot_dir)
        plot_sample_predictions(
            self.model, train_loader, self.device, plot_dir,
            split_name="train", num_classes=self.num_classes,
        )
        plot_sample_predictions(
            self.model, val_loader, self.device, plot_dir,
            split_name="val", num_classes=self.num_classes,
        )

        return best_metric

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(**_autocast_kwargs(self.device.type)):
                    logits = self._forward(images)
                    loss = self.loss_fn(logits, masks.long())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self._forward(images)
                loss = self.loss_fn(logits, masks.long())
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def _validate_epoch(self, loader, epoch):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self._forward(images)
            loss = self.loss_fn(logits, masks.long())
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets, self.num_classes)
        metrics["val_loss"] = total_loss / len(loader)
        return metrics

    def _save_checkpoint(self, epoch, metrics, is_best):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.cfg,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        filename = "best.pt" if is_best else "latest.pt"
        torch.save(state, os.path.join(self.checkpoint_dir, filename))

    def _forward(self, images):
        """Forward pass with support for torchvision OrderedDict outputs."""
        output = self.model(images)
        if isinstance(output, dict):
            output = output["out"]
        return output

    @staticmethod
    def _get_device(device_str):
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)
