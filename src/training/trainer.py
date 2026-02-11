"""
Training Loop Orchestration

This module provides the Trainer class for training and validating the U-Net model.

Features:
- Training and validation loops
- Early stopping with configurable patience
- Checkpointing (best and latest models)
- Progress tracking with tqdm
- Metric computation and logging
- TensorBoard integration (optional)
"""

import os
from pathlib import Path
from typing import Dict, Optional, Callable
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.

    Args:
        patience (int): Number of epochs to wait after last improvement. Default: 50
        min_delta (float): Minimum change to qualify as improvement. Default: 0.0001
        mode (str): "min" for loss, "max" for accuracy/F1. Default: "min"
        verbose (bool): Print messages when stopping. Default: True
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.0001,
        mode: str = "min",
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = float('inf')
        elif mode == "max":
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'.")

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score (float): Current validation metric value

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose and self.counter >= self.patience:
                print(f"\nEarly stopping triggered: no improvement for {self.patience} epochs")
            self.early_stop = self.counter >= self.patience
            return self.early_stop


class Trainer:
    """
    Trainer class for U-Net semantic segmentation.

    Args:
        model (nn.Module): U-Net model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (nn.Module): Loss function
        device (torch.device): Device to run training on (cuda/mps/cpu)
        config (Dict): Configuration dictionary with training parameters
        metrics_fn (Optional[Callable]): Function to compute metrics. Default: None
        checkpoint_dir (str): Directory to save checkpoints. Default: "checkpoints"
        log_dir (Optional[str]): Directory for TensorBoard logs. Default: None
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        config: Dict,
        metrics_fn: Optional[Callable] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: Optional[str] = None,
        scheduler=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.metrics_fn = metrics_fn
        self.scheduler = scheduler

        # Mixed precision (AMP)
        self.use_amp = config.get("device", {}).get("mixed_precision", False) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            print("Mixed precision training (AMP) enabled")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        early_stop_config = config.get("training", {}).get("early_stopping", {})
        if early_stop_config.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=early_stop_config.get("patience", 50),
                min_delta=early_stop_config.get("min_delta", 0.0001),
                mode=early_stop_config.get("mode", "min"),
                verbose=config.get("logging", {}).get("print_metrics", True)
            )
        else:
            self.early_stopping = None

        self.writer = None
        if log_dir and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.log_every_n_batches = config.get("logging", {}).get("log_every_n_batches", 10)
        self.print_metrics = config.get("logging", {}).get("print_metrics", True)

    def train_epoch(self) -> float:
        """Run one training epoch and return average loss."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]", leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device).long()

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            if self.writer and (batch_idx % self.log_every_n_batches == 0):
                global_step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar("Train/BatchLoss", batch_loss, global_step)

        return epoch_loss / num_batches

    def validate_epoch(self) -> Dict[str, float]:
        """Run one validation epoch and return metrics."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(self.val_loader)

        all_predictions = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]", leave=False)

        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                batch_loss = loss.item()
                epoch_loss += batch_loss

                if self.metrics_fn:
                    all_predictions.append(outputs.cpu())
                    all_targets.append(labels.cpu())

                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        avg_loss = epoch_loss / num_batches
        results = {"val_loss": avg_loss}

        if self.metrics_fn and len(all_predictions) > 0:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            metrics = self.metrics_fn(all_predictions, all_targets)
            results.update(metrics)

        return results

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)

        if self.print_metrics:
            print(f"{'[BEST] ' if is_best else ''}Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint to resume training."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.print_metrics:
            print(f"Checkpoint loaded: {filepath} (epoch {self.current_epoch})")

    def fit(self, num_epochs: int, resume_from: Optional[str] = None):
        """Train the model for specified epochs."""
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1

        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print("-" * 70)

        start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_results = self.validate_epoch()
            val_loss = val_results["val_loss"]
            self.val_losses.append(val_loss)

            # Report parameter count after first epoch (LazyConv layers now initialized)
            if epoch == start_epoch:
                try:
                    num_params = self.model.count_parameters()
                    if num_params > 0:
                        print(f"\nModel parameters: {num_params:,} ({num_params/1e6:.2f}M)")
                except:
                    pass  # Silently skip if still fails

            # Step LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.print_metrics:
                print(f"\nEpoch {epoch}/{num_epochs-1}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  LR:         {current_lr:.2e}")

                for key, value in val_results.items():
                    if key != "val_loss" and "macro" in key:
                        print(f"  {key}: {value:.4f}")

            if self.writer:
                self.writer.add_scalar("Train/EpochLoss", train_loss, epoch)
                self.writer.add_scalar("Val/EpochLoss", val_loss, epoch)
                self.writer.add_scalar("Train/LR", current_lr, epoch)
                for key, value in val_results.items():
                    if key != "val_loss":
                        self.writer.add_scalar(f"Val/{key}", value, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.checkpoint_dir / "best_model.pth"
                self.save_checkpoint(str(best_path), is_best=True)

            latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
            self.save_checkpoint(str(latest_path), is_best=False)

            if self.early_stopping:
                if self.early_stopping(val_loss):
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Training complete!")
        print(f"Total time: {elapsed_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model saved at: {self.checkpoint_dir / 'best_model.pth'}")
        print("=" * 70)

        if self.writer:
            self.writer.close()
