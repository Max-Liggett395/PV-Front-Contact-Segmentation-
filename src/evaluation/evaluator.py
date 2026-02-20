"""Run model on a test set and save results."""

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_metrics
from src.data.dataset import CLASS_NAMES


class Evaluator:
    """Run inference on a test set and save results."""

    def __init__(self, model, device, num_classes=6):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def run(self, dataloader):
        """Run on dataloader and return metrics dict."""
        all_preds = []
        all_targets = []

        for images, masks in tqdm(dataloader, desc="Running"):
            images = images.to(self.device)
            logits = self.model(images)
            if isinstance(logits, dict):
                logits = logits["out"]
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(masks)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        metrics = compute_metrics(all_preds, all_targets, self.num_classes)

        # Format per-class results
        per_class = {}
        for i, name in enumerate(CLASS_NAMES[:self.num_classes]):
            per_class[name] = {"iou": metrics["per_class_iou"][i]}
        metrics["per_class"] = per_class

        return metrics

    def save_results(self, metrics, output_path):
        """Save metrics to JSON file."""
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, float)):
                serializable[k] = round(float(v), 4)
            elif isinstance(v, list):
                serializable[k] = [round(float(x), 4) for x in v]
            elif isinstance(v, dict):
                serializable[k] = {
                    sk: {sk2: round(float(sv2), 4) for sk2, sv2 in sv.items()}
                    if isinstance(sv, dict) else sv
                    for sk, sv in v.items()
                }
            else:
                serializable[k] = v

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {output_path}")
