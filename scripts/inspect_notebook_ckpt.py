"""Inspect the notebook checkpoint to understand training history."""
import torch
import sys

ckpt_path = "/home/mis60/CSE_MSE_RXF131/cradle-members/mds3/mis60/mds3-advman-2/25-mds3-data-segmentation/models/sem/unet-83/checkpoints/latest_checkpoint.pth"
best_path = "/home/mis60/CSE_MSE_RXF131/cradle-members/mds3/mis60/mds3-advman-2/25-mds3-data-segmentation/models/sem/unet-83/checkpoints/best_model.pth"

print("=== LATEST CHECKPOINT ===")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"Keys: {list(ckpt.keys())}")
print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Best val_loss: {ckpt.get('best_val_loss', 'N/A')}")
print(f"Num train_losses: {len(ckpt.get('train_losses', []))}")
print(f"Num val_losses: {len(ckpt.get('val_losses', []))}")

train_losses = ckpt.get("train_losses", [])
val_losses = ckpt.get("val_losses", [])

if train_losses:
    print(f"\nTrain loss range: {min(train_losses):.4f} - {max(train_losses):.4f}")
    print(f"  First 5: {[round(x,4) for x in train_losses[:5]]}")
    print(f"  Last 5: {[round(x,4) for x in train_losses[-5:]]}")

if val_losses:
    print(f"\nVal loss range: {min(val_losses):.4f} - {max(val_losses):.4f}")
    print(f"  First 5: {[round(x,4) for x in val_losses[:5]]}")
    print(f"  Last 5: {[round(x,4) for x in val_losses[-5:]]}")
    best_epoch = val_losses.index(min(val_losses))
    print(f"  Best val_loss epoch: {best_epoch + 1} (val_loss={min(val_losses):.4f})")
    print(f"  Train loss at best epoch: {train_losses[best_epoch]:.4f}")

# Check optimizer state for clues
opt_state = ckpt.get("optimizer_state_dict", {})
param_groups = opt_state.get("param_groups", [])
if param_groups:
    print(f"\nOptimizer: lr={param_groups[0].get('lr')}, weight_decay={param_groups[0].get('weight_decay')}")

# Compare model weights
print("\n=== BEST MODEL ===")
best_state = torch.load(best_path, map_location="cpu", weights_only=False)
print(f"Keys: {list(best_state.keys())[:10]}...")
print(f"Num parameters: {len(best_state)}")

# Check if best model weights differ from latest
print("\n=== COMPARISON ===")
latest_state = ckpt["model_state_dict"]
same = all(
    torch.equal(latest_state[k], best_state[k])
    for k in best_state.keys()
)
print(f"Best model == latest model weights? {same}")
