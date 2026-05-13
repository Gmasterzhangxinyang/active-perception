#!/usr/bin/env python3
"""Train BEVFusion on nuScenes mini dataset.

Usage:
    python train.py --dataroot /path/to/nuscenes --epochs 10
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BEVConfig
from models.bevfusion import BEVFusion
from data.nuscenes_loader import NuScenesLoader
from data.bev_gt import generate_bev_gt
from utils.visualize import visualize_bev_result


def train(args):
    import shutil

    # ---- Config ----
    device_name = (
        "mps" if torch.backends.mps.is_available() and args.device == "auto" else "cpu"
    )
    if args.device in ("cpu", "mps"):
        device_name = args.device
    device = torch.device(device_name)

    cfg = BEVConfig(device=device_name)

    print("=" * 60)
    print("BEVFusion Training")
    print("=" * 60)
    print(f"Device:     {device}")
    print(f"Epochs:     {args.epochs}")
    print(f"LR:         {args.lr}")
    print(f"Dataroot:   {args.dataroot}")
    print()

    # Progress display width
    bar_width = 40

    # ---- Data ----
    print("Loading nuScenes...")
    loader = NuScenesLoader(args.dataroot, args.version, cfg)
    nusc = loader.nusc
    n_samples = len(loader)
    print(f"Total samples: {n_samples}")

    # Pre-generate all BEV ground truths
    print("Generating BEV ground truth maps...")
    gt_maps = []
    valid_indices = []
    for i in range(n_samples):
        sample = nusc.sample[i]
        bev_gt = generate_bev_gt(nusc, sample, cfg)
        n_objects = (bev_gt > 0).sum().item()
        gt_maps.append(bev_gt)
        if n_objects > 0:
            valid_indices.append(i)
    print(f"Samples with objects: {len(valid_indices)} / {n_samples}")
    print()

    # ---- Model ----
    camera_only = False
    print("Building model... (camera_only=" + str(camera_only) + ")")
    model = BEVFusion(cfg, camera_only=camera_only).to(device)

    # Freeze ResNet backbone to speed up training
    for param in model.camera_encoder.layer1.parameters():
        param.requires_grad = False
    for param in model.camera_encoder.layer2.parameters():
        param.requires_grad = False
    for param in model.camera_encoder.layer3.parameters():
        param.requires_grad = False
    for param in model.camera_encoder.layer4.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable (backbone frozen)")

    # ---- Loss & Optimizer ----
    # GT currently uses two labels (0=free, 1=occupied), while the model head may
    # expose more semantic classes. CrossEntropyLoss requires one weight per logit
    # class, so keep unused extra classes lightly weighted.
    class_weights = torch.ones(cfg.num_classes, dtype=torch.float32)
    class_weights[0] = 0.1
    class_weights[1] = 10.0
    class_weights[2:] = 0.1
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 5
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    os.makedirs(args.output_dir, exist_ok=True)

    def format_time(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    train_indices = valid_indices  # Only train on samples with objects
    total_steps_per_epoch = len(train_indices)
    total_steps = total_steps_per_epoch * args.epochs

    print(
        f"\nTotal training steps: {total_steps} ({total_steps_per_epoch} steps/epoch x {args.epochs} epochs)"
    )
    print("-" * 60)

    best_loss = float("inf")
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t_epoch = time.time()

        indices = train_indices.copy()
        np.random.shuffle(indices)

        for step, idx in enumerate(indices):
            t0 = time.time()

            data = loader[idx]
            bev_gt = gt_maps[idx].unsqueeze(0).to(device)

            images = data["images"].to(device)
            intrinsics = data["intrinsics"].to(device)
            extrinsics = data["extrinsics"].to(device)
            lidar_pts = data["lidar_points"].to(device)
            lidar_mask = data["lidar_mask"].to(device)

            logits, bev_seg = model(
                images, intrinsics, extrinsics, lidar_pts, lidar_mask
            )

            loss = criterion(logits, bev_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct = (bev_seg == bev_gt).sum().item()
            total_px = bev_gt.numel()
            epoch_correct += correct
            epoch_total += total_px

            cur_loss = epoch_loss / (step + 1)
            cur_acc = epoch_correct / epoch_total
            elapsed = time.time() - t_start
            global_step = epoch * total_steps_per_epoch + step + 1
            eta = (elapsed / global_step) * (total_steps - global_step)

            filled = int(bar_width * global_step / total_steps)
            bar = "█" * filled + "░" * (bar_width - filled)
            pct = 100.0 * global_step / total_steps
            print(
                f"\r[{bar}] {pct:.1f}% | E{epoch + 1}/{args.epochs} S{step + 1}/{len(indices)} | Loss:{cur_loss:.4f} Acc:{cur_acc * 100:.1f}% | ETA:{format_time(eta)}",
                end="",
                flush=True,
            )
            with open(args.output_dir + "/progress.txt", "w") as f:
                f.write(
                    f"{pct:.1f},{epoch + 1},{args.epochs},{step + 1},{len(indices)},{cur_loss:.4f},{cur_acc * 100:.1f},{eta:.0f}\n"
                )

        epoch_time = time.time() - t_epoch
        avg_loss = epoch_loss / len(indices)
        avg_acc = 100.0 * epoch_correct / epoch_total

        total_elapsed = time.time() - t_start
        print(
            f"\n  ✓ Epoch {epoch + 1}: Loss={avg_loss:.4f} Acc={avg_acc:.1f}% Time={format_time(epoch_time)} Total={format_time(total_elapsed)}"
        )

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> Saved best checkpoint: {ckpt_path}")

        # Visualize one sample at end of each epoch
        model.eval()
        with torch.no_grad():
            vis_idx = valid_indices[0] if valid_indices else 0
            vis_data = loader[vis_idx]
            vis_gt = gt_maps[vis_idx]
            for k, v in vis_data.items():
                vis_data[k] = v.to(device)
            vis_logits, vis_seg = model(
                vis_data["images"],
                vis_data["intrinsics"],
                vis_data["extrinsics"],
                vis_data["lidar_points"],
                vis_data["lidar_mask"],
            )

            seg_np = vis_seg[0].cpu().numpy()
            gt_np = vis_gt.numpy()
            logits_np = vis_logits[0].cpu().numpy()

            # Save prediction vs GT comparison
            save_comparison(seg_np, gt_np, logits_np, epoch + 1, args.output_dir)

        scheduler.step()
        print()

    # Final checkpoint
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model: {final_path}")
    print(f"Best loss: {best_loss:.4f}")


def save_comparison(pred, gt, logits, epoch, output_dir):
    """Save side-by-side prediction vs ground truth."""
    import matplotlib.pyplot as plt
    from utils.visualize import colorize_bev, CLASS_COLORS, CLASS_NAMES
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ground truth
    axes[0].imshow(colorize_bev(gt), origin="lower")
    axes[0].set_title("Ground Truth")

    # Prediction
    axes[1].imshow(colorize_bev(pred), origin="lower")
    axes[1].set_title(f"Prediction (Epoch {epoch})")

    # Confidence
    conf = logits.max(axis=0)
    im = axes[2].imshow(conf, origin="lower", cmap="hot")
    axes[2].set_title("Confidence")
    plt.colorbar(im, ax=axes[2])

    patches = [
        mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k])
        for k in sorted(CLASS_COLORS.keys())
    ]
    axes[0].legend(handles=patches, loc="upper right", fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, f"epoch_{epoch:02d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved visualization: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "mps"]
    )
    parser.add_argument("--output_dir", type=str, default="train_output")
    parser.add_argument(
        "--objects_only",
        action="store_true",
        help="Only train on samples that contain objects",
    )
    args = parser.parse_args()
    train(args)
