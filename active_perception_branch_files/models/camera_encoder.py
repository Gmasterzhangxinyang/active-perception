"""Camera branch: ResNet-50 backbone + FPN neck + LSS view transform."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.geometry import create_frustum, frustum_to_world, points_to_bev_indices


class FPN(nn.Module):
    """Simplified Feature Pyramid Network — fuse C3, C4, C5 into a single scale."""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features):
        # features: list of [C3, C4, C5] from ResNet
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False
            )

        out = self.output_conv(laterals[0])  # use finest scale
        return out


class DepthNet(nn.Module):
    """Predict discrete depth distribution for each pixel."""

    def __init__(self, in_channels, depth_bins):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, depth_bins, 1),
        )

    def forward(self, x):
        # x: (B*N, C, fH, fW)  -> (B*N, D, fH, fW)
        return self.net(x)


class LSSViewTransform(nn.Module):
    """Lift-Splat-Shoot: lift 2D features to 3D using predicted depth, then splat to BEV.

    This is the core of camera-to-BEV projection, replacing CUDA bev_pool with simple
    scatter-add operations that work on CPU/MPS.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.depth_net = DepthNet(64, cfg.depth_bins)

        # Pre-compute frustum grid (shared across all images)
        self.frustum = create_frustum(
            cfg.depth_bins, cfg.depth_min, cfg.depth_max,
            cfg.image_size[0], cfg.image_size[1], downsample=16
        )  # (D, fH, fW, 3)

    def forward(self, features, intrinsics, extrinsics):
        """
        Args:
            features: (B, N, C, fH, fW) image features from backbone+FPN
            intrinsics: (B, N, 3, 3) camera intrinsics
            extrinsics: (B, N, 4, 4) cam2ego transforms

        Returns:
            bev_features: (B, C, bev_H, bev_W)
        """
        B, N, C, fH, fW = features.shape
        cfg = self.cfg
        D = cfg.depth_bins
        bev_H, bev_W = cfg.bev_size
        device = features.device

        # Predict depth distribution
        feat_flat = features.reshape(B * N, C, fH, fW)
        depth_logits = self.depth_net(feat_flat)            # (B*N, D, fH, fW)
        depth_probs = depth_logits.softmax(dim=1)           # (B*N, D, fH, fW)

        # Outer product: feature * depth -> 3D volume
        # feat_flat: (B*N, C, fH, fW) -> (B*N, C, 1, fH, fW)
        # depth_probs: (B*N, D, fH, fW) -> (B*N, 1, D, fH, fW)
        volume = feat_flat.unsqueeze(2) * depth_probs.unsqueeze(1)  # (B*N, C, D, fH, fW)
        volume = volume.reshape(B, N, C, D, fH, fW)

        # Project frustum points to ego coordinates for each camera
        frustum = self.frustum.to(device)  # (D, fH, fW, 3)

        bev_out = torch.zeros(B, C, bev_H, bev_W, device=device)

        for b in range(B):
            for n in range(N):
                # Project this camera's frustum to ego frame
                pts_ego = frustum_to_world(
                    frustum, intrinsics[b, n], extrinsics[b, n]
                )  # (D, fH, fW, 3)

                # Get BEV indices
                bev_ix, bev_iy, valid = points_to_bev_indices(
                    pts_ego, cfg.bev_x_range, cfg.bev_y_range, cfg.bev_size
                )  # each: (D, fH, fW)

                # Scatter-add valid features to BEV grid
                if valid.any():
                    valid_flat = valid.reshape(-1)
                    ix_flat = bev_ix.reshape(-1)[valid_flat]
                    iy_flat = bev_iy.reshape(-1)[valid_flat]

                    # volume for this camera: (C, D, fH, fW)
                    vol = volume[b, n]  # (C, D, fH, fW)
                    vol_flat = vol.reshape(C, -1)[:, valid_flat]  # (C, n_valid)

                    # Linear index into BEV
                    bev_idx = iy_flat * bev_W + ix_flat  # (n_valid,)
                    bev_flat = bev_out[b].reshape(C, -1)  # (C, bev_H*bev_W)
                    bev_flat = bev_flat.scatter_add(1, bev_idx.unsqueeze(0).expand(C, -1), vol_flat)
                    bev_out = bev_out.clone()
                    bev_out[b] = bev_flat.reshape(C, bev_H, bev_W)

        return bev_out


class CameraEncoder(nn.Module):
    """Full camera branch: ResNet-50 -> FPN -> LSS -> Camera BEV features."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ResNet-50 backbone. Keep pretrained weights opt-in so local runs do not
        # unexpectedly download from torchvision.
        weights = models.ResNet50_Weights.DEFAULT if getattr(cfg, "pretrained_camera_backbone", False) else None
        resnet = models.resnet50(weights=weights)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # FPN: fuse layer2(512), layer3(1024), layer4(2048) -> 64 channels
        self.fpn = FPN([512, 1024, 2048], cfg.cam_channels)

        # LSS view transform
        self.view_transform = LSSViewTransform(cfg)

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            images: (B, N, 3, H, W)
            intrinsics: (B, N, 3, 3)
            extrinsics: (B, N, 4, 4)

        Returns:
            cam_bev: (B, C, bev_H, bev_W)
        """
        B, N = images.shape[:2]

        # Process all camera images together
        imgs = images.reshape(B * N, 3, *self.cfg.image_size)

        c1 = self.layer1(imgs)
        c2 = self.layer2(c1)   # stride 8,  512 ch
        c3 = self.layer3(c2)   # stride 16, 1024 ch
        c4 = self.layer4(c3)   # stride 32, 2048 ch

        feat = self.fpn([c2, c3, c4])  # (B*N, 64, fH, fW) at stride 8

        # Downsample to stride 16 to match frustum
        feat = F.avg_pool2d(feat, 2)

        fH, fW = feat.shape[2:]
        feat = feat.reshape(B, N, self.cfg.cam_channels, fH, fW)

        # LSS: lift to BEV
        cam_bev = self.view_transform(feat, intrinsics, extrinsics)
        return cam_bev
