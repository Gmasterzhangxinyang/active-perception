"""
BEV Quality Evaluator - 评估BEV分割质量
"""

import torch
import numpy as np
import cv2


class BEVEvaluator:
    """评估BEV分割质量"""

    def __init__(self, device="cpu"):
        self.device = device

    def evaluate(self, bev_seg):
        """
        评估BEV分割质量（无GT版本）

        Args:
            bev_seg: BEV分割结果 (H, W) 值是类别ID

        Returns:
            dict: 包含edge_density, class_balance等指标
        """
        if bev_seg.dim() == 3:
            bev_seg = bev_seg.squeeze(0)

        bev_seg = bev_seg.float()

        # 1. 边缘清晰度 - 用Canny检测边缘
        edge_density = self._compute_edge_density(bev_seg)

        # 2. 类别分布 - 是否合理
        class_counts = self._count_classes(bev_seg)

        # 3. 物体完整性 - 分割区域的大小分布
        integrity = self._compute_integrity(bev_seg)

        # 4. 质量分数 (综合)
        score = 0.4 * edge_density + 0.3 * integrity + 0.3 * min(1.0, class_counts.get(1, 0) / 100)

        # 5. 问题区域中心坐标（用于相机映射）
        problem_coords = self._find_problem_centers(bev_seg)

        return {
            "edge_density": edge_density,
            "class_counts": class_counts,
            "integrity": integrity,
            "score": score,
            "needs_optimization": edge_density < 0.05 or integrity < 0.5 or len(problem_coords) > 3,
            "problem_mask": self._find_problem_regions(bev_seg),
            "problem_coords": problem_coords
        }

    def evaluate_with_gt(self, bev_pred, bev_gt):
        """
        用GT评估BEV质量

        Args:
            bev_pred: 预测的BEV分割 (H, W)
            bev_gt: GT的BEV分割 (H, W)

        Returns:
            dict: 包含IoU、Accuracy等指标
        """
        if bev_pred.dim() == 3:
            bev_pred = bev_pred.squeeze(0)
        if bev_gt.dim() == 3:
            bev_gt = bev_gt.squeeze(0)

        bev_pred = bev_pred.long()
        bev_gt = bev_gt.long()

        # 对于多类模型，转为二值: class 1=占用, 其他=空闲
        # 因为GT只有0和1，模型可能有更多类
        bev_pred_binary = (bev_pred == 1).long()
        bev_gt_binary = (bev_gt == 1).long()

        # 总体IoU (二值)
        intersection = (bev_pred_binary & bev_gt_binary).sum()
        union = (bev_pred_binary | bev_gt_binary).sum()
        iou = intersection.float() / (union.float() + 1e-6)

        # 各类别IoU (原始多类)
        iou_per_class = {}
        unique_classes = torch.unique(torch.cat([bev_pred.flatten(), bev_gt.flatten()])).tolist()
        for cls in unique_classes:
            pred_cls = (bev_pred == cls)
            gt_cls = (bev_gt == cls)
            inter = (pred_cls & gt_cls).sum()
            un = (pred_cls | gt_cls).sum()
            iou_per_class[cls] = inter.float() / (un.float() + 1e-6)

        # Accuracy (二值)
        accuracy = (bev_pred_binary == bev_gt_binary).float().mean()

        return {
            "iou": iou.item(),
            "iou_per_class": {k: v.item() for k, v in iou_per_class.items()},
            "accuracy": accuracy.item()
        }

    def bev_to_camera_mapping(self, problem_coords, extrinsics, intrinsics, bev_cfg):
        """
        BEV问题区域 → 对应相机ID

        映射逻辑:
        1. 问题区域中心 (bev_x, bev_y) 在ego坐标系下，z=0
        2. 逆投影: ego → cam 使用ego2cam = inverse(cam2ego)
        3. 检查点是否在相机前方 (z > 0)
        4. 投影到图像平面: u = fx*x/z + cx, v = fy*y/z + cy
        5. 检查是否在图像范围内 [0, W] x [0, H]

        Args:
            problem_coords: 问题区域坐标列表 [{bbox, center, area}, ...]
            extrinsics: 相机外参 (B, N_cams, 4, 4) cam2ego
            intrinsics: 相机内参 (B, N_cams, 3, 3)
            bev_cfg: BEV配置 dict，包含x_range, y_range, bev_size, image_size

        Returns:
            list: 每个问题区域对应的相机ID列表
                  [{bev_center: [x,y], camera_ids: [0,1], image_coords: (u,v)}, ...]
        """
        # BEV配置
        bev_x_range = bev_cfg.get("bev_x_range", (-30, 30))
        bev_y_range = bev_cfg.get("bev_y_range", (-30, 30))
        bev_size = bev_cfg.get("bev_size", (120, 120))
        image_size = bev_cfg.get("image_size", (128, 352))  # (H, W)

        x_min, x_max = bev_x_range
        y_min, y_max = bev_y_range
        bev_h, bev_w = bev_size
        img_h, img_w = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        # 计算分辨率
        resolution_x = (x_max - x_min) / bev_w
        resolution_y = (y_max - y_min) / bev_h

        # 移到CPU做计算，先转numpy再处理维度
        extrinsics_np = extrinsics.detach().cpu().numpy() if hasattr(extrinsics, 'cpu') else np.array(extrinsics)
        intrinsics_np = intrinsics.detach().cpu().numpy() if hasattr(intrinsics, 'cpu') else np.array(intrinsics)

        # 处理batch维度，取第一个样本
        while extrinsics_np.ndim > 3:
            extrinsics_np = extrinsics_np[0]
        while intrinsics_np.ndim > 3:
            intrinsics_np = intrinsics_np[0]

        n_cams = extrinsics_np.shape[0]
        results = []

        for region in problem_coords:
            center = region.get("center", [60, 60])  # 默认中心
            bev_pixel_x, bev_pixel_y = center

            # BEV像素 → ego米坐标
            # 原点(0,0)在BEV中心
            bev_x = x_min + bev_pixel_x * resolution_x
            bev_y = y_max - bev_pixel_y * resolution_y  # y轴翻转

            valid_cameras = []
            image_coords = []

            for cam_id in range(n_cams):
                # 外参 cam2ego: 从相机到ego (numpy)
                cam2ego = extrinsics_np[cam_id]  # (4, 4)
                ego2cam = np.linalg.inv(cam2ego)  # ego到相机

                # ego坐标 → 相机坐标
                point_ego = np.array([bev_x, bev_y, 0.0, 1.0])
                point_cam = ego2cam @ point_ego

                # 确保是1D数组
                point_cam = np.asarray(point_cam).flatten()

                # 检查是否在相机前方 (z > 0)
                cam_z = float(point_cam[2])
                if cam_z > 0.1:  # 需要在前方一段距离
                    # 内参
                    fx = float(intrinsics_np[cam_id, 0, 0])
                    fy = float(intrinsics_np[cam_id, 1, 1])
                    cx = float(intrinsics_np[cam_id, 0, 2])
                    cy = float(intrinsics_np[cam_id, 1, 2])

                    # 投影到图像平面
                    cam_x = float(point_cam[0])
                    cam_y = float(point_cam[1])
                    u = fx * cam_x / cam_z + cx
                    v = fy * cam_y / cam_z + cy

                    # 检查是否在图像范围内
                    in_w = bool(0 <= u <= img_w)
                    in_h = bool(0 <= v <= img_h)
                    if in_w and in_h:
                        valid_cameras.append(cam_id)
                        image_coords.append((float(u), float(v)))

            # 如果没有相机能看到，说明在盲区，返回所有相机
            if not valid_cameras:
                valid_cameras = list(range(n_cams))
                image_coords = [(img_w // 2, img_h // 2)] * n_cams

            results.append({
                "bev_center": [bev_pixel_x, bev_pixel_y],
                "bev_meters": [bev_x, bev_y],
                "camera_ids": valid_cameras,
                "image_coords": image_coords
            })

        return results

    def _compute_edge_density(self, bev):
        """计算边缘密度"""
        bev_np = bev.cpu().numpy().astype(np.uint8)
        edges = cv2.Canny(bev_np * 255, 50, 150)
        edge_density = (edges > 0).sum() / (edges.size + 1e-6)
        return edge_density

    def _count_classes(self, bev):
        """统计各类别数量"""
        bev_np = bev.cpu().numpy()
        unique, counts = np.unique(bev_np, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def _compute_integrity(self, bev, min_area=20):
        """计算物体完整性 - 有多少完整物体 vs 碎片"""
        bev_np = (bev.cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(bev_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        valid_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                valid_count += 1

        # 完整性指标：完整物体数量越多越好
        integrity = min(1.0, valid_count / 10)  # 假设10个物体算完整
        return integrity

    def _find_problem_regions(self, bev, min_area=20):
        """找出问题区域（过小/过碎的物体）"""
        bev_np = (bev.cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(bev_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        problem_mask = np.zeros_like(bev_np)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                problem_mask[y:y+h, x:x+w] = 255

        return problem_mask

    def _find_problem_centers(self, bev, max_regions=5):
        """找出问题区域的中心点"""
        bev_np = (bev.cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(bev_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        problem_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:  # 太小
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2
                problem_regions.append({
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "center": [int(center_x), int(center_y)],
                    "area": float(area)
                })

        # 按面积排序，取最大的几个
        problem_regions.sort(key=lambda r: r["area"], reverse=True)
        return problem_regions[:max_regions]
