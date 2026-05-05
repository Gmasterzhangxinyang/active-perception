"""
Image Refiner - 根据Agent决策修改输入图像
"""

import torch
import numpy as np
import cv2


class ImageRefiner:
    """图像优化器，执行Agent选择的工具"""

    def __init__(self):
        self.current_state = None

    def enhance_image(self, images, camera_ids, enhancement_type="contrast", factor=1.5):
        """
        图像增强

        Args:
            images: (B, N_cams, 3, H, W) tensor
            camera_ids: list 要处理的相机ID
            enhancement_type: "contrast" | "sharpness" | "denoise" | "gamma"
            factor: 增强强度

        Returns:
            处理后的tensor
        """
        images = images.clone()

        for cam_id in camera_ids:
            if cam_id >= images.shape[1]:
                continue

            # (B, 3, H, W) -> (B, H, W, 3) -> numpy
            img_batch = images[:, cam_id]
            img_np = img_batch.permute(0, 2, 3, 1).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            enhanced_list = []
            for img in img_np:
                if enhancement_type == "contrast":
                    img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
                elif enhancement_type == "sharpness":
                    kernel = np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]]) * factor
                    img = cv2.filter2D(img, -1, kernel)
                    img = np.clip(img, 0, 255).astype(np.uint8)
                elif enhancement_type == "gamma":
                    inv_gamma = 1.0 / factor
                    table = np.array([((i / 255.0) ** inv_gamma) * 255
                                      for i in np.arange(0, 256)]).astype("uint8")
                    img = cv2.LUT(img, table)
                elif enhancement_type == "denoise":
                    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

                enhanced_list.append(img)

            # 转回tensor
            enhanced_np = np.array(enhanced_list)
            enhanced_np = enhanced_np.astype(np.float32) / 255.0
            enhanced_tensor = torch.from_numpy(enhanced_np).permute(0, 3, 1, 2)

            images[:, cam_id] = enhanced_tensor.to(images.device)

        return images

    def remove_rain(self, images, camera_ids, method="CLAHE", regions=None):
        """
        去雨

        Args:
            images: (B, N_cams, 3, H, W) tensor
            camera_ids: list
            method: "CLAHE" | "Gaussian"
            regions: list of [x1,y1,x2,y2] 如果提供，只处理这些区域

        Returns:
            处理后的tensor
        """
        images = images.clone()

        for cam_id in camera_ids:
            if cam_id >= images.shape[1]:
                continue

            img_batch = images[:, cam_id]
            img_np = img_batch.permute(0, 2, 3, 1).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            enhanced_list = []
            for img in img_np:
                if regions:
                    # 只处理指定区域
                    for region in regions:
                        x1, y1, x2, y2 = region
                        # 确保区域在图像范围内
                        x1, x2 = max(0, x1), min(img.shape[1], x2)
                        y1, y2 = max(0, y1), min(img.shape[0], y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        roi = img[y1:y2, x1:x2]
                        if method == "CLAHE":
                            lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            l = clahe.apply(l)
                            lab = cv2.merge([l, a, b])
                            roi_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                        else:
                            roi_processed = cv2.GaussianBlur(roi, (5, 5), 0)

                        img[y1:y2, x1:x2] = roi_processed
                    enhanced_list.append(img)
                else:
                    # 全图处理
                    if method == "CLAHE":
                        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l = clahe.apply(l)
                        lab = cv2.merge([l, a, b])
                        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    else:
                        img = cv2.GaussianBlur(img, (5, 5), 0)
                    enhanced_list.append(img)

            enhanced_np = np.array(enhanced_list)
            enhanced_np = enhanced_np.astype(np.float32) / 255.0
            enhanced_tensor = torch.from_numpy(enhanced_np).permute(0, 3, 1, 2)

            images[:, cam_id] = enhanced_tensor.to(images.device)

        return images

    def dehaze(self, images, camera_ids, method="CLAHE", regions=None):
        """
        去雾

        Args:
            images: (B, N_cams, 3, H, W) tensor
            camera_ids: list
            method: "CLAHE" | "HE"
            regions: list of [x1,y1,x2,y2] 如果提供，只处理这些区域

        Returns:
            处理后的tensor
        """
        images = images.clone()

        for cam_id in camera_ids:
            if cam_id >= images.shape[1]:
                continue

            img_batch = images[:, cam_id]
            img_np = img_batch.permute(0, 2, 3, 1).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            enhanced_list = []
            for img in img_np:
                if regions:
                    # 只处理指定区域
                    for region in regions:
                        x1, y1, x2, y2 = region
                        x1, x2 = max(0, x1), min(img.shape[1], x2)
                        y1, y2 = max(0, y1), min(img.shape[0], y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        roi = img[y1:y2, x1:x2]
                        if method == "CLAHE":
                            lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            l = clahe.apply(l)
                            lab = cv2.merge([l, a, b])
                            roi_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                        else:
                            yuv = cv2.cvtColor(roi, cv2.COLOR_RGB2YUV)
                            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                            roi_processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

                        img[y1:y2, x1:x2] = roi_processed
                    enhanced_list.append(img)
                else:
                    # 全图处理
                    if method == "CLAHE":
                        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l = clahe.apply(l)
                        lab = cv2.merge([l, a, b])
                        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    else:
                        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
                    enhanced_list.append(img)

            enhanced_np = np.array(enhanced_list)
            enhanced_np = enhanced_np.astype(np.float32) / 255.0
            enhanced_tensor = torch.from_numpy(enhanced_np).permute(0, 3, 1, 2)

            images[:, cam_id] = enhanced_tensor.to(images.device)

        return images

    def crop_and_zoom(self, images, camera_ids, bbox, zoom_factor):
        """
        裁剪并放大 - 简化版（需要配合内参调整）

        Args:
            images: (B, N_cams, 3, H, W) tensor
            camera_ids: list
            bbox: [x_min, y_min, x_max, y_max] 归一化坐标
            zoom_factor: float

        Returns:
            处理后的tensor
        """
        # 注意：完整的crop_and_zoom需要修改相机内参
        # 这里只是简单实现，放大效果有限
        x_min, y_min, x_max, y_max = bbox

        images = images.clone()

        for cam_id in camera_ids:
            if cam_id >= images.shape[1]:
                continue

            img_batch = images[:, cam_id]
            B, C, H, W = img_batch.shape

            # 裁剪坐标
            x1, y1 = int(x_min * W), int(y_min * H)
            x2, y2 = int(x_max * W), int(y_max * H)

            # 提取裁剪区域
            cropped = img_batch[:, :, y1:y2, x1:x2]

            # 放大到原尺寸 * zoom_factor
            new_H, new_W = int((y2 - y1) * zoom_factor), int((x2 - x1) * zoom_factor)

            # 转换用于cv2
            cropped_np = cropped.permute(0, 2, 3, 1).cpu().numpy()
            cropped_np = (cropped_np * 255).astype(np.uint8)

            resized_list = []
            for img in cropped_np:
                resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
                resized_list.append(resized)

            resized_np = np.array(resized_list)
            resized_np = resized_np.astype(np.float32) / 255.0
            resized_tensor = torch.from_numpy(resized_np).permute(0, 3, 1, 2)

            # 裁剪到原图尺寸后放回
            out_H = min(new_H, H)
            out_W = min(new_W, W)
            images[:, cam_id, :, :out_H, :out_W] = resized_tensor[:, :, :out_H, :out_W].to(images.device)

        return images
