"""
Image Refiner - 根据Agent决策修改输入图像
"""

import torch
import numpy as np
import cv2


class ImageRefiner:
    """图像优化器，执行Agent选择的工具。"""

    def __init__(self):
        self.current_state = None

    def enhance_image(self, images, camera_ids, enhancement_type="contrast", factor=1.15):
        """
        通用图像增强。

        enhancement_type:
            contrast: 温和对比度增强
            sharpness: unsharp mask 提高清晰度
            denoise: 彩色降噪
            gamma: Gamma亮度校正，factor > 1 变亮，factor < 1 变暗
        """
        def process(img):
            if enhancement_type == "contrast":
                return self._enhance_contrast(img, factor=factor)
            if enhancement_type == "sharpness":
                return self._unsharp_mask(img, amount=factor, radius=1.0)
            if enhancement_type == "gamma":
                return self._apply_gamma_factor(img, factor=factor)
            if enhancement_type == "denoise":
                return cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
            return img

        return self._apply_to_cameras(images, camera_ids, process)

    def enhance_low_light(self, images, camera_ids, strength=0.65, gamma=1.25, clip_limit=2.0, regions=None):
        """
        夜间/弱光增强：只重点提升暗部，尽量保留高光，避免把车灯和路灯进一步冲爆。
        """
        def process(img):
            return self._enhance_low_light_image(
                img,
                strength=float(strength),
                gamma=float(gamma),
                clip_limit=float(clip_limit),
            )

        return self._apply_to_cameras(images, camera_ids, process, regions=regions)

    def reduce_glare(self, images, camera_ids, threshold=210, strength=0.55, regions=None):
        """
        压制车灯/路灯眩光：检测高亮区域并压缩亮度，保留原图结构。
        """
        def process(img):
            return self._reduce_glare_image(
                img,
                threshold=int(threshold),
                strength=float(strength),
            )

        return self._apply_to_cameras(images, camera_ids, process, regions=regions)

    def sharpen_image(self, images, camera_ids, strength=0.65, radius=1.0, regions=None):
        """温和锐化，适合轻微模糊或远处目标边缘不清。"""
        def process(img):
            return self._unsharp_mask(img, amount=float(strength), radius=float(radius))

        return self._apply_to_cameras(images, camera_ids, process, regions=regions)

    def deblur_image(self, images, camera_ids, strength=0.75, regions=None):
        """
        轻量去模糊：双边滤波保边，再做高频增强。不是盲去卷积，避免生成明显伪影。
        """
        def process(img):
            base = cv2.bilateralFilter(img, 5, 30, 30)
            sharp = self._unsharp_mask(base, amount=float(strength), radius=1.2)
            return cv2.addWeighted(sharp, 0.85, img, 0.15, 0)

        return self._apply_to_cameras(images, camera_ids, process, regions=regions)

    def remove_rain(self, images, camera_ids, method="CLAHE", regions=None):
        """
        去雨/水渍的轻量版本。这里不是深度去雨模型，主要提供几种保守候选给 ablation 选择。
        """
        def process(img):
            if method == "Median":
                filtered = cv2.medianBlur(img, 3)
                return cv2.addWeighted(img, 0.65, filtered, 0.35, 0)
            if method == "Bilateral":
                filtered = cv2.bilateralFilter(img, 5, 35, 35)
                return cv2.addWeighted(img, 0.55, filtered, 0.45, 0)
            if method == "Gaussian":
                filtered = cv2.GaussianBlur(img, (3, 3), 0)
                return cv2.addWeighted(img, 0.7, filtered, 0.3, 0)
            return self._clahe_luminance(img, clip_limit=1.8)

        return self._apply_to_cameras(images, camera_ids, process, regions=regions)

    def dehaze(self, images, camera_ids, method="CLAHE", regions=None):
        """去雾/轻霾，提供 CLAHE/HE/DCP 三类候选。"""
        def process(img):
            if method == "DCP":
                return self._dark_channel_dehaze(img)
            if method == "HE":
                yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            return self._clahe_luminance(img, clip_limit=2.0)

        return self._apply_to_cameras(images, camera_ids, process, regions=regions)

    def crop_and_zoom(self, images, camera_ids, bbox, zoom_factor):
        """
        裁剪并放大 - 简化版（需要配合内参调整）。
        注意：这个工具会破坏相机几何，默认只用于实验，不建议作为主工具。
        """
        x_min, y_min, x_max, y_max = bbox
        images = images.clone()

        for cam_id in camera_ids:
            if cam_id >= images.shape[1]:
                continue

            img_batch = images[:, cam_id]
            B, C, H, W = img_batch.shape

            x1 = max(0, min(W - 1, int(x_min * W)))
            y1 = max(0, min(H - 1, int(y_min * H)))
            x2 = max(x1 + 1, min(W, int(x_max * W)))
            y2 = max(y1 + 1, min(H, int(y_max * H)))

            cropped = img_batch[:, :, y1:y2, x1:x2]
            new_H = max(1, int((y2 - y1) * zoom_factor))
            new_W = max(1, int((x2 - x1) * zoom_factor))

            cropped_np = cropped.permute(0, 2, 3, 1).cpu().numpy()
            cropped_np = (np.clip(cropped_np, 0.0, 1.0) * 255).astype(np.uint8)

            resized_list = [
                cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
                for img in cropped_np
            ]
            resized_np = np.array(resized_list).astype(np.float32) / 255.0
            resized_tensor = torch.from_numpy(resized_np).permute(0, 3, 1, 2)

            out_H = min(new_H, H)
            out_W = min(new_W, W)
            images[:, cam_id, :, :out_H, :out_W] = resized_tensor[:, :, :out_H, :out_W].to(images.device)

        return images

    def _apply_to_cameras(self, images, camera_ids, processor, regions=None):
        images = images.clone()

        for cam_id in camera_ids:
            if cam_id >= images.shape[1]:
                continue

            img_batch = images[:, cam_id]
            img_np = img_batch.permute(0, 2, 3, 1).detach().cpu().numpy()
            img_np = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)

            processed_list = []
            for img in img_np:
                if regions:
                    out = img.copy()
                    for region in regions:
                        clipped = self._clip_region(region, img.shape)
                        if clipped is None:
                            continue
                        x1, y1, x2, y2 = clipped
                        out[y1:y2, x1:x2] = processor(out[y1:y2, x1:x2])
                    processed_list.append(out)
                else:
                    processed_list.append(processor(img))

            processed_np = np.array(processed_list).astype(np.float32) / 255.0
            processed_tensor = torch.from_numpy(processed_np).permute(0, 3, 1, 2)
            images[:, cam_id] = processed_tensor.to(images.device)

        return images

    def _clip_region(self, region, shape):
        h, w = shape[:2]
        if len(region) != 4:
            return None
        x1, y1, x2, y2 = [int(v) for v in region]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _enhance_contrast(self, img, factor=1.15):
        factor = float(np.clip(factor, 0.7, 1.6))
        mean = img.astype(np.float32).mean(axis=(0, 1), keepdims=True)
        contrast = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        if factor <= 1.12:
            return contrast
        clahe = self._clahe_luminance(img, clip_limit=1.5)
        return cv2.addWeighted(contrast, 0.75, clahe, 0.25, 0)

    def _apply_gamma_factor(self, img, factor=1.0):
        factor = float(np.clip(factor, 0.5, 1.8))
        inv_gamma = 1.0 / max(factor, 1e-6)
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)
        ]).astype(np.uint8)
        return cv2.LUT(img, table)

    def _clahe_luminance(self, img, clip_limit=2.0, tile_grid_size=(8, 8)):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    def _enhance_low_light_image(self, img, strength=0.65, gamma=1.25, clip_limit=2.0):
        strength = float(np.clip(strength, 0.0, 1.0))
        gamma_img = self._apply_gamma_factor(img, factor=gamma)
        clahe_img = self._clahe_luminance(img, clip_limit=clip_limit)
        enhanced = cv2.addWeighted(gamma_img, 0.6, clahe_img, 0.4, 0)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        value = hsv[:, :, 2].astype(np.float32) / 255.0
        dark_weight = np.clip((0.85 - value) / 0.85, 0.0, 1.0)[:, :, None]
        blend_weight = strength * dark_weight
        out = img.astype(np.float32) * (1.0 - blend_weight) + enhanced.astype(np.float32) * blend_weight
        return np.clip(out, 0, 255).astype(np.uint8)

    def _reduce_glare_image(self, img, threshold=210, strength=0.55):
        threshold = int(np.clip(threshold, 150, 245))
        strength = float(np.clip(strength, 0.0, 0.9))

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        value = hsv[:, :, 2]
        mask = (value > threshold).astype(np.float32)
        if mask.max() <= 0:
            return img

        soft = cv2.GaussianBlur(mask, (0, 0), 5.0)[:, :, None]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        l = lab[:, :, 0]
        compressed_l = np.where(
            l > threshold,
            threshold + (l - threshold) * (1.0 - strength),
            l,
        )
        lab[:, :, 0] = l * (1.0 - soft[:, :, 0]) + compressed_l * soft[:, :, 0]

        glare_reduced = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        hsv_reduced = cv2.cvtColor(glare_reduced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_reduced[:, :, 1] *= (1.0 - 0.2 * soft[:, :, 0])
        return cv2.cvtColor(np.clip(hsv_reduced, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _unsharp_mask(self, img, amount=0.65, radius=1.0):
        amount = float(np.clip(amount, 0.0, 1.8))
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=max(float(radius), 0.1))
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _dark_channel_dehaze(self, img, omega=0.85, t0=0.35):
        image = img.astype(np.float32) / 255.0
        dark = image.min(axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dark = cv2.erode(dark, kernel)

        flat_dark = dark.reshape(-1)
        flat_img = image.reshape(-1, 3)
        top_n = max(1, int(0.001 * flat_dark.size))
        top_idx = np.argpartition(flat_dark, -top_n)[-top_n:]
        airlight = flat_img[top_idx].mean(axis=0)
        airlight = np.maximum(airlight, 0.1)

        normalized = image / airlight.reshape(1, 1, 3)
        transmission = 1.0 - omega * normalized.min(axis=2)
        transmission = cv2.GaussianBlur(transmission, (0, 0), 3.0)
        transmission = np.clip(transmission, t0, 1.0)

        recovered = (image - airlight.reshape(1, 1, 3)) / transmission[:, :, None] + airlight.reshape(1, 1, 3)
        recovered = np.clip(recovered, 0.0, 1.0)
        return (recovered * 255).astype(np.uint8)
