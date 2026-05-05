"""
Vision LLM - 使用视觉语言模型分析图像
"""

import json
import base64
import io
from PIL import Image
import torch
import numpy as np
import requests


class VisionLLM:
    """使用视觉LLM分析相机图像，输出结构化判断"""

    def __init__(self, llm_url="http://localhost:11434", model_name="qwen2.5vl:7b"):
        """
        Args:
            llm_url: Ollama服务地址
            model_name: 模型名称
        """
        self.llm_url = llm_url
        self.model_name = model_name

        # 相机ID到名称的映射
        self.camera_names = {
            0: "CAM_FRONT",
            1: "CAM_FRONT_RIGHT",
            2: "CAM_FRONT_LEFT",
            3: "CAM_BACK",
            4: "CAM_BACK_RIGHT",
            5: "CAM_BACK_LEFT"
        }

    def encode_image(self, image_tensor):
        """
        将tensor图像转换为base64编码

        Args:
            image_tensor: (3, H, W) 或 (H, W, 3) 的tensor

        Returns:
            base64编码的PNG图像
        """
        # 确保是CHW格式
        if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
            img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image_tensor.cpu().numpy()

        # 归一化到0-255
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # 转为PIL Image再转base64
        pil_img = Image.fromarray(img_np)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def analyze_single_image(self, image_tensor, camera_id):
        """
        分析单张图像

        Args:
            image_tensor: (3, H, W) tensor
            camera_id: int

        Returns:
            dict: 分析结果
        """
        prompt = """分析这张图片。用JSON格式回答:
{"conditions":["rain/fog/haze/clear/low_light/glare"],
"problem_regions":[[x1,y1,x2,y2]],
"suggested_tools":["remove_rain/dehaze/enhance_image"]}"""

        # 编码图像
        image_base64 = self.encode_image(image_tensor)

        try:
            response = requests.post(
                f"{self.llm_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_base64]
                        }
                    ],
                    "stream": False
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")

                # 尝试解析JSON
                # 尝试提取并解析JSON
                parsed = self._extract_json(content)
                if parsed is not None:
                    parsed["camera_id"] = camera_id
                    parsed["camera_name"] = self.camera_names.get(camera_id, f"Camera{camera_id}")
                    parsed.setdefault("analysis", content[:100])
                    return parsed

                # 如果解析失败，返回默认
                return self._default_result(camera_id, "分析失败")

            else:
                return self._default_result(camera_id, f"LLM请求失败: {response.status_code}")

        except Exception as e:
            return self._default_result(camera_id, f"异常: {str(e)}")

    def analyze_images(self, images, camera_ids):
        """
        批量分析多张图像

        Args:
            images: (B, N_cams, 3, H, W) tensor
            camera_ids: list 要分析的相机ID

        Returns:
            list: 每个相机的分析结果
        """
        results = []

        for cam_id in camera_ids:
            if cam_id >= images.shape[1]:
                continue

            # 提取单个相机的图像 (B, 3, H, W) 取第一张
            img = images[0, cam_id]  # (3, H, W)
            result = self.analyze_single_image(img, cam_id)
            results.append(result)

        return results

    def _extract_json(self, text):
        """从文本中提取JSON，支持嵌套结构"""
        import re
        # 先尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 找到最外层的{}
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _default_result(self, camera_id, error_msg=""):
        """默认结果"""
        return {
            "camera_id": camera_id,
            "camera_name": self.camera_names.get(camera_id, f"Camera{camera_id}"),
            "analysis": error_msg if error_msg else "无法分析",
            "conditions": [],
            "problem_regions": [],
            "suggested_tools": []
        }

    def merge_analyses(self, analyses):
        """
        合并多个相机的分析结果，生成统一的工具执行计划

        Args:
            analyses: list 每个相机的分析结果

        Returns:
            dict: 合并后的工具执行计划
        """
        # 按工具分组
        tool_plan = {
            "remove_rain": {"camera_ids": [], "regions": []},
            "dehaze": {"camera_ids": [], "regions": []},
            "enhance_image": {"camera_ids": [], "regions": [], "params": {}},
            "crop_and_zoom": {"camera_ids": [], "regions": []}
        }

        for analysis in analyses:
            cam_id = analysis.get("camera_id", 0)
            tools = analysis.get("suggested_tools", [])

            for tool_info in tools:
                # Handle both string tools and dict tools
                if isinstance(tool_info, str):
                    tool_name = tool_info
                    regions = []
                else:
                    tool_name = tool_info.get("tool", "")
                    regions = tool_info.get("target_regions", [])

                if tool_name == "remove_rain":
                    tool_plan["remove_rain"]["camera_ids"].append(cam_id)
                    tool_plan["remove_rain"]["regions"].extend(regions)

                elif tool_name == "dehaze":
                    tool_plan["dehaze"]["camera_ids"].append(cam_id)
                    tool_plan["dehaze"]["regions"].extend(regions)

                elif tool_name == "enhance_image":
                    if cam_id not in tool_plan["enhance_image"]["camera_ids"]:
                        tool_plan["enhance_image"]["camera_ids"].append(cam_id)
                    tool_plan["enhance_image"]["regions"].extend(regions)

                elif tool_name == "crop_and_zoom":
                    tool_plan["crop_and_zoom"]["camera_ids"].append(cam_id)
                    tool_plan["crop_and_zoom"]["regions"].extend(regions)

        return tool_plan
