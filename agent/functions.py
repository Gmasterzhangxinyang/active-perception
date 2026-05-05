"""
Function Calling definitions for Agent
"""

# 可用工具列表
AVAILABLE_TOOLS = [
    {
        "name": "crop_and_zoom",
        "description": "裁剪并放大图像的特定区域，用于处理BEV中边界模糊的区域",
        "parameters": {
            "type": "object",
            "properties": {
                "camera_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "要处理的相机ID列表，如[0,1]表示前置摄像头"
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "裁剪区域[x_min, y_min, x_max, y_max]，归一化坐标0-1",
                    "minItems": 4,
                    "maxItems": 4
                },
                "zoom_factor": {
                    "type": "number",
                    "description": "放大倍数，如2.0表示放大2倍",
                    "default": 2.0
                }
            },
            "required": ["camera_ids", "bbox"]
        }
    },
    {
        "name": "enhance_image",
        "description": "增强图像质量，对比度、锐化、降噪或Gamma校正",
        "parameters": {
            "type": "object",
            "properties": {
                "camera_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "要处理的相机ID列表"
                },
                "enhancement_type": {
                    "type": "string",
                    "enum": ["contrast", "sharpness", "denoise", "gamma"],
                    "description": "增强类型"
                },
                "factor": {
                    "type": "number",
                    "description": "增强强度因子",
                    "default": 1.5
                }
            },
            "required": ["camera_ids", "enhancement_type"]
        }
    },
    {
        "name": "remove_rain",
        "description": "去除图像中的雨滴痕迹",
        "parameters": {
            "type": "object",
            "properties": {
                "camera_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "要处理的相机ID列表"
                },
                "method": {
                    "type": "string",
                    "enum": ["CLAHE", "Gaussian"],
                    "description": "去雨方法",
                    "default": "CLAHE"
                }
            },
            "required": ["camera_ids"]
        }
    },
    {
        "name": "dehaze",
        "description": "去除图像中的雾霾",
        "parameters": {
            "type": "object",
            "properties": {
                "camera_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "要处理的相机ID列表"
                },
                "method": {
                    "type": "string",
                    "enum": ["CLAHE", "AOD-Net"],
                    "description": "去雾方法",
                    "default": "CLAHE"
                }
            },
            "required": ["camera_ids"]
        }
    },
    {
        "name": "finalize",
        "description": "确认当前BEV结果为最终输出，停止优化",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]


def get_tool_by_name(name):
    """根据名称获取工具定义"""
    for tool in AVAILABLE_TOOLS:
        if tool["name"] == name:
            return tool
    return None
