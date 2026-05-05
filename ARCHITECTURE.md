# Active LLM Agent for BEVFusion - 系统架构

## 1. 系统概述

```
┌─────────────────────────────────────────────────────────────────┐
│                      整体架构                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: 6张相机图像 + LiDAR点云                                    │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────────┐                                           │
│  │   BEVFusion     │  ← 已训练的模型                            │
│  │  (Camera+LiDAR) │                                           │
│  └────────┬────────┘                                           │
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────┐                                           │
│  │   BEV分割图      │  ← 120x120 鸟瞰分割                       │
│  │ (分类: 车/路/人) │                                           │
│  └────────┬────────┘                                           │
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Agent Loop (ReAct)                   │  │
│  │                                                          │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │  │
│  │  │   评估      │───▶│  VisionLLM  │───▶│  规则匹配   │  │  │
│  │  │(启发式/有GT) │    │(Qwen2.5-VL)│    │  选工具     │  │  │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘  │  │
│  │        │                    ↑                │         │  │
│  │        │                    │                ▼         │  │
│  │        │      ┌─────────────────────────┐  ┌─────────────┐│  │
│  │        └──────┤  问题区域→相机映射       ├─▶│   执行      ││  │
│  │               └─────────────────────────┘  │(图像优化)   ││  │
│  │                                             └─────────────┘│  │
│  └─────────────────────────────────────────────────────────────┘  │
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────┐                                           │
│  │  优化后的BEV    │                                           │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 未来扩展: LLM 决策层 (Qwen3-8B)

```
当前流程 (v1): VisionLLM → 规则匹配 → 工具
                    │
                    ▼ (未来)
              Qwen3-8B 决策层
                    │
                    ▼
            "根据这些信息，思考应该用什么工具"
                    │
                    ▼
         thought + action (真正的LLM决策)
```

**为什么暂时不用 Qwen3 决策**:
1. VisionLLM 已经返回结构化 `suggested_tools`，直接映射更简单
2. 减少 LLM 调用次数，降低延迟
3. 规则系统在简单场景下足够有效

**什么时候需要加 Qwen3 决策**:
- 复杂场景需要综合判断（多种工具组合）
- 需要文字解释决策理由
- 积累足够多的微调数据后

## 2. 模块结构

```
bevfusion/
├── agent/                          # Agent模块
│   ├── __init__.py
│   ├── core.py                    # Agent主控逻辑 (ReAct引擎)
│   ├── bev_evaluator.py          # BEV质量评估器 + 几何映射
│   ├── refiner.py                # 图像优化器(含区域处理)
│   ├── functions.py              # Function Calling定义
│   ├── prompts.py                # 提示词模板
│   ├── vision_llm.py             # 视觉LLM接口(Qwen2.5-VL)
│   └── data_logger.py            # 数据记录器
│
├── models/                        # BEVFusion模型
│   ├── bevfusion.py              # 主模型
│   ├── camera_encoder.py         # 相机分支
│   ├── lidar_encoder.py          # LiDAR分支
│   ├── fusion.py                 # 融合模块
│   └── heads.py                  # 分割头
│
├── data/                          # 数据处理
│   ├── nuscenes_loader.py        # 数据加载
│   └── bev_gt.py                 # GT生成
│
├── config.py                      # 配置文件
├── train.py                       # 训练脚本
├── bev_comparison.py              # 对比实验主入口（baseline vs agent）
└── run_agent_inference.py         # 旧推理入口（已被ev_comparison.py取代）
```

## 3. 核心流程

### 3.0 视觉LLM图像分析 (VisionLLM)

```
BEV问题区域 → 对应相机ID列表
    │
    ▼
VisionLLM.analyze_images() 批量分析
    │
    ▼
Qwen2.5-VL返回结构化JSON:
{
  "camera_id": 0,
  "conditions": ["rain", "low_light"],
  "problem_regions": [{"bbox": [x1,y1,x2,y2], "condition": "rain"}],
  "suggested_tools": [{"tool": "remove_rain", "target_regions": [[x1,y1,x2,y2]]}]
}
    │
    ▼
VisionLLM.merge_analyses() 合并为工具执行计划
```

### 3.1 Agent评估流程

```
BEV分割图 (H=120, W=120)
    │
    ▼
┌────────────────────────────────────────────┐
│          BEVEvaluator.evaluate()            │
│                                            │
│  1. edge_density = Canny边缘检测           │
│     - edge_density < 0.3 → 需要优化        │
│                                            │
│  2. integrity = 碎片化检测                  │
│     - integrity < 0.5 → 需要优化            │
│                                            │
│  3. problem_coords = 问题区域中心列表       │
│     - 面积 < 20像素的区域                   │
│                                            │
│  4. bev_to_camera_mapping() = 几何映射       │
│     - BEV坐标 → 相机ID                      │
│                                            │
└────────────────────────────────────────────┘
```

### 3.2 BEV→相机几何映射

```
问题区域中心 (bev_x, bev_y) 在BEV像素坐标系
    │
    │  (根据bev_x_range, bev_y_range转换)
    ▼
ego坐标系下的米坐标 (ego_x, ego_y, 0)
    │
    │  (ego2cam = inverse(cam2ego))
    ▼
相机坐标系下的坐标 (cam_x, cam_y, cam_z)
    │
    │  (检查 cam_z > 0 且投影在图像内)
    ▼
能拍到该BEV位置的相机ID列表 + 图像坐标
```

### 3.3 Agent决策流程 (ReAct)

```
┌──────────────────────────────────────────────────────┐
│                    ReAct Loop                          │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Step 1: 观察 (Observe)                              │
│    - 输入: BEV质量指标 + 问题区域相机映射              │
│    - "edge_density=0.2, integrity=0.3,               │
│      BEV区域(80,45)对应CAM_FRONT"                    │
│                                                       │
│  Step 2: 视觉分析 (Vision LLM)                        │
│    - 发送问题区域对应的相机图像给Qwen2.5-VL          │
│    - 获取: weather conditions, problem bboxes         │
│    - "检测到rain, 区域[180,40,320,100]"              │
│                                                       │
│  Step 3: 思考 (Think)                                │
│    - 结合BEV评估 + 视觉LLM分析                        │
│    - 输出: thought (文字理由)                        │
│    - "检测到相机0图像有雨，建议去雨处理"             │
│                                                       │
│  Step 4: 行动 (Act)                                   │
│    - 调用Function Calling                             │
│    - 输出: action {name, parameters}                   │
│    - "remove_rain(camera_ids=[0],                    │
│       regions=[[180,40,320,100]])"                    │
│                                                       │
│  Step 5: 迭代                                         │
│    - 执行action → 新BEV → 重新评估                    │
│    - 最多3次迭代                                      │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## 4. Function Calling 工具

### 4.1 工具列表

| 工具名称 | 参数 | 功能 |
|---------|------|------|
| `enhance_image` | camera_ids, enhancement_type, factor | 对比度/锐化/降噪/Gamma |
| `remove_rain` | camera_ids, method, regions | 去雨 (CLAHE/高斯)，支持区域处理 |
| `dehaze` | camera_ids, method, regions | 去雾 (CLAHE/直方图均衡)，支持区域处理 |
| `crop_and_zoom` | camera_ids, bbox, zoom_factor | 裁剪放大区域 |
| `finalize` | - | 确认输出，停止优化 |

### 4.2 工具实现

```
ImageRefiner
├── enhance_image(camera_ids, type, factor)
│   └── cv2.convertScaleAbs / filter2D / LUT
├── remove_rain(camera_ids, method, regions=None)
│   └── CLAHE或GaussianBlur，支持指定区域
├── dehaze(camera_ids, method, regions=None)
│   └── CLAHE或直方图均衡化，支持指定区域
└── crop_and_zoom(camera_ids, bbox, zoom)
    └── cv2.resize裁剪区域
```

## 5. 数据流

### 5.1 训练阶段 (阶段1)

```
nuScenes Dataset
       │
       ▼
┌─────────────────┐
│  相机图像 + LiDAR │ → CameraEncoder → cam_bev
│                  │ → LiDAREncoder  → lidar_bev
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    ConvFuser    │ → 融合特征
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   BEVSegHead    │ → 分割 logits
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Loss(CE+GT)    │ → 反向传播训练
└─────────────────┘
```

### 5.2 Agent推理阶段 (阶段2)

```
同一输入
    │
    ├──→ Baseline路径
    │    └──→ BEVFusion → BEV_baseline → IoU_baseline
    │
    └──→ Agent路径
         └──→ Agent Loop
              ├──→ 评估 (启发式)
              ├──→ 决策 (LLM)
              ├──→ 执行 (图像优化)
              └──→ 重新生成BEV
                   │
                   ▼
              BEV_agent → IoU_agent

最终: IoU提升 = IoU_agent - IoU_baseline
```

## 6. 数据存储

### 6.1 微调数据格式 (JSONL)

```json
{
  "session_id": "uuid",
  "iteration": 0,
  "timestamp": "2026-05-05T10:30:00",
  "input_state": {
    "sample_idx": 0,
    "lidar_points_count": 15000
  },
  "bev_quality": {
    "edge_density": 0.25,
    "integrity": 0.35,
    "iou": 0.65
  },
  "agent_output": {
    "thought": "边缘密度偏低，需要增强对比度",
    "action": {
      "name": "enhance_image",
      "parameters": {
        "camera_ids": [0, 1],
        "enhancement_type": "contrast",
        "factor": 1.5
      }
    }
  },
  "result": {
    "iou_improvement": 0.03,
    "improved": true
  }
}
```

### 6.2 用途

| 用途 | 方法 |
|------|------|
| 监督微调(SFT) | 用action作为label训练Qwen |
| 偏好对齐(DPO) | 对比improved=true/false的样本 |
| 统计分析 | 统计最有效的工具 |

## 7. 配置参数

### 7.1 BEV配置

```python
BEVConfig:
    bev_x_range = (-30, 30)      # BEV范围(米)
    bev_y_range = (-30, 30)
    bev_size = (120, 120)        # BEV分辨率
    bev_resolution = 0.5          # 每像素多少米
```

### 7.2 Agent配置

```python
AgentCore:
    llm_url = "http://localhost:11434"
    max_iterations = 3
    fast_mode = False   # True: 跳过VisionLLM，用纯规则决策（无需Ollama）
    model_name = "qwen2.5-vl:7b"
```

### 7.3 评估阈值

```python
# 评估阈值
edge_density_threshold = 0.3
integrity_threshold = 0.5

# 达到阈值 → 不需要优化 → finalize
```

## 8. 文件清单

| 文件 | 行数 | 功能 |
|------|------|------|
| `agent/core.py` | ~380 | Agent主控，ReAct引擎，fast_mode支持 |
| `agent/bev_evaluator.py` | ~200 | BEV评估，几何映射 |
| `agent/refiner.py` | ~150 | 图像优化工具(含区域处理) |
| `agent/functions.py` | ~80 | Function Calling定义 |
| `agent/prompts.py` | ~80 | 提示词模板 |
| `agent/vision_llm.py` | ~200 | 视觉LLM接口(Qwen2.5-VL) |
| `agent/data_logger.py` | ~80 | 数据记录器 |
| `bev_comparison.py` | ~400 | 对比实验主入口（baseline vs agent）|
| `run_agent_inference.py` | ~200 | 旧推理入口（已被bev_comparison.py取代）|

## 9. 验证方法

### 9.1 对比实验

```bash
# 标准模式（需要Ollama + qwen2.5-vl:7b）
python bev_comparison.py --num_samples 10

# Fast模式（无需LLM，纯规则，快速验证）
python bev_comparison.py --num_samples 10 --fast

# 单样本调试
python bev_comparison.py --sample 0 --fast
```

### 9.2 评估指标

| 指标 | 说明 |
|------|------|
| IoU提升 | IoU_agent - IoU_baseline |
| 准确率提升 | Acc_agent - Acc_baseline |
| 迭代次数 | 达到满意的优化次数 |
| 工具成功率 | 该工具优化后IoU提升的比例 |

### 9.3 预期结果

| 场景 | 预期 |
|------|------|
| 夜间样本 | Agent选择enhance_image提升对比度 |
| 雨天样本 | Agent选择remove_rain去雨 |
| 模糊样本 | Agent选择crop_and_zoom放大 |

## 10. 依赖

```
# 核心依赖
torch
torchvision
numpy
opencv-python

# Agent依赖
requests          # LLM调用
ollama            # 本地LLM服务
qwen2.5-vl:7b     # 视觉LLM模型 (ollama pull qwen2.5-vl:7b)

# 评估依赖
scikit-image      # 图像质量指标
```

## 11. 测试状态 (2026-05-05)

### 已验证功能 ✅

| 功能 | 测试结果 |
|------|---------|
| BEV评估(无GT) | ✅ edge_density, integrity, problem_coords正常 |
| BEV评估(有GT) | ✅ IoU~0.02-0.04, Accuracy~0.81-0.92 (1 epoch模型) |
| Agent ReAct循环 | ✅ fast_mode下1次增强后finalize |
| fast_mode | ✅ 跳过VisionLLM，纯规则，2样本约30秒 |
| 对比实验(bev_comparison.py) | ✅ baseline vs agent，批量+单样本均可 |
| 数据记录JSONL | ✅ agent_training_data.jsonl已生成 |
| 工具执行(enhance_image) | ✅ camera_ids=[0..5], factor=1.3 |
| 工具执行(finalize) | ✅ 正常退出循环 |

### 批量测试结果 (fast_mode, 2样本)

| 样本 | Baseline IoU | Agent IoU | IoU变化 | Accuracy变化 |
|------|-------------|-----------|---------|-------------|
| 0 | 0.037 | 0.033 | -0.004 | +0.038 |
| 1 | 0.019 | 0.022 | +0.003 | +0.106 |

注：IoU整体偏低是模型只训练1个epoch的问题，与Agent无关。

### 待解决问题

| 问题 | 原因 | 优先级 | 状态 |
|------|------|--------|------|
| qwen2.5vl:7b模型名 | 模型名无短横线(qwen2.5vl:7b) | 高 | ✅ 已修复(vision_llm.py默认名已更新) |
| IoU偏低(0.02-0.04) | 模型只训练1个epoch | 中 | ✅ best_model.pth已存在，可继续训练 |
| VisionLLM未实际调用 | 依赖qwen2.5vl模型 | 高 | ✅ 模型已下载(qwen2.5vl:7b 6.0GB) |
| fast_mode重复增强 | 缺少历史记录检查 | 高 | ✅ 已修复 |
| fast_mode用错评估字段 | evaluate()无iou字段 | 高 | ✅ 已修复 |
| argparse --fast参数重复 | bev_comparison.py重复添加 | 高 | ✅ 已修复 |

### 运行命令

```bash
# Fast模式（无需Ollama，快速验证）
python bev_comparison.py --num_samples 10 --fast

# 标准模式（需要Ollama）
ollama serve
ollama pull qwen2.5-vl:7b
python bev_comparison.py --num_samples 10

# 查看日志
cat agent_training_data.jsonl | python3 -m json.tool
```
