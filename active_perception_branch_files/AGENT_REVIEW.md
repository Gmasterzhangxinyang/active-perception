# Agent Review Checklist

## 当前问题

- [x] Agent 会在没有明确视觉证据时 fallback 到强对比度增强，容易破坏模型输入分布。
- [x] Agent 可能连续重复同一个 action，例如多轮 `enhance_image`，但没有判断重复动作是否有效。
- [x] 每轮 action 后没有验收机制，质量下降时仍继续使用被修改后的图像。
- [x] 日志只记录最终 IoU/Accuracy，缺少每轮 action 前后的 `score`、`edge_density`、`integrity`。
- [ ] 当前 `best_model.pth` 只训练 1 epoch，BEVFusion 本身仍很弱，agent 效果会被底座模型能力限制。
- [ ] 训练 GT 是二值占用，但模型配置是 6 类，类别定义和训练目标仍不完全一致。
- [ ] mini 数据集多数样本未必有雨雾弱光，视觉工具触发场景可能不足。
- [ ] 图像工具仍是传统 CV 处理，虽然已增强为低光/眩光/清晰度工具，但和 BEVFusion 训练分布仍需用 ablation 验证。

## 本轮已修改

- [x] 没有明确视觉问题时直接 `finalize`，不再盲目增强。
- [x] 降低 `enhance_image` 强度，默认从 1.5/1.8 收敛到 1.15，fast mode 使用 1.1。
- [x] 避免重复执行同一种 action。
- [x] action 后比较无 GT 质量指标；如果 `score`/`integrity` 没有提升，则回滚图像并停止。
- [x] 推理日志打印每轮 before/after 的 `score`、`edge_density`、`integrity`、accepted/reason。
- [x] 升级图像工具层：新增 `enhance_low_light`、`reduce_glare`、`sharpen_image`、`deblur_image`，并把旧锐化改为更温和的 unsharp mask。

## 下一步方向

- [ ] 继续训练或重新设计为二分类输出，使 baseline IoU 先达到可用水平。
- [ ] 把 GPT 视觉判断、BEV 指标、历史 action 合并成结构化 state，让 GPT 直接选择工具和参数。
- [ ] 增加 action 黑名单：若某工具在当前样本下降，则本样本不再尝试同类工具。
- [x] 引入 GT-aware debug 模式：实验阶段用 IoU 判断是否接受 action，正式模式再切回无 GT score。
- [x] 引入多候选 action ablation：同一轮对 no-op 和多个工具参数做 forward 对比，再选择最佳候选。
- [ ] 增加样本筛选：优先挑低光、雨雾、模糊、强眩光样本测试 agent。
- [ ] 对每种工具做离线 ablation，确认它对当前 BEVFusion 是否真的有正向作用。
