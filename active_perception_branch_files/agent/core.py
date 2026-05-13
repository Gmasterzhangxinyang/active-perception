"""
ReAct Agent Core - Agent主控逻辑
"""

import json
import uuid
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, FEW_SHOT_EXAMPLES
from .functions import AVAILABLE_TOOLS
from .bev_evaluator import BEVEvaluator
from .refiner import ImageRefiner
from .vision_llm import VisionLLM


class AgentCore:
    """Agent核心，ReAct循环引擎"""

    def __init__(
        self,
        model_name="gpt-5.4-mini",
        max_iterations=3,
        fast_mode=False,
        api_key=None,
        min_score_delta=0.005,
        gt_debug=False,
        ablation=False,
        min_iou_delta=1e-6,
    ):
        """
        Args:
            model_name: OpenAI GPT模型名称
            max_iterations: 最大迭代次数
            fast_mode: 跳过VisionLLM，用纯规则决策（快速模式）
            api_key: OpenAI API Key；默认读取 OPENAI_API_KEY 环境变量
            min_score_delta: 接受一次图像修改所需的最小无GT质量提升
            gt_debug: 实验模式，用GT IoU验收动作
            ablation: 对同一决策扩展多个候选动作并逐个评估
            min_iou_delta: GT debug模式下接受动作所需的最小IoU提升
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.fast_mode = fast_mode
        self.min_score_delta = min_score_delta
        self.gt_debug = gt_debug
        self.ablation = ablation
        self.min_iou_delta = min_iou_delta
        self.evaluator = BEVEvaluator()
        self.refiner = ImageRefiner()
        self.vision_llm = VisionLLM(model_name=model_name, api_key=api_key) if not fast_mode else None
        self.session_id = str(uuid.uuid4())

    def run(self, model, images, intrinsics, extrinsics, lidar_points, lidar_mask, bev_cfg=None, gt_bev=None):
        """
        运行Agent循环

        Args:
            model: BEVFusion模型
            images: (B, N_cams, 3, H, W)
            intrinsics: (B, N_cams, 3, 3)
            extrinsics: (B, N_cams, 4, 4)
            lidar_points: (B, N_pts, 5)
            lidar_mask: (B, N_pts)
            bev_cfg: BEV配置字典
            gt_bev: GT BEV，仅用于gt_debug实验模式

        Returns:
            dict: 最终结果和决策历史
        """
        history = []
        bev_cfg = bev_cfg or {}

        # 首次生成BEV
        logits, bev_seg = model(images, intrinsics, extrinsics, lidar_points, lidar_mask)
        cam_bev = bev_seg[0] if bev_seg.dim() > 2 else bev_seg

        # 评估
        eval_result = self.evaluator.evaluate(cam_bev)
        gt_eval = self._evaluate_with_gt(cam_bev, gt_bev)
        history.append({"iteration": 0, "eval": eval_result, "gt_eval": gt_eval, "action": None})

        # Agent循环
        for i in range(self.max_iterations):
            iteration = i + 1

            # 检查是否需要优化
            if not eval_result["needs_optimization"]:
                return {
                    "final_bev": bev_seg,
                    "history": history,
                    "finalized": True
                }

            # 生成问题区域到相机的映射
            problem_camera_mapping = self._get_problem_camera_mapping(
                eval_result["problem_coords"], extrinsics, intrinsics, bev_cfg
            )

            # 获取需要分析的相机ID
            camera_ids_to_analyze = self._get_unique_camera_ids(problem_camera_mapping)

            # 使用视觉LLM分析这些相机的图像（fast_mode跳过）
            if self.fast_mode:
                vision_analysis = []
            else:
                vision_analysis = self._analyze_images_with_vision_llm(
                    images, camera_ids_to_analyze
                )

            # 生成描述
            problem_areas = self._format_problem_areas(
                eval_result["problem_coords"],
                problem_camera_mapping,
                vision_analysis
            )

            # 结合BEV评估和视觉LLM分析做决策
            decision = self._make_decision(
                eval_result,
                vision_analysis,
                problem_areas,
                history=history
            )

            if decision is None:
                decision = {
                    "thought": "无法决定，使用finalize",
                    "action": {"name": "finalize", "parameters": {}}
                }

            history.append({
                "iteration": iteration,
                "decision": decision,
                "vision_analysis": vision_analysis,
                "eval_before": eval_result
            })

            # 检查是否是finalize
            if decision["action"]["name"] == "finalize":
                return {
                    "final_bev": bev_seg,
                    "history": history,
                    "finalized": True
                }

            # 执行action。先在候选图像上尝试，若质量下降则回滚。
            prev_images = images
            prev_bev_seg = bev_seg
            prev_eval = eval_result
            prev_gt_eval = gt_eval
            candidate_results = self._evaluate_action_candidates(
                model,
                decision["action"],
                images,
                intrinsics,
                extrinsics,
                lidar_points,
                lidar_mask,
                gt_bev,
            )
            selected = candidate_results[0]

            # 评估
            candidate_images = selected["images"]
            candidate_bev_seg = selected["bev_seg"]
            new_eval = selected["eval"]
            new_gt_eval = selected["gt_eval"]
            accepted, accept_reason = self._accept_candidate(prev_eval, new_eval, prev_gt_eval, new_gt_eval)
            score_delta = new_eval.get("score", 0.0) - prev_eval.get("score", 0.0)
            history.append({
                "iteration": iteration,
                "eval": new_eval,
                "gt_eval": new_gt_eval,
                "accepted": accepted,
                "accept_reason": accept_reason,
                "score_delta": score_delta,
                "selected_action": selected["action"],
                "candidate_results": [
                    self._summarize_candidate_result(result, prev_eval, prev_gt_eval)
                    for result in candidate_results
                ]
            })

            if not accepted:
                history.append({
                    "iteration": iteration,
                    "decision": {
                        "thought": f"动作后质量未提升，回滚并停止: {accept_reason}",
                        "action": {"name": "finalize", "parameters": {}}
                    },
                    "rolled_back": True,
                    "eval_before": prev_eval,
                    "eval_after": new_eval,
                    "gt_before": prev_gt_eval,
                    "gt_after": new_gt_eval
                })
                images = prev_images
                bev_seg = prev_bev_seg
                eval_result = prev_eval
                gt_eval = prev_gt_eval
                return {
                    "final_bev": bev_seg,
                    "history": history,
                    "finalized": True,
                    "reason": "action_rejected"
                }

            images = candidate_images
            bev_seg = candidate_bev_seg
            eval_result = new_eval
            gt_eval = new_gt_eval

        # 达到最大迭代次数
        return {
            "final_bev": bev_seg,
            "history": history,
            "finalized": False,
            "reason": "达到最大迭代次数"
        }

    def _get_problem_camera_mapping(self, problem_coords, extrinsics, intrinsics, bev_cfg):
        """获取问题区域对应的相机"""
        if not problem_coords:
            return []

        try:
            mapping = self.evaluator.bev_to_camera_mapping(
                problem_coords, extrinsics, intrinsics, bev_cfg
            )
            return mapping
        except Exception as e:
            import traceback
            print(f"映射失败: {e}")
            traceback.print_exc()
            return []

    def _get_unique_camera_ids(self, problem_camera_mapping):
        """从映射中获取需要分析的相机ID"""
        camera_ids = set()
        for m in problem_camera_mapping:
            camera_ids.update(m.get("camera_ids", []))
        return list(camera_ids) if camera_ids else [0, 1, 2, 3, 4, 5]

    def _analyze_images_with_vision_llm(self, images, camera_ids):
        """使用视觉LLM分析图像"""
        try:
            analyses = self.vision_llm.analyze_images(images, camera_ids)
            return analyses
        except Exception as e:
            print(f"视觉LLM分析失败: {e}")
            return []

    def _make_decision(self, eval_result, vision_analysis, problem_areas, history=None):
        """根据BEV评估和视觉LLM分析做决策"""

        # fast_mode: 纯规则决策，不依赖VisionLLM
        # 只做一次增强，之后直接finalize
        if self.fast_mode:
            integrity = eval_result.get("integrity", 1.0)
            already_enhanced = any(
                h.get("decision", {}).get("action", {}).get("name") == "enhance_image"
                for h in (history or []) if isinstance(h, dict)
            )
            edge_density = eval_result.get("edge_density", 1.0)
            if edge_density < 0.02 and integrity < 0.5 and not already_enhanced:
                return {
                    "thought": f"[FastMode] edge_density={edge_density:.3f}, integrity={integrity:.3f}，保守增强一次",
                    "action": {
                        "name": "enhance_image",
                        "parameters": {"camera_ids": [0, 1, 2], "enhancement_type": "contrast", "factor": 1.1}
                    }
                }
            else:
                return {
                    "thought": f"[FastMode] 无可靠视觉证据或已尝试动作，完成",
                    "action": {"name": "finalize", "parameters": {}}
                }

        # 如果有视觉LLM的分析结果，优先使用
        if vision_analysis:
            if self._vision_has_only_clear_or_errors(vision_analysis):
                return {
                    "thought": "视觉分析未发现明确雨雾/弱光问题，避免无依据增强，直接finalize",
                    "action": {"name": "finalize", "parameters": {}}
                }

            # 直接从analysis中提取conditions来决定工具
            for analysis in vision_analysis:
                cam_id = analysis.get("camera_id", 0)
                conditions = self._normalize_conditions(analysis.get("conditions", []))

                # 根据conditions决定工具（更准确的匹配）
                if "rain" in conditions:
                    decision = {
                        "thought": f"检测到{analysis.get('camera_name', cam_id)}相机图像有雨，建议去雨处理",
                        "action": {
                            "name": "remove_rain",
                            "parameters": {"camera_ids": [cam_id], "regions": None}
                        }
                    }
                    return self._avoid_repeated_action(decision, history)
                elif "fog" in conditions or "haze" in conditions:
                    decision = {
                        "thought": f"检测到{analysis.get('camera_name', cam_id)}相机图像有雾/霾，建议去雾处理",
                        "action": {
                            "name": "dehaze",
                            "parameters": {"camera_ids": [cam_id], "regions": None}
                        }
                    }
                    return self._avoid_repeated_action(decision, history)
                elif "blur" in conditions or "motion_blur" in conditions:
                    decision = {
                        "thought": f"检测到{analysis.get('camera_name', cam_id)}相机图像有模糊，建议轻量去模糊",
                        "action": {
                            "name": "deblur_image",
                            "parameters": {"camera_ids": [cam_id], "strength": 0.75}
                        }
                    }
                    return self._avoid_repeated_action(decision, history)
                elif "noise" in conditions:
                    decision = {
                        "thought": f"检测到{analysis.get('camera_name', cam_id)}相机图像噪声明显，建议降噪",
                        "action": {
                            "name": "enhance_image",
                            "parameters": {"camera_ids": [cam_id], "enhancement_type": "denoise", "factor": 1.0}
                        }
                    }
                    return self._avoid_repeated_action(decision, history)
                elif "glare" in conditions or "overexposed" in conditions:
                    decision = {
                        "thought": f"检测到{analysis.get('camera_name', cam_id)}相机图像有眩光/过曝，建议压制高光",
                        "action": {
                            "name": "reduce_glare",
                            "parameters": {"camera_ids": [cam_id], "threshold": 210, "strength": 0.55}
                        }
                    }
                    return self._avoid_repeated_action(decision, history)
                elif "low_light" in conditions or "underexposed" in conditions:
                    decision = {
                        "thought": f"检测到{analysis.get('camera_name', cam_id)}相机图像弱光，建议低光增强",
                        "action": {
                            "name": "enhance_low_light",
                            "parameters": {"camera_ids": [cam_id], "strength": 0.65, "gamma": 1.25}
                        }
                    }
                    return self._avoid_repeated_action(decision, history)

            # Fallback: 如果没有匹配到conditions，使用merge_analyses的suggested_tools
            tool_plan = self.vision_llm.merge_analyses(vision_analysis)

            # 按优先级选择工具
            if tool_plan["remove_rain"]["camera_ids"]:
                cam_ids = tool_plan["remove_rain"]["camera_ids"]
                regions = tool_plan["remove_rain"]["regions"]
                decision = {
                    "thought": f"检测到{cam_ids}相机图像有雨，建议去雨处理",
                    "action": {
                        "name": "remove_rain",
                        "parameters": {
                            "camera_ids": cam_ids,
                            "regions": regions if regions else None
                        }
                    }
                }
                return self._avoid_repeated_action(decision, history)

            if tool_plan["dehaze"]["camera_ids"]:
                cam_ids = tool_plan["dehaze"]["camera_ids"]
                regions = tool_plan["dehaze"]["regions"]
                decision = {
                    "thought": f"检测到{cam_ids}相机图像有雾/霾，建议去雾处理",
                    "action": {
                        "name": "dehaze",
                        "parameters": {
                            "camera_ids": cam_ids,
                            "regions": regions if regions else None
                        }
                    }
                }
                return self._avoid_repeated_action(decision, history)

            if tool_plan["enhance_image"]["camera_ids"]:
                cam_ids = tool_plan["enhance_image"]["camera_ids"]
                decision = {
                    "thought": f"检测到{cam_ids}相机图像需要增强",
                    "action": {
                        "name": "enhance_image",
                        "parameters": {
                            "camera_ids": cam_ids,
                            "enhancement_type": "contrast",
                            "factor": 1.15
                        }
                    }
                }
                return self._avoid_repeated_action(decision, history)

            for tool_name, thought in [
                ("reduce_glare", "检测到相机图像存在眩光/过曝，建议高光压制"),
                ("deblur_image", "检测到相机图像存在模糊，建议轻量去模糊"),
                ("enhance_low_light", "检测到相机图像弱光，建议低光增强"),
                ("sharpen_image", "检测到相机图像边缘不清，建议温和锐化"),
            ]:
                if tool_plan[tool_name]["camera_ids"]:
                    cam_ids = tool_plan[tool_name]["camera_ids"]
                    decision = {
                        "thought": f"{thought}: {cam_ids}",
                        "action": {
                            "name": tool_name,
                            "parameters": {
                                "camera_ids": cam_ids,
                                "regions": tool_plan[tool_name]["regions"] or None,
                            }
                        }
                    }
                    return self._avoid_repeated_action(decision, history)

        # 没有明确视觉证据时不再盲目增强。当前模型对输入分布很敏感，强增强更容易降质。
        return {
            "thought": "没有可靠视觉问题或工具建议，保留当前BEV结果",
            "action": {"name": "finalize", "parameters": {}}
        }

    def _accept_action(self, before, after):
        """用无GT指标判断动作是否值得保留。"""
        score_delta = after.get("score", 0.0) - before.get("score", 0.0)
        edge_delta = after.get("edge_density", 0.0) - before.get("edge_density", 0.0)
        integrity_delta = after.get("integrity", 0.0) - before.get("integrity", 0.0)

        if score_delta >= self.min_score_delta:
            return True, f"score提升 {score_delta:+.4f}"

        if integrity_delta >= 0.05 and edge_delta >= -0.01:
            return True, f"integrity提升 {integrity_delta:+.4f}, edge变化 {edge_delta:+.4f}"

        return False, f"score变化 {score_delta:+.4f}, integrity变化 {integrity_delta:+.4f}, edge变化 {edge_delta:+.4f}"

    def _accept_candidate(self, before_eval, after_eval, before_gt_eval=None, after_gt_eval=None):
        """根据当前模式验收候选动作。gt_debug开启时优先用真实IoU。"""
        if self.gt_debug and before_gt_eval and after_gt_eval:
            iou_delta = after_gt_eval.get("iou", 0.0) - before_gt_eval.get("iou", 0.0)
            acc_delta = after_gt_eval.get("accuracy", 0.0) - before_gt_eval.get("accuracy", 0.0)
            if iou_delta >= self.min_iou_delta:
                return True, f"GT IoU提升 {iou_delta:+.4f}, Acc变化 {acc_delta:+.4f}"
            return False, f"GT IoU变化 {iou_delta:+.4f}, Acc变化 {acc_delta:+.4f}"

        return self._accept_action(before_eval, after_eval)

    def _evaluate_action_candidates(
        self,
        model,
        action,
        images,
        intrinsics,
        extrinsics,
        lidar_points,
        lidar_mask,
        gt_bev=None,
    ):
        """评估一个动作或其ablation候选，返回按质量排序后的候选列表。"""
        actions = self._build_candidate_actions(action) if self.ablation else [action]
        results = []

        for candidate_action in actions:
            candidate_images = self._execute_action(candidate_action, images)
            logits, candidate_bev_seg = model(
                candidate_images, intrinsics, extrinsics, lidar_points, lidar_mask
            )
            cam_bev = candidate_bev_seg[0] if candidate_bev_seg.dim() > 2 else candidate_bev_seg
            candidate_eval = self.evaluator.evaluate(cam_bev)
            candidate_gt_eval = self._evaluate_with_gt(cam_bev, gt_bev)
            results.append({
                "action": candidate_action,
                "images": candidate_images,
                "bev_seg": candidate_bev_seg,
                "eval": candidate_eval,
                "gt_eval": candidate_gt_eval,
            })

        results.sort(key=self._candidate_rank_key, reverse=True)
        return results

    def _build_candidate_actions(self, action):
        """从一个初始动作扩展出可比较的候选动作，包含no-op基线。"""
        candidates = [{"name": "finalize", "parameters": {}, "label": "no_op"}]
        name = action.get("name")
        params = action.get("parameters", {})
        camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
        regions = params.get("regions")

        if name == "enhance_image":
            for factor in [1.05, 1.1, 1.15, 1.25]:
                candidates.append({
                    "name": "enhance_image",
                    "parameters": {
                        "camera_ids": camera_ids,
                        "enhancement_type": "contrast",
                        "factor": factor,
                    }
                })
            for factor in [0.9, 1.05, 1.15]:
                candidates.append({
                    "name": "enhance_image",
                    "parameters": {
                        "camera_ids": camera_ids,
                        "enhancement_type": "gamma",
                        "factor": factor,
                    }
                })
            candidates.append({
                "name": "enhance_image",
                "parameters": {
                    "camera_ids": camera_ids,
                    "enhancement_type": "denoise",
                    "factor": 1.0,
                }
            })
            candidates.extend(self._low_light_candidates(camera_ids, regions))
            candidates.extend(self._clarity_candidates(camera_ids, regions))
        elif name == "enhance_low_light":
            candidates.extend(self._low_light_candidates(camera_ids, regions))
            candidates.extend([
                {
                    "name": "enhance_image",
                    "parameters": {
                        "camera_ids": camera_ids,
                        "enhancement_type": "gamma",
                        "factor": factor,
                    }
                }
                for factor in [1.1, 1.25, 1.4]
            ])
        elif name == "reduce_glare":
            for threshold in [195, 210, 225]:
                for strength in [0.4, 0.6]:
                    candidates.append({
                        "name": "reduce_glare",
                        "parameters": {
                            "camera_ids": camera_ids,
                            "threshold": threshold,
                            "strength": strength,
                            "regions": regions,
                        }
                    })
            candidates.extend(self._low_light_candidates(camera_ids, regions))
            candidates.append({
                "name": "enhance_image",
                "parameters": {
                    "camera_ids": camera_ids,
                    "enhancement_type": "denoise",
                    "factor": 1.0,
                }
            })
        elif name in {"sharpen_image", "deblur_image"}:
            candidates.extend(self._clarity_candidates(camera_ids, regions))
            candidates.append({
                "name": "enhance_image",
                "parameters": {
                    "camera_ids": camera_ids,
                    "enhancement_type": "denoise",
                    "factor": 1.0,
                }
            })
        elif name == "remove_rain":
            for method in ["CLAHE", "Gaussian", "Median", "Bilateral"]:
                candidates.append({
                    "name": "remove_rain",
                    "parameters": {
                        "camera_ids": camera_ids,
                        "method": method,
                        "regions": regions,
                    }
                })
            candidates.append({
                "name": "enhance_image",
                "parameters": {
                    "camera_ids": camera_ids,
                    "enhancement_type": "denoise",
                    "factor": 1.0,
                }
            })
        elif name == "dehaze":
            for method in ["CLAHE", "HE", "DCP"]:
                candidates.append({
                    "name": "dehaze",
                    "parameters": {
                        "camera_ids": camera_ids,
                        "method": method,
                        "regions": regions,
                    }
                })
        elif name == "crop_and_zoom":
            candidates.append(action)
        elif name != "finalize":
            candidates.append(action)

        return self._dedupe_actions(candidates)

    def _low_light_candidates(self, camera_ids, regions=None):
        candidates = []
        for strength, gamma in [(0.45, 1.15), (0.65, 1.25), (0.8, 1.35)]:
            candidates.append({
                "name": "enhance_low_light",
                "parameters": {
                    "camera_ids": camera_ids,
                    "strength": strength,
                    "gamma": gamma,
                    "regions": regions,
                }
            })
        return candidates

    def _clarity_candidates(self, camera_ids, regions=None):
        candidates = []
        for strength in [0.45, 0.65, 0.85]:
            candidates.append({
                "name": "sharpen_image",
                "parameters": {
                    "camera_ids": camera_ids,
                    "strength": strength,
                    "regions": regions,
                }
            })
        for strength in [0.55, 0.75, 0.95]:
            candidates.append({
                "name": "deblur_image",
                "parameters": {
                    "camera_ids": camera_ids,
                    "strength": strength,
                    "regions": regions,
                }
            })
        return candidates

    def _dedupe_actions(self, actions):
        seen = set()
        unique = []
        for action in actions:
            signature = json.dumps(action, sort_keys=True, default=str)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(action)
        return unique

    def _candidate_rank_key(self, result):
        """排序候选。gt_debug下用IoU，否则用无GT质量分。"""
        eval_result = result.get("eval") or {}
        gt_eval = result.get("gt_eval")
        action_name = result.get("action", {}).get("name")
        action_bonus = 0 if action_name == "finalize" else 1

        if self.gt_debug and gt_eval:
            return (
                gt_eval.get("iou", 0.0),
                gt_eval.get("accuracy", 0.0),
                eval_result.get("score", 0.0),
                action_bonus,
            )

        return (
            eval_result.get("score", 0.0),
            eval_result.get("integrity", 0.0),
            eval_result.get("edge_density", 0.0),
            action_bonus,
        )

    def _evaluate_with_gt(self, bev_seg, gt_bev):
        if not self.gt_debug or gt_bev is None:
            return None
        return self.evaluator.evaluate_with_gt(bev_seg, gt_bev)

    def _summarize_candidate_result(self, result, baseline_eval, baseline_gt_eval=None):
        eval_result = result.get("eval") or {}
        gt_eval = result.get("gt_eval")
        summary = {
            "action": result.get("action"),
            "eval": self._compact_eval(eval_result),
            "score_delta": float(eval_result.get("score", 0.0) - baseline_eval.get("score", 0.0)),
        }

        if gt_eval:
            summary["gt_eval"] = {
                "iou": float(gt_eval.get("iou", 0.0)),
                "accuracy": float(gt_eval.get("accuracy", 0.0)),
            }
            if baseline_gt_eval:
                summary["iou_delta"] = float(
                    gt_eval.get("iou", 0.0) - baseline_gt_eval.get("iou", 0.0)
                )

        return summary

    def _compact_eval(self, eval_result):
        return {
            "edge_density": float(eval_result.get("edge_density", 0.0)),
            "integrity": float(eval_result.get("integrity", 0.0)),
            "score": float(eval_result.get("score", 0.0)),
            "needs_optimization": bool(eval_result.get("needs_optimization", False)),
            "class_counts": {
                str(k): int(v) for k, v in eval_result.get("class_counts", {}).items()
            },
            "num_problem_coords": len(eval_result.get("problem_coords", [])),
        }

    def _normalize_conditions(self, conditions):
        """兼容字符串、列表和模型偶发的复合输出。"""
        if isinstance(conditions, str):
            conditions = [conditions]
        normalized = set()
        for condition in conditions or []:
            for part in str(condition).replace("/", ",").split(","):
                value = part.strip().lower()
                if value:
                    normalized.add(value)
        return normalized

    def _vision_has_only_clear_or_errors(self, analyses):
        """没有明确可操作问题时让agent停手。"""
        actionable = {
            "rain", "fog", "haze", "low_light", "glare",
            "overexposed", "underexposed", "blur", "motion_blur", "noise"
        }
        saw_valid_analysis = False
        for analysis in analyses:
            text = str(analysis.get("analysis", ""))
            if text.startswith("异常") or text.startswith("分析失败") or text.startswith("无法分析"):
                continue
            saw_valid_analysis = True
            conditions = self._normalize_conditions(analysis.get("conditions", []))
            tools = analysis.get("suggested_tools", [])
            if conditions & actionable or tools:
                return False
        return True if saw_valid_analysis else True

    def _action_signature(self, action):
        params = action.get("parameters", {})
        return (
            action.get("name"),
            tuple(params.get("camera_ids", [])),
            params.get("enhancement_type"),
            params.get("method"),
        )

    def _avoid_repeated_action(self, decision, history=None):
        signature = self._action_signature(decision.get("action", {}))
        for item in history or []:
            old_action = item.get("decision", {}).get("action") if isinstance(item, dict) else None
            if old_action and self._action_signature(old_action) == signature:
                return {
                    "thought": f"动作 {signature[0]} 已经尝试过，避免重复修改输入，直接finalize",
                    "action": {"name": "finalize", "parameters": {}}
                }
        return decision

    def _format_problem_areas(self, problem_coords, problem_camera_mapping, vision_analysis=None):
        """格式化问题区域描述"""
        if not problem_coords:
            return None

        mapping_dict = {}
        for m in problem_camera_mapping:
            bev_center = tuple(m.get("bev_center", [0, 0]))
            mapping_dict[bev_center] = m.get("camera_ids", [])

        areas = []
        for idx, region in enumerate(problem_coords[:3]):
            bbox = region["bbox"]
            center = region["center"]
            camera_ids = mapping_dict.get(tuple(center), [])

            # 如果有视觉LLM分析，添加更多信息
            vision_info = ""
            if vision_analysis:
                for analysis in vision_analysis:
                    if analysis.get("camera_id") in camera_ids:
                        conditions = analysis.get("conditions", [])
                        if conditions:
                            vision_info = f" [视觉检测: {','.join(conditions)}]"

            if camera_ids:
                camera_names = self._get_camera_names(camera_ids)
                areas.append(
                    f"BEV区域({center[0]},{center[1]})"
                    f"，对应{camera_names}(ID:{camera_ids})"
                    f"，bbox:[{bbox[0]},{bbox[1]}-{bbox[2]},{bbox[3]}]"
                    f"{vision_info}"
                )

        return ", ".join(areas) if areas else None

    def _get_camera_names(self, camera_ids):
        """相机ID转名称"""
        camera_names = {
            0: "CAM_FRONT",
            1: "CAM_FRONT_RIGHT",
            2: "CAM_FRONT_LEFT",
            3: "CAM_BACK",
            4: "CAM_BACK_RIGHT",
            5: "CAM_BACK_LEFT"
        }
        return [camera_names.get(i, f"Camera{i}") for i in camera_ids]

    def _execute_action(self, action, images):
        """执行action"""
        name = action["name"]
        params = action.get("parameters", {})
        regions = params.get("regions")

        if name == "enhance_image":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            enhancement_type = params.get("enhancement_type", "contrast")
            factor = params.get("factor", 1.5)
            return self.refiner.enhance_image(images, camera_ids, enhancement_type, factor)

        elif name == "enhance_low_light":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            strength = params.get("strength", 0.65)
            gamma = params.get("gamma", 1.25)
            clip_limit = params.get("clip_limit", 2.0)
            return self.refiner.enhance_low_light(images, camera_ids, strength, gamma, clip_limit, regions)

        elif name == "reduce_glare":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            threshold = params.get("threshold", 210)
            strength = params.get("strength", 0.55)
            return self.refiner.reduce_glare(images, camera_ids, threshold, strength, regions)

        elif name == "sharpen_image":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            strength = params.get("strength", 0.65)
            radius = params.get("radius", 1.0)
            return self.refiner.sharpen_image(images, camera_ids, strength, radius, regions)

        elif name == "deblur_image":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            strength = params.get("strength", 0.75)
            return self.refiner.deblur_image(images, camera_ids, strength, regions)

        elif name == "remove_rain":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            method = params.get("method", "CLAHE")
            return self.refiner.remove_rain(images, camera_ids, method, regions)

        elif name == "dehaze":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            method = params.get("method", "CLAHE")
            return self.refiner.dehaze(images, camera_ids, method, regions)

        elif name == "crop_and_zoom":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            bbox = params.get("bbox", [0.3, 0.3, 0.7, 0.7])
            zoom_factor = params.get("zoom_factor", 2.0)
            return self.refiner.crop_and_zoom(images, camera_ids, bbox, zoom_factor)

        return images
