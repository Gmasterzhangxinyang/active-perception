#!/usr/bin/env python3
"""
Agent Inference Entry - Agent推理入口

对比实验设计：
同一输入 → 原始BEVFusion → BEV_A → 与GT计算IoU_A
同一输入 → BEVFusion + Agent → BEV_B → 与GT计算IoU_B
                                 ↓
                         IoU_B - IoU_A = 提升效果
"""

import argparse
import sys
import os
import random
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BEVConfig
from models.bevfusion import BEVFusion
from data.nuscenes_loader import NuScenesLoader
from data.bev_gt import generate_bev_gt
from agent.core import AgentCore
from agent.bev_evaluator import BEVEvaluator
from agent.data_logger import DataLogger


def load_env_file(path=".env"):
    """Load simple KEY=VALUE pairs from a local env file if it exists."""
    if not path or not os.path.exists(path):
        return

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def compact_eval(eval_result):
    """Keep JSONL logs small and serializable."""
    if not eval_result:
        return {}
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


def get_scene_sample_indices(loader, scene_name):
    """Return dataset sample indices for one nuScenes scene name."""
    matching = [scene for scene in loader.nusc.scene if scene["name"] == scene_name]
    if not matching:
        available = ", ".join(scene["name"] for scene in loader.nusc.scene)
        raise ValueError(f"Scene {scene_name!r} not found. Available scenes: {available}")

    scene = matching[0]
    idx_by_token = {sample["token"]: idx for idx, sample in enumerate(loader.nusc.sample)}
    indices = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        indices.append(idx_by_token[sample_token])
        sample = loader.nusc.get("sample", sample_token)
        sample_token = sample["next"]

    return scene, indices


def run_agent_inference(args):
    """运行Agent推理"""
    load_env_file(args.env_file)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not args.fast_mode and not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env or run "
            "`export OPENAI_API_KEY=...` before using GPT mode. "
            "Use --fast_mode to skip GPT calls."
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = BEVConfig(device=device)

    print("=" * 60)
    print("BEVFusion + Active Agent Inference")
    print("=" * 60)
    print(f"Device: {device}")

    # 加载数据
    print("\nLoading nuScenes...")
    loader = NuScenesLoader(args.dataroot, args.version, cfg)
    print(f"Total samples: {len(loader)}")

    # 加载模型
    print("\nBuilding model...")
    model = BEVFusion(cfg)
    model = model.to(device)
    model.eval()

    # 加载训练好的权重
    model_path = "best_model.pth"  # 用户提供的新权重
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: No trained weights found, using random initialization")

    # BEV配置 - 传给Agent用于几何映射
    bev_cfg = {
        "bev_x_range": cfg.bev_x_range,
        "bev_y_range": cfg.bev_y_range,
        "bev_size": cfg.bev_size,
        "image_size": cfg.image_size,  # (H, W) tuple
    }

    # 初始化
    agent = AgentCore(
        model_name=args.gpt_model,
        max_iterations=args.max_iterations,
        fast_mode=args.fast_mode,
        api_key=openai_api_key,
        gt_debug=args.gt_debug,
        ablation=args.ablation,
        min_iou_delta=args.min_iou_delta,
    )
    evaluator = BEVEvaluator()
    data_logger = DataLogger(args.log_file)

    # 选择样本
    if args.sample is not None:
        sample_indices = [args.sample]
    elif args.scene_name is not None:
        scene, sample_indices = get_scene_sample_indices(loader, args.scene_name)
        if args.scene_stride > 1:
            sample_indices = sample_indices[::args.scene_stride]
        if args.scene_max_samples is not None:
            sample_indices = sample_indices[:args.scene_max_samples]
        print(
            f"Selected scene: {scene['name']} | "
            f"frames={len(sample_indices)}/{scene['nbr_samples']} | "
            f"description={scene.get('description', '')}"
        )
    elif args.random_samples is not None:
        rng = random.Random(args.seed)
        n_samples = min(args.random_samples, len(loader))
        sample_indices = rng.sample(range(len(loader)), n_samples)
    else:
        sample_indices = list(range(0, min(args.num_samples, len(loader))))
    print(f"Selected sample indices: {sample_indices}")

    results = []

    for idx in sample_indices:
        print(f"\n{'='*60}")
        print(f"Processing sample {idx}...")
        print("=" * 60)

        sample = loader[idx]
        raw_sample = loader.samples[idx]  # 原始nuScenes sample dict

        # 准备输入
        images = sample["images"].to(device)
        intrinsics = sample["intrinsics"].to(device)
        extrinsics = sample["extrinsics"].to(device)
        lidar_points = sample["lidar_points"].to(device)
        lidar_mask = sample["lidar_mask"].to(device)

        # 生成GT BEV (需要原始sample dict)
        gt_bev = generate_bev_gt(loader.nusc, raw_sample, cfg)
        gt_bev_tensor = gt_bev.long().to(device)

        # Baseline: 原始BEVFusion
        with torch.no_grad():
            logits_baseline, bev_baseline = model(images, intrinsics, extrinsics, lidar_points, lidar_mask)
            bev_baseline = bev_baseline[0] if bev_baseline.dim() > 2 else bev_baseline

        # 评估Baseline
        baseline_eval = evaluator.evaluate_with_gt(bev_baseline, gt_bev_tensor)
        print(f"\nBaseline (无Agent):")
        print(f"  IoU: {baseline_eval['iou']:.3f}")
        print(f"  Accuracy: {baseline_eval['accuracy']:.3f}")

        # Agent优化后
        with torch.no_grad():
            agent_result = agent.run(
                model, images, intrinsics, extrinsics,
                lidar_points, lidar_mask, bev_cfg,
                gt_bev=gt_bev_tensor if args.gt_debug else None
            )

        bev_agent = agent_result["final_bev"]
        if bev_agent.dim() > 2:
            bev_agent = bev_agent[0]

        # 评估Agent结果
        agent_eval = evaluator.evaluate_with_gt(bev_agent, gt_bev_tensor)
        print(f"\nAgent优化后:")
        print(f"  IoU: {agent_eval['iou']:.3f}")
        print(f"  Accuracy: {agent_eval['accuracy']:.3f}")
        print(f"  Iterations: {len([h for h in agent_result['history'] if 'decision' in h])}")

        # 计算提升
        iou_improvement = agent_eval['iou'] - baseline_eval['iou']
        acc_improvement = agent_eval['accuracy'] - baseline_eval['accuracy']

        print(f"\n提升效果:")
        print(f"  IoU: {baseline_eval['iou']:.3f} → {agent_eval['iou']:.3f} ({iou_improvement:+.3f})")
        print(f"  Accuracy: {baseline_eval['accuracy']:.3f} → {agent_eval['accuracy']:.3f} ({acc_improvement:+.3f})")

        # 打印决策历史（含相机映射信息）
        for h in agent_result['history']:
            if 'decision' in h:
                print(f"\n  Iteration {h['iteration']}:")
                print(f"    Thought: {h['decision'].get('thought', 'N/A')}")
                action = h['decision'].get('action', {})
                print(f"    Action: {action.get('name', 'N/A')}")
                print(f"    Camera IDs: {action.get('parameters', {}).get('camera_ids', 'all')}")
                eval_before = h.get("eval_before", {})
                if eval_before:
                    print(
                        "    Before: "
                        f"score={eval_before.get('score', 0):.4f}, "
                        f"edge={eval_before.get('edge_density', 0):.4f}, "
                        f"integrity={eval_before.get('integrity', 0):.4f}"
                    )

                vision_analysis = h.get("vision_analysis", [])
                for analysis in vision_analysis[:3]:
                    print(
                        "    Vision: "
                        f"{analysis.get('camera_name', analysis.get('camera_id'))} "
                        f"conditions={analysis.get('conditions', [])} "
                        f"analysis={analysis.get('analysis', '')[:80]}"
                    )

                eval_after = next(
                    (
                        item for item in agent_result["history"]
                        if item.get("iteration") == h["iteration"]
                        and "eval" in item
                        and "accepted" in item
                    ),
                    None
                )
                if eval_after:
                    after = eval_after["eval"]
                    print(
                        "    After:  "
                        f"score={after.get('score', 0):.4f}, "
                        f"edge={after.get('edge_density', 0):.4f}, "
                        f"integrity={after.get('integrity', 0):.4f}, "
                        f"accepted={eval_after.get('accepted')}, "
                        f"reason={eval_after.get('accept_reason')}"
                    )
                    selected_action = eval_after.get("selected_action")
                    if selected_action:
                        selected_params = selected_action.get("parameters", {})
                        print(
                            "    Selected: "
                            f"{selected_action.get('name')} "
                            f"params={selected_params}"
                        )

                    candidate_results = eval_after.get("candidate_results", [])
                    for rank, candidate in enumerate(candidate_results[:5], start=1):
                        action_info = candidate.get("action", {})
                        candidate_eval = candidate.get("eval", {})
                        candidate_gt = candidate.get("gt_eval")
                        metric_text = (
                            f"score_delta={candidate.get('score_delta', 0):+.4f}, "
                            f"score={candidate_eval.get('score', 0):.4f}"
                        )
                        if candidate_gt:
                            metric_text += (
                                f", iou={candidate_gt.get('iou', 0):.4f}, "
                                f"iou_delta={candidate.get('iou_delta', 0):+.4f}"
                            )
                        print(
                            f"    Candidate {rank}: "
                            f"{action_info.get('name')} "
                            f"{action_info.get('parameters', {})} | {metric_text}"
                        )

                # 记录数据
                data_logger.log(
                    iteration=h['iteration'],
                    input_state={"sample_idx": idx},
                    bev_quality={
                        "eval_before": compact_eval(eval_before),
                        "eval_after": compact_eval(eval_after.get("eval", {}) if eval_after else {}),
                        "iou": agent_eval['iou'],
                        "accuracy": agent_eval['accuracy']
                    },
                    agent_output=h['decision'],
                    result={
                        "iou_improvement": iou_improvement,
                        "improved": iou_improvement > 0,
                        "accepted": eval_after.get("accepted") if eval_after else None,
                        "accept_reason": eval_after.get("accept_reason") if eval_after else None,
                        "selected_action": eval_after.get("selected_action") if eval_after else None,
                        "candidate_results": eval_after.get("candidate_results", []) if eval_after else [],
                    }
                )

        results.append({
            "sample_idx": idx,
            "baseline_iou": baseline_eval['iou'],
            "agent_iou": agent_eval['iou'],
            "iou_improvement": iou_improvement,
            "baseline_acc": baseline_eval['accuracy'],
            "agent_acc": agent_eval['accuracy'],
            "acc_improvement": acc_improvement,
            "iterations": len([h for h in agent_result['history'] if 'decision' in h])
        })

    # 汇总统计
    print(f"\n{'='*60}")
    print("汇总统计")
    print("=" * 60)

    total_iou_improvement = sum(r['iou_improvement'] for r in results)
    avg_iou_improvement = total_iou_improvement / len(results) if results else 0

    improved_count = sum(1 for r in results if r['iou_improvement'] > 0)
    degraded_count = sum(1 for r in results if r['iou_improvement'] < 0)

    print(f"样本数: {len(results)}")
    print(f"IoU提升: 平均{avg_iou_improvement:+.3f}")
    print(f"提升样本: {improved_count}/{len(results)}")
    print(f"下降样本: {degraded_count}/{len(results)}")

    # 数据分析
    print(f"\n{'='*60}")
    print("数据分析")
    print("=" * 60)
    stats = data_logger.analyze()
    if stats:
        for action_name, stat in stats.items():
            print(f"  {action_name}: {stat['count']}次, 成功率{stat.get('success_rate', 0):.1%}")
    else:
        print("  暂无数据记录")


def main():
    parser = argparse.ArgumentParser(description="BEVFusion + Active Agent Inference")
    parser.add_argument("--dataroot", type=str, default="./data/sets/nuscenes", help="Path to nuScenes")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="Dataset version")
    parser.add_argument("--sample", type=int, default=None, help="Sample index")
    parser.add_argument("--scene_name", type=str, default=None, help="nuScenes scene name to run, e.g. scene-1094")
    parser.add_argument("--scene_stride", type=int, default=1, help="Use every Nth sample when running --scene_name")
    parser.add_argument("--scene_max_samples", type=int, default=None, help="Limit number of samples when running --scene_name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--random_samples", type=int, default=None, help="Randomly sample this many samples instead of taking the first N")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --random_samples")
    parser.add_argument("--max_iterations", type=int, default=3, help="Max agent iterations")
    parser.add_argument("--gpt_model", type=str, default="gpt-5.4-mini", help="OpenAI GPT model for vision analysis")
    parser.add_argument("--env_file", type=str, default=".env", help="Optional env file containing OPENAI_API_KEY")
    parser.add_argument("--fast_mode", action="store_true", help="Skip GPT vision calls and use rule-based fallback decisions")
    parser.add_argument("--gt_debug", action="store_true", help="Debug only: use GT IoU to accept/reject agent actions")
    parser.add_argument("--ablation", action="store_true", help="Evaluate multiple candidate action variants and choose the best one")
    parser.add_argument("--min_iou_delta", type=float, default=1e-6, help="Minimum IoU gain required to accept an action in --gt_debug")
    parser.add_argument("--log_file", type=str, default="agent_training_data.jsonl", help="Log file path")
    args = parser.parse_args()

    run_agent_inference(args)


if __name__ == "__main__":
    main()
