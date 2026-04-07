"""
入口脚本：运行阶段一 VLM 轨迹巡检器

使用方式：
  python run_inspector.py \
      --frames   /path/to/frames/ \
      --mot      /path/to/mot_result.txt \
      --output   suspicions.json \
      --backend  ollama \
      --model    qwen2-vl \
      --threshold 0.5

MOT 结果文件格式（MOTChallenge 标准）：
  frame_id, track_id, x, y, w, h, conf, -1, -1, -1
"""
import argparse
import cv2
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

from data_types import Box, Track
from track_preprocessor import load_frames
from vlm_inspector import QwenVLBackend, TrackInspector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# MOT 结果加载
# ──────────────────────────────────────────────

def load_mot_result(mot_path: str) -> List[Track]:
    """
    加载 MOTChallenge 格式的跟踪结果文件。
    每行：frame_id, track_id, x, y, w, h, conf, ...
    """
    from collections import defaultdict

    track_boxes: dict = defaultdict(list)

    with open(mot_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x   = float(parts[2])
            y   = float(parts[3])
            w   = float(parts[4])
            h   = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            track_boxes[track_id].append(Box(frame_id, x, y, w, h, conf))

    tracks = []
    for tid, boxes in track_boxes.items():
        boxes.sort(key=lambda b: b.frame_id)
        tracks.append(Track(track_id=tid, boxes=boxes))

    tracks.sort(key=lambda t: t.track_id)
    logger.info(f"加载轨迹 {len(tracks)} 条，来自 {mot_path}")
    return tracks


# ──────────────────────────────────────────────
# Demo：用合成数据测试（无需真实视频）
# ──────────────────────────────────────────────

def generate_demo_data(n_tracks: int = 5, n_frames: int = 60):
    """生成合成帧和轨迹，用于快速验证流程"""
    import random

    frames = {}
    for fid in range(1, n_frames + 1):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        # 随机背景噪点
        noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        frames[fid] = img

    tracks = []
    for tid in range(1, n_tracks + 1):
        boxes = []
        x, y = random.randint(50, 400), random.randint(50, 300)
        vx, vy = random.choice([-2, 2, 3, -3]), random.choice([-1, 1])
        for fid in range(1, n_frames + 1):
            # 模拟一次位移突变（制造 drift 错误）
            if fid == 30 and tid == 2:
                x += 80
            x = max(10, min(580, x + vx + random.randint(-1, 1)))
            y = max(10, min(420, y + vy + random.randint(-1, 1)))

            # 在帧上绘制目标
            color = [(200, 80, 80), (80, 200, 80), (80, 80, 200),
                     (200, 200, 80), (200, 80, 200)][tid % 5]
            cv2.rectangle(frames[fid], (int(x), int(y)), (int(x + 50), int(y + 100)), color, -1)
            cv2.putText(frames[fid], str(tid), (int(x + 5), int(y + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 模拟漏检（轨迹 3 在帧 20-30 中断）
            if tid == 3 and 20 <= fid <= 30:
                continue

            boxes.append(Box(fid, float(x), float(y), 50.0, 100.0, 0.9))

        if boxes:
            tracks.append(Track(track_id=tid, boxes=boxes))

    return frames, tracks


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MOT 轨迹巡检器（阶段一）")
    parser.add_argument("--frames",    type=str, default=None,     help="帧图像目录")
    parser.add_argument("--mot",       type=str, default=None,     help="MOT 结果文件（MOTChallenge 格式）")
    parser.add_argument("--output",    type=str, default="suspicions.json", help="输出 JSON 路径")
    parser.add_argument("--backend",   type=str, default="ollama", choices=["ollama", "hf"], help="推理后端")
    parser.add_argument("--model",     type=str, default="qwen3-vl", help="模型名称")
    parser.add_argument("--ollama-url",type=str, default="http://localhost:11434")
    parser.add_argument("--threshold", type=float, default=0.5,   help="可疑置信度阈值")
    parser.add_argument("--demo",      action="store_true",        help="使用合成数据演示（无需真实视频）")
    args = parser.parse_args()

    # ── 数据加载 ──
    if args.demo:
        logger.info("Demo 模式：生成合成数据...")
        import cv2  # noqa
        frames, tracks = generate_demo_data()
    else:
        if not args.frames or not args.mot:
            parser.error("非 demo 模式必须指定 --frames 和 --mot")
        frames = load_frames(args.frames)
        tracks = load_mot_result(args.mot)

    # ── 后端初始化 ──
    if args.backend == "ollama":
        backend = QwenVLBackend(
            model_name=args.model,
            use_ollama=True,
            ollama_url=args.ollama_url,
        )
    else:
        backend = QwenVLBackend(model_name=args.model, use_ollama=False)

    # ── 运行巡检 ──
    inspector = TrackInspector(backend, suspicion_threshold=args.threshold)
    report = inspector.inspect(tracks, frames)

    # ── 输出结果 ──
    print("\n" + "=" * 50)
    print(report.summary())
    print("=" * 50 + "\n")

    output_data = {
        "total_tracks": report.total_tracks,
        "suspicious_count": report.suspicious_count,
        "ok_count": len(report.ok_tracks),
        "suspicious": [
            {
                "track_id":        s.track_id,
                "anomaly_type":    s.anomaly_type.value,
                "frame_range":     list(s.frame_range),
                "confidence":      s.confidence,
                "reason":          s.reason,
                "related_track_id": s.related_track_id,
            }
            for s in report.suspicious
        ],
        "ok_tracks": report.ok_tracks,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存至 {args.output}")


if __name__ == "__main__":
    main()