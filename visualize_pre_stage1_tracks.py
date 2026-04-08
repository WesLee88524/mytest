"""
可视化阶段一之前的跟踪结果（原始 MOT 结果）：

用途：
  - 在进入阶段一（VLM 巡检）前，把跟踪框 + ID 叠加到原始帧上导出，
    方便人工核对“阶段一判断是否合理”。
  - 支持单结果可视化，也支持两个结果对比可视化（可选）。

示例：
python visualize_pre_stage1_tracks.py \
  --seq dancetrack0003 \
  --tracks /path/to/dancetrack0003_pred1_base.txt \
  --image-dir /path/to/DanceTrack/test/dancetrack0003/img1 \
  --output-dir /path/to/viz/dancetrack0003/raw \
  --start-frame 1 \
  --end-frame 330

python visualize_pre_stage1_tracks.py \
  --seq dancetrack0003 \
  --tracks /path/to/base.txt \
  --compare-tracks /path/to/ours.txt \
  --image-dir /path/to/DanceTrack/test/dancetrack0003/img1 \
  --output-dir /path/to/viz/dancetrack0003/base_only \
  --output-compare-dir /path/to/viz/dancetrack0003/base_plus_compare
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def id_to_color(obj_id: int) -> Tuple[int, int, int]:
    np.random.seed(int(obj_id))
    c = np.random.randint(0, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def infer_end_frame(tracks_path: str, fallback: int) -> int:
    try:
        with open(tracks_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return fallback
        last = lines[-1].split(",")
        return int(float(last[0])) if last else fallback
    except Exception:
        return fallback


def _parse_box(fields: List[str], box_format: str) -> Tuple[float, float, float, float]:
    a, b, c, d = map(float, fields[2:6])
    if box_format == "xyxy":
        x1, y1, x2, y2 = a, b, c, d
    else:  # xywh
        x1, y1 = a, b
        x2, y2 = a + c, b + d
    return x1, y1, x2, y2


def read_tracks(
    txt_path: str,
    start_frame: int,
    end_frame: int,
    box_format: str = "xyxy",
) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    tracks: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")
            if len(fields) < 6:
                continue
            frame_id = int(float(fields[0]))
            obj_id = int(float(fields[1]))
            if not (start_frame <= frame_id <= end_frame):
                continue
            x1, y1, x2, y2 = _parse_box(fields, box_format=box_format)
            tracks.setdefault(frame_id, []).append((obj_id, x1, y1, x2, y2))
    return tracks


def _resolve_frame_path(image_dir: str, frame_id: int, frame_digits: int) -> str:
    return os.path.join(image_dir, f"{frame_id:0{frame_digits}d}.jpg")


def draw_tracks_on_image(
    image: np.ndarray,
    tracks: List[Tuple[int, float, float, float, float]],
    font_scale: float = 0.8,
    thickness: int = 2,
    id_prefix: str = "ID",
    color_bias: int = 0,
) -> np.ndarray:
    out = image.copy()
    for obj_id, x1, y1, x2, y2 in tracks:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        color = id_to_color(obj_id + color_bias)
        cv2.rectangle(out, p1, p2, color, thickness)
        cv2.putText(
            out,
            f"{id_prefix}:{obj_id}",
            (int(x1), max(12, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="可视化阶段一前的跟踪结果（支持可选对比）")
    parser.add_argument("--seq", type=str, default="", help="序列名，仅用于日志")
    parser.add_argument("--tracks", type=str, required=True, help="主跟踪结果 txt")
    parser.add_argument("--compare-tracks", type=str, default=None, help="可选：对比跟踪结果 txt")
    parser.add_argument("--image-dir", type=str, required=True, help="帧目录（img1）")
    parser.add_argument("--output-dir", type=str, required=True, help="主结果可视化输出目录")
    parser.add_argument("--output-compare-dir", type=str, default=None, help="主+对比叠加输出目录")
    parser.add_argument("--start-frame", type=int, default=1, help="起始帧")
    parser.add_argument("--end-frame", type=int, default=-1, help="结束帧；-1 表示自动推断")
    parser.add_argument("--frame-digits", type=int, default=8, choices=[6, 8], help="帧文件位数（DanceTrack 常用 8）")
    parser.add_argument("--box-format", type=str, default="xyxy", choices=["xyxy", "xywh"], help="txt 中框格式")
    parser.add_argument("--font-scale", type=float, default=0.8, help="ID 字体缩放")
    parser.add_argument("--thickness", type=int, default=2, help="框线与字体粗细")
    args = parser.parse_args()

    seq_name = args.seq or Path(args.image_dir).parent.name
    out_main = Path(args.output_dir)
    out_main.mkdir(parents=True, exist_ok=True)

    if args.compare_tracks and args.output_compare_dir:
        out_compare = Path(args.output_compare_dir)
        out_compare.mkdir(parents=True, exist_ok=True)
    elif args.compare_tracks:
        out_compare = out_main.parent / f"{out_main.name}_compare"
        out_compare.mkdir(parents=True, exist_ok=True)
    else:
        out_compare = None

    end_frame = args.end_frame
    if end_frame < 0:
        end_frame = infer_end_frame(args.tracks, fallback=args.start_frame)

    main_tracks = read_tracks(
        args.tracks,
        start_frame=args.start_frame,
        end_frame=end_frame,
        box_format=args.box_format,
    )
    cmp_tracks = read_tracks(
        args.compare_tracks,
        start_frame=args.start_frame,
        end_frame=end_frame,
        box_format=args.box_format,
    ) if args.compare_tracks else {}

    print(f"[{seq_name}] 可视化帧范围: {args.start_frame}-{end_frame}")
    for frame_id in range(args.start_frame, end_frame + 1):
        img_path = _resolve_frame_path(args.image_dir, frame_id, args.frame_digits)
        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue
        image = cv2.imread(img_path)
        if image is None:
            print(f"Read failed: {img_path}")
            continue

        vis_main = draw_tracks_on_image(
            image,
            main_tracks.get(frame_id, []),
            font_scale=args.font_scale,
            thickness=args.thickness,
            id_prefix="ID",
            color_bias=3,
        )
        cv2.imwrite(str(out_main / f"{frame_id:06d}.jpg"), vis_main)

        if out_compare is not None:
            vis_cmp = draw_tracks_on_image(
                vis_main,
                cmp_tracks.get(frame_id, []),
                font_scale=args.font_scale,
                thickness=args.thickness,
                id_prefix="CMP",
                color_bias=11,
            )
            cv2.imwrite(str(out_compare / f"{frame_id:06d}.jpg"), vis_cmp)

    print(f"[{seq_name}] 主可视化目录: {out_main}")
    if out_compare is not None:
        print(f"[{seq_name}] 对比可视化目录: {out_compare}")


if __name__ == "__main__":
    main()
