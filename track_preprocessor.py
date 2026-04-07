"""
轨迹预处理：
  1. 从帧序列中裁剪目标图像块
  2. 将关键帧拼成横向拼贴图（contact sheet）发给 VLM
  3. 计算数值特征（位移、IoU、长度）辅助 VLM 判断
"""
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from data_types import Box, Track


# ──────────────────────────────────────────────
# 1. 帧图像加载
# ──────────────────────────────────────────────

def load_frames(frame_dir: str) -> Dict[int, np.ndarray]:
    """从目录加载所有帧，文件名格式：000001.jpg / frame_000001.png 等"""
    frame_dir = Path(frame_dir)
    frames: Dict[int, np.ndarray] = {}
    for p in sorted(frame_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        # 提取数字部分作为帧 ID
        digits = "".join(filter(str.isdigit, p.stem))
        if not digits:
            continue
        frame_id = int(digits)
        img = cv2.imread(str(p))
        if img is not None:
            frames[frame_id] = img
    return frames


# ──────────────────────────────────────────────
# 2. 关键帧采样
# ──────────────────────────────────────────────

def sample_key_frames(track: Track, max_frames: int = 12) -> List[Box]:
    """
    均匀采样关键帧，避免拼贴图过大。
    优先保留首帧、尾帧和运动突变帧。
    """
    boxes = track.boxes
    if len(boxes) <= max_frames:
        return boxes

    # 均匀步进采样
    indices = set(np.linspace(0, len(boxes) - 1, max_frames, dtype=int).tolist())
    # 强制包含首尾
    indices.add(0)
    indices.add(len(boxes) - 1)
    return [boxes[i] for i in sorted(indices)]


# ──────────────────────────────────────────────
# 3. 单帧目标裁剪
# ──────────────────────────────────────────────

def crop_target(
    frame: np.ndarray,
    box: Box,
    padding: float = 0.25,
    target_size: Tuple[int, int] = (128, 192),  # (W, H)
) -> np.ndarray:
    """裁剪目标区域，加一点 padding 以保留上下文"""
    h_img, w_img = frame.shape[:2]
    px = box.w * padding
    py = box.h * padding
    x1 = max(0, int(box.x - px))
    y1 = max(0, int(box.y - py))
    x2 = min(w_img, int(box.x + box.w + px))
    y2 = min(h_img, int(box.y + box.h + py))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    return cv2.resize(crop, target_size)


# ──────────────────────────────────────────────
# 4. 拼贴图生成
# ──────────────────────────────────────────────

def build_contact_sheet(
    track: Track,
    frames: Dict[int, np.ndarray],
    max_frames: int = 12,
    thumb_size: Tuple[int, int] = (128, 192),
    gap: int = 4,
) -> np.ndarray:
    """
    生成横向拼贴图：
    [帧号标签]
    [目标裁剪图] × N 列
    """
    key_boxes = sample_key_frames(track, max_frames)
    thumbs = []
    labels = []

    for box in key_boxes:
        frame = frames.get(box.frame_id)
        if frame is None:
            thumb = np.full((thumb_size[1], thumb_size[0], 3), 128, dtype=np.uint8)
        else:
            thumb = crop_target(frame, box, target_size=thumb_size)

        # 在裁剪图上画出 bbox（红色）
        pw = thumb_size[0]
        ph = thumb_size[1]
        # 相对坐标
        scale_x = pw / (box.w * 1.5 + 1e-6)
        scale_y = ph / (box.h * 1.5 + 1e-6)
        bx1 = int(pw * 0.17)
        by1 = int(ph * 0.17)
        bx2 = int(pw * 0.83)
        by2 = int(ph * 0.83)
        cv2.rectangle(thumb, (bx1, by1), (bx2, by2), (0, 0, 220), 2)

        thumbs.append(thumb)
        labels.append(f"f{box.frame_id}")

    # 拼接
    label_h = 20
    row_h = thumb_size[1] + label_h + gap
    total_w = (thumb_size[0] + gap) * len(thumbs) - gap

    sheet = np.full((row_h, total_w, 3), 240, dtype=np.uint8)

    for i, (thumb, label) in enumerate(zip(thumbs, labels)):
        x_off = i * (thumb_size[0] + gap)
        # 标签
        cv2.putText(
            sheet, label, (x_off + 2, label_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA,
        )
        # 图像
        sheet[label_h: label_h + thumb_size[1], x_off: x_off + thumb_size[0]] = thumb

    return sheet


# ──────────────────────────────────────────────
# 5. 数值特征计算
# ──────────────────────────────────────────────

def compute_numeric_features(track: Track) -> Dict:
    """
    计算轨迹数值特征，作为文本附加给 VLM 参考。
    返回：位移统计、IoU 变化、轨迹长度、中断帧段
    """
    boxes = track.boxes
    if len(boxes) < 2:
        return {"length": len(boxes), "note": "轨迹过短，无法计算特征"}

    # 逐帧位移（中心点欧氏距离）
    displacements = []
    for a, b in zip(boxes[:-1], boxes[1:]):
        d = math.hypot(b.cx - a.cx, b.cy - a.cy)
        displacements.append(d)

    disp_arr = np.array(displacements)
    mean_d = float(disp_arr.mean())
    std_d  = float(disp_arr.std())
    max_d  = float(disp_arr.max())
    max_d_frame = boxes[int(disp_arr.argmax())].frame_id

    # 找到位移突变帧（超过均值 + 3*std）
    threshold = mean_d + 3 * std_d
    jump_frames = [
        boxes[i].frame_id
        for i, d in enumerate(displacements)
        if d > threshold
    ]

    # 逐帧 IoU
    def iou(a: Box, b: Box) -> float:
        ax2, ay2 = a.x + a.w, a.y + a.h
        bx2, by2 = b.x + b.w, b.y + b.h
        ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    ious = [iou(a, b) for a, b in zip(boxes[:-1], boxes[1:])]
    mean_iou = float(np.mean(ious)) if ious else 0.0
    min_iou  = float(np.min(ious))  if ious else 0.0

    # 检测帧间隔（判断中断）
    frame_ids = [b.frame_id for b in boxes]
    gaps = [(frame_ids[i + 1] - frame_ids[i], frame_ids[i], frame_ids[i + 1])
            for i in range(len(frame_ids) - 1)]
    # 超过5帧的间隔认为是中断
    breaks = [(gap, f1, f2) for gap, f1, f2 in gaps if gap > 5]

    return {
        "track_id":       track.track_id,
        "length":         track.length,
        "start_frame":    track.start_frame,
        "end_frame":      track.end_frame,
        "mean_displacement": round(mean_d, 2),
        "std_displacement":  round(std_d, 2),
        "max_displacement":  round(max_d, 2),
        "max_disp_frame":    max_d_frame,
        "jump_frames":       jump_frames,
        "mean_iou":          round(mean_iou, 3),
        "min_iou":           round(min_iou, 3),
        "trajectory_breaks": [{"gap": g, "from": f1, "to": f2} for g, f1, f2 in breaks],
    }
