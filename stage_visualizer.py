"""
阶段可视化导出工具：
  - 阶段一：可疑轨迹片段
  - 阶段二：调查前后（初筛区间 vs 复核后区间）
  - 阶段三：修正前后
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from data_types import Box, Suspicion, Track
from llm_investigator import Investigation
from track_corrector import CorrectionAction


def _crop_with_padding(frame: np.ndarray, box: Box, padding: float = 0.30) -> np.ndarray:
    h, w = frame.shape[:2]
    px = box.w * padding
    py = box.h * padding
    x1 = max(0, int(box.x - px))
    y1 = max(0, int(box.y - py))
    x2 = min(w, int(box.x + box.w + px))
    y2 = min(h, int(box.y + box.h + py))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((180, 120, 3), dtype=np.uint8)
    return crop


def _ascii_text(s: str) -> str:
    # OpenCV 默认字体不支持中文，统一降级为可渲染 ASCII，避免“????”乱码
    return s.encode("ascii", errors="replace").decode("ascii")


def _sample_boxes_in_range(track: Track, start: int, end: int, max_frames: int = 10) -> List[Box]:
    boxes = [b for b in track.boxes if start <= b.frame_id <= end]
    if not boxes:
        return []
    if len(boxes) <= max_frames:
        return boxes
    idx = np.linspace(0, len(boxes) - 1, max_frames, dtype=int).tolist()
    return [boxes[i] for i in idx]


def _build_track_strip(
    track: Optional[Track],
    frames: Dict[int, np.ndarray],
    frame_range: Tuple[int, int],
    title: str,
    thumb_size: Tuple[int, int] = (120, 180),
    max_frames: int = 10,
) -> np.ndarray:
    start, end = frame_range
    label_h = 48
    gap = 4
    w_t, h_t = thumb_size

    if track is None:
        canvas = np.full((label_h + h_t, w_t, 3), 220, dtype=np.uint8)
        cv2.putText(canvas, _ascii_text(title), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, "track missing", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 180), 1, cv2.LINE_AA)
        return canvas

    sampled = _sample_boxes_in_range(track, start, end, max_frames=max_frames)
    if not sampled:
        canvas = np.full((label_h + h_t, w_t, 3), 230, dtype=np.uint8)
        cv2.putText(canvas, _ascii_text(title), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"no boxes in {start}-{end}", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
        return canvas

    total_w = (w_t + gap) * len(sampled) - gap
    canvas = np.full((label_h + h_t, total_w, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, _ascii_text(title), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"track={track.track_id} frames={start}-{end}", (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)

    for i, b in enumerate(sampled):
        x = i * (w_t + gap)
        frame = frames.get(b.frame_id)
        if frame is None:
            thumb = np.full((h_t, w_t, 3), 120, dtype=np.uint8)
        else:
            frame_marked = frame.copy()
            x1, y1 = int(max(0, b.x)), int(max(0, b.y))
            x2, y2 = int(min(frame.shape[1] - 1, b.x + b.w)), int(min(frame.shape[0] - 1, b.y + b.h))
            cv2.rectangle(frame_marked, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame_marked,
                f"id:{track.track_id}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            crop = _crop_with_padding(frame_marked, b, padding=0.30)
            thumb = cv2.resize(crop, (w_t, h_t))
            cv2.rectangle(thumb, (2, 2), (w_t - 3, h_t - 3), (0, 210, 255), 2)
        canvas[label_h:label_h + h_t, x:x + w_t] = thumb
        cv2.putText(canvas, f"f{b.frame_id}", (x + 4, label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def _stack_vertical(images: List[np.ndarray], bg: int = 255, gap: int = 6) -> np.ndarray:
    if not images:
        return np.full((120, 320, 3), bg, dtype=np.uint8)
    max_w = max(im.shape[1] for im in images)
    total_h = sum(im.shape[0] for im in images) + gap * (len(images) - 1)
    out = np.full((total_h, max_w, 3), bg, dtype=np.uint8)
    y = 0
    for im in images:
        out[y:y + im.shape[0], :im.shape[1]] = im
        y += im.shape[0] + gap
    return out


def _safe_text(s: str, max_len: int = 160) -> str:
    s = s.replace("\n", " ").strip()
    return s[:max_len]


def visualize_stage1(
    seq_name: str,
    suspicions: List[Suspicion],
    tracks: Dict[int, Track],
    frames: Dict[int, np.ndarray],
    out_dir: str,
) -> None:
    base = Path(out_dir) / seq_name / "stage1"
    base.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(suspicions, start=1):
        tr = tracks.get(s.track_id)
        title = f"[Stage1] #{i} {s.anomaly_type.value} conf={s.confidence:.2f} reason={_safe_text(s.reason, 60)}"
        strip = _build_track_strip(tr, frames, s.frame_range, title=title)
        out = base / f"{i:03d}_track{s.track_id}_{s.anomaly_type.value}_{s.frame_range[0]}_{s.frame_range[1]}.jpg"
        cv2.imwrite(str(out), strip)


def visualize_stage2(
    seq_name: str,
    investigations: List[Investigation],
    tracks: Dict[int, Track],
    frames: Dict[int, np.ndarray],
    out_dir: str,
) -> None:
    base = Path(out_dir) / seq_name / "stage2"
    base.mkdir(parents=True, exist_ok=True)
    for i, inv in enumerate(investigations, start=1):
        tr = tracks.get(inv.track_id)
        init_range = inv.initial_suspicion.frame_range
        final_range = inv.frame_range
        top = _build_track_strip(
            tr, frames, init_range,
            title=(
                f"[Stage2-Before] track={inv.track_id} "
                f"type={inv.initial_suspicion.anomaly_type.value} conf={inv.initial_suspicion.confidence:.2f}"
            ),
        )
        tools = " -> ".join(inv.tool_calls) if inv.tool_calls else "none"
        bottom = _build_track_strip(
            tr, frames, final_range,
            title=(
                f"[Stage2-After] confirmed={inv.confirmed} type={inv.anomaly_type.value} "
                f"conf={inv.confidence:.2f} tools={_safe_text(tools, 80)}"
            ),
        )
        panel = _stack_vertical([top, bottom], bg=250, gap=8)
        out = base / f"{i:03d}_track{inv.track_id}_{inv.anomaly_type.value}_{final_range[0]}_{final_range[1]}.jpg"
        cv2.imwrite(str(out), panel)


def visualize_stage3(
    seq_name: str,
    actions: List[CorrectionAction],
    tracks_before: Dict[int, Track],
    tracks_after: Dict[int, Track],
    frames: Dict[int, np.ndarray],
    out_dir: str,
) -> None:
    base = Path(out_dir) / seq_name / "stage3"
    base.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(actions, start=1):
        tr_before = tracks_before.get(a.track_id)
        tr_after = tracks_after.get(a.track_id)
        top = _build_track_strip(
            tr_before, frames, a.frame_range,
            title=(
                f"[Stage3-Before] track={a.track_id} type={a.anomaly_type.value} "
                f"conf={a.confidence:.2f}"
            ),
        )
        bottom = _build_track_strip(
            tr_after, frames, a.frame_range,
            title=(
                f"[Stage3-After] applied={a.applied} rollback={a.rolled_back} "
                f"verified={a.verified} detail={_safe_text(a.details, 80)}"
            ),
        )
        panel = _stack_vertical([top, bottom], bg=252, gap=8)
        out = base / (
            f"{i:03d}_track{a.track_id}_{a.anomaly_type.value}_"
            f"{a.frame_range[0]}_{a.frame_range[1]}.jpg"
        )
        cv2.imwrite(str(out), panel)
