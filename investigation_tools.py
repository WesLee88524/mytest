"""
阶段二工具集 —— LLM 可调用的视觉分析工具

每个工具函数：
  - 接收结构化参数
  - 返回 ToolResult（包含 base64 图像 和/或 文本描述）
  - 供 LLM 在 ReAct 循环中按需调用
"""
import base64
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from data_types import Box, Track

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 工具返回值
# ──────────────────────────────────────────────

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    text: str                          # 文本描述，始终存在
    images_b64: List[str] = field(default_factory=list)  # base64 编码图像列表
    metadata: Dict = field(default_factory=dict)

    def to_llm_text(self) -> str:
        """转为 LLM 可读的文本摘要（图像单独传入）"""
        lines = [f"[{self.tool_name}] {'成功' if self.success else '失败'}"]
        lines.append(self.text)
        if self.metadata:
            lines.append(f"附加数据：{json.dumps(self.metadata, ensure_ascii=False)}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 工具注册表
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "replay_segment",
        "description": (
            "获取指定帧范围内的目标图像序列，用于逐帧观察目标变化。"
            "返回多帧裁剪图拼成的横向拼贴图。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "track_id":    {"type": "integer", "description": "目标轨迹 ID"},
                "start_frame": {"type": "integer", "description": "起始帧号"},
                "end_frame":   {"type": "integer", "description": "结束帧号"},
                "step":        {"type": "integer", "description": "采样步长，默认1（每帧都取）", "default": 1},
            },
            "required": ["track_id", "start_frame", "end_frame"],
        },
    },
    {
        "name": "zoom_region",
        "description": (
            "裁剪并放大指定帧中目标所在区域，用于细节观察。"
            "可选超分辨率锐化，适合目标较小或图像模糊的情况。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "track_id":   {"type": "integer", "description": "目标轨迹 ID"},
                "frame_id":   {"type": "integer", "description": "要放大的帧号"},
                "scale":      {"type": "number",  "description": "放大倍数，默认3.0", "default": 3.0},
                "padding":    {"type": "number",  "description": "目标周围留白比例，默认0.5", "default": 0.5},
                "sharpen":    {"type": "boolean", "description": "是否做锐化增强，默认true", "default": True},
            },
            "required": ["track_id", "frame_id"],
        },
    },
    {
        "name": "enhance_image",
        "description": (
            "对指定帧做图像增强处理（去模糊/对比度拉伸/边缘强化），"
            "用于在低质量图像中找回目标细节。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "track_id": {"type": "integer", "description": "目标轨迹 ID"},
                "frame_id": {"type": "integer", "description": "要增强的帧号"},
                "method":   {
                    "type": "string",
                    "enum": ["contrast", "deblur", "edge", "clahe"],
                    "description": "增强方式：contrast=对比度拉伸, deblur=去模糊, edge=边缘强化, clahe=局部直方图均衡",
                    "default": "contrast",
                },
            },
            "required": ["track_id", "frame_id"],
        },
    },
    {
        "name": "compare_appearances",
        "description": (
            "对比同一轨迹两个不同时刻的目标外观，"
            "计算颜色直方图相似度和像素差异，判断是否发生外观突变（ID 切换）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "track_id":     {"type": "integer", "description": "目标轨迹 ID"},
                "frame_id_a":   {"type": "integer", "description": "参考帧号（突变前）"},
                "frame_id_b":   {"type": "integer", "description": "对比帧号（突变后）"},
            },
            "required": ["track_id", "frame_id_a", "frame_id_b"],
        },
    },
    {
        "name": "check_nearby_tracks",
        "description": (
            "在指定帧的指定位置附近，查找其他轨迹的目标，"
            "用于判断是否存在遮挡、ID 互换等情况。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "frame_id":    {"type": "integer", "description": "要查找的帧号"},
                "track_id":    {"type": "integer", "description": "当前可疑轨迹 ID"},
                "radius":      {"type": "number",  "description": "搜索半径（像素），默认150", "default": 150},
            },
            "required": ["frame_id", "track_id"],
        },
    },
]


# ──────────────────────────────────────────────
# 工具实现
# ──────────────────────────────────────────────

def _ndarray_to_b64(img: np.ndarray, fmt: str = ".jpg") -> str:
    ok, buf = cv2.imencode(fmt, img)
    if not ok or not hasattr(buf, "tobytes"):
        # 兜底：先缩放再用 PNG 编码，避免超大图导致 jpg 编码失败
        h, w = img.shape[:2]
        max_side = 4096
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        ok2, buf2 = cv2.imencode(".png", img)
        if not ok2 or not hasattr(buf2, "tobytes"):
            raise ValueError("图像编码失败")
        return base64.b64encode(buf2.tobytes()).decode("utf-8")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _get_box_for_frame(track: Track, frame_id: int) -> Optional[Box]:
    """取轨迹在指定帧的 box，若无则取最近帧"""
    exact = [b for b in track.boxes if b.frame_id == frame_id]
    if exact:
        return exact[0]
    # 取距离最近的帧
    if not track.boxes:
        return None
    return min(track.boxes, key=lambda b: abs(b.frame_id - frame_id))


def _crop_box(frame: np.ndarray, box: Box, padding: float = 0.3) -> np.ndarray:
    h, w = frame.shape[:2]
    px, py = box.w * padding, box.h * padding
    x1 = max(0, int(box.x - px))
    y1 = max(0, int(box.y - py))
    x2 = min(w, int(box.x + box.w + px))
    y2 = min(h, int(box.y + box.h + py))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else np.zeros((64, 64, 3), dtype=np.uint8)


# ── 工具1：回放片段 ───────────────────────────

def replay_segment(
    track: Track,
    frames: Dict[int, np.ndarray],
    start_frame: int,
    end_frame: int,
    step: int = 1,
) -> ToolResult:
    thumb_w, thumb_h = 120, 180
    gap = 4
    label_h = 18
    max_show = 32
    max_cols = 8

    selected_boxes = [
        b for b in track.boxes
        if start_frame <= b.frame_id <= end_frame
    ][::max(1, step)]

    if not selected_boxes:
        return ToolResult(
            tool_name="replay_segment",
            success=False,
            text=f"轨迹 {track.track_id} 在帧 {start_frame}-{end_frame} 内无数据",
        )

    thumbs = []
    for box in selected_boxes:
        frame = frames.get(box.frame_id)
        if frame is None:
            thumb = np.full((thumb_h, thumb_w, 3), 100, dtype=np.uint8)
        else:
            crop = _crop_box(frame, box, padding=0.3)
            thumb = cv2.resize(crop, (thumb_w, thumb_h))
            # 标注 bbox 位置
            cv2.rectangle(thumb,
                (int(thumb_w * 0.15), int(thumb_h * 0.15)),
                (int(thumb_w * 0.85), int(thumb_h * 0.85)),
                (0, 80, 220), 1)
        thumbs.append((box.frame_id, thumb))

    sampled = thumbs
    if len(sampled) > max_show:
        keep_idx = np.linspace(0, len(sampled) - 1, max_show, dtype=int).tolist()
        sampled = [sampled[i] for i in keep_idx]

    n = len(sampled)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    total_w = cols * thumb_w + (cols - 1) * gap
    total_h = rows * (label_h + thumb_h + gap) - gap
    sheet = np.full((total_h, total_w, 3), 235, dtype=np.uint8)

    for i, (fid, thumb) in enumerate(sampled):
        r = i // cols
        c = i % cols
        x = c * (thumb_w + gap)
        y = r * (label_h + thumb_h + gap)
        cv2.putText(sheet, f"f{fid}", (x + 2, y + label_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (50, 50, 50), 1)
        sheet[y + label_h:y + label_h + thumb_h, x:x + thumb_w] = thumb

    return ToolResult(
        tool_name="replay_segment",
        success=True,
        text=(
            f"轨迹 {track.track_id} 帧 {start_frame}-{end_frame}，"
            f"原始 {len(thumbs)} 帧，展示 {len(sampled)} 帧（步长 {step}）"
        ),
        images_b64=[_ndarray_to_b64(sheet)],
        metadata={"frame_count": len(sampled), "frame_ids": [t[0] for t in sampled]},
    )


# ── 工具2：区域放大 ───────────────────────────

def zoom_region(
    track: Track,
    frames: Dict[int, np.ndarray],
    frame_id: int,
    scale: float = 3.0,
    padding: float = 0.5,
    sharpen: bool = True,
) -> ToolResult:
    frame = frames.get(frame_id)
    box = _get_box_for_frame(track, frame_id)

    if frame is None or box is None:
        return ToolResult(
            tool_name="zoom_region",
            success=False,
            text=f"帧 {frame_id} 或轨迹 {track.track_id} 数据不存在",
        )

    crop = _crop_box(frame, box, padding=padding)
    h, w = crop.shape[:2]
    zoomed = cv2.resize(crop, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_CUBIC)

    if sharpen:
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        zoomed = cv2.filter2D(zoomed, -1, kernel)
        zoomed = np.clip(zoomed, 0, 255).astype(np.uint8)

    # 画出目标框
    zh, zw = zoomed.shape[:2]
    pad_x = int(zw * padding / (1 + 2 * padding))
    pad_y = int(zh * padding / (1 + 2 * padding))
    cv2.rectangle(zoomed, (pad_x, pad_y), (zw - pad_x, zh - pad_y), (0, 80, 220), 2)
    cv2.putText(zoomed, f"ID:{track.track_id} f{frame_id}",
                (pad_x + 4, pad_y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 80, 220), 1)

    return ToolResult(
        tool_name="zoom_region",
        success=True,
        text=f"轨迹 {track.track_id} 第 {frame_id} 帧放大 {scale}x，{'已锐化' if sharpen else '未锐化'}",
        images_b64=[_ndarray_to_b64(zoomed)],
        metadata={"scale": scale, "original_size": [w, h],
                  "zoomed_size": [zw, zh], "box": [box.x, box.y, box.w, box.h]},
    )


# ── 工具3：图像增强 ───────────────────────────

def enhance_image(
    track: Track,
    frames: Dict[int, np.ndarray],
    frame_id: int,
    method: str = "contrast",
) -> ToolResult:
    frame = frames.get(frame_id)
    box = _get_box_for_frame(track, frame_id)

    if frame is None or box is None:
        return ToolResult(
            tool_name="enhance_image",
            success=False,
            text=f"帧 {frame_id} 数据不存在",
        )

    crop = _crop_box(frame, box, padding=0.4)
    result = crop.copy()

    if method == "contrast":
        # 线性对比度拉伸
        result = cv2.convertScaleAbs(result, alpha=1.8, beta=20)

    elif method == "deblur":
        # 非锐化掩模模拟去模糊
        blur = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.8, blur, -0.8, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)

    elif method == "edge":
        # 边缘叠加增强
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(result, 0.8, edges_3ch, 0.5, 0)

    elif method == "clahe":
        # 局部直方图均衡（对低光照场景有效）
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 并排对比原图和增强图
    h = max(crop.shape[0], result.shape[0])
    crop_r = cv2.resize(crop, (int(crop.shape[1] * h / crop.shape[0]), h))
    result_r = cv2.resize(result, (int(result.shape[1] * h / result.shape[0]), h))
    divider = np.full((h, 4, 3), 180, dtype=np.uint8)
    comparison = np.hstack([crop_r, divider, result_r])
    cv2.putText(comparison, "原图", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,50,50), 1)
    cv2.putText(comparison, method, (crop_r.shape[1] + 8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,50,50), 1)

    return ToolResult(
        tool_name="enhance_image",
        success=True,
        text=f"轨迹 {track.track_id} 第 {frame_id} 帧增强方式：{method}，左为原图右为增强结果",
        images_b64=[_ndarray_to_b64(comparison)],
        metadata={"method": method, "frame_id": frame_id},
    )


# ── 工具4：外观对比 ───────────────────────────

def compare_appearances(
    track: Track,
    frames: Dict[int, np.ndarray],
    frame_id_a: int,
    frame_id_b: int,
) -> ToolResult:
    frame_a = frames.get(frame_id_a)
    frame_b = frames.get(frame_id_b)
    box_a = _get_box_for_frame(track, frame_id_a)
    box_b = _get_box_for_frame(track, frame_id_b)

    if frame_a is None or frame_b is None or box_a is None or box_b is None:
        return ToolResult(
            tool_name="compare_appearances",
            success=False,
            text="所需帧数据不存在",
        )

    crop_a = _crop_box(frame_a, box_a, padding=0.2)
    crop_b = _crop_box(frame_b, box_b, padding=0.2)

    # 统一大小用于对比
    size = (128, 192)
    img_a = cv2.resize(crop_a, size)
    img_b = cv2.resize(crop_b, size)

    # 颜色直方图相似度（HSV 空间）
    def hist_sim(i1, i2):
        h1 = cv2.calcHist([cv2.cvtColor(i1, cv2.COLOR_BGR2HSV)],
                          [0, 1], None, [18, 16], [0, 180, 0, 256])
        h2 = cv2.calcHist([cv2.cvtColor(i2, cv2.COLOR_BGR2HSV)],
                          [0, 1], None, [18, 16], [0, 180, 0, 256])
        cv2.normalize(h1, h1); cv2.normalize(h2, h2)
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

    sim = hist_sim(img_a, img_b)

    # 像素差异图
    diff = cv2.absdiff(img_a, img_b)
    diff_enhanced = cv2.convertScaleAbs(diff, alpha=3.0)

    # 拼接：原A | 原B | 差异图
    gap = np.full((size[1], 4, 3), 180, dtype=np.uint8)
    panel = np.hstack([img_a, gap, img_b, gap, diff_enhanced])

    # 标注
    for x, label in [(4, f"f{frame_id_a}"),
                     (size[0] + 8, f"f{frame_id_b}"),
                     (size[0] * 2 + 12, "差异")]:
        cv2.putText(panel, label, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30,30,30), 1)

    # 相似度颜色标注
    sim_color = (0, 180, 0) if sim > 0.7 else (0, 100, 220) if sim > 0.4 else (0, 0, 200)
    sim_text = f"颜色相似度: {sim:.3f} ({'正常' if sim > 0.7 else '可疑' if sim > 0.4 else '异常'})"
    cv2.putText(panel, sim_text, (4, size[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, sim_color, 1)

    judgment = (
        "外观基本一致，非 ID 切换" if sim > 0.7
        else "外观有差异，可能存在遮挡或光照变化" if sim > 0.4
        else "外观差异显著，高度怀疑 ID 切换"
    )

    return ToolResult(
        tool_name="compare_appearances",
        success=True,
        text=f"轨迹 {track.track_id}：帧{frame_id_a} vs 帧{frame_id_b}，{judgment}",
        images_b64=[_ndarray_to_b64(panel)],
        metadata={"color_similarity": round(sim, 4), "judgment": judgment},
    )


# ── 工具5：附近轨迹查找 ──────────────────────

def check_nearby_tracks(
    all_tracks: Dict[int, Track],
    frames: Dict[int, np.ndarray],
    frame_id: int,
    track_id: int,
    radius: float = 150,
) -> ToolResult:
    target_track = all_tracks.get(track_id)
    if target_track is None:
        return ToolResult(
            tool_name="check_nearby_tracks",
            success=False,
            text=f"轨迹 {track_id} 不存在",
        )

    target_box = _get_box_for_frame(target_track, frame_id)
    if target_box is None:
        return ToolResult(
            tool_name="check_nearby_tracks",
            success=False,
            text=f"轨迹 {track_id} 在帧 {frame_id} 无数据",
        )

    frame = frames.get(frame_id)
    if frame is None:
        return ToolResult(
            tool_name="check_nearby_tracks",
            success=False,
            text=f"帧 {frame_id} 图像不存在",
        )

    vis = frame.copy()
    nearby = []

    # 画出当前目标（蓝色）
    cv2.rectangle(vis,
        (int(target_box.x), int(target_box.y)),
        (int(target_box.x + target_box.w), int(target_box.y + target_box.h)),
        (220, 80, 0), 2)
    cv2.putText(vis, f"ID:{track_id}", (int(target_box.x), int(target_box.y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 80, 0), 1)

    # 搜索附近其他轨迹（绿色）
    for tid, track in all_tracks.items():
        if tid == track_id:
            continue
        box = _get_box_for_frame(track, frame_id)
        if box is None:
            continue
        dist = math.hypot(box.cx - target_box.cx, box.cy - target_box.cy)
        if dist <= radius:
            nearby.append({"track_id": tid, "distance": round(dist, 1),
                           "box": [box.x, box.y, box.w, box.h]})
            cv2.rectangle(vis,
                (int(box.x), int(box.y)),
                (int(box.x + box.w), int(box.y + box.h)),
                (0, 180, 0), 2)
            cv2.putText(vis, f"ID:{tid} d={dist:.0f}",
                        (int(box.x), int(box.y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 0), 1)

    # 画搜索半径圆
    cv2.circle(vis, (int(target_box.cx), int(target_box.cy)),
               int(radius), (180, 180, 0), 1)

    # 缩放到合理大小输出
    h, w = vis.shape[:2]
    max_w = 800
    if w > max_w:
        scale = max_w / w
        vis = cv2.resize(vis, (max_w, int(h * scale)))

    nearby_desc = (
        f"发现 {len(nearby)} 个附近轨迹：" +
        "，".join([f"ID {n['track_id']}（距离{n['distance']}px）" for n in nearby])
        if nearby else "附近无其他轨迹"
    )

    return ToolResult(
        tool_name="check_nearby_tracks",
        success=True,
        text=f"帧 {frame_id} 轨迹 {track_id} 附近（{radius}px）：{nearby_desc}",
        images_b64=[_ndarray_to_b64(vis)],
        metadata={"nearby_tracks": nearby, "search_radius": radius},
    )


# ──────────────────────────────────────────────
# 工具分发器
# ──────────────────────────────────────────────

class ToolExecutor:
    """统一工具调用入口，供 LLM 调查器使用"""

    def __init__(
        self,
        tracks: Dict[int, Track],
        frames: Dict[int, np.ndarray],
    ):
        self.tracks = tracks
        self.frames = frames

    def execute(self, tool_name: str, params: Dict,
                fallback_track_id=None) -> ToolResult:
        params = self._normalize_params(tool_name, params, fallback_track_id)
        track_id = params.get("track_id")
        track = self.tracks.get(track_id) if track_id is not None else None

        try:
            if tool_name == "replay_segment":
                if track is None:
                    return ToolResult(tool_name=tool_name, success=False, text="缺少有效 track_id")
                if "start_frame" not in params or "end_frame" not in params:
                    return ToolResult(tool_name=tool_name, success=False, text="缺少 start_frame/end_frame")
                return replay_segment(
                    track, self.frames,
                    params["start_frame"], params["end_frame"],
                    params.get("step", 1),
                )
            elif tool_name == "zoom_region":
                if track is None:
                    return ToolResult(tool_name=tool_name, success=False, text="缺少有效 track_id")
                if "frame_id" not in params:
                    return ToolResult(tool_name=tool_name, success=False, text="缺少 frame_id")
                return zoom_region(
                    track, self.frames,
                    params["frame_id"],
                    params.get("scale", 3.0),
                    params.get("padding", 0.5),
                    params.get("sharpen", True),
                )
            elif tool_name == "enhance_image":
                if track is None:
                    return ToolResult(tool_name=tool_name, success=False, text="缺少有效 track_id")
                if "frame_id" not in params:
                    return ToolResult(tool_name=tool_name, success=False, text="缺少 frame_id")
                return enhance_image(
                    track, self.frames,
                    params["frame_id"],
                    params.get("method", "contrast"),
                )
            elif tool_name == "compare_appearances":
                if track is None:
                    return ToolResult(tool_name=tool_name, success=False, text="缺少有效 track_id")
                if "frame_id_a" not in params or "frame_id_b" not in params:
                    return ToolResult(tool_name=tool_name, success=False, text="缺少 frame_id_a/frame_id_b")
                return compare_appearances(
                    track, self.frames,
                    params["frame_id_a"], params["frame_id_b"],
                )
            elif tool_name == "check_nearby_tracks":
                if "frame_id" not in params or "track_id" not in params:
                    return ToolResult(tool_name=tool_name, success=False, text="缺少 frame_id/track_id")
                return check_nearby_tracks(
                    self.tracks, self.frames,
                    params["frame_id"],
                    params["track_id"],
                    params.get("radius", 150),
                )
            else:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    text=f"未知工具：{tool_name}",
                )
        except Exception as e:
            logger.exception(f"工具 {tool_name} 执行出错")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                text=f"工具执行异常：{e}",
            )

    def _normalize_params(self, tool_name: str, params: Dict,
                          fallback_track_id) -> Dict:
        p = dict(params)
        track = self.tracks.get(fallback_track_id) if fallback_track_id is not None else None

        # track_id 兜底
        if "track_id" not in p and fallback_track_id is not None:
            p["track_id"] = fallback_track_id

        # frame_ids / frames 别名（列表）
        frame_list = None
        for alias in ("frame_ids", "frames", "key_frames"):
            if alias in p and isinstance(p[alias], list) and p[alias]:
                frame_list = [int(v) for v in p[alias]]
                break

        # frame_id 别名归一
        if "frame_id" not in p:
            for alias in ("frame", "frame_num", "frame_number", "frameid"):
                if alias in p:
                    p["frame_id"] = p.pop(alias)
                    break
            if "frame_id" not in p and frame_list:
                p["frame_id"] = int(frame_list[0])
            if "frame_id" not in p and "frame_range" in p:
                fr = p["frame_range"]
                if isinstance(fr, list) and len(fr) >= 2:
                    p["frame_id"] = (int(fr[0]) + int(fr[1])) // 2

        # frame_id_a 别名归一
        if "frame_id_a" not in p:
            for alias in ("frame_a", "frame_1", "frame_id_1", "frame_before",
                          "reference_frame_id", "ref_frame_id"):
                if alias in p:
                    p["frame_id_a"] = p.pop(alias)
                    break
            # compare_appearances: frame_id 本身作为 frame_id_a
            if "frame_id_a" not in p and tool_name == "compare_appearances":
                if "frame_id" in p:
                    p["frame_id_a"] = p.pop("frame_id")
            if "frame_id_a" not in p and frame_list:
                p["frame_id_a"] = int(frame_list[0])
            if "frame_id_a" not in p and "frame_range" in p:
                fr = p["frame_range"]
                if isinstance(fr, list) and len(fr) >= 1:
                    p["frame_id_a"] = int(fr[0])

        # frame_id_b 别名归一
        if "frame_id_b" not in p:
            for alias in ("frame_b", "frame_2", "frame_id_2", "frame_after",
                          "compare_frame_id", "target_frame_id"):
                if alias in p:
                    p["frame_id_b"] = p.pop(alias)
                    break
            if "frame_id_b" not in p:
                if tool_name == "compare_appearances" and "end_frame" in p:
                    p["frame_id_b"] = p["end_frame"]
                elif frame_list and len(frame_list) >= 2:
                    p["frame_id_b"] = int(frame_list[-1])
                elif "frame_range" in p:
                    fr = p["frame_range"]
                    if isinstance(fr, list) and len(fr) >= 2:
                        p["frame_id_b"] = int(fr[1])
                elif "next_frame_id" in p:
                    p["frame_id_b"] = int(p["next_frame_id"])
                elif "reference_frame_id" in p:
                    p["frame_id_b"] = int(p["reference_frame_id"])

        # compare_appearances 最终兜底
        if tool_name == "compare_appearances" and "frame_id_a" in p and "frame_id_b" not in p:
            p["frame_id_b"] = int(p["frame_id_a"]) + 1
        if tool_name == "compare_appearances" and "frame_id_b" in p and "frame_id_a" not in p:
            p["frame_id_a"] = int(p["frame_id_b"]) - 1

        # replay_segment 若区间过大，自动增大步长，避免超大拼图
        if tool_name == "replay_segment":
            if "start_frame" in p and "end_frame" in p:
                start = int(p["start_frame"])
                end = int(p["end_frame"])
                if end < start:
                    start, end = end, start
                p["start_frame"], p["end_frame"] = start, end
                span = end - start + 1
                default_step = max(1, int(np.ceil(span / 48)))
                p["step"] = max(int(p.get("step", 1)), default_step)
            elif track is not None and track.boxes:
                p["start_frame"] = track.start_frame
                p["end_frame"] = track.end_frame
                span = p["end_frame"] - p["start_frame"] + 1
                p["step"] = max(1, int(np.ceil(span / 48)))

        return p
