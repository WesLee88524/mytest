"""
阶段三：轨迹修正器

目标：
1. 根据阶段二调查结论执行可解释的规则修正；
2. 每次修正后进行轻量级局部验证（可选）；
3. 验证失败则自动回滚，避免“修错比不修更糟”。
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_types import AnomalyType, Box, Track
from llm_investigator import Investigation
from vlm_inspector import QwenVLBackend, TrackInspector

logger = logging.getLogger(__name__)


@dataclass
class CorrectionAction:
    track_id: int
    anomaly_type: AnomalyType
    frame_range: Tuple[int, int]
    confidence: float
    related_track_id: Optional[int]
    attempted: bool
    applied: bool
    verified: bool
    rolled_back: bool
    details: str


@dataclass
class CorrectionReport:
    before_track_count: int
    after_track_count: int
    actions: List[CorrectionAction] = field(default_factory=list)

    @property
    def applied_count(self) -> int:
        return sum(1 for a in self.actions if a.applied and not a.rolled_back)

    @property
    def rollback_count(self) -> int:
        return sum(1 for a in self.actions if a.rolled_back)

    def summary(self) -> str:
        lines = [
            f"阶段三修正报告：共尝试 {len(self.actions)} 条",
            f"成功生效 {self.applied_count} 条，回滚 {self.rollback_count} 条",
            f"轨迹数：{self.before_track_count} -> {self.after_track_count}",
        ]
        for a in self.actions:
            lines.append(
                f"  轨迹 {a.track_id} | {a.anomaly_type.value} | "
                f"帧 {a.frame_range[0]}-{a.frame_range[1]} | "
                f"applied={a.applied} verified={a.verified} rollback={a.rolled_back} | "
                f"{a.details}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "before_track_count": self.before_track_count,
            "after_track_count": self.after_track_count,
            "applied_count": self.applied_count,
            "rollback_count": self.rollback_count,
            "actions": [
                {
                    "track_id": a.track_id,
                    "anomaly_type": a.anomaly_type.value,
                    "frame_range": list(a.frame_range),
                    "confidence": a.confidence,
                    "related_track_id": a.related_track_id,
                    "attempted": a.attempted,
                    "applied": a.applied,
                    "verified": a.verified,
                    "rolled_back": a.rolled_back,
                    "details": a.details,
                }
                for a in self.actions
            ],
        }


def clone_tracks(tracks: Dict[int, Track]) -> Dict[int, Track]:
    return {
        tid: Track(track_id=t.track_id, boxes=[copy.deepcopy(b) for b in t.boxes])
        for tid, t in tracks.items()
    }


def _dedup_sort_boxes(boxes: List[Box]) -> List[Box]:
    best = {}
    for b in boxes:
        old = best.get(b.frame_id)
        if old is None or b.conf >= old.conf:
            best[b.frame_id] = b
    return sorted(best.values(), key=lambda x: x.frame_id)


def _interpolate_box(a: Box, b: Box, fid: int) -> Box:
    if b.frame_id == a.frame_id:
        return Box(fid, a.x, a.y, a.w, a.h, min(a.conf, b.conf))
    t = (fid - a.frame_id) / (b.frame_id - a.frame_id)
    return Box(
        frame_id=fid,
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        w=max(1.0, a.w + (b.w - a.w) * t),
        h=max(1.0, a.h + (b.h - a.h) * t),
        conf=float(max(0.4, min(a.conf, b.conf, 0.85))),
    )


class TrackCorrector:
    """
    阶段三修正器：
    - 规则执行（ID 切换、漂移、漏检、碎片）
    - 局部验证（可选）
    - 失败回滚
    """

    def __init__(
        self,
        frames: Dict[int, np.ndarray],
        backend: Optional[QwenVLBackend] = None,
        enable_validation: bool = True,
        validation_gain: float = 0.15,
    ):
        self.frames = frames
        self.backend = backend
        self.enable_validation = enable_validation and backend is not None
        self.validation_gain = validation_gain
        self._validator = (
            TrackInspector(backend, suspicion_threshold=0.0, max_contact_frames=8)
            if self.enable_validation
            else None
        )

    def correct(
        self,
        tracks: Dict[int, Track],
        investigations: List[Investigation],
        min_confidence: float = 0.60,
        only_confirmed: bool = True,
    ) -> Tuple[Dict[int, Track], CorrectionReport]:
        corrected = clone_tracks(tracks)
        report = CorrectionReport(
            before_track_count=len(corrected),
            after_track_count=len(corrected),
        )

        for inv in investigations:
            if only_confirmed and not inv.confirmed:
                continue
            if inv.confidence < min_confidence:
                continue
            if inv.anomaly_type == AnomalyType.OK:
                continue

            snapshot = clone_tracks(corrected)
            applied, detail = self._apply_one(corrected, inv)
            verified = True
            rolled_back = False

            if applied and self.enable_validation:
                verified = self._validate_local(corrected, inv)
                if not verified:
                    corrected = snapshot
                    rolled_back = True
                    detail = f"{detail}；局部验证未通过，已回滚"

            report.actions.append(
                CorrectionAction(
                    track_id=inv.track_id,
                    anomaly_type=inv.anomaly_type,
                    frame_range=inv.frame_range,
                    confidence=inv.confidence,
                    related_track_id=inv.related_track_id,
                    attempted=True,
                    applied=applied,
                    verified=verified,
                    rolled_back=rolled_back,
                    details=detail,
                )
            )

            if applied and not rolled_back:
                logger.info(
                    "修正生效：track=%s type=%s frames=%s-%s",
                    inv.track_id, inv.anomaly_type.value, inv.frame_range[0], inv.frame_range[1]
                )
            elif applied and rolled_back:
                logger.info(
                    "修正回滚：track=%s type=%s",
                    inv.track_id, inv.anomaly_type.value
                )
            else:
                logger.info(
                    "修正跳过/失败：track=%s type=%s detail=%s",
                    inv.track_id, inv.anomaly_type.value, detail
                )

        report.after_track_count = len(corrected)
        return corrected, report

    def _apply_one(self, tracks: Dict[int, Track], inv: Investigation) -> Tuple[bool, str]:
        if inv.anomaly_type == AnomalyType.ID_SWITCH:
            return self._fix_id_switch(tracks, inv)
        if inv.anomaly_type == AnomalyType.DRIFT:
            return self._fix_drift(tracks, inv)
        if inv.anomaly_type == AnomalyType.MISS:
            return self._fix_miss(tracks, inv)
        if inv.anomaly_type == AnomalyType.FRAGMENT:
            return self._fix_fragment(tracks, inv)
        return False, "未知错误类型"

    def _fix_id_switch(self, tracks: Dict[int, Track], inv: Investigation) -> Tuple[bool, str]:
        main = tracks.get(inv.track_id)
        other = tracks.get(inv.related_track_id) if inv.related_track_id is not None else None
        if main is None:
            return False, "主轨迹不存在"
        if other is None:
            # 兜底策略：阶段二未给出 related_track_id 时，尝试剔除可疑帧段，
            # 避免“确认了 ID 切换但完全不修”的情况。
            start, end = inv.frame_range
            before = len(main.boxes)
            kept = [b for b in main.boxes if not (start <= b.frame_id <= end)]
            removed = before - len(kept)
            if removed <= 0:
                return False, "缺少 related_track_id，且可疑区间无可剔除帧"
            main.boxes = kept
            return True, f"缺少 related_track_id，已剔除主轨迹可疑帧段 {start}-{end}（{removed} 帧）"

        start, _ = inv.frame_range
        move_boxes = [copy.deepcopy(b) for b in other.boxes if b.frame_id >= start]
        if not move_boxes:
            return False, "related 轨迹在可疑区间后无可迁移帧"

        main.boxes.extend(move_boxes)
        main.boxes = _dedup_sort_boxes(main.boxes)

        other.boxes = [b for b in other.boxes if b.frame_id < start]
        if not other.boxes:
            del tracks[other.track_id]
            return True, f"将轨迹 {other.track_id} 从帧 {start} 起并入 {main.track_id}，并删除空轨迹"
        return True, f"将轨迹 {other.track_id} 从帧 {start} 起并入 {main.track_id}"

    def _fix_fragment(self, tracks: Dict[int, Track], inv: Investigation) -> Tuple[bool, str]:
        main = tracks.get(inv.track_id)
        other = tracks.get(inv.related_track_id) if inv.related_track_id is not None else None
        if main is None:
            return False, "主轨迹不存在"
        if other is None:
            return False, "缺少 related_track_id 或 related 轨迹不存在"

        main.boxes.extend(copy.deepcopy(other.boxes))
        main.boxes = _dedup_sort_boxes(main.boxes)
        del tracks[other.track_id]
        return True, f"合并碎片轨迹 {other.track_id} -> {main.track_id}"

    def _fix_drift(self, tracks: Dict[int, Track], inv: Investigation) -> Tuple[bool, str]:
        track = tracks.get(inv.track_id)
        if track is None or not track.boxes:
            return False, "轨迹不存在或为空"

        start, end = inv.frame_range
        prev_boxes = [b for b in track.boxes if b.frame_id < start]
        next_boxes = [b for b in track.boxes if b.frame_id > end]
        if not prev_boxes or not next_boxes:
            return False, "缺少前后锚点，无法插值平滑"

        a = prev_boxes[-1]
        b = next_boxes[0]
        updated = 0
        added = 0
        by_fid = {bx.frame_id: bx for bx in track.boxes}
        for fid in range(start, end + 1):
            existed = fid in by_fid
            by_fid[fid] = _interpolate_box(a, b, fid)
            if existed:
                updated += 1
            else:
                added += 1
        track.boxes = _dedup_sort_boxes(list(by_fid.values()))
        changed = updated + added
        return (changed > 0), f"平滑修正漂移帧 {start}-{end}，更新 {updated} 帧，新增 {added} 帧"

    def _fix_miss(self, tracks: Dict[int, Track], inv: Investigation) -> Tuple[bool, str]:
        track = tracks.get(inv.track_id)
        if track is None or len(track.boxes) < 2:
            return False, "轨迹太短，无法补点"

        start, end = inv.frame_range
        prev_boxes = [b for b in track.boxes if b.frame_id < start]
        next_boxes = [b for b in track.boxes if b.frame_id > end]
        if not prev_boxes or not next_boxes:
            return False, "缺少前后锚点，无法插值补全"

        a = prev_boxes[-1]
        b = next_boxes[0]
        if b.frame_id - a.frame_id <= 1:
            return False, "前后锚点相邻，无需补全"

        added = []
        existing = {bx.frame_id for bx in track.boxes}
        for fid in range(max(a.frame_id + 1, start), min(b.frame_id - 1, end) + 1):
            if fid not in existing:
                added.append(_interpolate_box(a, b, fid))

        if not added:
            return False, "可疑区间内无缺失帧可补全"

        track.boxes.extend(added)
        track.boxes = _dedup_sort_boxes(track.boxes)
        return True, f"补全漏检帧 {len(added)} 个（{added[0].frame_id}-{added[-1].frame_id}）"

    def _validate_local(self, tracks: Dict[int, Track], inv: Investigation) -> bool:
        if self._validator is None:
            return True

        track = tracks.get(inv.track_id)
        if track is None:
            return False

        suspicion = self._validator.inspect_one(track, self.frames)
        if suspicion.anomaly_type != inv.anomaly_type:
            return True
        # 同类型但发生在不同帧段，说明当前修正目标区域已缓解，也可放行
        if not self._ranges_overlap(suspicion.frame_range, inv.frame_range):
            return True
        # 同类型且同区域：置信度有下降即可通过（默认下降 15%）
        return suspicion.confidence <= max(0.10, inv.confidence * (1.0 - self.validation_gain))

    @staticmethod
    def _ranges_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return max(a[0], b[0]) <= min(a[1], b[1])


def dump_mot_result(tracks: Dict[int, Track], output_path: str) -> None:
    """
    将修正后的轨迹写回 MOTChallenge 文本：
      frame_id, track_id, x, y, w, h, conf, -1, -1, -1
    """
    rows = []
    for tid, track in tracks.items():
        for b in track.boxes:
            rows.append((b.frame_id, tid, b.x, b.y, b.w, b.h, b.conf))
    rows.sort(key=lambda r: (r[0], r[1]))

    with open(output_path, "w", encoding="utf-8") as f:
        for fid, tid, x, y, w, h, conf in rows:
            f.write(f"{fid},{tid},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.4f},-1,-1,-1\n")


# python run_pipeline.py \
#   --frames /public/home/lyh_npu/code/RT-MOTRv2/data/DanceTrack/test/dancetrack0003/img1 \
#   --mot /public/home/lyh_npu/code/RT-MOTRv2/exps/test/tracker-66.5/dancetrack0003.txt \
#   --model /public/home/lyh_npu/models/Qwen2.5-VL-7B-Instruct \
#   --output pipeline_report.json \
#   --enable-stage3 \
#   --stage3-min-confidence 0.65

# nohup python -u run_pipeline.py \
#   --frames-root /public/home/lyh_npu/code/RT-MOTRv2/data/DanceTrack/test \
#   --mot-root /public/home/lyh_npu/code/RT-MOTRv2/exps/test/tracker-66.5 \
#   --model /public/home/lyh_npu/models/Qwen2.5-VL-7B-Instruct \
#   --output /public/home/lyh_npu/code/PostMOT/pipeline_report_batch.json \
#   --enable-stage3 \
#   --stage3-min-confidence 0.65 \
#   --corrected-mot /public/home/lyh_npu/code/PostMOT/corrected_mot > test.log 2>&1 &
