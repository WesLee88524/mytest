"""
数据结构定义 —— MOT 轨迹巡检器
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class AnomalyType(str, Enum):
    ID_SWITCH   = "id_switch"    # 目标 ID 在中途切换成另一个目标
    DRIFT       = "drift"         # 轨迹漂移，位置跳变
    MISS        = "miss"          # 漏检，轨迹中断
    FRAGMENT    = "fragment"      # 同一目标被拆成多段短轨迹
    OK          = "ok"            # 无异常


@dataclass
class Box:
    """单帧 bounding box"""
    frame_id: int
    x: float       # 左上角 x
    y: float       # 左上角 y
    w: float       # 宽
    h: float       # 高
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class Track:
    """一条完整轨迹"""
    track_id: int
    boxes: List[Box] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.boxes[0].frame_id if self.boxes else -1

    @property
    def end_frame(self) -> int:
        return self.boxes[-1].frame_id if self.boxes else -1

    @property
    def length(self) -> int:
        return len(self.boxes)

    def frame_ids(self) -> List[int]:
        return [b.frame_id for b in self.boxes]


@dataclass
class Suspicion:
    """VLM 输出的可疑片段记录"""
    track_id: int
    anomaly_type: AnomalyType
    frame_range: Tuple[int, int]   # (start_frame, end_frame)
    confidence: float              # 0~1，越高越可疑
    reason: str                    # VLM 给出的自然语言理由
    related_track_id: Optional[int] = None  # 疑似混入的另一条轨迹 ID


@dataclass
class InspectionReport:
    """阶段一完整输出"""
    total_tracks: int
    suspicious: List[Suspicion] = field(default_factory=list)
    ok_tracks: List[int] = field(default_factory=list)

    @property
    def suspicious_count(self) -> int:
        return len(self.suspicious)

    def summary(self) -> str:
        lines = [
            f"共检查轨迹：{self.total_tracks} 条",
            f"可疑轨迹：{self.suspicious_count} 条",
            f"正常轨迹：{len(self.ok_tracks)} 条",
        ]
        for s in self.suspicious:
            lines.append(
                f"  轨迹 {s.track_id} | {s.anomaly_type.value} | "
                f"帧 {s.frame_range[0]}-{s.frame_range[1]} | "
                f"置信度 {s.confidence:.2f} | {s.reason}"
            )
        return "\n".join(lines)
