"""
第一版：从 tracker 结果 + GT 构建阶段一训练样本（JSONL）

目标：
  1) 自动发现 id_switch / miss / drift / fragment / false_positive 事件；
  2) 生成可用于 SFT / RL 的结构化样本；
  3) 与当前阶段一输出结构对齐（anomaly_type + frame_range + confidence）。

说明：
  - 输入 MOT 文本默认为 MOTChallenge xywh 格式：
      frame_id, track_id, x, y, w, h, conf, ...
  - 不依赖 scipy，使用贪心 IoU 匹配。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MotBox:
    frame_id: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h


def iou(a: MotBox, b: MotBox) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    if union <= 0:
        return 0.0
    return inter / union


def read_mot(txt_path: str) -> Dict[int, List[MotBox]]:
    by_frame: Dict[int, List[MotBox]] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            fid = int(float(p[0]))
            tid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            conf = float(p[6]) if len(p) > 6 else 1.0
            by_frame.setdefault(fid, []).append(MotBox(fid, tid, x, y, w, h, conf))
    return by_frame


def greedy_match(
    gt_boxes: List[MotBox],
    trk_boxes: List[MotBox],
    iou_thr: float,
) -> List[Tuple[int, int, float]]:
    """
    返回匹配列表：(gt_idx, trk_idx, iou)
    """
    pairs: List[Tuple[int, int, float]] = []
    for gi, g in enumerate(gt_boxes):
        for ti, t in enumerate(trk_boxes):
            pairs.append((gi, ti, iou(g, t)))
    pairs.sort(key=lambda x: x[2], reverse=True)

    used_g, used_t = set(), set()
    out = []
    for gi, ti, v in pairs:
        if v < iou_thr:
            break
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi)
        used_t.add(ti)
        out.append((gi, ti, v))
    return out


def group_consecutive(frames: List[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    frames = sorted(frames)
    segs = []
    s = p = frames[0]
    for f in frames[1:]:
        if f == p + 1:
            p = f
        else:
            segs.append((s, p))
            s = p = f
    segs.append((s, p))
    return segs


def build_events_for_sequence(
    seq_name: str,
    gt_by_frame: Dict[int, List[MotBox]],
    trk_by_frame: Dict[int, List[MotBox]],
    iou_match_thr: float = 0.5,
    drift_iou_thr: float = 0.3,
    miss_gap_min: int = 2,
    fp_min_len: int = 2,
) -> List[dict]:
    """
    产出事件级样本（可用于阶段一训练）：
      - id_switch / miss / drift / fragment / false_positive / ok
    """
    all_frames = sorted(set(gt_by_frame.keys()) | set(trk_by_frame.keys()))
    if not all_frames:
        return []

    # gt_id -> [(frame_id, matched_tracker_id or None, matched_iou)]
    gt_timeline: Dict[int, List[Tuple[int, Optional[int], float]]] = {}
    # tracker_id -> list[frame_id]
    tracker_frames: Dict[int, List[int]] = {}
    # tracker_id -> list[frame_id]（该帧该目标没有匹配到任何 GT）
    tracker_unmatched_frames: Dict[int, List[int]] = {}

    for fid in all_frames:
        gt_boxes = gt_by_frame.get(fid, [])
        trk_boxes = trk_by_frame.get(fid, [])
        for t in trk_boxes:
            tracker_frames.setdefault(t.track_id, []).append(fid)
        matches = greedy_match(gt_boxes, trk_boxes, iou_thr=iou_match_thr)

        g_to_t: Dict[int, Tuple[Optional[int], float]] = {i: (None, 0.0) for i in range(len(gt_boxes))}
        matched_ti = set()
        for gi, ti, ov in matches:
            g_to_t[gi] = (trk_boxes[ti].track_id, ov)
            matched_ti.add(ti)

        for ti, t in enumerate(trk_boxes):
            if ti not in matched_ti:
                tracker_unmatched_frames.setdefault(t.track_id, []).append(fid)

        for gi, g in enumerate(gt_boxes):
            tid, ov = g_to_t[gi]
            gt_timeline.setdefault(g.track_id, []).append((fid, tid, ov))

    events: List[dict] = []

    for gt_id, timeline in gt_timeline.items():
        # 1) id_switch：同一 gt 连续被不同 tracker id 覆盖
        prev_tid = None
        for fid, tid, ov in timeline:
            if tid is None:
                continue
            if prev_tid is not None and tid != prev_tid:
                events.append({
                    "seq_name": seq_name,
                    "gt_id": gt_id,
                    "track_id": tid,
                    "anomaly_type": "id_switch",
                    "frame_range": [max(1, fid - 2), fid + 2],
                    "confidence": 0.95,
                    "meta": {"prev_track_id": prev_tid, "iou": round(ov, 4)},
                })
            prev_tid = tid

        # 2) miss：GT 存在但没匹配 tracker
        miss_frames = [fid for fid, tid, _ in timeline if tid is None]
        for s, e in group_consecutive(miss_frames):
            if e - s + 1 >= miss_gap_min:
                events.append({
                    "seq_name": seq_name,
                    "gt_id": gt_id,
                    "track_id": None,
                    "anomaly_type": "miss",
                    "frame_range": [s, e],
                    "confidence": 0.90,
                    "meta": {"length": e - s + 1},
                })

        # 3) drift：匹配到了 tracker 但 IoU 很低
        drift_frames = [fid for fid, tid, ov in timeline if (tid is not None and ov < drift_iou_thr)]
        for s, e in group_consecutive(drift_frames):
            events.append({
                "seq_name": seq_name,
                "gt_id": gt_id,
                "track_id": None,
                "anomaly_type": "drift",
                "frame_range": [s, e],
                "confidence": 0.75,
                "meta": {"drift_iou_thr": drift_iou_thr},
            })

        # 4) fragment：同一 gt 由多个不连续 tracker 段表示
        matched_frames = [fid for fid, tid, _ in timeline if tid is not None]
        segs = group_consecutive(matched_frames)
        if len(segs) >= 2:
            for s, e in segs[1:]:
                events.append({
                    "seq_name": seq_name,
                    "gt_id": gt_id,
                    "track_id": None,
                    "anomaly_type": "fragment",
                    "frame_range": [s, e],
                    "confidence": 0.70,
                    "meta": {"segments": segs},
                })

    # 5) false_positive：tracker 连续出现，但无法匹配到任何 GT → 归为 fragment
    for tid, fids in tracker_unmatched_frames.items():
        for s, e in group_consecutive(fids):
            if e - s + 1 < fp_min_len:
                continue
            events.append({
                "seq_name": seq_name,
                "gt_id": None,
                "track_id": tid,
                "anomaly_type": "fragment",
                "frame_range": [s, e],
                "confidence": 0.88,
                "meta": {"length": e - s + 1},
            })

    # 6) ok 样本：从 gt_timeline 正向提取"连续稳定跟踪"片段
    # 条件：连续帧内同一 tracker id 且 IoU >= iou_match_thr，长度 >= ok_min_len
    ok_min_len = 10
    for gt_id, timeline in gt_timeline.items():
        # 按连续段分组：同一 tid 且 IoU 达标
        seg_start = None
        seg_tid = None
        seg_frames = []
        for fid, tid, ov in timeline:
            if tid is not None and ov >= iou_match_thr:
                if tid == seg_tid:
                    seg_frames.append(fid)
                else:
                    # 新 tid，先保存上一段
                    if seg_frames and len(seg_frames) >= ok_min_len:
                        events.append({
                            "seq_name": seq_name,
                            "gt_id": gt_id,
                            "track_id": seg_tid,
                            "anomaly_type": "ok",
                            "frame_range": [seg_frames[0], seg_frames[-1]],
                            "confidence": 0.90,
                            "meta": {"source": "gt_stable", "length": len(seg_frames)},
                        })
                    seg_tid = tid
                    seg_frames = [fid]
            else:
                # 中断，保存当前段
                if seg_frames and len(seg_frames) >= ok_min_len:
                    events.append({
                        "seq_name": seq_name,
                        "gt_id": gt_id,
                        "track_id": seg_tid,
                        "anomaly_type": "ok",
                        "frame_range": [seg_frames[0], seg_frames[-1]],
                        "confidence": 0.90,
                        "meta": {"source": "gt_stable", "length": len(seg_frames)},
                    })
                seg_tid = None
                seg_frames = []
        # 收尾
        if seg_frames and len(seg_frames) >= ok_min_len:
            events.append({
                "seq_name": seq_name,
                "gt_id": gt_id,
                "track_id": seg_tid,
                "anomaly_type": "ok",
                "frame_range": [seg_frames[0], seg_frames[-1]],
                "confidence": 0.90,
                "meta": {"source": "gt_stable", "length": len(seg_frames)},
            })

    return events


def write_jsonl(events: List[dict], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _find_gt_file(seq_dir: Path) -> Optional[Path]:
    cands = [
        seq_dir / "gt" / "gt.txt",
        seq_dir / "gt.txt",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def _find_tracker_file(tracker_root: Path, seq_name: str) -> Optional[Path]:
    cands = [
        tracker_root / f"{seq_name}.txt",
        tracker_root / seq_name / "tracks.txt",
        tracker_root / seq_name / "result.txt",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def run_batch(
    gt_root: str,
    tracker_root: str,
    output_dir: str,
    seq_glob: str,
    iou_match_thr: float,
    drift_iou_thr: float,
    miss_gap_min: int,
    fp_min_len: int,
) -> dict:
    gt_root_p = Path(gt_root)
    trk_root_p = Path(tracker_root)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = [p for p in sorted(gt_root_p.glob(seq_glob)) if p.is_dir()]
    summary = {
        "mode": "batch",
        "gt_root": str(gt_root_p),
        "tracker_root": str(trk_root_p),
        "sequence_count": 0,
        "sequences": [],
    }

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        gt_file = _find_gt_file(seq_dir)
        trk_file = _find_tracker_file(trk_root_p, seq_name)
        if gt_file is None or trk_file is None:
            summary["sequences"].append({
                "seq_name": seq_name,
                "status": "skipped",
                "reason": f"missing gt/tracker file: gt={gt_file}, tracker={trk_file}",
            })
            continue

        gt_by_frame = read_mot(str(gt_file))
        trk_by_frame = read_mot(str(trk_file))
        events = build_events_for_sequence(
            seq_name=seq_name,
            gt_by_frame=gt_by_frame,
            trk_by_frame=trk_by_frame,
            iou_match_thr=iou_match_thr,
            drift_iou_thr=drift_iou_thr,
            miss_gap_min=miss_gap_min,
            fp_min_len=fp_min_len,
        )
        out_jsonl = out_root / f"{seq_name}.jsonl"
        write_jsonl(events, str(out_jsonl))
        summary["sequence_count"] += 1
        summary["sequences"].append({
            "seq_name": seq_name,
            "status": "ok",
            "gt_file": str(gt_file),
            "tracker_file": str(trk_file),
            "output_jsonl": str(out_jsonl),
            "event_count": len(events),
        })
        print(f"[{seq_name}] 写出 {len(events)} 条事件到 {out_jsonl}")

    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[batch] 汇总写出: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="构建阶段一 RL/SFT 训练样本（第一版）")
    # 单序列模式
    parser.add_argument("--seq-name", type=str, default=None)
    parser.add_argument("--gt", type=str, default=None, help="GT MOT txt")
    parser.add_argument("--tracker", type=str, default=None, help="tracker MOT txt (ByteTrack/OCSORT)")
    parser.add_argument("--output", type=str, default=None, help="输出 JSONL")
    # 批量模式
    parser.add_argument("--gt-root", type=str, default=None, help="多序列 GT 根目录（如 DanceTrack/val）")
    parser.add_argument("--tracker-root", type=str, default=None, help="多序列 tracker 结果根目录")
    parser.add_argument("--output-dir", type=str, default=None, help="批量模式输出目录（每序列一个 jsonl）")
    parser.add_argument("--seq-glob", type=str, default="dancetrack*", help="序列匹配模式")
    parser.add_argument("--iou-match-thr", type=float, default=0.5)
    parser.add_argument("--drift-iou-thr", type=float, default=0.3)
    parser.add_argument("--miss-gap-min", type=int, default=2)
    parser.add_argument("--fp-min-len", type=int, default=2)
    args = parser.parse_args()

    batch_mode = bool(args.gt_root and args.tracker_root and args.output_dir)
    if batch_mode:
        run_batch(
            gt_root=args.gt_root,
            tracker_root=args.tracker_root,
            output_dir=args.output_dir,
            seq_glob=args.seq_glob,
            iou_match_thr=args.iou_match_thr,
            drift_iou_thr=args.drift_iou_thr,
            miss_gap_min=args.miss_gap_min,
            fp_min_len=args.fp_min_len,
        )
        return

    if not (args.seq_name and args.gt and args.tracker and args.output):
        parser.error("单序列模式需要 --seq-name --gt --tracker --output；或使用批量参数 --gt-root --tracker-root --output-dir")

    gt_by_frame = read_mot(args.gt)
    trk_by_frame = read_mot(args.tracker)
    events = build_events_for_sequence(
        seq_name=args.seq_name,
        gt_by_frame=gt_by_frame,
        trk_by_frame=trk_by_frame,
        iou_match_thr=args.iou_match_thr,
        drift_iou_thr=args.drift_iou_thr,
        miss_gap_min=args.miss_gap_min,
        fp_min_len=args.fp_min_len,
    )
    write_jsonl(events, args.output)
    print(f"[{args.seq_name}] 写出 {len(events)} 条事件到 {args.output}")


if __name__ == "__main__":
    main()
