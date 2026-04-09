"""
完整 Pipeline 运行脚本
阶段一（VLM 巡检）→ 阶段二（LLM 调查）→ 输出报告

使用方式：
  # Demo 模式
  python run_pipeline.py --demo \
      --model /public/home/lyh_npu/models/Qwen2.5-VL-7B-Instruct

  # 真实数据
  python run_pipeline.py \
      --frames /path/to/frames/ \
      --mot    /path/to/result.txt \
      --model  /public/home/lyh_npu/models/Qwen2.5-VL-7B-Instruct
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from data_types import Box, Track, Suspicion, AnomalyType
from track_preprocessor import load_frames
from vlm_inspector import QwenVLBackend, TrackInspector
from investigation_tools import ToolExecutor
from llm_investigator import LLMInvestigator
from track_corrector import TrackCorrector, dump_mot_result, clone_tracks
from stage_visualizer import visualize_stage1, visualize_stage2, visualize_stage3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────

def load_mot_result(mot_path: str) -> Dict[int, Track]:
    track_boxes: dict = defaultdict(list)
    with open(mot_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            fid  = int(float(parts[0]))
            tid  = int(float(parts[1]))
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            track_boxes[tid].append(Box(fid, x, y, w, h, conf))

    tracks = {}
    for tid, boxes in track_boxes.items():
        boxes.sort(key=lambda b: b.frame_id)
        tracks[tid] = Track(track_id=tid, boxes=boxes)
    logger.info(f"加载轨迹 {len(tracks)} 条")
    return tracks


def resolve_batch_pairs(
    frames_root: str,
    mot_root: str,
    seq_glob: str = "dancetrack*",
) -> List[Tuple[str, str, str]]:
    """
    从多序列目录中构建 (seq_name, frames_dir, mot_file) 配对列表。
    约定：
      - 帧目录结构：{frames_root}/{seq_name}/img1
      - 结果文件：  {mot_root}/{seq_name}.txt
    """
    pairs: List[Tuple[str, str, str]] = []
    root = Path(frames_root)
    mot_dir = Path(mot_root)
    for seq_dir in sorted(root.glob(seq_glob)):
        if not seq_dir.is_dir():
            continue
        seq_name = seq_dir.name
        img1 = seq_dir / "img1"
        frames_dir = img1 if img1.is_dir() else seq_dir
        mot_file = mot_dir / f"{seq_name}.txt"
        if not mot_file.exists():
            logger.warning(f"[{seq_name}] 跳过：未找到 MOT 文件 {mot_file}")
            continue
        pairs.append((seq_name, str(frames_dir), str(mot_file)))
    return pairs


def generate_demo_data(n_tracks=5, n_frames=60):
    import random
    frames = {}
    for fid in range(1, n_frames + 1):
        img = np.full((480, 640, 3), 30, dtype=np.uint8)
        img += np.random.randint(0, 40, img.shape, dtype=np.uint8)
        frames[fid] = np.clip(img, 0, 255).astype(np.uint8)

    tracks = {}
    for tid in range(1, n_tracks + 1):
        boxes = []
        x = random.randint(50, 400)
        y = random.randint(50, 300)
        vx = random.choice([-2, 2, 3, -3])
        vy = random.choice([-1, 1])
        for fid in range(1, n_frames + 1):
            if fid == 30 and tid == 2:
                x += 80  # 制造 drift
            x = max(10, min(580, x + vx + random.randint(-1, 1)))
            y = max(10, min(420, y + vy + random.randint(-1, 1)))
            color = [(200,80,80),(80,200,80),(80,80,200),(200,200,80),(200,80,200)][tid%5]
            cv2.rectangle(frames[fid], (int(x),int(y)), (int(x+50),int(y+100)), color, -1)
            cv2.putText(frames[fid], str(tid), (int(x+5),int(y+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if tid == 3 and 20 <= fid <= 30:  # 制造 miss
                continue
            boxes.append(Box(fid, float(x), float(y), 50.0, 100.0, 0.9))
        if boxes:
            tracks[tid] = Track(track_id=tid, boxes=boxes)
    return frames, tracks


def run_single_sequence(
    seq_name: str,
    frames: Dict[int, np.ndarray],
    tracks: Dict[int, Track],
    backend: QwenVLBackend,
    args,
    output_path: str,
    viz_dir: Optional[str] = None,
) -> dict:
    # ── 阶段一：VLM 巡检 ──
    stage1_meta = {}
    if args.stage1_json:
        logger.info(f"[{seq_name}] 跳过阶段一，读取：{args.stage1_json}")
        with open(args.stage1_json) as f:
            s1_data = json.load(f)
        suspicions = []
        for s in s1_data.get("suspicious", []):
            suspicions.append(Suspicion(
                track_id=s["track_id"],
                anomaly_type=AnomalyType(s["anomaly_type"]),
                frame_range=tuple(s["frame_range"]),
                confidence=s["confidence"],
                reason=s["reason"],
                related_track_id=s.get("related_track_id"),
            ))
        stage1_meta = {
            "from_json": args.stage1_json,
            "total_tracks": s1_data.get("total_tracks", len(tracks)),
            "suspicious_count": len(suspicions),
        }
    else:
        logger.info("=" * 40)
        logger.info(f"[{seq_name}] 阶段一：VLM 轨迹巡检")
        logger.info("=" * 40)
        inspector = TrackInspector(backend, suspicion_threshold=args.threshold)
        track_list = list(tracks.values())
        s1_report = inspector.inspect(track_list, frames)
        suspicions = s1_report.suspicious

        s1_path = output_path.replace(".json", "_stage1.json")
        with open(s1_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_tracks": s1_report.total_tracks,
                "suspicious": [
                    {"track_id": s.track_id, "anomaly_type": s.anomaly_type.value,
                     "frame_range": list(s.frame_range), "confidence": s.confidence,
                     "reason": s.reason, "related_track_id": s.related_track_id}
                    for s in suspicions
                ],
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"[{seq_name}] 阶段一完成：{len(suspicions)} 条可疑轨迹，已保存至 {s1_path}")
        stage1_meta = {
            "output_json": s1_path,
            "total_tracks": s1_report.total_tracks,
            "suspicious_count": len(suspicions),
            "ok_count": len(s1_report.ok_tracks),
        }

    if viz_dir:
        try:
            visualize_stage1(
                seq_name=seq_name,
                suspicions=suspicions,
                tracks=tracks,
                frames=frames,
                out_dir=viz_dir,
            )
        except Exception as e:
            logger.warning(f"[{seq_name}] 阶段一可视化失败：{e}")

    final_payload = {
        "sequence": seq_name,
        "stage1": stage1_meta,
        "stage2": {"investigations": []},
    }

    if args.stage1_only or not suspicions:
        logger.info(f"[{seq_name}] 阶段一结束（stage1-only 或无可疑轨迹）")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, ensure_ascii=False, indent=2)
        return final_payload

    # ── 阶段二：LLM 调查 ──
    logger.info("=" * 40)
    logger.info(f"[{seq_name}] 阶段二：LLM 工具调用调查")
    logger.info("=" * 40)
    executor = ToolExecutor(tracks=tracks, frames=frames)
    investigator = LLMInvestigator(backend=backend, executor=executor)
    inv_report = investigator.investigate_all(suspicions, tracks, frames)
    final_payload["stage2"] = inv_report.to_dict()
    if viz_dir:
        try:
            visualize_stage2(
                seq_name=seq_name,
                investigations=inv_report.investigations,
                tracks=tracks,
                frames=frames,
                out_dir=viz_dir,
            )
        except Exception as e:
            logger.warning(f"[{seq_name}] 阶段二可视化失败：{e}")

    # ── 阶段三：轨迹修正（可选） ──
    if args.enable_stage3:
        logger.info("=" * 40)
        logger.info(f"[{seq_name}] 阶段三：轨迹修正")
        logger.info("=" * 40)
        tracks_before_stage3 = clone_tracks(tracks)
        corrector = TrackCorrector(
            frames=frames,
            backend=backend,
            enable_validation=(not args.stage3_no_verify),
        )
        corrected_tracks, c_report = corrector.correct(
            tracks=tracks,
            investigations=inv_report.investigations,
            min_confidence=args.stage3_min_confidence,
            only_confirmed=True,
        )
        corrected_mot = args.corrected_mot
        if not corrected_mot:
            corrected_mot = output_path.replace(".json", "_corrected.txt")
        dump_mot_result(corrected_tracks, corrected_mot)
        logger.info(f"[{seq_name}] 修正后 MOT 已保存至 {corrected_mot}")
        final_payload["stage3"] = {
            **c_report.to_dict(),
            "corrected_mot_path": corrected_mot,
            "validation_enabled": (not args.stage3_no_verify),
            "min_confidence": args.stage3_min_confidence,
        }
        if viz_dir:
            try:
                visualize_stage3(
                    seq_name=seq_name,
                    actions=c_report.actions,
                    tracks_before=tracks_before_stage3,
                    tracks_after=corrected_tracks,
                    frames=frames,
                    out_dir=viz_dir,
                )
            except Exception as e:
                logger.warning(f"[{seq_name}] 阶段三可视化失败：{e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[{seq_name}] 报告已保存至 {output_path}")
    return final_payload


def _parse_gpu_devices(gpu_devices: str) -> List[str]:
    if not gpu_devices:
        return []
    out = []
    for item in gpu_devices.split(","):
        item = item.strip()
        if item:
            out.append(item)
    return out


def _auto_detect_gpu_devices() -> List[str]:
    try:
        import torch
        n = torch.cuda.device_count()
        return [str(i) for i in range(n)] if n > 0 else []
    except Exception:
        return []


def _query_gpu_free_gb(gpu_id: str) -> Optional[float]:
    """
    查询指定 GPU 可用显存（GiB），失败返回 None。
    """
    try:
        cmd = [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        if not out:
            return None
        free_mb = float(out[0].strip())
        return free_mb / 1024.0
    except Exception:
        return None


def _process_sequence_worker(task: dict) -> dict:
    """
    多进程 worker：每个进程独立加载模型并处理一个序列。
    task 字段：
      - seq_name, frames_dir, mot_file, seq_output
      - args_dict（run_pipeline 参数 dict）
      - gpu_device（如 "0"/"1"）
      - viz_dir（可视化输出根目录）
    """
    seq_name = task["seq_name"]
    frames_dir = task["frames_dir"]
    mot_file = task["mot_file"]
    seq_output = task["seq_output"]
    args_dict = task["args_dict"]
    gpu_device = task.get("gpu_device", "")
    viz_dir = task.get("viz_dir", None)

    if gpu_device != "":
        # 子进程内固定可见 GPU，避免多进程争抢同卡
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        device = "cuda:0"
    else:
        device = "cuda"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    backend = QwenVLBackend(
        model_name=args_dict["model"],
        use_ollama=False,
        device=device,
        max_new_tokens=args_dict.get("max_new_tokens", 160),
    )
    backend.max_investigator_images = args_dict.get("investigator_max_images", 8)
    frames = load_frames(frames_dir)
    tracks = load_mot_result(mot_file)

    seq_args = argparse.Namespace(**args_dict)
    if args_dict.get("corrected_mot"):
        corr_root = Path(args_dict["corrected_mot"])
        corr_root.mkdir(parents=True, exist_ok=True)
        seq_args.corrected_mot = str(corr_root / f"{seq_name}.txt")
    else:
        seq_args.corrected_mot = None

    try:
        result = run_single_sequence(
            seq_name=seq_name,
            frames=frames,
            tracks=tracks,
            backend=backend,
            args=seq_args,
            output_path=seq_output,
            viz_dir=viz_dir,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        logger.warning(f"[{seq_name}] 首次运行 OOM，降级参数重试一次")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        backend = QwenVLBackend(
            model_name=args_dict["model"],
            use_ollama=False,
            device=device,
            max_new_tokens=max(64, int(args_dict.get("max_new_tokens", 160) * 0.5)),
        )
        backend.max_investigator_images = max(4, int(args_dict.get("investigator_max_images", 8) * 0.5))
        result = run_single_sequence(
            seq_name=seq_name,
            frames=frames,
            tracks=tracks,
            backend=backend,
            args=seq_args,
            output_path=seq_output,
            viz_dir=viz_dir,
        )
    return {
        "sequence": seq_name,
        "frames_dir": frames_dir,
        "mot_file": mot_file,
        "report_path": seq_output,
        "stage1_suspicious": result.get("stage1", {}).get("suspicious_count", 0),
        "stage2_count": len(result.get("stage2", {}).get("investigations", [])),
    }


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MOT 错误检测完整 Pipeline")
    parser.add_argument("--frames",     type=str, default=None)
    parser.add_argument("--mot",        type=str, default=None)
    parser.add_argument("--frames-root", type=str, default=None,
                        help="多序列帧根目录，如 .../DanceTrack/test")
    parser.add_argument("--mot-root", type=str, default=None,
                        help="多序列 MOT 结果根目录，如 .../tracker-xx")
    parser.add_argument("--seq-glob", type=str, default="dancetrack*",
                        help="多序列模式下匹配序列目录名的 glob")
    parser.add_argument("--model",      type=str,
                        default="/public/home/lyh_npu/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output",     type=str, default="pipeline_report.json")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--demo",       action="store_true")
    parser.add_argument("--stage1-only",action="store_true", help="只运行阶段一")
    parser.add_argument("--stage1-json",type=str, default=None,
                        help="跳过阶段一，直接用已有的 suspicions.json 运行阶段二")
    parser.add_argument("--enable-stage3", action="store_true", help="启用阶段三轨迹修正")
    parser.add_argument("--stage3-min-confidence", type=float, default=0.60,
                        help="阶段三执行修正的最小置信度")
    parser.add_argument("--stage3-no-verify", action="store_true",
                        help="关闭阶段三局部 VLM 验证（更快，但更激进）")
    parser.add_argument("--corrected-mot", type=str, default=None,
                        help="阶段三修正后的 MOT 输出路径（默认 output 同名 *_corrected.txt）")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="多序列并行进程数（仅 batch 模式生效，建议与 GPU 数一致）")
    parser.add_argument("--gpu-devices", type=str, default="",
                        help="逗号分隔 GPU 列表，如 0,1,2,3；为空则不限制可见 GPU")
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="可视化输出目录（按 seq_name/stage1|2|3 导出 jpg）")
    parser.add_argument("--only-seq", "--only_seq", dest="only_seq", type=str, default=None,
                        help="仅处理指定序列名（用于快速排查单序列）")
    parser.add_argument("--max-new-tokens", type=int, default=160,
                        help="VLM 生成最大 token（降显存优先建议 96~192）")
    parser.add_argument("--investigator-max-images", type=int, default=8,
                        help="阶段二每轮最多使用图片数（越小越省显存）")
    parser.add_argument("--min-gpu-free-gb", type=float, default=32.0,
                        help="并行前筛选 GPU 的最小空闲显存（GiB）")
    args = parser.parse_args()
    batch_mode = bool(args.frames_root and args.mot_root)
    if batch_mode and args.stage1_json:
        parser.error("多序列模式暂不支持 --stage1-json，请对每个序列单独运行或关闭该参数")

    if batch_mode:
        pairs = resolve_batch_pairs(args.frames_root, args.mot_root, args.seq_glob)
        if args.only_seq:
            pairs = [p for p in pairs if p[0] == args.only_seq]
            if not pairs:
                parser.error(f"--only-seq={args.only_seq} 未匹配到序列")
        if not pairs:
            parser.error("未发现可处理序列，请检查 --frames-root/--mot-root/--seq-glob")

        out_path = Path(args.output)
        seq_out_dir = out_path.parent / f"{out_path.stem}_seq_reports"
        seq_out_dir.mkdir(parents=True, exist_ok=True)

        aggregate = {
            "batch_mode": True,
            "frames_root": args.frames_root,
            "mot_root": args.mot_root,
            "sequence_count": len(pairs),
            "num_workers": args.num_workers,
            "gpu_devices": _parse_gpu_devices(args.gpu_devices),
            "sequences": [],
        }
        logger.info(f"多序列模式：共 {len(pairs)} 个序列，num_workers={args.num_workers}")

        if args.num_workers <= 1:
            logger.info(f"加载模型：{args.model}")
            backend = QwenVLBackend(
                model_name=args.model,
                use_ollama=False,
                device="cuda",
                max_new_tokens=args.max_new_tokens,
            )
            backend.max_investigator_images = args.investigator_max_images
            for idx, (seq_name, frames_dir, mot_file) in enumerate(pairs, start=1):
                logger.info(f"[{idx}/{len(pairs)}] 处理序列 {seq_name}")
                frames = load_frames(frames_dir)
                tracks = load_mot_result(mot_file)
                seq_output = str(seq_out_dir / f"{seq_name}.json")
                seq_args = argparse.Namespace(**vars(args))
                if args.corrected_mot:
                    corr_root = Path(args.corrected_mot)
                    corr_root.mkdir(parents=True, exist_ok=True)
                    seq_args.corrected_mot = str(corr_root / f"{seq_name}.txt")
                else:
                    seq_args.corrected_mot = None
                result = run_single_sequence(
                    seq_name=seq_name,
                    frames=frames,
                    tracks=tracks,
                    backend=backend,
                    args=seq_args,
                    output_path=seq_output,
                    viz_dir=args.viz_dir,
                )
                aggregate["sequences"].append({
                    "sequence": seq_name,
                    "frames_dir": frames_dir,
                    "mot_file": mot_file,
                    "report_path": seq_output,
                    "stage1_suspicious": result.get("stage1", {}).get("suspicious_count", 0),
                    "stage2_count": len(result.get("stage2", {}).get("investigations", [])),
                })
        else:
            gpu_list = _parse_gpu_devices(args.gpu_devices)
            if not gpu_list:
                gpu_list = _auto_detect_gpu_devices()
                if gpu_list:
                    logger.info(f"未显式指定 --gpu-devices，自动探测到 GPU: {gpu_list}")

            if gpu_list:
                usable = []
                for gid in gpu_list:
                    free_gb = _query_gpu_free_gb(gid)
                    if free_gb is None:
                        usable.append(gid)  # 查询失败时不强行剔除
                        continue
                    if free_gb >= args.min_gpu_free_gb:
                        usable.append(gid)
                    else:
                        logger.warning(
                            "GPU %s 可用显存 %.2f GiB < 阈值 %.2f GiB，跳过该卡",
                            gid, free_gb, args.min_gpu_free_gb
                        )
                gpu_list = usable
                if not gpu_list:
                    parser.error("无可用 GPU（空闲显存不足），请降低 --min-gpu-free-gb 或释放显存后重试")

                actual_workers = min(args.num_workers, len(gpu_list))
                if actual_workers < args.num_workers:
                    logger.warning(
                        "num_workers=%s 大于 GPU 数=%s，自动降为 %s 以避免多进程抢同卡导致 OOM",
                        args.num_workers, len(gpu_list), actual_workers
                    )
            else:
                # 未指定 GPU 且无法探测，保守退化到单进程，避免所有进程挤到 cuda:0
                actual_workers = 1
                logger.warning(
                    "未指定 GPU 且无法探测可用 GPU，已退化到单进程以降低 OOM 风险；"
                    "建议显式传 --gpu-devices"
                )

            logger.info(
                "并行模式启用：workers=%s（requested=%s），gpu_list=%s",
                actual_workers, args.num_workers, gpu_list or "未指定"
            )
            tasks = []
            args_dict = vars(args).copy()
            for idx, (seq_name, frames_dir, mot_file) in enumerate(pairs):
                seq_output = str(seq_out_dir / f"{seq_name}.json")
                gpu_device = gpu_list[idx % len(gpu_list)] if gpu_list else ""
                tasks.append({
                    "seq_name": seq_name,
                    "frames_dir": frames_dir,
                    "mot_file": mot_file,
                    "seq_output": seq_output,
                    "args_dict": args_dict,
                    "gpu_device": gpu_device,
                    "viz_dir": args.viz_dir,
                })

            done = 0
            with ProcessPoolExecutor(max_workers=actual_workers) as ex:
                futures = [ex.submit(_process_sequence_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    done += 1
                    try:
                        res = fut.result()
                        logger.info(f"[{done}/{len(tasks)}] 完成序列 {res['sequence']}")
                        aggregate["sequences"].append(res)
                    except Exception as e:
                        logger.exception(f"[{done}/{len(tasks)}] 子任务失败：{e}")
                        aggregate["sequences"].append({
                            "sequence": "unknown",
                            "frames_dir": "",
                            "mot_file": "",
                            "report_path": "",
                            "stage1_suspicious": 0,
                            "stage2_count": 0,
                            "error": str(e),
                        })

            aggregate["sequences"].sort(key=lambda x: x["sequence"])

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, ensure_ascii=False, indent=2)
        logger.info(f"多序列汇总报告已保存至 {args.output}")
        return

    # ── 模型加载（单序列） ──
    logger.info(f"加载模型：{args.model}")
    backend = QwenVLBackend(
        model_name=args.model,
        use_ollama=False,
        device="cuda",
        max_new_tokens=args.max_new_tokens,
    )
    backend.max_investigator_images = args.investigator_max_images

    # 单序列模式
    if args.demo:
        logger.info("Demo 模式：生成合成数据")
        frames, tracks = generate_demo_data()
        seq_name = "demo"
    elif args.stage1_json:
        if not args.frames or not args.mot:
            parser.error("使用 --stage1-json 时仍需指定 --frames 和 --mot")
        frames = load_frames(args.frames)
        tracks = load_mot_result(args.mot)
        seq_name = Path(args.mot).stem
    else:
        if not args.frames or not args.mot:
            parser.error("需要指定 --frames 和 --mot，或使用 --demo，或使用多序列参数")
        frames = load_frames(args.frames)
        tracks = load_mot_result(args.mot)
        seq_name = Path(args.mot).stem

    run_single_sequence(
        seq_name=seq_name,
        frames=frames,
        tracks=tracks,
        backend=backend,
        args=args,
        output_path=args.output,
        viz_dir=args.viz_dir,
    )


if __name__ == "__main__":
    main()
