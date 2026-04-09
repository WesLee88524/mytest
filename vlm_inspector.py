"""
VLM 轨迹巡检器
使用 Qwen3-VL（本地）分析每条轨迹的拼贴图 + 数值特征，
输出可疑片段清单。

依赖：transformers>=4.51, torch, Pillow, opencv-python-headless
不依赖 qwen-vl-utils（av / ffmpeg 等重量级依赖）。
"""
import base64
import json
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from data_types import AnomalyType, InspectionReport, Suspicion, Track
from track_preprocessor import (
    build_contact_sheet,
    compute_numeric_features,
    load_frames,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Prompt 模板
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个多目标跟踪（MOT）质检专家。
你的任务是检查单条目标轨迹是否存在跟踪错误。

可能的错误类型：
- id_switch：目标在中途被错误地切换成了另一个目标（外观突变）
- drift：目标位置发生异常跳变（不符合运动规律）
- miss：轨迹中断，目标消失超过5帧（漏检）
- fragment：该轨迹疑似是某个更长轨迹的碎片
- ok：轨迹正常，无明显错误

请严格按 JSON 格式输出，不要输出任何其他内容。"""

USER_PROMPT_TEMPLATE = """这是目标 ID={track_id} 的跟踪轨迹拼贴图（从左到右按时间顺序排列，每张图左上角显示帧号）。

数值特征参考：
{numeric_features}

请分析：
1. 目标外观是否在某帧发生突变（衣服、体型、颜色明显不同）？
2. 目标位置是否有异常跳变（max_displacement={max_disp}，max_disp_frame={max_disp_frame}）？
3. 轨迹是否存在中断（trajectory_breaks={breaks}）？

输出格式（严格 JSON，不含注释）：
{{
  "anomaly_type": "id_switch|drift|miss|fragment|ok",
  "frame_range": [起始帧, 结束帧],
  "confidence": 0.0到1.0之间的小数,
  "reason": "简短说明（中文，50字以内）",
  "related_track_id": null或疑似混入的轨迹ID
}}"""


# ──────────────────────────────────────────────
# 图像编码
# ──────────────────────────────────────────────

def ndarray_to_base64(img: np.ndarray, fmt: str = ".jpg") -> str:
    """将 numpy 图像编码为 base64 字符串"""
    success, buf = cv2.imencode(fmt, img)
    if not success:
        raise ValueError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def base64_to_pil(image_b64: str):
    """base64 字符串 → PIL.Image，不依赖任何额外包"""
    from PIL import Image
    img_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


# ──────────────────────────────────────────────
# Qwen3-VL 后端适配
# ──────────────────────────────────────────────

class QwenVLBackend:
    """
    本地 Qwen3-VL 推理后端。
    支持两种加载方式：
      - transformers (HuggingFace) —— 默认，不依赖 qwen-vl-utils
      - ollama HTTP API           —— 设置 use_ollama=True
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-7B-Instruct",
        use_ollama: bool = False,
        ollama_url: str = "http://localhost:11434",
        device: str = "cuda",
        max_new_tokens: int = 256,
    ):
        self.use_ollama = use_ollama
        self.max_new_tokens = max_new_tokens

        if use_ollama:
            import requests
            self._requests = requests
            self._ollama_url = ollama_url
            self._model_name = model_name
            logger.info(f"使用 Ollama 后端：{ollama_url}，模型：{model_name}")
        else:
            self._init_hf(model_name, device)

    def _init_hf(self, model_name: str, device: str):
        """加载 HuggingFace 本地模型，完全不触碰 qwen-vl-utils"""
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(f"加载本地模型：{model_name}（device={device}）")
        self._device = device

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            backend="torchvision",
        )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        self._model.eval()
        logger.info("模型加载完成")

    # ── 统一推理入口 ──────────────────────────

    def infer(self, image_b64: str, system: str, user: str) -> str:
        """统一推理接口，返回原始文本"""
        if self.use_ollama:
            return self._infer_ollama(image_b64, system, user)
        return self._infer_hf(image_b64, system, user)

    # ── HuggingFace 推理 ──────────────────────

    def _infer_hf(self, image_b64: str, system: str, user: str) -> str:
        """
        纯 transformers + Pillow 推理。
        核心策略：图像以 PIL Image 对象直接传入 processor 的 images= 参数，
        消息里用 {"type": "image"} 占位符，完全绕开 qwen-vl-utils 的
        fetch_image / video_reader 代码路径。
        """
        import torch

        pil_img = base64_to_pil(image_b64)

        # {"type": "image"} 不带 "image" 字段 —— processor 从 images= 按顺序匹配
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 图像直接以 PIL Image 列表传入，processor 内部自己做 resize/patch
        inputs = self._processor(
            text=[text],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # 只解码新生成的 token
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        return self._processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

    # ── Ollama 推理 ───────────────────────────

    def _infer_ollama(self, image_b64: str, system: str, user: str) -> str:
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user,
                    "images": [image_b64],
                },
            ],
            "stream": False,
            "options": {"num_predict": self.max_new_tokens},
        }
        resp = self._requests.post(
            f"{self._ollama_url}/api/chat", json=payload, timeout=120
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# ──────────────────────────────────────────────
# JSON 解析与容错
# ──────────────────────────────────────────────

def parse_vlm_output(raw: str, track: Track) -> Suspicion:
    """解析 VLM 输出的 JSON，容错处理"""
    raw = raw.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning(f"轨迹 {track.track_id}：VLM 未返回 JSON，原始输出：{raw[:200]}")
        return Suspicion(
            track_id=track.track_id,
            anomaly_type=AnomalyType.OK,
            frame_range=(track.start_frame, track.end_frame),
            confidence=0.0,
            reason="VLM 输出解析失败",
        )

    try:
        data: Dict[str, Any] = json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"轨迹 {track.track_id}：JSON 解析错误 {e}，原始：{raw[start:end][:200]}")
        return Suspicion(
            track_id=track.track_id,
            anomaly_type=AnomalyType.OK,
            frame_range=(track.start_frame, track.end_frame),
            confidence=0.0,
            reason="JSON 解析失败",
        )

    anomaly_str = data.get("anomaly_type", "ok")
    try:
        anomaly = AnomalyType(anomaly_str)
    except ValueError:
        anomaly = AnomalyType.OK

    fr = data.get("frame_range", [track.start_frame, track.end_frame])
    frame_range = (
        (int(fr[0]), int(fr[1]))
        if isinstance(fr, list) and len(fr) >= 2
        else (track.start_frame, track.end_frame)
    )

    return Suspicion(
        track_id=track.track_id,
        anomaly_type=anomaly,
        frame_range=frame_range,
        confidence=float(data.get("confidence", 0.0)),
        reason=str(data.get("reason", "")),
        related_track_id=data.get("related_track_id"),
    )


# ──────────────────────────────────────────────
# 主巡检器
# ──────────────────────────────────────────────

class TrackInspector:
    """
    阶段一：VLM 轨迹巡检器

    用法：
        backend = QwenVLBackend(use_ollama=True, model_name="qwen3-vl")
        inspector = TrackInspector(backend, suspicion_threshold=0.5)
        report = inspector.inspect(tracks, frames)
        print(report.summary())
    """

    def __init__(
        self,
        backend: QwenVLBackend,
        suspicion_threshold: float = 0.5,
        max_contact_frames: int = 12,
    ):
        self.backend = backend
        self.suspicion_threshold = suspicion_threshold
        self.max_contact_frames = max_contact_frames

    def inspect_one(
        self,
        track: Track,
        frames: Dict[int, np.ndarray],
    ) -> Suspicion:
        """对单条轨迹进行 VLM 分析"""
        sheet     = build_contact_sheet(track, frames, self.max_contact_frames)
        image_b64 = ndarray_to_base64(sheet)
        feats     = compute_numeric_features(track)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            track_id=track.track_id,
            numeric_features=json.dumps(feats, ensure_ascii=False, indent=2),
            max_disp=feats.get("max_displacement", "N/A"),
            max_disp_frame=feats.get("max_disp_frame", "N/A"),
            breaks=feats.get("trajectory_breaks", []),
        )

        logger.info(f"正在分析轨迹 {track.track_id}（长度 {track.length} 帧）...")
        raw_output = self.backend.infer(image_b64, SYSTEM_PROMPT, user_prompt)
        logger.debug(f"轨迹 {track.track_id} VLM 原始输出：{raw_output}")

        return parse_vlm_output(raw_output, track)

    def inspect(
        self,
        tracks: List[Track],
        frames: Dict[int, np.ndarray],
        skip_short: int = 3,
    ) -> InspectionReport:
        """
        遍历所有轨迹，生成巡检报告。

        Args:
            tracks:      所有轨迹列表
            frames:      帧字典 {frame_id: ndarray}
            skip_short:  跳过长度不足 N 帧的轨迹（默认 3）
        """
        report = InspectionReport(total_tracks=len(tracks))

        for track in tracks:
            if track.length < skip_short:
                logger.debug(f"轨迹 {track.track_id} 过短（{track.length} 帧），跳过")
                report.ok_tracks.append(track.track_id)
                continue

            suspicion = self.inspect_one(track, frames)

            if (
                suspicion.anomaly_type != AnomalyType.OK
                and suspicion.confidence >= self.suspicion_threshold
            ):
                report.suspicious.append(suspicion)
                logger.info(
                    f"[可疑] 轨迹 {track.track_id} — {suspicion.anomaly_type.value} "
                    f"置信度 {suspicion.confidence:.2f}：{suspicion.reason}"
                )
            else:
                report.ok_tracks.append(track.track_id)

        return report
