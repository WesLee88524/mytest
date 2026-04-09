"""
阶段二：LLM 工具调用调查器

对阶段一输出的每个可疑片段，运行 ReAct 循环：
  Thought → Action（调用工具）→ Observation → ... → 最终诊断

使用与阶段一相同的 VLM 后端（Qwen2.5-VL），
图像和文本一起传入，模型可以"看着"工具返回的图像做推理。
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_types import AnomalyType, Suspicion, Track
from investigation_tools import TOOL_DEFINITIONS, ToolExecutor, ToolResult
from vlm_inspector import QwenVLBackend, ndarray_to_base64, base64_to_pil

logger = logging.getLogger(__name__)

MAX_REACT_STEPS = 6  # 最多调用工具次数，防止死循环


# ──────────────────────────────────────────────
# 诊断结果
# ──────────────────────────────────────────────

@dataclass
class Investigation:
    """单个可疑片段的调查结论"""
    track_id: int
    initial_suspicion: Suspicion          # 阶段一给出的初步怀疑
    confirmed: bool                       # 是否确认存在错误
    anomaly_type: AnomalyType             # 确认的错误类型
    frame_range: Tuple[int, int]          # 精确的问题帧范围
    confidence: float                     # 最终置信度
    conclusion: str                       # 自然语言结论
    tool_calls: List[str] = field(default_factory=list)   # 调用过的工具序列
    related_track_id: Optional[int] = None


@dataclass
class InvestigationReport:
    """阶段二完整报告"""
    investigations: List[Investigation] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"调查报告（共 {len(self.investigations)} 条可疑轨迹）", "=" * 50]
        for inv in self.investigations:
            status = "✓ 确认错误" if inv.confirmed else "✗ 误报"
            lines.append(
                f"轨迹 {inv.track_id} [{status}] {inv.anomaly_type.value} "
                f"帧{inv.frame_range[0]}-{inv.frame_range[1]} "
                f"置信度{inv.confidence:.2f}"
            )
            lines.append(f"  结论：{inv.conclusion}")
            lines.append(f"  调查步骤：{' → '.join(inv.tool_calls)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "investigations": [
                {
                    "track_id":          inv.track_id,
                    "confirmed":         inv.confirmed,
                    "anomaly_type":      inv.anomaly_type.value,
                    "frame_range":       list(inv.frame_range),
                    "confidence":        inv.confidence,
                    "conclusion":        inv.conclusion,
                    "tool_calls":        inv.tool_calls,
                    "related_track_id":  inv.related_track_id,
                }
                for inv in self.investigations
            ]
        }


# ──────────────────────────────────────────────
# Prompt 模板
# ──────────────────────────────────────────────

INVESTIGATOR_SYSTEM = """你是一个多目标跟踪（MOT）错误调查专家。
你的任务是对一条可疑轨迹进行深入调查，确认或排除跟踪错误。

你可以调用以下工具：
- replay_segment：回放一段帧序列，逐帧观察目标
- zoom_region：放大某一帧的目标区域，看清细节
- enhance_image：增强图像质量（contrast/deblur/edge/clahe）
- compare_appearances：对比两帧的目标外观，检测ID切换
- check_nearby_tracks：查找附近其他轨迹，分析遮挡情况

调查策略：
- id_switch 嫌疑：先用 compare_appearances 对比突变前后，再用 zoom_region 细看
- drift 嫌疑：先用 replay_segment 回放突变片段，再用 check_nearby_tracks 看是否有遮挡
- miss 嫌疑：用 replay_segment 看中断前后，再用 check_nearby_tracks 看目标去了哪里

每次只调用一个工具。工具调用格式（严格 JSON）：
{"tool": "工具名", "params": {参数}}

调查完成后，输出最终结论（严格 JSON）：
{"done": true, "confirmed": true或false, "anomaly_type": "类型", "frame_range": [start, end], "confidence": 0.0-1.0, "conclusion": "结论说明", "related_track_id": null或ID}

每次回复只能是工具调用 JSON 或最终结论 JSON，不要输出任何其他文字。"""

INVESTIGATOR_INIT_TEMPLATE = """请调查以下可疑轨迹：

轨迹 ID：{track_id}
初步怀疑类型：{anomaly_type}
可疑帧范围：{start_frame} - {end_frame}
阶段一理由：{reason}
置信度：{confidence}

轨迹数值特征：
{numeric_features}

请开始调查。"""


# ──────────────────────────────────────────────
# ReAct 调查循环
# ──────────────────────────────────────────────

class LLMInvestigator:
    """
    阶段二：LLM 工具调用调查器

    用法：
        investigator = LLMInvestigator(backend, executor)
        report = investigator.investigate_all(suspicions, tracks, frames)
        print(report.summary())
    """

    def __init__(
        self,
        backend: QwenVLBackend,
        executor: ToolExecutor,
        max_steps: int = MAX_REACT_STEPS,
    ):
        self.backend = backend
        self.executor = executor
        self.max_steps = max_steps

    def _call_vlm(
        self,
        messages: List[Dict],
    ) -> str:
        """
        调用 VLM，支持多轮对话历史。
        messages 格式：[{"role": "user/assistant", "content": str或list, "images_b64": [...]}]
        """
        from transformers import AutoProcessor
        import torch
        from PIL import Image
        from io import BytesIO
        import base64

        if self.backend.use_ollama:
            return self._call_ollama(messages)
        return self._call_hf(messages)

    def _call_hf(self, messages: List[Dict]) -> str:
        import torch

        hf_messages = [{"role": "system", "content": INVESTIGATOR_SYSTEM}]
        all_pil_images = []
        max_images = getattr(self.backend, "max_investigator_images", 12)

        for msg in messages:
            role = msg["role"]
            images_b64 = msg.get("images_b64", [])
            text = msg.get("content", "")

            if images_b64:
                content = []
                for b64 in images_b64:
                    if len(all_pil_images) >= max_images:
                        break
                    content.append({"type": "image"})
                    pil = base64_to_pil(b64)
                    pil = self._sanitize_pil_for_qwen(pil)
                    all_pil_images.append(pil)
                content.append({"type": "text", "text": text})
            else:
                content = text

            hf_messages.append({"role": role, "content": content})

        text_input = self.backend._processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.backend._processor(
            text=[text_input],
            images=all_pil_images if all_pil_images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.backend._model.device)

        max_new_tokens = getattr(self.backend, "max_new_tokens", 96)
        with torch.no_grad():
            output_ids = self.backend._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        result = self.backend._processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        del inputs, output_ids, generated
        torch.cuda.empty_cache()
        return result

    def _sanitize_pil_for_qwen(self, pil_img):
        """
        避免 Qwen2-VL 的 aspect ratio 限制（<200）和过大分辨率导致预处理报错。
        """
        from PIL import Image

        w, h = pil_img.size
        if w <= 0 or h <= 0:
            return pil_img

        # 控制最长边
        max_side = 560
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            pil_img = pil_img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.BICUBIC,
            )
            w, h = pil_img.size

        # 控制极端长宽比
        ratio = max(w / h, h / w)
        if ratio > 120:
            if w > h:
                target_h = max(1, int(w / 120.0))
                canvas = Image.new("RGB", (w, target_h), (240, 240, 240))
                y = (target_h - h) // 2
                canvas.paste(pil_img, (0, y))
            else:
                target_w = max(1, int(h / 120.0))
                canvas = Image.new("RGB", (target_w, h), (240, 240, 240))
                x = (target_w - w) // 2
                canvas.paste(pil_img, (x, 0))
            pil_img = canvas

        return pil_img

    def _call_ollama(self, messages: List[Dict]) -> str:
        import requests

        ollama_messages = [{"role": "system", "content": INVESTIGATOR_SYSTEM}]
        for msg in messages:
            ollama_msg = {"role": msg["role"], "content": msg.get("content", "")}
            if msg.get("images_b64"):
                ollama_msg["images"] = msg["images_b64"]
            ollama_messages.append(ollama_msg)

        payload = {
            "model": self.backend._model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {"num_predict": 300},
        }
        resp = self.backend._requests.post(
            f"{self.backend._ollama_url}/api/chat",
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _parse_llm_response(self, raw: str) -> Optional[Dict]:
        """从 LLM 输出中提取 JSON"""
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None

    def investigate_one(
        self,
        suspicion: Suspicion,
        tracks: Dict[int, Track],
        frames: Dict[int, np.ndarray],
    ) -> Investigation:
        """对单个可疑片段运行 ReAct 调查循环"""
        from track_preprocessor import compute_numeric_features

        track = tracks.get(suspicion.track_id)
        if track is None:
            return Investigation(
                track_id=suspicion.track_id,
                initial_suspicion=suspicion,
                confirmed=False,
                anomaly_type=AnomalyType.OK,
                frame_range=suspicion.frame_range,
                confidence=0.0,
                conclusion="轨迹数据不存在，无法调查",
            )

        feats = compute_numeric_features(track)
        init_text = INVESTIGATOR_INIT_TEMPLATE.format(
            track_id=suspicion.track_id,
            anomaly_type=suspicion.anomaly_type.value,
            start_frame=suspicion.frame_range[0],
            end_frame=suspicion.frame_range[1],
            reason=suspicion.reason,
            confidence=suspicion.confidence,
            numeric_features=json.dumps(feats, ensure_ascii=False, indent=2),
        )

        # 对话历史
        history: List[Dict] = [{"role": "user", "content": init_text}]
        tool_calls_log: List[str] = []

        logger.info(f"开始调查轨迹 {suspicion.track_id}（{suspicion.anomaly_type.value}）")

        for step in range(self.max_steps):
            raw = self._call_vlm(history)
            logger.debug(f"  步骤 {step + 1} LLM 输出：{raw[:200]}")

            parsed = self._parse_llm_response(raw)
            if parsed is None:
                logger.warning(f"  步骤 {step + 1}：LLM 输出解析失败，终止调查")
                break

            # 最终结论
            if parsed.get("done"):
                anomaly_str = parsed.get("anomaly_type", "ok")
                try:
                    anomaly = AnomalyType(anomaly_str)
                except ValueError:
                    anomaly = suspicion.anomaly_type

                fr = parsed.get("frame_range", list(suspicion.frame_range))
                return Investigation(
                    track_id=suspicion.track_id,
                    initial_suspicion=suspicion,
                    confirmed=bool(parsed.get("confirmed", False)),
                    anomaly_type=anomaly,
                    frame_range=(int(fr[0]), int(fr[1])),
                    confidence=float(parsed.get("confidence", 0.5)),
                    conclusion=str(parsed.get("conclusion", "")),
                    tool_calls=tool_calls_log,
                    related_track_id=parsed.get("related_track_id"),
                )

            # 工具调用
            tool_name = parsed.get("tool")
            params = parsed.get("params", {})
            if not tool_name:
                logger.warning(f"  步骤 {step + 1}：未识别到工具调用，终止")
                break

            logger.info(f"  步骤 {step + 1}：调用 {tool_name}({params})")
            tool_calls_log.append(tool_name)

            # 把 LLM 的工具调用加入历史
            history.append({"role": "assistant", "content": raw})

            # 执行工具
            result: ToolResult = self.executor.execute(tool_name, params,
                                                             fallback_track_id=suspicion.track_id)
            logger.info(f"  工具结果：{result.text[:100]}")

            # 把工具结果（含图像）加入历史
            obs_msg = {
                "role": "user",
                "content": f"工具执行结果：\n{result.to_llm_text()}\n\n请继续调查，或输出最终结论。",
                "images_b64": result.images_b64,
            }
            history.append(obs_msg)

        # 超出步数限制，用最后的信息给出默认结论
        logger.warning(f"轨迹 {suspicion.track_id} 达到最大调查步数，使用初始判断")
        return Investigation(
            track_id=suspicion.track_id,
            initial_suspicion=suspicion,
            confirmed=suspicion.confidence >= 0.7,
            anomaly_type=suspicion.anomaly_type,
            frame_range=suspicion.frame_range,
            confidence=suspicion.confidence * 0.8,
            conclusion=f"调查步数耗尽，沿用阶段一结论：{suspicion.reason}",
            tool_calls=tool_calls_log,
        )

    def investigate_all(
        self,
        suspicions: List[Suspicion],
        tracks: Dict[int, Track],
        frames: Dict[int, np.ndarray],
    ) -> InvestigationReport:
        report = InvestigationReport()
        for i, sus in enumerate(suspicions):
            logger.info(f"[{i+1}/{len(suspicions)}] 调查轨迹 {sus.track_id}")
            inv = self.investigate_one(sus, tracks, frames)
            report.investigations.append(inv)
            status = "确认" if inv.confirmed else "误报"
            logger.info(f"  → {status}：{inv.conclusion}")
        return report
