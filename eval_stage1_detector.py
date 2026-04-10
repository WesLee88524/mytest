"""
阶段一 SFT 模型评估脚本

指标：
  - Type-F1（五分类，macro + per-class）
  - Range-IoU（帧段定位，预测 vs 标注）
  - Confidence Calibration（ECE）
  - JSON 合法率

用法：
  python eval_stage1_detector.py \
    --model /public/home/lyh_npu/code/RT-MOTRv2/data/DanceTrack/val \
    --input /public/home/lyh_npu/code/PostMOT/stage1_events_val \
    --output eval_results.json
"""
from __future__ import annotations

import argparse
import json
import glob
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VALID_TYPES = {"id_switch", "miss", "drift", "fragment", "false_positive", "ok"}
SYSTEM_INST = (
    "你是多目标跟踪(MOT)质检模型。"
    "给定轨迹事件上下文，请输出严格 JSON："
    '{"anomaly_type":"id_switch|miss|drift|fragment|false_positive|ok",'
    '"frame_range":[start,end],'
    '"confidence":0.0-1.0,'
    '"related_track_id":null或整数,'
    '"reason":"简短中文说明"}'
)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def read_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                data.append(json.loads(ln))
    return data


def load_events(input_path: str) -> List[dict]:
    p = Path(input_path)
    if p.is_file():
        return read_jsonl(str(p))
    items: List[dict] = []
    for jf in sorted(glob.glob(str(p / "*.jsonl"))):
        items.extend(read_jsonl(jf))
    return items


def event_to_prompt(ev: dict) -> str:
    user = {
        "seq_name": ev.get("seq_name", ""),
        "track_id": ev.get("track_id"),
        "gt_id": ev.get("gt_id"),
        "frame_range": ev.get("frame_range", [0, 0]),
        "meta": ev.get("meta", {}),
        "image_path": ev.get("contact_sheet_path"),
    }
    return (
        f"<|system|>\n{SYSTEM_INST}\n"
        f"<|user|>\n{json.dumps(user, ensure_ascii=False)}\n"
        f"<|assistant|>\n"
    )


# ── 推理 ──────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    import torch
    from transformers import AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def run_inference(prompt: str, tokenizer, model, max_new_tokens: int = 128) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = out[:, inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    del inputs, out, generated
    torch.cuda.empty_cache()
    return result


def parse_output(raw: str) -> Tuple[Optional[dict], bool]:
    """返回 (parsed_dict_or_None, json_valid)"""
    raw = raw.strip()
    # 尝试提取第一个 {...}
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end + 1]), True
        except json.JSONDecodeError:
            pass
    return None, False


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def range_iou(pred: List[int], gt: List[int]) -> float:
    s1, e1 = pred[0], pred[1]
    s2, e2 = gt[0], gt[1]
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = max(e1, e2) - min(s1, s2) + 1
    return inter / union if union > 0 else 0.0


def compute_ece(confidences: List[float], corrects: List[bool], n_bins: int = 10) -> float:
    bins = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, corrects):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))
    ece = 0.0
    n = len(confidences)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(x[0] for x in b) / len(b)
        avg_acc = sum(x[1] for x in b) / len(b)
        ece += len(b) / n * abs(avg_conf - avg_acc)
    return ece


def compute_f1_per_class(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, dict]:
    results = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": tp + fn}
    macro_f1 = sum(v["f1"] for v in results.values()) / len(labels)
    results["macro"] = {"f1": round(macro_f1, 4)}
    return results


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SFT checkpoint 路径")
    parser.add_argument("--input", required=True, help="val 事件 JSONL 文件或目录")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--max-samples", type=int, default=None, help="限制评估样本数（调试用）")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    events = load_events(args.input)
    if args.max_samples:
        events = events[:args.max_samples]
    print(f"评估样本数: {len(events)}")

    tokenizer, model = load_model(args.model)

    y_true, y_pred = [], []
    range_ious = []
    confidences, corrects = [], []
    json_valid_count = 0
    raw_outputs = []

    for i, ev in enumerate(events):
        prompt = event_to_prompt(ev)
        raw = run_inference(prompt, tokenizer, model, max_new_tokens=args.max_new_tokens)
        parsed, valid = parse_output(raw)

        gt_type = ev.get("anomaly_type", "ok")
        gt_range = ev.get("frame_range", [0, 0])

        if valid:
            json_valid_count += 1
            pred_type = parsed.get("anomaly_type", "ok")
            pred_range = parsed.get("frame_range", gt_range)
            pred_conf = float(parsed.get("confidence", 0.5))
        else:
            pred_type = "ok"
            pred_range = gt_range
            pred_conf = 0.0

        y_true.append(gt_type)
        y_pred.append(pred_type)
        range_ious.append(range_iou(pred_range, gt_range))
        confidences.append(pred_conf)
        corrects.append(pred_type == gt_type)

        raw_outputs.append({"gt": ev, "raw": raw, "parsed": parsed, "valid": valid})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(events)}] json_valid={json_valid_count}/{i+1} "
                  f"acc={sum(corrects)/(i+1):.3f} range_iou={sum(range_ious)/(i+1):.3f}")

    labels = sorted(VALID_TYPES)
    f1_results = compute_f1_per_class(y_true, y_pred, labels)
    ece = compute_ece(confidences, corrects)
    avg_range_iou = sum(range_ious) / len(range_ious) if range_ious else 0.0
    json_valid_rate = json_valid_count / len(events) if events else 0.0
    accuracy = sum(corrects) / len(corrects) if corrects else 0.0

    results = {
        "n_samples": len(events),
        "accuracy": round(accuracy, 4),
        "macro_f1": f1_results["macro"]["f1"],
        "per_class_f1": {k: v for k, v in f1_results.items() if k != "macro"},
        "avg_range_iou": round(avg_range_iou, 4),
        "ece": round(ece, 4),
        "json_valid_rate": round(json_valid_rate, 4),
    }

    print("\n===== 评估结果 =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    out = {
        "metrics": results,
        "samples": raw_outputs,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
