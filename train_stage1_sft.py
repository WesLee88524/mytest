"""
阶段二（SFT）第一版：
基于 build_rl_dataset.py 产出的 JSONL 训练“阶段一判断器”。

输入样本（每行 JSON）示例：
{
  "seq_name": "dancetrack0003",
  "track_id": 250,
  "anomaly_type": "drift",
  "frame_range": [367, 903],
  "confidence": 0.75,
  "meta": {...}
}

本脚本采用 instruction tuning 方式，将结构化事件转成
“指令 + 上下文 -> JSON 输出”的监督样本进行 SFT。

注意：
  - v1 默认是 text-only SFT baseline；
  - 若要做真正 VLM SFT，请在事件中提供图像字段（如 contact_sheet_path），
    并用 --require-image 强制筛选，仅训练带图样本。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers import Qwen2_5_VLForConditionalGeneration


SYSTEM_INST = (
    "你是多目标跟踪(MOT)质检模型。"
    "给定轨迹事件上下文，请输出严格 JSON："
    '{"anomaly_type":"id_switch|miss|drift|fragment|false_positive|ok",'
    '"frame_range":[start,end],'
    '"confidence":0.0-1.0,'
    '"related_track_id":null或整数,'
    '"reason":"简短中文说明"}'
)


def read_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            data.append(json.loads(ln))
    return data


def load_events(input_path: str) -> List[dict]:
    p = Path(input_path)
    if p.is_file():
        return read_jsonl(str(p))
    if p.is_dir():
        items: List[dict] = []
        for jf in sorted(glob.glob(str(p / "*.jsonl"))):
            items.extend(read_jsonl(jf))
        return items
    raise FileNotFoundError(f"input not found: {input_path}")


def enrich_image_path_from_root(
    events: List[dict],
    image_root: str,
    image_field: str,
) -> int:
    """
    尝试从 stage1 可视化目录自动回填 image_field。
    默认按以下模式查找：
      {image_root}/{seq_name}/stage1/*_track{track_id}_*_{start}_{end}.jpg
    """
    if not image_root:
        return 0
    filled = 0
    for ev in events:
        if ev.get(image_field):
            continue
        seq = ev.get("seq_name")
        tid = ev.get("track_id")
        fr = ev.get("frame_range", [None, None])
        if seq is None or tid is None or fr[0] is None or fr[1] is None:
            continue
        s, e = fr[0], fr[1]
        patt_exact = os.path.join(
            image_root, str(seq), "stage1", f"*_track{tid}_*_{s}_{e}.jpg"
        )
        cand = sorted(glob.glob(patt_exact))
        if not cand:
            patt_relax = os.path.join(
                image_root, str(seq), "stage1", f"*_track{tid}_*.jpg"
            )
            cand = sorted(glob.glob(patt_relax))
        if cand:
            ev[image_field] = cand[0]
            filled += 1
    return filled


def event_to_example(ev: dict, image_field: str) -> Dict[str, str]:
    anomaly_type = ev.get("anomaly_type", "ok")
    frame_range = ev.get("frame_range", [0, 0])
    conf = float(ev.get("confidence", 0.5))
    track_id = ev.get("track_id", None)
    seq_name = ev.get("seq_name", "")
    gt_id = ev.get("gt_id", None)
    meta = ev.get("meta", {})
    image_path = ev.get(image_field, None)

    user = {
        "seq_name": seq_name,
        "track_id": track_id,
        "gt_id": gt_id,
        "frame_range": frame_range,
        "meta": meta,
        "image_path": image_path,
    }
    # 第一版：related_track_id 无法从自动标注稳定恢复，先置 null
    assistant = {
        "anomaly_type": anomaly_type,
        "frame_range": frame_range,
        "confidence": round(conf, 4),
        "related_track_id": None,
        "reason": "自动标注样本",
    }

    prompt = (
        f"<|system|>\n{SYSTEM_INST}\n"
        f"<|user|>\n{json.dumps(user, ensure_ascii=False)}\n"
        f"<|assistant|>\n"
    )
    answer = json.dumps(assistant, ensure_ascii=False)
    return {"prompt": prompt, "answer": answer}


def balance_events(events: List[dict], seed: int, max_ratio: int = 5) -> List[dict]:
    """
    对多数类欠采样，使最多类样本数 <= min_class_count * max_ratio。
    少数类保持不变（不过采样，避免重复样本影响泛化）。
    """
    from collections import defaultdict
    by_type: dict = defaultdict(list)
    for ev in events:
        by_type[ev.get("anomaly_type", "ok")].append(ev)
    min_count = min(len(v) for v in by_type.values())
    cap = max(min_count * max_ratio, 1)
    rng = random.Random(seed)
    result = []
    for evs in by_type.values():
        if len(evs) > cap:
            result.extend(rng.sample(evs, cap))
        else:
            result.extend(evs)
    rng.shuffle(result)
    return result



    rng = random.Random(seed)
    arr = items[:]
    rng.shuffle(arr)
    n_val = int(len(arr) * val_ratio)
    val = arr[:n_val]
    train = arr[n_val:]
    return train, val


@dataclass
class EncodedDataset:
    train: Dataset
    val: Dataset


def build_hf_dataset(
    train_items: List[dict],
    val_items: List[dict],
    tokenizer,
    max_length: int,
) -> EncodedDataset:
    class TokenizedDataset(Dataset):
        def __init__(self, items: List[dict]):
            self.samples = []
            for ex in items:
                full_text = ex["prompt"] + ex["answer"] + tokenizer.eos_token
                tok = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                self.samples.append(tok)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    train_ds = TokenizedDataset(train_items)
    val_ds = TokenizedDataset(val_items)
    return EncodedDataset(train=train_ds, val=val_ds)


def maybe_apply_lora(model, use_lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    if not use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError("启用 --use-lora 需要安装 peft") from e

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser(description="训练阶段一 SFT（第一版）")
    parser.add_argument("--input", type=str, required=True, help="事件 JSONL 文件或目录")
    parser.add_argument("--model", type=str, required=True, help="基础模型名/路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--image-field", type=str, default="contact_sheet_path",
                        help="事件中图像路径字段名（用于 VLM 样本）")
    parser.add_argument("--require-image", action="store_true",
                        help="只保留包含 image-field 的样本")
    parser.add_argument("--image-root", type=str, default=None,
                        help="可选：从该目录自动回填图像路径（如 /path/to/viz_root）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    events = load_events(args.input)
    if not events:
        raise RuntimeError("输入事件为空，无法训练")

    if args.image_root:
        n = enrich_image_path_from_root(
            events=events,
            image_root=args.image_root,
            image_field=args.image_field,
        )
        print(f"auto-filled image paths: {n} (field={args.image_field})")

    if args.require_image:
        events = [ev for ev in events if ev.get(args.image_field)]
        if not events:
            raise RuntimeError(f"启用 --require-image 后无样本，请检查字段 {args.image_field}")
        print(f"after image filter ({args.image_field}): {len(events)}")
    else:
        print("warning: 当前为 text-only SFT baseline；建议后续切换到带图样本训练。")

    events = balance_events(events, seed=args.seed)
    print(f"after balancing: {len(events)}")
    from collections import Counter
    print("class dist:", dict(Counter(ev["anomaly_type"] for ev in events)))

    examples = [event_to_example(ev, image_field=args.image_field) for ev in events]
    train_items, val_items = train_val_split(examples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"events={len(events)} train={len(train_items)} val={len(val_items)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )
    model = maybe_apply_lora(
        model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    ds = build_hf_dataset(
        train_items=train_items,
        val_items=val_items if val_items else train_items[: min(128, len(train_items))],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=20,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        fsdp="full_shard auto_wrap",
        fsdp_config={"fsdp_transformer_layer_cls_to_wrap": "Qwen2_5_VLDecoderLayer"},
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds.train,
        eval_dataset=ds.val,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
