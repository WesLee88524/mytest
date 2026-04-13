from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_events(input_path: str) -> Tuple[List[dict], bool]:
    p = Path(input_path)
    if p.is_file():
        return read_jsonl(p), False
    if p.is_dir():
        events: List[dict] = []
        for jf in sorted(glob.glob(str(p / "*.jsonl"))):
            events.extend(read_jsonl(Path(jf)))
        return events, True
    raise FileNotFoundError(f"input not found: {input_path}")


def write_jsonl(path: Path, events: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def split_by_sequence(events: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for ev in events:
        groups[str(ev.get("seq_name", "__unknown__"))].append(ev)

    seq_names = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(seq_names)

    n_val_seq = max(1, int(len(seq_names) * val_ratio)) if len(seq_names) > 1 else 0
    val_seqs = set(seq_names[:n_val_seq])

    train_events, val_events = [], []
    for seq_name, seq_events in groups.items():
        if seq_name in val_seqs:
            val_events.extend(seq_events)
        else:
            train_events.extend(seq_events)

    return train_events, val_events


def split_by_sample(events: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    arr = list(events)
    rng = random.Random(seed)
    rng.shuffle(arr)
    n_val = int(len(arr) * val_ratio)
    return arr[n_val:], arr[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description="把 stage1 事件数据拆分为 train/val，避免训练和评估使用同一份数据")
    parser.add_argument("--input", required=True, help="事件 JSONL 文件或目录")
    parser.add_argument("--train-output", required=True, help="train 输出路径（.jsonl 或目录）")
    parser.add_argument("--val-output", required=True, help="val 输出路径（.jsonl 或目录）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-by",
        choices=["seq", "sample"],
        default="seq",
        help="按序列切分（推荐，防泄漏）或按样本随机切分",
    )
    args = parser.parse_args()

    events, source_is_dir = load_events(args.input)
    if not events:
        raise RuntimeError("输入为空，无法切分")

    if args.split_by == "seq":
        train_events, val_events = split_by_sequence(events, args.val_ratio, args.seed)
    else:
        train_events, val_events = split_by_sample(events, args.val_ratio, args.seed)

    if not train_events or not val_events:
        raise RuntimeError(
            f"切分后样本为空: train={len(train_events)} val={len(val_events)}，请调整 --val-ratio 或 --split-by"
        )

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)

    if source_is_dir:
        if train_output.suffix or val_output.suffix:
            raise ValueError("输入是目录时，请把 --train-output/--val-output 设为目录路径")

        train_grouped: Dict[str, List[dict]] = defaultdict(list)
        val_grouped: Dict[str, List[dict]] = defaultdict(list)
        for ev in train_events:
            train_grouped[str(ev.get("seq_name", "__unknown__"))].append(ev)
        for ev in val_events:
            val_grouped[str(ev.get("seq_name", "__unknown__"))].append(ev)

        for seq_name, items in train_grouped.items():
            write_jsonl(train_output / f"{seq_name}.jsonl", items)
        for seq_name, items in val_grouped.items():
            write_jsonl(val_output / f"{seq_name}.jsonl", items)
    else:
        write_jsonl(train_output, train_events)
        write_jsonl(val_output, val_events)

    def count_types(items: List[dict]) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for ev in items:
            out[str(ev.get("anomaly_type", "ok"))] += 1
        return dict(sorted(out.items(), key=lambda x: x[0]))

    print(f"total={len(events)} train={len(train_events)} val={len(val_events)}")
    print(f"split_by={args.split_by} seed={args.seed} val_ratio={args.val_ratio}")
    print(f"train type dist: {count_types(train_events)}")
    print(f"val   type dist: {count_types(val_events)}")
    print(f"written train -> {train_output}")
    print(f"written val   -> {val_output}")


if __name__ == "__main__":
    main()
