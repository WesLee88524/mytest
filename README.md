# MOT 轨迹巡检器（阶段一）

## 文件结构

```
mot_inspector/
├── data_types.py          # 数据结构定义（Track, Box, Suspicion 等）
├── track_preprocessor.py  # 拼贴图生成 + 数值特征计算
├── vlm_inspector.py       # VLM 巡检器核心（Qwen-VL 后端）
└── run_inspector.py       # 入口脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install opencv-python-headless numpy transformers torch
# 如果使用 Ollama 后端（推荐本地部署）：
# https://ollama.com/library/qwen3-vl
ollama pull qwen3-vl
```

### 2. Demo 模式（合成数据，无需真实视频）

```bash
cd mot_inspector
python run_inspector.py --demo --backend ollama
```

### 3. 真实数据

```bash
python run_inspector.py \
    --frames  /data/MOT17/train/MOT17-02/img1/ \
    --mot     /data/results/MOT17-02.txt \
    --output  suspicions.json \
    --backend ollama \
    --model   qwen3-vl \
    --threshold 0.5
```

MOT 结果文件格式（MOTChallenge 标准）：
```
frame_id, track_id, x, y, w, h, conf, -1, -1, -1
1, 1, 100.5, 200.3, 50.0, 120.0, 0.95, -1, -1, -1
```

## 输出格式

`suspicions.json`：
```json
{
  "total_tracks": 42,
  "suspicious_count": 5,
  "ok_count": 37,
  "suspicious": [
    {
      "track_id": 7,
      "anomaly_type": "id_switch",
      "frame_range": [120, 145],
      "confidence": 0.87,
      "reason": "目标在第132帧外观突变，体型和颜色与之前不一致",
      "related_track_id": 12
    }
  ]
}
```

## 设计要点

### 拼贴图策略
- 每条轨迹均匀采样最多 12 帧，拼成横向接触表
- 每帧裁剪目标区域（含25%padding以保留上下文）
- 帧号标注在每张图左上角

### 数值特征辅助
VLM 同时接收文本形式的数值特征：
- `max_displacement`：最大帧间位移（检测 drift）
- `trajectory_breaks`：帧间隔超过5帧的中断点（检测 miss）
- `min_iou`：最小连续帧 IoU（检测突变）
- `jump_frames`：位移突变帧列表

### 阈值建议
| 场景 | 推荐阈值 |
|------|---------|
| 高召回（宁可多查） | 0.3 |
| 平衡 | 0.5 |
| 高精度（减少误报） | 0.7 |

## 与阶段二的接口

`InspectionReport.suspicious` 列表直接传入阶段二（LLM 工具调用调查器），
每个 `Suspicion` 提供：`track_id`、`anomaly_type`、`frame_range`、`confidence`。

## 训练/评估数据切分（避免 val 泄漏）

如果你当前用 `DanceTrack val` 同时做训练和评估，会导致指标失真（数据泄漏）。
建议先把 `stage1_events_val` 切成 `train/val`：

```bash
python split_stage1_events.py \
  --input ./stage1_events_val \
  --train-output ./stage1_events_val_train \
  --val-output ./stage1_events_val_holdout \
  --split-by seq \
  --val-ratio 0.2 \
  --seed 42
```

然后训练和评估分开使用：

```bash
# 训练只用 train 子集
nohup torchrun --nproc_per_node=4 train_stage1_sft.py \
  --input ./stage1_events_val_train \
  --model /public/home/lyh_npu/models/Qwen2.5-VL-7B-Instruct \
  --output-dir exps/val_v2 \
  --use-lora --bf16 \
  --image-field contact_sheet_path \
  --image-root ./viz_val \
  --require-image \
  --epochs 3 > log/val_train_stage1_sft2.log 2>&1 &

# 评估只用 holdout 子集
python eval_stage1_detector.py \
  --model exps/val_v2 \
  --input ./stage1_events_val_holdout \
  --output eval_results_holdout.json
```
