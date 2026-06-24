# Fine-Tuned YOLO Detector

This page documents how to use `scripts/run_models/yolo/yolo.py` to run
cross-sucking detection using a fine-tuned YOLOv26 model, producing structured
JSON event metadata for downstream evaluation and clipping.

---

## Summary

This script runs a fine-tuned YOLOv26 model frame-by-frame on source videos
from a test split CSV, detecting cross-sucking events and saving structured
metadata. Unlike the baseline script which flags frames based on simple
bounding box overlap between any two calves, this script detects
cross-sucking directly using a model fine-tuned on labeled cross-sucking clips.

The script runs **two detection approaches in a single execution**:

- **YOLO** — groups consecutive frames with detections into events using a
  buffer-based approach. Frames within `--buffer` seconds of each other are
  merged into a single event.
- **Seq-NMS** — links detections across consecutive frames using spatial IoU
  into tubes, then converts tubes into event windows. See the
  [Seq-NMS page](seq_nms.md) for more detail on the algorithm.

Both outputs are saved in separate subfolders under `--output_dir`, in the
same JSON format as `baseline.py` so `evaluation.py` works unchanged.

---

## Environment Setup

Before running this script, ensure your local workspace satisfies the
following:

- `.env` is populated with `ROOT_DIR` and `LOCAL_DIR` (used by `config.py`).
- Source videos exist under `SOURCE_VIDEOS_DIR`.
- A test split CSV exists under `data/processed/<split_name>/test.csv`,
  produced by running `scripts/split_data/split_data.py`.
- Fine-tuned YOLO model weights exist at the path specified by `--model_path`,
  produced by running `scripts/training/training_yolo.py`.

---

## Output

Results are saved to two subfolders under `--output_dir`:

```text
<output_dir>/
├── yolo/
│   ├── ch02_20250913094601_results.json   # YOLO buffer-based events
│   └── ch02_20250913094602_results.json
└── seq-nms/
    ├── ch02_20250913094601_results.json   # YOLO + Seq-NMS events
    └── ch02_20250913094602_results.json
```

Each JSON file contains event timestamps, confidence scores, and per-frame
bounding box coordinates:

```json
{
  "identifier": "ch02_20250913094601.mp4",
  "model": "weights/split_1/best.pt",
  "conf_threshold": 0.1,
  "iou_threshold": 0.0,
  "num_events": 2,
  "events": [
    {
      "start_sec": 12.4,
      "end_sec": 19.1,
      "duration_sec": 6.7,
      "avg_confidence": 0.71,
      "intersection_box": [...]
    }
  ]
}
```

---

## How it works

1. `collect_frames` opens the source video and runs fine-tuned YOLOv26
   inference on every Nth frame, collecting all per-frame detections with
   bounding box coordinates and confidence scores.
2. `extract_events` groups consecutive detections into events using a
   buffer-based approach — frames within `--buffer` seconds of each other
   are merged into a single event. Results are saved to `yolo/`.
3. `build_tubes` (from `seq_NMS.py`) links detections across frames using
   spatial IoU into consistent tubes. `suppress_weak_detections` removes low
   confidence frames within tubes. `tubes_to_events` converts tubes into
   event windows. Results are saved to `seq-nms/`.
4. `build_metadata` assembles both sets of events into structured JSON files
   and saves them to their respective output subdirectories.

---

## Usage

### Basic usage

```bash
uv run scripts/run_models/yolo/yolo.py \
    --video_path "$ROOT_DIR/raw_cross_sucking_datalog/videos/Pen2/POSTWEANING/Day3/ch02_20251104074042.mp4" \
    --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" \
    --output_dir "results/metadata/pipeline_demo"
```

### Run with frame skipping for faster processing

```bash
uv run scripts/run_models/yolo/yolo.py \
    --video_path "$ROOT_DIR/raw_cross_sucking_datalog/videos/Pen2/POSTWEANING/Day3/ch02_20251104074042.mp4" \
    --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" \
    --output_dir "results/metadata/pipeline_demo" \
    --frame_skip 30
```

### Show video during inference

```bash
uv run scripts/run_models/yolo/yolo.py \
    --video_path "$ROOT_DIR/raw_cross_sucking_datalog/videos/Pen2/POSTWEANING/Day3/ch02_20251104074042.mp4" \
    --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" \
    --output_dir "results/metadata/pipeline_demo" \
    --show_video
```

### Run with custom thresholds

```bash
uv run scripts/run_models/yolo/yolo.py \
    --video_path "$ROOT_DIR/raw_cross_sucking_datalog/videos/Pen2/POSTWEANING/Day3/ch02_20251104074042.mp4" \
    --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" \
    --output_dir "results/metadata/pipeline_demo" \
    --conf_threshold 0.25 \
    --iou_threshold 0.3 \
    --min_duration 1.0
```

---

## Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video_path` | str | required | Path to input source video file |
| `--model_path` | str | required | Path to fine-tuned YOLO weights file |
| `--output_dir` | str | required | Directory to save metadata output |
| `--conf_threshold` | float | `0.1` | Minimum YOLO detection confidence to keep a box |
| `--iou_threshold` | float | `0.0` | Minimum spatial IoU to link boxes across frames into tubes |
| `--min_duration` | float | `0.0` | Minimum event duration in seconds |
| `--frame_skip` | int | `10` | Process every Nth frame |
| `--buffer` | int | `60` | Seconds without detection before ending a YOLO event |
| `--target_class` | str | `cross-sucking` | Target class for detection |
| `--show_video` | flag | `False` | Display annotated video during inference |

---

## Function Reference

::: scripts.run_models.yolo.yolo
    options:
        show_source: false
        show_root_heading: true
        members:
            - extract_events
            - collect_frames
            - build_metadata
            - run_models