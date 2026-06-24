# Seq-NMS Cross-Sucking Detector

This page documents how to use `scripts/run_models/seq-NMS/seq_NMS.py` to
run cross-sucking detection using a fine-tuned YOLOv26 model with Sequential
Non-Maximum Suppression (Seq-NMS) for temporal linking.

---

## Summary

This script provides a standalone implementation of the YOLOv26 + Seq-NMS
detection pipeline. While `yolo.py` runs both YOLO and Seq-NMS together on
a full test split CSV, this script runs Seq-NMS only on a single video file,
making it useful for testing and debugging on individual videos.

The key difference from the baseline and plain YOLO approaches is temporal
linking via Seq-NMS:

1. Fine-tuned YOLOv26 detects cross-sucking bounding boxes **per frame**
2. Seq-NMS links detections across consecutive frames into **tubes** based
   on spatial IoU overlap
3. Weak detections within tubes are suppressed
4. Tubes are converted into event windows with defined start and end times

The output JSON format matches `baseline.py` so `evaluation.py` and
`clipping.py` work unchanged.

---

## Environment Setup

Before running this script, ensure your local workspace satisfies the
following:

- `.env` is populated with `ROOT_DIR` and `LOCAL_DIR` (used by `config.py`).
- Fine-tuned YOLO model weights exist at the path specified by `--model_path`,
  produced by running `scripts/training/training_yolo.py`.

---

## Output

Results are saved to `--output_dir`:

```text
<output_dir>/
└── <video_name>_results.json
```

```json
{
  "identifier": "ch02_20251104074042.mp4",
  "model": "weights/split_1/best.pt",
  "conf_threshold": 0.15,
  "iou_threshold": 0.3,
  "min_duration_sec": 1.0,
  "frame_skip": 30,
  "fps": 30.0,
  "total_frames": 97200,
  "cross_sucking_detected": true,
  "num_events": 3,
  "events": [
    {
      "start_sec": 154.3,
      "end_sec": 167.8,
      "duration_sec": 13.5,
      "avg_confidence": 0.612,
      "intersection_box": [...]
    }
  ]
}
```

---

## How it works

1. `run_seq_nms_detection` loads the fine-tuned YOLOv26 model and opens
   the source video.
2. For every Nth frame (controlled by `--frame_skip`), YOLO inference is
   run and all detections above `--conf_threshold` are collected.
3. `build_tubes` links detections across consecutive frames using spatial
   IoU — detections in adjacent frames that overlap above `--iou_threshold`
   are grouped into the same tube.
4. `suppress_weak_detections` removes frames within each tube where
   confidence is below `--conf_threshold`.
5. `tubes_to_events` converts surviving tubes into event windows, discarding
   any tube shorter than `--min_duration`.
6. Results are saved as a structured JSON file.

### Edge Cases

**"Target class not found"**
    If the fine-tuned model does not contain the `cross-sucking` class (e.g.
    when testing with pretrained weights), the script falls back to the
    `cross-sucking` class if available, or raises a `ValueError` if neither
    class is found.

**"No detections on a frame"**
    Frames with no detections above `--conf_threshold` are stored as empty
    lists. Seq-NMS treats these as breaks between tubes — any active tube is
    closed when an empty frame is encountered.

---

## Core Pipeline Concepts

### Tubes

A tube is a sequence of bounding boxes linked across consecutive frames that
all belong to the same detected interaction. Seq-NMS builds tubes by checking
whether each new detection overlaps with the last box in any active tube above
`--iou_threshold`. If it does, the tube is extended. If not, a new tube is
started.

### Why Seq-NMS over baseline

The baseline script flags any frame where two calves overlap sufficiently,
treating each frame independently. This produces fragmented detections that
don't naturally group into events. Seq-NMS solves this by explicitly linking
detections through time, producing cleaner event boundaries and filtering out
brief spurious detections via `--min_duration`.

### Limitation

Seq-NMS links detections based purely on spatial overlap between consecutive
frames. If two calves shift position significantly during a cross-sucking
interaction, the bounding box location may change enough that Seq-NMS breaks
the tube into two separate events, inflating event counts and reducing measured
duration accuracy.

---

## Usage

### Basic usage

```bash
uv run scripts/run_models/seq-NMS/seq_NMS.py \
    --video_path "$ROOT_DIR/raw_cross_sucking_datalog/videos/Pen2/POSTWEANING/Day3/ch02_20251104074042.mp4" \
    --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" \
    --output_dir "results/metadata/pipeline_demo/seq-nms"
```

### Run on a sample clip with video display

```bash
uv run scripts/run_models/seq-NMS/seq_NMS.py \
    --video_path "sample_videos/cross_sucking_clip_sample/CS_0276_WEAN_d1_p2_cowT_16102025_ch02-20251016124717_19033_19045.mp4" \
    --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" \
    --output_dir "results/demo" \
    --show_video
```

### Run with custom thresholds

```bash
uv run scripts/run_models/seq-NMS/seq_NMS.py \
    --video_path "$ROOT_DIR/raw_cross_sucking_datalog/videos/Pen2/POSTWEANING/Day3/ch02_20251104074042.mp4" \
    --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" \
    --output_dir "results/metadata/pipeline_demo/seq-nms" \
    --conf_threshold 0.25 \
    --iou_threshold 0.3 \
    --min_duration 1.0 \
    --frame_skip 30
```

---

## Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video_path` | str | required | Path to input video file |
| `--model_path` | str | required | Path to fine-tuned YOLO weights file |
| `--output_dir` | str | required | Directory to save metadata output |
| `--conf_threshold` | float | `0.15` | Minimum YOLO detection confidence to keep a box |
| `--iou_threshold` | float | `0.3` | Minimum spatial IoU to link detections into tubes |
| `--min_duration` | float | `1.0` | Minimum event duration in seconds |
| `--frame_skip` | int | `30` | Process every Nth frame |
| `--target_class` | str | `cross-sucking` | Target class for detection |
| `--show_video` | flag | `False` | Display annotated video during inference |

---

## Function Reference

::: scripts.run_models.seq-NMS.seq_NMS
    options:
        show_source: false
        show_root_heading: true
        members:
            - compute_iou
            - build_tubes
            - suppress_weak_detections
            - tubes_to_events
            - run_seq_nms_detection