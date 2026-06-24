# CS Detection Evaluation

This page documents how to use `scripts/predict_bboxes.py` to evaluate the
spatial accuracy of the fine-tuned YOLO model's cross-sucking detection,
independently of event-level temporal matching.

---

## Summary

While `evaluation.py` measures whether the model correctly identifies *when*
cross-sucking events occur, `predict_bboxes.py` measures whether the model
correctly identifies *which frames* contain cross-sucking and *where* in the
frame it is occurring. It does this by:

1. Running the fine-tuned YOLO model frame-by-frame on each video clip
2. Saving predicted bounding boxes as `.txt` files in YOLO normalized format,
   mirroring the CVAT ground truth annotation folder structure
3. Optionally comparing predicted `.txt` files against CVAT ground truth zip
   annotations to compute frame-level precision, recall, F1, F2, and average
   bounding box IoU

Frames with no detections are skipped — no empty `.txt` file is written,
matching the behaviour of CVAT ground truth exports which only contain frames
where a bounding box was annotated.

---

## Input Requirements

| Property | Details |
|---|---|
| Format | `.csv` file |
| Content | Processed clips index CSV containing `clip_relative_path` and `labelled_clip_relative_path` columns. Must be `processed_clips_index.csv`, not `all_clips_index_raw.csv` — only the processed index contains the `labelled_clip_relative_path` column needed for ground truth bounding box comparison |

| Property | Details |
|---|---|
| Format | `.pt` file |
| Content | Fine-tuned YOLO model weights produced by `training_yolo.py` |

| Property | Details |
|---|---|
| Format | Directory of CVAT annotation `.zip` files |
| Content | Ground truth bounding box annotations exported from CVAT — required only if frame-level evaluation metrics are needed. Without this, `--labelled_clips_dir` can be omitted and only predicted `.txt` files will be produced |

---

## Output

Predicted bounding box `.txt` files are saved to `ROOT_DIR/output_path/`,
using the same naming convention as CVAT ground truth annotations:

```text
<numeric_id>_<part_id>_frame_<XXXXXX>.txt
```

Each `.txt` file contains one line per detected bounding box in YOLO
normalized format:

```text
class_id  center_x  center_y  width  height
```

If `--labelled_clips_dir` is provided, a frame-level evaluation report is
also saved to the output directory:

```text
results/predicted_bboxes/<split_name>/frame_level_evaluation.json
```

```json
{
  "true_positives": 0,
  "false_positives": 0,
  "false_negatives": 0,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "f2": 0.0,
  "avg_bbox_iou": 0.0
}
```

---

## How it works

1. `extract_predicted_bboxes` reads the input CSV, loads the fine-tuned YOLO
   model once, then loops through each clip frame by frame.
2. `save_bbox_as_yolo_txt` converts raw YOLO model output for each frame into
   normalized YOLO format and saves it as a `.txt` file. Frames with no
   detections above `--conf_threshold` are skipped entirely.
3. If `--labelled_clips_dir` is provided, `evaluate_frame_level` is called
   after inference completes. It loads ground truth boxes from CVAT zip files
   and compares them against predicted `.txt` files frame by frame, counting
   true positives, false positives, and false negatives.

### Edge Cases

**"No detections on a frame"**
    If the model makes no detections above `--conf_threshold` on a given
    frame, no `.txt` file is written for that frame. This matches the CVAT
    ground truth convention where unannotated frames have no corresponding
    `.txt` file.

**"Missing clip files"**
    If a clip listed in the input CSV cannot be found on disk, a warning is
    printed and that clip is skipped. All other clips continue processing.

**"Missing CVAT zip files"**
    If a CVAT zip file referenced in the ground truth CSV cannot be found,
    `load_gt_boxes_from_zip` returns an empty dict and that clip contributes
    no ground truth frames to the evaluation.

---

## Core Pipeline Concepts

### Frame-Level vs Event-Level Evaluation

This script evaluates at the **frame level** — each individual frame is
either a true positive, false positive, or false negative independently.
This is different from `evaluation.py` which evaluates at the **event
level**, grouping consecutive detections into events and comparing event
time windows using temporal IoU.

Frame-level evaluation directly measures the YOLO model's raw CS detection
accuracy: when the model draws a bounding box, how accurate is it spatially,
and how often does it detect CS in frames that actually contain CS?

### Bounding Box Matching

A predicted frame is counted as a true positive if the model detected a
bounding box on that frame AND the ground truth has a bounding box for that
frame, with IoU >= `--iou_threshold`. If the model detects something but the
ground truth has no box (or IoU is too low), it is a false positive. If the
ground truth has a box but the model detected nothing, it is a false negative.

### F2 Score

In addition to F1, the evaluation computes F2, which weights recall twice as
heavily as precision. This reflects the project's risk tolerance: missing a
real cross-sucking frame is considered more costly than occasionally flagging
a false one.

---

## Usage

### Inference Only

Run inference and save predicted bounding boxes without evaluation:

```bash
uv run python scripts/predict_bboxes.py \
    --input_path data/processed/pipeline_demo/test.csv \
    --model_path data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt \
    --output_path results/predicted_bboxes/pipeline_demo/ \
    --skip 10
```

### Inference + Evaluation

Run inference and evaluate against CVAT ground truth annotations:

```bash
uv run python scripts/predict_bboxes.py \
    --input_path data/processed/<split_name>/test.csv \
    --model_path data/yolo_training_runs/<split_name>/<run_name>/weights/best.pt \
    --output_path results/predicted_bboxes/<split_name>/<run_name>/ \
    --labelled_clips_dir /path/to/cross_sucking_labelled \
    --skip 10
```

### Overwriting Existing Results

By default the script skips frames that already have a predicted `.txt` file.
Pass `--FORCE` to overwrite existing files:

```bash
uv run python scripts/predict_bboxes.py \
    --input_path data/processed/pipeline_demo/test.csv \
    --model_path data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt \
    --output_path results/predicted_bboxes/pipeline_demo/ \
    --skip 10 \
    --FORCE
```

### Adjusting Thresholds

Use `--conf_threshold` to control the minimum YOLO confidence score required
to save a detection, and `--iou_threshold` to control the minimum bounding
box IoU required to count a frame as a true positive during evaluation:

```bash
uv run python scripts/predict_bboxes.py \
    --input_path data/processed/pipeline_demo/test.csv \
    --model_path data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt \
    --output_path results/predicted_bboxes/pipeline_demo/ \
    --labelled_clips_dir /path/to/cross_sucking_labelled \
    --conf_threshold 0.25 \
    --iou_threshold 0.5 \
    --skip 10
```

---

## Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input_path` | str | required | Path to input CSV relative to `ROOT_DIR` |
| `--model_path` | str | required | Path to YOLO model weights relative to `ROOT_DIR` |
| `--output_path` | str | required | Output directory relative to `ROOT_DIR` |
| `--labelled_clips_dir` | str | None | Path to CVAT annotation zip files. If provided, runs frame-level evaluation after inference |
| `--skip` | int | 1 | Process every Nth frame. Should match skip used in preprocessing |
| `--conf_threshold` | float | 0.0 | Minimum YOLO confidence score to save a detection |
| `--iou_threshold` | float | 0.5 | Minimum bbox IoU to count as a True Positive during evaluation |
| `--FORCE` | flag | False | Overwrite existing output files |

---

## Function Reference

::: scripts.predict_bboxes
    options:
        show_source: false
        show_root_heading: true