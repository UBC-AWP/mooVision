# Baseline Cross-Sucking Detector

This page documents how to use `scripts/run_models/baseline/baseline.py` to
detect cross-sucking behaviour in calf videos using YOLO bounding box overlap,
and write structured JSON event metadata for downstream use.

---

## Summary

The baseline detector provides a deliberately naive lower-bound reference for
cross-sucking detection. It runs a COCO-pretrained YOLO26 model across each
video in a test split CSV, computing pairwise IoU between all detected calf
bounding boxes in each frame. Consecutive frames where overlap exceeds a
minimum threshold are grouped into events, and events shorter than a minimum
duration are discarded as noise.

This detector operates on proximity alone and has no understanding of
cross-sucking as a behaviour. Its purpose is to provide a reference point
against which fine-tuned model performance can be meaningfully compared. Any
improvement from the fine-tuned YOLO26 or Seq-NMS models is meaningful
precisely because it demonstrates better performance than simply flagging every
sufficiently long calf contact as cross-sucking.

One JSON metadata file is written per video containing event timestamps,
confidence scores, and per-frame intersection box coordinates for downstream
clipping and analysis.

---

## Environment Setup

Before running the baseline script, ensure your local workspace satisfies
the following:

- `.env` is populated with `ROOT_DIR`, `LOCAL_DIR`, and `SOURCE_VIDEOS_DIR`
  (used by `config.py`).
- Source videos exist under `SOURCE_VIDEOS_DIR`.
- A test split CSV exists under `data/processed/<split_name>/test.csv`,
  produced by running `scripts/split_data/split_data.py`.

---

## Output

Results are saved to `results/metadata/<split_name>/baseline/` automatically.
The split name is inferred from the parent folder of the input CSV
(e.g. `day_based`).

```text
results/
└── metadata/
    └── <split_name>/
        └── baseline/
            ├── ch02_20250913094601_results.json
            └── ch02_20250913094602_results.json
```

Videos whose JSON already exists are skipped automatically, making the script
safe to re-run after interruptions.

---

## Usage

> **Tip:** You can run this step with `make baseline` instead of the
> commands below. See
> [Running the Pipeline (Makefile)](index.md#running-the-pipeline-makefile)
> for details.

### Basic usage

```bash
uv run scripts/run_models/baseline/baseline.py --csv data/processed/day_based/test.csv
```

### Run with frame skipping for faster processing

```bash
uv run scripts/run_models/baseline/baseline.py \
  --csv data/processed/day_based/test.csv \
  --frame_skip 30
```

### Run with custom thresholds

```bash
uv run scripts/run_models/baseline/baseline.py \
  --csv data/processed/day_based/test.csv \
  --iou_threshold 0.05 \
  --conf_threshold 0.4 \
  --min_duration 2.0
```

### Specifying a different split

The split name is inferred from the parent folder of the CSV. To run on a
different split, pass the corresponding test CSV:

```bash
uv run scripts/run_models/baseline/baseline.py \
  --csv data/processed/pen_based/pen_2/test.csv
```

---

## Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--csv` | str | required | Path to a split test CSV (e.g. `data/processed/day_based/test.csv`) |
| `--model` | str | `yolo26x.pt` | YOLO model weights filename |
| `--iou_threshold` | float | `0.1` | Minimum IoU overlap to flag a frame |
| `--conf_threshold` | float | `0.5` | Minimum YOLO detection confidence to keep a box |
| `--min_duration` | float | `0.5` | Minimum event duration in seconds |
| `--frame_skip` | int | `1` | Process every Nth frame |

---

## Function Reference

::: scripts.run_models.baseline.baseline
    options:
        show_source: false
        show_root_heading: true
        members:
            - compute_iou
            - frame_has_overlap
            - extract_events
            - load_split_from_csv
            - detect_video
            - run_all