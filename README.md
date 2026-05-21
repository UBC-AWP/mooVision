# MooVision

MooVision is a computer-vision pipeline for detecting **cross-sucking behaviour** in socially housed dairy calves from angled overhead pen video. The goal is to reduce the time and effort required for manual review/labeling by automatically producing candidate events, clipped video segments, and structured metadata for downstream analysis.

## Project Overview

Cross-sucking (here: sucking directed at various body parts of other calves) is a welfare concern in group-housed calves and is currently studied via manual labeling of long video recordings. This project builds a scalable workflow that:

- takes raw pen video as input
- runs a baseline detector (pretrained YOLOv8) with simple logic on top (e.g., proximity/overlap + temporal persistence)
- outputs predicted event windows and metadata (start/end time, confidence, pen, weaning stage, day)
- optionally generates clipped videos for review and evaluation

## Repository Structure (high level)

- `src/`: library code (config, preprocessing, baseline inference, evaluation)
- `scripts/`: runnable entry points (run baseline, build clip index, etc.)
- `eda/`: EDA notebooks
- `tests/`: unit tests
- `docs/` : project documentation
- `report/`: report assets

## Environment Setup

This project uses `uv` for package management.

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Clone the repo and cd into it
3. Run `uv sync` to install all dependencies
4. Run scripts with `uv run python <script.py>`

## Local `.env` configuration (required)

We use a local `.env` file (stored at the **repo root**) to configure machine-specific paths (e.g., where the video clips live). This avoids hardcoding absolute paths in code.

1. Create a `.env` file at the repo root:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set your local data path, for example:

   ```bash
   MOOVISION_DATA_ROOT=/Users/<you>/path/to/data_root
   ```

3. `.env` is ignored by git (do not commit). If you need to change what variables exist, update `.env.example` instead.

---

## Baseline Cross-Sucking Detector

A baseline script for detecting cross-sucking behaviour in calves using YOLO bounding box overlap. For each input video, the script produces a JSON metadata file containing the time windows where cross-sucking may have occurred, along with the per-frame intersection box coordinates of the overlapping region.

---

### How It Works

1. Loads a pretrained YOLO26 model by default (the model works on COCO dataset which detects `cow` class as a proxy for calves)
2. Loads one video at a time (this setting might be changed in the future)
3. Reads every Nth frame as set by `--frame_skip` (default: 1 = every frame). This will be seen while running `baseline.py`
4. For each processed frame, detects all calves and checks if any two bounding boxes overlap beyond a configurable IoU threshold
5. Groups consecutive flagged frames into events and filters out events shorter than a minimum duration
6. Saves a JSON metadata file with the flagged events and their intersection box coordinates

**Notes:**

1. The COCO-pretrained model was trained on adult cattle outdoors. Detection accuracy will improve significantly once fine-tuned on your own labelled calf footage
2. The model, IoU threshold, confidence threshold, and number of skipped frames can be changed into other baseline models using arguments which will be described below
3. The argument `skip_frame` allows to skip N number of frames at a time (e.g. `skip_frame = 5` means instead of )

### Input (baseline)

| Property | Details |
|---|---|
| Format | `.mp4`, `.avi`, or any format supported by OpenCV |
| Content | Single video clip of calves in a pen |

### Output (baseline)

Results are saved to `results/metadata/baseline/` automatically.

**File:** `results/metadata/baseline/<video_name>_results.json`

**Example output:**

```json
{
  "identifier": "pen3_morning.mp4",
  "video_path": "/data/videos/pen3_morning.mp4",
  "model": "yolo26m.pt",
  "iou_threshold": 0.1,
  "conf_threshold": 0.5,
  "min_duration_sec": 1.0,
  "fps": 25.0,
  "total_frames": 7500,
  "total_duration_sec": 300.0,
  "cross_sucking_detected": true,
  "num_events": 2,
  "events": [
    {
      "start_sec": 12.4,
      "end_sec": 19.1,
      "duration_sec": 6.7,
      "avg_confidence": 0.71,
      "intersection_boxes": [
        { "frame": 310, "x1": 290, "y1": 95, "x2": 340, "y2": 280 },
        { "frame": 311, "x1": 291, "y1": 96, "x2": 341, "y2": 281 }
      ]
    }
  ]
}
```

**Output fields:**

| Field | Description |
|---|---|
| `identifier` | Video filename — used as the unique ID for this clip |
| `cross_sucking_detected` | `true` if at least one event was flagged |
| `num_events` | Total number of flagged events |
| `start_sec` / `end_sec` | Start and end time of the event in seconds |
| `duration_sec` | Length of the event in seconds |
| `avg_confidence` | Average YOLO detection confidence across all frames in the event |
| `intersection_boxes` | Per-frame pixel coordinates of the overlapping region for downstream annotation |

### How to run the baseline

1. Using the default parameters:

    ```bash
    uv run scripts/baseline/baseline.py --video "<path/to/video.mp4>"
    ```

2. Using custom parameters:

    ```bash
    uv run baseline.py --video "<path/to/video.mp4>" \
                    --iou_threshold <insert_threshold> \
                    --conf_threshold <insert_threshold> \
                    --min_duration <insert_in_seconds> \
                    --model <insert_model_name> \
                    --frame_skip <insert_integer> 
    ```

**Note:** If your file path contains spaces, wrapping it in quotes will avoid shell parsing errors. The file path should look like `"\Users\mickeymouse\data\video_cross_sucking.mp4"`

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video` | str | - | Path to input video file (required) |
| `--model` | str | `yolo26m.pt` | YOLO model weights filename e.g. `yolo26n.pt`, `yolo26m.pt`, `yolo26l.pt` |
| `--iou_threshold` | float | `0.1` | Minimum IoU overlap to flag a frame. Must be between `0.0` and `1.0` |
| `--conf_threshold` | float | `0.5` | Minimum YOLO detection confidence to keep a box. Must be between `0.0` and `1.0` |
| `--min_duration` | float | `1.0` | Minimum duration in seconds a continuous overlap must last to be flagged as an event |
| `--frame_skip` | int | `1` | Process every Nth frame. Higher values are faster but may miss short events |

### Tuning tips

- **`--iou_threshold`** — lower values (e.g. `0.05`) catch more events but increase false positives. Raise it if you're getting too many flags from calves just standing close together
- **`--conf_threshold`** — raise if the model is detecting non-calf objects. Lower if calves are being missed in difficult lighting
- **`--min_duration`** — raise if brief accidental box overlaps are being flagged. A value of `1.0–2.0` seconds works well as a starting point
- **`--frame_skip`** — raise for faster processing on long videos (5–15 is a good range). Keep at `1` if you need precise event boundaries or are looking for very short events

#### Demo Examples

For demonstration purposes, we provide 2 clip samples each for cross-sucking and non-cross-sucking examples (each video ~15-17s).

- Cross-sucking examples:

    1. Example 1 (~2-3 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --video "sample_videos/cross_sucking_clip_sample/CS_0276_WEAN_d1_p2_cowT_16102025_ch02-20251016124717_19033_19045.mp4"
        ```

    2. Example 2 (~4-5 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --video "sample_videos/cross_sucking_clip_sample/CS_0002_POSTWEAN_d1_p2_cow3_02112025_ch02-20251103001956_60818_60835.mp4"
        ```

- Non-cross-sucking examples

    1. Example 1 (~1-2 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --video "sample_videos/non_cross_sucking_clip_sample/ch05_20251114073451_15s.mp4"
        ```

    2. Example 2 (~3-4 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --video "sample_videos/non_cross_sucking_clip_sample/ch04_20250828075551_15s.mp4"
        ```
