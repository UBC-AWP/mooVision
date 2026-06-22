# Baseline Cross-Sucking Detector

A baseline script for detecting cross-sucking behaviour in calves using YOLO bounding box overlap. For each input video, the script produces a JSON metadata file containing the time windows where cross-sucking may have occurred, along with the per-frame intersection box coordinates of the overlapping region.

## How It Works

1. Loads a pretrained YOLO26 model by default (the model works on COCO dataset which detects `cow` class as a proxy for calves)
2. Reads a test CSV (e.g. `data/processed/day_based/test.csv`), deduplicates video paths, and processes each video sequentially. Videos whose JSON output already exists are skipped automatically
3. Reads every Nth frame as set by `--frame_skip` (default: 1 = every frame). Progress is printed while running `baseline.py`
4. For each frame, detects all calves and identifies the pair with the highest overlap (IoU). If IoU ≥ threshold, flags the frame and records that pair's intersection box
5. Groups consecutive flagged frames into events and filters out events shorter than a minimum duration
6. Saves a JSON metadata file per video with the flagged events and their intersection box coordinates

**Notes:**

1. The COCO-pretrained model was trained on adult cattle outdoors. Detection accuracy will improve significantly once fine-tuned on your own labelled calf footage
2. The model, IoU threshold, confidence threshold, and number of skipped frames can be changed using arguments described below
3. The argument `--frame_skip` allows skipping N frames at a time (e.g. `--frame_skip 5` means processing every 5th frame instead of every frame)

## Input (baseline)

| Property | Details |
|---|---|
| Format | A test CSV file (e.g. `data/processed/day_based/test.csv`) containing a `source_video_path` column |
| Content | Each row points to a single `.mp4` or `.avi` video clip of calves in a pen |

## Output (baseline)

Results are saved to `results/metadata/baseline/` automatically. The split name is inferred from the parent folder of the input CSV (e.g. `day_based`), and the directory hierarchy mirrors the source video structure.

**File:** `results/metadata/baseline/<split_name>/<Pen>/<Stage>/<Day>/<video_stem>_results.json`

**Example:** `results/metadata/baseline/day_based/Pen 2 - Group 2/POSTWEANING/Day 1/ch02_results.json`

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
  "frame_skip": 1,
  "cross_sucking_detected": true,
  "num_events": 2,
  "events": [
    {
      "start_sec": 12.4,
      "end_sec": 19.1,
      "duration_sec": 6.7,
      "avg_confidence": 0.71,
      "intersection_box": [
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
| `model` | YOLO model name used for detection |
| `iou_threshold` | IoU threshold used to flag overlapping boxes |
| `conf_threshold` | YOLO detection confidence threshold |
| `min_duration_sec` | Minimum event duration in seconds (shorter events are filtered out) |
| `fps` | Frames per second of the video |
| `total_frames` | Total number of frames in the video |
| `frame_skip` | Frame skip value (1 = every frame, 2 = every 2nd frame, etc.) |
| `cross_sucking_detected` | `true` if at least one event was flagged |
| `num_events` | Total number of flagged events |
| `start_sec` / `end_sec` | Start and end time of the event in seconds |
| `duration_sec` | Length of the event in seconds |
| `avg_confidence` | Average YOLO detection confidence across all frames in the event |
| `intersection_box` | Per-frame pixel coordinates of the overlapping region for downstream annotation |

## How to run the baseline

1. Using the default parameters:

    ```bash
    uv run scripts/baseline/baseline.py --csv "<path/to/split/test.csv>"
    ```

2. Using custom parameters:

    ```bash
    uv run scripts/baseline/baseline.py --csv "<path/to/split/test.csv>" \
                    --iou_threshold <insert_threshold> \
                    --conf_threshold <insert_threshold> \
                    --min_duration <insert_in_seconds> \
                    --model <insert_model_name> \
                    --frame_skip <insert_integer>
    ```

**Note:** If your file path contains spaces, wrap it in quotes. The script infers the split name (e.g. `day_based`) from the parent folder of the CSV and writes results automatically to the corresponding subfolder under `results/metadata/baseline/`.

## Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--csv` | str | - | Path to a split test CSV (e.g. `data/processed/day_based/test.csv`). Required. |
| `--model` | str | `yolo26x.pt` | YOLO model weights filename e.g. `yolo26n.pt`, `yolo26m.pt`, `yolo26l.pt` |
| `--iou_threshold` | float | `0.1` | Minimum IoU overlap to flag a frame. Must be between `0.0` and `1.0` |
| `--conf_threshold` | float | `0.5` | Minimum YOLO detection confidence to keep a box. Must be between `0.0` and `1.0` |
| `--min_duration` | float | `1.0` | Minimum duration in seconds a continuous overlap must last to be flagged as an event |
| `--frame_skip` | int | `1` | Process every Nth frame. Higher values are faster but may miss short events |

### Tuning tips

- **`--iou_threshold`** — lower values (e.g. `0.05`) catch more events but increase false positives. Raise it if you're getting too many flags from calves just standing close together
- **`--conf_threshold`** — raise if the model is detecting non-calf objects. Lower if calves are being missed in difficult lighting
- **`--min_duration`** — raise if brief accidental box overlaps are being flagged. A value of `1.0–2.0` seconds works well as a starting point
- **`--frame_skip`** — raise for faster processing on long videos (5–15 is a good range). Keep at `1` if you need precise event boundaries or are looking for very short events

### Demo Examples

For demonstration purposes, we provide 2 clip samples each for cross-sucking and non-cross-sucking examples (each video ~15-17s).

- Cross-sucking examples:

    1. Example 1 (~2-3 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --csv "data/processed/day_based/test.csv"
        ```

    2. Example 2 (~4-5 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --csv "data/processed/pen_based/test.csv"
        ```

- Non-cross-sucking examples: run with the same commands above — the script processes all videos in the CSV, including non-cross-sucking ones. Videos with no detected events will have `"cross_sucking_detected": false` in their JSON output.

## Limitation

- **Single highest overlap per frame:** Only the pair with the best overlap is recorded per frame. If multiple calves are simultaneously cross-sucking, only the largest IoU is captured

- **YOLO detection instability and flickering:** The detector relies on YOLO correctly identifying all calves. Missed or misidentified calves will cause missed or false overlaps. The instability of YOLO detection would also result in "flickering" in the intersection box coordinates even for the same interaction and can falsely split a single cross-sucking event into multiple short events

- **Camera perspective and distance:** IoU is sensitive to bounding box size and camera angle. Calves cross-sucking at the back of the pen (far from camera) produce small bounding boxes with low IoU values, causing missed detections. The metric underestimates true overlap when viewed from an angle

- **Sensitive to threshold tuning:**
  - **IoU threshold (`--iou_threshold`):**
    - Too high: Misses true overlaps, especially distant or angled interactions
    - Too low: False positives from calves simply standing close together
  - **Confidence threshold (`--conf_threshold`):**
    - Too high: YOLO misses weakly-detected calves, reducing overlap detection
    - Too low: Increases false detections of non-calf objects, creating spurious overlaps
  - **Minimum duration (`--min_duration`):**
    - Too high: Misses shorter but real cross-sucking events
    - Too low: Reports accidental brief touches or flickering frames as events
  - **Frame skip (`--frame_skip`):**
    - Too high: Misses short events entirely, event boundaries become imprecise
    - Too low: Increases processing time
  - No universal defaults exist; tuning is dataset, model, and camera-dependent

- **Model limitation:** The COCO-pretrained model was trained on outdoor cattle, not calves. Hence, the accuracy performance to detect calves can be limited. Accuracy improves significantly after fine-tuning on your own labeled calf footage, especially in varied lighting or pen angles

## Future Improvements

- Consider using other proportion-based overlaps (e.g. intersection over minimum, etc.) as a better alternative or addition to using overlapping area, which is sensitive to camera angle and distance

- Use a "grace period" for handling brief detection gaps (0.5–1.0 seconds) due to YOLO missed detections that break event continuity, splitting a single cross-sucking event into multiple fragmented events. This will allow events separated by brief gaps to be treated as a single continuous interaction

- Output all detected bounding boxes, IoU values for all pairs, and confidence scores per frame (not just the highest pair). This enables offline algorithm refinement and re-analysis without re-running YOLO inference

- Apply smoothing filters (e.g. Kalman filter, moving average) to bounding box trajectories to reduce flickering. Implement frame correlation analysis to determine the minimum number of consecutive frames required to accept a detection, reducing false events from transient YOLO errors

- Implement intelligent frame skipping to downscale processing as data volume increases. For instance, process frames densely in regions of detected overlap, sparsely elsewhere. This will reduce storage and computation while maintaining event precision

## Functions

::: scripts.models.baseline.baseline