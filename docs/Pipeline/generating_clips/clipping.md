# Clipping Pipeline

This page documents how to use `scripts/clipping.py` to reproduce short annotated clips from long-form source videos using JSON event metadata produced by the baseline detection pipeline.

---

## Summary

The clipping pipeline replays precise temporal segments from raw `.mp4` files into standalone annotated clip files. It supports two input modes:

1) **Index-based clipping**: reads a CSV index with clip start/end timestamps.
2) **JSON events-based clipping**: reads JSON event metadata produced by the baseline pipeline and produces boxed clips with per-frame bounding box annotations.

All clips are written to `results/result_clips/`, organised by split name and source video hierarchy.

---

## Input Requirements

| Property | Details |
|---|---|
| Format | CSV index **or** JSON events metadata |
| Content | Clip start/end times + source video references (optionally bounding boxes) |

Before running the clipping script, ensure your local workspace satisfies the following:

- `.env` is populated with `ROOT_DIR` and `LOCAL_DIR` (used by `config.py`).
- Source videos exist under `ROOT_DIR/raw_cross_sucking_datalog/videos`.
- For **index mode**, `INDEX_PATH` points to a CSV index (default: `ROOT_DIR/cross_sucking_clips/all_clips_index.csv`).
- For **JSON events mode**, JSON files exist under `LOCAL_DIR/results/metadata/baseline/` produced by the baseline script.

### Required CSV Columns (Index Mode)

At minimum, the CSV must include:

- `source_video_basename`
- `source_video_path`
- `clip_name`
- `clip_start_in_source_sec`
- `clip_end_in_source_sec`

### Expected JSON Schema (Events Mode)

Minimum fields:

```json
{
  "video_path": "path/to/source.mp4",
  "events": [
    { "start_sec": 12.3, "end_sec": 16.8 }
  ]
}
```

Optional fields used for annotation:

- `identifier` (string used for clip naming)
- `fps` (float; used for annotation frame alignment)
- `events[*].intersection_box`: list of per-frame boxes with `frame`, `x1`, `y1`, `x2`, `y2`

---

## Output

Clips are written to `results/result_clips/` with the split name inferred from the JSON directory structure, mirroring the baseline metadata hierarchy.

```text
results/result_clips/
├── day_based/
│   └── Pen 2 - Group 2/POSTWEANING/Day 1/
│       ├── <identifier>_event001_12.3-16.8_boxed.mp4
│       └── <identifier>_event002_22.1-25.4_boxed.mp4
└── pen_based/
    └── Pen 2 - Group 2/POSTWEANING/Day 1/
        └── <identifier>_event001_12.3-16.8_boxed.mp4
```

Boxed clips are saved with the `_boxed.mp4` suffix. The unboxed intermediate clip is removed after annotation succeeds. If annotation fails, the unboxed version is kept as a fallback.

---

## How It Works

1) Load configuration paths from `config.py` (sourced from `.env`).
2) Choose a clipping strategy: CSV index mode or JSON events mode.
3) Recursively search the input directory for `*.json` metadata files.
4) For each JSON, resolve the source video path and infer the split name from the directory structure.
5) `reproduce_clip` reads frames from `start_sec` to `end_sec` and writes a temporary `.mp4`.
6) `annotate_clip_with_boxes` renders per-frame bounding boxes into the final `_boxed.mp4`, then removes the unboxed intermediate.

### Edge Cases

**"Annotation Alignment"**
The JSON `fps` and `intersection_box` frame indices must align with the source video. Frame indices in `intersection_box` are in source-video coordinates and are converted to clip-local indices by subtracting the event start frame before annotation.

**"Missing Video Paths"**
If a JSON file references a `video_path` that does not exist on disk, a `FileNotFoundError` is raised.

**"Index Mode Processes a Single Row"**
`split_by_index` currently iterates `matched[:1]`, so only the first matched clip is reproduced. Remove the slice to enable batch reproduction.

---

## Core Pipeline Concepts

### Input Modes

- **CSV index mode** reads the clip list from a single index and writes clips using each row's `clip_name`.
- **JSON events mode** recursively searches a directory for JSON files produced by the baseline pipeline, creating one annotated clip per event.

### Split Name Inference

In JSON events mode, the split name (e.g. `day_based`) is inferred from the JSON file's directory structure — specifically four levels up from the JSON file, matching the baseline output layout:

```text
results/metadata/baseline/<split_name>/<Pen>/<Stage>/<Day>/<file>.json
```

This split name is then prepended to the output path under `results/result_clips/`.

### Clip Naming

In JSON mode, clip names follow:

```text
<identifier_stem>_event###_<start_sec>-<end_sec>_boxed.mp4
```

### Annotation Rendering

`annotate_clip_with_boxes` converts source-video frame indices to clip-local indices by subtracting the event's start frame, keeping boxes aligned to the clipped segment.

---

## Usage

### Basic (JSON events mode)

```bash
uv run scripts/clipping.py
```

### Index mode (CSV index)

```bash
uv run python - <<'PY'
from scripts.clipping import split_by_index
from config import INDEX_PATH, REPRODUCED_CLIPS_DIR

split_by_index(INDEX_PATH, REPRODUCED_CLIPS_DIR)
PY
```

---

## Function Reference

::: scripts.clipping
    options:
      show_source: false
      show_root_heading: false
      heading_level: 2
      show_signature_annotations: true
      members:
        - reproduce_clip
        - split_by_index
        - split_by_json_events
        - annotate_clip_with_boxes
        - run_splitting
        - main