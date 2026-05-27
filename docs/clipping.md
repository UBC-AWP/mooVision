# Clipping Pipeline

This page documents how to use `scripts/clipping.py` to reproduce short clips from long-form videos, either from a CSV index or from JSON event metadata produced by the baseline pipeline.

---

## Summary

The clipping pipeline replays precise temporal segments from raw `.mp4` files into standalone clip files. It supports two input modes:

1) **Index-based clipping**: reads a CSV index with clip start/end timestamps.  
2) **JSON events-based clipping**: reads JSON event metadata and optionally produces boxed clips with per-frame annotations.

All clips are written to the configured `REPRODUCED_CLIPS_DIR`, with names derived from either the CSV `clip_name` or a deterministic event naming scheme.

---

## Input Requirements

| Property | Details |
|---|---|
| Format | CSV index **or** JSON events metadata |
| Content | Clip start/end times + source video references (optionally bounding boxes) |

Before running the clipping script, ensure your local workspace satisfies the following dependencies:

- `.env` is populated with `ROOT_DIR` and `LOCAL_DIR` (used by `config.py`).
- Source videos exist under `ROOT_DIR/raw_cross_sucking_datalog/videos`.
- For **index mode**, `INDEX_PATH` points to a CSV index (default: `ROOT_DIR/cross_sucking_clips/all_clips_index.csv`).
- For **JSON events mode**, JSON files exist under `LOCAL_DIR/results/metadata/baseline`.

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

Optional fields for annotation:

- `identifier` (string used for clip naming)
- `fps` (float; used for annotation alignment)
- `events[*].intersection_box`: list of per-frame boxes with `frame`, `x1`, `y1`, `x2`, `y2`

---

## Output

By default, results are saved to `REPRODUCED_CLIPS_DIR` (defined in `config.py`).

```text
reproduced_clips/
├── Pen 2 - Group 2/
│   └── POSTWEANING/Day 1/
│       ├── <identifier>_event001_12.3-16.8.mp4
│       └── <identifier>_event001_12.3-16.8_boxed.mp4
└── <clip_name_from_csv>.mp4
```

When `annotate=True`, boxed clips are saved alongside the unboxed clips using the `_boxed.mp4` suffix.

---

## How it works

1) Load configuration paths from `config.py` (sourced from `.env`).  
2) Choose a clipping strategy: CSV index mode or JSON events mode.  
3) Resolve each clip’s source video and timestamps.  
4) `reproduce_clip` reads frames from `start_sec` to `end_sec` and writes a new `.mp4`.  
5) In JSON mode, `annotate_clip_with_boxes` can optionally render bounding boxes into a second clip.

### Edge Cases

**"Annotation Alignment Requirement"**  
When annotations are used, the JSON `fps` and `intersection_box` frame indices must align with the source video. If they do not, boxes will be offset from the target frames.

**"Missing Video Paths"**  
If a JSON file references a `video_path` that does not exist, a `FileNotFoundError` is raised.

**"Index Mode Processes a Single Row"**  
`split_by_index` currently iterates `matched[:1]`, so only the first matched clip is reproduced. Remove the slice to enable batch reproduction.

---

## Core Pipeline Concepts

### Input Modes

- **CSV index mode** reads the clip list from a single index and writes clips using each row’s `clip_name`.
- **JSON events mode** reads one JSON file or a folder of JSON files, creating per-event clips and optionally boxed versions.

### Clip Naming and Foldering

In JSON mode, clip names follow:

```text
<identifier_stem>_event###_<start_sec>-<end_sec>.mp4
```

Clips are written into a folder derived from the source video path segment after `cross_sucking_clips/`, preserving pen/stage/day structure when available.

### Annotation Rendering

`annotate_clip_with_boxes` converts **source-video frame indices** to **clip-local indices** by subtracting the event’s start frame. This keeps boxes aligned to the clipped segment.

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
