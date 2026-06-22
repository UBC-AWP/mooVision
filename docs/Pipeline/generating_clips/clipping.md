# Clipping Pipeline

This page documents how to use `scripts/clipping.py` to reproduce short annotated clips from long-form source videos using JSON event metadata produced by the detection pipeline.

---

## Summary

The clipping pipeline replays precise temporal segments from raw `.mp4` files into standalone annotated clip files. It reads JSON event metadata produced by any stage of the detection pipeline (baseline, fine-tuned YOLO, or sequence-linked) and produces boxed clips with per-frame bounding box annotations.

All clips are written to `results/result_clips/`, preserving the directory structure of the input JSON files relative to the `metadata/` folder.

---

## Input Requirements

| Property | Details |
|---|---|
| Format | JSON event metadata files |
| Content | Clip start/end times, source video references, and optionally per-frame bounding boxes |

Before running the clipping script, ensure your local workspace satisfies the following:

- `.env` is populated with `ROOT_DIR` and `LOCAL_DIR` (used by `config.py`).
- Source videos exist under `ROOT_DIR/raw_cross_sucking_datalog/`.
- JSON metadata files exist under `LOCAL_DIR/results/metadata/` produced by the detection pipeline.

### Expected JSON Schema

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

- `identifier` (string; used as the per-video output folder name and clip filename stem)
- `fps` (float; used for annotation frame alignment)
- `events[*].intersection_box`: list of per-frame boxes with `frame`, `x1`, `y1`, `x2`, `y2`

---

## Output

Clips are written to `results/result_clips/` preserving everything after `metadata/` in the input JSON path. Each video gets its own subfolder named after its `identifier` stem.

```text
results/result_clips/
├── pipeline_demo/
│   └── yolo/
│       └── ch02_20250913094601/
│           ├── ch02_20250913094601_event001_12.3-16.8_boxed.mp4
│           └── ch02_20250913094601_event002_22.1-25.4_boxed.mp4
│   └── baseline/
│       └── ch02_20250913094601/
│           ├── ch02_20250913094601_event001_12.3-16.8_boxed.mp4
│           └── ch02_20250913094601_event002_22.1-25.4_boxed.mp4
```

Boxed clips are saved with the `_boxed.mp4` suffix. The unboxed intermediate clip is removed after annotation succeeds. If annotation fails, the unboxed version is kept as a fallback.

---

## How It Works

1. Load configuration paths from `config.py` (sourced from `.env`).
2. Accept `--input` as a single JSON file or a directory to search recursively for `*.json` files.
3. For each JSON, resolve the source video path using `ROOT_DIR`.
4. Extract everything after `metadata/` in the JSON file's parent path to construct the output subdirectory.
5. Create a per-video output folder named after the `identifier` stem.
6. `reproduce_clip` reads frames from `start_sec` to `end_sec` and writes a temporary `.mp4`.
7. `annotate_clip_with_boxes` renders per-frame bounding boxes into the final `_boxed.mp4`, then removes the unboxed intermediate.

### Output Path Derivation

The output path is derived entirely from the JSON file's location relative to the `metadata/` folder. Everything after `metadata/` in the JSON's parent directory is preserved as-is in the output, so no level-counting is needed and any input structure is handled correctly.

```text
input  → results/metadata/pipeline_demo/yolo/ch02_20250913094601_results.json
output → results/result_clips/pipeline_demo/yolo/ch02_20250913094601/ch02_20250913094601_event001_12.3-16.8_boxed.mp4

input  → results/metadata/pipeline_demo/baseline/ch02_20250913094601_results.json
output → results/result_clips/pipeline_demo/baseline/ch02_20250913094601/ch02_20250913094601_event001_12.3-16.8_boxed.mp4
```

### Edge Cases

**Annotation Alignment**
The JSON `fps` and `intersection_box` frame indices must align with the source video. Frame indices in `intersection_box` are in source-video coordinates and are converted to clip-local indices by subtracting the event start frame before annotation.

**Missing Video Paths**
If a JSON file references a `video_path` that does not exist on disk, a `FileNotFoundError` is raised.

**No Events in JSON**
If a JSON file contains an empty `events` list, the file is skipped and a message is printed.

**Annotation Failure**
If `annotate_clip_with_boxes` fails for an event, the unboxed intermediate clip is kept as a fallback and a warning is printed.

---

### Clip Naming

Clips follow this naming convention:

```text
<identifier_stem>_event###_<start_sec>-<end_sec>_boxed.mp4
```

Example:

```text
ch02_20250913094601_event001_12.3-16.8_boxed.mp4
```

---

## Usage

### Clip from a single JSON file

```bash
uv run scripts/clipping.py --input results/metadata/pipeline_demo/yolo/ch02_20250913094601_results.json
```

### Clip from a directory of JSON files

```bash
uv run scripts/clipping.py --input results/metadata/pipeline_demo/yolo/
```

### Clip to a custom output directory

```bash
uv run scripts/clipping.py --input results/metadata/pipeline_demo/yolo/ --output /tmp/clips/
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
        - split_by_json_events
        - annotate_clip_with_boxes
        - main