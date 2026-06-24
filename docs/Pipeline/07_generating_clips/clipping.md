# Clipping Pipeline

This page documents how to use `scripts/clipping/clipping.py` to reproduce
short annotated clips from long-form source videos using JSON event metadata
produced by the detection pipeline.

---

## Summary

The clipping pipeline reads JSON event metadata produced by any stage of the
detection pipeline (baseline, fine-tuned YOLO, or Seq-NMS) and reproduces
the corresponding video segments as standalone annotated clips with per-frame
bounding box overlays. All clips are written to `results/result_clips/`,
preserving the input directory structure relative to the `metadata/` folder.

---

## Environment Setup

Before running the clipping script, ensure your local workspace satisfies
the following:

- `.env` is populated with `ROOT_DIR` and `LOCAL_DIR` (used by `config.py`).
- Source videos exist under `ROOT_DIR/raw_cross_sucking_datalog/`.
- JSON metadata files exist under `LOCAL_DIR/results/metadata/` produced
  by the detection pipeline.

---

## Output Structure

```text
results/result_clips/
├── pipeline_demo/
│   ├── yolo/
│   │   └── ch02_20250913094601/
│   │       ├── ch02_20250913094601_event001_12.3-16.8_boxed.mp4
│   │       └── ch02_20250913094601_event002_22.1-25.4_boxed.mp4
│   └── baseline/
│       └── ch02_20250913094601/
│           └── ch02_20250913094601_event001_12.3-16.8_boxed.mp4
```

---

## Usage

### Clip from a single JSON file

```bash
uv run scripts/clipping/clipping.py \
  --input results/metadata/pipeline_demo/yolo/ch02_20250913094601_results.json
```

### Clip from a directory of JSON files

```bash
uv run scripts/clipping/clipping.py \
  --input results/metadata/pipeline_demo/yolo/
```

### Clip to a custom output directory

```bash
uv run scripts/clipping/clipping.py \
  --input results/metadata/pipeline_demo/yolo/ \
  --output /tmp/clips/
```

### Specifying a different model output

The output path mirrors the input path relative to `metadata/`. To clip
results from a different model, pass the corresponding metadata directory:

```bash
uv run scripts/clipping/clipping.py \
  --input results/metadata/pipeline_demo/seq-nms/
```

---

## Function Reference

::: scripts.clipping.clipping
    options:
        show_source: false
        show_root_heading: true
        members:
            - reproduce_clip
            - split_by_json_events
            - annotate_clip_with_boxes