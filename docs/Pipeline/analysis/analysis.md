# Distribution Analysis

This page documents how to use `scripts/notebooks/distribution_analysis.ipynb` to visualise and compare model prediction outputs across splits and models.

---

## Summary

The distribution analysis notebook loads per-video prediction JSONs and evaluation reports produced by the MooVision pipeline, and generates a series of diagnostic charts for comparing YOLO and Seq-NMS model behaviour across three experimental splits (POSTWEAN, WEAN, Random).

The notebook produces nine figures grouped by analysis type:

| Figure | Type | Description |
|--------|------|-------------|
| Fig 1 | Histogram | Confidence score distribution per model and split |
| Fig 2 | Bar chart | Predicted event count per video |
| Fig 3 | Histogram | Event duration distribution per model and split |
| Fig 4 | Scatter | Confidence vs event duration |
| Fig 5 | Dot plot | Temporal density of events within videos — POSTWEAN |
| Fig 6 | Dot plot | Temporal density of events within videos — WEAN |
| Fig 7 | Dot plot | Temporal density of events within videos — Random |
| Fig 8 | Heatmap | Event-level confusion matrix by model and split |

All charts render inline — nothing is saved to disk.

---

## Inputs (Data requirements)

The notebook requires two sets of outputs produced by the MooVision training and evaluation pipeline.

### Prediction JSONs

Per-video prediction JSON files output by `baseline.py` or `clipping.py`. These must be organised into the following directory structure under `results/metadata/`:

```
results/
└── metadata/
    ├── period_based/
    │   ├── POSTWEAN/
    │   │   ├── yolo/
    │   │   │   └── *.json
    │   │   └── seq-nms/
    │   │       └── *.json
    │   └── WEAN/
    │       ├── yolo/
    │       │   └── *.json
    │       └── seq-nms/
    │           └── *.json
    └── random/
        ├── yolo/
        │   └── *.json
        └── seq-nms/
            └── *.json
```

Each prediction JSON must conform to the MooVision output schema and contain the following fields:

```json
{
  "identifier": "video_name.mp4",
  "video_path": "/path/to/video",
  "total_duration_sec": 120.0,
  "cross_sucking_detected": true,
  "num_events": 3,
  "events": [
    {
      "start_sec": 10.5,
      "end_sec": 18.2,
      "duration_sec": 7.7,
      "avg_confidence": 0.43
    }
  ]
}
```

### Evaluation Reports

Per-split evaluation JSON files output by the evaluation pipeline. These must be located in `results/evaluation/` and follow the naming convention:

```
evaluation_report_{split}_{model}.json
```

For example:

```
results/
└── evaluation/
    ├── evaluation_report_period_based_postwean_yolo.json
    ├── evaluation_report_period_based_postwean_seq_nms.json
    ├── evaluation_report_period_based_wean_yolo.json
    ├── evaluation_report_period_based_wean_seq_nms.json
    ├── evaluation_report_random_yolo.json
    └── evaluation_report_random_seq_nms.json
```

Each evaluation JSON must contain the following structure:

```json
{
  "event_level": {
    "true_positives": 9,
    "false_positives": 193,
    "false_negatives": 44,
    "precision": 0.044,
    "recall": 0.17,
    "f2": 0.11
  },
  "sequence_level": {
    "avg_temporal_iou": 0.68
  },
  "frame_level": {
    "avg_bbox_iou": 0.52
  }
}
```

---

## Usage

### Basic Usage

Open and run the notebook from the project root:

```bash
uv run jupyter notebook scripts/notebooks/distribution_analysis.ipynb
```

Or with JupyterLab:

```bash
uv run jupyter lab scripts/notebooks/distribution_analysis.ipynb
```

Run all cells in order using **Kernel → Restart & Run All**. Charts will render inline after each figure cell.

### Vegafusion Note

The notebook uses `vegafusion` as the default Altair data transformer for handling large prediction dataframes (Figures 1–7). Figures 8 and 9 switch to the `default` transformer automatically since they use pre-aggregated data. Do not manually re-enable vegafusion after the Figure 8 cell or the charts will fail to render.

### Adding a New Split

To add a new split, define its prediction JSON paths and add an entry to the `SPLITS` dictionary at the top of the notebook:

```python
SPLITS = {
    "POSTWEAN": {
        "YOLO":    YOLO_MODEL_OUTPUT_DIR_PERIOD_POST,
        "Seq-NMS": SEQ_MODEL_OUTPUT_DIR_PERIOD_POST,
    },
    "MY_NEW_SPLIT": {           # add your split here
        "YOLO":    Path("results/metadata/my_split/yolo"),
        "Seq-NMS": Path("results/metadata/my_split/seq-nms"),
    },
}
```

Also update `SPLIT_ORDER` and `MODEL_ORDER` to include the new split label so facet ordering is preserved.

### Adding a New Model

To add a new model alongside YOLO and Seq-NMS, add its path to each relevant split entry in `SPLITS` and update `MODEL_ORDER`:

```python
MODEL_ORDER = [f"{m} — {s}" for s in SPLIT_ORDER for m in ["YOLO", "Seq-NMS", "MY_MODEL"]]
```

---

## Outputs

The notebook does not write any files to disk. All figures render inline in the notebook. To export a figure for use in the report:

```python
fig5.save("reports/figures/fig5_temporal_density_postwean.html")  # interactive
fig9.save("reports/figures/fig9_confusion_matrix.png", scale_factor=2.0)  # static
```

Note: PNG export requires the `vl-convert-python` package:

```bash
uv add vl-convert-python
```

---

## Figure Reference

### Figure 1 — Confidence Score Distribution

Six-panel histogram of `avg_confidence` per predicted event, arranged as YOLO (top row) vs Seq-NMS (bottom row) across POSTWEAN, WEAN, and Random splits. Seq-NMS is expected to show lower and more spread confidence than YOLO due to the sequence linking step chaining lower-confidence frame detections.

### Figure 2 — Predicted Events per Video

Bar chart of predicted event count per video, sorted descending, coloured by pen. Uniform tall bars across all videos indicate the model is over-detecting rather than concentrating detections on genuine cross-sucking videos.

### Figure 3 — Event Duration Distribution

Histogram of event duration in seconds. YOLO tends to produce a small number of very long events; Seq-NMS fragments these into many shorter events. A spike at short durations alongside a long tail suggests the short cluster is mostly false positives.

### Figure 4 — Confidence vs Event Duration

Scatter plot of `avg_confidence` against `duration_sec`, coloured by pen. Events in the short + low-confidence region are the strongest false positive candidates. Events in the long + high-confidence region are the strongest true positive candidates for manual review.

### Figures 5–7 — Temporal Density of Events Within Videos

Dot plots showing where within each video events fire. Dot size encodes duration, dot colour encodes confidence. Genuine cross-sucking bouts are expected to appear as clusters of large, high-confidence dots within a narrow time window. Sparse dots scattered uniformly across the full timeline suggest the model is not detecting real behavioural bouts.

### Figure 8 — Event-Level Confusion Matrix

Faceted heatmap showing TP, FP, and FN counts for each model and split. True negatives are excluded as they are not well-defined at the event level for continuous video. The top-right cell (Actual CS, Predicted No CS) shows the false negative count and is typically the dominant cell, reflecting the near-zero recall observed across all splits.