#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
MooVision Prediction Distribution Analysis — Multi-Model, Multi-Split Comparison
---------------------------------------------------------------------------------
Loads prediction JSONs for fine-tuned YOLO and Seq-NMS across three splits
(POSTWEAN, WEAN, Random) and produces side-by-side Altair charts grouped by
chart type. All charts render inline — nothing is saved locally.
"""

import json
import re
from pathlib import Path

import pandas as pd
import altair as alt
import sys

# ── Point to project root so config.py is importable ──────────────────────
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import ROOT_DIR

alt.data_transformers.enable("vegafusion")

EVAL_PATH = ROOT_DIR / "results" / "evaluation"

YOLO_MODEL_OUTPUT_DIR_PERIOD_POST  = ROOT_DIR / "results" / "metadata" / "period_based" / "POSTWEAN" / "yolo"
YOLO_MODEL_OUTPUT_DIR_PERIOD_WEAN  = ROOT_DIR / "results" / "metadata" / "period_based" / "WEAN"     / "yolo"
SEQ_MODEL_OUTPUT_DIR_PERIOD_POST   = ROOT_DIR / "results" / "metadata" / "period_based" / "POSTWEAN" / "seq-nms"
SEQ_MODEL_OUTPUT_DIR_PERIOD_WEAN   = ROOT_DIR / "results" / "metadata" / "period_based" / "WEAN"     / "seq-nms"
YOLO_MODEL_OUTPUT_DIR_RANDOM       = ROOT_DIR / "results" / "metadata" / "random"        / "yolo"
SEQ_MODEL_OUTPUT_DIR_RANDOM        = ROOT_DIR / "results" / "metadata" / "random"        / "seq-nms"

# ── Split → model → path mapping ──────────────────────────────────────────
SPLITS = {
    "POSTWEAN": {
        "YOLO":    YOLO_MODEL_OUTPUT_DIR_PERIOD_POST,
        "Seq-NMS": SEQ_MODEL_OUTPUT_DIR_PERIOD_POST,
    },
    "WEAN": {
        "YOLO":    YOLO_MODEL_OUTPUT_DIR_PERIOD_WEAN,
        "Seq-NMS": SEQ_MODEL_OUTPUT_DIR_PERIOD_WEAN,
    },
    "Random": {
        "YOLO":    YOLO_MODEL_OUTPUT_DIR_RANDOM,
        "Seq-NMS": SEQ_MODEL_OUTPUT_DIR_RANDOM,
    },
}

# Display order for facets: YOLO vs Seq-NMS within each split
SPLIT_ORDER = ["POSTWEAN", "WEAN", "Random"]
MODEL_ORDER  = [f"{m} — {s}" for s in SPLIT_ORDER for m in ["YOLO", "Seq-NMS"]]


# ## Load Predictions

# In[2]:


def _extract_pen(video_path: str) -> str:
    m = re.search(r"Pen\s*(\d+)", video_path, re.IGNORECASE)
    return f"Pen {m.group(1)}" if m else "Unknown"


def _extract_stage(video_path: str) -> str:
    for stage in ("PREWEANING", "WEANING", "POSTWEANING"):
        if stage.lower() in video_path.lower():
            return stage
    return "Unknown"


def load_predictions(input_dir: Path, model_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load per-video prediction JSONs from a directory.

    Parameters
    ----------
    input_dir : Path
        Directory containing per-video prediction JSON files.
    model_label : str
        Label for this model, used as a column value for faceting.

    Returns
    -------
    video_df : pd.DataFrame
        One row per video.
    event_df : pd.DataFrame
        One row per event.
    """
    video_rows, event_rows = [], []

    for path in sorted(Path(input_dir).glob("*.json")):
        with open(path) as f:
            data = json.load(f)

        vid   = data["identifier"]
        pen   = _extract_pen(data.get("video_path", ""))
        stage = _extract_stage(data.get("video_path", ""))

        video_rows.append({
            "model":                  model_label,
            "video":                  vid,
            "pen":                    pen,
            "weaning_stage":          stage,
            "total_duration_sec":     data.get("total_duration_sec"),
            "cross_sucking_detected": data.get("cross_sucking_detected", False),
            "num_events":             data.get("num_events", 0),
        })

        for i, ev in enumerate(data.get("events", [])):
            event_rows.append({
                "model":          model_label,
                "video":          vid,
                "pen":            pen,
                "weaning_stage":  stage,
                "event_idx":      i,
                "start_sec":      ev["start_sec"],
                "end_sec":        ev["end_sec"],
                "duration_sec":   ev["duration_sec"],
                "avg_confidence": ev["avg_confidence"],
            })

    video_df = pd.DataFrame(video_rows)
    event_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame(
        columns=["model", "video", "pen", "weaning_stage", "event_idx",
                 "start_sec", "end_sec", "duration_sec", "avg_confidence"]
    )
    return video_df, event_df


# ── Load all splits and models ─────────────────────────────────────────────
video_dfs, event_dfs = [], []

for split_name, models in SPLITS.items():
    for model_label, path in models.items():
        if not path or not Path(path).exists():
            print(f"[SKIP] {model_label} / {split_name} — path not found: {path}")
            continue
        vdf, edf = load_predictions(path, model_label)
        vdf["split"] = split_name
        edf["split"] = split_name
        video_dfs.append(vdf)
        event_dfs.append(edf)

all_video_df = pd.concat(video_dfs, ignore_index=True)
all_event_df = pd.concat(event_dfs, ignore_index=True)

# Combined label used for all faceting
all_event_df["model_split"] = all_event_df["model"] + " — " + all_event_df["split"]
all_video_df["model_split"] = all_video_df["model"] + " — " + all_video_df["split"]
all_event_df["short_name"]  = all_event_df["video"].str.replace(r"\.mp4$", "", regex=True)
all_video_df["short_name"]  = all_video_df["video"].str.replace(r"\.mp4$", "", regex=True)

print(f"Loaded {len(all_event_df):,} events across {all_event_df['model_split'].nunique()} model/split combinations")
all_event_df.groupby(["split", "model"])[["avg_confidence", "duration_sec", "event_idx"]].agg(
    {"avg_confidence": ["mean", "std"], "duration_sec": ["mean", "std"], "event_idx": "count"}
).round(3)


# ## Evaluation Summary Table
# 
# F2 and recall are highlighted — F2 is the primary metric since missing a real CS event is worse than a false detection.

# In[3]:


import re as _re

def _parse_filename(filename: str) -> dict:
    stem = Path(filename).stem
    m = _re.match(r"evaluation_report_(.+?)_(yolo|seq_nms)$", stem)
    if not m:
        return {"split": stem, "model": "unknown"}
    return {"split": m.group(1), "model": m.group(2)}


def load_evaluation_folder(folder: Path) -> pd.DataFrame:
    json_files = sorted(Path(folder).glob("*.json"))
    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        meta  = _parse_filename(jf.name)
        ev    = data.get("event_level", {})
        seq   = data.get("sequence_level", {})
        frame = data.get("frame_level", {})
        rows.append({
            "split":            meta["split"],
            "model":            meta["model"],
            "TP":               ev.get("true_positives",  0),
            "FP":               ev.get("false_positives", 0),
            "FN":               ev.get("false_negatives", 0),
            "precision":        ev.get("precision",  0.0),
            "recall":           ev.get("recall",     0.0),
            "f2":               ev.get("f2",         0.0),
            "avg_temporal_iou": seq.get("avg_temporal_iou", 0.0),
            "frame_bbox_iou":   frame.get("avg_bbox_iou",   0.0),
        })
    return pd.DataFrame(rows).sort_values(["split", "model"]).reset_index(drop=True)


eval_df = load_evaluation_folder(EVAL_PATH)
eval_df.style.format({
    "precision": "{:.4f}", "recall": "{:.4f}", "f2": "{:.4f}",
    "avg_temporal_iou": "{:.4f}", "frame_bbox_iou": "{:.4f}",
}).background_gradient(subset=["f2", "recall"], cmap="YlGn")


# ## Figure 1 — Confidence Score Distribution
# 
# Six panels (YOLO vs Seq-NMS × POSTWEAN / WEAN / Random) arranged in two rows. Seq-NMS is expected to show lower and more spread confidence than YOLO.

# In[4]:


def _confidence_chart(ms: str) -> alt.Chart:
    df = all_event_df[all_event_df["model_split"] == ms]
    return (
        alt.Chart(df, title=ms)
        .mark_bar(opacity=0.8, color="steelblue")
        .encode(
            alt.X("avg_confidence:Q", bin=alt.Bin(step=0.02), title="Avg Confidence"),
            alt.Y("count():Q", title="# Events"),
            tooltip=["count():Q"],
        )
        .properties(width=260, height=200)
    )

fig1 = alt.vconcat(
    alt.hconcat(
        _confidence_chart("YOLO — POSTWEAN"),
        _confidence_chart("YOLO — WEAN"),
        _confidence_chart("YOLO — Random"),
    ),
    alt.hconcat(
        _confidence_chart("Seq-NMS — POSTWEAN"),
        _confidence_chart("Seq-NMS — WEAN"),
        _confidence_chart("Seq-NMS — Random"),
    ),
).properties(title="Confidence Score Distribution (top: YOLO, bottom: Seq-NMS)")

fig1


# ## Figure 2 — Predicted Events per Video
# 
# Highlights which videos the model is over-detecting on. Uniform tall bars across all videos suggests spam rather than genuine detections.

# In[5]:


def _events_per_video_chart(ms: str) -> alt.Chart:
    df = all_video_df[all_video_df["model_split"] == ms].sort_values("num_events", ascending=False)
    return (
        alt.Chart(df, title=ms)
        .mark_bar()
        .encode(
            alt.X("short_name:N", sort="-y", title="Video",
                  axis=alt.Axis(labelAngle=-60, labelLimit=140)),
            alt.Y("num_events:Q", title="# Events"),
            alt.Color("pen:N", title="Pen"),
            tooltip=["short_name:N", "pen:N", "weaning_stage:N", "num_events:Q"],
        )
        .properties(width=260, height=220)
    )

fig2 = alt.vconcat(
    alt.hconcat(
        _events_per_video_chart("YOLO — POSTWEAN"),
        _events_per_video_chart("YOLO — WEAN"),
        _events_per_video_chart("YOLO — Random"),
    ),
    alt.hconcat(
        _events_per_video_chart("Seq-NMS — POSTWEAN"),
        _events_per_video_chart("Seq-NMS — WEAN"),
        _events_per_video_chart("Seq-NMS — Random"),
    ),
).properties(title="Predicted Events per Video (top: YOLO, bottom: Seq-NMS)")

fig2


# ## Figure 3 — Event Duration Distribution
# 
# Seq-NMS is expected to shift durations longer due to sequence linking. A bimodal distribution (short spike + long tail) suggests the short cluster is mostly FPs.

# In[6]:


def _duration_chart(ms: str) -> alt.Chart:
    df = all_event_df[all_event_df["model_split"] == ms]
    return (
        alt.Chart(df, title=ms)
        .mark_bar(opacity=0.8, color="#F58518")
        .encode(
            alt.X("duration_sec:Q", bin=alt.Bin(maxbins=30), title="Duration (s)"),
            alt.Y("count():Q", title="# Events"),
            tooltip=["count():Q"],
        )
        .properties(width=260, height=200)
    )

fig3 = alt.vconcat(
    alt.hconcat(
        _duration_chart("YOLO — POSTWEAN"),
        _duration_chart("YOLO — WEAN"),
        _duration_chart("YOLO — Random"),
    ),
    alt.hconcat(
        _duration_chart("Seq-NMS — POSTWEAN"),
        _duration_chart("Seq-NMS — WEAN"),
        _duration_chart("Seq-NMS — Random"),
    ),
).properties(title="Event Duration Distribution (top: YOLO, bottom: Seq-NMS)")

fig3


# ## Figure 4 — Confidence vs Event Duration
# 
# Short + low-confidence = likely FPs. Long + high-confidence = strongest TP candidates to pull for manual review.

# In[7]:


def _conf_dur_chart(ms: str) -> alt.Chart:
    df = all_event_df[all_event_df["model_split"] == ms]
    return (
        alt.Chart(df, title=ms)
        .mark_point(filled=True, opacity=0.6, size=35)
        .encode(
            alt.X("duration_sec:Q", title="Duration (s)"),
            alt.Y("avg_confidence:Q", title="Avg Confidence"),
            alt.Color("pen:N", title="Pen"),
            tooltip=["short_name:N", "pen:N", "weaning_stage:N",
                     "duration_sec:Q", "avg_confidence:Q"],
        )
        .properties(width=260, height=220)
    )

fig4 = alt.vconcat(
    alt.hconcat(
        _conf_dur_chart("YOLO — POSTWEAN"),
        _conf_dur_chart("YOLO — WEAN"),
        _conf_dur_chart("YOLO — Random"),
    ),
    alt.hconcat(
        _conf_dur_chart("Seq-NMS — POSTWEAN"),
        _conf_dur_chart("Seq-NMS — WEAN"),
        _conf_dur_chart("Seq-NMS — Random"),
    ),
).properties(title="Confidence vs Duration (top: YOLO, bottom: Seq-NMS)")

fig4


# ## Figure 5 — Temporal Density of Events Within Videos
# 
# Shows *where* in each video events fire. Dot size = duration, colour = confidence. Shown for POSTWEAN only (most TP signal); WEAN and Random follow in Figures 6 and 7.

# In[8]:


def _temporal_chart(ms: str, width: int = 500) -> alt.Chart:
    df = all_event_df[all_event_df["model_split"] == ms].copy()
    n  = df["video"].nunique()
    return (
        alt.Chart(df, title=ms)
        .mark_point(filled=True, opacity=0.75)
        .encode(
            alt.X("start_sec:Q", title="Start Time (s)"),
            alt.Y("short_name:N", title="Video", sort="-x"),
            alt.Size("duration_sec:Q", title="Duration (s)",
                     scale=alt.Scale(range=[10, 220])),
            alt.Color("avg_confidence:Q",
                      scale=alt.Scale(scheme="viridis"),
                      title="Confidence"),
            tooltip=["short_name:N", "start_sec:Q", "end_sec:Q",
                     "duration_sec:Q", "avg_confidence:Q",
                     "pen:N", "weaning_stage:N"],
        )
        .properties(width=width, height=max(240, n * 20))
    )

fig5 = alt.hconcat(
    _temporal_chart("YOLO — POSTWEAN"),
    _temporal_chart("Seq-NMS — POSTWEAN"),
).properties(title="Temporal Density: POSTWEAN (left: YOLO, right: Seq-NMS)")

fig5


# ## Figure 6 — Temporal Density: WEAN

# In[9]:


fig6 = alt.hconcat(
    _temporal_chart("YOLO — WEAN"),
    _temporal_chart("Seq-NMS — WEAN"),
).properties(title="Temporal Density: WEAN (left: YOLO, right: Seq-NMS)")

fig6


# ## Figure 7 — Temporal Density: Random

# In[10]:


fig7 = alt.hconcat(
    _temporal_chart("YOLO — Random"),
    _temporal_chart("Seq-NMS — Random"),
).properties(title="Temporal Density: Random (left: YOLO, right: Seq-NMS)")

fig7


# ## Figure 8 — Confusion Matrix (TP / FP / FN per Model × Split)
# 

# In[11]:


rows = []
for _, meta in eval_df.iterrows():
    split_label = meta["split"].upper()  
    model_label = "YOLO" if meta["model"] == "yolo" else "Seq-NMS"
    ms = f"{model_label} — {split_label}"
    rows.extend([
        {"model_split": ms, "Actual": "CS",    "Predicted": "CS",    "n": int(meta["TP"]), "cell": "TP"},
        {"model_split": ms, "Actual": "No CS", "Predicted": "CS",    "n": int(meta["FP"]), "cell": "FP"},
        {"model_split": ms, "Actual": "CS",    "Predicted": "No CS", "n": int(meta["FN"]), "cell": "FN"},
    ])
cm_df = pd.DataFrame(rows)

alt.data_transformers.enable("default")

# Share base encodings — layer first, then facet once
_base = alt.Chart(cm_df).encode(
    alt.X("Predicted:N", sort=["CS", "No CS"],
          axis=alt.Axis(labelAngle=0), title="Predicted"),
    alt.Y("Actual:N",    sort=["CS", "No CS"], title="Actual"),
)

_rect = _base.mark_rect().encode(
    alt.Color("n:Q", scale=alt.Scale(scheme="blues"), title="Count"),
    tooltip=["model_split:N", "cell:N", "n:Q"],
)

_text = _base.mark_text(fontSize=18, fontWeight="bold").encode(
    alt.Text("n:Q"),
    color=alt.condition(
        "datum.n > 30", alt.value("white"), alt.value("black")
    ),
)

fig9 = (
    (_rect + _text)
    .properties(width=130, height=130)
    .facet(
        facet=alt.Facet("model_split:N", sort=MODEL_ORDER),
        columns=3,
    )
    .properties(title="Event-Level Confusion Matrix by Model and Split")
)

alt.data_transformers.enable("vegafusion")
fig9


# In[12]:


FIG_OUT = Path("../reports/final/img/results")
FIG_OUT.mkdir(parents=True, exist_ok=True)

def save_chart(chart, filename, scale_factor=2.0):
    """Save an Altair chart as PNG to the report img folder."""
    out = FIG_OUT / filename
    # Switch to default transformer for export (vegafusion not needed for static save)
    alt.data_transformers.enable("default")
    alt.data_transformers.disable_max_rows()
    chart.save(str(out), scale_factor=scale_factor)
    alt.data_transformers.enable("vegafusion")
    print(f"Saved {out}")

# ── Figure 8 — Confusion matrix ────────────────────────────────────────────
# Rebuild cm_df and fig in case cell order has changed
_cm_rows = []
for _, meta in eval_df.iterrows():
    split_label = meta["split"].upper()
    model_label = "YOLO" if meta["model"] == "yolo" else "Seq-NMS"
    ms = f"{model_label} — {split_label}"
    _cm_rows.extend([
        {"model_split": ms, "Actual": "CS",    "Predicted": "CS",    "n": int(meta["TP"]), "cell": "TP"},
        {"model_split": ms, "Actual": "No CS", "Predicted": "CS",    "n": int(meta["FP"]), "cell": "FP"},
        {"model_split": ms, "Actual": "CS",    "Predicted": "No CS", "n": int(meta["FN"]), "cell": "FN"},
    ])
_cm_df = pd.DataFrame(_cm_rows)

_base_export = alt.Chart(_cm_df).encode(
    alt.X("Predicted:N", sort=["CS", "No CS"],
          axis=alt.Axis(labelAngle=0), title="Predicted"),
    alt.Y("Actual:N", sort=["CS", "No CS"], title="Actual"),
)
_rect_export = _base_export.mark_rect().encode(
    alt.Color("n:Q", scale=alt.Scale(scheme="blues"), title="Count"),
)
_text_export = _base_export.mark_text(fontSize=18, fontWeight="bold").encode(
    alt.Text("n:Q"),
    color=alt.condition("datum.n > 30", alt.value("white"), alt.value("black")),
)
_fig_confusion_export = (
    (_rect_export + _text_export)
    .properties(width=130, height=130)
    .facet(facet=alt.Facet("model_split:N", sort=MODEL_ORDER), columns=3)
    .properties(title="Event-Level Confusion Matrix by Model and Split")
)
save_chart(_fig_confusion_export, "fig_confusion.png", scale_factor=2.5)

# ── Figures 5–7 — Temporal density (vconcat for PDF, matches report layout) ─
def _temporal_chart_export(ms, width=460):
    df = all_event_df[all_event_df["model_split"] == ms].copy()
    n  = df["video"].nunique()
    return (
        alt.Chart(df, title=ms)
        .mark_point(filled=True, opacity=0.75)
        .encode(
            alt.X("start_sec:Q", title="Start Time (s)"),
            alt.Y("short_name:N", title="Video", sort="-x",
                  axis=alt.Axis(labelLimit=100, labelFontSize=8)),
            alt.Size("duration_sec:Q", title="Duration (s)",
                     scale=alt.Scale(range=[5, 120])),
            alt.Color("avg_confidence:Q",
                      scale=alt.Scale(scheme="viridis"),
                      title="Confidence"),
            tooltip=["short_name:N", "start_sec:Q", "end_sec:Q",
                     "duration_sec:Q", "avg_confidence:Q",
                     "pen:N", "weaning_stage:N"],
        )
        .properties(width=width, height=max(200, n * 16))
    )

save_chart(
    alt.vconcat(
        _temporal_chart_export("YOLO — POSTWEAN"),
        _temporal_chart_export("Seq-NMS — POSTWEAN"),
    ).properties(title="Temporal Density: POSTWEAN"),
    "fig_temporal_postwean.png",
    scale_factor=2.0,
)

save_chart(
    alt.hconcat(
        _temporal_chart_export("YOLO — WEAN"),
        _temporal_chart_export("Seq-NMS — WEAN"),
    ).properties(title="Temporal Density: WEAN"),
    "fig_temporal_wean.png",
    scale_factor=2.0,
)

save_chart(
    alt.vconcat(
        _temporal_chart_export("YOLO — Random"),
        _temporal_chart_export("Seq-NMS — Random"),
    ).properties(title="Temporal Density: Random"),
    "fig_temporal_random.png",
    scale_factor=2.0,
)

print("\nAll figures exported to", FIG_OUT.resolve())

