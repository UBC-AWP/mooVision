"""
test_evaluation.py
------------------
pytest test suite for evaluation.py

A light-coverage suite: 2-3 tests per function, focused on the
core behavior of each one rather than exhaustive edge cases.

Run with:
    pytest tests/test_evaluation.py -v
"""

import json
import io
import zipfile
import pytest
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.evaluation import (
    compute_temporal_iou,
    compute_bbox_iou,
    compute_avg_bbox_iou_for_event,
    compute_frame_level_bbox_iou,
    compute_precision_recall_f,
    match_predictions_to_ground_truth,
    evaluate_by_stratum,
    load_gt_boxes_from_zip,
    load_predictions,
    load_ground_truth,
    generate_evaluation_report,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
)


# ===========================================================================
# Helpers
# ===========================================================================

def make_predictions_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "source_video_basename": "video.mp4",
        "start_sec": 0.0,
        "end_sec": 5.0,
        "duration_sec": 5.0,
        "avg_confidence": 0.9,
        "intersection_box": [],
        "fps": 30.0,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def make_ground_truth_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "source_video_basename": "video.mp4",
        "start_sec": 0.0,
        "end_sec": 5.0,
        "pen": 2,
        "weaning_stage": "PREWEAN",
        "day": 1,
        "labelled_clip_relative_path": "",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def write_cvat_zip(path: Path, annotations: dict):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in annotations.items():
            z.writestr(name, content)
    path.write_bytes(buf.getvalue())


def make_box(frame, x1=0, y1=0, x2=100, y2=100):
    return {"frame": frame, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


# ===========================================================================
# compute_temporal_iou
# ===========================================================================

def test_temporal_iou_perfect_overlap():
    assert compute_temporal_iou(0.0, 5.0, 0.0, 5.0) == pytest.approx(1.0)


def test_temporal_iou_no_overlap():
    assert compute_temporal_iou(0.0, 3.0, 5.0, 8.0) == pytest.approx(0.0)


def test_temporal_iou_partial_overlap():
    # GT: 5→10, Pred: 7→13 → intersection=3, union=8
    assert compute_temporal_iou(7.0, 13.0, 5.0, 10.0) == pytest.approx(3 / 8)


# ===========================================================================
# compute_bbox_iou
# ===========================================================================

def test_bbox_iou_perfect_overlap():
    box = [0, 0, 100, 100]
    assert compute_bbox_iou(box, box) == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    assert compute_bbox_iou([0, 0, 50, 50], [100, 100, 200, 200]) == pytest.approx(0.0)


def test_bbox_iou_normalized_to_smaller_box():
    # small box fully inside large box → score should be 1.0
    result = compute_bbox_iou([0, 0, 200, 200], [50, 50, 100, 100])
    assert result == pytest.approx(1.0)


# ===========================================================================
# compute_avg_bbox_iou_for_event
# ===========================================================================

def test_avg_bbox_iou_exact_match():
    pred = [make_box(10)]
    gt = [make_box(10)]
    assert compute_avg_bbox_iou_for_event(pred, gt) == pytest.approx(1.0)


def test_avg_bbox_iou_empty_inputs():
    assert compute_avg_bbox_iou_for_event([], []) == pytest.approx(0.0)


def test_avg_bbox_iou_outside_tolerance():
    pred = [make_box(10)]
    gt = [make_box(25)]  # 15 frames away, beyond default tolerance of 10
    assert compute_avg_bbox_iou_for_event(pred, gt) == pytest.approx(0.0)


# ===========================================================================
# compute_frame_level_bbox_iou
# ===========================================================================

def test_frame_level_bbox_iou_perfect_match(tmp_path):
    write_cvat_zip(tmp_path / "labels.zip", {
        "obj_train_data/frame_000010.txt": "0 0.5 0.5 0.1 0.1"
    })
    x1, y1 = int(0.45 * VIDEO_WIDTH), int(0.45 * VIDEO_HEIGHT)
    x2, y2 = int(0.55 * VIDEO_WIDTH), int(0.55 * VIDEO_HEIGHT)

    preds = make_predictions_df([{
        "intersection_box": [{"frame": 10, "x1": x1, "y1": y1, "x2": x2, "y2": y2}],
    }])
    gt = make_ground_truth_df([{"labelled_clip_relative_path": "labels.zip"}])

    result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
    assert result == pytest.approx(1.0, abs=0.01)


def test_frame_level_bbox_iou_no_predictions(tmp_path):
    write_cvat_zip(tmp_path / "labels.zip", {
        "obj_train_data/frame_000010.txt": "0 0.5 0.5 0.1 0.1"
    })
    preds = make_predictions_df([])
    gt = make_ground_truth_df([{"labelled_clip_relative_path": "labels.zip"}])
    result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
    assert result == pytest.approx(0.0)


def test_frame_level_bbox_iou_missing_zip_no_crash(tmp_path):
    preds = make_predictions_df([{
        "intersection_box": [make_box(10)],
    }])
    gt = make_ground_truth_df([{"labelled_clip_relative_path": "nonexistent.zip"}])
    result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
    assert result == pytest.approx(0.0)


# ===========================================================================
# compute_precision_recall_f
# ===========================================================================

def test_precision_recall_f_perfect():
    result = compute_precision_recall_f(true_positives=5, false_positives=0, false_negatives=0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f_score"] == pytest.approx(1.0)


def test_precision_recall_f_zero_counts():
    result = compute_precision_recall_f(true_positives=0, false_positives=0, false_negatives=0)
    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)
    assert result["f_score"] == pytest.approx(0.0)


def test_precision_recall_f2_weights_recall_more():
    f1 = compute_precision_recall_f(true_positives=8, false_positives=8, false_negatives=2, beta=1.0)
    f2 = compute_precision_recall_f(true_positives=8, false_positives=8, false_negatives=2, beta=2.0)
    assert f2["f_score"] > f1["f_score"]


# ===========================================================================
# match_predictions_to_ground_truth
# ===========================================================================

def test_match_perfect_overlap_is_tp():
    preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
    gt = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
    result = match_predictions_to_ground_truth(preds, gt)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_match_no_overlap_is_fp_and_fn():
    preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 3.0}])
    gt = make_ground_truth_df([{"start_sec": 10.0, "end_sec": 15.0}])
    result = match_predictions_to_ground_truth(preds, gt)
    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1


def test_match_below_confidence_threshold_ignored():
    preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0, "avg_confidence": 0.3}])
    gt = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
    result = match_predictions_to_ground_truth(preds, gt, confidence_threshold=0.5)
    assert result["true_positives"] == 0
    assert result["false_negatives"] == 1


# ===========================================================================
# evaluate_by_stratum
# ===========================================================================

def test_evaluate_by_stratum_returns_one_row_per_value():
    gt = pd.DataFrame([
        {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0,
         "pen": 2, "weaning_stage": "PREWEAN", "day": 1, "labelled_clip_relative_path": ""},
        {"source_video_basename": "v2.mp4", "start_sec": 0.0, "end_sec": 5.0,
         "pen": 3, "weaning_stage": "WEAN", "day": 2, "labelled_clip_relative_path": ""},
    ])
    preds = make_predictions_df([
        {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0},
        {"source_video_basename": "v2.mp4", "start_sec": 0.0, "end_sec": 5.0},
    ])
    result = evaluate_by_stratum(preds, gt, stratum="pen")
    assert len(result) == 2
    assert set(result["pen"]) == {2, 3}


def test_evaluate_by_stratum_perfect_match_metrics():
    gt = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
    preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
    result = evaluate_by_stratum(preds, gt, stratum="pen")
    row = result.iloc[0]
    assert row["true_positives"] == 1
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)


# ===========================================================================
# load_gt_boxes_from_zip
# ===========================================================================

def test_load_gt_boxes_basic(tmp_path):
    write_cvat_zip(tmp_path / "labels.zip", {
        "obj_train_data/frame_000010.txt": "0 0.5 0.5 0.5 0.5"
    })
    boxes = load_gt_boxes_from_zip("labels.zip", tmp_path, clip_start_frame=0)
    assert len(boxes) == 1
    assert boxes[0]["frame"] == 10


def test_load_gt_boxes_missing_zip_returns_empty(tmp_path):
    boxes = load_gt_boxes_from_zip("nonexistent.zip", tmp_path, clip_start_frame=0)
    assert boxes == []


def test_load_gt_boxes_clip_start_frame_offset(tmp_path):
    write_cvat_zip(tmp_path / "labels.zip", {
        "obj_train_data/frame_000005.txt": "0 0.5 0.5 0.5 0.5"
    })
    boxes = load_gt_boxes_from_zip("labels.zip", tmp_path, clip_start_frame=100)
    assert boxes[0]["frame"] == 105


# ===========================================================================
# load_predictions
# ===========================================================================

def test_load_predictions_basic(tmp_path):
    (tmp_path / "preds.json").write_text(json.dumps({
        "identifier": "video.mp4",
        "fps": 30.0,
        "events": [{
            "start_sec": 1.0, "end_sec": 4.0, "duration_sec": 3.0,
            "avg_confidence": 0.8, "intersection_box": [],
        }],
    }))
    df = load_predictions(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["source_video_basename"] == "video.mp4"


def test_load_predictions_empty_dir(tmp_path):
    df = load_predictions(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_load_predictions_multiple_files_combined(tmp_path):
    for name, vid in [("a.json", "v1.mp4"), ("b.json", "v2.mp4")]:
        (tmp_path / name).write_text(json.dumps({
            "identifier": vid, "fps": 30.0,
            "events": [{"start_sec": 0.0, "end_sec": 5.0, "duration_sec": 5.0,
                        "avg_confidence": 0.9, "intersection_box": []}],
        }))
    df = load_predictions(tmp_path)
    assert len(df) == 2
    assert set(df["source_video_basename"]) == {"v1.mp4", "v2.mp4"}


# ===========================================================================
# load_ground_truth
# ===========================================================================

def test_load_ground_truth_renames_columns(tmp_path):
    p = tmp_path / "gt.csv"
    pd.DataFrame([{
        "source_video_basename": "video.mp4",
        "clip_start_in_source_sec": 1.0,
        "clip_end_in_source_sec": 6.0,
        "phase": "PREWEAN",
        "pen": 2, "day": 1, "labelled_clip_relative_path": "a.zip",
    }]).to_csv(p, index=False)

    df = load_ground_truth(p)
    assert "start_sec" in df.columns
    assert "weaning_stage" in df.columns
    assert df.iloc[0]["start_sec"] == 1.0


def test_load_ground_truth_returns_dataframe(tmp_path):
    p = tmp_path / "gt.csv"
    pd.DataFrame([{
        "source_video_basename": "video.mp4",
        "clip_start_in_source_sec": 1.0,
        "clip_end_in_source_sec": 6.0,
        "phase": "PREWEAN",
        "pen": 2, "day": 1, "labelled_clip_relative_path": "a.zip",
    }]).to_csv(p, index=False)
    df = load_ground_truth(p)
    assert isinstance(df, pd.DataFrame)


# ===========================================================================
# generate_evaluation_report
# ===========================================================================

def test_generate_report_has_expected_keys():
    preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
    gt = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
    report = generate_evaluation_report(preds, gt)
    for key in ("thresholds", "frame_level", "event_level",
                "sequence_level", "by_pen", "by_weaning_stage", "matched_pairs"):
        assert key in report


def test_generate_report_perfect_match():
    preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
    gt = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
    report = generate_evaluation_report(preds, gt)
    el = report["event_level"]
    assert el["true_positives"] == 1
    assert el["precision"] == pytest.approx(1.0)
    assert el["recall"] == pytest.approx(1.0)


def test_generate_report_saves_to_file(tmp_path):
    preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
    gt = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
    output = tmp_path / "report.json"
    generate_evaluation_report(preds, gt, output_path=output)
    assert output.exists()