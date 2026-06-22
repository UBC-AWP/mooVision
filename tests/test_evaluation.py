"""
test_evaluation.py
------------------
pytest test suite for evaluation.py

Tests cover:
  - compute_temporal_iou
  - compute_bbox_iou
  - compute_avg_bbox_iou_for_event
  - compute_frame_level_bbox_iou
  - compute_precision_recall_f
  - evaluate_by_stratum
  - match_predictions_to_ground_truth
  - load_gt_boxes_from_zip
  - load_predictions
  - load_ground_truth
  - generate_evaluation_report

Run with:
    pytest test_evaluation.py -v
"""

import io
import json
import zipfile
import textwrap
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from evaluation import (
    compute_temporal_iou,
    compute_bbox_iou,
    compute_avg_bbox_iou_for_event,
    compute_frame_level_bbox_iou,
    compute_precision_recall_f,
    match_predictions_to_ground_truth,
    load_gt_boxes_from_zip,
    load_predictions,
    load_ground_truth,
    generate_evaluation_report,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
)


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def make_predictions_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal predictions DataFrame from a list of dicts."""
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
    """Build a minimal ground truth DataFrame from a list of dicts."""
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


def make_cvat_zip(annotations: dict[str, str]) -> bytes:
    """
    Build an in-memory CVAT annotation zip.

    annotations: {filename_inside_zip: yolo_line_content}
    e.g. {"obj_train_data/frame_000010.txt": "0 0.5 0.5 0.4 0.4"}
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in annotations.items():
            z.writestr(name, content)
    return buf.getvalue()


# ===========================================================================
# compute_temporal_iou
# ===========================================================================

class TestComputeTemporalIou:

    def test_perfect_overlap(self):
        assert compute_temporal_iou(0.0, 5.0, 0.0, 5.0) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_temporal_iou(0.0, 3.0, 5.0, 8.0) == pytest.approx(0.0)

    def test_partial_overlap_from_docstring(self):
        # GT: 5→10, Pred: 7→13  → intersection=3, union=8
        result = compute_temporal_iou(7.0, 13.0, 5.0, 10.0)
        assert result == pytest.approx(3 / 8)

    def test_prediction_contained_within_gt(self):
        # Pred: 2→4 fully inside GT: 0→10  → intersection=2, union=10
        result = compute_temporal_iou(2.0, 4.0, 0.0, 10.0)
        assert result == pytest.approx(2 / 10)

    def test_gt_contained_within_prediction(self):
        # GT: 2→4 fully inside Pred: 0→10
        result = compute_temporal_iou(0.0, 10.0, 2.0, 4.0)
        assert result == pytest.approx(2 / 10)

    def test_touching_edges_no_overlap(self):
        # Windows touch but don't overlap
        assert compute_temporal_iou(0.0, 5.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_zero_length_union(self):
        # Degenerate: both zero-length at same point
        result = compute_temporal_iou(3.0, 3.0, 3.0, 3.0)
        assert result == pytest.approx(0.0)

    def test_symmetry(self):
        a = compute_temporal_iou(1.0, 6.0, 4.0, 9.0)
        b = compute_temporal_iou(4.0, 9.0, 1.0, 6.0)
        assert a == pytest.approx(b)

    def test_return_type_is_float(self):
        result = compute_temporal_iou(0.0, 5.0, 0.0, 5.0)
        assert isinstance(result, float)


# ===========================================================================
# compute_bbox_iou
# ===========================================================================

class TestComputeBboxIou:

    def test_perfect_overlap(self):
        box = [0, 0, 100, 100]
        assert compute_bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_bbox_iou([0, 0, 50, 50], [100, 100, 200, 200]) == pytest.approx(0.0)

    def test_partial_overlap_normalised_to_smaller_box(self):
        # pred: 0→100 x 0→100 (area=10000)
        # gt:   50→150 x 50→150 (area=10000)
        # intersection: 50→100 x 50→100 = 50*50 = 2500
        # min_area = 10000  → score = 0.25
        result = compute_bbox_iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert result == pytest.approx(2500 / 10000)

    def test_small_box_fully_inside_large_box(self):
        # Small box fully covered → score should be 1.0 (min_area = small box)
        result = compute_bbox_iou([0, 0, 200, 200], [50, 50, 100, 100])
        assert result == pytest.approx(1.0)

    def test_zero_area_box(self):
        assert compute_bbox_iou([0, 0, 0, 0], [0, 0, 100, 100]) == pytest.approx(0.0)

    def test_symmetry(self):
        a = compute_bbox_iou([0, 0, 60, 60], [40, 40, 100, 100])
        b = compute_bbox_iou([40, 40, 100, 100], [0, 0, 60, 60])
        assert a == pytest.approx(b)

    def test_touching_edges(self):
        assert compute_bbox_iou([0, 0, 50, 50], [50, 0, 100, 50]) == pytest.approx(0.0)

    def test_return_type_is_float(self):
        result = compute_bbox_iou([0, 0, 100, 100], [0, 0, 100, 100])
        assert isinstance(result, float)


# ===========================================================================
# compute_avg_bbox_iou_for_event
# ===========================================================================

class TestComputeAvgBboxIouForEvent:

    def _box(self, frame, x1=0, y1=0, x2=100, y2=100):
        return {"frame": frame, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def test_single_exact_match(self):
        pred = [self._box(10)]
        gt   = [self._box(10)]
        assert compute_avg_bbox_iou_for_event(pred, gt) == pytest.approx(1.0)

    def test_empty_pred_returns_zero(self):
        assert compute_avg_bbox_iou_for_event([], [self._box(10)]) == pytest.approx(0.0)

    def test_empty_gt_returns_zero(self):
        assert compute_avg_bbox_iou_for_event([self._box(10)], []) == pytest.approx(0.0)

    def test_both_empty_returns_zero(self):
        assert compute_avg_bbox_iou_for_event([], []) == pytest.approx(0.0)

    def test_nearest_frame_match_within_tolerance(self):
        pred = [self._box(10)]
        gt   = [self._box(15)]  # 5 frames away, within default tolerance 10
        result = compute_avg_bbox_iou_for_event(pred, gt)
        assert result == pytest.approx(1.0)

    def test_frame_beyond_tolerance_not_matched(self):
        pred = [self._box(10)]
        gt   = [self._box(25)]  # 15 frames away, beyond tolerance 10
        result = compute_avg_bbox_iou_for_event(pred, gt)
        assert result == pytest.approx(0.0)

    def test_gt_frame_matched_only_once(self):
        # Two pred frames both close to same GT frame — GT used only once
        pred = [self._box(10), self._box(11)]
        gt   = [self._box(10)]
        result = compute_avg_bbox_iou_for_event(pred, gt)
        # Only one match → mean of one IoU = 1.0
        assert result == pytest.approx(1.0)

    def test_multiple_frames_averaged(self):
        # Frame 0: perfect overlap (IoU=1.0), Frame 1: no overlap (IoU=0.0)
        pred = [
            self._box(0, 0, 0, 100, 100),
            self._box(1, 0, 0, 100, 100),
        ]
        gt = [
            self._box(0, 0, 0, 100, 100),   # perfect match
            self._box(1, 200, 200, 300, 300),  # no overlap
        ]
        result = compute_avg_bbox_iou_for_event(pred, gt)
        assert result == pytest.approx(0.5)

    def test_custom_tolerance(self):
        pred = [self._box(10)]
        gt   = [self._box(13)]  # 3 frames away
        # With tolerance=2 → no match
        assert compute_avg_bbox_iou_for_event(pred, gt, frame_tolerance=2) == pytest.approx(0.0)
        # With tolerance=5 → match
        assert compute_avg_bbox_iou_for_event(pred, gt, frame_tolerance=5) == pytest.approx(1.0)


# ===========================================================================
# compute_precision_recall_f
# ===========================================================================

class TestComputePrecisionRecallF:

    def test_perfect_predictions(self):
        result = compute_precision_recall_f(tp=5, fp=0, fn=0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"]    == pytest.approx(1.0)
        assert result["f_score"]   == pytest.approx(1.0)

    def test_all_false_positives(self):
        result = compute_precision_recall_f(tp=0, fp=5, fn=0)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"]    == pytest.approx(0.0)
        assert result["f_score"]   == pytest.approx(0.0)

    def test_all_false_negatives(self):
        result = compute_precision_recall_f(tp=0, fp=0, fn=5)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"]    == pytest.approx(0.0)
        assert result["f_score"]   == pytest.approx(0.0)

    def test_zero_counts(self):
        result = compute_precision_recall_f(tp=0, fp=0, fn=0)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"]    == pytest.approx(0.0)
        assert result["f_score"]   == pytest.approx(0.0)

    def test_known_values(self):
        # tp=3, fp=1, fn=2  → precision=3/4=0.75, recall=3/5=0.6
        # F1 = 2*0.75*0.6/(0.75+0.6) = 0.9/1.35 ≈ 0.6667
        result = compute_precision_recall_f(tp=3, fp=1, fn=2, beta=1.0)
        assert result["precision"] == pytest.approx(0.75,   rel=1e-3)
        assert result["recall"]    == pytest.approx(0.6,    rel=1e-3)
        assert result["f_score"]   == pytest.approx(2/3,    rel=1e-3)

    def test_f2_weights_recall_more(self):
        # High recall, low precision scenario
        result_f1 = compute_precision_recall_f(tp=8, fp=8, fn=2, beta=1.0)
        result_f2 = compute_precision_recall_f(tp=8, fp=8, fn=2, beta=2.0)
        # F2 should be higher than F1 when recall > precision
        assert result_f2["f_score"] > result_f1["f_score"]

    def test_rounding_to_4_decimal_places(self):
        result = compute_precision_recall_f(tp=1, fp=2, fn=3)
        for key in ("precision", "recall", "f_score"):
            val = result[key]
            assert val == round(val, 4)

    def test_return_keys(self):
        result = compute_precision_recall_f(tp=1, fp=1, fn=1)
        assert set(result.keys()) == {"precision", "recall", "f_score"}


# ===========================================================================
# match_predictions_to_ground_truth
# ===========================================================================

class TestMatchPredictionsToGroundTruth:

    def test_perfect_match_single_event(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        result = match_predictions_to_ground_truth(preds, gt)
        assert result["true_positives"]  == 1
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0

    def test_no_overlap_is_fp_and_fn(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 3.0}])
        gt    = make_ground_truth_df([{"start_sec": 10.0, "end_sec": 15.0}])
        result = match_predictions_to_ground_truth(preds, gt)
        assert result["true_positives"]  == 0
        assert result["false_positives"] == 1
        assert result["false_negatives"] == 1

    def test_below_confidence_threshold_is_ignored(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0, "avg_confidence": 0.3}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        result = match_predictions_to_ground_truth(preds, gt, confidence_threshold=0.5)
        assert result["true_positives"]  == 0
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 1

    def test_below_temporal_iou_threshold_is_fp(self):
        # Small overlap that won't reach 0.5 IoU
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 2.0}])
        gt    = make_ground_truth_df([{"start_sec": 1.5, "end_sec": 10.0}])
        result = match_predictions_to_ground_truth(preds, gt, temporal_iou_threshold=0.5)
        assert result["true_positives"]  == 0
        assert result["false_positives"] == 1
        assert result["false_negatives"] == 1

    def test_gt_matched_only_once(self):
        # Two predictions overlap same GT event → only first should match
        preds = make_predictions_df([
            {"start_sec": 0.0, "end_sec": 5.0},
            {"start_sec": 0.5, "end_sec": 5.5},
        ])
        gt = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        result = match_predictions_to_ground_truth(preds, gt)
        assert result["true_positives"]  == 1
        assert result["false_positives"] == 1
        assert result["false_negatives"] == 0

    def test_multiple_videos_independent(self):
        preds = make_predictions_df([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0},
            {"source_video_basename": "v2.mp4", "start_sec": 0.0, "end_sec": 5.0},
        ])
        gt = make_ground_truth_df([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0},
            {"source_video_basename": "v2.mp4", "start_sec": 0.0, "end_sec": 5.0},
        ])
        result = match_predictions_to_ground_truth(preds, gt)
        assert result["true_positives"]  == 2
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0

    def test_no_predictions_all_fn(self):
        preds = make_predictions_df([])
        gt    = make_ground_truth_df([
            {"start_sec": 0.0, "end_sec": 5.0},
            {"start_sec": 10.0, "end_sec": 15.0},
        ])
        result = match_predictions_to_ground_truth(preds, gt)
        assert result["true_positives"]  == 0
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 2

    def test_no_ground_truth_all_fp(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([])
        result = match_predictions_to_ground_truth(preds, gt)
        assert result["true_positives"]  == 0
        assert result["false_positives"] == 1
        assert result["false_negatives"] == 0

    def test_matched_pairs_populated(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        result = match_predictions_to_ground_truth(preds, gt)
        assert len(result["matched_pairs"]) == 1
        pair = result["matched_pairs"][0]
        assert "temporal_iou" in pair
        assert "bbox_iou"     in pair
        assert pair["temporal_iou"] == pytest.approx(1.0)

    def test_temporal_ious_list_length_equals_tp(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        result = match_predictions_to_ground_truth(preds, gt)
        assert len(result["temporal_ious"]) == result["true_positives"]


# ===========================================================================
# load_gt_boxes_from_zip
# ===========================================================================

class TestLoadGtBoxesFromZip:

    def _write_zip(self, tmp_path: Path, annotations: dict) -> Path:
        zip_bytes = make_cvat_zip(annotations)
        zip_file = tmp_path / "labels.zip"
        zip_file.write_bytes(zip_bytes)
        return zip_file

    def test_basic_single_frame(self, tmp_path):
        # YOLO: class cx cy w h  (normalized, 1920x1080)
        # cx=0.5 cy=0.5 w=0.5 h=0.5 → x1=480, y1=270, x2=1440, y2=810
        zip_file = self._write_zip(tmp_path, {
            "obj_train_data/frame_000010.txt": "0 0.5 0.5 0.5 0.5"
        })
        boxes = load_gt_boxes_from_zip(
            labelled_clip_relative_path="labels.zip",
            labelled_clips_dir=tmp_path,
            clip_start_frame=0,
        )
        assert len(boxes) == 1
        b = boxes[0]
        assert b["frame"] == 10
        assert b["x1"] == int((0.5 - 0.25) * VIDEO_WIDTH)
        assert b["y1"] == int((0.5 - 0.25) * VIDEO_HEIGHT)
        assert b["x2"] == int((0.5 + 0.25) * VIDEO_WIDTH)
        assert b["y2"] == int((0.5 + 0.25) * VIDEO_HEIGHT)

    def test_clip_start_frame_offset_applied(self, tmp_path):
        zip_file = self._write_zip(tmp_path, {
            "obj_train_data/frame_000005.txt": "0 0.5 0.5 0.5 0.5"
        })
        boxes = load_gt_boxes_from_zip(
            labelled_clip_relative_path="labels.zip",
            labelled_clips_dir=tmp_path,
            clip_start_frame=100,
        )
        assert boxes[0]["frame"] == 105  # 5 + 100

    def test_missing_zip_returns_empty_list(self, tmp_path):
        boxes = load_gt_boxes_from_zip(
            labelled_clip_relative_path="nonexistent.zip",
            labelled_clips_dir=tmp_path,
            clip_start_frame=0,
        )
        assert boxes == []

    def test_empty_txt_file_skipped(self, tmp_path):
        zip_file = self._write_zip(tmp_path, {
            "obj_train_data/frame_000001.txt": ""
        })
        boxes = load_gt_boxes_from_zip(
            labelled_clip_relative_path="labels.zip",
            labelled_clips_dir=tmp_path,
            clip_start_frame=0,
        )
        assert boxes == []

    def test_malformed_txt_skipped(self, tmp_path):
        zip_file = self._write_zip(tmp_path, {
            "obj_train_data/frame_000001.txt": "0 0.5"  # too few parts
        })
        boxes = load_gt_boxes_from_zip(
            labelled_clip_relative_path="labels.zip",
            labelled_clips_dir=tmp_path,
            clip_start_frame=0,
        )
        assert boxes == []

    def test_non_annotation_files_ignored(self, tmp_path):
        zip_file = self._write_zip(tmp_path, {
            "obj_train_data/frame_000001.txt": "0 0.5 0.5 0.5 0.5",
            "README.md":                       "This is a readme",
            "obj_train_data/classes.txt":      "cross_sucking",
        })
        boxes = load_gt_boxes_from_zip(
            labelled_clip_relative_path="labels.zip",
            labelled_clips_dir=tmp_path,
            clip_start_frame=0,
        )
        assert len(boxes) == 1

    def test_multiple_frames_all_loaded(self, tmp_path):
        zip_file = self._write_zip(tmp_path, {
            "obj_train_data/frame_000001.txt": "0 0.5 0.5 0.5 0.5",
            "obj_train_data/frame_000002.txt": "0 0.3 0.3 0.2 0.2",
            "obj_train_data/frame_000003.txt": "0 0.7 0.7 0.3 0.3",
        })
        boxes = load_gt_boxes_from_zip(
            labelled_clip_relative_path="labels.zip",
            labelled_clips_dir=tmp_path,
            clip_start_frame=0,
        )
        assert len(boxes) == 3

    def test_backslash_path_normalised(self, tmp_path):
        zip_bytes = make_cvat_zip({"obj_train_data/frame_000001.txt": "0 0.5 0.5 0.5 0.5"})
        (tmp_path / "labels.zip").write_bytes(zip_bytes)
        # Simulate Windows-style path with backslashes
        boxes = load_gt_boxes_from_zip(
            labelled_clip_relative_path="labels.zip".replace("/", "\\"),
            labelled_clips_dir=tmp_path,
            clip_start_frame=0,
        )
        assert len(boxes) == 1


# ===========================================================================
# load_predictions
# ===========================================================================

class TestLoadPredictions:

    def _write_json(self, tmp_path: Path, data: dict, filename: str = "preds.json") -> Path:
        p = tmp_path / filename
        p.write_text(json.dumps(data))
        return p

    def _base_metadata(self, identifier="video.mp4", fps=30.0, events=None):
        return {
            "identifier": identifier,
            "fps": fps,
            "events": events or [],
        }

    def test_basic_load(self, tmp_path):
        self._write_json(tmp_path, self._base_metadata(
            events=[{
                "start_sec": 1.0,
                "end_sec": 4.0,
                "duration_sec": 3.0,
                "avg_confidence": 0.8,
                "intersection_box": [],
            }]
        ))
        df = load_predictions(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["source_video_basename"] == "video.mp4"
        assert df.iloc[0]["start_sec"] == 1.0
        assert df.iloc[0]["avg_confidence"] == 0.8

    def test_empty_directory_returns_empty_df(self, tmp_path):
        df = load_predictions(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_multiple_json_files_combined(self, tmp_path):
        self._write_json(tmp_path, self._base_metadata("v1.mp4", events=[{
            "start_sec": 0.0, "end_sec": 5.0, "duration_sec": 5.0,
            "avg_confidence": 0.9, "intersection_box": [],
        }]), "v1.json")
        self._write_json(tmp_path, self._base_metadata("v2.mp4", events=[{
            "start_sec": 10.0, "end_sec": 15.0, "duration_sec": 5.0,
            "avg_confidence": 0.7, "intersection_box": [],
        }]), "v2.json")
        df = load_predictions(tmp_path)
        assert len(df) == 2
        assert set(df["source_video_basename"]) == {"v1.mp4", "v2.mp4"}

    def test_no_events_key_returns_empty(self, tmp_path):
        self._write_json(tmp_path, {"identifier": "video.mp4", "fps": 30.0})
        df = load_predictions(tmp_path)
        assert len(df) == 0

    def test_intersection_box_defaults_to_empty_list(self, tmp_path):
        self._write_json(tmp_path, self._base_metadata(events=[{
            "start_sec": 0.0, "end_sec": 5.0, "duration_sec": 5.0,
            "avg_confidence": 0.9,
            # no "intersection_box" key
        }]))
        df = load_predictions(tmp_path)
        assert df.iloc[0]["intersection_box"] == []

    def test_required_columns_present(self, tmp_path):
        self._write_json(tmp_path, self._base_metadata(events=[{
            "start_sec": 0.0, "end_sec": 5.0, "duration_sec": 5.0,
            "avg_confidence": 0.9, "intersection_box": [],
        }]))
        df = load_predictions(tmp_path)
        for col in ("source_video_basename", "start_sec", "end_sec",
                    "duration_sec", "avg_confidence", "intersection_box", "fps"):
            assert col in df.columns


# ===========================================================================
# load_ground_truth
# ===========================================================================

class TestLoadGroundTruth:

    def _write_csv(self, tmp_path: Path, rows: list[dict]) -> Path:
        p = tmp_path / "gt.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def test_basic_load_and_column_rename(self, tmp_path):
        p = self._write_csv(tmp_path, [{
            "source_video_basename": "video.mp4",
            "clip_start_in_source_sec": 1.0,
            "clip_end_in_source_sec": 6.0,
            "phase": "PREWEAN",
            "pen": 2,
            "day": 1,
            "labelled_clip_relative_path": "zip/file.zip",
        }])
        df = load_ground_truth(p)
        assert "start_sec"     in df.columns
        assert "end_sec"       in df.columns
        assert "weaning_stage" in df.columns
        assert "clip_start_in_source_sec" not in df.columns
        assert df.iloc[0]["start_sec"] == 1.0
        assert df.iloc[0]["weaning_stage"] == "PREWEAN"

    def test_returns_dataframe(self, tmp_path):
        p = self._write_csv(tmp_path, [])
        df = load_ground_truth(p)
        assert isinstance(df, pd.DataFrame)

    def test_multiple_rows(self, tmp_path):
        p = self._write_csv(tmp_path, [
            {"source_video_basename": "v1.mp4", "clip_start_in_source_sec": 0.0,
             "clip_end_in_source_sec": 5.0, "phase": "PREWEAN", "pen": 2, "day": 1,
             "labelled_clip_relative_path": "a.zip"},
            {"source_video_basename": "v2.mp4", "clip_start_in_source_sec": 10.0,
             "clip_end_in_source_sec": 20.0, "phase": "WEAN", "pen": 3, "day": 2,
             "labelled_clip_relative_path": "b.zip"},
        ])
        df = load_ground_truth(p)
        assert len(df) == 2


# ===========================================================================
# generate_evaluation_report
# ===========================================================================

class TestGenerateEvaluationReport:

    def test_report_keys_present(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        report = generate_evaluation_report(preds, gt)
        for key in ("thresholds", "frame_level", "event_level",
                    "sequence_level", "by_pen", "by_weaning_stage", "matched_pairs"):
            assert key in report

    def test_perfect_predictions_full_report(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        report = generate_evaluation_report(preds, gt)
        el = report["event_level"]
        assert el["true_positives"]  == 1
        assert el["false_positives"] == 0
        assert el["false_negatives"] == 0
        assert el["precision"] == pytest.approx(1.0)
        assert el["recall"]    == pytest.approx(1.0)
        assert el["f1"]        == pytest.approx(1.0)

    def test_no_matches_all_fp_fn(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 3.0}])
        gt    = make_ground_truth_df([{"start_sec": 20.0, "end_sec": 25.0}])
        report = generate_evaluation_report(preds, gt)
        el = report["event_level"]
        assert el["true_positives"]  == 0
        assert el["false_positives"] == 1
        assert el["false_negatives"] == 1

    def test_saves_report_to_file(self, tmp_path):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        output = tmp_path / "report.json"
        generate_evaluation_report(preds, gt, output_path=output)
        assert output.exists()
        loaded = json.loads(output.read_text())
        assert "event_level" in loaded

    def test_thresholds_reflected_in_report(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        report = generate_evaluation_report(
            preds, gt,
            temporal_iou_threshold=0.3,
            confidence_threshold=0.7,
        )
        assert report["thresholds"]["temporal_iou_threshold"] == 0.3
        assert report["thresholds"]["confidence_threshold"]   == 0.7

    def test_frame_level_bbox_iou_zero_without_labelled_clips_dir(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        report = generate_evaluation_report(preds, gt, labelled_clips_dir=None)
        assert report["frame_level"]["avg_bbox_iou"] == pytest.approx(0.0)

    def test_by_pen_and_by_weaning_stage_are_lists(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        report = generate_evaluation_report(preds, gt)
        assert isinstance(report["by_pen"],           list)
        assert isinstance(report["by_weaning_stage"], list)

    def test_sequence_level_avg_temporal_iou(self):
        preds = make_predictions_df([{"start_sec": 0.0, "end_sec": 5.0}])
        gt    = make_ground_truth_df([{"start_sec": 0.0, "end_sec": 5.0}])
        report = generate_evaluation_report(preds, gt)
        assert report["sequence_level"]["avg_temporal_iou"] == pytest.approx(1.0)

    def test_empty_predictions_and_gt(self):
        preds  = make_predictions_df([])
        gt     = make_ground_truth_df([])
        report = generate_evaluation_report(preds, gt)
        el = report["event_level"]
        assert el["true_positives"]  == 0
        assert el["false_positives"] == 0
        assert el["false_negatives"] == 0


# ===========================================================================
# compute_frame_level_bbox_iou
# ===========================================================================

class TestComputeFrameLevelBboxIou:
    """
    Tests for compute_frame_level_bbox_iou.

    This function is integration-heavy: it loads CVAT zips from disk and
    correlates predicted frames with GT frames across all videos. The
    strategy here is to write real zip files to tmp_path so the file I/O
    path is exercised, while keeping bounding boxes simple so the expected
    IoU is easy to calculate by hand.
    """

    # ---- shared helpers -------------------------------------------------------

    def _write_zip(self, directory: Path, name: str, annotations: dict) -> None:
        """Write a CVAT annotation zip to directory/name."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for fname, content in annotations.items():
                z.writestr(fname, content)
        (directory / name).write_bytes(buf.getvalue())

    def _pred_with_boxes(self, video: str, start: float, end: float, boxes: list) -> dict:
        """Shorthand for a prediction row that includes intersection_box frames."""
        return {
            "source_video_basename": video,
            "start_sec":            start,
            "end_sec":              end,
            "duration_sec":         end - start,
            "avg_confidence":       0.9,
            "intersection_box":     boxes,
            "fps":                  30.0,
        }

    def _frame_box(self, frame: int, x1=0, y1=0, x2=100, y2=100) -> dict:
        return {"frame": frame, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

    # ---- tests ----------------------------------------------------------------

    def test_perfect_overlap_single_video(self, tmp_path):
        """Predicted box exactly matches GT box → IoU = 1.0."""
        # GT box: cx=0.5 cy=0.5 w=0.1 h=0.1 → x1=912 y1=486 x2=1008 y2=594
        self._write_zip(tmp_path, "labels.zip", {
            "obj_train_data/frame_000010.txt": "0 0.5 0.5 0.1 0.1"
        })
        # Build the matching predicted pixel box
        x1 = int((0.5 - 0.05) * VIDEO_WIDTH)
        y1 = int((0.5 - 0.05) * VIDEO_HEIGHT)
        x2 = int((0.5 + 0.05) * VIDEO_WIDTH)
        y2 = int((0.5 + 0.05) * VIDEO_HEIGHT)

        preds = make_predictions_df([self._pred_with_boxes(
            "video.mp4", 0.0, 5.0,
            [self._frame_box(10, x1, y1, x2, y2)]
        )])
        gt = make_ground_truth_df([{
            "source_video_basename":       "video.mp4",
            "start_sec":                   0.0,
            "end_sec":                     5.0,
            "labelled_clip_relative_path": "labels.zip",
        }])

        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_no_overlap_returns_zero(self, tmp_path):
        """Predicted box and GT box don't overlap → IoU = 0.0 → excluded from mean."""
        self._write_zip(tmp_path, "labels.zip", {
            "obj_train_data/frame_000010.txt": "0 0.1 0.1 0.05 0.05"
        })
        preds = make_predictions_df([self._pred_with_boxes(
            "video.mp4", 0.0, 5.0,
            # Box in opposite corner — no overlap
            [self._frame_box(10, 1800, 1000, 1920, 1080)]
        )])
        gt = make_ground_truth_df([{
            "source_video_basename":       "video.mp4",
            "start_sec":                   0.0,
            "end_sec":                     5.0,
            "labelled_clip_relative_path": "labels.zip",
        }])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
        # IoU == 0 so no entries added to all_ious → returns 0.0
        assert result == pytest.approx(0.0)

    def test_no_predictions_returns_zero(self, tmp_path):
        self._write_zip(tmp_path, "labels.zip", {
            "obj_train_data/frame_000010.txt": "0 0.5 0.5 0.1 0.1"
        })
        preds = make_predictions_df([])
        gt    = make_ground_truth_df([{
            "source_video_basename":       "video.mp4",
            "start_sec":                   0.0,
            "end_sec":                     5.0,
            "labelled_clip_relative_path": "labels.zip",
        }])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
        assert result == pytest.approx(0.0)

    def test_no_ground_truth_for_video_returns_zero(self, tmp_path):
        """Predictions exist but no GT rows for that video → returns 0.0."""
        preds = make_predictions_df([self._pred_with_boxes(
            "video.mp4", 0.0, 5.0, [self._frame_box(10)]
        )])
        gt = make_ground_truth_df([{
            "source_video_basename":       "other_video.mp4",
            "start_sec":                   0.0,
            "end_sec":                     5.0,
            "labelled_clip_relative_path": "labels.zip",
        }])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
        assert result == pytest.approx(0.0)

    def test_missing_zip_skipped_gracefully(self, tmp_path):
        """GT row pointing to a non-existent zip → no crash, returns 0.0."""
        preds = make_predictions_df([self._pred_with_boxes(
            "video.mp4", 0.0, 5.0, [self._frame_box(10)]
        )])
        gt = make_ground_truth_df([{
            "source_video_basename":       "video.mp4",
            "start_sec":                   0.0,
            "end_sec":                     5.0,
            "labelled_clip_relative_path": "nonexistent.zip",
        }])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
        assert result == pytest.approx(0.0)

    def test_prediction_without_intersection_box_skipped(self, tmp_path):
        """Predictions with empty intersection_box are skipped."""
        self._write_zip(tmp_path, "labels.zip", {
            "obj_train_data/frame_000010.txt": "0 0.5 0.5 0.1 0.1"
        })
        preds = make_predictions_df([{
            "source_video_basename": "video.mp4",
            "start_sec":            0.0,
            "end_sec":              5.0,
            "duration_sec":         5.0,
            "avg_confidence":       0.9,
            "intersection_box":     [],   # empty — should be skipped
            "fps":                  30.0,
        }])
        gt = make_ground_truth_df([{
            "source_video_basename":       "video.mp4",
            "start_sec":                   0.0,
            "end_sec":                     5.0,
            "labelled_clip_relative_path": "labels.zip",
        }])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
        assert result == pytest.approx(0.0)

    def test_clip_start_frame_offset_used_correctly(self, tmp_path):
        """
        GT clip starts at 10s (300 frames at 30fps).
        CVAT zip has frame_000005.txt → source frame = 305.
        Prediction also uses source frame 305 → should match.
        """
        self._write_zip(tmp_path, "labels.zip", {
            "obj_train_data/frame_000005.txt": "0 0.5 0.5 0.5 0.5"
        })
        x1 = int(0.25 * VIDEO_WIDTH)
        y1 = int(0.25 * VIDEO_HEIGHT)
        x2 = int(0.75 * VIDEO_WIDTH)
        y2 = int(0.75 * VIDEO_HEIGHT)

        preds = make_predictions_df([self._pred_with_boxes(
            "video.mp4", 0.0, 20.0,
            [self._frame_box(305, x1, y1, x2, y2)]  # source frame 305
        )])
        gt = make_ground_truth_df([{
            "source_video_basename":       "video.mp4",
            "start_sec":                   10.0,   # → clip_start_frame = 300
            "end_sec":                     15.0,
            "labelled_clip_relative_path": "labels.zip",
        }])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path, fps=30.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_multiple_gt_clips_aggregated(self, tmp_path):
        """GT boxes from two clips for the same video are pooled together."""
        self._write_zip(tmp_path, "clip1.zip", {
            "obj_train_data/frame_000000.txt": "0 0.5 0.5 0.5 0.5"
        })
        self._write_zip(tmp_path, "clip2.zip", {
            "obj_train_data/frame_000000.txt": "0 0.5 0.5 0.5 0.5"
        })
        x1 = int(0.25 * VIDEO_WIDTH)
        y1 = int(0.25 * VIDEO_HEIGHT)
        x2 = int(0.75 * VIDEO_WIDTH)
        y2 = int(0.75 * VIDEO_HEIGHT)

        preds = make_predictions_df([self._pred_with_boxes(
            "video.mp4", 0.0, 30.0,
            [self._frame_box(0, x1, y1, x2, y2),
             self._frame_box(300, x1, y1, x2, y2)]
        )])
        gt = pd.DataFrame([
            {"source_video_basename": "video.mp4", "start_sec": 0.0,
             "end_sec": 5.0,  "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": "clip1.zip"},
            {"source_video_basename": "video.mp4", "start_sec": 10.0,
             "end_sec": 15.0, "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": "clip2.zip"},
        ])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path, fps=30.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_multiple_videos_averaged(self, tmp_path):
        """Result is averaged across multiple videos."""
        for name in ("v1.zip", "v2.zip"):
            self._write_zip(tmp_path, name, {
                "obj_train_data/frame_000000.txt": "0 0.5 0.5 0.5 0.5"
            })
        x1 = int(0.25 * VIDEO_WIDTH)
        y1 = int(0.25 * VIDEO_HEIGHT)
        x2 = int(0.75 * VIDEO_WIDTH)
        y2 = int(0.75 * VIDEO_HEIGHT)

        preds = pd.DataFrame([
            {**{"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0,
                "duration_sec": 5.0, "avg_confidence": 0.9, "fps": 30.0,
                "intersection_box": [self._frame_box(0, x1, y1, x2, y2)]}},
            {**{"source_video_basename": "v2.mp4", "start_sec": 0.0, "end_sec": 5.0,
                "duration_sec": 5.0, "avg_confidence": 0.9, "fps": 30.0,
                "intersection_box": [self._frame_box(0, x1, y1, x2, y2)]}},
        ])
        gt = pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0,
             "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": "v1.zip"},
            {"source_video_basename": "v2.mp4", "start_sec": 0.0, "end_sec": 5.0,
             "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": "v2.zip"},
        ])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_result_is_rounded_to_4_decimal_places(self, tmp_path):
        self._write_zip(tmp_path, "labels.zip", {
            "obj_train_data/frame_000010.txt": "0 0.5 0.5 0.1 0.1"
        })
        x1 = int(0.25 * VIDEO_WIDTH)
        y1 = int(0.25 * VIDEO_HEIGHT)
        x2 = int(0.75 * VIDEO_WIDTH)
        y2 = int(0.75 * VIDEO_HEIGHT)

        preds = make_predictions_df([self._pred_with_boxes(
            "video.mp4", 0.0, 5.0, [self._frame_box(10, x1, y1, x2, y2)]
        )])
        gt = make_ground_truth_df([{
            "source_video_basename":       "video.mp4",
            "start_sec":                   0.0,
            "end_sec":                     5.0,
            "labelled_clip_relative_path": "labels.zip",
        }])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
        assert result == round(result, 4)

    def test_gt_row_without_zip_path_skipped(self, tmp_path):
        """GT rows with empty labelled_clip_relative_path contribute no GT boxes."""
        preds = make_predictions_df([self._pred_with_boxes(
            "video.mp4", 0.0, 5.0, [self._frame_box(10)]
        )])
        gt = make_ground_truth_df([{
            "source_video_basename":       "video.mp4",
            "start_sec":                   0.0,
            "end_sec":                     5.0,
            "labelled_clip_relative_path": "",  # no zip → skipped
        }])
        result = compute_frame_level_bbox_iou(preds, gt, labelled_clips_dir=tmp_path)
        assert result == pytest.approx(0.0)


# ===========================================================================
# evaluate_by_stratum
# ===========================================================================

class TestEvaluateByStratum:

    def _multi_stratum_gt(self) -> pd.DataFrame:
        """GT with two pens and two weaning stages across two videos."""
        return pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0,  "end_sec": 5.0,
             "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": ""},
            {"source_video_basename": "v2.mp4", "start_sec": 0.0,  "end_sec": 5.0,
             "pen": 3, "weaning_stage": "WEAN",    "day": 2,
             "labelled_clip_relative_path": ""},
        ])

    def _multi_stratum_preds(self) -> pd.DataFrame:
        """Predictions that perfectly match the GT above."""
        return pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0,
             "duration_sec": 5.0, "avg_confidence": 0.9,
             "intersection_box": [], "fps": 30.0,
             "pen": 2, "weaning_stage": "PREWEAN"},
            {"source_video_basename": "v2.mp4", "start_sec": 0.0, "end_sec": 5.0,
             "duration_sec": 5.0, "avg_confidence": 0.9,
             "intersection_box": [], "fps": 30.0,
             "pen": 3, "weaning_stage": "WEAN"},
        ])

    def test_returns_dataframe(self):
        from evaluation import evaluate_by_stratum
        gt    = self._multi_stratum_gt()
        preds = self._multi_stratum_preds()
        result = evaluate_by_stratum(preds, gt, stratum="pen")
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_stratum_value(self):
        from evaluation import evaluate_by_stratum
        gt    = self._multi_stratum_gt()
        preds = self._multi_stratum_preds()
        result = evaluate_by_stratum(preds, gt, stratum="pen")
        assert len(result) == 2  # pen 2 and pen 3
        assert set(result["pen"]) == {2, 3}

    def test_expected_columns_present(self):
        from evaluation import evaluate_by_stratum
        gt    = self._multi_stratum_gt()
        preds = self._multi_stratum_preds()
        result = evaluate_by_stratum(preds, gt, stratum="pen")
        for col in ("pen", "true_positives", "false_positives", "false_negatives",
                    "precision", "recall", "f1", "f2",
                    "avg_temporal_iou", "avg_bbox_iou"):
            assert col in result.columns

    def test_perfect_match_per_stratum(self):
        from evaluation import evaluate_by_stratum
        gt    = self._multi_stratum_gt()
        preds = self._multi_stratum_preds()
        result = evaluate_by_stratum(preds, gt, stratum="pen")
        for _, row in result.iterrows():
            assert row["true_positives"]  == 1
            assert row["false_positives"] == 0
            assert row["false_negatives"] == 0
            assert row["precision"] == pytest.approx(1.0)
            assert row["recall"]    == pytest.approx(1.0)
            assert row["f1"]        == pytest.approx(1.0)

    def test_weaning_stage_stratum(self):
        from evaluation import evaluate_by_stratum
        gt    = self._multi_stratum_gt()
        preds = self._multi_stratum_preds()
        result = evaluate_by_stratum(preds, gt, stratum="weaning_stage")
        assert len(result) == 2
        assert set(result["weaning_stage"]) == {"PREWEAN", "WEAN"}

    def test_strata_are_independent(self):
        """A miss in one stratum should not affect another."""
        from evaluation import evaluate_by_stratum
        gt = pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0,  "end_sec": 5.0,
             "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": ""},
            {"source_video_basename": "v2.mp4", "start_sec": 0.0,  "end_sec": 5.0,
             "pen": 3, "weaning_stage": "WEAN",    "day": 2,
             "labelled_clip_relative_path": ""},
        ])
        # Only a prediction for pen 2 — pen 3 will be a missed FN
        preds = pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0,
             "duration_sec": 5.0, "avg_confidence": 0.9,
             "intersection_box": [], "fps": 30.0, "pen": 2},
        ])
        result = evaluate_by_stratum(preds, gt, stratum="pen")
        pen2 = result[result["pen"] == 2].iloc[0]
        pen3 = result[result["pen"] == 3].iloc[0]
        assert pen2["true_positives"]  == 1
        assert pen2["false_negatives"] == 0
        assert pen3["true_positives"]  == 0
        assert pen3["false_negatives"] == 1

    def test_stratum_not_in_predictions_uses_all_preds(self):
        """
        When the stratum column isn't in predictions (e.g. pen isn't tagged
        on each prediction row), all predictions are used for every stratum
        value rather than filtering to an empty subset.
        """
        from evaluation import evaluate_by_stratum
        gt = pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0,
             "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": ""},
            {"source_video_basename": "v1.mp4", "start_sec": 10.0, "end_sec": 15.0,
             "pen": 3, "weaning_stage": "WEAN",    "day": 1,
             "labelled_clip_relative_path": ""},
        ])
        # Predictions do NOT have a "pen" column
        preds = pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0,
             "duration_sec": 5.0, "avg_confidence": 0.9,
             "intersection_box": [], "fps": 30.0},
            {"source_video_basename": "v1.mp4", "start_sec": 10.0, "end_sec": 15.0,
             "duration_sec": 5.0, "avg_confidence": 0.9,
             "intersection_box": [], "fps": 30.0},
        ])
        # Should not raise; all preds used for each pen subset
        result = evaluate_by_stratum(preds, gt, stratum="pen")
        assert len(result) == 2
        # Both pen subsets should find their matching prediction
        for _, row in result.iterrows():
            assert row["true_positives"] >= 1

    def test_f2_greater_than_f1_when_recall_dominates(self):
        """When recall > precision, F2 should exceed F1 within a stratum."""
        from evaluation import evaluate_by_stratum
        # Two GT events, one prediction that matches one → recall=0.5, precision=1.0
        gt = pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0,  "end_sec": 5.0,
             "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": ""},
            {"source_video_basename": "v1.mp4", "start_sec": 20.0, "end_sec": 25.0,
             "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": ""},
        ])
        preds = pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0,
             "duration_sec": 5.0, "avg_confidence": 0.9,
             "intersection_box": [], "fps": 30.0},
        ])
        result = evaluate_by_stratum(preds, gt, stratum="pen")
        row = result[result["pen"] == 2].iloc[0]
        # precision=1.0, recall=0.5 → F2 weights recall more → F2 < F1 here
        # (recall < precision so F1 > F2 in this case — assert they differ)
        assert row["f1"] != pytest.approx(row["f2"])

    def test_avg_temporal_iou_zero_when_no_matches(self):
        from evaluation import evaluate_by_stratum
        gt = pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 0.0, "end_sec": 5.0,
             "pen": 2, "weaning_stage": "PREWEAN", "day": 1,
             "labelled_clip_relative_path": ""},
        ])
        # Prediction far outside GT window → FP, no TP → no temporal IoUs
        preds = pd.DataFrame([
            {"source_video_basename": "v1.mp4", "start_sec": 50.0, "end_sec": 55.0,
             "duration_sec": 5.0, "avg_confidence": 0.9,
             "intersection_box": [], "fps": 30.0},
        ])
        result = evaluate_by_stratum(preds, gt, stratum="pen")
        assert result.iloc[0]["avg_temporal_iou"] == pytest.approx(0.0)