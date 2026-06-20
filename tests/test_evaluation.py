"""
test_evaluation.py
------------------
Comprehensive tests for the MooVision evaluation module.
 
Tests cover:
    - compute_temporal_iou
    - compute_bbox_iou (with camera distance bias correction)
    - compute_precision_recall_f
    - compute_avg_bbox_iou_for_event
    - load_predictions
    - load_ground_truth
    - match_predictions_to_ground_truth
 
Run with:
    uv run pytest tests/test_evaluation.py -v
"""
 
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
 
# Import functions from evaluation script
from scripts.evaluation import (
    compute_temporal_iou,
    compute_bbox_iou,
    compute_precision_recall_f,
    compute_avg_bbox_iou_for_event,
    load_predictions,
    load_ground_truth,
    match_predictions_to_ground_truth,
)

# ===========================================================================
# TESTS: compute_temporal_iou
# ===========================================================================
 
class TestComputeTemporalIou:
    """Tests for compute_temporal_iou()."""
 
    def test_perfect_overlap(self):
        """
        Identical time windows should give IoU of 1.0.
        Both start and end at the same time.
        """
        iou = compute_temporal_iou(5.0, 10.0, 5.0, 10.0)
        assert iou == 1.0
 
    def test_no_overlap(self):
        """
        Non-overlapping time windows should give IoU of 0.0.
        Prediction ends before ground truth starts.
        """
        iou = compute_temporal_iou(1.0, 5.0, 6.0, 10.0)
        assert iou == 0.0
 
    def test_partial_overlap(self):
        """
        Partially overlapping windows should give IoU between 0 and 1.
 
        Prediction:   5.0 -> 10.0  (5 seconds)
        Ground truth: 7.0 -> 13.0  (6 seconds)
        Intersection: 7.0 -> 10.0  (3 seconds)
        Union:        5.0 -> 13.0  (8 seconds)
        IoU = 3/8 = 0.375
        """
        iou = compute_temporal_iou(5.0, 10.0, 7.0, 13.0)
        assert abs(iou - 0.375) < 0.001
 
    def test_prediction_contains_ground_truth(self):
        """
        Prediction window completely contains ground truth window.
        IoU should be less than 1.0 since union is larger than intersection.
        """
        iou = compute_temporal_iou(1.0, 20.0, 5.0, 10.0)
        assert 0.0 < iou < 1.0
 
    def test_ground_truth_contains_prediction(self):
        """
        Ground truth window completely contains prediction window.
        Same as above but reversed.
        """
        iou = compute_temporal_iou(5.0, 10.0, 1.0, 20.0)
        assert 0.0 < iou < 1.0
 
    def test_touching_at_boundary(self):
        """
        Windows that touch exactly at one point should give IoU of 0.0
        since intersection has zero duration.
        """
        iou = compute_temporal_iou(1.0, 5.0, 5.0, 10.0)
        assert iou == 0.0
 
    def test_zero_duration_prediction(self):
        """
        Prediction with zero duration (start == end) should give 0.0.
        Union would be zero so we return 0.0 to avoid division by zero.
        """
        iou = compute_temporal_iou(5.0, 5.0, 5.0, 10.0)
        assert iou == 0.0
 
    def test_zero_duration_ground_truth(self):
        """
        Ground truth with zero duration should give 0.0.
        """
        iou = compute_temporal_iou(5.0, 10.0, 7.0, 7.0)
        assert iou == 0.0
 
    def test_symmetry(self):
        """
        Swapping prediction and ground truth should give the same IoU.
        IoU is symmetric by definition.
        """
        iou_a = compute_temporal_iou(5.0, 10.0, 7.0, 13.0)
        iou_b = compute_temporal_iou(7.0, 13.0, 5.0, 10.0)
        assert abs(iou_a - iou_b) < 0.001
 
    def test_result_between_zero_and_one(self):
        """
        IoU should always be between 0 and 1 for any valid input.
        """
        iou = compute_temporal_iou(3.0, 8.0, 6.0, 12.0)
        assert 0.0 <= iou <= 1.0

# ===========================================================================
# TESTS: compute_bbox_iou
# ===========================================================================
 
class TestComputeBboxIou:
    """
    Tests for compute_bbox_iou().
 
    Note: this function uses min area normalization instead of union
    to correct for camera distance bias. A score of 1.0 means the
    smaller box is completely covered by the overlap.
    """
 
    def test_perfect_overlap(self):
        """
        Identical boxes should give overlap ratio of 1.0.
        """
        box = [0, 0, 100, 100]
        iou = compute_bbox_iou(box, box)
        assert iou == 1.0
 
    def test_no_overlap(self):
        """
        Completely non-overlapping boxes should give 0.0.
        """
        box_pred = [0,   0,  50,  50]
        box_gt   = [60, 60, 110, 110]
        iou = compute_bbox_iou(box_pred, box_gt)
        assert iou == 0.0
 
    def test_partial_overlap(self):
        """
        Partially overlapping boxes should give ratio between 0 and 1.
        """
        box_pred = [0,  0, 100, 100]
        box_gt   = [50, 0, 150, 100]
        iou = compute_bbox_iou(box_pred, box_gt)
        assert 0.0 < iou < 1.0
 
    def test_camera_distance_bias_correction(self):
        """
        Tests the key feature of this function: correcting for camera distance.
 
        Two scenarios with the same proportional overlap but different box sizes
        (simulating calves at different distances from camera).
        Both should give similar overlap ratios, unlike standard IoU which
        would give very different scores.
 
        Small boxes (far away calves):
            pred: 100x100, gt: 100x100, overlap: 80x80
            min_area ratio = 6400/10000 = 0.64
 
        Large boxes (close up calves):
            pred: 400x400, gt: 400x400, overlap: 320x320
            min_area ratio = 102400/160000 = 0.64
 
        Both should give the same score since proportional overlap is identical.
        """
        # Small boxes — far away calves
        small_pred = [0,   0,  100, 100]
        small_gt   = [20,  20, 120, 120]
        iou_small  = compute_bbox_iou(small_pred, small_gt)
 
        # Large boxes — close up calves (4x scale)
        large_pred = [0,   0,  400, 400]
        large_gt   = [80,  80, 480, 480]
        iou_large  = compute_bbox_iou(large_pred, large_gt)
 
        # Scores should be similar despite different box sizes
        assert abs(iou_small - iou_large) < 0.05
 
    def test_smaller_box_fully_inside_larger(self):
        """
        If the smaller box is completely inside the larger box,
        the overlap ratio should be 1.0 since the entire smaller
        box is covered.
        """
        box_pred = [25, 25, 75, 75]    # smaller box
        box_gt   = [0,  0,  100, 100]  # larger box containing pred
        iou = compute_bbox_iou(box_pred, box_gt)
        assert iou == 1.0
 
    def test_zero_area_box(self):
        """
        A box with zero area (point) should return 0.0
        to avoid division by zero.
        """
        box_pred = [50, 50, 50, 50]    # zero area
        box_gt   = [0,  0,  100, 100]
        iou = compute_bbox_iou(box_pred, box_gt)
        assert iou == 0.0
 
    def test_result_between_zero_and_one(self):
        """
        Overlap ratio should always be between 0 and 1.
        """
        box_pred = [10, 10, 60, 60]
        box_gt   = [30, 30, 80, 80]
        iou = compute_bbox_iou(box_pred, box_gt)
        assert 0.0 <= iou <= 1.0
 
    def test_symmetry(self):
        """
        Note: unlike standard IoU, min-area normalization is NOT fully
        symmetric when boxes have different sizes. This test documents
        that behavior so it is understood and expected.
        """
        box_pred = [0,  0,  100, 100]  # larger box
        box_gt   = [25, 25, 75,  75]   # smaller box inside pred
 
        iou_ab = compute_bbox_iou(box_pred, box_gt)
        iou_ba = compute_bbox_iou(box_gt,   box_pred)
 
        # Both should be 1.0 since smaller box is fully inside larger
        assert iou_ab == 1.0
        assert iou_ba == 1.0