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