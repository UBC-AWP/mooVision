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