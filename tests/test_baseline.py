import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.baseline.baseline import compute_iou, frame_has_overlap, extract_events, run_detection

# Compute IoU tests
def test_compute_iou_perfect_overlap():
    """Verify that identical boxes yield an IoU of 1.0 and correct intersection."""
    box_a = [100, 100, 200, 200]
    box_b = [100, 100, 200, 200]
    iou, inter_box = compute_iou(box_a, box_b)
    
    assert iou == 1.0
    assert inter_box == [100, 100, 200, 200]


def test_compute_iou_partial_overlap():
    """Verify standard box math for partial overlap."""
    box_a = [0, 0, 10, 10]    # Area = 100
    box_b = [5, 0, 15, 10]    # Area = 100
    # Intersection is from x=5 to 10, y=0 to 10 -> Area = 50
    # Union = 100 + 100 - 50 = 150
    # Expected IoU = 50 / 150 = 0.3333...
    iou, inter_box = compute_iou(box_a, box_b)
    
    assert pytest.approx(iou, rel=1e-4) == 0.3333
    assert inter_box == [5, 0, 10, 10]


def test_compute_iou_no_overlap():
    """Verify that completely disconnected boxes yield 0.0 IoU."""
    box_a = [0, 0, 50, 50]
    box_b = [100, 100, 150, 150]
    iou, inter_box = compute_iou(box_a, box_b)
    
    assert iou == 0.0
    assert inter_box is None

# Frame. has overlap tests

def test_frame_has_overlap_picks_highest_iou_pair():
    """Ensure frame parsing identifies the highest overlapping pair correctly."""
    boxes = [
        [0, 0, 100, 100],     # Box 0
        [90, 90, 190, 190],   # Box 1 (Tiny overlap with Box 0)
        [10, 10, 90, 90]      # Box 2 (Massive overlap inside Box 0)
    ]
    # The pair (Box 0, Box 2) should easily beat the pair (Box 0, Box 1)
    has_overlap, best_box = frame_has_overlap(boxes, iou_threshold=0.2)
    
    assert has_overlap is True
    assert best_box == [10, 10, 90, 90]

