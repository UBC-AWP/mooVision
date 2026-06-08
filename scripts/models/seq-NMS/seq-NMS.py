"""
seq_nms_inference.py
--------------------
Inference script for cross-sucking detection using a fine-tuned YOLOv26 model
with Seq-NMS (Sequential Non-Maximum Suppression) for temporal linking.
 
Unlike the baseline script which flags frames based on simple bounding box
overlap, this script:
    1. Runs fine-tuned YOLOv26 on each frame to detect cross-sucking directly
    2. Collects all per-frame detections across the full video
    3. Applies Seq-NMS to link detections across frames into consistent tubes
    4. Converts tubes into event windows with start/end times
 
The output JSON format matches baseline.py so evaluation.py works unchanged.
 
Usage:
    uv run python scripts/seq_nms_inference.py \
        --video path/to/video.mp4 \
        --model runs/detect/cross-sucking/weights/best.pt \
        --frame_skip 1
"""
 
import argparse
import json
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
 
DEFAULT_CONF_THRESHOLD    = 0.25   # Lower than baseline since fine-tuned model
                                   # is more specific — catches more real events
DEFAULT_IOU_THRESHOLD     = 0.5    # Minimum spatial IoU to link boxes across frames
DEFAULT_MIN_DURATION      = 1.0    # Minimum tube length in seconds to keep as event
DEFAULT_FRAME_SKIP        = 1      # Process every Nth frame
TARGET_CLASS_NAME         = "cross-sucking"  # Class name from fine-tuned model
                                             # Change to "cow" if using pretrained weights
 
# ---------------------------------------------------------------------------
# STEP 1: BOUNDING BOX IoU
# ---------------------------------------------------------------------------
 
def compute_iou(box_a: list, box_b: list) -> float:
    """
    Compute IoU between two bounding boxes.
 
    Used by Seq-NMS to decide whether two detections in consecutive
    frames belong to the same tube.
 
    Parameters
    ----------
    box_a, box_b : list [x1, y1, x2, y2]
        Bounding boxes in pixel coordinates.
 
    Returns
    -------
    float
        IoU score between 0 and 1.
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
 
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
 
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - intersection
 
    return intersection / union if union > 0 else 0.0