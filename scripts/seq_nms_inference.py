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

# ---------------------------------------------------------------------------
# STEP 2: SEQ-NMS — BUILD TUBES
# ---------------------------------------------------------------------------
 
def build_tubes(
    frame_detections: list,
    iou_threshold: float,
) -> list:
    """
    Link per-frame detections into tubes using Seq-NMS.
 
    A tube is a sequence of bounding boxes linked across consecutive frames
    that all belong to the same detected event. Think of it as tracking
    one cross-sucking interaction through time.
 
    How it works:
        For each frame, for each detection in that frame:
            - Look at all active tubes from the previous frame
            - If this detection overlaps with the last box in a tube
              above iou_threshold, extend that tube
            - Otherwise start a new tube
 
    Parameters
    ----------
    frame_detections : list of list of dict
        Per-frame detections. Each entry is a list of detections for that
        frame. Each detection is a dict with keys:
            frame (int), x1, y1, x2, y2 (float), confidence (float)
    iou_threshold : float
        Minimum IoU between consecutive boxes to link them into the same tube.
 
    Returns
    -------
    list of list of dict
        Each tube is a list of detection dicts linked across frames.
        Example:
            [
                [  # tube 1
                    {"frame": 10, "x1": 100, "y1": 200, "x2": 300, "y2": 400, "confidence": 0.8},
                    {"frame": 11, "x1": 102, "y1": 201, "x2": 302, "y2": 401, "confidence": 0.75},
                ],
                [  # tube 2
                    ...
                ]
            ]
    """
    active_tubes    = []  # tubes still being extended
    completed_tubes = []  # tubes that have ended
 
    for frame_dets in frame_detections:
        if not frame_dets:
            # No detections this frame — close all active tubes
            completed_tubes.extend(active_tubes)
            active_tubes = []
            continue
 
        # Track which active tubes got extended this frame
        extended = set()
        new_tubes = []
 
        for det in frame_dets:
            best_iou   = 0.0
            best_tube  = None
            best_idx   = None
 
            # Try to find the best active tube to extend
            for i, tube in enumerate(active_tubes):
                if i in extended:
                    continue  # tube already extended this frame
 
                last_box = tube[-1]
                iou = compute_iou(
                    [det["x1"], det["y1"], det["x2"], det["y2"]],
                    [last_box["x1"], last_box["y1"], last_box["x2"], last_box["y2"]]
                )
                if iou > best_iou:
                    best_iou  = iou
                    best_tube = tube
                    best_idx  = i
 
            if best_iou >= iou_threshold and best_tube is not None:
                # Extend existing tube
                best_tube.append(det)
                extended.add(best_idx)
            else:
                # Start a new tube
                new_tubes.append([det])
 
        # Close any active tubes that were not extended this frame
        for i, tube in enumerate(active_tubes):
            if i not in extended:
                completed_tubes.append(tube)
 
        # Keep extended tubes + new tubes as active for next frame
        active_tubes = [
            tube for i, tube in enumerate(active_tubes) if i in extended
        ] + new_tubes
 
    # Close any remaining active tubes at end of video
    completed_tubes.extend(active_tubes)
 
    return completed_tubes

# ---------------------------------------------------------------------------
# STEP 3: SEQ-NMS — SUPPRESS WEAK DETECTIONS WITHIN TUBES
# ---------------------------------------------------------------------------
 
def suppress_weak_detections(
    tubes: list,
    conf_threshold: float
) -> list:
    """
    Remove low confidence detections from within each tube.
 
    After linking, some frames within a tube may have low confidence
    detections. This step removes them to clean up the tube, keeping
    only frames where the model was confident enough.
 
    Tubes that become empty after suppression are discarded.
 
    Parameters
    ----------
    tubes : list
        Output of build_tubes().
    conf_threshold : float
        Minimum confidence score to keep a detection within a tube.
 
    Returns
    -------
    list
        Cleaned tubes with low confidence detections removed.
    """
    cleaned = []
    for tube in tubes:
        filtered = [det for det in tube if det["confidence"] >= conf_threshold]
        if filtered:
            cleaned.append(filtered)
    return cleaned
 
 # ---------------------------------------------------------------------------
# STEP 4: CONVERT TUBES TO EVENTS
# ---------------------------------------------------------------------------
 
def tubes_to_events(
    tubes: list,
    fps: float,
    min_duration: float,
) -> list:
    """
    Convert Seq-NMS tubes into event dictionaries.
 
    Takes the linked tubes and converts them into the same event format
    used by baseline.py so the rest of the pipeline (clipping, evaluation)
    works without any changes.
 
    Tubes shorter than min_duration are discarded as noise.
 
    Parameters
    ----------
    tubes : list
        Output of suppress_weak_detections().
    fps : float
        Frames per second of the video — used to convert frame numbers
        to timestamps in seconds.
    min_duration : float
        Minimum event duration in seconds. Shorter tubes are discarded.
 
    Returns
    -------
    list of dict
        Each dict is one event with keys:
            start_sec, end_sec, duration_sec, avg_confidence,
            intersection_box (per-frame box coordinates)
    """
    events = []
 
    for tube in tubes:
        start_frame = tube[0]["frame"]
        end_frame   = tube[-1]["frame"]
        duration    = (end_frame - start_frame) / fps
 
        # Discard short tubes
        if duration < min_duration:
            continue
 
        avg_confidence = float(np.mean([det["confidence"] for det in tube]))
 
        # Build per-frame intersection box list — matches baseline.py format
        intersection_box = [
            {
                "frame": det["frame"],
                "x1":    int(det["x1"]),
                "y1":    int(det["y1"]),
                "x2":    int(det["x2"]),
                "y2":    int(det["y2"]),
            }
            for det in tube
        ]
 
        events.append({
            "start_sec":        round(start_frame / fps, 2),
            "end_sec":          round(end_frame / fps, 2),
            "duration_sec":     round(duration, 2),
            "avg_confidence":   round(avg_confidence, 3),
            "intersection_box": intersection_box,
        })
 
    return events