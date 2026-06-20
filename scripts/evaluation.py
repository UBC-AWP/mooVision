"""
evaluation.py
-------------
Evaluation module for the MooVision cross-sucking detection pipeline.

Compares baseline model predictions (JSON) against ground truth annotations
(processed clips index CSV) across three levels:
  - Frame level:    bounding box IoU for every predicted frame vs ground truth
                    frame, independent of temporal matching. Measures spatial
                    detection accuracy of the YOLO model directly.
  - Event level:    bounding box IoU, precision, recall, F1, F2 for matched
                    events (only events that passed temporal IoU threshold).
  - Sequence level: temporal IoU — how well predicted event time windows
                    overlap with labeled event time windows.

How to run:
    python scripts/evaluation.py \
        --predictions results/metadata/baseline/ \
        --ground_truth data/raw/all_clips_index_raw.csv \
        --output results/evaluation_report.json \
        --labelled_clips_dir /path/to/cross_sucking_labelled \
        --fps 30.0
"""

import json
import zipfile
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Video dimensions — used to convert YOLO normalized coords to pixels
VIDEO_WIDTH  = 1920
VIDEO_HEIGHT = 1080


# ===========================================================================
# STEP 1: LOADING DATA
# ===========================================================================
# These functions load the predictions and ground truth into DataFrames
# so the rest of the script can work with them in a consistent format.

def load_predictions(predictions_dir: Path) -> pd.DataFrame:
    """
    Load all prediction JSON files from a directory into a DataFrame.

    Each JSON file is produced by running inference (baseline.py or
    seq_NMS.py) on one source video. It contains a list of detected
    events with start/end times, confidence scores, and per-frame
    bounding boxes.

    This function reads all JSON files in the directory and flattens
    them into a single DataFrame where each row is one predicted event.

    Parameters
    ----------
    predictions_dir : Path
        Folder containing prediction JSON files
        (e.g. results/metadata/seq_nms/).

    Returns
    -------
    pd.DataFrame
        One row per predicted event. Columns:
        source_video_basename, start_sec, end_sec, duration_sec,
        avg_confidence, intersection_box, fps.
        Returns empty DataFrame if no JSON files found.
    """
    records = []

    for json_file in Path(predictions_dir).glob("*.json"):
        with open(json_file) as f:
            metadata = json.load(f)

        # identifier is the source video filename e.g. ch02_20251102075200.mp4
        source_video_basename = metadata["identifier"]

        for event in metadata.get("events", []):
            records.append({
                "source_video_basename": source_video_basename,
                "start_sec":            event["start_sec"],
                "end_sec":              event["end_sec"],
                "duration_sec":         event["duration_sec"],
                "avg_confidence":       event["avg_confidence"],
                "intersection_box":     event.get("intersection_box", []),
                "fps":                  metadata["fps"],
            })

    return pd.DataFrame(records)


def load_ground_truth(path: Path) -> pd.DataFrame:
    """
    Load ground truth annotations from the processed clips index CSV.

    This CSV is produced by read_all_clips_index.py and contains one
    row per confirmed cross-sucking clip, with timing, pen, weaning
    stage, day, and the path to the corresponding CVAT annotation zip
    file for bounding box evaluation.

    Key columns used for evaluation:
        - source_video_basename:    links ground truth to predictions
        - clip_start_in_source_sec: CS event start time in source video
        - clip_end_in_source_sec:   CS event end time in source video
        - pen:                      which pen the calf was in
        - phase:                    preweaning / weaning / postweaning
        - day:                      which day of the observation period
        - labelled_clip_relative_path: path to CVAT zip file with
                                        ground truth bounding boxes

    Parameters
    ----------
    path : Path
        Path to the processed clips index CSV.

    Returns
    -------
    pd.DataFrame
        One row per confirmed cross-sucking event, with renamed columns:
        clip_start_in_source_sec → start_sec
        clip_end_in_source_sec   → end_sec
        phase                    → weaning_stage
    """
    df = pd.read_csv(path)

    # Rename timing columns to match prediction column names
    # so comparisons are straightforward
    df = df.rename(columns={
        "clip_start_in_source_sec": "start_sec",
        "clip_end_in_source_sec":   "end_sec",
        "phase":                    "weaning_stage",
    })

    return df


def load_gt_boxes_from_zip(
    labelled_clip_relative_path: str,
    labelled_clips_dir: Path,
    clip_start_frame: int,
    img_width: int = VIDEO_WIDTH,
    img_height: int = VIDEO_HEIGHT,
) -> list:
    """
    Load ground truth bounding boxes from a CVAT annotation zip file.

    CVAT exports annotations as zip files containing per-frame .txt
    files in YOLO format:
        class_id  center_x  center_y  width  height
    All coordinate values are normalized between 0 and 1 relative to
    the image dimensions.

    This function:
        1. Opens the zip file from labelled_clips_dir
        2. Reads each frame_XXXXXX.txt annotation file
        3. Converts normalized YOLO coordinates to pixel coordinates
           using VIDEO_WIDTH and VIDEO_HEIGHT (1920x1080)
        4. Offsets clip-level frame numbers by clip_start_frame so
           they align with source video frame numbers used in predictions

    Parameters
    ----------
    labelled_clip_relative_path : str
        Relative path to the zip file from the ground truth CSV column
        labelled_clip_relative_path.
    labelled_clips_dir : Path
        Root directory of CVAT annotation zip files on OneDrive
        (cross_sucking_labelled/).
    clip_start_frame : int
        Frame number in the source video where this clip starts.
        Computed as int(clip_start_sec * fps). Used to convert
        clip-level frame numbers to source video frame numbers.
    img_width : int
        Video frame width in pixels. Default 1920.
    img_height : int
        Video frame height in pixels. Default 1080.

    Returns
    -------
    list of dict
        Per-frame ground truth boxes in pixel coordinates:
        [{"frame": int, "x1": int, "y1": int, "x2": int, "y2": int}, ...]
        Returns empty list if zip file not found or contains no annotations.
    """
    zip_path = labelled_clips_dir / labelled_clip_relative_path.replace("\\", "/")

    if not zip_path.exists():
        return []

    gt_boxes = []

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            txt_files = [
                f for f in z.namelist()
                if f.startswith("obj_train_data/") and f.endswith(".txt")
            ]

            for txt_file in txt_files:
                # Extract frame number from filename e.g. frame_000441.txt → 441
                # String split is faster than regex for this fixed format
                try:
                    clip_frame_num = int(txt_file.split("frame_")[1].replace(".txt", ""))
                except (IndexError, ValueError):
                    continue

                # Offset clip frame number to source video frame number
                source_frame_num = clip_frame_num + clip_start_frame

                with z.open(txt_file) as f:
                    content = f.read().decode().strip()
                    if not content:
                        continue

                    # Parse YOLO format: class_id cx cy w h
                    parts = content.split()
                    if len(parts) < 5:
                        continue

                    cx = float(parts[1])
                    cy = float(parts[2])
                    w  = float(parts[3])
                    h  = float(parts[4])

                    # Convert normalized coords to pixel coords
                    x1 = int((cx - w / 2) * img_width)
                    y1 = int((cy - h / 2) * img_height)
                    x2 = int((cx + w / 2) * img_width)
                    y2 = int((cy + h / 2) * img_height)

                    gt_boxes.append({
                        "frame": source_frame_num,
                        "x1":    x1,
                        "y1":    y1,
                        "x2":    x2,
                        "y2":    y2,
                    })

    except Exception as e:
        print(f"[WARN] Could not read {zip_path}: {e}")
        return []

    return gt_boxes


# ===========================================================================
# STEP 2: CORE METRICS
# ===========================================================================
# These functions compute the actual evaluation metrics.
# They are pure math functions that take numbers and return numbers.

def compute_temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float
) -> float:
    """
    Compute temporal IoU between a predicted and ground truth event window.

    Measures how well the predicted time window overlaps with the labeled
    time window. Works exactly like bounding box IoU but in 1D (time
    instead of 2D space):
        - Intersection: the overlapping time segment
        - Union: total time covered by both windows
        - IoU = intersection / union

    Example:
        Ground truth: 5.0s → 10.0s  (5 seconds)
        Prediction:   7.0s → 13.0s  (6 seconds)
        Intersection: 7.0s → 10.0s  (3 seconds)
        Union:        5.0s → 13.0s  (8 seconds)
        Temporal IoU = 3 / 8 = 0.375

    Parameters
    ----------
    pred_start : float
        Predicted event start time in seconds.
    pred_end : float
        Predicted event end time in seconds.
    gt_start : float
        Ground truth event start time in seconds.
    gt_end : float
        Ground truth event end time in seconds.

    Returns
    -------
    float
        Temporal IoU score between 0.0 and 1.0.
        0.0 means no overlap, 1.0 means perfect overlap.
    """
    intersection_start = max(pred_start, gt_start)
    intersection_end   = min(pred_end,   gt_end)
    intersection = max(0.0, intersection_end - intersection_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    return intersection / union if union > 0 else 0.0


def compute_bbox_iou(box_pred: list, box_gt: list) -> float:
    """
    Compute bounding box overlap normalized by the smaller box area.

    Instead of standard IoU (which divides by union), this divides by
    the minimum of the two box areas. This corrects for camera distance
    bias: calves closer to the camera have larger bounding boxes, which
    would unfairly inflate standard IoU scores. By normalizing to the
    smaller box, the score reflects how well the predicted box covers
    the ground truth region regardless of box size or camera distance.

    Both boxes use [x1, y1, x2, y2] format (pixel coordinates), which
    matches the intersection_box format from baseline.py and seq_NMS.py.

    Parameters
    ----------
    box_pred : list [x1, y1, x2, y2]
        Predicted bounding box in pixel coordinates.
    box_gt : list [x1, y1, x2, y2]
        Ground truth bounding box in pixel coordinates.

    Returns
    -------
    float
        Overlap ratio between 0.0 and 1.0.
        1.0 means the smaller box is completely covered by the overlap.
        0.0 means no overlap at all.
    """
    inter_x1 = max(box_pred[0], box_gt[0])
    inter_y1 = max(box_pred[1], box_gt[1])
    inter_x2 = min(box_pred[2], box_gt[2])
    inter_y2 = min(box_pred[3], box_gt[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt   = (box_gt[2]   - box_gt[0])   * (box_gt[3]   - box_gt[1])

    min_area = min(area_pred, area_gt)

    return intersection / min_area if min_area > 0 else 0.0


def compute_avg_bbox_iou_for_event(
    pred_boxes: list,
    gt_boxes: list,
    frame_tolerance: int = 10,
) -> float:
    """
    Compute average bounding box IoU across all frames in a matched event.

    Each event stores per-frame bounding boxes. This function averages
    the bbox IoU over all frames where both a predicted box and a ground
    truth box are available.

    Matching uses the NEAREST ground truth frame within frame_tolerance
    rather than requiring an exact frame number match. This is necessary
    because:
        - Predictions using frame_skip (e.g. every 5th frame) produce
          frame numbers that rarely align exactly with ground truth
          frame numbers from the CVAT zip files.
        - Small fps rounding differences when converting timestamps to
          frame numbers can shift frame numbers by 1-2 frames.

    Each ground truth frame is only matched once (closest prediction
    wins) to avoid one ground truth box being reused for many nearby
    predicted frames.

    Parameters
    ----------
    pred_boxes : list of dict
        Per-frame predicted boxes:
        [{"frame": int, "x1": int, "y1": int, "x2": int, "y2": int}, ...]
    gt_boxes : list of dict
        Per-frame ground truth boxes in the same format.
    frame_tolerance : int
        Maximum frame distance to consider a ground truth box a match
        for a predicted box. Default 10 frames (~0.3s at 30fps).

    Returns
    -------
    float
        Average bbox IoU across all matched frames.
        Returns 0.0 if no frames match within tolerance.
    """
    if not pred_boxes or not gt_boxes:
        return 0.0

    gt_sorted = sorted(gt_boxes, key=lambda b: b["frame"])
    used_gt_frames = set()
    ious = []

    for pb in pred_boxes:
        pred_frame = pb["frame"]

        best_gt   = None
        best_dist = None
        for gb in gt_sorted:
            if gb["frame"] in used_gt_frames:
                continue
            dist = abs(gb["frame"] - pred_frame)
            if dist <= frame_tolerance and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_gt   = gb

        if best_gt is not None:
            iou = compute_bbox_iou(
                [pb["x1"], pb["y1"], pb["x2"], pb["y2"]],
                [best_gt["x1"], best_gt["y1"], best_gt["x2"], best_gt["y2"]]
            )
            ious.append(iou)
            used_gt_frames.add(best_gt["frame"])

    return float(np.mean(ious)) if ious else 0.0


def compute_frame_level_bbox_iou(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    labelled_clips_dir: Path,
    fps: float = 30.0,
    frame_tolerance: int = 10,
) -> float:
    """
    Compute frame-level bounding box IoU across ALL predicted frames.

    This is the primary metric for evaluating the YOLO CS detection
    model independently of event-level temporal matching. It measures
    how accurately the model draws bounding boxes around cross-sucking
    interactions regardless of whether the event timing was correct.

    Unlike avg_bbox_iou in the event-level report (which only computes
    bbox IoU for events that already passed the temporal IoU threshold),
    this function computes bbox IoU for every predicted frame that has
    a corresponding ground truth annotation in the CVAT zip files.

    For each predicted event:
        1. Find all ground truth clips for the same source video
        2. Load ground truth boxes from each clip's CVAT zip file
        3. For each predicted frame box, find the nearest ground truth
           frame within frame_tolerance
        4. Compute bbox IoU for matched frame pairs
        5. Average across all matched frames across all videos

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of load_predictions(). Must contain intersection_box
        column with per-frame predicted bounding boxes.
    ground_truth : pd.DataFrame
        Output of load_ground_truth(). Must contain
        labelled_clip_relative_path column linking to CVAT zip files.
    labelled_clips_dir : Path
        Root directory of CVAT annotation zip files.
    fps : float
        Frames per second — used to convert clip start times to frame
        numbers for offsetting ground truth frame numbers.
        Default 30.0.
    frame_tolerance : int
        Maximum frame distance for nearest-frame matching.
        Default 10 frames (~0.3s at 30fps).

    Returns
    -------
    float
        Average bbox IoU across all matched predicted frames.
        Returns 0.0 if no frames could be matched.
    """
    all_ious = []

    print(f"Processing {predictions['source_video_basename'].nunique()} unique videos from predictions")

    # Process each source video that appears in predictions
    for video in predictions["source_video_basename"].unique():
        video_preds = predictions[
            predictions["source_video_basename"] == video
        ].to_dict("records")

        # Get all ground truth clips for this source video
        video_gt = ground_truth[
            ground_truth["source_video_basename"] == video
        ].to_dict("records")

        if not video_gt:
            print(f"{video} -> no matching ground truth rows, skipping")
            continue

        # Load all ground truth boxes for this video from zip files
        all_gt_boxes = []
        for gt_event in video_gt:
            if not gt_event.get("labelled_clip_relative_path"):
                print(f"{video} -> xxx")
                continue
            clip_start_frame = int(gt_event["start_sec"] * fps)
            gt_boxes = load_gt_boxes_from_zip(
                labelled_clip_relative_path=gt_event["labelled_clip_relative_path"],
                labelled_clips_dir=labelled_clips_dir,
                clip_start_frame=clip_start_frame,
                img_width=VIDEO_WIDTH,
                img_height=VIDEO_HEIGHT,
            )
            all_gt_boxes.extend(gt_boxes)

        print(f"{video} -> {len(video_gt)} gt clips, {len(all_gt_boxes)} total gt boxes loaded, {len(video_preds)} predicted events")

        if not all_gt_boxes:
            print(f"{video} -> no gt boxes available, skipping")
            continue

        # Compute bbox IoU for each predicted event's frames
        for pred in video_preds:
            if not pred.get("intersection_box"):
                continue
            iou = compute_avg_bbox_iou_for_event(
                pred["intersection_box"],
                all_gt_boxes,
                frame_tolerance=frame_tolerance,
            )
            if iou > 0:
                all_ious.append(iou)

    print(f"Matched {len(all_ious)} predicted events with non-zero bbox IoU")

    return round(float(np.mean(all_ious)), 4) if all_ious else 0.0


def compute_precision_recall_f(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    beta: float = 1.0
) -> dict:
    """
    Compute precision, recall, and F-beta score from TP/FP/FN counts.

    - Precision: of all events the model flagged, how many were real?
                 High precision = few false alarms.
    - Recall:    of all real events, how many did the model find?
                 High recall = few missed events.
    - F1:        balanced harmonic mean of precision and recall.
    - F2:        recall weighted twice as heavily as precision.
                 Used in this project because missing a real CS event
                 is more costly than occasionally flagging a false one.

    Parameters
    ----------
    true_positives : int
        Predicted events that correctly match a ground truth event.
    false_positives : int
        Predicted events with no matching ground truth event.
    false_negatives : int
        Ground truth events the model missed entirely.
    beta : float
        Beta value for F-score. 1.0 = F1 (balanced), 2.0 = F2
        (recall-weighted). Default 1.0.

    Returns
    -------
    dict
        Keys: precision (float), recall (float), f_score (float).
        All values between 0.0 and 1.0.
    """
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0 else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0 else 0.0
    )
    beta_sq = beta ** 2
    f_score = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if (beta_sq * precision + recall) > 0 else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f_score":   round(f_score, 4),
    }


# ===========================================================================
# STEP 3: MATCHING PREDICTIONS TO GROUND TRUTH
# ===========================================================================
# The core logic that decides which predictions are correct.
# Loops through each video, pairs each predicted event with the best
# matching ground truth event using temporal IoU, and counts TP/FP/FN.

def match_predictions_to_ground_truth(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    temporal_iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    labelled_clips_dir: Path = None,
    fps: float = 30.0,
) -> dict:
    """
    Match predicted events to ground truth events using temporal IoU.

    Matching logic per source video:
        For each predicted event above confidence_threshold:
            - Find the ground truth event with the highest temporal IoU
            - If best temporal IoU >= temporal_iou_threshold → True Positive
              and compute bbox IoU for that matched pair using CVAT zip files
            - Otherwise → False Positive
        Any ground truth events with no matching prediction → False Negative

    Each ground truth event can only be matched once (greedy matching)
    to prevent one real event from counting as multiple true positives.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of load_predictions().
    ground_truth : pd.DataFrame
        Output of load_ground_truth().
    temporal_iou_threshold : float
        Minimum temporal IoU to count a prediction as a True Positive.
        Default 0.5.
    confidence_threshold : float
        Minimum avg_confidence score to consider a prediction.
        Predictions below this are ignored entirely. Default 0.5.
    labelled_clips_dir : Path, optional
        Root directory of CVAT annotation zip files. If provided,
        bbox IoU is computed for each True Positive using ground truth
        boxes from the zip file. If None, bbox IoU is 0.0.
    fps : float
        Frames per second of source videos. Used to convert clip start
        times to frame numbers for zip file offset. Default 30.0.

    Returns
    -------
    dict
        Keys:
            true_positives (int): correctly detected events
            false_positives (int): incorrectly flagged events
            false_negatives (int): missed real events
            matched_pairs (list of dict): details of each TP match
            temporal_ious (list of float): temporal IoU per TP
            bbox_ious (list of float): bbox IoU per TP
    """
    preds = predictions[
        predictions["avg_confidence"] >= confidence_threshold
    ].copy()

    true_positives  = 0
    false_positives = 0
    false_negatives = 0
    matched_pairs   = []
    temporal_ious   = []
    bbox_ious       = []

    all_videos = set(preds["source_video_basename"]).union(
        set(ground_truth["source_video_basename"])
    )

    for video in all_videos:
        video_preds = preds[
            preds["source_video_basename"] == video
        ].to_dict("records")

        video_gt = ground_truth[
            ground_truth["source_video_basename"] == video
        ].to_dict("records")

        matched_gt = set()

        for pred in video_preds:
            best_iou    = 0.0
            best_gt_idx = None

            for gt_idx, gt in enumerate(video_gt):
                if gt_idx in matched_gt:
                    continue
                t_iou = compute_temporal_iou(
                    pred["start_sec"], pred["end_sec"],
                    gt["start_sec"],   gt["end_sec"]
                )
                if t_iou > best_iou:
                    best_iou    = t_iou
                    best_gt_idx = gt_idx

            if best_iou >= temporal_iou_threshold and best_gt_idx is not None:
                true_positives += 1
                matched_gt.add(best_gt_idx)
                temporal_ious.append(best_iou)

                gt_event = video_gt[best_gt_idx]
                b_iou = 0.0
                if pred.get("intersection_box") and labelled_clips_dir and gt_event.get("labelled_clip_relative_path"):
                    clip_start_frame = int(gt_event["start_sec"] * fps)
                    gt_boxes = load_gt_boxes_from_zip(
                        labelled_clip_relative_path=gt_event["labelled_clip_relative_path"],
                        labelled_clips_dir=labelled_clips_dir,
                        clip_start_frame=clip_start_frame,
                        img_width=VIDEO_WIDTH,
                        img_height=VIDEO_HEIGHT,
                    )
                    if gt_boxes:
                        b_iou = compute_avg_bbox_iou_for_event(
                            pred["intersection_box"],
                            gt_boxes
                        )
                elif pred.get("intersection_box") and gt_event.get("intersection_box"):
                    b_iou = compute_avg_bbox_iou_for_event(
                        pred["intersection_box"],
                        gt_event["intersection_box"]
                    )
                bbox_ious.append(b_iou)

                matched_pairs.append({
                    "video":        video,
                    "pred_start":   pred["start_sec"],
                    "pred_end":     pred["end_sec"],
                    "gt_start":     gt_event["start_sec"],
                    "gt_end":       gt_event["end_sec"],
                    "temporal_iou": round(best_iou, 4),
                    "bbox_iou":     round(b_iou, 4),
                    "confidence":   pred["avg_confidence"],
                })
            else:
                false_positives += 1

        false_negatives += len(video_gt) - len(matched_gt)

    return {
        "true_positives":  true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matched_pairs":   matched_pairs,
        "temporal_ious":   temporal_ious,
        "bbox_ious":       bbox_ious,
    }


# ===========================================================================
# STEP 4: STRATIFIED EVALUATION
# ===========================================================================
# Breaks evaluation down by pen or weaning stage so the partner can ask:
# "Does the model perform better in certain pens or weaning stages?"

def evaluate_by_stratum(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    stratum: str,
    temporal_iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    labelled_clips_dir: Path = None,
    fps: float = 30.0,
) -> pd.DataFrame:
    """
    Evaluate model performance broken down by a grouping variable.

    Runs the full matching and metric computation separately for each
    unique value of the stratum column — for example each pen value
    (2, 3, 5) or each weaning stage (PREWEAN, WEAN, POSTWEAN).

    This directly supports the partner's research questions about
    whether the model performs differently across different pens or
    developmental stages.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of load_predictions().
    ground_truth : pd.DataFrame
        Output of load_ground_truth().
    stratum : str
        Column to group by. One of: 'pen', 'weaning_stage', 'day'.
    temporal_iou_threshold : float
        Passed to match_predictions_to_ground_truth(). Default 0.5.
    confidence_threshold : float
        Passed to match_predictions_to_ground_truth(). Default 0.5.
    labelled_clips_dir : Path, optional
        Passed to match_predictions_to_ground_truth() for bbox IoU.
    fps : float
        Frames per second. Default 30.0.

    Returns
    -------
    pd.DataFrame
        One row per unique stratum value with columns:
        stratum_value, true_positives, false_positives, false_negatives,
        precision, recall, f1, f2, avg_temporal_iou, avg_bbox_iou.
    """
    results = []

    for value in ground_truth[stratum].unique():
        preds_subset = predictions[predictions[stratum] == value] \
            if stratum in predictions.columns else predictions
        gt_subset = ground_truth[ground_truth[stratum] == value]

        match_result = match_predictions_to_ground_truth(
            preds_subset, gt_subset,
            temporal_iou_threshold, confidence_threshold,
            labelled_clips_dir=labelled_clips_dir,
            fps=fps,
        )

        tp = match_result["true_positives"]
        fp = match_result["false_positives"]
        fn = match_result["false_negatives"]

        f1 = compute_precision_recall_f(tp, fp, fn, beta=1.0)
        f2 = compute_precision_recall_f(tp, fp, fn, beta=2.0)

        results.append({
            stratum:             value,
            "true_positives":    tp,
            "false_positives":   fp,
            "false_negatives":   fn,
            "precision":         f1["precision"],
            "recall":            f1["recall"],
            "f1":                f1["f_score"],
            "f2":                f2["f_score"],
            "avg_temporal_iou":  round(float(np.mean(
                                     match_result["temporal_ious"])), 4)
                                 if match_result["temporal_ious"] else 0.0,
            "avg_bbox_iou":      round(float(np.mean(
                                     match_result["bbox_ious"])), 4)
                                 if match_result["bbox_ious"] else 0.0,
        })

    return pd.DataFrame(results)


# ===========================================================================
# STEP 5: GENERATE FULL REPORT
# ===========================================================================
# Pulls everything together into one structured report and optionally
# saves it as a JSON file.

def generate_evaluation_report(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    output_path: Path = None,
    temporal_iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    labelled_clips_dir: Path = None,
    fps: float = 30.0,
) -> dict:
    """
    Generate a full evaluation report comparing predictions to ground truth.

    Computes metrics at three levels:
        1. Frame level: bbox IoU across ALL predicted frames vs ground
           truth frames, independent of temporal matching. This measures
           raw YOLO detection accuracy — when the model draws a box,
           how accurate is it spatially?
        2. Event level: precision, recall, F1, F2, and bbox IoU for
           temporally matched events (True Positives only).
        3. Sequence level: average temporal IoU for matched events.

    Also stratifies all event-level metrics by pen and weaning stage
    to support the partner's research questions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of load_predictions().
    ground_truth : pd.DataFrame
        Output of load_ground_truth().
    output_path : Path, optional
        If provided, saves the full report as a JSON file.
    temporal_iou_threshold : float
        Minimum temporal IoU to count as a True Positive. Default 0.5.
    confidence_threshold : float
        Minimum confidence score to consider a prediction. Default 0.5.
    labelled_clips_dir : Path, optional
        Root directory of CVAT annotation zip files. Required for
        frame-level and event-level bbox IoU computation. If None,
        all bbox IoU values will be 0.0.
    fps : float
        Frames per second of source videos. Default 30.0.

    Returns
    -------
    dict
        Full evaluation report with keys:
            thresholds: confidence and temporal IoU thresholds used
            frame_level: avg bbox IoU across all predicted frames
            event_level: TP/FP/FN, precision, recall, F1, F2, avg bbox IoU
            sequence_level: avg temporal IoU for matched events
            by_pen: event-level metrics broken down by pen
            by_weaning_stage: event-level metrics broken down by weaning stage
            matched_pairs: details of each True Positive match
    """
    # --- Frame level bbox IoU (independent of temporal matching) ---
    frame_level_bbox_iou = 0.0
    if labelled_clips_dir:
        frame_level_bbox_iou = compute_frame_level_bbox_iou(
            predictions=predictions,
            ground_truth=ground_truth,
            labelled_clips_dir=labelled_clips_dir,
            fps=fps,
        )

    # --- Event level matching ---
    match_result = match_predictions_to_ground_truth(
        predictions, ground_truth,
        temporal_iou_threshold, confidence_threshold,
        labelled_clips_dir=labelled_clips_dir,
        fps=fps,
    )

    tp = match_result["true_positives"]
    fp = match_result["false_positives"]
    fn = match_result["false_negatives"]

    f1_scores = compute_precision_recall_f(tp, fp, fn, beta=1.0)
    f2_scores = compute_precision_recall_f(tp, fp, fn, beta=2.0)

    avg_temporal_iou = (
        round(float(np.mean(match_result["temporal_ious"])), 4)
        if match_result["temporal_ious"] else 0.0
    )
    avg_bbox_iou = (
        round(float(np.mean(match_result["bbox_ious"])), 4)
        if match_result["bbox_ious"] else 0.0
    )

    # --- Stratified metrics ---
    by_pen = evaluate_by_stratum(
        predictions, ground_truth, "pen",
        temporal_iou_threshold, confidence_threshold,
        labelled_clips_dir=labelled_clips_dir,
        fps=fps,
    ).to_dict("records")

    by_weaning_stage = evaluate_by_stratum(
        predictions, ground_truth, "weaning_stage",
        temporal_iou_threshold, confidence_threshold,
        labelled_clips_dir=labelled_clips_dir,
        fps=fps,
    ).to_dict("records")

    # --- Build report ---
    report = {
        "thresholds": {
            "temporal_iou_threshold": temporal_iou_threshold,
            "confidence_threshold":   confidence_threshold,
        },
        "frame_level": {
            "avg_bbox_iou": frame_level_bbox_iou,
            "description":  (
                "Average bbox IoU across ALL predicted frames vs ground truth "
                "frames, independent of temporal event matching. Measures raw "
                "YOLO CS detection spatial accuracy."
            ),
        },
        "event_level": {
            "true_positives":  tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision":       f1_scores["precision"],
            "recall":          f1_scores["recall"],
            "f1":              f1_scores["f_score"],
            "f2":              f2_scores["f_score"],
            "avg_bbox_iou":    avg_bbox_iou,
            "description":     (
                "Metrics for events that passed temporal IoU threshold. "
                "avg_bbox_iou only computed for True Positives."
            ),
        },
        "sequence_level": {
            "avg_temporal_iou": avg_temporal_iou,
            "description":      (
                "Average temporal IoU for matched events. Measures how well "
                "predicted event time windows overlap with labeled windows."
            ),
        },
        "by_pen":           by_pen,
        "by_weaning_stage": by_weaning_stage,
        "matched_pairs":    match_result["matched_pairs"],
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Evaluation report saved to {output_path}")

    return report


# ===========================================================================
# MAIN — command line entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MooVision predictions against ground truth."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory containing prediction JSON files."
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        required=True,
        help="Path to processed clips index CSV (ground truth)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save evaluation report as JSON."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score for predictions. Default 0.5."
    )
    parser.add_argument(
        "--temporal_iou_threshold",
        type=float,
        default=0.5,
        help="Minimum temporal IoU to count as a match. Default 0.5."
    )
    parser.add_argument(
        "--labelled_clips_dir",
        type=Path,
        default=None,
        help=(
            "Path to root directory of CVAT annotation zip files "
            "(cross_sucking_labelled/). Required for bbox IoU computation. "
            "If not provided, all bbox IoU values will be 0.0."
        )
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second of source videos. Default 30.0."
    )
    args = parser.parse_args()

    preds = load_predictions(args.predictions)
    gt    = load_ground_truth(args.ground_truth)

    report = generate_evaluation_report(
        predictions=preds,
        ground_truth=gt,
        output_path=args.output,
        temporal_iou_threshold=args.temporal_iou_threshold,
        confidence_threshold=args.confidence_threshold,
        labelled_clips_dir=args.labelled_clips_dir,
        fps=args.fps,
    )

    print(json.dumps(report, indent=2))