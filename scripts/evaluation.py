"""
evaluation.py
-------------
Evaluation module for the MooVision cross-sucking detection pipeline.

Compares baseline model predictions (JSON) against ground truth annotations
(processed clips index CSV) across two levels:
  - Event level:    bounding box IoU, precision, recall, F1, F2
  - Sequence level: temporal IoU

How to run:
    python src/evaluation.py \
        --predictions results/metadata/baseline/ \
        --ground_truth data/raw/all_clips_index_raw.csv \
        --output results/evaluation_report.json
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ===========================================================================
# STEP 1: LOADING DATA
# ===========================================================================
# These two functions load the predictions and ground truth into DataFrames
# so the rest of the script can work with them in a consistent format.

def load_predictions(predictions_dir: Path) -> pd.DataFrame:
    """
    Load all baseline prediction JSON files from a directory.

    The baseline script (baseline.py) saves one JSON file per video.
    Each JSON contains a list of detected events with start/end times,
    confidence scores, and per-frame bounding boxes.

    This function reads all those JSON files and flattens them into a
    single DataFrame where each row is one predicted event.

    Parameters
    ----------
    predictions_dir : Path
        Folder containing baseline JSON files
        (e.g. results/metadata/baseline/).

    Returns
    -------
    pd.DataFrame
        One row per predicted event. Columns:
        source_video_basename, start_sec, end_sec, duration_sec,
        avg_confidence, intersection_box, fps.
    """
    records = []

    for json_file in Path(predictions_dir).glob("*.json"):
        with open(json_file) as f:
            metadata = json.load(f)

        # identifier is the video filename e.g. ch02_20251102075200.mp4
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

    This CSV was produced by the read_data_from_index script and contains
    one row per confirmed cross-sucking clip, with timing, pen, phase,
    and day information.

    The key columns we use for evaluation are:
        - source_video_basename: links ground truth to baseline predictions
        - clip_start_in_source_sec: when the CS event starts in the raw video
        - clip_end_in_source_sec:   when the CS event ends in the raw video
        - pen:   which pen the calf was in
        - phase: preweaning / weaning / postweaning
        - day:   which day of the observation period

    Parameters
    ----------
    path : Path
        Path to the processed clips index CSV.

    Returns
    -------
    pd.DataFrame
        One row per confirmed cross-sucking event.
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


# ===========================================================================
# STEP 2: CORE METRICS
# ===========================================================================
# These three functions compute the actual evaluation metrics.
# They are simple math functions that take numbers and return numbers.

def compute_temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float
) -> float:
    """
    Compute temporal IoU between a predicted and ground truth event window.

    Asks: how well does the predicted time window overlap with the
    labeled time window?

    Works exactly like bounding box IoU but in 1D (time instead of space):
        - Find the overlapping time segment (intersection)
        - Find the total time covered by both windows (union)
        - IoU = intersection / union

    Example:
        Ground truth: 5.0s -> 10.0s  (5 seconds long)
        Prediction:   7.0s -> 13.0s  (6 seconds long)
        Intersection: 7.0s -> 10.0s  (3 seconds)
        Union:        5.0s -> 13.0s  (8 seconds)
        Temporal IoU = 3 / 8 = 0.375

    Parameters
    ----------
    pred_start, pred_end : float
        Predicted event start and end time in seconds.
    gt_start, gt_end : float
        Ground truth event start and end time in seconds.

    Returns
    -------
    float
        Temporal IoU score between 0 and 1.
    """
    intersection_start = max(pred_start, gt_start)
    intersection_end   = min(pred_end,   gt_end)
    intersection = max(0.0, intersection_end - intersection_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    return intersection / union if union > 0 else 0.0


def compute_bbox_iou(box_pred: list, box_gt: list) -> float:
    """
    Compute bounding box IoU between a predicted and ground truth box.

    Asks: how well does the predicted bounding box overlap with the
    labeled bounding box spatially?

    Both boxes are in [x1, y1, x2, y2] format (pixel coordinates),
    which matches the intersection_box format from baseline.py.

    Parameters
    ----------
    box_pred : list [x1, y1, x2, y2]
        Predicted bounding box.
    box_gt : list [x1, y1, x2, y2]
        Ground truth bounding box.

    Returns
    -------
    float
        IoU score between 0 and 1.
    """
    # Find the overlapping rectangle
    inter_x1 = max(box_pred[0], box_gt[0])
    inter_y1 = max(box_pred[1], box_gt[1])
    inter_x2 = min(box_pred[2], box_gt[2])
    inter_y2 = min(box_pred[3], box_gt[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt   = (box_gt[2]   - box_gt[0])   * (box_gt[3]   - box_gt[1])
    union = area_pred + area_gt - intersection

    return intersection / union if union > 0 else 0.0


def compute_avg_bbox_iou_for_event(
    pred_boxes: list,
    gt_boxes: list
) -> float:
    """
    Compute average bounding box IoU across all frames in a matched event.

    Since each event stores per-frame bounding boxes, this averages
    the IoU over all frames where both a predicted and ground truth
    box are available.

    Parameters
    ----------
    pred_boxes : list of dict
        Per-frame predicted boxes from baseline.py:
        [{"frame": int, "x1": int, "y1": int, "x2": int, "y2": int}, ...]
    gt_boxes : list of dict
        Per-frame ground truth boxes in same format.

    Returns
    -------
    float
        Average IoU across matched frames. 0.0 if no frames match.
    """
    # Build a lookup of ground truth boxes by frame number
    gt_by_frame = {b["frame"]: b for b in gt_boxes}

    ious = []
    for pb in pred_boxes:
        frame = pb["frame"]
        if frame in gt_by_frame:
            gb = gt_by_frame[frame]
            iou = compute_bbox_iou(
                [pb["x1"], pb["y1"], pb["x2"], pb["y2"]],
                [gb["x1"], gb["y1"], gb["x2"], gb["y2"]]
            )
            ious.append(iou)

    return float(np.mean(ious)) if ious else 0.0


def compute_precision_recall_f(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    beta: float = 1.0
) -> dict:
    """
    Compute precision, recall, and F-beta score.

    - Precision: of all events the model flagged, how many were real?
    - Recall:    of all real events, how many did the model find?
    - F1:        balanced average of precision and recall
    - F2:        like F1 but recall counts twice as much as precision
                 We use F2 because missing a real CS event is worse
                 than occasionally flagging a false one.

    Parameters
    ----------
    true_positives : int
        Predicted events that correctly match a ground truth event.
    false_positives : int
        Predicted events with no matching ground truth event.
    false_negatives : int
        Ground truth events the model missed.
    beta : float
        1.0 = F1, 2.0 = F2.

    Returns
    -------
    dict
        Keys: precision, recall, f_score.
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
# This is the core logic that decides which predictions are correct.
# It loops through each video, tries to pair each predicted event with
# a ground truth event using temporal IoU, and counts TP/FP/FN.

def match_predictions_to_ground_truth(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    temporal_iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5
) -> dict:
    """
    Match predicted events to ground truth events using temporal IoU.

    Logic per video:
        For each predicted event (above confidence threshold):
            - Find the ground truth event with the highest temporal IoU
            - If that IoU >= temporal_iou_threshold → True Positive
            - Otherwise → False Positive
        Any ground truth events with no matching prediction → False Negative

    Matching is done per source video so predictions and ground truth
    are only compared within the same video.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of load_predictions().
    ground_truth : pd.DataFrame
        Output of load_ground_truth().
    temporal_iou_threshold : float
        Minimum temporal IoU to count as a match. Default 0.5.
    confidence_threshold : float
        Minimum confidence score to consider a prediction. Default 0.5.

    Returns
    -------
    dict
        true_positives, false_positives, false_negatives,
        matched_pairs, temporal_ious, bbox_ious.
    """
    # Filter out low confidence predictions
    preds = predictions[
        predictions["avg_confidence"] >= confidence_threshold
    ].copy()

    true_positives  = 0
    false_positives = 0
    false_negatives = 0
    matched_pairs   = []
    temporal_ious   = []
    bbox_ious       = []

    # Get all unique videos across both predictions and ground truth
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

        matched_gt = set()  # track which GT events have been matched

        for pred in video_preds:
            best_iou    = 0.0
            best_gt_idx = None

            # Find the best matching ground truth event
            for gt_idx, gt in enumerate(video_gt):
                if gt_idx in matched_gt:
                    continue  # already matched, skip
                t_iou = compute_temporal_iou(
                    pred["start_sec"], pred["end_sec"],
                    gt["start_sec"],   gt["end_sec"]
                )
                if t_iou > best_iou:
                    best_iou    = t_iou
                    best_gt_idx = gt_idx

            if best_iou >= temporal_iou_threshold and best_gt_idx is not None:
                # Good match — True Positive
                true_positives += 1
                matched_gt.add(best_gt_idx)
                temporal_ious.append(best_iou)

                # Compute bbox IoU if boxes are available
                gt_event = video_gt[best_gt_idx]
                b_iou = 0.0
                if pred.get("intersection_box") and gt_event.get("intersection_box"):
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
                # No good match — False Positive
                false_positives += 1

        # Unmatched ground truth events — False Negatives
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
    confidence_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Evaluate model performance broken down by a grouping variable.

    Runs the full matching and metric computation separately for each
    unique value of the stratum column (e.g. each pen, or each phase).

    Parameters
    ----------
    stratum : str
        Column to group by. One of: 'pen', 'weaning_stage', 'day'.

    Returns
    -------
    pd.DataFrame
        One row per stratum value with all metrics.
    """
    results = []

    for value in ground_truth[stratum].unique():
        # Filter both dataframes to just this stratum value
        preds_subset = predictions[predictions[stratum] == value] \
            if stratum in predictions.columns else predictions
        gt_subset = ground_truth[ground_truth[stratum] == value]

        match_result = match_predictions_to_ground_truth(
            preds_subset, gt_subset,
            temporal_iou_threshold, confidence_threshold
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
    confidence_threshold: float = 0.5
) -> dict:
    """
    Generate a full evaluation report.

    Runs matching, computes all metrics, stratifies by pen and weaning
    stage, and returns everything as a structured dictionary.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of load_predictions().
    ground_truth : pd.DataFrame
        Output of load_ground_truth().
    output_path : Path, optional
        If provided, saves the report as a JSON file.
    temporal_iou_threshold : float
        Minimum temporal IoU to count as a match. Default 0.5.
    confidence_threshold : float
        Minimum confidence score to consider a prediction. Default 0.5.

    Returns
    -------
    dict
        Full report with all metrics, stratified results, and matched pairs.
    """
    # --- Overall metrics ---
    match_result = match_predictions_to_ground_truth(
        predictions, ground_truth,
        temporal_iou_threshold, confidence_threshold
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
        temporal_iou_threshold, confidence_threshold
    ).to_dict("records")

    by_weaning_stage = evaluate_by_stratum(
        predictions, ground_truth, "weaning_stage",
        temporal_iou_threshold, confidence_threshold
    ).to_dict("records")

    # --- Build report ---
    report = {
        "thresholds": {
            "temporal_iou_threshold": temporal_iou_threshold,
            "confidence_threshold":   confidence_threshold,
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
        },
        "sequence_level": {
            "avg_temporal_iou": avg_temporal_iou,
        },
        "by_pen":           by_pen,
        "by_weaning_stage": by_weaning_stage,
        "matched_pairs":    match_result["matched_pairs"],
    }

    # --- Optionally save to disk ---
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
        description="Evaluate MooVision baseline predictions against ground truth."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory containing baseline JSON prediction files."
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
    args = parser.parse_args()

    preds = load_predictions(args.predictions)
    gt    = load_ground_truth(args.ground_truth)

    report = generate_evaluation_report(
        predictions=preds,
        ground_truth=gt,
        output_path=args.output,
        temporal_iou_threshold=args.temporal_iou_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    print(json.dumps(report, indent=2))