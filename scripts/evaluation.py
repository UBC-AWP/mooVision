"""
evaluation.py
-------------
Evaluation module for the MooVision cross-sucking detection pipeline.

Compares baseline model predictions against ground truth annotations
across two levels:
  - Event level: bounding box IoU, precision, recall, F1, F2
  - Sequence level: temporal IoU

Prediction input: JSON files output by baseline.py, one per video.
Ground truth input: CSV file with one row per confirmed CS event,
                    with columns: identifier, start_sec, end_sec,
                    pen, weaning_stage, day, and optionally bounding boxes.

Usage:
    uv run python src/evaluation.py \
        --predictions results/metadata/baseline/ \
        --ground_truth data/metadata/ground_truth.csv \
        --output results/evaluation_report.json
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Event Level: Bounding Box IoU
# ---------------------------------------------------------------------------

def compute_bbox_iou(box_pred: list, box_gt: list) -> float:
    """
    Compute bounding box IoU between a predicted and ground truth box.

    Both boxes use [x1, y1, x2, y2] format (pixel coordinates), which
    matches the intersection_box format output by baseline.py.

    Parameters
    ----------
    box_pred : list
        Predicted bounding box [x1, y1, x2, y2].
    box_gt : list
        Ground truth bounding box [x1, y1, x2, y2].

    Returns
    -------
    float
        IoU score between 0 and 1. Returns 0.0 if union is zero.
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
    union = area_pred + area_gt - intersection

    return intersection / union if union > 0 else 0.0


def compute_avg_bbox_iou_for_event(
    pred_boxes: list,
    gt_boxes: list
) -> float:
    """
    Compute average bounding box IoU across all frames in a matched event.

    Since each event has per-frame intersection boxes, this averages
    the IoU over all frames where both a predicted and ground truth
    box are available.

    Parameters
    ----------
    pred_boxes : list of dict
        Per-frame predicted boxes: [{"frame": int, "x1", "y1", "x2", "y2"}, ...]
    gt_boxes : list of dict
        Per-frame ground truth boxes in same format.

    Returns
    -------
    float
        Average IoU across matched frames. Returns 0.0 if no frames match.
    """
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


# ---------------------------------------------------------------------------
# Event Level: Precision, Recall, F1, F2
# ---------------------------------------------------------------------------

def compute_precision_recall_f(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    beta: float = 1.0
) -> dict:
    """
    Compute precision, recall, and F-beta score.

    Use beta=1.0 for F1 (equal weight to precision and recall).
    Use beta=2.0 for F2 (recall weighted more heavily than precision).
    F2 is preferred for this project because missing a cross-sucking
    event is more costly than an occasional false positive.

    Parameters
    ----------
    true_positives : int
        Predicted events that correctly match a ground truth event.
    false_positives : int
        Predicted events with no matching ground truth event.
    false_negatives : int
        Ground truth events that were not detected by the model.
    beta : float
        Beta value. 1.0 = F1, 2.0 = F2.

    Returns
    -------
    dict
        Keys: precision, recall, f_score. All values between 0 and 1.
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


# ---------------------------------------------------------------------------
# Sequence Level: Temporal IoU
# ---------------------------------------------------------------------------

def compute_temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float
) -> float:
    """
    Compute temporal IoU between a predicted and ground truth event window.

    Measures how well the predicted event duration overlaps with the
    manually labeled time window in seconds.

    Parameters
    ----------
    pred_start : float
        Predicted event start time in seconds (start_sec from baseline).
    pred_end : float
        Predicted event end time in seconds (end_sec from baseline).
    gt_start : float
        Ground truth event start time in seconds.
    gt_end : float
        Ground truth event end time in seconds.

    Returns
    -------
    float
        Temporal IoU score between 0 and 1. Returns 0.0 if no overlap.
    """
    intersection_start = max(pred_start, gt_start)
    intersection_end   = min(pred_end,   gt_end)
    intersection = max(0.0, intersection_end - intersection_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(predictions_dir: Path) -> pd.DataFrame:
    """
    Load all baseline prediction JSON files from a directory into a DataFrame.

    Each JSON file corresponds to one video processed by baseline.py.
    Flattens the nested events list so each row is one predicted event.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing one JSON file per video
        (e.g. results/metadata/baseline/).

    Returns
    -------
    pd.DataFrame
        One row per predicted event with columns:
        identifier, start_sec, end_sec, duration_sec,
        avg_confidence, intersection_box, fps, pen, weaning_stage, day.
    """
    records = []
    for json_file in Path(predictions_dir).glob("*.json"):
        with open(json_file) as f:
            metadata = json.load(f)

        identifier = metadata["identifier"]
        pen, weaning_stage, day = parse_identifier(identifier)

        for event in metadata.get("events", []):
            records.append({
                "identifier":       identifier,
                "start_sec":        event["start_sec"],
                "end_sec":          event["end_sec"],
                "duration_sec":     event["duration_sec"],
                "avg_confidence":   event["avg_confidence"],
                "intersection_box": event.get("intersection_box", []),
                "fps":              metadata["fps"],
                "pen":              pen,
                "weaning_stage":    weaning_stage,
                "day":              day,
            })

    return pd.DataFrame(records)


def load_ground_truth(path: Path) -> pd.DataFrame:
    """
    Load ground truth annotations from CSV.

    Expected columns: identifier, start_sec, end_sec, pen,
    weaning_stage, day, and optionally intersection_box (per-frame boxes).

    Parameters
    ----------
    path : Path
        Path to ground truth CSV file.

    Returns
    -------
    pd.DataFrame
        One row per confirmed cross-sucking event.
    """
    return pd.read_csv(path)


def parse_identifier(identifier: str) -> tuple:
    """
    Parse pen, weaning stage and day from a video filename identifier.

    Expects filenames following the MooVision naming convention:
    CS_XXXX_<STAGE>_d<DAY>_p<PEN>_...

    Parameters
    ----------
    identifier : str
        Video filename e.g. CS_0001_POSTWEAN_d1_p2_cow6_...mp4

    Returns
    -------
    tuple
        (pen, weaning_stage, day). Returns ('unknown', 'unknown', 'unknown')
        if parsing fails.
    """
    try:
        parts = identifier.split("_")
        weaning_stage = parts[2]   # e.g. POSTWEAN
        day           = parts[3]   # e.g. d1
        pen           = parts[4]   # e.g. p2
        return pen, weaning_stage, day
    except (IndexError, ValueError):
        return "unknown", "unknown", "unknown"


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_predictions_to_ground_truth(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    temporal_iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5
) -> dict:
    """
    Match predicted events to ground truth events using temporal IoU.

    Matching is done per video (identifier). A predicted event is a
    true positive if:
      1. It shares the same identifier as a ground truth event
      2. Its temporal IoU with that ground truth event >= temporal_iou_threshold
      3. Its avg_confidence >= confidence_threshold

    Each ground truth event can only be matched once (greedy matching).

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of load_predictions().
    ground_truth : pd.DataFrame
        Output of load_ground_truth().
    temporal_iou_threshold : float
        Minimum temporal IoU to count as a match. Default 0.5.
    confidence_threshold : float
        Minimum confidence to consider a prediction. Default 0.5.

    Returns
    -------
    dict
        Keys:
          - true_positives (int)
          - false_positives (int)
          - false_negatives (int)
          - matched_pairs (list of dict)
          - temporal_ious (list of float)
          - bbox_ious (list of float)
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

    all_identifiers = set(preds["identifier"]).union(set(ground_truth["identifier"]))

    for identifier in all_identifiers:
        video_preds = preds[preds["identifier"] == identifier].to_dict("records")
        video_gt    = ground_truth[ground_truth["identifier"] == identifier].to_dict("records")

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
                if pred.get("intersection_box") and gt_event.get("intersection_box"):
                    b_iou = compute_avg_bbox_iou_for_event(
                        pred["intersection_box"],
                        gt_event["intersection_box"]
                    )
                bbox_ious.append(b_iou)

                matched_pairs.append({
                    "identifier":   identifier,
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


# ---------------------------------------------------------------------------
# Stratified Evaluation
# ---------------------------------------------------------------------------

def evaluate_by_stratum(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    stratum: str,
    temporal_iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Evaluate model performance broken down by pen or weaning stage.

    Allows the partner (UBC AWP) to ask questions like:
    "Does the model perform better in certain pens or weaning stages?"

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of load_predictions().
    ground_truth : pd.DataFrame
        Output of load_ground_truth().
    stratum : str
        Column to group by. One of: 'pen', 'weaning_stage', 'day'.
    temporal_iou_threshold : float
        Passed to match_predictions_to_ground_truth().
    confidence_threshold : float
        Passed to match_predictions_to_ground_truth().

    Returns
    -------
    pd.DataFrame
        One row per stratum value with columns:
        stratum_value, true_positives, false_positives, false_negatives,
        precision, recall, f1, f2, avg_temporal_iou, avg_bbox_iou.
    """
    results = []
    for value in ground_truth[stratum].unique():
        preds_subset = predictions[predictions[stratum] == value]
        gt_subset    = ground_truth[ground_truth[stratum] == value]

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
            "avg_temporal_iou":  round(float(np.mean(match_result["temporal_ious"])), 4)
                                 if match_result["temporal_ious"] else 0.0,
            "avg_bbox_iou":      round(float(np.mean(match_result["bbox_ious"])), 4)
                                 if match_result["bbox_ious"] else 0.0,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def generate_evaluation_report(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    output_path: Path = None,
    temporal_iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5
) -> dict:
    """
    Generate a full evaluation report comparing predictions to ground truth.

    Computes all metrics at both event and sequence level, and stratifies
    results by pen and weaning stage for the partner.

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
        Minimum confidence to consider a prediction. Default 0.5.

    Returns
    -------
    dict
        Full evaluation report with all metrics.
    """
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

    by_pen = evaluate_by_stratum(
        predictions, ground_truth, "pen",
        temporal_iou_threshold, confidence_threshold
    ).to_dict("records")

    by_weaning_stage = evaluate_by_stratum(
        predictions, ground_truth, "weaning_stage",
        temporal_iou_threshold, confidence_threshold
    ).to_dict("records")

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

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Evaluation report saved to {output_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        help="Path to ground truth CSV file."
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