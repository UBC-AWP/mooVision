"""
predict_bboxes.py
-----------------
Runs a fine-tuned YOLO model frame-by-frame on cross-sucking video clips,
saves predicted bounding boxes as .txt files in YOLO format mirroring the
CVAT ground truth annotation folder structure, and optionally evaluates
frame-level CS detection accuracy against ground truth annotations.

Output files use the same YOLO format as ground truth annotations:
    class_id  center_x  center_y  width  height
    (all values normalized between 0 and 1)

Frames with no detections are skipped — no empty .txt file is written,
matching the behavior of CVAT ground truth exports which only contain
frames where a bounding box was annotated.

Evaluation (optional):
    When --labelled_clips_dir is provided, computes frame-level precision,
    recall, F1, F2 and avg bbox IoU by comparing predicted .txt files against
    CVAT ground truth zip annotations. A TP is a frame where the model
    predicted a bbox AND ground truth has a bbox with IoU >= iou_threshold.

How to run (inference only):
    uv run python scripts/evaluation/predict_bboxes.py \
        --input_path data/processed/pipeline_demo/test.csv \
        --model_path data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt \
        --output_path results/predicted_bboxes/pipeline_demo/ \
        --skip 1

How to run (inference + evaluation):
    uv run python scripts/evaluation/predict_bboxes.py \
        --input_path data/processed/<split_name>/test.csv \
        --model_path data/yolo_training_runs/<split_name>/<run_name>/weights/best.pt \
        --output_path results/predicted_bboxes/<split_name>/<run_name>/ \
        --labelled_clips_dir /path/to/cross_sucking_labelled \
        --skip 1
"""

from pathlib import Path
import sys
import cv2
import json
import zipfile
import argparse
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
from scripts.data_reading.matching import parse_unlabelled_name
from config import UNLABELLED_CLIPS_DIR, ROOT_DIR


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def save_bbox_as_yolo_txt(
    path: Path,
    results,
    frame_shape: tuple,
    conf_threshold: float = 0.0,
) -> bool:
    """
    Save YOLO model predictions for a single frame as a .txt annotation file.

    Uses YOLO normalized format matching CVAT ground truth exports:
        class_id  center_x  center_y  width  height
    All coordinate values are normalized between 0 and 1 relative to
    frame dimensions.

    Class ID is forced to 0 (cross-sucking) regardless of the raw model
    output, since the fine-tuned model was trained on a single class and
    this guarantees exact format matching with ground truth annotation files.

    Parameters
    ----------
    path : Path
        Output path for the .txt file.
    results : ultralytics Results
        Raw output from model(frame) — contains predicted boxes.
    frame_shape : tuple
        Shape of the video frame (height, width, channels) from frame.shape.
        Used to normalize pixel coordinates.
    conf_threshold : float
        Minimum confidence score to save a detection. Default 0.0 (save all).

    Returns
    -------
    bool
        True if at least one detection was saved, False if no detections
        above threshold (caller should skip writing the file).
    """
    h, w = frame_shape[:2]
    lines = []

    for box in results[0].boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Convert pixel coords to normalized YOLO format
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # Force class ID to 0 (cross-sucking) — fine-tuned model is single class
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        return False  # No detections — caller should skip writing file

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))

    return True


def load_gt_boxes_from_zip(zip_path: Path, skip: int = 1) -> dict:
    """
    Load ground truth bounding boxes from a CVAT annotation zip file.

    Reads all frame_XXXXXX.txt files from the obj_train_data folder inside
    the zip, parses YOLO normalized format, and returns a dict mapping
    frame number to bounding box coordinates (still normalized).

    Parameters
    ----------
    zip_path : Path
        Full path to the CVAT annotation zip file.
    skip : int
        Frame skip value — only loads frames that match the skip pattern
        so ground truth frames align with predicted frames. Default 1.

    Returns
    -------
    dict
        {frame_num: (cx, cy, w, h)} in normalized YOLO format.
        Empty dict if zip not found or no annotations.
    """
    if not zip_path.exists():
        return {}

    gt_boxes = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            txt_files = [
                f for f in z.namelist()
                if f.startswith("obj_train_data/") and f.endswith(".txt")
            ]
            for txt_file in txt_files:
                try:
                    frame_num = int(txt_file.split("frame_")[1].replace(".txt", ""))
                except (IndexError, ValueError):
                    continue

                if frame_num % skip != 0:
                    continue

                with z.open(txt_file) as f:
                    content = f.read().decode().strip()
                    if not content:
                        continue
                    parts = content.split()
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gt_boxes[frame_num] = (cx, cy, bw, bh)
    except Exception as e:
        print(f"[WARN] Could not read {zip_path}: {e}")

    return gt_boxes


def compute_bbox_iou_normalized(pred: tuple, gt: tuple) -> float:
    """
    Compute IoU between two bounding boxes in normalized YOLO format.

    Both boxes are (cx, cy, w, h) with values between 0 and 1.
    Converts to (x1, y1, x2, y2) format for intersection computation.

    Parameters
    ----------
    pred : tuple (cx, cy, w, h)
        Predicted bounding box in normalized YOLO format.
    gt : tuple (cx, cy, w, h)
        Ground truth bounding box in normalized YOLO format.

    Returns
    -------
    float
        IoU score between 0.0 and 1.0.
    """
    px1, py1 = pred[0] - pred[2] / 2, pred[1] - pred[3] / 2
    px2, py2 = pred[0] + pred[2] / 2, pred[1] + pred[3] / 2
    gx1, gy1 = gt[0] - gt[2] / 2,   gt[1] - gt[3] / 2
    gx2, gy2 = gt[0] + gt[2] / 2,   gt[1] + gt[3] / 2

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_pred = pred[2] * pred[3]
    area_gt   = gt[2] * gt[3]
    union = area_pred + area_gt - intersection

    return intersection / union if union > 0 else 0.0


def load_pred_boxes_from_txt(txt_path: Path) -> list:
    """
    Load predicted bounding boxes from a saved .txt file.

    Parameters
    ----------
    txt_path : Path
        Path to the predicted .txt file in YOLO format.

    Returns
    -------
    list of tuple (cx, cy, w, h)
        All predicted boxes for this frame. Empty list if file missing.
    """
    if not txt_path.exists():
        return []
    boxes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def evaluate_frame_level(
    df: pd.DataFrame,
    output_dir: Path,
    labelled_clips_dir: Path,
    skip: int = 1,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Evaluate frame-level CS detection accuracy by comparing predicted
    bounding box .txt files against ground truth CVAT zip annotations.

    For each frame in each clip:
        - TP: model predicted a bbox AND ground truth has a bbox,
              and their IoU >= iou_threshold
        - FP: model predicted a bbox but ground truth has no bbox
              (or IoU < iou_threshold)
        - FN: ground truth has a bbox but model predicted nothing

    Aggregates TP/FP/FN across all clips and all frames to compute
    a single set of frame-level precision, recall, F1, and F2 scores.
    F2 is prioritized since missing a real CS frame is more costly
    than a false positive.

    Parameters
    ----------
    df : pd.DataFrame
        Input CSV DataFrame containing clip paths and labelled_clip_relative_path.
    output_dir : Path
        Directory containing predicted .txt files from extract_predicted_bboxes().
    labelled_clips_dir : Path
        Root directory of CVAT annotation zip files (cross_sucking_labelled/).
    skip : int
        Frame skip value — must match the skip used during prediction
        so predicted and ground truth frame numbers align. Default 1.
    iou_threshold : float
        Minimum IoU between predicted and ground truth box to count as TP.
        Default 0.5.

    Returns
    -------
    dict
        Frame-level evaluation metrics:
        {precision, recall, f1, f2, avg_bbox_iou, tp, fp, fn,
         total_frames_with_gt, total_frames_with_pred}
    """
    tp = 0
    fp = 0
    fn = 0
    bbox_ious = []

    for row in df.itertuples():
        # Get file naming info
        clip_path = str(row.clip_relative_path).replace("\\", "/")
        clip_name = Path(clip_path).name

        from scripts.data_reading.matching import parse_unlabelled_name
        numeric_id, part_id = parse_unlabelled_name(clip_name)
        part_str = f"part0{part_id}" if part_id else None
        file_prefix = f"{int(numeric_id):04}_{part_str}_frame_"

        # Load ground truth boxes from zip
        if not hasattr(row, "labelled_clip_relative_path") or pd.isna(row.labelled_clip_relative_path):
            continue

        zip_path = labelled_clips_dir / str(row.labelled_clip_relative_path).replace("\\", "/")
        gt_boxes = load_gt_boxes_from_zip(zip_path, skip=skip)

        # Get all frame numbers we processed for this clip
        pred_files = {
            int(f.stem.split("frame_")[1]): f
            for f in output_dir.glob(f"{file_prefix}*.txt")
        }

        # All frame numbers from either predictions or ground truth
        all_frames = set(gt_boxes.keys()) | set(pred_files.keys())

        for frame_num in all_frames:
            has_gt   = frame_num in gt_boxes
            has_pred = frame_num in pred_files

            if has_gt and has_pred:
                # Load predicted boxes and find best IoU against ground truth
                pred_box_list = load_pred_boxes_from_txt(pred_files[frame_num])
                if not pred_box_list:
                    fn += 1
                    continue
                best_iou = max(
                    compute_bbox_iou_normalized(pb, gt_boxes[frame_num])
                    for pb in pred_box_list
                )
                if best_iou >= iou_threshold:
                    tp += 1
                    bbox_ious.append(best_iou)
                else:
                    fp += 1
                    fn += 1
            elif has_pred and not has_gt:
                fp += 1
            elif has_gt and not has_pred:
                fn += 1

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
    avg_bbox_iou = float(np.mean(bbox_ious)) if bbox_ious else 0.0

    return {
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "f2":              round(f2, 4),
        "avg_bbox_iou":    round(avg_bbox_iou, 4),
    }

def extract_predicted_bboxes(
    input_path: str,
    model_path: str,
    output_path: str,
    labelled_clips_dir: str = None,
    skip: int = 1,
    conf_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    FORCE: bool = False,
) -> None:
    """
    Run YOLO inference frame-by-frame on video clips, save predicted
    bounding boxes as .txt files, and optionally evaluate frame-level
    detection accuracy against CVAT ground truth annotations.

    For each video clip in the input CSV:
        1. Opens the video file from UNLABELLED_CLIPS_DIR
        2. Reads every Nth frame (controlled by skip)
        3. Passes each frame to the fine-tuned YOLO model
        4. Saves predicted bounding boxes as a .txt file in YOLO format
           using the same naming convention as CVAT ground truth exports
        5. Skips frames with no detections (no empty .txt file written)

    If labelled_clips_dir is provided, runs frame-level evaluation after
    inference completes, comparing predicted .txt files against CVAT
    ground truth zip annotations to compute precision, recall, F1, F2,
    and average bbox IoU across all frames.

    Output folder structure mirrors the ground truth zip file structure:
        ROOT_DIR/output_path/
            <numeric_id>_<part_id>_frame_<XXXXXX>.txt

    Parameters
    ----------
    input_path : str
        Path to CSV file containing clip paths and labels.
        Relative to ROOT_DIR (e.g. data/processed/pipeline_demo/test.csv).
        Must contain 'clip_relative_path' and 'labelled_clip_relative_path' columns.
    model_path : str
        Path to fine-tuned YOLO model weights (.pt file).
        Relative to ROOT_DIR
        (e.g. data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt).
    output_path : str
        Output directory for predicted bounding box .txt files.
        Relative to ROOT_DIR (e.g. results/predicted_bboxes/pipeline_demo/).
    labelled_clips_dir : str, optional
        Path to root directory of CVAT annotation zip files
        (cross_sucking_labelled/). If provided, frame-level evaluation
        metrics are computed after inference. If None, skips evaluation.
    skip : int
        Process every Nth frame. skip=1 processes every frame, skip=5
        processes every 5th frame. Should match the skip value used in
        preprocessing so frame numbers align with ground truth. Default 1.
    conf_threshold : float
        Minimum YOLO confidence score to save a detection. Default 0.0
        (save all detections regardless of confidence).
    iou_threshold : float
        Minimum bbox IoU between predicted and ground truth box to count
        as a True Positive during evaluation. Default 0.5.
    FORCE : bool
        If True, overwrite existing output files. Default False.

    Returns
    -------
    None
        Saves .txt prediction files to ROOT_DIR/output_path/ and
        optionally prints and saves a frame-level evaluation report.
    """
    # Resolve paths
    root = Path(ROOT_DIR) if not isinstance(ROOT_DIR, Path) else ROOT_DIR
    df = pd.read_csv(root / input_path, index_col=0)
    output_dir = root / output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=================================")
    print(f"PREDICTING BOUNDING BOXES")
    print(f"=================================\n")
    print(f"Model:          {model_path}")
    print(f"Input CSV:      {input_path} ({len(df)} clips)")
    print(f"Output dir:     {output_dir}")
    print(f"Frame skip:     {skip}")
    print(f"Conf threshold: {conf_threshold}")
    print(f"Evaluate:       {'Yes' if labelled_clips_dir else 'No'}\n")

    # Load model once before the loop — expensive operation
    print("Loading model...")
    model = YOLO(str(root / model_path))
    print("Model loaded.\n")

    n_clips = len(df)
    total_frames_processed = 0
    total_detections_saved = 0

    for n, row in enumerate(df.itertuples(), 1):
        clip_path = UNLABELLED_CLIPS_DIR / str(row.clip_relative_path).replace("\\", "/")

        if not clip_path.exists():
            print(f"[WARN] Clip not found, skipping: {clip_path}")
            continue

        # Get numeric id and part id for output file naming
        numeric_id, part_id = parse_unlabelled_name(str(clip_path.name))
        part_str = f"part0{part_id}" if part_id else None
        file_prefix = f"{int(numeric_id):04}_{part_str}_frame_"

        # Open video
        cap = cv2.VideoCapture(str(clip_path))
        total_clip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing clip ({n}/{n_clips}): {clip_path.name} ({total_clip_frames} frames)")

        frame_pointer = 0
        clip_detections = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_pointer % skip == 0:
                # Run YOLO inference on this frame
                results = model(frame, verbose=False)

                # Save predicted bbox as .txt file if detections exist
                txt_path = output_dir / f"{file_prefix}{frame_pointer:06d}.txt"

                if not txt_path.exists() or FORCE:
                    saved = save_bbox_as_yolo_txt(
                        path=txt_path,
                        results=results,
                        frame_shape=frame.shape,
                        conf_threshold=conf_threshold,
                    )
                    if saved:
                        clip_detections += 1
                        total_detections_saved += 1

                total_frames_processed += 1

            # Fast frame skipping
            if skip > 1:
                video_ended_early = False
                for _ in range(skip - 1):
                    if not cap.grab():
                        video_ended_early = True
                        break
                if video_ended_early:
                    break
                frame_pointer += skip
            else:
                frame_pointer += 1

        cap.release()
        print(f"  → {clip_detections} frames with detections saved out of {frame_pointer} frames processed")

    print(f"\n=================================")
    print(f"INFERENCE COMPLETE")
    print(f"=================================")
    print(f"Total frames processed:      {total_frames_processed}")
    print(f"Total detection files saved: {total_detections_saved}")
    print(f"Output saved to: {output_dir}\n")

    # --- Frame-level evaluation ---
    if labelled_clips_dir:
        print(f"\n=================================")
        print(f"FRAME-LEVEL EVALUATION")
        print(f"=================================\n")

        metrics = evaluate_frame_level(
            df=df,
            output_dir=output_dir,
            labelled_clips_dir=Path(labelled_clips_dir),
            skip=skip,
            iou_threshold=iou_threshold,
        )

        print(f"IoU threshold:   {iou_threshold}")
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"Precision:       {metrics['precision']}")
        print(f"Recall:          {metrics['recall']}")
        print(f"F1:              {metrics['f1']}")
        print(f"F2:              {metrics['f2']}")
        print(f"Avg Bbox IoU:    {metrics['avg_bbox_iou']}")

        # Save evaluation report
        report_path = output_dir / "frame_level_evaluation.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nEvaluation report saved to: {report_path}")


# ===========================================================================
# ARGUMENT PARSING
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference and save predicted bboxes as CVAT-format .txt files."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input CSV file relative to ROOT_DIR (e.g. data/processed/pipeline_demo/test.csv)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to YOLO model weights relative to ROOT_DIR (e.g. data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory relative to ROOT_DIR (e.g. results/predicted_bboxes/pipeline_demo/)."
    )
    parser.add_argument(
        "--labelled_clips_dir",
        type=str,
        default=None,
        help="Path to CVAT annotation zip files (cross_sucking_labelled/). If provided, runs frame-level evaluation after inference."
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Process every Nth frame. Should match skip used in preprocessing. Default 1."
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Minimum YOLO confidence score to save a detection. Default 0.0."
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="Minimum bbox IoU to count as a True Positive during evaluation. Default 0.5."
    )
    parser.add_argument(
        "--FORCE",
        default=False,
        action="store_true",
        help="Overwrite existing output files."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_predicted_bboxes(
        input_path=args.input_path,
        model_path=args.model_path,
        output_path=args.output_path,
        labelled_clips_dir=args.labelled_clips_dir,
        skip=args.skip,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        FORCE=args.FORCE,
    )