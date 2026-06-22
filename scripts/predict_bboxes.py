"""
predict_bboxes.py
-----------------
Runs a fine-tuned YOLO model frame-by-frame on cross-sucking video clips
and saves predicted bounding boxes as .txt files in YOLO format, mirroring
the CVAT ground truth annotation folder structure.

This script is used to evaluate the spatial accuracy of the YOLO CS detection
model by producing predicted bounding box files that can be directly compared
against ground truth CVAT annotation files using the same naming convention.

Output files use the same YOLO format as ground truth annotations:
    class_id  center_x  center_y  width  height
    (all values normalized between 0 and 1)

Frames with no detections are skipped — no empty .txt file is written,
matching the behavior of CVAT ground truth exports which only contain
frames where a bounding box was annotated.

How to run:
    python scripts/predict_bboxes.py \
        --input_path data/processed/pipeline_demo/test.csv \
        --model_path data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt \
        --output_path results/predicted_bboxes/pipeline_demo/ \
        --skip 1

For any split or model:
    python scripts/predict_bboxes.py \
        --input_path data/processed/<split_name>/test.csv \
        --model_path data/yolo_training_runs/<split_name>/<run_name>/weights/best.pt \
        --output_path results/predicted_bboxes/<split_name>/<run_name>/ \
        --skip 1
"""

from pathlib import Path
import sys
import cv2
import argparse
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

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


# ===========================================================================
# MAIN FUNCTION
# ===========================================================================

def extract_predicted_bboxes(
    input_path: str,
    model_path: str,
    output_path: str,
    skip: int = 1,
    conf_threshold: float = 0.0,
    FORCE: bool = False,
) -> None:
    """
    Run YOLO inference frame-by-frame on video clips and save predicted
    bounding boxes as .txt files mirroring the CVAT ground truth structure.

    For each video clip in the input CSV:
        1. Opens the video file from UNLABELLED_CLIPS_DIR
        2. Reads every Nth frame (controlled by skip)
        3. Passes each frame to the fine-tuned YOLO model
        4. Saves predicted bounding boxes as a .txt file in YOLO format
           using the same naming convention as CVAT ground truth exports
        5. Skips frames with no detections (no empty .txt file written)

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
    skip : int
        Process every Nth frame. skip=1 processes every frame, skip=5
        processes every 5th frame. Higher values are faster but may miss
        short events. Should match the skip value used in preprocessing
        so frame numbers align with ground truth annotations. Default 1.
    conf_threshold : float
        Minimum YOLO confidence score to save a detection. Default 0.0
        (save all detections regardless of confidence).
    FORCE : bool
        If True, overwrite existing output files. Default False.

    Returns
    -------
    None
        Saves .txt prediction files to ROOT_DIR/output_path/.
    """
    # Resolve paths
    root = Path(ROOT_DIR) if not isinstance(ROOT_DIR, Path) else ROOT_DIR
    df = pd.read_csv(root / input_path, index_col=0)
    output_dir = root / output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=================================")
    print(f"PREDICTING BOUNDING BOXES")
    print(f"=================================\n")
    print(f"Model:      {model_path}")
    print(f"Input CSV:  {input_path} ({len(df)} clips)")
    print(f"Output dir: {output_dir}")
    print(f"Frame skip: {skip}")
    print(f"Conf threshold: {conf_threshold}\n")

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
    print(f"COMPLETE")
    print(f"=================================")
    print(f"Total frames processed:  {total_frames_processed}")
    print(f"Total detection files saved: {total_detections_saved}")
    print(f"Output saved to: {output_dir}\n")


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
        skip=args.skip,
        conf_threshold=args.conf_threshold,
        FORCE=args.FORCE,
    )