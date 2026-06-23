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
    uv run python scripts/models/seq_NMS/seq_NMS.py \
        --video "ROOT_DIR/raw_cross_sucking_datalog/videos/Pen 2 - Group 2/WEANING/Day 2/ch02_20251018035506.mp4" \
        --model weights/split_1/best.pt \
        --output_dir=results/test\
        --frame_skip 30 \
        --show_video
"""

import sys
import argparse
import json
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from config import ROOT_DIR

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONF_THRESHOLD = 0.15  # Lower than baseline since fine-tuned model
# is more specific — catches more real events is good.
DEFAULT_IOU_THRESHOLD = 0.3  # Minimum spatial IoU to link boxes across frames
DEFAULT_MIN_DURATION = 1.0  # Minimum tube length in seconds to keep as event
DEFAULT_FRAME_SKIP = 30  # Process every Nth frame
TARGET_CLASS_NAME = "cross-sucking"  # Class name from fine-tuned model
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
    union = area_a + area_b - intersection

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
    active_tubes = []  # tubes still being extended
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
            best_iou = 0.0
            best_tube = None
            best_idx = None

            # Try to find the best active tube to extend
            for i, tube in enumerate(active_tubes):
                if i in extended:
                    continue  # tube already extended this frame

                last_box = tube[-1]
                iou = compute_iou(
                    [det["x1"], det["y1"], det["x2"], det["y2"]],
                    [last_box["x1"], last_box["y1"], last_box["x2"], last_box["y2"]],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_tube = tube
                    best_idx = i

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


def suppress_weak_detections(tubes: list, conf_threshold: float) -> list:
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
        end_frame = tube[-1]["frame"]
        duration = (end_frame - start_frame) / fps

        # Discard short tubes
        if duration < min_duration:
            continue

        avg_confidence = float(np.mean([det["confidence"] for det in tube]))

        # Build per-frame intersection box list — matches baseline.py format
        intersection_box = [
            {
                "frame": det["frame"],
                "x1": int(det["x1"]),
                "y1": int(det["y1"]),
                "x2": int(det["x2"]),
                "y2": int(det["y2"]),
            }
            for det in tube
        ]

        events.append(
            {
                "start_sec": round(start_frame / fps, 2),
                "end_sec": round(end_frame / fps, 2),
                "duration_sec": round(duration, 2),
                "avg_confidence": round(avg_confidence, 3),
                "intersection_box": intersection_box,
            }
        )

    return events


# ---------------------------------------------------------------------------
# STEP 5: FULL DETECTION PIPELINE
# ---------------------------------------------------------------------------


def run_seq_nms_detection(
    video_path: str,
    model_path: str,
    output_dir: str,
    conf_threshold: float,
    iou_threshold: float,
    min_duration: float,
    frame_skip: int,
    target_class: str,
    show_video: bool = False,
) -> dict:
    """
    Full YOLOv26 + Seq-NMS detection pipeline.

    Steps:
        1. Load fine-tuned YOLOv26 model
        2. Open video and process every Nth frame
        3. Run YOLO inference on each frame
        4. Collect all per-frame detections
        5. Apply Seq-NMS to link detections into tubes
        6. Suppress weak detections within tubes
        7. Convert tubes to event windows
        8. Save results as JSON

    Parameters
    ----------
    video_path : str
        Path to input video file.
    model_path : str
        Path to fine-tuned YOLO weights file (e.g. best.pt).
    conf_threshold : float
        Minimum YOLO detection confidence to keep a box.
    iou_threshold : float
        Minimum spatial IoU to link boxes across frames into tubes.
    min_duration : float
        Minimum event duration in seconds.
    frame_skip : int
        Process every Nth frame. 1 = every frame.

    Returns
    -------
    dict
        Full metadata dict, also saved as JSON to
        results/metadata/seq_nms/<video_name>_results.json
    """
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    # Get target class ID from model
    class_name_to_id = {v: k for k, v in model.names.items()}
    if target_class in class_name_to_id:
        target_ids = {class_name_to_id[target_class]}
    else:
        # Fall back to cow class if cross-sucking not found
        # (for testing with pretrained weights)
        print(f"[WARN] '{target_class}' not found in model classes.")
        print(f"[WARN] Available classes: {list(model.names.values())}")
        print(f"[WARN] Falling back to 'cross-sucking' class for testing.")
        target_ids = (
            {class_name_to_id["cross-sucking"]}
            if "cross-sucking" in class_name_to_id
            else set()
        )

    if not target_ids:
        raise ValueError(
            f"Neither '{target_class}' nor 'cow' found in model classes: "
            f"{list(model.names.values())}"
        )

    print(f"[INFO] Detecting class IDs: {target_ids}")
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps | {total_frames} frames")

    # Collect per-frame detections
    # Each entry is a list of detections for that frame
    frame_detections = []
    frame_idx = 0

    print("[INFO] Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fast frame skipping without running the above
        if frame_skip > 1:
            for _ in range(frame_skip - 1):
                if not cap.grab():
                    break
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"  ...frame {frame_idx}/{total_frames}")
        else:
            frame_idx += 1

        # Run YOLO inference
        results = model(frame, imgsz=640, conf=conf_threshold, verbose=False)[0]

        # Show annotated frame
        if show_video:
            annotated = results.plot()
            cv2.imshow("Seq-NMS Detection", annotated)
            cv2.waitKey(1)

        # Collect detections for this frame
        dets = []
        for box in results.boxes:
            if int(box.cls[0].item()) in target_ids:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(
                    {
                        "frame": frame_idx,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": float(box.conf[0].item()),
                    }
                )

        frame_detections.append(dets)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  ...frame {frame_idx}/{total_frames}")

    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    print(f"[INFO] Processed {frame_idx} frames, collected detections.")

    print("Tracking Cross Sucking EVents Across Frames")
    # Apply Seq-NMS
    print("[INFO] Building tubes with Seq-NMS...")
    tubes = build_tubes(frame_detections, iou_threshold)
    print(f"[INFO] Built {len(tubes)} tubes before suppression.")

    # Suppress weak detections
    tubes = suppress_weak_detections(tubes, conf_threshold)
    print(f"[INFO] {len(tubes)} tubes after suppression.")

    # Convert tubes to events
    events = tubes_to_events(tubes, fps, min_duration)
    print(f"[INFO] {len(events)} events after filtering by min duration.")

    # Build metadata — same format as baseline.py
    print("Building Metadata...")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # output_dir = ROOT_DIR / "results/metadata/seq_nms"
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "identifier": os.path.basename(video_path),
        "video_path": os.path.abspath(video_path),
        "model": model_path,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "min_duration_sec": min_duration,
        "frame_skip": frame_skip,
        "fps": fps,
        "total_frames": total_frames,
        "total_duration_sec": round(total_frames / fps, 2),
        "cross_sucking_detected": len(events) > 0,
        "num_events": len(events),
        "events": events,
    }

    # Save JSON
    print()
    print("Saving Metadata...")
    json_path = os.path.join(output_dir, f"{video_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print()
    print("\n" + "═" * 50)
    print(f"  VIDEO:    {metadata['identifier']}")
    print(f"  FLAGGED:  {metadata['cross_sucking_detected']}")
    print(f"  EVENTS:   {metadata['num_events']}")
    for i, ev in enumerate(events):
        print(
            f"    Event {i+1}: {ev['start_sec']}s → {ev['end_sec']}s "
            f"({ev['duration_sec']}s) | conf={ev['avg_confidence']}"
        )
    print(f"  OUTPUT:   {json_path}")
    print("═" * 50 + "\n")

    return metadata


# ---------------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv26 + Seq-NMS cross-sucking detector."
    )
    parser.add_argument("--video_path", required=True, help="Path to input video file.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to fine-tuned YOLO weights file (e.g. runs/detect/cross-sucking/weights/best.pt).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory to save metadata to.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Minimum YOLO detection confidence. Default: {DEFAULT_CONF_THRESHOLD}",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"Minimum IoU to link detections across frames into tubes. Default: {DEFAULT_IOU_THRESHOLD}",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help=f"Minimum event duration in seconds. Default: {DEFAULT_MIN_DURATION}",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=DEFAULT_FRAME_SKIP,
        help=f"Process every Nth frame. Default: {DEFAULT_FRAME_SKIP}",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default=TARGET_CLASS_NAME,
        help=f"Process every Nth frame. Default: {DEFAULT_FRAME_SKIP}",
    )
    parser.add_argument(
        "--show_video",
        default=False,
        action="store_true",
        help="Display video during annotation, useful for running on sample videos (default: False)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_seq_nms_detection(
        video_path=args.video_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        min_duration=args.min_duration,
        frame_skip=args.frame_skip,
        target_class=args.target_class,
        show_video=args.show_video,
    )
