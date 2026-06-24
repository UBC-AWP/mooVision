"""
scripts.run_models.baseline.baseline
=====================================

Baseline cross-sucking detector using YOLO bounding box overlap.

Overview
--------
This module provides a baseline detection pipeline for identifying
cross-sucking behaviour in dairy calf videos. For each input video,
the script computes pairwise IoU between all detected calf bounding
boxes per frame, groups consecutive overlapping frames into events,
and writes a JSON metadata file containing event timestamps, confidence
scores, and per-frame intersection box coordinates.

Input/Output
------------
Inputs
  - A test split CSV (e.g. ``data/processed/day_based/test.csv``)
    containing a ``source_video_path`` column, passed via ``--csv``.

Outputs
  - One JSON metadata file per video written under
    ``results/metadata/<split_name>/baseline/``::

        results/metadata/
        └── day_based/
            └── baseline/
                └── ch02_20250913094601_results.json

Notes
-----
  - Uses the COCO-pretrained YOLO26 model by default. The ``cow`` class
    is used as a proxy for calves. Detection accuracy improves after
    fine-tuning on labelled calf footage.
  - Videos whose JSON output already exists are skipped automatically,
    making the script safe to re-run after interruptions.
  - Configuration is centralised in ``config.py``, driven by environment
    variables in a ``.env`` file at the repo root.
"""
import argparse
import json
import os
import cv2
import re
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent)) 
from config import ROOT_DIR,SOURCE_VIDEOS_DIR,METADATA_DIR_CLOUD
from scripts.run_models.run_testing import get_split_label

# Defaults
DEFAULT_MODEL = "yolo26x.pt"    
DEFAULT_IOU_THRESHOLD = 0.1    # Minimum IoU to consider two boxes "overlapping"
DEFAULT_MIN_DURATION = 0.5       # Minimum seconds of continuous overlap to flag an event
DEFAULT_CONF_THRESHOLD = 0.5      # Minimum YOLO detection confidence to keep a box
TARGET_CLASS_NAME = "cow"
DEFAULT_FRAME_SKIP = 1

def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) between two bounding boxes
    and return the pixel coordinates of their intersection rectangle.

    IoU measures how much two boxes overlap:
        - IoU = 0.0  → boxes don't touch at all
        - IoU = 1.0  → boxes are perfectly identical

    The intersection rectangle is the overlapping region, stored in
    the metadata so the clipping script knows where to draw the annotation.

    Args:
        box_a (list): [x1, y1, x2, y2] (left edge, top edge, right edge, bottom edge of box A)
        box_b (list): [x1, y1, x2, y2] (left edge, top edge, right edge, bottom edge of box B)

    Returns:
        tuple:
            iou (float): IoU score in [0, 1]
            intersection_box (list | None): [x1, y1, x2, y2] pixel coordinates of
                                            the overlapping region, or None if no overlap
    """
    # Find the coordinates of the intersection rectangle
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    # If there's no intersection, width or height will be ≤ 0
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Area of each box individually
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Union = sum of areas minus the part we counted twice
    union = area_a + area_b - intersection

    # Compute iou
    iou = intersection / union if union > 0 else 0.0

    intersection_box = [round(inter_x1), round(inter_y1), 
                        round(inter_x2), round(inter_y2)] if intersection > 0 else None

    return iou, intersection_box


def frame_has_overlap(boxes, iou_threshold):
    """
    Check whether any two bounding boxes in a single frame overlap
    above the IoU threshold.

    Checks every possible pair of detected calf boxes. Keeps track of
    the pair with the highest IoU and returns that pair's intersection
    box to be stored in the metadata.

    Args:
        boxes (list): List of [x1, y1, x2, y2] bounding boxes for all
                      detected calves in this frame
        iou_threshold (float): Minimum IoU score to consider an overlap

    Returns:
        tuple:
            overlap_detected (bool): True if any pair exceeds the threshold
            best_inter_box (list | None): [x1, y1, x2, y2] of the intersection
                                          region of the highest-IoU pair,
                                          or None if no overlap was found
    """
    max_iou = 0.0
    best_inter_box = None
    n = len(boxes)

    for i in range(n):
        for j in range(i + 1, n):
            iou, inter_box = compute_iou(boxes[i], boxes[j])
            if iou > max_iou:
                max_iou = iou
                best_inter_box = inter_box

    return (max_iou >= iou_threshold), best_inter_box

# Event extraction
def extract_events(
    frame_flags:   list[bool],
    fps:           float,
    min_duration:  float,
    frame_skip:    int,
    confidences:   list[float],
    frame_boxes:   list,        # list[list | None] — None for non-flagged frames
    frame_indices: list[int],
) -> list[dict]:
    """
    Group consecutive flagged frames into discrete cross-sucking events and
    collect per-frame intersection box coordinates for each event.
 
    A frame is "flagged" when its IoU exceeds the overlap threshold. Consecutive
    flagged frames are merged into a single event. Events shorter than
    `min_duration` seconds are discarded as noise.
 
    Args:
        frame_flags:
            Per-frame overlap flag. True when bounding-box overlap was detected.
        fps:
            Frames per second of the source video, used to convert frame
            numbers to timestamps in seconds.
        min_duration:
            Minimum event length in seconds. Events below this are dropped.
        frame_skip:
            The frame-skip factor used during inference, needed to correctly
            compute the minimum frame count threshold.
        confidences:
            Per-frame maximum YOLO detection confidence (0.0 when no detections).
        frame_boxes:
            Per-frame intersection box [x1, y1, x2, y2], or None for frames
            where no overlap was detected. Must be the same length as
            `frame_flags`.
        frame_indices:
            Actual frame indices in the source video (accounts for frame skip).
 
    Returns:
        List of event dicts, each containing:
            start_sec (float):          Event start time in seconds.
            end_sec (float):            Event end time in seconds.
            duration_sec (float):       Total duration of the event.
            avg_confidence (float):     Mean YOLO confidence across event frames.
            intersection_box (list):    Per-frame dicts with keys
                                        'frame', 'x1', 'y1', 'x2', 'y2'.
    """
    events    = []
    in_event  = False
    start_frame   = 0
    event_confs   = []
    event_interbox = []
    min_frames = max(1, round(min_duration * fps / frame_skip))
 
    for idx, flagged in enumerate(frame_flags):
        actual_frame = frame_indices[idx]
 
        if flagged and not in_event:
            in_event    = True
            start_frame = actual_frame
            event_confs = [confidences[idx]]
            box = frame_boxes[idx]
            event_interbox = [{"frame": actual_frame,
                                "x1": box[0], "y1": box[1],
                                "x2": box[2], "y2": box[3]}]
 
        elif flagged and in_event:
            event_confs.append(confidences[idx])
            box = frame_boxes[idx]
            if box is not None:
                event_interbox.append({"frame": actual_frame,
                                        "x1": box[0], "y1": box[1],
                                        "x2": box[2], "y2": box[3]})
 
        elif not flagged and in_event:
            end_frame = actual_frame
            if len(event_confs) >= min_frames:
                events.append({
                    "start_sec":        round(start_frame / fps, 2),
                    "end_sec":          round(end_frame / fps, 2),
                    "duration_sec":     round((end_frame - start_frame) / fps, 2),
                    "avg_confidence":   round(float(np.mean(event_confs)), 3),
                    "intersection_box": event_interbox,
                })
            in_event       = False
            event_confs    = []
            event_interbox = []
 
    # Flush event that runs to the very last frame
    if in_event and len(event_confs) >= min_frames:
        end_frame = frame_indices[-1]
        events.append({
            "start_sec":        round(start_frame / fps, 2),
            "end_sec":          round(end_frame / fps, 2),
            "duration_sec":     round((end_frame - start_frame) / fps, 2),
            "avg_confidence":   round(float(np.mean(event_confs)), 3),
            "intersection_box": event_interbox,
        })
 
    return events

def extract_video_path(video_path):
    """
    Extract the video name from the full path for use in output naming.

    Args:
        video_path (str): Full path to the input video file
    Returns:
        list: [video_path (str), video_path (str)]
    """
    # enforece Path object for consistent handling
    video_path = Path(video_path)
    # 
    return [video_path.joinpath(f.name) for f in video_path.glob("*.mp4")]


 # Data loading      
def load_split_from_csv(csv_path: Path) -> list[Path]:
    """
    Load video paths from a single split CSV.

    Args:
        csv_path: Path to a test.csv file (e.g. data/processed/day_based/test.csv).

    Returns:
        List of resolved absolute video paths, deduplicated.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df = df.drop_duplicates(subset=["source_video_path"])
    if df.empty:
        raise ValueError(f"No rows in {csv_path} after deduplication.")

    clean_paths = []
    for raw in df["source_video_path"]:
        cln = Path(raw.replace("\\", "/"))
        rel = Path(*cln.parts[-4:])
        clean_paths.append(SOURCE_VIDEOS_DIR / rel)
    return clean_paths

# Detection pipeline
def detect_video(
    video_path:    Path,
    model,
    target_ids:    set,
    iou_threshold: float,
    conf_threshold:float,
    min_duration:  float,
    frame_skip:    int,
) -> dict:
    """
    Run the full detection pipeline on a single video file.
 
    Pipeline:
        open video → detect calves per frame (YOLO) →
        check pairwise IoU overlap → collect intersection boxes →
        group consecutive flagged frames into events →
        build and return metadata dict
 
    Args:
        video_path:
            Absolute path to the input video file.
        model:
            Loaded YOLO model instance (ultralytics.YOLO).
        target_ids:
            Set of YOLO class IDs corresponding to the target animal class.
        iou_threshold:
            Minimum IoU to flag a frame as containing an overlap event.
        conf_threshold:
            Minimum YOLO detection confidence to keep a bounding box.
        min_duration:
            Minimum event duration in seconds; shorter events are discarded.
        frame_skip:
            Process every Nth frame. Frame indices in output reflect the
            actual source video frame numbers.
 
    Returns:
        Metadata dict written to JSON, containing:
            identifier (str):               Video filename.
            video_path (str):               Absolute path to the source video.
            model (str):                    Model weights filename.
            iou_threshold (float):          IoU threshold used.
            conf_threshold (float):         Confidence threshold used.
            min_duration_sec (float):       Minimum event duration used.
            fps (float):                    Source video frame rate.
            total_frames (int):             Total frame count of source video.
            total_duration_sec (float):     Total video duration in seconds.
            frame_skip (int):               Frame-skip factor used.
            cross_sucking_detected (bool):  True if at least one event found.
            num_events (int):               Number of events detected.
            events (list[dict]):            Per-event metadata (see extract_events).
 
    Raises:
        FileNotFoundError: If the video file cannot be opened by OpenCV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
 
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {video_path.name} | {width}x{height} @ {fps:.1f} fps | {total_frames} frames")
 
    frame_flags   = []
    frame_confs   = []
    frame_interbox = []
    frame_indices  = []
    frame_idx      = 0
    processed      = 0
    skipped        = 0
 
    print(f"[INFO] Processing: {video_path.name}")
    print(f"       frames={total_frames} | fps={fps:.1f} | skip=1/{frame_skip} "
          f"(~{total_frames // frame_skip} frames to process)")
    print("[INFO] Running detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            skipped += 1
            continue
 
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Show the frame with bounding boxes drawn
        # annotated_frame = results.plot()
        # cv2.imshow("Calf Detection", annotated_frame)
        # cv2.waitKey(1)  # 1ms delay, keeps the window responsive
        
        boxes = []
        confs = []
        for box in results.boxes:
            if int(box.cls[0].item()) in target_ids:
                boxes.append(box.xyxy[0].tolist())
                confs.append(float(box.conf[0].item()))
 
        overlap_detected, inter_box = frame_has_overlap(boxes, iou_threshold)
 
        frame_flags.append(overlap_detected)
        frame_confs.append(max(confs) if confs else 0.0)
        frame_indices.append(frame_idx)
        frame_interbox.append(inter_box if overlap_detected else None)
        frame_idx += 1
        processed += 1
 
        if processed % 100 == 0:
            pct = 100 * frame_idx / total_frames if total_frames else 0
            print(f"  ...processed {processed} frames | "
                  f"skipped {skipped} | "
                  f"video pos {frame_idx}/{total_frames} ({pct:.1f}%) | "
                  f"events so far: {sum(frame_flags)}")
 
    cap.release()
    print(f"[INFO] Done — processed {processed} frames, skipped {skipped} | "
          f"flagged frames: {sum(frame_flags)}")
 
    events = extract_events(
        frame_flags, fps, min_duration, frame_skip,
        frame_confs, frame_interbox, frame_indices,
    )
 
    metadata = {
        "identifier":              video_path.name,
        "video_path":             str(video_path.resolve()),
        "model":                  str(DEFAULT_MODEL),
        "iou_threshold":          iou_threshold,
        "conf_threshold":         conf_threshold,
        "min_duration_sec":       min_duration,
        "fps":                    fps,
        "total_frames":           total_frames,
        "total_duration_sec":     round(total_frames / fps, 2),
        "frame_skip":             frame_skip,
        "cross_sucking_detected": len(events) > 0,
        "num_events":             len(events),
        "events":                 events,
    }
    return metadata

def run_all(
    csv_path:       Path,
    model_path:     str,
    iou_threshold:  float,
    conf_threshold: float,
    min_duration:   float,
    frame_skip:     int,
) -> None:
    """
    End-to-end pipeline: load all splits → run detection → save JSON metadata.
 
    Iterates over every split under `data_root`, runs `detect_video` on each
    unique video, and writes results to the baseline metadata directory.
    Videos whose output JSON already exists are skipped automatically.
 
    Args:
        data_root:      Root directory containing split subdirectories.
        model_path:     YOLO weights filename (e.g. 'yolo26x.pt').
                        Downloaded automatically on first run if not found locally.
        iou_threshold:  IoU threshold to flag a frame as containing an overlap.
        conf_threshold: Minimum YOLO detection confidence to keep a bounding box.
        min_duration:   Minimum event duration in seconds.
        frame_skip:     Process every Nth frame (1 = every frame).
 
    Raises:
        ValueError: If the target class is not present in the loaded model.
    """
    split_name = get_split_label(csv_path)
    video_paths = load_split_from_csv(csv_path)
    if not video_paths:
        print("[ERROR] No videos loaded.")
        return

    from ultralytics import YOLO
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    class_name_to_id = {v: k for k, v in model.names.items()}
    target_ids = {class_name_to_id[TARGET_CLASS_NAME]} if TARGET_CLASS_NAME in class_name_to_id else set()
    if not target_ids:
        raise ValueError(f"'{TARGET_CLASS_NAME}' not found in model: {list(model.names.values())}")

    for video_path in video_paths:
        # m = re.search(r"videos[\\/](.*)$", str(video_path))
        # rel_parent = Path(m.group(1)).parent if m else Path()
        
        json_path = METADATA_DIR_CLOUD/ split_name / "baseline" / f"{video_path.stem}_results.json"

        if json_path.exists():
            print(f"[SKIP] {json_path.name}")
            continue
        if not video_path.exists():
            print(f"[WARN] Video not found, skipping: {video_path}")
            continue

        try:
            metadata = detect_video(
                video_path, model, target_ids,
                iou_threshold, conf_threshold, min_duration, frame_skip,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {video_path.name}: {e} — skipping.")
            continue

        os.makedirs(json_path.parent, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "═" * 50)
        print(f"  VIDEO:   {metadata['identifier']}")
        print(f"  FLAGGED: {metadata['cross_sucking_detected']}")
        print(f"  EVENTS:  {metadata['num_events']}")
        for i, ev in enumerate(metadata["events"]):
            print(f"    Event {i+1}: {ev['start_sec']}s → {ev['end_sec']}s "
                  f"({ev['duration_sec']}s) | conf={ev['avg_confidence']}")
        print(f"  OUTPUT:  {json_path}")
        print("═" * 50 + "\n")

# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Baseline cross-sucking detector using {DEFAULT_MODEL} bounding box overlap."
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="Path to a split test.csv (e.g. data/processed/day_based/test.csv)")
    parser.add_argument("--model", 
                        default=DEFAULT_MODEL, 
                        help=f"YOLO weights file (default: {DEFAULT_MODEL})")
    parser.add_argument("--iou_threshold", 
                        type=float, 
                        default=DEFAULT_IOU_THRESHOLD,  
                        help=f"IoU overlap threshold (default: {DEFAULT_IOU_THRESHOLD})")
    parser.add_argument("--conf_threshold",
                        type=float, default=DEFAULT_CONF_THRESHOLD, 
                        help=f"YOLO detection confidence threshold (default: {DEFAULT_CONF_THRESHOLD})")
    parser.add_argument("--min_duration",  
                        type=float, 
                        default=DEFAULT_MIN_DURATION, 
                        help=f"Minimum event duration in seconds (default: {DEFAULT_MIN_DURATION})")
    parser.add_argument("--frame_skip",  
                        type=int, 
                        default=DEFAULT_FRAME_SKIP, 
                        help=f"Process every Nth frame (default: {DEFAULT_FRAME_SKIP})")
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    run_all(
        csv_path   = ROOT_DIR / args.csv,
        model_path    = args.model,
        iou_threshold = args.iou_threshold,
        conf_threshold= args.conf_threshold,
        min_duration  = args.min_duration,
        frame_skip    = args.frame_skip,
    )
    