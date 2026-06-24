"""
Baseline cross-sucking detector using YOLO bounding box overlap.
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
from config import ROOT_DIR, SOURCE_VIDEOS_DIR, METADATA_DIR_CLOUD
from scripts.run_models.run_testing import get_split_label

# Defaults
DEFAULT_MODEL = "yolo26x.pt"
DEFAULT_IOU_THRESHOLD = 0.1
DEFAULT_MIN_DURATION = 0.5
DEFAULT_CONF_THRESHOLD = 0.5
TARGET_CLASS_NAME = "cow"
DEFAULT_FRAME_SKIP = 1


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) between two bounding boxes
    and return the pixel coordinates of their intersection rectangle.

    Parameters
    ----------
    box_a : list
        [x1, y1, x2, y2] coordinates of box A.
    box_b : list
        [x1, y1, x2, y2] coordinates of box B.

    Returns
    -------
    iou : float
        IoU score in [0, 1]. 0.0 means no overlap; 1.0 means identical boxes.
    intersection_box : list or None
        [x1, y1, x2, y2] pixel coordinates of the overlapping region,
        or None if the boxes do not overlap.

    Examples
    --------
    .. code-block:: python

        iou, box = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        print(iou)   # 0.142...
        print(box)   # [5, 5, 10, 10]
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

    iou = intersection / union if union > 0 else 0.0
    intersection_box = (
        [round(inter_x1), round(inter_y1), round(inter_x2), round(inter_y2)]
        if intersection > 0 else None
    )
    return iou, intersection_box


def frame_has_overlap(boxes, iou_threshold):
    """
    Check whether any two bounding boxes in a single frame overlap
    above the IoU threshold.

    Checks every possible pair of detected calf boxes and tracks the pair
    with the highest IoU. Returns that pair's intersection box for storage
    in the event metadata.

    Parameters
    ----------
    boxes : list
        List of [x1, y1, x2, y2] bounding boxes for all detected calves
        in this frame.
    iou_threshold : float
        Minimum IoU score to consider an overlap.

    Returns
    -------
    overlap_detected : bool
        True if any pair of boxes exceeds the IoU threshold.
    best_inter_box : list or None
        [x1, y1, x2, y2] of the intersection region of the highest-IoU pair,
        or None if no overlap was found.

    Examples
    --------
    .. code-block:: python

        boxes = [[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 200, 200]]
        detected, box = frame_has_overlap(boxes, iou_threshold=0.1)
        print(detected)  # True
        print(box)       # [5, 5, 10, 10]
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


def extract_events(
    frame_flags: list[bool],
    fps: float,
    min_duration: float,
    frame_skip: int,
    confidences: list[float],
    frame_boxes: list,
    frame_indices: list[int],
) -> list[dict]:
    """
    Group consecutive flagged frames into discrete cross-sucking events.

    A frame is flagged when its IoU exceeds the overlap threshold. Consecutive
    flagged frames are merged into a single event. Events shorter than
    ``min_duration`` seconds are discarded as noise.

    Parameters
    ----------
    frame_flags : list of bool
        Per-frame overlap flag. True when bounding-box overlap was detected.
    fps : float
        Frames per second of the source video, used to convert frame
        numbers to timestamps in seconds.
    min_duration : float
        Minimum event length in seconds. Events below this threshold are dropped.
    frame_skip : int
        Frame-skip factor used during inference, needed to compute the minimum
        frame count threshold correctly.
    confidences : list of float
        Per-frame maximum YOLO detection confidence (0.0 when no detections).
    frame_boxes : list
        Per-frame intersection box [x1, y1, x2, y2], or None for frames where
        no overlap was detected. Must be the same length as ``frame_flags``.
    frame_indices : list of int
        Actual frame indices in the source video, accounting for frame skip.

    Returns
    -------
    list of dict
        Each dict contains:

        - ``start_sec`` (float): Event start time in seconds.
        - ``end_sec`` (float): Event end time in seconds.
        - ``duration_sec`` (float): Total duration of the event.
        - ``avg_confidence`` (float): Mean YOLO confidence across event frames.
        - ``intersection_box`` (list): Per-frame dicts with keys
          ``frame``, ``x1``, ``y1``, ``x2``, ``y2``.

    Notes
    -----
    An event that runs to the very last frame of the video is flushed
    and saved even if no explicit end flag is encountered.
    """
    events = []
    in_event = False
    start_frame = 0
    event_confs = []
    event_interbox = []
    min_frames = max(1, round(min_duration * fps / frame_skip))

    for idx, flagged in enumerate(frame_flags):
        actual_frame = frame_indices[idx]

        if flagged and not in_event:
            in_event = True
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
            in_event = False
            event_confs = []
            event_interbox = []

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


def load_split_from_csv(csv_path: Path) -> list[Path]:
    """
    Load video paths from a single split CSV.

    Parameters
    ----------
    csv_path : Path
        Path to a test.csv file (e.g. ``data/processed/day_based/test.csv``).

    Returns
    -------
    list of Path
        Resolved absolute video paths, deduplicated.

    Raises
    ------
    ValueError
        If the CSV is empty after deduplication.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path
        from scripts.run_models.baseline.baseline import load_split_from_csv

        paths = load_split_from_csv(Path("data/processed/day_based/test.csv"))
        print(paths[0])  # /path/to/source/video.mp4
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


def detect_video(
    video_path: Path,
    model,
    target_ids: set,
    iou_threshold: float,
    conf_threshold: float,
    min_duration: float,
    frame_skip: int,
) -> dict:
    """
    Run the full baseline detection pipeline on a single video file.

    Opens the video, detects calves per frame using YOLO, computes pairwise
    IoU overlap, groups consecutive flagged frames into events, and returns
    a metadata dict ready to be written to JSON.

    Parameters
    ----------
    video_path : Path
        Absolute path to the input video file.
    model : ultralytics.YOLO
        Loaded YOLO model instance.
    target_ids : set
        Set of YOLO class IDs corresponding to the target animal class.
    iou_threshold : float
        Minimum IoU to flag a frame as containing an overlap event.
    conf_threshold : float
        Minimum YOLO detection confidence to keep a bounding box.
    min_duration : float
        Minimum event duration in seconds; shorter events are discarded.
    frame_skip : int
        Process every Nth frame. Frame indices in output reflect actual
        source video frame numbers.

    Returns
    -------
    dict
        Metadata dict containing detection parameters and per-event results.
        Written directly to JSON by ``run_all``. Fields include
        ``identifier``, ``video_path``, ``model``, ``iou_threshold``,
        ``conf_threshold``, ``min_duration_sec``, ``fps``, ``total_frames``,
        ``total_duration_sec``, ``frame_skip``, ``cross_sucking_detected``,
        ``num_events``, and ``events``.

    Raises
    ------
    FileNotFoundError
        If the video file cannot be opened by OpenCV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {video_path.name} | {width}x{height} @ {fps:.1f} fps | {total_frames} frames")

    frame_flags    = []
    frame_confs    = []
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

    return {
        "identifier":              video_path.name,
        "video_path":              str(video_path.resolve()),
        "model":                   str(DEFAULT_MODEL),
        "iou_threshold":           iou_threshold,
        "conf_threshold":          conf_threshold,
        "min_duration_sec":        min_duration,
        "fps":                     fps,
        "total_frames":            total_frames,
        "total_duration_sec":      round(total_frames / fps, 2),
        "frame_skip":              frame_skip,
        "cross_sucking_detected":  len(events) > 0,
        "num_events":              len(events),
        "events":                  events,
    }


def run_all(
    csv_path: Path,
    model_path: str,
    iou_threshold: float,
    conf_threshold: float,
    min_duration: float,
    frame_skip: int,
) -> None:
    """
    End-to-end pipeline: load split CSV → run detection → save JSON metadata.

    Iterates over every unique video in the split CSV, runs ``detect_video``
    on each, and writes one JSON metadata file per video to the baseline
    metadata directory. Videos whose output JSON already exists are skipped
    automatically, making this function safe to re-run after interruptions.

    Parameters
    ----------
    csv_path : Path
        Path to a split test CSV (e.g. ``data/processed/day_based/test.csv``).
    model_path : str
        YOLO weights filename (e.g. ``'yolo26x.pt'``).
        Downloaded automatically on first run if not found locally.
    iou_threshold : float
        IoU threshold to flag a frame as containing an overlap event.
    conf_threshold : float
        Minimum YOLO detection confidence to keep a bounding box.
    min_duration : float
        Minimum event duration in seconds.
    frame_skip : int
        Process every Nth frame (1 = every frame).

    Returns
    -------
    None
        Writes JSON metadata files to disk and does not return anything.

    Raises
    ------
    ValueError
        If the target class is not present in the loaded model.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path
        from scripts.run_models.baseline.baseline import run_all

        run_all(
            csv_path=Path("data/processed/day_based/test.csv"),
            model_path="yolo26x.pt",
            iou_threshold=0.1,
            conf_threshold=0.5,
            min_duration=1.0,
            frame_skip=30,
        )
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
    target_ids = (
        {class_name_to_id[TARGET_CLASS_NAME]}
        if TARGET_CLASS_NAME in class_name_to_id else set()
    )
    if not target_ids:
        raise ValueError(
            f"'{TARGET_CLASS_NAME}' not found in model: {list(model.names.values())}"
        )

    for video_path in video_paths:
        json_path = (
            METADATA_DIR_CLOUD / split_name / "baseline"
            / f"{video_path.stem}_results.json"
        )

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Baseline cross-sucking detector using {DEFAULT_MODEL} bounding box overlap."
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="Path to a split test.csv (e.g. data/processed/day_based/test.csv)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"YOLO weights file (default: {DEFAULT_MODEL})")
    parser.add_argument("--iou_threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
                        help=f"IoU overlap threshold (default: {DEFAULT_IOU_THRESHOLD})")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f"YOLO detection confidence threshold (default: {DEFAULT_CONF_THRESHOLD})")
    parser.add_argument("--min_duration", type=float, default=DEFAULT_MIN_DURATION,
                        help=f"Minimum event duration in seconds (default: {DEFAULT_MIN_DURATION})")
    parser.add_argument("--frame_skip", type=int, default=DEFAULT_FRAME_SKIP,
                        help=f"Process every Nth frame (default: {DEFAULT_FRAME_SKIP})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(
        csv_path=ROOT_DIR / args.csv,
        model_path=args.model,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        min_duration=args.min_duration,
        frame_skip=args.frame_skip,
    )