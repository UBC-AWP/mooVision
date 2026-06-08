"""
Module for running YOLO models to detect cross-sucking events and output event metadata.

NOTE: This file is meant to send clips for human review: add pre and post
buffer of 10-30s for each video meta-data output to ensure we capture the
whole event

NOTE: Redo Docstrings

NOTE: Make Tests
"""

import sys
from pathlib import Path
import os
import json
import argparse
from ultralytics import YOLO
import numpy as np
import cv2

sys.path.append(str(Path(__file__).parent.parent.parent))


def extract_events(
    fps: int, buffer: int, frame_detections: list[list[dict]]
) -> list[dict]:
    """

        # NOTE: This is not designed for event tracking!

    Flagging frames into events with start/end timestamps and collect the
    intersection box coordinates for every flagged frame within each event.

    Args:
        frame_flags (list[bool]): Per-frame overlap flag — True if overlap detected
        fps (float): Frames per second of the video, used to convert frame
                     numbers into timestamps in seconds
        min_duration (float): Minimum event length in seconds to keep
        confidences (list[float]): Per-frame max YOLO detection confidence
        frame_boxes (list): Per-frame intersection box [x1, y1, x2, y2],
                            or None if that frame was not flagged
        frame_indices (list): Actual frame indices in the video (accounting for frame skip)

    Returns:
        list[dict]: Each dict represents one flagged event:
            - start_sec (float): Event start time in seconds
            - end_sec (float): Event end time in seconds
            - duration_sec (float): Total event duration in seconds
            - avg_confidence (float): Average YOLO detection confidence
                                      across all frames in the event
            - intersection_box (list[dict]): Per-frame intersection coordinates,
                                             each entry has 'frame', 'x1', 'y1',
                                             'x2', 'y2'
    """
    # Max number of frames before separating cross-sucking events
    max_dist = buffer * fps
    end_frame = -1
    start_frame = frame_detections[0][0]["frame"]

    events = []  # Cross-Sucking Events
    event_interbox = []  # Bounding Boxes
    event_confs = []  # Confidence Scores

    for frame_dets in frame_detections:
        current_frame = frame_dets[0]["frame"]
        current_dist = current_frame - start_frame

        if current_dist <= max_dist:
            for det in frame_dets:
                event_confs.append(frame_detections[52][0]["confidence"])
                event_interbox.append(
                    {
                        "frame": det["frame"],
                        "x1": det["x1"],
                        "y1": det["y1"],
                        "x2": det["x2"],
                        "y2": det["y2"],
                    }
                )
            end_frame = current_frame  # Store previous frame!

        else:
            events.append(
                {
                    "start_sec": round(start_frame / fps, 2),
                    "end_sec": round(end_frame / fps, 2),
                    "duration_sec": round(end_frame - start_frame, 2),
                    "avg_confidence": round(float(np.mean(event_confs)), 3),
                    "intersection_box": event_interbox,
                }
            )
            event_interbox = []
            event_confs = []
            start_frame = current_frame

    # End logic, add last event.
    if (end_frame == frame_detections[-1][0]["frame"]) and (end_frame != start_frame):
        events.append(
            {
                "start_sec": round(start_frame / fps, 2),
                "end_sec": round(end_frame / fps, 2),
                "duration_sec": round(end_frame - start_frame, 2),
                "avg_confidence": round(float(np.mean(event_confs)), 3),
                "intersection_box": event_interbox,
            }
        )
    return events


def run_detection(
    video_path: str,
    model_path: str,
    conf_threshold: float,
    iou_threshold: float,
    min_duration: float,
    frame_skip: int,
    show_video: bool = False,
    target_class: str = "cross-sucking",
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
            f"Neither '{target_class}' nor 'cross-sucking' found in model classes: "
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

    frame_detections = []
    frame_idx = 0

    print("[INFO] Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            # frame_detections.append([])  # empty detection for skipped frame
            continue

        # Run YOLO inference
        results = model(frame, conf=conf_threshold, verbose=False)[0]

        # Show annotated frame
        if show_video:
            annotated = results.plot()
            cv2.imshow("Basic Detection", annotated)
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

        if dets:
            frame_detections.append(dets)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  ...frame {frame_idx}/{total_frames}")

    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    print(f"[INFO] Processed {frame_idx} frames, collected detections.")

    # SAVE FRAMES HERE
    print("Frames saved to <output_path>")

    # Apply Event Extraction
    events = extract_events

    # Build metadata — same format as baseline.py
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = "data/results/metadata/yolo-basic"
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
    json_path = os.path.join(output_dir, f"{video_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Cross-sucking detection and event linking using {YOLO} and baseline logic."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        default="runs/detect/MooVision/cross-sucking/weights/best.pt",
        help="YOLO weights file (default: runs/detect/MooVision/cross-sucking/weights/best.pt)",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0,
        help="IoU overlap threshold (default: 0)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0,
        help="YOLO detection confidence threshold (default: 0)",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0,
        help="Minimum event duration in seconds (default: 0)",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--show_video",
        type=bool,
        default=False,
        action="store_true",
        help="Display video during annotation, useful for running on sample videos (default: False)",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default="cross-sucking",
        help="Target class for detection (default: cross-sucking)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_detection(
        video_path=args.video,
        model_path=args.model,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        min_duration=args.min_duration,
        frame_skip=args.frame_skip,
        show_video=args.show_video,
        target_class=args.target_class,
    )
