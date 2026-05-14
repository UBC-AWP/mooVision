import argparse
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n.pt"    
DEFAULT_IOU_THRESHOLD = 0.1     # Minimum IoU to consider two boxes "overlapping"
DEFAULT_MIN_DURATION = 10       # Minimum seconds of continuous overlap to flag an event
DEFAULT_CONF_THRESHOLD = 0.7       # Minimum YOLO detection confidence to keep a box
TARGET_CLASS_NAME = "cow"


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        box_a, box_b: [x1, y1, x2, y2]  (left edge, top edge, right edge, bottom edge)

    Returns:
        float: IoU score in [0, 1]
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

    return intersection / union if union > 0 else 0.0


def frame_has_overlap(boxes, iou_threshold):
    """
    Check whether any two bounding boxes in a frame overlap above the threshold.

    Args:
        boxes:          List of [x1, y1, x2, y2] arrays
        iou_threshold:  Float in [0, 1]

    Returns:
        (bool, float): (overlap_detected, max_iou_found)
    """

def extract_events(frame_flags, fps, min_duration, confidences):
    """
    Group consecutive flagged frames into events with start/end timestamps.

    Args:
        frame_flags:  List of bool, one per frame (True = overlap detected)
        fps:          Frames per second of the video
        min_duration: Minimum event length in seconds
        confidences:  List of float, detection confidence per frame

    Returns:
        List of dicts: [{start_sec, end_sec, avg_confidence}, ...]
    """


def run_detection(video_path, model_path, iou_threshold, conf_threshold, min_duration):
    """
    Full pipeline: load video → detect calves → check overlap → output metadata.

    Args:
        video_path:     Path to the input video file (.mp4, .avi, etc.)
        model_path:     YOLOv8 model weights (e.g. 'yolov8n.pt')
        iou_threshold:  IoU threshold to flag overlapping boxes
        conf_threshold: Minimum YOLO detection confidence
        min_duration:   Minimum event length in seconds

    Returns:
        dict: metadata dict (also written to JSON)
    """

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    class_name_to_id = {v: k for k, v in model.names.items()}
    target_ids = {class_name_to_id[TARGET_CLASS_NAME] if TARGET_CLASS_NAME in class_name_to_id else set()}

    if not target_ids:
        raise ValueError(
            f"{TARGET_CLASS_NAME} is not found in model classes: {list(model.names.values())}"
        )
    print(f"[INFO] Tracking class IDs: {target_ids}  ({TARGET_CLASS_NAME})")

    # Opening the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps | {total_frames} frames")


    frame_flags    = []   # True/False per frame: overlap detected?
    frame_confs    = []   # Max detection confidence in that frame
    frame_idx      = 0

    print("[INFO] Processing frames...")
    while True:
        ret, frame = cap.read() # reads the next frame
        if not ret:
            break  # End of video

        # Run YOLO inference on the frame
        results = model(frame, conf=conf_threshold, verbose=False)[0]

        # Show the frame with bounding boxes drawn
        annotated_frame = results.plot()
        cv2.imshow("Calf Detection", annotated_frame)
        cv2.waitKey(1)  # 1ms delay, keeps the window responsive

        # Filter detections to only our target classes (cows/calves)
        boxes   = []
        confs   = []
        for box in results.boxes:
            if int(box.cls[0].item()) in target_ids:
                boxes.append(box.xyxy[0].tolist())
                confs.append(float(box.conf[0].item()))
 
        # Check for overlapping pairs
        overlap_detected = False
        if len(boxes) >= 2:
            overlap_detected, _ = frame_has_overlap(boxes, iou_threshold)
 
        frame_flags.append(overlap_detected)
        frame_confs.append(max(confs) if confs else 0.0)
        frame_idx += 1
 
        if frame_idx % 100 == 0:
            print(f"  ...frame {frame_idx}/{total_frames}")
 
    cap.release()
 
    # Group flagged frames into events
    events = extract_events(frame_flags, fps, min_duration, frame_confs)
 
    # Build metadata
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = "results/metadata/baseline"
    os.makedirs(output_dir, exist_ok=True)
 
    metadata = {
        "identifier":             os.path.basename(video_path),
        "video_path":             os.path.abspath(video_path),
        "model":                  model_path,
        "iou_threshold":          iou_threshold,
        "conf_threshold":         conf_threshold,
        "min_duration_sec":       min_duration,
        "fps":                    fps,
        "total_frames":           total_frames,
        "total_duration_sec":     round(total_frames / fps, 2),
        "cross_sucking_detected": len(events) > 0,
        "num_events":             len(events),
        "events":                 events,
    }
 
    # Save JSON
    json_path = os.path.join(output_dir, f"{video_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
 
    # Print summary to console
    print("\n" + "═" * 50)
    print(f"  VIDEO:    {metadata['identifier']}")
    print(f"  FLAGGED:  {metadata['cross_sucking_detected']}")
    print(f"  EVENTS:   {metadata['num_events']}")
    for i, ev in enumerate(events):
        print(f"    Event {i+1}: {ev['start_sec']}s → {ev['end_sec']}s "
              f"({ev['duration_sec']}s) | conf={ev['avg_confidence']}")
    print(f"  OUTPUT:   {json_path}")
    print("═" * 50 + "\n")
 
    return metadata
