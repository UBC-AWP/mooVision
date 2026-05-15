import argparse
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL = "yolo26m.pt"    
DEFAULT_IOU_THRESHOLD = 0.1     # Minimum IoU to consider two boxes "overlapping"
DEFAULT_MIN_DURATION = 1       # Minimum seconds of continuous overlap to flag an event
DEFAULT_CONF_THRESHOLD = 0.5       # Minimum YOLO detection confidence to keep a box
TARGET_CLASS_NAME = "cow"


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


def extract_events(frame_flags, fps, min_duration, confidences, frame_boxes):
    """
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
    events = []
    in_event = False
    start_frame = 0
    event_confs = []
    event_interbox = []

    for idx, flagged in enumerate(frame_flags):
        if flagged and not in_event:
            # Event starts
            in_event = True
            start_frame = idx
            event_confs = [confidences[idx]]
            event_interbox.append({"frame": idx, "x1": frame_boxes[idx][0], "y1": frame_boxes[idx][1],
                                    "x2": frame_boxes[idx][2], "y2": frame_boxes[idx][3]})

        elif flagged and in_event:
            # Event continues
            event_confs.append(confidences[idx])
            event_interbox.append({"frame": idx, "x1": frame_boxes[idx][0], "y1": frame_boxes[idx][1],
                                    "x2": frame_boxes[idx][2], "y2": frame_boxes[idx][3]})

        elif not flagged and in_event:
            # Event just ended — evaluate it
            end_frame = idx - 1
            duration = (end_frame - start_frame) / fps
            if duration >= min_duration:
                events.append({
                    "start_sec":        round(start_frame / fps, 2),
                    "end_sec":          round(end_frame / fps, 2),
                    "duration_sec":     round(duration, 2),
                    "avg_confidence":   round(float(np.mean(event_confs)), 3),
                    "intersection_box": event_interbox,
                })

            in_event = False
            event_confs = []

    # Handle event that runs to the very last frame
    if in_event:
        end_frame = len(frame_flags) - 1
        duration = (end_frame - start_frame) / fps
        if duration >= min_duration:
            events.append({
                "start_sec":      round(start_frame / fps, 2),
                "end_sec":        round(end_frame / fps, 2),
                "duration_sec":   round(duration, 2),
                "avg_confidence": round(float(np.mean(event_confs)), 3),
                "intersection_box": event_interbox,
            })

    return events


def run_detection(video_path, model_path, iou_threshold, conf_threshold, min_duration):
    """
    Full detection pipeline:
        load model → open video → detect calves per frame →
        check IoU overlap → collect intersection boxes →
        group into events → save JSON metadata

    Args:
        video_path (str): Path to the input video file (.mp4, .avi, etc.)
        model_path (str): YOLO model weights filename (e.g. 'yolo26m.pt').
                          Downloaded automatically on first run.
        iou_threshold (float): IoU threshold to flag a frame as overlapping
        conf_threshold (float): Minimum YOLO detection confidence to keep a box.
                                Detections below this are ignored before IoU check.
        min_duration (float): Minimum event duration in seconds. Events shorter
                              than this are discarded as noise.

    Returns:
        dict: Full metadata dict 
            also written to JSON at results/metadata/baseline/<video_name>_results.json
    """
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    class_name_to_id = {v: k for k, v in model.names.items()}
    target_ids = {class_name_to_id[TARGET_CLASS_NAME]} if TARGET_CLASS_NAME in class_name_to_id else set()

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
    frame_interbox = []

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
            overlap_detected, intersection_box  = frame_has_overlap(boxes, iou_threshold)
 
        frame_flags.append(overlap_detected)
        frame_confs.append(max(confs) if confs else 0.0)
        frame_idx += 1
        frame_interbox.append(intersection_box if overlap_detected else None)
 
        if frame_idx % 100 == 0:
            print(f"  ...frame {frame_idx}/{total_frames}")
 
    cap.release()
 
    # Group flagged frames into events
    events = extract_events(frame_flags, fps, min_duration, frame_confs, frame_interbox)
 
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Baseline cross-sucking detector using {DEFAULT_MODEL} bounding box overlap."
    )
    parser.add_argument("--video",
                        required=True, 
                        help="Path to input video file")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_detection(
        video_path    = args.video,
        model_path    = args.model,
        iou_threshold = args.iou_threshold,
        conf_threshold= args.conf_threshold,
        min_duration  = args.min_duration,
    )