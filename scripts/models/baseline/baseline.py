import argparse
import json
import os
import cv2
import re
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent)) 
from config import SOURCE_VIDEOS_DIR,BASELINE_METADATA_DIR_NEW,READ_DF_PATH

DEFAULT_DATA_ROOT = READ_DF_PATH
DEFAULT_MODEL = "yolo26x.pt"    
DEFAULT_IOU_THRESHOLD = 0.1    # Minimum IoU to consider two boxes "overlapping"
DEFAULT_MIN_DURATION = 0.5       # Minimum seconds of continuous overlap to flag an event
DEFAULT_CONF_THRESHOLD = 0.5      # Minimum YOLO detection confidence to keep a box
TARGET_CLASS_NAME = "cow"
DEFAULT_FRAME_SKIP = 1

SPLIT_NAMES = ["day_based", "pen_based", "period_based", "random", "pipeline_demo"]

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


def extract_events(frame_flags, fps, min_duration, confidences, frame_boxes, frame_indices):
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
    events = []
    in_event = False
    start_frame = 0
    event_confs = []
    event_interbox = []
    min_frames = max(1, round(min_duration * fps / DEFAULT_FRAME_SKIP))

    for idx, flagged in enumerate(frame_flags):
        actual_frame = frame_indices[idx]

        if flagged and not in_event:
            # Event starts
            in_event = True
            start_frame = actual_frame
            event_confs = [confidences[idx]]
            event_interbox.append({"frame": actual_frame, "x1": frame_boxes[idx][0], "y1": frame_boxes[idx][1],
                                    "x2": frame_boxes[idx][2], "y2": frame_boxes[idx][3]})

        elif flagged and in_event:
            # Event continues
            event_confs.append(confidences[idx])
            event_interbox.append({"frame": actual_frame, "x1": frame_boxes[idx][0], "y1": frame_boxes[idx][1],
                                    "x2": frame_boxes[idx][2], "y2": frame_boxes[idx][3]})

        elif not flagged and in_event:
            # Event just ended — evaluate it
            end_frame = actual_frame
            duration = (end_frame - start_frame) / fps
            
            if len(event_confs) >= min_frames:
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
        end_frame = frame_indices[-1]
        duration = (end_frame - start_frame) / fps
        if len(event_confs) >= min_frames:
            events.append({
                "start_sec":      round(start_frame / fps, 2),
                "end_sec":        round(end_frame / fps, 2),
                "duration_sec":   round(duration, 2),
                "avg_confidence": round(float(np.mean(event_confs)), 3),
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

def run_detection(video_paths, model_path, iou_threshold, conf_threshold, min_duration, frame_skip):
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
        frame_skip (int): Process every Nth frame

    Returns:
        dict: Full metadata dict 
            also written to JSON at results/metadata/baseline/<video_name>_results.json

    Raises:
        FileNotFoundError: If video or model file cannot be found
        ValueError: If target class is not in model
    """
    video_paths,folder_name = read__df(video_paths)
    # Download and loading the model
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    # video_paths = extract_video_path(video_paths)
    
    # Validate inputs
    # Extract relative parent path for output directory
    m = re.search(r"videos[\\/](.*)$", str(video_paths))
    if m:
        rel = Path(m.group(1))  # Pen 2 - Group 2/POSTWEANING/Day 1/<file>.mp4
        rel_parent = rel.parent  # Pen 2 - Group 2/POSTWEANING/Day 1
    else:
        # fallback if pattern not found
        rel_parent = Path()
    
    output_dir = BASELINE_METADATA_DIR_NEW / folder_name / rel_parent
    os.makedirs(output_dir, exist_ok=True)
    
    for video_path in video_paths:
        expected_metadata_file = output_dir / f"{video_path.stem}_results.json"
        
        if expected_metadata_file.is_file():
            print(f"Metadata already exists for {video_path.name}. Skipping detection.")
            continue
        
        print(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

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
        frame_indices = []

        print("[INFO] Processing frames...")
        while True:
            ret, frame = cap.read() # reads the next frame
            if not ret:
                break  # End of video

            # Skip frames — only process every Nth frame
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # Run YOLO inference on the frame
            results = model(frame, conf=conf_threshold, verbose=False)[0]

            # Show the frame with bounding boxes drawn
            annotated_frame = results.plot()
            # cv2.imshow("Calf Detection", annotated_frame)
            # cv2.waitKey(1)  # 1ms delay, keeps the window responsive

            # Filter detections to only our target classes (cows/calves)
            boxes   = []
            confs   = []
            for box in results.boxes:
                if int(box.cls[0].item()) in target_ids:
                    boxes.append(box.xyxy[0].tolist())
                    confs.append(float(box.conf[0].item()))
    
            # Check for overlapping pairs
            overlap_detected, intersection_box  = frame_has_overlap(boxes, iou_threshold)
    
            frame_flags.append(overlap_detected)
            frame_confs.append(max(confs) if confs else 0.0)
            frame_indices.append(frame_idx)
            frame_interbox.append(intersection_box if overlap_detected else None)
            frame_idx += 1
    
            if frame_idx % 100 == 0:
                print(f"  ...frame {frame_idx}/{total_frames}")
    
        cap.release()
    
        # Group flagged frames into events
        events = extract_events(frame_flags, fps, min_duration, frame_confs, frame_interbox, frame_indices)
    
        # Build metadata
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
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
            "frame_skip":             frame_skip,
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
    
def read__df(data_path):
    """
        Read a CSV file, deduplicate rows, and extract absolute video paths.

        This function loads a dataframe from the specified `data_path`, removes
        duplicate rows based on the 'source_video_path' column, and slices the 
        dataframe to inspect just the first row. It then converts the relative 
        video path into an absolute path using a predefined directory constant.

        Parameters
        ----------
        data_path : str or pathlib.Path
            The file path to the CSV configuration or data file.

        Returns
        -------
        list of pathlib.Path
            A list containing the resolved absolute path(s) to the video file(s).

        Raises
        ------
        ValueError
            If the resulting dataframe is empty after loading or deduplication.

        See Also
        --------
        pandas.read_csv : Underlying function used to load the data.

        Examples
        --------
        >>> read_df("metadata.csv")
        [PosixPath('/path/to/source_videos/dataset/category/video/file.mp4')]
        """
    split_name = ["day_based", "pen_based", "period_based", "random", "pipeline_demo"]

    for name in split_name:
        try:
            data_path = DEFAULT_PATH / name / "test.csv"
        except Exception as e:
            print(f"Error constructing data path for {name}: {e}")
            continue
        df = pd.read_csv(data_path, index_col=0)
        df = df.drop_duplicates(subset=["source_video_path"])
        df = df.iloc[:1]
        if df.empty:
            raise ValueError("df is empty.")
        
        video_paths = df["source_video_path"]

        clean_paths = []
        for video_path in video_paths:

            cln_str = video_path.replace("\\", "/")
            cln_path = Path(cln_str)
            rel_path = Path(*cln_path.parts[-4:])  # Relies on file naming conventions...
            abs_path = SOURCE_VIDEOS_DIR / rel_path
            
            clean_paths.append(abs_path)
        return clean_paths,name
    
def load_split(data_root: Path, split_name: str) -> list[Path]:
    """
    Load the test CSV for a single split and resolve video paths to absolute paths.

    Reads <data_root>/<split_name>/test.csv, drops duplicate rows on
    'source_video_path', and converts each relative path to an absolute
    path under SOURCE_VIDEOS_DIR.

    The last 4 path components of each relative video path are kept
    (e.g. Pen/Stage/Day/file.mp4) and joined onto SOURCE_VIDEOS_DIR.
    This relies on a consistent directory depth in the video naming convention.

    Args:
        data_root:
            Root directory containing split subdirectories.
        split_name:
            Name of the split subdirectory (e.g. 'day_based').

    Returns:
        List of resolved absolute Path objects pointing to video files.

    Raises:
        FileNotFoundError: If the CSV for this split does not exist.
        ValueError:        If the CSV is empty after deduplication.
    """
    csv_path = data_root / split_name / "test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    df = df.drop_duplicates(subset=["source_video_path"])

    if df.empty:
        raise ValueError(f"No rows in {csv_path} after deduplication.")

    clean_paths = []
    for raw in df["source_video_path"]:
        cln  = Path(raw.replace("\\", "/"))
        rel  = Path(*cln.parts[-4:])   # Pen/Stage/Day/file.mp4
        clean_paths.append(SOURCE_VIDEOS_DIR / rel)

    return clean_paths
   
def load_all_splits(data_root: Path) -> list[tuple[str, list[Path]]]:
    """
    Load video paths for every known split under `data_root`.
 
    Skips splits whose CSV is missing or empty with a warning rather
    than raising, so a partially populated data directory still runs.
 
    Args:
        data_root:
            Root directory containing split subdirectories
            (day_based/, pen_based/, etc.).
 
    Returns:
        List of (split_name, video_paths) tuples for each split that
        loaded successfully. Empty if no splits could be loaded.
    """
    results = []
    for name in SPLIT_NAMES:
        try:
            paths = load_split(data_root, name)
            results.append((name, paths))
            print(f"[INFO] Loaded split '{name}': {len(paths)} unique video(s)")
        except FileNotFoundError:
            print(f"[WARN] Split '{name}' not found — skipping.")
        except ValueError as e:
            print(f"[WARN] Split '{name}' skipped: {e}")
    return results

def resolve_output_path(video_path: Path, split_name: str) -> Path:
    """
    Resolve the output JSON path for a given video under the baseline results directory.
 
    Mirrors the source video's Pen/Stage/Day directory hierarchy under:
        BASELINE_METADATA_DIR_NEW / <split_name> / <Pen> / <Stage> / <Day> /
 
    The relative parent is extracted from the video path by matching everything
    after the 'videos' directory separator. Falls back to a flat output directory
    if the expected pattern is not found.
 
    Args:
        video_path: Absolute path to the source video file.
        split_name: Name of the data split (e.g. 'day_based').
 
    Returns:
        Full Path to the output JSON file (not yet created).
    """
    m = re.search(r"videos[\\/](.*)$", str(video_path))
    rel_parent = Path(m.group(1)).parent if m else Path()
    output_dir = BASELINE_METADATA_DIR_NEW / split_name / rel_parent
    return output_dir / f"{video_path.stem}_results.json"   

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Baseline cross-sucking detector using {DEFAULT_MODEL} bounding box overlap."
    )
    parser.add_argument(
        "--data_root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Root directory containing split CSVs (day_based/, pen_based/, etc.)",
    )
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
    run_detection(
        video_paths   = args.video,
        model_path    = args.model,
        iou_threshold = args.iou_threshold,
        conf_threshold= args.conf_threshold,
        min_duration  = args.min_duration,
        frame_skip    = args.frame_skip,
    )
    # extract_video_path(args.video)
    