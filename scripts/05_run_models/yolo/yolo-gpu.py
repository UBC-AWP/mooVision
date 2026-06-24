"""
Module for running YOLO models to detect cross-sucking events and output event metadata.

Optimized for GPU Batched Streaming on HPC clusters (UBC Sockeye).
"""

import sys
from pathlib import Path
import os
import json
import argparse
from ultralytics import YOLO
import numpy as np
import cv2

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
# from config import ROOT_DIR


def extract_events(
    fps: int, buffer: int, frame_detections: list[list[dict]]
) -> list[dict]:
    """
    [Your exact original event extraction logic remains completely unchanged here]
    """
    if not frame_detections:
        return []

    max_dist = buffer * fps
    end_frame = -1
    start_frame = frame_detections[0][0]["frame"]
    last_frame = start_frame

    events = []
    event_interbox = []
    event_confs = []

    for frame_dets in frame_detections:
        current_frame = frame_dets[0]["frame"]
        current_dist = current_frame - last_frame

        if current_dist < max_dist:
            for det in frame_dets:
                event_confs.append(frame_dets[0]["confidence"])
                event_interbox.append(
                    {
                        "frame": det["frame"],
                        "x1": det["x1"],
                        "y1": det["y1"],
                        "x2": det["x2"],
                        "y2": det["y2"],
                    }
                )
            last_frame = current_frame

        else:
            end_frame = last_frame
            events.append(
                {
                    "start_sec": round(start_frame / fps, 2),
                    "end_sec": round(end_frame / fps, 2),
                    "duration_sec": round((end_frame - start_frame) / fps, 2),
                    "avg_confidence": round(float(np.mean(event_confs)), 3),
                    "intersection_box": event_interbox,
                }
            )
            event_interbox = []
            event_confs = []
            for det in frame_dets:
                event_confs.append(frame_dets[0]["confidence"])
                event_interbox.append(
                    {
                        "frame": det["frame"],
                        "x1": det["x1"],
                        "y1": det["y1"],
                        "x2": det["x2"],
                        "y2": det["y2"],
                    }
                )
            start_frame = current_frame
            last_frame = current_frame

    end_frame = current_frame
    events.append(
        {
            "start_sec": round(start_frame / fps, 2),
            "end_sec": round(end_frame / fps, 2),
            "duration_sec": round((end_frame - start_frame) / fps, 2),
            "avg_confidence": round(float(np.mean(event_confs)), 3),
            "intersection_box": event_interbox,
        }
    )
    return events


def run_gpu_batch_detection(
    model_path: str,
    video_paths: list[str],  # Accept a list of paths instead of just one!
    output_dir: str,
    conf_threshold: float,
    iou_threshold: float,
    min_duration: float,
    buffer: int,
    frame_skip: int,
    target_class: str = "cross-sucking",
):
    """
    Optimized GPU Batch Pipeline.
    Streams frames from multiple videos sequentially without loading full datasets into memory.
    """
    print(f"[INFO] Loading model onto GPU: {model_path}")
    model = YOLO(model_path)

    # Resolve target class ID
    class_name_to_id = {v: k for k, v in model.names.items()}
    target_id = class_name_to_id.get(target_class, 0)
    print(f"[INFO] Targeting class: '{target_class}' (ID: {target_id})")

    os.makedirs(output_dir, exist_ok=True)

    # === NEW: BULLETPROOF FILE INTEGRITY CHECK ===
    print(
        f"[INFO] Auditing {len(video_paths)} videos for corruption before inference..."
    )
    verified_video_paths = []

    for path in video_paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"[WARNING] Skipping {Path(path).name}: File is empty or missing.")
            continue

        # Try to briefly open the video header
        cap = cv2.VideoCapture(path)
        is_opened = cap.isOpened()
        if is_opened:
            ret, _ = cap.read()  # Try to read just the very first frame
            if ret:
                verified_video_paths.append(path)
            else:
                print(
                    f"[CRITICAL WARNING] Skipping {Path(path).name}: Found file but cannot read frames (Corrupt video stream)."
                )
        else:
            print(
                f"[CRITICAL WARNING] Skipping {Path(path).name}: OpenCV failed to open codec stream."
            )
        cap.release()

    print(
        f"[INFO] Audit complete. {len(verified_video_paths)} / {len(video_paths)} videos passed integrity checks."
    )

    if not verified_video_paths:
        print("[ERROR] No valid videos left to process. Exiting safely.")
        return

    # Initialize variables tracking the generator's state
    current_video_path = None
    frame_detections = []
    frame_counter = 0
    fps = 30.0
    total_frames = 0

    # 1. Fire up the high-speed streaming engine
    # vid_stride handles frame skipping instantly in C++ while decoding
    print(
        f"[INFO] Initiating streaming pipeline for {len(verified_video_paths)} videos..."
    )
    results_generator = model.predict(
        source=verified_video_paths,
        conf=conf_threshold,
        iou=iou_threshold,
        device=0,  # <--- Hard enforces Sockeye's GPU
        stream=True,  # <--- Generates frames lazily to preserve VRAM
        vid_stride=frame_skip,  # <--- Native frame skip optimization
        verbose=False,
    )

    def process_and_save_metadata(v_path, f_dets, video_fps, total_f):
        """Helper to isolate calculations and JSON dumping when a video finishes"""
        if not f_dets:
            print(f"  --> No target behaviors found in: {Path(v_path).name}")
            return

        print(f"  --> Processing metadata & events for: {Path(v_path).name}")
        events = extract_events(video_fps, buffer, f_dets)

        video_name = Path(v_path).stem
        metadata = {
            "identifier": Path(v_path).name,
            "video_path": os.path.abspath(v_path),
            "model": model_path,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "min_duration_sec": min_duration,
            "frame_skip": frame_skip,
            "fps": video_fps,
            "total_frames": total_f,
            "total_duration_sec": round(total_f / video_fps, 2),
            "cross_sucking_detected": len(events) > 0,
            "num_events": len(events),
            "events": events,
        }

        json_path = os.path.join(output_dir, f"{video_name}_results.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # 2. Consume the frame stream dynamically
    for result in results_generator:
        frame_orig_path = result.path  # Tracks which file this specific frame is from

        # Check if the generator just shifted to a brand new video file
        if frame_orig_path != current_video_path:
            # Save the previous video's results if it exists
            if current_video_path is not None:
                process_and_save_metadata(
                    current_video_path, frame_detections, fps, total_frames
                )

            # Switch focus to the incoming video file
            current_video_path = frame_orig_path
            frame_detections = []
            frame_counter = 0

            # Quickly query metadata from the file header using OpenCV
            cap = cv2.VideoCapture(current_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(
                f"\n[STREAMING] Now evaluating video: {Path(current_video_path).name}"
            )

        # Sync frame indexes accurately with the native stride skipping
        frame_counter += frame_skip

        # Collect target array detections for this exact frame
        dets = []
        for box in result.boxes:
            if int(box.cls[0].item()) == target_id:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(
                    {
                        "frame": frame_counter,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": float(box.conf[0].item()),
                    }
                )

        if dets:
            frame_detections.append(dets)

    # 3. Handle the final video remaining in the pipeline upon stream closure
    if current_video_path is not None:
        process_and_save_metadata(
            current_video_path, frame_detections, fps, total_frames
        )

    print("\n═" * 50 + "\n[SUCCESS] Entire data batch complete on GPU!\n" + "═" * 50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-sucking detection event linking using GPU optimization."
    )
    parser.add_argument(
        "--video_paths",
        required=True,
        nargs="+",  # <--- Allows passing multiple space-separated paths from your orchestrator!
        help="Space-separated paths to input video files",
    )
    parser.add_argument("--model_path", required=True, help="YOLO weights file.")
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save metadata output."
    )
    parser.add_argument("--iou_threshold", type=float, default=0)
    parser.add_argument("--conf_threshold", type=float, default=0.1)
    parser.add_argument("--min_duration", type=float, default=0)
    parser.add_argument("--frame_skip", type=int, default=10)
    parser.add_argument("--buffer", type=int, default=60)
    parser.add_argument("--target_class", type=str, default="cross-sucking")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gpu_batch_detection(
        video_paths=args.video_paths,
        model_path=args.model_path,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        min_duration=args.min_duration,
        frame_skip=args.frame_skip,
        buffer=args.buffer,
        target_class=args.target_class,
    )
