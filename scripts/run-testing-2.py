"""
Orchestrator script for running models on testing data (raw source videos)

NOTE: Need to add functionality for adding other arguments too! seq_NMS and yolo have different arguments, should be call the functions instead?
"""

from pathlib import Path
import sys
import json
import argparse
import subprocess
import pandas as pd
from ultralytics import YOLO
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.append(str(Path(__file__).parent.parent))
from config import ROOT_DIR, SOURCE_VIDEOS_DIR, LOCAL_DIR
from scripts.models.yolo.yolo import run_detection
from scripts.models.seq_NMS.seq_NMS import run_seq_nms_detection


def run_testing(
    model_type: str,
    model_path: str,
    data_path: str,
    conf_threshold: float,
    iou_threshold: float,
    min_duration: float,
    buffer: int,
    frame_skip: int,
    target_class: str = "cross-sucking",
):
    """
    Run a model script on all source videos in testing set.

    Loads in a test.csv at `data_path` and runs the model
    specified at `model` on all source videos listed if they exist
    in the data directory. Specifically, this function loops through each
    video path in the data frame, and passes these paths the the script
    in models/ specified as `model`. This script will run the model on
    said video and output metadata to the data directory.

    Parameters
    ----------
    model : str
        Type of model to use. One of ['yolo', 'seq-NMS']
    model_path : str
        Path to model to use.
    data_path : str
        Relative Path to test.csv file inside ROOT_DIR.

    Returns
    -------
    None
        This function reads to disk and does not return anything.


    Raises
    ------


    Examples
    --------

    """
    if model_type not in ["yolo", "seq-NMS"]:
        raise ValueError("model must be one of:['yolo', 'seq-NMS']")

    # Read in Data
    if not (ROOT_DIR / data_path).exists():
        raise FileNotFoundError(f"Could not find file: {ROOT_DIR / data_path}")
    df = pd.read_csv(ROOT_DIR / data_path, index_col=0)

    if df.empty:
        raise ValueError("df is empty.")

    # the split label is the folder path between data/processed/ and test.csv
    # e.g. data/processed/pen_based/pen_2/test.csv  ->  pen_based/pen_2
    split_label = str(
        Path(data_path).parent.relative_to(ROOT_DIR / "data" / "processed")
    )
    print(f"\nSplit Label: {split_label}")

    # Clean and Build Video Paths
    video_paths = df["source_video_path"]
    print("Cleaning Video Paths ...")
    print(f"Videos to clean: {len(video_paths)}")
    clean_paths = []
    for video_path in video_paths:

        cln_str = video_path.replace("\\", "/")
        cln_path = Path(cln_str)
        rel_path = Path(*cln_path.parts[-4:])  # Relies on file naming conventions...
        abs_path = SOURCE_VIDEOS_DIR / rel_path
        print(abs_path)

        # Do not add video if path does not exist
        if not abs_path.exists():
            continue
        clean_paths.append(abs_path)

    difference = len(clean_paths) - len(video_paths)
    print(f"{difference} videos removed.")
    output_dir = ROOT_DIR / "results" / "metadata" / model_type / split_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine how many CPU cores to use
    # Read how many slots Slurm actually assigned us. Fallback to 2 if not set.
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
    print(
        f"[INFO] Found {len(clean_paths)} videos. Processing using {num_workers} parallel workers..."
    )

    def worker_initializer():
        """Prevents each process from grabbing all threads for internal PyTorch math"""
        import os

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        try:
            import torch

            torch.set_num_threads(1)
        except ImportError:
            pass

    # Spin up the parallel execution pool
    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=worker_initializer
    ) as executor:
        # Submit all videos to the processing queue
        if model_type == "yolo":

            futures = [
                executor.submit(
                    run_detection,
                    video_path=v_path,
                    model_path=model_path,
                    output_dir=output_dir,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    min_duration=min_duration,
                    frame_skip=frame_skip,
                    target_class=target_class,
                    buffer=buffer,
                    show_video=False,
                )
                for v_path in clean_paths
            ]
        elif model_type == "seq_NMS":
            futures = [
                executor.submit(
                    run_seq_nms_detection,
                    video_path=v_path,
                    model_path=model_path,
                    output_dir=output_dir,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    min_duration=min_duration,
                    frame_skip=frame_skip,
                    target_class=target_class,
                    show_video=False,
                )
                for v_path in clean_paths
            ]

        # Monitor progress as they finish
        for i, future in enumerate(futures):
            try:
                future.result()
                print(f"[PROGRESS] Completed video {i+1}/{len(clean_paths)}")
            except Exception as e:
                print(f"[ERROR] Video {i+1} failed with error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Cross-sucking detection event linking using {YOLO} and basic logic or seq_NMS."
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="YOLO weights file (default: runs/detect/MooVision/cross-sucking/weights/best.pt)",
    )
    parser.add_argument(
        "--model_type",
        required=True,
        help="Model type to use ('yolo' or 'seq_NMS'",
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
        default=10,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=1,
        help="Number of seconds to wait without CS until ending an event.",
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
    run_testing(
        model_path=args.model_path,
        data_path=args.data_path,
        model_type=args.model_type,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        min_duration=args.min_duration,
        buffer=args.buffer,
        frame_skip=args.frame_skip,
        target_class=args.target_class,
    )
