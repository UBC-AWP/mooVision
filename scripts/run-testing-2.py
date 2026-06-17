"""
Orchestrator script for running models on testing data (raw source videos)

NOTE: Need to add functionality for adding other arguments too! seq_NMS and yolo have different arguments, should be call the functions instead?
"""

from pathlib import Path
import sys
import argparse
import pandas as pd
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from config import ROOT_DIR, SOURCE_VIDEOS_DIR, LOCAL_DIR
from scripts.models.yolo.yolo import run_models
from scripts.models.seq_NMS.seq_NMS import (
    run_seq_nms_detection,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_FRAME_SKIP,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MIN_DURATION,
    TARGET_CLASS_NAME,
)


def run_testing(
    model_path: str,
    data_path: str,
    conf_threshold: float,
    iou_threshold: float,
    min_duration: float,
    buffer: int,
    frame_skip: int,
    target_class: str = "cross-sucking",
    offset: int = 0,
):
    """
    Run a model script on all source videos in testing set.

    Loads in a test.csv at `data_path` and runs the model
    specified at `model` on all source videos listed if they exist
    in the data directory. Specifically, this function loops through each
    video path in the data frame, and passes these paths the the script
    in models/ specified as `model`. This script will run the model on
    said video and output metadata to the data directory.

    Outputs both yolo and seq-NMS metadata.

    Parameters
    ----------
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
    # Read in Data
    if not (ROOT_DIR / data_path).exists():
        raise FileNotFoundError(f"Could not find file: {ROOT_DIR / data_path}")
    df = pd.read_csv(ROOT_DIR / data_path, index_col=0)

    if df.empty:
        raise ValueError("df is empty.")

    # the split label is the folder path between data/processed/ and test.csv
    # e.g. data/processed/pen_based/pen_2/test.csv  ->  pen_based/pen_2
    split_label = str(Path(data_path).parent.relative_to("data/processed"))
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

        # Do not add video if path does not exist
        try:
            if not abs_path.exists():
                continue
            clean_paths.append(abs_path)
        except Exception as e:
            print(f"{e}")

    unique_video_strings = sorted(list(set(str(p) for p in clean_paths)))
    n_unique_paths = len(unique_video_strings)
    n_video_paths = len(video_paths)

    difference = n_unique_paths - n_video_paths
    print(f"{difference} videos removed.")

    output_dir = ROOT_DIR / "results" / "metadata" / split_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # # Cut to 20 videos to presentation results!
    # if n_unique_paths > 20:
    #     unique_video_strings_short = unique_video_strings[:30]
    # else:
    #     unique_video_strings_short = unique_video_strings

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Could not find {model_path}.")
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Running models on 20 unique video paths... ")
    idx = 0
    for n, video_str in enumerate(unique_video_strings[offset : offset + 20], 1):
        try:

            path = Path(video_str)
            print(f"Running {path.stem} {idx}/{20}")
            run_models(
                model=model,
                model_path=model_path,
                video_path=str(path),
                output_dir=output_dir,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                min_duration=min_duration,
                buffer=buffer,
                frame_skip=frame_skip,
                target_class=target_class,
                show_video=False,
            )
            idx += 1

        except Exception:
            print(f"Skipped {path.stem}")
            continue


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
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU overlap threshold (default: 0)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help="YOLO detection confidence threshold (default: 0)",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help="Minimum event duration in seconds (default: 0)",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=DEFAULT_FRAME_SKIP,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=30,
        help="Number of seconds to wait without CS until ending an event.",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default="cross-sucking",
        help="Target class for detection (default: cross-sucking)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index into the video list (for parallel chunking)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_testing(
        model_path=args.model_path,
        data_path=args.data_path,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        min_duration=args.min_duration,
        buffer=args.buffer,
        frame_skip=args.frame_skip,
        target_class=args.target_class,
    )
