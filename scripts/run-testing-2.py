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
    chunk: int = 0,
    chunk_pct: float = 0.1,
    overwrite: bool = False,
):
    """
    Run a model script on all source videos in testing set.

    Loads a test.csv at `data_path`, cleans video paths, and runs both
    the YOLO and seq-NMS models on each video. Supports chunked parallel
    execution via `chunk` and `chunk_pct` for use with SLURM array jobs.
    Skips videos that already have a results JSON by default — pass
    `--overwrite` to reprocess all videos regardless.

    Outputs both yolo and seq-NMS metadata.

    Parameters
    ----------
    model_path : str
        Path to YOLO weights file (e.g. best.pt).
    data_path : str
        Relative path to test.csv inside ROOT_DIR
        (e.g. data/processed/random/test.csv).
    conf_threshold : float
        Minimum YOLO detection confidence to keep a box.
    iou_threshold : float
        Minimum IoU overlap threshold.
    min_duration : float
        Minimum event duration in seconds.
    buffer : int
        Seconds to wait without detection before ending an event.
    frame_skip : int
        Process every Nth frame (1 = every frame).
    target_class : str
        Target class name for detection (default: 'cross-sucking').
    chunk : int
        Which chunk to process (0-indexed). -1 runs all videos.
        Default is 0 (first 10% chunk). Used by SLURM array jobs to
        parallelise across subsets of the video list.
    chunk_pct : float
        Fraction of total videos per chunk (default: 0.10 = 10%).
        Combined with `chunk` to determine which videos this job processes.
    overwrite : bool
        If False (default), skip videos that already have a results JSON.
        If True, reprocess and overwrite existing results. Use --overwrite
        when rerunning after parameter changes or pipeline fixes.

    Returns
    -------
    None
        This function reads to disk and does not return anything.

    Raises
    ------
    FileNotFoundError
        If the test.csv at `data_path` does not exist under ROOT_DIR.
        If the model weights file at `model_path` does not exist.
    ValueError
        If the test.csv is empty after loading.

    Examples
    --------
    run_testing(
        model_path="/scratch/st-nina-1/moovision/yolo_training_runs/split_1_model/weights/best.pt",
        data_path="data/processed/random/test.csv",
        conf_threshold=0.25,
        iou_threshold=0.5,
        min_duration=1.0,
        buffer=30,
        frame_skip=10,
        )

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

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Could not find {model_path}.")
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    idx = 0
    if chunk == -1:
        selected = unique_video_strings
        print(f"Running models on all {len(selected)} videos...")
    else:
        chunk_size = max(1, int(len(unique_video_strings) * chunk_pct))
        start = chunk * chunk_size
        selected = unique_video_strings[start : start + chunk_size]
        print(f"Running models on chunk {chunk} ({chunk_pct*100:.0f}%): videos {start}–{start + len(selected)} of {n_unique_paths}...")
    
    for n, video_str in enumerate(selected, 1):
        try:

            path = Path(video_str)
            json_path = output_dir / f"{path.stem}_results.json"
            if json_path.exists() and not overwrite:
                print(f"Skipping {path.stem} — already processed.")
                idx += 1
                continue

            print(f"Running {path.stem} {idx}/{len(selected)}")
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
        "--chunk",
        type=int,
        default=0,
        help="Which chunk to process (0-indexed). -1 runs all videos."
    )
    parser.add_argument(
        "--chunk_pct",
        type=float,
        default=0.10,
        help="Percentage of videos per chunk (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing results. If not set, skips already processed videos."
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
        chunk=args.chunk,
        chunk_pct=args.chunk_pct,
        overwrite=args.overwrite,
    )
