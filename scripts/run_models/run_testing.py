"""
Orchestrator script for running models on testing data (raw source videos)

NOTE: Need to add functionality for adding other arguments too! seq_NMS and yolo have different arguments, should be call the functions instead?
"""

from pathlib import Path
import sys
import argparse
import pandas as pd
from ultralytics import YOLO
import importlib

sys.path.append(str(Path(__file__).parent))
from yolo.yolo import run_models
from seq_NMS.seq_NMS import (
    run_seq_nms_detection,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MIN_DURATION,
    TARGET_CLASS_NAME,
)

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ROOT_DIR, SOURCE_VIDEOS_DIR, LOCAL_DIR

DEFAULT_FRAME_SKIP = 10

def load_test_csv(data_path: str) -> pd.DataFrame:
    """
    Load and validate the test.csv for a given split.
 
    Parameters
    ----------
    data_path : str
        Relative path to test.csv inside ROOT_DIR.
 
    Returns
    -------
    pd.DataFrame
        The loaded test set.
 
    Raises
    ------
    FileNotFoundError
        If the test.csv at `data_path` does not exist under ROOT_DIR.
    ValueError
        If the test.csv is empty after loading.
    """
    full_path = ROOT_DIR / data_path
    if not full_path.exists():
        raise FileNotFoundError(f"Could not find file: {full_path}")
 
    df = pd.read_csv(full_path, index_col=0)
    if df.empty:
        raise ValueError("df is empty.")
 
    return df

def get_split_label(data_path: str) -> str:
    """
    Derive the split label from a test.csv path.
 
    The split label is the folder path between data/processed/ and
    test.csv, e.g. data/processed/pen_based/pen_2/test.csv -> pen_based/pen_2.
 
    Parameters
    ----------
    data_path : str
        Relative path to test.csv inside ROOT_DIR.
 
    Returns
    -------
    str
        The split label.
    """
    # Ensure the csv path is an absolute Path object
    csv_path = Path(data_path)
    if not csv_path.is_absolute():
        abs_csv_path = (ROOT_DIR / csv_path).resolve()
    else:
        abs_csv_path = csv_path.resolve()
    
    # Build the absolute path to your data/processed directory
    abs_processed_dir = (ROOT_DIR / "data" / "processed").resolve()
    
    # Safely find the relative difference between the two absolute paths
    return str(abs_csv_path.parent.relative_to(abs_processed_dir))

def clean_video_paths(video_paths: pd.Series) -> list[str]:
    """
    Resolve raw video paths from test.csv into absolute, existing,
    de-duplicated source video paths.
 
    Parameters
    ----------
    video_paths : pd.Series
        Raw `source_video_path` column from test.csv.
 
    Returns
    -------
    list[str]
        Sorted, unique, absolute video path strings that exist on disk.
    """
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
 
    difference = len(unique_video_strings) - len(video_paths)
    print(f"{difference} videos removed.")
 
    return unique_video_strings
 
 
def select_chunk(unique_video_strings: list[str], chunk: int, chunk_pct: float) -> list[str]:
    """
    Select the slice of videos this chunk/array task is responsible for.
 
    Parameters
    ----------
    unique_video_strings : list[str]
        Full list of unique video paths for the split.
    chunk : int
        Which chunk to process (0-indexed).
    chunk_pct : float
        Fraction of total videos per chunk (e.g. 0.10 = 10%).
 
    Returns
    -------
    list[str]
        The subset of video paths assigned to this chunk.
 
    Raises
    ------
    ValueError
        If `chunk` is out of range for the computed number of chunks,
        if `chunk` starts beyond the available videos, or if the
        resulting selection is empty.
    """
    # set number of chunks and number of unique videos
    n = len(unique_video_strings)
    n_chunks = round(1 / chunk_pct)

    # throw error when chunk is out of bounds
    if chunk < 0 or chunk >= n_chunks:
        raise ValueError(f"chunk {chunk} is out of range for {n_chunks} chunks (0–{n_chunks-1})")

    start = chunk * n // n_chunks
    end = (chunk + 1) * n // n_chunks

    # throw error when it starts more than the number of existing videos
    if start >= n:
        raise ValueError(f"chunk {chunk} starts at index {start} but only {n} videos exist")

    selected = unique_video_strings[start:end]

    # throw error when no video strings are selected
    if not selected:
        raise ValueError(f"chunk {chunk} is empty — check chunk_pct and total video count")
 
    return selected
 
 
def load_yolo_model(model_path: str) -> tuple[YOLO, str]:
    """
    Resolve a model weights path and load it as a YOLO model.
 
    Parameters
    ----------
    model_path : str
        Path to YOLO weights file (e.g. best.pt). May be relative to
        ROOT_DIR or absolute.
 
    Returns
    -------
    tuple[YOLO, str]
        The loaded model, and the resolved absolute path string used to load it.
 
    Raises
    ------
    FileNotFoundError
        If the model weights file does not exist.
    """
    resolved_path = Path(model_path)
    if not resolved_path.is_absolute():
        resolved_path = ROOT_DIR / resolved_path
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Could not find model at {resolved_path} (checked relative to ROOT_DIR if not absolute)."
        )
 
    resolved_path = str(resolved_path)
    print(f"[INFO] Loading model: {resolved_path}")
    model = YOLO(resolved_path)
 
    return model, resolved_path
 
 
def check_processed(video_path: Path, output_dir: Path) -> bool:
    """
    Check whether a video already has both yolo and seq-nms results.
 
    Parameters
    ----------
    video_path : Path
        Path to the source video.
    output_dir : Path
        Split-level output directory containing yolo/ and seq-nms/ subfolders.
 
    Returns
    -------
    bool
        True if both results JSON files already exist.
    """
    yolo_json = output_dir / "yolo" / f"{video_path.stem}_results.json"
    seq_nms_json = output_dir / "seq-nms" / f"{video_path.stem}_results.json"
    return yolo_json.exists() and seq_nms_json.exists()
 

def run_single_video(
    video_str: str,
    model: YOLO,
    model_path: str,
    output_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
    min_duration: float,
    buffer: int,
    frame_skip: int,
    target_class: str,
    overwrite: bool,
) -> str:
    """
    Run yolo + seq-nms models on a single video, skipping if already
    processed (unless `overwrite` is set).
 
    Parameters
    ----------
    video_str : str
        Absolute path to the source video, as a string.
    model : YOLO
        Pre-loaded YOLO model instance.
    model_path : str
        Resolved path string to the model weights (passed through to run_models).
    output_dir : Path
        Split-level output directory.
    conf_threshold : float
        Minimum YOLO detection confidence to keep a box.
    iou_threshold : float
        Minimum IoU overlap threshold.
    min_duration : float
        Minimum event duration in seconds.
    buffer : int
        Seconds to wait without detection before ending an event.
    frame_skip : int
        Process every Nth frame.
    target_class : str
        Target class name for detection.
    overwrite : bool
        If False, skip videos that already have results. If True, reprocess.
 
    Returns
    -------
    str
        One of "skipped", "processed", or "failed".
    """
    path = Path(video_str)
 
    if check_processed(path, output_dir) and not overwrite:
        print(f"Skipping {path.stem} — already processed.")
        return "skipped"
 
    try:
        run_models(
            model=model,
            model_path=model_path,
            video_path=video_str,
            output_dir=output_dir,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            min_duration=min_duration,
            buffer=buffer,
            frame_skip=frame_skip,
            target_class=target_class,
            show_video=False,
        )
        return "processed"
    except Exception as e:
        print(f"Skipped {path.stem}: {e!r}")
        return "failed"
 

def force_onedrive_download(file_path):
    """Force file download before sending to model."""
    # We must actually attempt to read a single byte to force macOS to download files
    try:
        with open(file_path, "rb") as f:
            f.read(1)  # Reads just the first byte, forcing the download
        print(f"Successfully synced: {(file_path.name)}")
    except Exception as e:
        print(f"Failed to force download: {e}")


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
        Which chunk to process (0-indexed).
        Default is 0 (first 10% chunk). Used by SLURM array jobs to
        parallelise across subsets of the video list.
        Must be in range [0, n_chunks).
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
    df = load_test_csv(data_path)

    # the split label is the folder path between data/processed/ and test.csv
    # e.g. data/processed/pen_based/pen_2/test.csv  ->  pen_based/pen_2
    split_label = get_split_label(data_path)
    print(f"\nSplit Label: {split_label}")

    # Clean and Build Video Paths
    video_paths = df["source_video_path"]
    unique_video_strings = clean_video_paths(video_paths)

    # Select chunks
    selected = select_chunk(unique_video_strings, chunk, chunk_pct)

    # Load model
    model, resolved_model_path = load_yolo_model(model_path)

    # Prepare the output directory
    output_dir = ROOT_DIR / "results" / "metadata" / split_label
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    for i, video_str in enumerate(selected):
        print(f"Running {Path(video_str).stem} {i}/{len(selected)}")
        status = run_single_video(
            video_str=video_str,
            model=model,
            model_path=resolved_model_path,
            output_dir=output_dir,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            min_duration=min_duration,
            buffer=buffer,
            frame_skip=frame_skip,
            target_class=target_class,
            overwrite=overwrite,
        )
        counts[status] += 1
 
    print(
        f"\nDone. processed={counts['processed']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )


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
        "--chunk", type=int, default=0, help="Which chunk to process (0-indexed)."
    )
    parser.add_argument(
        "--chunk_pct",
        type=float,
        default=0.10,
        help="Percentage of videos per chunk (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing results. If not set, skips already processed videos.",
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
