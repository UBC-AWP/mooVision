"""
Functions for extracting frames from cross-sucking videos
"""

import sys
from typing import List, Tuple, Set, Dict
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
import threading
import cv2

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.read_data.matching import parse_unlabelled_name
from scripts.preprocessing.utils import validate_file_paths
from typing import Callable


def generate_video_metadata(video_name: str) -> Tuple[Tuple[int, str | None], str]:
    """
    Extract lookup keys and file prefixes from an unlabelled video clip name.

    Parameters
    ----------
    video_name : str
        The raw base name of the video file (e.g., 'CS_0042_clip.mp4').

    Returns
    -------
    video_key : tuple of (int, str or None)
        A tracking tuple pair composed of (numeric_id, part_id) used to index
        and align extracted video frames with corresponding labels.
    file_prefix : str
        The standardized prefix format used for outputting dataset frame files,
        structured as `"{numeric_id:04}_{part_str}_frame_"`.

    Raises
    ------
    TypeError
        If `video_name` is not a string.
    ValueError
        If `video_name` is empty or composed entirely of whitespace.

    See Also
    --------
    parse_unlabelled_name : Helper function that utilizes regex patterns to split
                           the filename into its raw constituent components.
    """
    # Runtime Type and Value Validation
    if not isinstance(video_name, str):
        raise TypeError(
            f"Input 'video_name' must be a string, received {type(video_name).__name__}"
        )

    if not video_name.strip():
        raise ValueError("Input 'video_name' cannot be an empty string or whitespace.")

    numeric_id, part_id = parse_unlabelled_name(video_name)
    video_key = (int(numeric_id), part_id)

    part_str = f"part0{part_id}" if part_id else None
    file_prefix = f"{int(numeric_id):04}_{part_str}_frame_"

    return video_key, file_prefix


def process_single_video(
    video_path: Path,
    file_prefix: str,
    final_output_dir: Path,
    skip: int,
    executor: ThreadPoolExecutor,
    semaphore: threading.BoundedSemaphore,
    safe_write_func: Callable[[str, cv2.Mat], None],
) -> Set[int]:
    """
    Extract and process downsampled frames from a single video stream.

    Decodes a video file sequentially using OpenCV. Frames matching the downsampling
    interval stride are pushed to an asynchronous multi-threaded writer pool for
    high-speed disk writing. Non-matching frames are rapidly bypassed directly via
    native decoder buffer grabbing.

    Parameters
    ----------
    video_path : pathlib.Path
        The absolute or relative filesystem path to the source video file clip.
    file_prefix : str
        The standardized filename prefix used to identify individual dataset frame assets.
    final_output_dir : pathlib.Path
        The destination directory path where output image frames will be written.
    skip : int
        The frame downsampling stride interval (e.g., `5` processes every 5th frame).
    executor : concurrent.futures.ThreadPoolExecutor
        An active thread pool workspace utilized to handle unblocked background file writes.
    semaphore : threading.BoundedSemaphore
        A resource lock counter limiting the maximum length of the thread queue pool
        to protect memory buffers.
    safe_write_func : callable
        The specialized thread target worker function executing image writes and resource releases.

    Returns
    -------
    saved_frames : set of int
        A set containing the absolute sequential integers of all frames successfully
        decoded and dispatched to the background file writer.

    Raises
    ------
    TypeError
        If parameter input arguments fail baseline structural type boundaries.
    ValueError
        If `skip` is not a positive integer greater than zero.
    FileNotFoundError
        If the file located at `video_path` does not exist on disk.
    """
    # Runtime Type Checking
    if not isinstance(video_path, Path):
        raise TypeError(
            f"Argument 'video_path' must be a Path object, received {type(video_path).__name__}"
        )
    if not video_path.exists():
        raise FileNotFoundError(f"Target video stream not found at path: {video_path}")
    if not isinstance(file_prefix, str):
        raise TypeError(
            f"Argument 'file_prefix' must be a string, received {type(file_prefix).__name__}"
        )
    if not isinstance(final_output_dir, Path):
        raise TypeError(
            f"Argument 'final_output_dir' must be a Path object, received {type(final_output_dir).__name__}"
        )
    if not isinstance(skip, int):
        raise TypeError(
            f"Argument 'skip' must be an int, received {type(skip).__name__}"
        )
    if skip <= 0:
        raise ValueError(
            f"Argument 'skip' must be a positive integer greater than 0, received {skip}"
        )
    if not isinstance(executor, ThreadPoolExecutor):
        raise TypeError(
            f"Argument 'executor' must be a ThreadPoolExecutor instance, received {type(executor).__name__}"
        )
    if not isinstance(semaphore, threading.BoundedSemaphore):
        raise TypeError(
            f"Argument 'semaphore' must be a BoundedSemaphore instance, received {type(semaphore).__name__}"
        )
    if not callable(safe_write_func):
        raise TypeError(
            f"Argument 'safe_write_func' must be a callable target, received {type(safe_write_func).__name__}"
        )

    saved_frames = set()
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    frame_pointer = 0

    try:
        while cap.isOpened():
            semaphore.acquire()
            ret, frame = cap.read()
            if not ret:
                semaphore.release()
                break

            if frame_pointer % skip == 0:
                frame_path = final_output_dir / f"{file_prefix}{frame_pointer:06d}.jpg"
                executor.submit(safe_write_func, str(frame_path), frame)
                saved_frames.add(frame_pointer)

            if skip > 1:
                video_ended_early = False
                for _ in range(skip - 1):
                    if not cap.grab():
                        video_ended_early = True
                        break
                if video_ended_early:
                    semaphore.release()
                    break
                frame_pointer += skip
            else:
                frame_pointer += 1
    finally:
        cap.release()

    return saved_frames


def extract_frames(
    video_paths: List[str],
    videos_root: Path,
    working_dir: Path,
    split: str,
    skip: int,
    force: bool = False,
) -> Dict[Tuple[int, Tuple[str, None]], Set[int]]:
    """
    Orchestrate asynchronous frame extraction for a collection of video files.

    Validates relative video path targets against a root directory, configures
    an isolated node-local staging environment for the specified dataset split,
    and handles multi-threaded background disk-writing workers.

    Parameters
    ----------
    video_paths : list of str
        Relative file paths to individual target video files inside `videos_root`.
    videos_root : pathlib.Path
        The absolute root path directory containing the unlabelled source videos.
    working_dir : pathlib.Path
        The core active staging directory workspace (typically a local compute NVMe partition).
    split : str
        The specific machine learning dataset subset split; must be `'train'` or `'val'`.
    skip : int
        The integer stride sequence interval used for frame downsampling. Must be greater than 0.
    force : bool, default False
        If True, ignores existing matching output data targets and forces re-extraction.

    Returns
    -------
    saved_frames_registry : dict of {tuple(int, str or None) : set of int}
        A multi-video frame tracking map. Keys are unique video identifying tuples,
        and values are sets of integers containing successfully processed frame indices.

    Raises
    ------
    TypeError
        If input arguments fail structural type matching validations.
    ValueError
        If `split` is an unsupported keyword string or `skip` is non-positive.
    """
    # 1. Pipeline Input Guarding
    if not isinstance(video_paths, list):
        raise TypeError(
            f"Argument 'video_paths' must be a list, received {type(video_paths).__name__}"
        )
    if not isinstance(videos_root, Path) or not isinstance(working_dir, Path):
        raise TypeError(
            "Arguments 'videos_root' and 'working_dir' must be pathlib.Path objects."
        )
    if split not in ["train", "val"]:
        raise ValueError("Argument 'split' must be either 'train' or 'val'.")
    if not isinstance(skip, int):
        raise TypeError(
            f"Argument 'skip' must be an int, received {type(skip).__name__}"
        )
    if skip <= 0:
        raise ValueError(
            f"Argument 'skip' must be a positive integer greater than 0, received {skip}"
        )
    if not isinstance(force, bool):
        raise TypeError(
            f"Argument 'force' must be a boolean value, received {type(force).__name__}"
        )

    print("\n\n--- Extracting Video Frames---\n")

    final_output_dir = working_dir / "images" / split
    if Path(final_output_dir).exists() and not force:
        print(f"\nFiles already extracted at {Path(__file__) / Path(final_output_dir)}")
        return

    final_output_dir.mkdir(parents=True, exist_ok=True)

    validated_video_paths = validate_file_paths(video_paths, videos_root)

    # Initialize ThreadPoolExecutor
    MAX_QUEUE_SIZE = 40
    semaphore = threading.BoundedSemaphore(MAX_QUEUE_SIZE)
    n_videos = len(validated_video_paths)
    saved_frames_registry = {}

    # Helper function to release the semaphore slot once disk write is complete
    def safe_write(frame_path, frame_data):
        try:
            cv2.imwrite(frame_path, frame_data)
        finally:
            semaphore.release()  # Opens up a slot for the main loop to read again

    with ThreadPoolExecutor(max_workers=4) as executor:
        for n, video_path in enumerate(validated_video_paths, 1):

            video_key, file_prefix = generate_video_metadata(video_name=video_path.name)
            # Print working video...
            print(f"Extracting frames from video ({n}/{n_videos}): {video_path.name}")

            video_saved_frames = process_single_video(
                video_path=video_path,
                file_prefix=file_prefix,
                final_output_dir=final_output_dir,
                skip=skip,
                executor=executor,
                semaphore=semaphore,
                safe_write_func=safe_write,
            )

            saved_frames_registry[video_key] = video_saved_frames

    print("\nExtraction Complete. Waiting for final thread queue to clear...\n")
    executor.shutdown(wait=True)
    print(f"\nFrames successfully saved to disk at {final_output_dir}\n")

    return saved_frames_registry
