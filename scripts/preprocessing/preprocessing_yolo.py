"""
Preprocessing functionality for creating training sets to fine-tune YOLO
object detection models.

WIP NOTES.
NOTE 1: Should change to save output to onedrive rather than locally (later on). Creates Too many files for local work.

NOTE 2: Look into sampling frames at x/second rather than a defined skip amount.

NOTE 3: Examples are not finished and need to be properly updated.
"""

from typing import List, Tuple, Dict, Any, Set
from pathlib import Path
import sys
import os

import shutil
import argparse
import tarfile
import zipfile
import yaml

import concurrent.futures
import threading
import cv2
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data_reading.matching import parse_labelled_name, parse_unlabelled_name
from config import UNLABELLED_CLIPS_DIR, LABELLED_CLIPS_DIR, ROOT_DIR


def validate_file_paths(
    label_paths: List[str],
    labels_root: Path,
) -> List[Path]:
    """
    Clean file paths to label_paths to standardize path structure
    and validate paths exist in the labelled root directory `labels_root`.

    Parameters
    ----------
    labels_path : List[str]
        List of relative paths within the labelled clips directory to
        zipped folders containing bounding box annotations for
        cross-sucking events.

    labels_root : pathlib.Path
        Path to the labelled clips directory containing zipped folders
        with bounding box annotations for cross-sucking events.

    Returns
    -------
    validated_paths : List[pathlib.Path]
        List of Paths to zipped folders with bounding box labels

    Raises
    ------
    TypeError
        If label_paths is not a list or labels_root is not a Path object.
    FileNotFoundError
        If a constructed path does not exist.
    """
    # Type Validation
    if not isinstance(label_paths, list):
        raise TypeError(
            f"Expected 'label_paths' to be a list, got {type(label_paths).__name__}"
        )

    if not isinstance(labels_root, Path):
        raise TypeError(
            f"Expected 'labels_root' to be a Path object, got {type(labels_root).__name__}"
        )
    # Input Pre-Validation (Fails fast before touching data)
    validated_paths = []
    for raw_path in label_paths:
        if not isinstance(raw_path, str):
            raise TypeError(f"ERROR: {raw_path} is not a string path.")

        # Standardize path string
        clean_path = labels_root / raw_path.replace("\\", "/")
        if not clean_path.exists():
            raise FileNotFoundError(
                f"Attempting clean file paths, cleaned label file not found: {clean_path}"
            )

        validated_paths.append(clean_path)

    return validated_paths


def generate_video_metadata(zip_name: str) -> tuple[tuple[int, str | None], str]:
    """
    Extract lookup keys and file prefixes from a CVAT export zip filename.

    Parameters
    ----------
    zip_name : str
        The base name of the zip file archive.

    Returns
    -------
    video_key : tuple of (int, str or None)
        A tracking tuple pair composed of (numeric_id, part_id) used to index
        and align annotations with processed frames.
    file_prefix : str
        The standardized prefix format used for outputting dataset files,
        formatted as `"{numeric_id:04}_{part_id_str}_"`.

    Raises
    ------
    TypeError
        If `zip_name` is not passed as a string.
    ValueError
        If `zip_name` is an empty string.

    See Also
    --------
    parse_labelled_name : Extracted internal helper that parses components via regex.
    """
    # Type Validation
    if not isinstance(zip_name, str):
        raise TypeError(
            f"Input 'zip_name' must be a string, received {type(zip_name).__name__}"
        )

    if not zip_name.strip():
        raise ValueError("Input 'zip_name' cannot be an empty string or whitespace.")

    numeric_id, part_id = parse_labelled_name(zip_name)
    video_key = (int(numeric_id), part_id)

    part_id_str = f"part0{part_id}" if part_id else None
    file_prefix = f"{int(numeric_id):04}_{part_id_str}_"

    return video_key, file_prefix


def extract_single_zip(
    zip_path: Path,
    frame_registry: dict,
    skip: int,
    file_prefix: str,
    video_key: tuple[int, str | None],
) -> tuple[dict[str, bytes], set[int]]:
    """
    Extract downsampled annotation files matching a valid frame registry from one zip.

    Parses a single zipped CVAT export file, extracting text annotation data for
    frames that match downsampling strides and exist within the provided tracking
    registry.

    Parameters
    ----------
    zip_path : pathlib.Path
        The filesystem path leading to the target zip file archive.
    frame_registry : dict
        A multi-video tracking lookup dictionary. Keys are tuples matching
        `(numeric_id, part_id)` and values are sets of integers representing
        successfully extracted video frames.
    skip : int
        The downsampling stride value (e.g., `5` extracts every 5th frame).
    file_prefix : str
        The standardized name prefix string generated for the specific video file clip.
    video_key : tuple of (int, str or None)
        A pair composed of `(numeric_id, part_id)` used as a unique identifier
        for logging and checking against registries.

    Returns
    -------
    label_batch : dict of {str : bytes}
        A mapping of target destination filenames to their raw, unwritten text-file
        binary data bytes.
    saved_frames : set of int
        A set tracking the sequential frame numbers successfully extracted and batched
        from this specific archive.

    Raises
    ------
    TypeError
        If `zip_path` is not a Path object, `frame_registry` is not a dict,
        `skip` is not an integer, `file_prefix` is not a string, or `video_key`
        is not a tuple.
    ValueError
        If `skip` is less than or equal to zero, or if a parsed file inside the
        zip archive contains an invalid frame index structure.
    FileNotFoundError
        If the file at `zip_path` does not exist on disk.
    """
    # Runtime Type Checking
    if not isinstance(zip_path, Path):
        raise TypeError(
            f"Argument 'zip_path' must be a Path object, received {type(zip_path).__name__}"
        )
    if not zip_path.exists():
        raise FileNotFoundError(f"Target archive zip not found at path: {zip_path}")
    if frame_registry is not None and not isinstance(frame_registry, dict):
        raise TypeError(
            f"Argument 'frame_registry' must be a dict, received {type(frame_registry).__name__}"
        )
    if not isinstance(skip, int):
        raise TypeError(
            f"Argument 'skip' must be an int, received {type(skip).__name__}"
        )
    if skip <= 0:
        raise ValueError(
            f"Argument 'skip' must be a positive integer greater than 0, received {skip}"
        )
    if not isinstance(file_prefix, str):
        raise TypeError(
            f"Argument 'file_prefix' must be a string, received {type(file_prefix).__name__}"
        )
    if not isinstance(video_key, tuple):
        raise TypeError(
            f"Argument 'video_key' must be a tuple, received {type(video_key).__name__}"
        )

    label_batch = {}
    saved_frames = set()
    target_folder = "obj_train_data"

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for zinfo in zip_ref.infolist():
            filename = zinfo.filename

            if filename.startswith(target_folder) and filename.endswith(".txt"):
                pure_name = filename.split("/")[-1]

                try:
                    frame_num = int(pure_name[6:12])
                except ValueError as e:
                    raise ValueError(
                        f"CRITICAL: Structural naming mismatch inside archive {zip_path.name}. "
                        f"Expected 'frame_XXXXXX.txt' but got '{pure_name}'."
                    ) from e

                if frame_num % skip == 0:
                    if frame_registry is not None:
                        if (
                            video_key not in frame_registry
                            or frame_num not in frame_registry[video_key]
                        ):
                            continue  # Drop safely if the image frame isn't present

                    new_filename = f"{file_prefix}{pure_name}"
                    label_batch[new_filename] = zip_ref.read(zinfo)
                    saved_frames.add(frame_num)

    return label_batch, saved_frames


def parse_zip_annotations(
    validated_paths: List[Path],
    frame_registry: Dict[Tuple[int, Any], Set[int]],
    skip: int,
) -> Tuple[Dict[str, bytes], Dict[Tuple[int, Any], Set[int]]]:
    """
    Parse zipped CVAT annotation exports into in-memory byte batches and matching registries.

    Iterates through a list of validated paths to zip files containing bounding box
    annotations. Extracts text streams for downsampled frames that possess a valid
    corresponding entry in the provided frame registry.

    Parameters
    ----------
    validated_paths : list of pathlib.Path
        A list of verified absolute paths to zipped CVAT data folders.
    frame_registry : dict
        A lookup dictionary tracking extracted video frames. Keys are tuples containing
        the numeric ID and part ID (`(int, str/None)`), and values are sets of integers
        representing successfully saved frame numbers.
    skip : int
        The sequence interval step size used for downsampling frame data (e.g.,
        passing 5 filters and extracts every 5th sequential frame annotation).

    Returns
    -------
    label_batch : dict of {str : bytes}
        An in-memory batch mapping target destination filenames (formatted for YOLO)
        to their raw text binary streams extracted directly from the zip file archives.
    saved_labels_registry : dict of {tuple(int, str or None) : set of int}
        A mapping of video keys to sets of frame indexes that were successfully
        processed and queued for extraction.

    Raises
    ------
    ValueError
        If a file within the target archive violates standard naming conventions
        preventing safe extraction of its sequential frame index.

    See Also
    --------
    extract_labels : Parent orchestrator executing actual I/O batch writes.

    Notes
    -----
    This function reads zipped files directly into RAM to minimize redundant storage
    overhead on high-performance compute node playgrounds (like Sockeye's local NVMe
    burst workspaces). The returned byte arrays should be written using binary file-mode
    handlers (`"wb"`).
    """
    # Loop over zip file paths (CVAT Outputs)
    print("\n--- Extracting Annotation Labels ---\n")
    global_label_batch = {}
    saved_labels_registry = {}
    n_files = len(validated_paths)
    for n, input_path in enumerate(validated_paths, start=1):
        print(f"Extracting files from {input_path.name} ({n}/{n_files} )...")

        video_key, file_prefix = generate_video_metadata(zip_name=input_path.name)
        saved_labels_registry[video_key] = set()

        label_batch, saved_frames = extract_single_zip(
            zip_path=input_path,
            frame_registry=frame_registry,
            skip=skip,
            file_prefix=file_prefix,
            video_key=video_key,
        )

        global_label_batch.update(label_batch)
        saved_labels_registry[video_key] = saved_frames

        print(f"Labels Saved: {len(saved_labels_registry[video_key])}")

    return label_batch, saved_labels_registry


def save_label_batch(
    label_batch: dict,
    final_output_dir: Path,
) -> None:
    """
    Commit a batch of in-memory annotation byte streams to disk.

    Iterates over a dictionary of pre-filtered filenames and raw text byte
    arrays, writing them directly to the specified local workspace directory
    (e.g., node-local NVMe staging or local repository folders). Prints
    periodic progress updates for large batch operations.

    Parameters
    ----------
    label_batch : dict of {str : bytes}
        A mapping of target destination filenames (e.g., `'0001_part01_frame_000000.txt'`)
        to their raw annotation contents stored as unwritten text bytes.
    final_output_dir : pathlib.Path
        The destination directory path where the text files will be saved.

    Returns
    -------
    None
        This function writes directly to the filesystem and does not return a value.

    Raises
    ------
    TypeError
        If `label_batch` is not a dictionary or if `final_output_dir` is not a
        pathlib.Path object.
    """
    # Runtime Type Checking
    if not isinstance(label_batch, dict):
        raise TypeError(
            f"Argument 'label_batch' must be a dict, received {type(label_batch).__name__}"
        )
    if not isinstance(final_output_dir, Path):
        raise TypeError(
            f"Argument 'final_output_dir' must be a Path object, received {type(final_output_dir).__name__}"
        )
    print("\nExtraction complete.\n")
    if label_batch:
        idx = 0
        batch_len = len(label_batch)
        print(f"--- Executing batch-write for {batch_len} annotation labels ---\n")
        # Create an isolated, hyper-fast playground inside the node's local memory
        for filename, text_bytes in label_batch.items():
            idx += 1
            if idx % 1000 == 0:
                print(f"Writing label: ({idx}/{batch_len})")
            file_path = final_output_dir / filename
            with open(file_path, "wb") as f:
                f.write(text_bytes)

        print(f"\nAll labels saved at: {final_output_dir}")


def extract_labels(
    label_paths: List[str],
    working_dir: Path,
    labels_root: Path,
    frame_registry: dict,
    split: str,
    skip: int,
    force: bool = False,
) -> dict:
    """
    Extract labels from annotated data folders and match them to frames
    extracted from corresponding videos.

    Extracts bounding box labels from .zip files (CVAT Output folders),
    and saves files to either a train, or val folder in the specified
    working directory directory.

    If force = True, and the output directory structure already exists,
    this will delete the existing folders and rebuild the data from scratch.

    `extract_labels` outputs data into a labels/ folder within the output directory.
    This is meant to mimic the data format required for fine-tuning YOLO models.

    This function assumes that bounding box labels are saved as `.txt` files in
    `obj_train_data` folders within .zip files, and that `labels_path` lists
    relative paths to these .zip files within the `labels_root` directory.

    Parameters
    ----------
    labels_path : List[str]
        List of relative paths within the labelled clips directory to
        zipped folders containing bounding box annotations for
        cross-sucking events.
    working_dir : Path
        Path to output directory. Points to node's local temp workspace when
        running on Sockeye.
    labels_root : Path
        Path to the labelled clips directory containing zipped folders
        with bounding box annotations for cross-sucking events.
    frame_registry : dict
        A dictionary folding sets of frames that have been saved for each
        video. Uses the unique numeric ID and part ID of each video as keys,
        to ensures all labels have a matching frame.
    split : str
        One of `train`, `val`. Dictates which split folder, train/ or val/, the
        labels should be extracted to.
    skip : int
        Controls teh downsampling density; number of frames to skip. `skip=5`
        will read every 5th frame.
    force : bool
        If True, overwritres the existing data. Defaults to False.

    Returns
    -------
    dict[set] :
        This function returns a dictionary of sets of saved labels for each
        video in videos. This is used via set subtraction to remove any video
        frames which do not have an associated label.

    Raises
    ------
    ValueError
        If inputs are empty, or if labels_path is an empty list.
        If split is not one of `train` or `val`.
    TypeError
        If inputs to not match the specififed types.
    FileNotFoundError
        If any relative paths do not lead to files for labelled data.
        If `labels_root` does not exist.
        If the target folder does not exist within the .zip files.
        If there are no `.txt` files found in the target folder.

    Notes
    -----
    `extract_labels` is meant to be used in conjunction with `extract_frames`
    to build datasets for training YOLO models. YOLO requires frames and labels
    are split into train/val sets with corresponding files names.

    """
    if split not in ["train", "val"]:
        raise ValueError("split must be either 'train' or 'val'.")

    # Add subdirectories to output directory
    final_output_dir = working_dir / "labels" / split

    if final_output_dir.exists() and not force:
        print(f"Files already extracted at {final_output_dir}. Skipping computation.")
        return
    # Build output directory
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Clean File Paths
    validated_paths = validate_file_paths(label_paths, labels_root)

    label_batch, saved_labels_registry = parse_zip_annotations(
        validated_paths,
        frame_registry,
        skip,
    )

    save_label_batch(label_batch, final_output_dir)

    return saved_labels_registry


def extract_frames(
    video_paths: List[str],
    videos_root: Path,
    working_dir: Path,
    split: str,
    skip: int,
    force: bool = False,
) -> None:
    """
    Extract frames from all videos in list.

    Takes in a list of realtive paths to video files inside the videos_root directory
    and extracts the frames from each video as .jpg files into train or val folders in
    output_dir. Split is one of train or val and dictates which folder the images are
    saved under. Skip dictates how many frames to skip (i.e. skip = 5 would mean
    extract every fifth frame).

    `extract_frames` outputs frames to an images/train/ or images/val/ folder within
    the output directory depending on the split argument.

    Parameters
    ----------
    videos : List[str]
        A list of relative paths to video files inside the videos_root directory.
    videos_root : Path
        Path to root directory containing videos.
    working_dir : Path
        Path to output directory. Points to node's local temp workspace on Sockeye.
    split : str
        One of `train`, `val`. Dictates which split folder, train/ or val/, the
        labels should be extracted to.
    skip : int
        Number of frames to skip.
    force : bool
        If True, overwritres the existing data. Defaults to False.


    Returns
    -------
    dict[set] :
        This function returns a dictionary of sets of saved frames for each
        video in videos. This is passed to extract_labels to act as a dynamic
        shield forensuring frame and label matches.


    Raises
    ------
    ValueError
        If inputs are empty, or if videos is an empty list.
        If split is not one of `train` or `val`.
        If a video cannot be read, or is corrupted.
        If skip is greater than the number of frames
    TypeError
        If inputs to not match the specififed types.
    FileNotFoundError
        If any relative paths in videos do not lead to files for labelled data.
        If `videos_root` does not exist.

    Notes
    -----
    `extract_frames` is meant to be used in conjunction with `extract_labels`
    to build datasets for training YOLO models. YOLO requires datasets in the
    format listed below. Frames and labels are split into train/val sets,
    and each frame has a corresponding label with the same name and .txt
    extension. This implies that there needs to be equal numbers of frames
    and labels, and that labels need to match to correct frames.

    """
    print("\n\n--- Extracting Video Frames---\n")

    # Output dir
    final_output_dir = working_dir / "images" / split

    # Rewrite files on force
    if Path(final_output_dir).exists() and not force:
        print(f"\nFiles already extracted at {Path(__file__) / Path(final_output_dir)}")
        return

    # Extraction directory
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Track total files
    n_videos = len(video_paths)

    # Initialize ThreadPoolExecutor
    MAX_QUEUE_SIZE = 40
    semaphore = threading.BoundedSemaphore(MAX_QUEUE_SIZE)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # Helper function to release the semaphore slot once disk write is complete
    def safe_write(frame_path, frame_data):
        try:
            cv2.imwrite(frame_path, frame_data)
        finally:
            semaphore.release()  # Opens up a slot for the main loop to read again

    n_frames = 0
    saved_frames_registry = {}
    for n, video_file in enumerate(video_paths, 1):

        # Standardize Path to Posix Standard
        video_file = videos_root / video_file.replace("\\", "/")

        # Get numeric id and part id of video clip, create lookup key
        numeric_id, part_id = parse_unlabelled_name(str(video_file.name))
        video_key = (int(numeric_id), part_id)

        # Initialize the inner set for this specific video file
        saved_frames_registry[video_key] = set()

        # File Naming
        part_str = f"part0{part_id}" if part_id else None
        file_prefix = f"{int(numeric_id):04}_{part_str}_frame_"

        # Video capture
        cap = cv2.VideoCapture(str(video_file))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        # Print working video...
        print(f"Extracting frames from video ({n}/{n_videos})")

        frame_pointer = 0

        while cap.isOpened():

            semaphore.acquire()
            # Decode frame
            ret, frame = cap.read()
            if not ret:  # Break on video corruption
                semaphore.release()
                break

            # Match your label filtering math exactly
            if frame_pointer % skip == 0:
                frame_path = final_output_dir / f"{file_prefix}{frame_pointer:06d}.jpg"
                executor.submit(safe_write, str(frame_path), frame)

                saved_frames_registry[video_key].add(frame_pointer)
                n_frames += 1

            # Fast frame skipping without running the above
            # 2. Handle fast-forwarding frame drops safely
            if skip > 1:
                video_ended_early = False
                for _ in range(skip - 1):
                    if not cap.grab():
                        video_ended_early = True
                        break

                if video_ended_early:
                    break  # Completely break outer loop if video file terminates mid-stream

                # Advance pointer by the exact math stride
                frame_pointer += skip
            else:
                # If skip is 1, we just advance step-by-step
                frame_pointer += 1

        # release video
        cap.release()

    print("\nExtraction Complete.")

    # Wait for all background thread writes to finish inside /tmp
    print("Waiting for final thread queue to clear...")
    executor.shutdown(wait=True)
    print(f"\n{n_frames} frames successfully saved to disk at {final_output_dir}\n")
    print()

    return saved_frames_registry


def create_yaml(
    working_dir: str,
    class_names: list[str] = ["cross-sucking"],
    output_filename: str = "dataset.yaml",
) -> None:
    """
    Write a YOLO dataset YAML configuration file.

    Parameters
    ----------
    dataset_path : str
        Absolute path to the dataset root directory.
    class_names : list[str]
        Ordered list of class labels matching the IDs in label files.
    output_path : str, optional
        Destination path for the YAML file.

    Returns
    -------
    None
        This function reads straight to disk and does not return anything.
    """
    config = {
        "path": "./dataset",
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    print("\n\n=========================")
    print("DATASET.YAML")
    out = working_dir + "/" + output_filename
    print(out)
    print(config)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print("=========================\n\n")


# UPDATE TO CLEAN AND TAKE IN ARGUMENTS
def run_yolo_preprocessing(
    train_path: str,
    val_path: str,
    output_path: str,
    skip: int,
    force: bool = False,
) -> None:
    """
    Execute the end-to-end YOLO preprocessing pipeline on a tracking index.

    Reads a processed dataset file, splits the data into training and
    validation subsets via a randomized train/test split, and systematically
    calls `extract_frames` and `extract_labels` to generate a YOLO-compliant
    object detection directory structure.

    Parameters
    ----------
    train_path : str
        Path to the source CSV file containing training video and label mappings.
    val_path : str
        Path to the source CSV file containing validation video and label mappings.
    output_path : str
        Relative directory inside ROOT_DIR path where the 'images/' and
        'labels/' subfolders will be compiled. Or where .tar file will be saved
        if on sockeye.
    skip : int
        The sequence interval step size for downsampling frame data (e.g.,
        passing 5 extracts every 5th sequential frame).
    force : bool, default False
        If True, overwrite files at target destination.

    Returns
    -------
    None
        This function saves image files and annotation arrays straight to disk.

    Raises
    ------
    FileNotFoundError
        If the target configuration index at `train_path` cannot be located on disk.
    ValueError
        If `test_size` or `val_size` parameter boundaries violate standard float constraints.
    OTHER TO BE NOTED

    See Also
    --------
    extract_frames : Image extraction function utilizing OpenCV streams.
    extract_labels : Zipfile extraction for bounding box coordinates.

    Examples
    --------
    WIP
    """
    if not isinstance(ROOT_DIR, Path):
        # Convert inputs to Path objects
        output_dir = str(Path(ROOT_DIR) / output_path)
        # Read in data
        train_df = pd.read_csv(Path(ROOT_DIR) / train_path, index_col=0)
        val_df = pd.read_csv(Path(ROOT_DIR) / val_path, index_col=0)
    else:
        output_dir = str(ROOT_DIR / output_path)
        # Read in data
        train_df = pd.read_csv(str(ROOT_DIR / train_path), index_col=0)
        val_df = pd.read_csv(str(ROOT_DIR / val_path), index_col=0)

    # Automatically resolve the fastest local playground available
    # If on Sockeye, it uses $SLURM_TMPDIR. If on a laptop, it falls back to output_dir
    node_local_storage = os.environ.get("SLURM_TMPDIR", output_dir)
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local_dev")
    on_cluster = "PBS_JOBID" in os.environ or "SLURM_JOB_ID" in os.environ

    if on_cluster:
        if (
            node_local_storage == output_dir
            or "scratch" in str(node_local_storage).lower()
        ):
            print(
                f"WARNING: Shared storage detected! Isolating paths manually via task ID: {task_id}"
            )
            base_local_dir = (
                Path(node_local_storage)
                / "data"
                / "training"
                / f"job_array_{task_id}_yolo_build"
            )
        else:
            # If safely inside an isolated Sockeye NVMe ($SLURM_TMPDIR), a clean static name is safe!
            base_local_dir = (
                Path(node_local_storage)
                / "data"
                / "training"
                / f"job_array_{task_id}_yolo_build"
            )
            print(f"Success: True Node-Local Storage engaged at: {base_local_dir}")
    else:
        print("Working on local node.")
        base_local_dir = Path(node_local_storage)

    # "dataset" Matches the internal name when archiving - see create_yaml
    working_directory = base_local_dir / "dataset"

    # 3. Create the directories safely
    working_directory.mkdir(parents=True, exist_ok=True)
    print(f"Configuring node-local staging environment: {working_directory}")

    # For Pipeline testing on Sockeye
    # Extract frames and bounding box annotations for the train set
    print("\n\n=========================")
    print("Preprocessing Train Split")
    print("=========================\n")

    train_registry = extract_frames(
        video_paths=train_df["clip_relative_path"],
        videos_root=UNLABELLED_CLIPS_DIR,
        working_dir=working_directory,
        split="train",
        skip=skip,
        force=force,
    )
    train_label_registry = extract_labels(
        label_paths=train_df["labelled_clip_relative_path"],
        labels_root=LABELLED_CLIPS_DIR,
        working_dir=working_directory,
        frame_registry=train_registry,
        split="train",
        skip=skip,
        force=force,
    )

    # Prune trailing video frames that don't have matching labels
    print("\nPurging orphaned training images with no corresponding labels...")
    train_img_dir = working_directory / "images" / "train"
    train_frames_dropped = 0
    for video_key, frame_set in train_registry.items():
        label_set = train_label_registry.get(video_key, set())
        # Find frames that have an image but NO matching label
        orphaned_frames = frame_set - label_set

        if orphaned_frames:
            part_str = f"part0{video_key[1]}" if video_key[1] else None
            file_prefix = f"{int(video_key[0]):04}_{part_str}_frame_"
            for orphan_frame in orphaned_frames:
                orphan_path = train_img_dir / f"{file_prefix}{orphan_frame:06d}.jpg"
                if orphan_path.exists():
                    orphan_path.unlink()  # Physically drop the unannotated trailing image
                    train_frames_dropped += 1
    print(f"Dropped {train_frames_dropped} frames.")

    # Extract frames and bounding box annotations for the val set
    print("\n\n=========================")
    print("Preprocessing Validation Split")
    print("=========================\n")
    val_registry = extract_frames(
        video_paths=val_df["clip_relative_path"],
        videos_root=UNLABELLED_CLIPS_DIR,
        working_dir=working_directory,
        split="val",
        skip=skip,
        force=force,
    )
    val_label_registry = extract_labels(
        label_paths=val_df["labelled_clip_relative_path"],
        labels_root=LABELLED_CLIPS_DIR,
        working_dir=working_directory,
        frame_registry=val_registry,
        split="val",
        skip=skip,
        force=force,
    )

    # Prune trailing validation video frames
    print("\nPurging orphaned validation images with no corresponding labels...")
    val_img_dir = working_directory / "images" / "val"
    val_frames_dropped = 0
    for video_key, frame_set in val_registry.items():
        label_set = val_label_registry.get(video_key, set())
        orphaned_frames = frame_set - label_set

        if orphaned_frames:
            part_str = f"part0{video_key[1]}" if video_key[1] else None
            file_prefix = f"{int(video_key[0]):04}_{part_str}_frame_"
            for orphan_frame in orphaned_frames:
                orphan_path = val_img_dir / f"{file_prefix}{orphan_frame:06d}.jpg"
                if orphan_path.exists():
                    orphan_path.unlink()
                    val_frames_dropped += 1

    print(f"Dropped {val_frames_dropped} frames.")

    # Create dataset.yaml
    print("\n--- Creating YAML file ---")
    create_yaml(str(working_directory))

    print("Processing complete. Checking file counts...")

    train_img_count = sum(
        1
        for f in (working_directory / "images" / "train").glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )
    train_lbl_count = sum(
        1 for f in (working_directory / "labels" / "train").glob("*.txt")
    )
    val_img_count = sum(
        1
        for f in (working_directory / "images" / "val").glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )
    val_lbl_count = sum(1 for f in (working_directory / "labels" / "val").glob("*.txt"))

    print(
        f"\nVerification Counts (Local NVMe):\n - Train Images: {train_img_count} | Labels: {train_lbl_count}"
    )
    print(f" - Val Images:   {val_img_count} | Labels: {val_lbl_count}")

    assert train_img_count == train_lbl_count, "Train mismatch detected!"
    assert val_img_count == val_lbl_count, "Validation mismatch detected!"

    if on_cluster:
        # Package everything into a single tarball inside local /tmp
        local_tar_file = base_local_dir / "dataset.tar"
        print(f"\nCompressing complete archive on local node: {local_tar_file}")

        with tarfile.open(local_tar_file, "w") as tar:
            # Packages 'dataset/' as the single root directory inside the archive
            tar.add(str(working_directory), arcname="dataset")

        # Ship the single archive file to /scratch (Instantaneous network transaction)
        cluster_scratch_path = Path(output_dir)
        cluster_scratch_path.mkdir(parents=True, exist_ok=True)
        final_scratch_target = (
            cluster_scratch_path / "dataset.tar"
        )  # f"dataset_{task_id}.tar"

        print(
            f"Transferring clean tar archive to network scratch storage: {final_scratch_target}"
        )
        shutil.move(str(local_tar_file), str(final_scratch_target))

        # 4. Cleanup node local memory entirely
        print("Clearing temporary node-local data directory...")
        shutil.rmtree(base_local_dir)
        print("Preprocessing execution complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing for YOLO models.")
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to data file.",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
        help="Path to data file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for dataset.",
    )
    parser.add_argument(
        "--skip",
        default=1,
        type=int,
        help="Downsampling density. Skip=5 means read every 5th frame.",
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("\n=================================")
    print("PREPROCESSING")
    print("=================================\n")

    print("\nRunning preprocessing for YOLO models...\n")
    args = parse_args()

    run_yolo_preprocessing(
        train_path=args.train_path,
        val_path=args.val_path,
        output_path=args.output_path,
        skip=args.skip,
        force=args.force,
    )
