"""
Functions for extracting labels from CVAT zip folders
"""

import sys
from typing import List, Tuple, Dict, Any, Set
from pathlib import Path
import zipfile

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data_reading.matching import parse_labelled_name
from scripts.preprocessing.utils import validate_file_paths


def generate_label_metadata(zip_name: str) -> Tuple[Tuple[int, str | None], str]:
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

        video_key, file_prefix = generate_label_metadata(zip_name=input_path.name)
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
