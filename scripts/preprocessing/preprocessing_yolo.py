"""
Preprocessing functionality for creating training sets to fine-tune YOLO
object detection models.

WIP NOTES.
NOTE 1: Should change to save output to onedrive rather than locally (later on). Creates Too many files for local work.

NOTE 2: Look into sampling frames at x/second rather than a defined skip amount.

NOTE 3: Examples are not finished and need to be properly updated.
"""

from pathlib import Path
import sys
import platform
import subprocess
import io
import os
import tarfile
import zipfile
import shutil
import concurrent.futures
import threading
from typing import List
import cv2
import argparse
import yaml

# import argparse
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data_reading.matching import parse_labelled_name, parse_unlabelled_name
from config import UNLABELLED_CLIPS_DIR, LABELLED_CLIPS_DIR, ROOT_DIR


def extract_labels(
    label_paths: List[str],
    working_dir: Path,
    labels_root: Path,
    frame_registry: dict,
    split: str,
    skip: int,
    FORCE: bool = False,
) -> None:
    """
    Extract labels from annotated data.

    Extracts bounding box labels from .zip files (corresponding to CVAT Output
    folders), and saves files to either a train, or val (split) folder in the
    specified output directory. If FORCE = True, and the output directory
    structure already exists, this will delete the existing folders and rebuild
    the data from scratch.

    `extract_labels` outputs data into a labels/ folder within the output directory.
    This is meant to mimic the data format required for fine-tuning YOLO models.

    This function assumes that bounding box labels are saved as `.txt` files in
    `obj_train_data` folders within .zip files, and that `labels_path` lists
    relative paths to these .zip files within the `labels_root` directory.

    Parameters
    ----------
    labels_path : List[Path]
        List of relative paths to zipped folders containing bounding box
        annotations for each cross-sucking event.
    working_dir : Path
        Path to output directory. Points to node's local temp workspace on
        Sockeye.
    labels_root : Path
        Path to root directory containing zipped folders containing bounding
        box annotations for each cross-sucking event.
    frame_registry : dict
        A dictionary of sets of frames that have been saved for each video.
        Keys are the unique numeric ID and part ID of each video. Used to
        ensure corruption during video reading does not break frame matches
        between labels and videos.
    split : str
        One of `train`, `val`. Dictates which split folder, train/ or val/, the
        labels should be extracted to.
    skip : int
        Number of frames to skip.
    FORCE : bool
        If True, overwritres the existing data. Defaults to False.
    on_cluster : bool
         States whether the function is running on a Sockeye Cluster or a
         local env (default: false, equivalent to a local environment.)

    Returns
    -------
    None :
        This function reads to disk and does not return anything.

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
    to build datasets for training YOLO models. YOLO requires datasets in the
    format listed below where frames and labels are split into train/val sets
    and each frame has a corresponding label with the same name and a .txt
    extension. This implies that there needs to be equal numbers of frames
    and labels, and that frames and labels need to match.


    Examples
    --------
    .. code-block:: python

        import shutil
        import zipfile
        from pathlib import Path
        from my_project.preprocessing import extract_labels

        # Setup temporary directories mimicking a real project workspace
        labels_root = Path("temp_labels_source")
        output_dir = Path("temp_yolo_output")
        labels_root.mkdir(parents=True, exist_ok=True)

        # Ensure the mock filename aligns with your expected identifier conventions.
        mock_zip_name = "0042_p1.zip"
        zip_path = labels_root / mock_zip_name

        # Build a mock CVAT export zip file on the fly
        # Creates files inside 'obj_train_data/' to replicate CVAT structure
        with zipfile.ZipFile(zip_path, "w") as archive:
            for frame_idx in range(10):
                # Simulated YOLO format line: <class_id> <x1> <y1> <x2> <y2>
                mock_annotation = f"0 0.50 0.50 0.22 0.34"
                archive.writestr(
                    f"obj_train_data/frame_{frame_idx:06d}.txt",
                    mock_annotation
                )

        # Execute the label extraction (skipping every 2nd frame)
        extract_labels(
            labels_path=[mock_zip_name],
            labels_root=labels_root,
            output_dir=output_dir,
            target_folder="obj_train_data",
            split="train",
            skip=2,
            FORCE=True,
        )

        # Clean up mock files after ensuring execution succeeded
        shutil.rmtree(labels_root)
        shutil.rmtree(output_dir)
    """
    if split not in ["train", "val"]:
        raise ValueError("split must be either 'train' or 'val'.")

    # Add subdirectories to output directory
    final_output_dir = working_dir / "labels" / split

    # Do nothing if files already exist
    if final_output_dir.exists() and not FORCE:
        print(f"Files already extracted at {final_output_dir}. Skipping computation.")
        return
    # Build output directory
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Input Pre-Validation (Fails fast before touching data)
    validated_paths = []
    for raw_path in label_paths:
        if not isinstance(raw_path, str):
            raise TypeError(f"ERROR: {raw_path} is not a string path.")

        # Standardize path string
        clean_path = labels_root / raw_path.replace("\\", "/")
        if not clean_path.exists():
            raise FileNotFoundError(f"Label file not found: {clean_path}")

        validated_paths.append(clean_path)

    label_batch = {}
    n_files = len(validated_paths)

    # Loop over zip file paths (CVAT Outputs)
    print("\n--- Extracting Annotation Labels ---\n")
    for n, input_path in enumerate(validated_paths, start=1):
        print(f"Extracting files from {input_path.name} ({n}/{n_files} )...")

        # Get numeric id and part id of labelled output
        numeric_id, part_id = parse_labelled_name(str(input_path.name))
        video_key = (int(numeric_id), part_id)

        part_id_str = f"part0{part_id}" if part_id else None
        file_prefix = f"{int(numeric_id):04}_{part_id_str}_"
        target_folder = "obj_train_data"

        # Extracting labels
        with zipfile.ZipFile(input_path, "r") as zip_ref:
            for zinfo in zip_ref.infolist():
                filename = zinfo.filename
                if filename.startswith(target_folder) and filename.endswith(".txt"):
                    pure_name = filename.split("/")[-1]

                    try:
                        frame_num = int(pure_name[6:12])
                    except ValueError:
                        raise (
                            f"ValueError: Incorrect naming conventions for {filename} in {input_path.name}"
                        )

                    if frame_num % skip == 0:

                        # Check video exist in our frame registry
                        # Check video loop successfully extract this specific frame number
                        if frame_registry is not None:
                            if (
                                video_key not in frame_registry
                                or frame_num not in frame_registry[video_key]
                            ):
                                continue  # Drop label safely if the frame isn't on disk

                        new_filename = f"{file_prefix}{pure_name}"
                        # Ready text directly into RAM dictionary
                        label_batch[new_filename] = zip_ref.read(zinfo)

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

        # ----- OLD WORKFLOW -----

        # if on_cluster:
        #     # Create an isolated, hyper-fast playground inside the node's local memory
        #     local_working_dir = Path(f"/tmp/{task_id}_label_extraction")
        #     local_working_dir.mkdir(parents=True, exist_ok=True)
        #     tar_path = local_working_dir / f"labels_batch_{split}.tar"
        # else:
        #     local_working_dir = final_output_dir
        #     tar_path = final_output_dir / f"labels_batch_{split}_{task_id}.tar"

        # print(f"Creating a single memory-tarball at: {tar_path}")

        # # Create a single tar file with all labels
        # with tarfile.open(tar_path, "w") as tar:
        #     for filename, text_bytes in label_batch.items():
        #         idx += 1
        #         if idx % 1000 == 0:
        #             print(f"Executing tar-write: label ({idx}/{batch_len})")
        #         tarinfo = tarfile.TarInfo(name=filename)
        #         tarinfo.size = len(text_bytes)
        #         tar.addfile(tarinfo, io.BytesIO(text_bytes))

        # if on_cluster and (platform.system() != "Windows"):
        #     print("Working on cluster.")

        #     # COPY THE TAR TO SCRATCH FIRST (Single file network transfer = instant)
        #     scratch_tar_path = (
        #         final_output_dir.parent / f"labels_batch_{split}_{task_id}.tar"
        #     )
        #     final_output_dir.mkdir(parents=True, exist_ok=True)

        #     # Move the single tar archive from /tmp to /scratch natively
        #     print("Move tar file to /scratch/...")
        #     shutil.move(str(tar_path), str(scratch_tar_path))

        #     # Wipe the local /tmp folder right away since the tar is safe on scratch
        #     print("Removing tmp directory on node...")
        #     shutil.rmtree(local_working_dir)
        #     print("Done.\n")

        #     # ---- DO NOT UNPACK FILES ON SCRATCH, TRANSFER TO TRAINING NODE AND UNPACK THERE ----

        #     # print(
        #     #     "Exploding files securely at the storage layer via native system tar tool..."
        #     # )
        #     # # 4. Explode the tarball directly into your shared scratch directory
        #     # # Sockeye's native tar tool handles this at hardware block speeds!
        #     # subprocess.run(
        #     #     ["tar", "-xf", str(scratch_tar_path), "-C", str(final_output_dir)],
        #     #     check=True,
        #     # )

        #     # Clean up the temporary archive file on scratch
        #     scratch_tar_path.unlink()

        # else:
        #     print("Working locally.")
        #     print("Exploding files securely via native system tar tool...")
        #     # Standard laptop execution (Mac/Linux optimized, Windows safe fallback)
        #     if platform.system() != "Windows":
        #         subprocess.run(
        #             ["tar", "-xf", str(tar_path), "-C", str(final_output_dir)],
        #             check=True,
        #         )
        #     else:
        #         with tarfile.open(tar_path, "r") as tar:
        #             tar.extractall(path=final_output_dir)

        #     tar_path.unlink()  # Clean up local tar file inside scratch/final directory

        # print(
        #     f"All {len(label_batch)} files successfully loaded to {final_output_dir}!\n"
        # )
        # print()


def extract_frames(
    video_paths: List[str],
    videos_root: Path,
    working_dir: Path,
    split: str,
    skip: int,
    FORCE: bool = False,
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
    FORCE : bool
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

    Examples
    --------
    .. code-block:: python

        from pathlib import Path
        from unittest.mock import MagicMock, patch
        from my_project.preprocessing import extract_frames

        # Create dummy directories
        video_root = Path("temp_videos")
        output_dir = Path("temp_output")
        video_root.mkdir(parents=True, exist_ok=True)
        (video_root / "CS_0042_clip.mp4").touch() # Just an empty file shell

        # Mock OpenCV so it simulates reading 10 successful frames
        with patch('cv2.VideoCapture') as mock_caps:
            instance = mock_caps.return_value
            instance.isOpened.side_effect = [True] * 10 + [False]
            instance.read.return_value = (True, "mock_frame_data")
            instance.grab.return_value = True

            # Run the extraction function safely without a real video file
            extract_frames(
                videos=["CS_0042_clip.mp4"],
                videos_root=video_root,
                output_dir=output_dir,
                split="train",
                skip=2
            )
    """
    print("\n\n--- Extracting Video Frames---\n")

    # Output dir
    final_output_dir = working_dir / "images" / split

    # Rewrite files on FORCE
    if Path(final_output_dir).exists() and not FORCE:
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"Extracting frames from video ({n}/{n_videos}); total frames = {total_frames}..."
        )

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

    # ----- OLD WORKFLOW -----

    # # Synchronize from compute node memory back to Sockeye /scratch Space
    # if on_cluster:
    #     print("Working on Cluster.")
    #     print("Transferring frames from node local memory to network scratch...")

    #     # Wipe old destination directory to avoid collisions
    #     if final_output_dir.exists():
    #         print(f"Cleaning out stale destination directory: {final_output_dir}")
    #         shutil.rmtree(final_output_dir)

    #     # Move the fully populated folder instantly across storage tiers
    #     print(f"Moving extracted frames from {working_dir} to {final_output_dir}.")
    #     shutil.move(str(working_dir), str(final_output_dir))
    #     print(
    #         f"Successfully transferred all frames to network scratch: {final_output_dir}"
    #     )
    # else:
    #     print(f"Local run complete. All frames natively verified at {final_output_dir}")


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
    FORCE: bool = False,
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
    FORCE : bool, default False
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
        skip = int(skip)
        output_dir = str(Path(ROOT_DIR) / output_path)
        # Read in data
        train_df = pd.read_csv(Path(ROOT_DIR) / train_path, index_col=0)
        val_df = pd.read_csv(Path(ROOT_DIR) / val_path, index_col=0)
    else:
        skip = int(skip)
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
        print(f"Success: True Node-Local Storage engaged at: {node_local_storage}")
        base_local_dir = Path(node_local_storage)

    # "dataset" Matches the internal name when archiving - see create_yaml
    base_local_dir = Path(node_local_storage)
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
        FORCE=FORCE,
    )
    extract_labels(
        label_paths=train_df["labelled_clip_relative_path"],
        labels_root=LABELLED_CLIPS_DIR,
        working_dir=working_directory,
        frame_registry=train_registry,
        split="train",
        skip=skip,
        FORCE=FORCE,
    )

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
        FORCE=FORCE,
    )
    extract_labels(
        label_paths=val_df["labelled_clip_relative_path"],
        labels_root=LABELLED_CLIPS_DIR,
        working_dir=working_directory,
        frame_registry=val_registry,
        split="val",
        skip=skip,
        FORCE=FORCE,
    )

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

        # 3. Ship the single archive file to /scratch (Instantaneous network transaction)
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
        help="Output directory for train/val splits.)",
    )
    parser.add_argument(
        "--skip",
        default=1,
        help="Downsampling density. Skip=5 means read every 5th frame.",
    )
    parser.add_argument(
        "--FORCE",
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
        FORCE=args.FORCE,
    )
