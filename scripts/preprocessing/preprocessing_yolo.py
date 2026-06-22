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
import os

import shutil
import argparse
import tarfile

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import UNLABELLED_CLIPS_DIR, LABELLED_CLIPS_DIR, ROOT_DIR
from scripts.preprocessing.extract_frames import extract_frames
from scripts.preprocessing.extract_labels import extract_labels
from scripts.preprocessing.utils import create_yaml


def resolve_working_directory(output_dir: Path) -> Path:
    """
    Resolve the optimal node-local staging folder based on the host system.

    Parameters
    ----------
    output_dir : pathlib.Path
        The default fallback compilation directory path.

    Returns
    -------
    working_directory : pathlib.Path
        The verified, isolated local scratch workspace directory path.

    Raises
    ------
    TypeError
        If `output_dir` is not an instance of `pathlib.Path`.
    """
    # Runtime Type Checking
    if not isinstance(output_dir, Path):
        raise TypeError(
            f"Argument 'output_dir' must be a Path object, received {type(output_dir).__name__}"
        )

    node_local_storage = os.environ.get("SLURM_TMPDIR", str(output_dir))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local_dev")
    on_cluster = "PBS_JOBID" in os.environ or "SLURM_JOB_ID" in os.environ

    if on_cluster:
        # Protect shared filesystems via isolated job IDs
        base_local_dir = (
            Path(node_local_storage)
            / "data"
            / "training"
            / f"job_array_{task_id}_yolo_build"
        )
        if (
            node_local_storage == str(output_dir)
            or "scratch" in str(node_local_storage).lower()
        ):
            print(
                f"WARNING: Shared storage detected! Isolating paths via task ID: {task_id}"
            )
        else:
            print(f"Success: True Node-Local NVMe Storage engaged at: {base_local_dir}")
    else:
        print("Working on local node playground environment.")
        base_local_dir = Path(node_local_storage)

    working_directory = base_local_dir / "dataset"
    print(f"\nConfiguring node-local staging environment: {working_directory}")
    working_directory.mkdir(parents=True, exist_ok=True)
    return working_directory, base_local_dir, on_cluster


def process_single_split(
    df: pd.DataFrame,
    working_dir: Path,
    split: str,
    skip: int,
    force: bool,
    clips_root_dir: Path,
    labels_root_dir: Path,
) -> None:
    """
    Process image frames, label batches, and purge orphans for a single dataset split.
    """
    print("\n\n=========================")
    print(f"Preprocessing {split.capitalize()} Split")
    print("=========================\n")

    # 1. Execute Custom Extraction Pipelines
    frame_registry = extract_frames(
        video_paths=df["clip_relative_path"].to_list(),
        videos_root=clips_root_dir,
        working_dir=working_dir,
        split=split,
        skip=skip,
        force=force,
    )

    label_registry = extract_labels(
        label_paths=df["labelled_clip_relative_path"].to_list(),
        labels_root=labels_root_dir,
        working_dir=working_dir,
        frame_registry=frame_registry,
        split=split,
        skip=skip,
        force=force,
    )

    # 2. Sanitize and Purge Orphans (Align image frames cleanly with bounding boxes)
    print(f"\nPurging orphaned {split} images with no corresponding labels...")
    img_dir = working_dir / "images" / split
    frames_dropped = 0

    for video_key, frame_set in frame_registry.items():
        label_set = label_registry.get(video_key, set())
        orphaned_frames = frame_set - label_set

        if orphaned_frames:
            part_str = f"part0{video_key[1]}" if video_key[1] else None
            file_prefix = f"{int(video_key[0]):04}_{part_str}_frame_"

            for orphan_frame in orphaned_frames:
                orphan_path = img_dir / f"{file_prefix}{orphan_frame:06d}.jpg"
                if orphan_path.exists():
                    orphan_path.unlink()
                    frames_dropped += 1

    print(f"Dropped {frames_dropped} unannotated trailing {split} frames.")


def validate_dataset(
    working_dir: Path,
) -> None:
    """
    Validates train and val folders have the same numbers of images and labels.
    """
    print("Processing complete. Checking file counts...")

    train_img_count = sum(
        1
        for f in (working_dir / "images" / "train").glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )
    train_lbl_count = sum(1 for f in (working_dir / "labels" / "train").glob("*.txt"))
    val_img_count = sum(
        1
        for f in (working_dir / "images" / "val").glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )
    val_lbl_count = sum(1 for f in (working_dir / "labels" / "val").glob("*.txt"))

    print(
        f"\nVerification Counts (Local NVMe):\n - Train Images: {train_img_count} | Labels: {train_lbl_count}"
    )
    print(f" - Val Images:   {val_img_count} | Labels: {val_lbl_count}")

    assert train_img_count == train_lbl_count, "Train mismatch detected!"
    assert val_img_count == val_lbl_count, "Validation mismatch detected!"


def package_tar_file(base_local_dir, working_dir, output_dir):
    """
    Package Dataset into tar file and transfer to path
    """
    # Package everything into a single tarball inside local /tmp
    local_tar_file = base_local_dir / "dataset.tar"
    print(f"\nCompressing complete archive on local node: {local_tar_file}")

    with tarfile.open(local_tar_file, "w") as tar:
        # Packages 'dataset/' as the single root directory inside the archive
        tar.add(str(working_dir), arcname="dataset")

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
    root = Path(ROOT_DIR)
    output_dir = root / output_path

    # Read source CSV allocations directly
    train_df = pd.read_csv(root / train_path, index_col=0)
    val_df = pd.read_csv(root / val_path, index_col=0)

    working_directory, base_local_dir, on_cluster = resolve_working_directory(
        output_dir=output_dir
    )

    process_single_split(
        df=train_df,
        working_dir=working_directory,
        split="train",
        skip=skip,
        force=force,
        clips_root_dir=UNLABELLED_CLIPS_DIR,
        labels_root_dir=LABELLED_CLIPS_DIR,
    )

    process_single_split(
        df=val_df,
        working_dir=working_directory,
        split="val",
        skip=skip,
        force=force,
        clips_root_dir=UNLABELLED_CLIPS_DIR,
        labels_root_dir=LABELLED_CLIPS_DIR,
    )

    create_yaml(str(working_directory))
    validate_dataset(working_directory)

    if on_cluster:
        package_tar_file(base_local_dir, working_directory, output_dir)


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
