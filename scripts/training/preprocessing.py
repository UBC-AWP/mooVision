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
import zipfile
import shutil
from typing import List
import cv2
import re
import argparse
import yaml

# import argparse
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.matching import parse_labelled_name, parse_unlabelled_name
from config import UNLABELLED_CLIPS_DIR, LABELLED_CLIPS_DIR


def extract_labels(
    labels_path: List[Path],
    output_dir: Path,
    labels_root: Path,
    target_folder: str,
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
        List of relative paths to label outputs of cross-sucking events.
    output_dir : Path
        Path to output directory.
    labels_root : Path
        Path to root directory of label outputs of cross-sucking events.
    target_folder : str
        The target folder to extract frames from in the .zip files conataining
        the annotated data. For CVAT outputs this should be `obj_train_data`.
    split : str
        One of `train`, `val`. Dictates which split folder, train/ or val/, the
        labels should be extracted to.
    skip : int
        Number of frames to skip.
    FORCE : bool
        If True, overwritres the existing data. Defaults to False.

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

    # Add subdirectories to output directory
    output_dir = Path(output_dir) / "labels" / split

    # Do nothing if files already exist
    if Path(output_dir).exists() and not FORCE:
        print(f"Files already extracted at {Path(__file__) / Path(output_dir)}")
    else:
        # Delete path and files if they already exist
        if output_dir.exists() and output_dir.is_dir():
            # Recursively deletes the directory and all contents
            shutil.rmtree(output_dir)

        #### ---- CHECK INPUT LIST IS NOT EMPTY ---- ####
        #### ---- CHECK INPUT TYPES ARE STRINGS ---- ####

        # Loop over zip file paths (CVAT Outputs)
        for input_path in labels_path:

            ### THIS SHOULD BE EARLIER MAYBE? = yes, else we might
            # # run through severral iterations of this list until we get to something that is not a string

            if not isinstance(input_path, str):
                raise TypeError(
                    f"{input_path} is not a string. labels_path should be a list of strings."
                )

            # Standardie Path to Posix Standard
            input_path = labels_root / input_path.replace("\\", "/")

            if not labels_path.exists():
                raise FileNotFoundError(f"{input_path} not found.")

            # Get numeric id and part id of labelled output
            numeric_id, part_id = parse_labelled_name(str(input_path.name))

            ### DO I NEED TO TEST THE OUTPUTS OF THIS??
            # --- NO? Already confirmed from other function inputs?

            if part_id:
                # Format nicely
                part_id = f"part0{part_id}"

            # Target folder in zip file
            target_folder = target_folder

            ## WHAT HAPPENS IF THIS THROWS AN ERROR!
            if not target_folder.exists() or not target_folder.is_dir():
                raise FileNotFoundError(f"{target_folder} not found at {input_path}")

            # Look in zip folder
            with zipfile.ZipFile(input_path, "r") as zip_ref:

                # List all files in zip folder
                all_files = zip_ref.namelist()

                # Isolate only the .txt files belonging to the target folder hierarchy
                files_to_extract = sorted(
                    [
                        f
                        for f in all_files
                        if f.startswith(
                            target_folder
                        )  # assumes files names: target_folder/frame_000000.txt
                        and ".txt" in f
                        and (
                            int(re.search(r"(\d+)", f).group(1)) % skip == 0
                        )  # Take every `skip` frame
                    ]
                )

                ### TEST LENGTH OF LIST HERE FOR .TXT FILES --- Return could not find labels at input_path/target_folder

                for file in files_to_extract:
                    # Extract individual files explicitly to target destination
                    zip_ref.extract(file, output_dir)

            # target_folder = "obj_train_data/"
            # path to target folder in output dir (extraction adds target folder in output hierarchy)
            target_folder = output_dir / target_folder

            if target_folder.exists() and target_folder.is_dir():
                # Iterate through all files inside the sub-folder
                for file_path in target_folder.iterdir():
                    if file_path.is_file():
                        # Define target path (e.g., extraction_output/train/0000_{part}_frame_000000.txt)
                        target_path = (
                            output_dir
                            / f"{int(numeric_id):04}_{part_id}_{str(file_path.name)}"
                        )

                        # Atomic filesystem move (Metadata update only, no disk write)
                        file_path.rename(target_path)

                # Delete the now-empty target folder from output dir
                target_folder.rmdir()
                print(f"Files saved to {output_dir}")
            else:
                raise FileNotFoundError(
                    f"{target_folder} structure not found or already processed."
                )


def extract_frames(
    videos: List[str],
    videos_root: Path,
    output_dir,
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
    output_dir : Path
        Path to output directory to save extracted frames to.
    split : str
        One of `train`, `val`. Dictates which split folder, train/ or val/, the
        labels should be extracted to.
    skip : int
        Number of frames to skip.
    FORCE : bool
        If True, overwritres the existing data. Defaults to False.


    Returns
    -------
    None :
        This function reads to disk and does not return anything.


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

    # Output dir
    output_dir = Path(output_dir) / "images" / split
    # Rewrite files on FORCE
    if Path(output_dir).exists() and not FORCE:
        print(f"Files already extracted at {Path(__file__) / Path(output_dir)}")
    else:
        # Delete path if it already exists
        if output_dir.exists() and output_dir.is_dir():
            # Recursively deletes the directory and all contents
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        for video_file in videos:

            # Standardize Path to Posix Standard
            video_file = videos_root / video_file.replace("\\", "/")

            # Print working video...
            print(f"Extracting frames from {video_file.name}...")

            # Get numeric id and part id of video clip
            numeric_id, part_id = parse_unlabelled_name(str(video_file.name))
            if part_id:
                part_id = f"part0{part_id}"

            # Video capture
            cap = cv2.VideoCapture(str(video_file))
            frame_idx = 0

            while cap.isOpened():
                # Read frames at every `skip`` position, ignore the rest
                if frame_idx % skip == 0:
                    # Decode frame
                    ret, frame = cap.read()
                    if not ret:  # Break if decoding fails
                        break
                    # Create Output Path
                    frame_path = (
                        output_dir
                        / f"{int(numeric_id):04}_{part_id}_frame_{frame_idx:06d}.jpg"
                    )

                    # Write frame to output dir
                    cv2.imwrite(str(frame_path), frame)
                else:
                    ret = cap.grab()  # advances position, does not decode frame.
                    if not ret:  # Skip efficiently
                        break

                frame_idx += 1

            # release video
            cap.release()
            print(f"{video_file.name} frames saved to {output_dir}")


def create_yaml(
    dataset_path: str,
    class_names: list[str] = ["cross-sucking"],
    output_path: str = "dataset.yaml",
    train_dir: str = "images/train",
    val_dir: str = "images/val",
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
    train_dir : str, optional
        Training images directory, relative to dataset_path.
    val_dir : str, optional
        Validation images directory, relative to dataset_path.

    Returns
    -------
    None
        This function reads straight to disk and does not return anything.
    """
    config = {
        "path": str(Path(dataset_path).resolve()),
        "train": train_dir,
        "val": val_dir,
        "nc": len(class_names),
        "names": class_names,
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# UPDATE TO CLEAN AND TAKE IN ARGUMENTS
def run_yolo_preprocessing(
    input_path: str,
    output_dir: str,
    target_folder: str,
    skip: int,
    val_size: float,
    random_state: int = 300,
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
    input_path : str
        Path to the source CSV file containing video and label mappings.
    output_dir : Path
        Base directory path where the 'images/' and 'labels/' subfolders
        will be compiled.
    skip : int
        The sequence interval step size for downsampling frame data (e.g.,
        passing 5 extracts every 5th sequential frame).
    test_size : float
        The proportion of the dataset to include in the validation split
        (between 0.0 and 1.0).
    val_size : float
        The proportion of the dataset to include in the training split
        (between 0.0 and 1.0).
    random_state : int, default 300
        The seed passed directly to `train_test_split` to ensure
        reproducibility.
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
    # Convert inputs to Path objects
    train_path = Path(input_path).absolute()
    output_path = Path(output_dir).absolute()

    train_df = pd.read_csv(train_path, index_col=0)

    train, val = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
    )

    # Extract frames and bounding box annotations for the train set
    extract_labels(
        input_paths=train["labelled_clip_relative_path"],
        labels_root=LABELLED_CLIPS_DIR,
        output_dir=output_path,
        target_folder=target_folder,
        split="train",
        skip=skip,
        FORCE=FORCE,
    )
    extract_frames(
        videos=train["clip_relative_path"],
        videos_root=UNLABELLED_CLIPS_DIR,
        output_dir=output_path,
        split="train",
        skip=skip,
        FORCE=FORCE,
    )

    # Extract frames and bounding box annotations for the val set
    extract_labels(
        input_paths=val["labelled_clip_relative_path"],
        labels_root=LABELLED_CLIPS_DIR,
        output_dir=output_path,
        target_folder=target_folder,
        split="val",
        skip=skip,
        FORCE=FORCE,
    )
    extract_frames(
        videos=val["clip_relative_path"],
        videos_root=UNLABELLED_CLIPS_DIR,
        output_dir=output_path,
        split="val",
        skip=skip,
        FORCE=FORCE,
    )

    # Create dataset.yaml
    create_yaml(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing for YOLO models.")
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help=f"Output directory for train/test splits (default: {Path("data/processed/pipeline_testing/yolo_format").absolute()})",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default="obj_train_data",
        help=f"Output directory for train/test splits (default: {Path("data/processed/pipeline_testing/yolo_format").absolute()})",
    )
    parser.add_argument(
        "--skip",
        default=1,
        help="Downsampling density. Skip=5 means read every 5th frame.",
    )
    parser.add_argument(
        "--val_size",
        default=0.2,
        type=float,
        help="Size of validation set for train/val split.",
    )
    parser.add_argument(
        "--random_state",
        default=300,
        type=int,
        help="Random state for reproducibility in train/val split.",
    )
    parser.add_argument(
        "--FORCE",
        default=False,
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("Running preprocessing for YOLO models...")
    args = parse_args()
    run_yolo_preprocessing(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_folder=args.target_folder,
        skip=args.skip,
        val_size=args.val_size,
        random_state=args.random_state,
        FORCE=args.FORCE,
    )
    print("All files created.")
    print("Preprocessing for YOLO models complete.")
