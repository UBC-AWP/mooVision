"""
Frame-by-frame object detection training using Ultralytics YOLO models.
"""

from ultralytics import YOLO
from pathlib import Path
import sys
import argparse
import ast
import os
import subprocess
import shutil
import yaml

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import ROOT_DIR


def setup_node_dataset(dataset: str, base_name: str = "dataset") -> Path:
    """
    Extracts and isolates datasets to process-specific node directories,
    preventing file collisions if multiple tasks run on the same node.
    Dynamically configures unique absolute routing paths inside the local YAML file.
    """
    # GENERATE AN ABSOLUTE ISOLATION SIGNATURE PER ARRAY TASK
    # This ensures that even if Task 1 and Task 2 share the same physical server node,
    # they write to completely separated folders on the NVMe storage drive.
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local_dev")
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    tmp_dir_env = os.environ.get("SLURM_TMPDIR")

    on_cluster = "PBS_JOBID" in os.environ or "SLURM_JOB_ID" in os.environ

    if on_cluster:

        print("Working on Cluster, searching for tar path...")
        tar_path = Path(dataset)
        if not tar_path.exists():
            raise FileNotFoundError(f"Dataset archive missing at: {tar_path}")

        if tmp_dir_env:
            # Combined both Job ID and Task ID to guarantee absolute isolation
            # across different jobs and task arrays sharing the same node.
            # e.g., /localscratch/11740177/job_11740177_task_1_dataset
            isolated_node_dir = (
                Path(tmp_dir_env)
                / f"job_{slurm_job_id}_task_{slurm_task_id}_{base_name}"
            )
        else:
            raise ValueError("Could not find SLURM_TMPDIR.")

        # ISOLATED NATIVE C TAR EXTRACTION
        # If this specific task has already successfully extracted its partition, skip to avoid overhead
        if not (isolated_node_dir.exists() and any(isolated_node_dir.iterdir())):
            isolated_node_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Task {slurm_task_id} Staging Area: {isolated_node_dir}")
            print(f"[INFO] Extracting archive via native system tar utility...")

            try:
                # --strip-components=1 strips the top-level folder 'dataset/' from the tarball
                # and unrolls everything straight into our task-isolated directory
                subprocess.run(
                    [
                        "tar",
                        "-xf",
                        str(tar_path),
                        "--strip-components=1",
                        "-C",
                        str(isolated_node_dir),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                if isolated_node_dir.exists():
                    shutil.rmtree(isolated_node_dir)
                error_msg = (
                    e.stderr.decode().strip() if e.stderr else "I/O System Error"
                )
                raise RuntimeError(
                    f"[ERROR] Native tar extraction failed for task {slurm_task_id}: {error_msg}"
                )

        # ────────────────────────────────────────────────────────────────
        # ADDED: POST-EXTRACTION VALIDATION CHECKS
        # ────────────────────────────────────────────────────────────────
        print(
            f"[INFO] Task {slurm_task_id}: Commencing dataset integrity validation..."
        )

        # Step A: Dynamically check for standard YOLO data splits
        valid_splits = [
            d.name
            for d in isolated_node_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name in ["train", "val"]
        ]

        if not valid_splits:
            if isolated_node_dir.exists():
                shutil.rmtree(isolated_node_dir)
            raise RuntimeError(
                f"[ERROR] Validation Failed: No valid YOLO folders ('train'/'val') found inside "
                f"{isolated_node_dir}. Verify if '--strip-components=1' fits your archive structure."
            )

        # Step B: Traverse each found data split to audit structural files
        for split in valid_splits:
            img_dir = isolated_node_dir / "images" / split
            lbl_dir = isolated_node_dir / "labels" / split

            if not img_dir.exists() or not lbl_dir.exists():
                if isolated_node_dir.exists():
                    shutil.rmtree(isolated_node_dir)
                raise FileNotFoundError(
                    f"[ERROR] Structural Mismatch in split [{split.upper()}]. "
                    f"Expected both branches to exist:\n  - {img_dir}\n  - {lbl_dir}"
                )

            # Map unique stems (filenames without extensions) while dropping hidden OS assets
            image_stems = {
                f.stem
                for f in img_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            }
            label_stems = {
                f.stem
                for f in lbl_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            }

            num_images = len(image_stems)
            num_labels = len(label_stems)

            # Step C: Catch empty directory states (indicates quota block or corrupted extraction)
            if num_images == 0:
                if isolated_node_dir.exists():
                    shutil.rmtree(isolated_node_dir)
                raise ValueError(
                    f"[ERROR] Integrity Failure: Image directory for split [{split.upper()}] is completely empty."
                )

            # Step D: Alert or flag on image vs bounding-box count variance
            if num_images != num_labels:
                print(
                    f"[WARNING] File count discrepancy in split [{split.upper()}]: "
                    f"Found {num_images} images but {num_labels} labels."
                )

                # Check for images missing corresponding annotation labels
                orphaned_images = image_stems - label_stems
                if orphaned_images:
                    print(
                        f"First 5 images missing a matching .txt label file: {list(orphaned_images)[:5]}"
                    )

            print(
                f"[SUCCESS] Split [{split.upper()}] checked: {num_images} images and {num_labels} labels verified."
            )

        print(
            f"[SUCCESS] Dataset structure validation completely passed for Task {slurm_task_id}."
        )
        # ────────────────────────────────────────────────────────────────
    else:
        # Laptop fallback strategy
        data_path = ROOT_DIR / dataset
        isolated_node_dir = data_path.parent

    # CONFIGURE INDEPENDENT PATHS IN THE LOCAL YAML FILE
    # The config file sits inside each task's isolated local folder footprint,
    # meaning tasks will never read or overwrite each other's paths.
    local_yaml_path = isolated_node_dir / "dataset.yaml"
    if not local_yaml_path.exists():
        raise FileNotFoundError(
            f"Missing dataset.yaml configuration file inside task directory: {isolated_node_dir}"
        )

    # Read the isolated config file
    with open(local_yaml_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Re-route the 'path' key from './dataset' to the absolute path of this task's folder
    config_data["path"] = str(isolated_node_dir.resolve())

    # Save the modifications back to the node-local storage drive
    with open(local_yaml_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    if on_cluster:
        print(
            f"[SUCCESS] Task {slurm_task_id} absolute dataset path locked to: {config_data['path']}"
        )
    else:
        print(f"[SUCCESS] Absolute dataset path locked to: {config_data['path']}")
    return local_yaml_path


def train_yolo_model(
    yaml_path: str,
    name: str,
    project: Path | str,
    weights_dir: str,
    model: int,
    model_size: int,
    device: str,
    workers: int,
    exist_ok: bool = False,  # YOLO default
    epochs: int = 100,  # YOLO default
    time: float = None,  # YOLO default
    patience: int = 50,  # YOLO default
    batch: int | float = 16,  # YOLO default
    img_size: int = 640,  # YOLO default
    save: bool = True,  # YOLO default
    rect: bool = True,  # Keep original aspect ratio
    **kwargs,
):
    """
    Train a YOLO model.

    Takes in training set from preprocessing.py and trains a YOLOv8 model of
    the specified size.

    Parameters are taken from ultralytics docs. For more information and
    arguments see:
    https://docs.ultralytics.com/modes/train.


    Parameters
    ----------
    yaml_path : str
        Path to data.yaml configuration file. File contains dataset paths,
        and classes (e.g. cross-sucking)
    name : str
        Experiment name. e.g. cross-sucking-yolo-v26-m
    project : Path | str
        Output directory. This is where output of model training is stored, including model
        weights and predictions.
    weights_dir : str
        Path to local pretrained YOLO weights (raw weights)
    model : int
        YOLO model version. i.e. 26 for v26, 8 for v8.
    model_size : str
        Model size (n=nano, s=small, m=medium, l=large, x=xlarge). A larger
        model may take longer to train or use, but it may also provide
        increased performance.
    device : str
        device to train on (0 for GPU, 'cuda' for CUDA, 'cpu' for CPU, 'mps' for GPU on Mac).
    workers : int
        Number of parallel threads to run on sockeye.
    exist_ok : bool
        If True, overwrite existing project/name directory.
    epochs : int
        Number of training epochs. Each epoch represents one full pass of the
        dataest. A larger epoch value will increase training duration, but may
        also increase accuracy.
    time : float
        Maximum training time in hours. This will override the number of epochs
        if reached first. Note, this may affect model performance and training
        times.
    patience : int
        Number of epochs to wait until stopping the training due no
        improvements in validation metrics.
    batch : int or float
        Batch size. See ultrlytics docs for more info:
        https://www.ultralytics.com/glossary/batch-size
    img_size : int
        Image size for training. Resized to squares of img_size x img_size if
        rect = False, else keeps aspect ratio. This may affect model accuracy
        and computational complexity.
    save : bool
        Enables saving of training checkpoints and final model weights. Set to True by default
        to save results of model training.
    rect : bool
        If True, keeps image ratio when rescaling images. The longest side is
        equal to img_size. Can improve efficiency and speed but may affect
        accuracy.
    **kwargs : named arguemnts
        See ultralytics docs for additional arguments.


    Returns
    -------
    None
        This function reads to disk and does not return anything.


    Examples
    --------
    .. code-block:: python

        # After running scripts read_all_clips_index.py, and splitting.py run...
        yaml_path = "data/processed/pipeline_testing/yolo_format"  #/path/to/yaml/file
        train_yolo_model(
            yaml_path=yaml_path,
            model_size="n",
            epochs=3,
            imgsz=640,
            batch=8,
            rect=True,
            device="mps",
            save=True,
            patience=50,
            name="cross_sucking",
            project="MooVision",
        )
    """
    # Loads pretrained model
    if weights_dir:
        print("Loading model from weights...")
        if model == 26:
            print("Option 1: Loading YOLO26 model")
            final_model_target = Path(weights_dir) / f"yolo{model}{model_size}.pt"
            if not final_model_target.exists():
                raise FileNotFoundError(
                    f"Could not find {final_model_target}. Set model to yolo{model}{model_size}.pt in .env_sockeye and run ./01_setup.sh on the login node."
                )
        else:
            final_model_target = Path(weights_dir) / f"yolov{model}{model_size}.pt"
            if not final_model_target.exists():
                raise FileNotFoundError(
                    f"Could not find {final_model_target}. Set model to yolov{model}{model_size}.pt in .env_sockeye and run ./01_setup.sh on the login node."
                )
    else:
        print("loading model from internet...")
        if model == 26:
            final_model_target = f"yolo{model}{model_size}.pt"

        else:
            final_model_target = f"yolov{model}{model_size}.pt"

    model = YOLO(final_model_target)
    print(f"Model loaded from: {final_model_target}")

    # Train
    model.train(
        data=yaml_path,
        name=name,
        project=project,
        device=device,
        exist_ok=exist_ok,
        epochs=epochs,
        time=time,
        patience=patience,
        batch=batch,
        imgsz=img_size,
        save=save,
        rect=rect,
        workers=workers,
        cache=False,
        **kwargs,
    )

    # return model, results


def parse_args():
    parser = argparse.ArgumentParser(description="Training for YOLO models.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to data file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to grain on '0' for GPU, 'cpu' for cpu, 'cuda' for CUDA, 'mps' for mac gpu.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loader workers.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="cross-sucking",
        help="Experiment name",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="MooVision",
        help="Output directory.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="",
        help="Path to local weights folder. Leave blank to auto-download from the internet.",
    )
    parser.add_argument(
        "--model",
        default=26,
        type=int,
        help="YOLO model version. i.e. 26 for v26, 8 for v8.",
    )
    parser.add_argument(
        "--model_size",
        default="n",  # nano
        type=str,
        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge).",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="Number of full passes over df during training.",
    )
    parser.add_argument(
        "--time",
        default=None,
        type=float,
        help="Max training time in hours.",
    )
    parser.add_argument(
        "--batch",
        default=32,
        type=int,
        help="Batch size.",
    )

    parser.add_argument(
        "--patience",
        default=20,
        type=int,
        help="Number of epochs to wait with no imrpovement before early stopping.",
    )
    parser.add_argument(
        "--img_size",
        default=640,
        type=int,
        help="Image size for training.",
    )
    parser.add_argument(
        "--not_rect",
        default=True,
        action="store_false",
        help="Do not maintain image ratio.",
    )
    parser.add_argument(
        "--do_not_save",
        default=True,
        action="store_false",
        help="Disables saving of training checkpoints and final model weights.",
    )
    parser.add_argument(
        "--exist_ok",
        default=False,
        action="store_true",
        help="Overwrite existing project.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        # If passed as '[0,1]', convert it directly to a native Python list: [0, 1]
        final_device = ast.literal_eval(args.device)
    except (ValueError, SyntaxError):
        # Fallback if passed without brackets (e.g., "0,1"), split by comma
        if "," in args.device:
            final_device = [int(x.strip()) for x in args.device.split(",")]
        else:
            # Single device (e.g., "0" or "cpu")
            final_device = args.device.strip()

    print("\nUnpacking tar file into dataset...\n")
    node_yaml_config = setup_node_dataset(
        dataset=args.dataset,
    )
    print(f"\nData set unpacked at {node_yaml_config.parent}")

    print("Training YOLO model ...")
    train_yolo_model(
        yaml_path=str(node_yaml_config),
        name=args.name,
        project=args.project,
        weights_dir=args.weights_dir,
        model=args.model,
        model_size=args.model_size,
        device=final_device,
        workers=args.workers,
        exist_ok=args.exist_ok,
        epochs=args.epochs,
        batch=args.batch,
        time=args.time,
        patience=args.patience,
        img_size=args.img_size,
        rect=args.not_rect,
        save=args.do_not_save,
    )

    print("Training complete!")
    print("Best model saved to: runs/detect/MooVision/cross-sucking/weights/best.pt")
