"""
Frame-by-frame object detection training using Ultralytics YOLO models.
"""

from ultralytics import YOLO
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))


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
    # Load pretrained model
    if weights_dir:
        print("loading model from weights...")
        if model == 26:
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
        print("loading model...")
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
        workers=8,
        cache=False,
        **kwargs,
    )

    # return model, results


def parse_args():
    parser = argparse.ArgumentParser(description="Training for YOLO models.")
    parser.add_argument(
        "--yaml_path",
        type=str,
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
        help="Device to grain on '0' for GPU, 'cpu' for cpu, 'cuda' for CUDA, 'mps' for mac gpu.",
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

    print("Training YOLO model ...")
    train_yolo_model(
        yaml_path=args.yaml_path,
        name=args.name,
        project=args.project,
        weights_dir=args.weights_dir,
        model=args.model,
        model_size=args.model_size,
        device=args.device,
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
