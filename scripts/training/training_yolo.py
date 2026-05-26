"""
Base function to be used for training a YOLO models based on a
frame-by-frame analysis.

NOTE:  IMAGE SIZE= 1920 x 1800 (for moovision cross-sucking-clips)
"""

from ultralytics import YOLO
from pathlib import Path
import sys

test_path = Path("data/processed/pipeline_testing/test.csv").absolute()
sys.path.append(str(Path(__file__).parent.parent.parent))


def train_yolo_model(
    yaml_path: Path,
    name: str,
    project: Path,
    model: int,
    model_size: int,
    device: str,
    exist_ok: bool = False,  # YOLO default
    epochs: int = 100,  # YOLO default
    time: float = None,  # YOLO default
    patience: int = 100,  # YOLO default
    batch: int | float = 16,  # YOLO default
    img_size: int = 640,  # YOLO default
    save: bool = True,  # YOLO default
    rect: bool = True,  # Keep original aspect ratio
    **kwargs,
):
    """
    Train YOLOv8 model.

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
    model : int
        YOLO model to use for training. i.e. 26 for v26, 8 for v8.
    model_size : str
        Model size (n=nano, s=small, m=medium, l=large, x=xlarge). A larger
        model may take longer to train or use, but it may also provide
        increased performance.
    device : str
        device to train on (0 for GPU, 'cuda' for CUDA, 'cpu' for CPU, 'mps' for GPU on Mac).
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
        plots=True,
        name="cross_sucking",
        project="runs/detect",
    )
    """
    # Load pretrained model
    model = YOLO(f"yolov{model}{model_size}.pt")

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
        img_size=img_size,
        save=save,
        rect=rect,
        **kwargs,
    )

    # return model, results


# UPDATE TO CLEAN AND TAKE IN ARGUMENTS
def main():
    # Train the model
    yaml_path = "data/processed/pipeline_testing/yolo_format"  #
    train_yolo_model(
        yaml_path=yaml_path,
        name="cross_sucking",
        project="runs/detect",
        model=26,
        model_size="n",  # Nano model
        device="mps",  # Mac GPU --- change to detect correct output here!
        epochs=3,
        batch=8,
        imgsz=640,  # adjusted image size from 1920x1800
        rect=True,  # Keep image ratio
        save=True,
        plots=True,
    )


print("Training complete!")
print("Best model saved to: ...")

if __name__ == "__main__":
    main()
