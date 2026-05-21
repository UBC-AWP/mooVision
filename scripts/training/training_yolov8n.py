"""
Base function to be used for training a YOLO models.

Start by training one model on the random shuffle data set. Frame by frame approach? Ignore this?
"""

from ultralytics import YOLO
from pathlib import Path
import sys

test_path = Path("data/processed/pipeline_testing/test.csv").absolute()
sys.path.append(str(Path(__file__).parent.parent.parent))

## IMAGE SIZE: 1920 x 1800


def train_yolo_model(
    yaml_path,
    model_size="n",  # n, s, m, l, x
    epochs=3,
    imgsz=640,  # adjusted image size from 1920 t0 1800
    batch=8,
    rect=True,  # Keep image ratio
    device="mps",  # GPU device, or 'cpu, or 'mps' on Mac   --make as argument
):
    """
    Train YOLOv8 model

    Args:
        yaml_path: path to data.yaml
        model_size: model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: training epochs
        imgsz: image size
        batch: batch size
        device: device to train on (0 for GPU, 'cpu' for CPU)
    """
    # Load pretrained model
    model = YOLO(f"yolov8{model_size}.pt")

    # Train
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        cache="ram",
        patience=50,  # Early stopping patience
        save=True,
        plots=True,
        rect=rect,
        name="cross_sucking",  # Experiment name
        project="data/processed/pipeline_testing/runs/detect",  # Output directory
    )

    # return model, results


# UPDATE TO CLEAN AND TAKE IN ARGUMENTS
def main():
    # Train the model
    yaml_path = "data/processed/pipeline_testing/yolo_format"  #
    train_yolo_model(
        yaml_path=yaml_path, model_size="n", epochs=3, batch=8  # Nano model
    )


print("Training complete!")
print("Best model saved to: ...")

if __name__ == "__main__":
    main()
