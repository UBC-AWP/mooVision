# YOLO Model Training

Frame-by-frame object detection training using Ultralytics YOLO models. Wraps the
Ultralytics training API with a fixed interface and CLI entry point, allowing
models to be trained from the command line or imported directly into a pipeline.

---

## Summary

| Property | Detail |
|---|---|
| Script | `train_yolo_model.py` |
| Framework | [Ultralytics YOLO](https://docs.ultralytics.com/modes/train) |
| Supported models | YOLOv8, YOLOv11 (YOLO26+) |
| Input | Dataset YAML + pretrained weights |
| Output | Model weights + training artefacts written to disk |
| CLI support | Yes |

---

## Inputs

The script loads a pretrained YOLO model and runs training against the dataset defined
in `yaml_path`.

## Output

This function writes directly to disk and returns `None`. All outputs are written to `project/name/`.

```text
project/
└── name/
    ├── weights/
    │   ├── best.pt       # Best checkpoint by validation metric
    │   └── last.pt       # Final epoch checkpoint
    ├── results.csv       # Per-epoch training metrics
    └── args.yaml         # Resolved training configuration
```

---

## How It Works

1. A pretrained YOLO model is loaded from Ultralytics using the version and size
   specified (`yolov{model}{model_size}.pt`).
2. The model is trained on the dataset described in the provided YAML file.
3. Training parameters — epochs, batch size, image size, patience, etc. — are
   forwarded directly to `model.train()`, along with any `**kwargs`.
4. Weights and training artefacts are written to `project/name/`.

---

## Core Pipeline Functionality

### Image Handling

Training images are resized to `img_size` pixels. When `rect=True` (default),
the longest side is set to `img_size` and the aspect ratio is preserved. This is
recommended for high-resolution footage such as the 1920×1800 Moovision clips
used in this pipeline.

### Early Stopping

If validation metrics do not improve for `patience` consecutive epochs, training
halts automatically. Set `time` to cap training wall-clock duration regardless
of epoch count.

### Model Selection

The pretrained weight file is resolved at runtime as `yolov{model}{model_size}.pt`.
For example, `model=26, model_size="m"` loads `yolov26m.pt`. Ultralytics downloads
the weights automatically if not found locally.

---

## Usage

### Python

```python
from train_yolo_model import train_yolo_model
 
train_yolo_model(
    yaml_path="data/processed/dataset.yaml",
    name="cross-sucking",
    project="runs/detect",
    model=26,
    model_size="m",
    device="mps",
    epochs=100,
    batch=16,
    img_size=640,
    patience=50,
    rect=True,
    save=True,
)
```

### Command Line

```bash
python train_yolo_model.py \
  --yaml_path data/processed/dataset.yaml \
  --name cross-sucking-yolov26-m \
  --project runs/detect \
  --model 26 \
  --model_size m \
  --device mps \
  --epochs 100 \
  --batch 16 \
  --img_size 640 \
  --patience 50
```

Pass additional Ultralytics arguments via `--kwargs`: (functionality not complete yet)

```bash
python train_yolo_model.py --yaml_path dataset.yaml --kwargs lr0=0.01 cos_lr=True
```

---

## Function Reference

::: scripts.training.training_yolo
    options:
        show_source: false
        show_root_heading: true