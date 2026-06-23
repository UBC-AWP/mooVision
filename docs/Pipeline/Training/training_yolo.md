# YOLO Model Training

Frame-by-frame object detection training using Ultralytics YOLO models. Wraps the Ultralytics training API with a fixed interface and CLI entry point, allowing models to be trained from the command line or imported directly into a pipeline.

---

## Summary

As with preprocessing, we specify two workflows for training models: local and Sockeye (HPC). Local workflows run linearly, where each training process must be called individually. HPC workflows parallelize training jobs to run each of the eight experimental splits simultaneously.

| Property | Detail |
| :--- | :--- |
| **Script** | `training_yolo.py` |
| **Framework** | [Ultralytics YOLO](https://docs.ultralytics.com/modes/train) |
| **Supported Models** | YOLOv8, YOLOv11 (YOLO26+) |
| **Input** | Dataset YAML + pretrained weights / Dataset.tar + pretrained weights |
| **Output** | Model weights + training artifacts written to disk |
| **CLI Support** | Yes |

> **See Also:** `04_train_yolo.sh` for the YOLO training Sockeye job submission script and `run_training_pipeline.sh` for the Sockeye orchestrator script.

---

## Inputs

* **Local:** Reads training data directly from the local `dataset.yaml` path, loads the pretrained YOLO weights, updates the path references, and runs training.
* **Sockeye (HPC):** Unpacks a compressed `dataset.tar` file locally to the compute node to avoid numerous slow file transfers over the HPC network. It updates `yaml_path` to the current node directory, loads pre-downloaded YOLO weights, and runs training against the freshly unpacked files.

## Output

This script writes directly to disk and returns `None`. All outputs are organized by experimental split and written to `project/name/split`.

```text
project/
└── name/
    ├── weights/
    │   ├── best.pt       # Best checkpoint evaluated by validation metric
    │   └── last.pt       # Final checkpoint from the last epoch
    ├── results.csv       # Per-epoch training and validation metrics
    └── args.yaml         # Resolved runtime training configuration

```

---

## How It Works

1. **Environment Detection:** The script detects whether the execution context is local or on Sockeye.

* On **Sockeye**, the `dataset.tar` file is unpacked directly on the local compute node, and the `yaml_path` variable inside `dataset.yaml` is updated dynamically to target the node location.
* On a **Local** machine, `yaml_path` inside `dataset.yaml` is updated to your current project path before training execution.

2. **Weight Allocation:**

* On **Sockeye**, a pretrained YOLO model is loaded from a local, pre-downloaded weights directory (configurable via `.env_sockeye` and prepared by running `bash 01_setup.sh`). *Ensure the model version downloaded and the variables specified in `04_train_yolo.sh` match before running.*
* **Locally**, weights are fetched directly from Ultralytics using the version and size flags passed via the CLI (e.g., resolving to `yolov8n.pt` or `yolo11n.pt`).

3. **Training Execution:** The model is trained on the dataset described in the generated YAML file.

4. **Parameter Forwarding:** Core training arguments—such as epochs, batch size, image size, patience, and device constraints—are passed directly to the underlying Ultralytics `model.train()` execution call.

---

## Core Pipeline Functionality

### YAML Path Configuration & Dataset Unpacking

Because absolute paths on shared HPC clusters change depending on the assigned compute node, YAML target paths are initially specified as relative links (`./dataset`). This script resolves and updates paths dynamically at runtime to prevent pathing failures during training.

On Sockeye, datasets are bundled into a single `dataset.tar` archive to maximize filesystem network efficiency. This script unpacks the dataset onto the local compute node, updates the underlying configuration paths, and validates folder lengths and file counts to ensure complete data integrity before initializing the model.

### Image Handling

Training images are dynamically resized to the target `img_size` resolution. When `rect=True` (default behavior), the longest side of the frame is restricted to `img_size` while fully preserving the original aspect ratio. This configuration is highly recommended for high-resolution footage like the 1920×1800 MooVision clips utilized in this pipeline.

### Early Stopping

If the chosen validation metrics fail to improve for `patience` consecutive epochs, training terminates automatically to save compute cycles. You can also pass the `time` parameter to hard-cap the absolute wall-clock runtime in hours, regardless of how many epochs have finished.

### Batch Size

Supports flexible, variable batch configuration. The default batch size on Sockeye is optimized at 64 to maximize throughput; allocating exceptionally large batch values on certain compute nodes may trigger Out Of Memory (OOM) faults.

### Epoch Setting

The `epochs` parameter specifies the maximum number of full training passes over the dataset. While higher epoch caps allow for deep feature optimization, they significantly increase the probability of model overfitting.

### Cache Optimization

Allows caching of dataset images and labels directly into memory using `--cache`. Caching is **generally discouraged** for this pipeline as the video frame datasets are exceptionally large and risk flooding memory limits. Given the high-speed local disk nodes on Sockeye, caching yields negligible performance benefits.

### Overwriting Existing Runs

To prevent accidental data loss, the pipeline prevents overwriting existing directories by default. To replace an existing run folder with a new experiment, pass the `--exist_ok` flag.

### Hardware & GPU Compatibility

The architecture is designed to scale across single or multi-GPU environments using the `device` and `workers` parameters. By default, `device` is set to `"cpu"` to ensure safe, error-free execution on machines lacking dedicated acceleration.

* To target a specific GPU, pass a single string identifier: `--device="0"`
* To run parallelized multi-GPU training, pass a collection list: `--device=[0,1]`
* The `workers` argument (defaults to 8) regulates the number of parallel CPU data-loading threads used to feed data to the GPU.

### Model Selection

Pretrained base weights are dynamically resolved at execution as `yolo{version}{size}.pt`. For example, setting `model=26` (MooVision internally aliases 26 for v11 architectures) and `model_size="m"` will resolve to `yolo11m.pt`. Ultralytics automatically handles fetching remote files if matching weights are not discovered locally.

---

## Command Line Arguments

For complete information regarding downstream engine configurations, visit the [Ultralytics Training Reference Documentation](https://docs.ultralytics.com/modes/train).

| Argument | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--dataset` | `str` | *None* | **Yes** | Path to the dataset configuration target file. |
| `--device` | `str` | `"cpu"` | No | Hardware acceleration choice. Examples: `'0'` or `'0,1'` for specific GPUs, `'cpu'`, `'cuda'`, or `'mps'`. |
| `--workers` | `int` | `8` | No | Number of parallel data loader worker processes. |
| `--name` | `str` | `"cross-sucking"` | No | Name identifier for the specific training run/experiment. |
| `--project` | `str` | `"MooVision"` | No | Name of the output project directory where results are saved. |
| `--weights_dir` | `str` | `""` | No | Path to a local folder containing model weights. Leave completely blank to automatically download weights from the internet. |
| `--model` | `int` | `26` | No | YOLO model version integer (e.g., `8` for YOLOv8, `26` for YOLO11/26+). |
| `--model_size` | `str` | `"n"` | No | Model scale/variant size configuration options: `n` (nano), `s` (small), `m` (medium), `l` (large), or `x` (xlarge). |
| `--epochs` | `int` | `50` | No | Maximum number of full validation passes over the entire dataframe during training. |
| `--time` | `float` | `None` | No | Maximum absolute training duration allowed, specified in hours. |
| `--batch` | `int` | `32` | No | Training batch size per iteration. |
| `--patience` | `int` | `20` | No | Early stopping threshold. Training stops early if no evaluation improvements are seen after this many consecutive epochs. |
| `--img_size` | `int` | `640` | No | Target image size (resolution pixel width/height) used for model training. |
| `--not_rect` | Flag | `True` | No | **Disables** rectangular training. Including this flag switches the underlying configuration (`rect`) to `False`, forcing the system to stop maintaining native image aspect ratios. |
| `--do_not_save` | Flag | `True` | No | **Disables** tracking saves. Including this flag switches the underlying configuration (`save`) to `False`, preventing training checkpoints and finalized model weights from saving to disk. |
| `--exist_ok` | Flag | `False` | No | **Enables** project overwriting. Including this flag sets the parameter to `True`, allowing the script to safely overwrite existing project directories without throwing errors. |
| `--cache` | Flag | `False` | No | **Enables** image caching. Including this flag sets the parameter to `True`, caching raw images in RAM/storage memory for significantly quicker batch step testing. |

---

## Usage

### Command Line (Local) Example

```bash
uv run scripts/training/training_yolo.py \
  --dataset="data/training/pipeline_demo/dataset/dataset.yaml" \
  --project="pipeline_demo" \
  --name="demo_01" \
  --device="cpu" \
  --epochs=1 \
  --batch=8

```

---

## Function Reference

::: scripts.training.training_yolo
options:
show_source: false
show_root_heading: true

---

## Work In Progress (WIP)

1. **Flexible Parameter Forwarding:** Integrate `kwargs` capability into parsed argument behaviors. This will unlock direct passthrough optimization, allowing users to leverage any latent arguments supported natively within the Ultralytics API straight from this interface wrapper.

### Target Interface Concept

Pass additional Ultralytics arguments via `--kwargs` (feature implementation pending):

```bash
python train_yolo_model.py --yaml_path dataset.yaml --kwargs lr0=0.01 cos_lr=True

```