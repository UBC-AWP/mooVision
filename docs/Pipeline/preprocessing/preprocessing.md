# # Data Preprocessing Pipeline

This page documents how to utilize `scripts/training/preprocessing_yolo.py` to compile video data and bounding box annotations into structured training configurations optimized for fine-tuning **YOLO object detection models**.

---

## Summary

To transition from abstract `train.csv` and `val.csv` splits to actual model optimization, the pipeline slices video clips into discrete images while simultaneously extracting compressed bounding box files.

This script processes videos alongside zipped annotation folders (e.g., CVAT output archives), synchronizes their frames, and produces a standardized YOLO-compliant architecture:

```text
my_yolo_dataset/
├── dataset.yaml             <-- Defines paths to train/val subsets & class names
├── images/                  <-- Main directory for image files (.jpg)
│   ├── train/               
│   └── val/                 
└── labels/                  <-- Main directory for text annotations (.txt)
    ├── train/               
    └── val/
```

To optimize for parallel executions, we makes use of UBC's high performance computing cluster (HPC), UBC ARC Sockeye.

A note on HPC's:

- HPC's are optimized for transfering large files across storage space, however they are notoriously slow at transferring many small files. Since, preprocessing for YOLO models requires the creation and transfer of many small files, there are slight differences in workflows when running locally vs running on Sockeye. This script will automatically detect when it is running locally or on a cluster, and run the optimal workflow for each scenario.

---

## Intput Requirements

| Property | Details |
|---|---|
| Format | `train.csv`, `val.csv` |
| Content | Upstream data indeces of a train/val split |

Before running the preprocessing execution block, ensure your local workspace satisfies the following dependencies:

- `train.csv` / `val.csv` Paths: An upstream data index mapping individual clips to their corresponding annotation zip files.
- Environment File: A fully populated local .env configuration mapping your local volume paths for ROOT_DIR and LOCAL_DIR.
- Downloaded videos and zipped annotations (else preprocessing time will be significantly extended by file downloads).

If your data indexes and splits are not yet generated, execute the upstream data layer pipeline sequentially via your terminal:

```bash
uv run scripts/read_all_clips_index.py

uv run scripts/splitting.py
```

---

## Output

Local: Outputs the file structure layed out below.
UBC ARC Sockeye: Outputs a tar file containing the dataset structure below. This is setup to avoid 

By default results are saved to `ROOT_DIR/data/training/yolo/split_name`.

```text
data/training/yolo/split_name/
├── dataset.yaml     
├── images/                  
│   ├── train/              
│   │   ├── 0001_None_frame_000001.jpg
│   │   ├── 0001_None_frame_000002.jpg
│   │   ├── ...
│   │   ├── 0002_part01_frame_000001.jpg
│   │   ├── 0002_part01_frame_000002.jpg
│   │   ├── ...
│   └── val/                
│       ├── 0003_part02_frame_000001.jpg
│       └── 0003_part02_frame_000002.jpg
│       └── ...
└── labels/                 
    ├── train/                  
    │   ├── 0001_None_frame_000001.txt  
    │   ├── 0001_None_frame_000002.txt
    │   ├── ...
    │   ├── 0002_part01_frame_000001.txt
    │   ├── 0002_part01_frame_000002.txt
    │   ├── ...
    └── val/                
        ├── 0003_part02_frame_000001.txt
        └── 0003_part02_frame_000001.txt


data.yaml
   "path": "./dataset",
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": "cross-sucking"

```

---

## How it works

Local Workflow: Working directory is the root directory

1) Read in `train.csv` and `val.csv` from the root directory.
2) Extract a list of relative paths to cross-sucking videos, and pass this to `extract_frames`. Extract a list of relative paths to zipped annotation folder, and pass this to `extract_labels`. Note, relative paths of cross-sucking videos are stored in the `clips_relative_path` column, and relative paths of annotations are stored in the `labelled_clips_relative_path` column.  
3) `extract_frames` reads in each video in the passed list, extracts frames as jpg images using MultiThreadPooling and Semaphore attributes, and saves frames to either an `images/train/` or an `images/val/` subfolder within the working directory depending on the split source.  
4) `extract_labels` opens each .zip file in the list, extracts bounding box annotations as .txt files from 'obj_train_data', removes the prefix 'obj_train_data', and saves these to either a `labels/train/` or a `labels/val/` subfolder within the output directory depending on the split source.  
5) `build_yaml` creates a yaml file at `data/training/yolo/split_name/` within the working directory with required classes and paths. Note that the dataset path in the dataset.yaml file is set within training_yolo.py to accomodate running jobs on Sockeye where this path cannot be predetermined.

UBC ARC Sockeye Workflow: Working directory is a high-speed tmp/ folder on the compute node.

6) Build a tar file from the dataset and transfer this to ROOT_DIR/data/training/yolo/split_name/dataset.tar e.g. ROOT_DIR/data/training/yolo/random/dataset.tar.

note: This is specified to be transferred back to the compute node as a .tar file and unpacked there for training. This workflow avoids the transportation of many small files across the HPC and makes use of the fast transfer system for large files. In this way, we are able to optimize preprocessing time.

---

### Edge Cases

**"Crucial Synchronization Requirement"**
    YOLO models link images to annotations by replacing sections of file paths: (`images` -> `labels`, `.jpg` -> `.txt`). For the dataset to sync correctly, video processing frame rates **must exactly match** the frame indexes exported in the annotation bounding boxes. If there is a frame rate or naming mismatch, your labels will misalign with your target images. It is assumed that both videos and CVAT outputs do not have missing intermediary frames in this regard. Preprocessing_yolo.py handles frame matching with unique naming conventions based on the numeric ID and part ID for each video, and a customizable skip parameter that is passed to both frame and label extraction functions. This ensures each frame and label pair has a unique naming convention and can be passed safely downstream to training.

**"Corrupted Videos, and Misaligned Annotations"**
    When extracting frames, video corruption may result in a premature exit of a video extraction loops. If not dealt with, this will result in unaligned annotation where extracted labels do not have matching video frames. Moreover, CVAT removes non-event labels (.txt files) leading to the end of a video clip. This leads to scenarios where videos extend past annotaion labels and some frames do not have matching labels. To account for this, preprocessing_yolo.py has functionality to compare and remove both labels and frames which are missing the other part. This ensures all frames and labels exist as matching pairs, and training sets will not break or cause downstream disruptions when training YOLO models.

**"Missing Videos, or Missing Annotation"**
    This is handled when reading in data, as 

---

## Core Pipeline Concepts

### File Structure and Naming Mechanics

To match names across video frame to annotation labels, and prevent namespace collision across overlapping frame numbers, both extract_frames and extract_labels use parsing functions from `scripts/matching.py` to extract the numeric ID and part ID of each video/zipped file. Adding the numericID and partID ensures there are no repeated names, and allows functionality for reading in multiple videos and zipped files.

Files are uniformly written out using a deterministic unique index:

```{markdown}
Format:
numericID_partID_frame_frameNum.jpg (or.txt)

Example:
0001_None_frame_000001.jpg, 0001_None_frame_000001.txt. 
0003_part01_frame_000645.jpg, 0003_part01_frame_000645.txt

```

#### Challenges

`extract_labels` assumes that within zipped annotation folders bounding boxes exist in a obj_train_data folder as `.txt` files, along with images highlighted with said bounding boxes. `extract_labels` extracts only the `.txt` files in this folder, forcibly carrying the hierarchy `obj_train_data/frame_000001.txt` with it as well. The file paths of `.txt` files are then rewritten to remove the `obj_train_data` hierarchy and flatten the output. i.e. this changes filepaths from `labels/train/obj_train_data/0001_None_frame_000001.txt` to `labels/train/0001_None_frame_000001.txt`

### Temporal Frame Skipping

The skip parameter controls downsampling density. For instance, setting skip=5 extracts every 5th frame. Using larger step intervals accelerates dataset generation and reduces spatial autocorrelation (redundant data), but excessive downsampling introduces temporal tracking errors across fast-moving targets.

### Frame and Label Matching Validation

The skip parameter controls downsampling density. For instance, setting skip=5 extracts every 5th frame. Using larger step intervals accelerates dataset generation and reduces spatial autocorrelation (redundant data), but excessive downsampling introduces temporal tracking errors across fast-moving targets.

### Overwrite Behaviours (FORCE)

By default, the pipeline preserves existing targets to save disk I/O time. If an output target directory is present, the script skips parsing. To discard stale data matrices and completely rebuild your dataset structures from scratch, pass the explicit overwrite flag: FORCE=True.

---

## Usage

| Argument | Type | Required / Default | Description |
| :--- | :--- | :--- | :--- |
| `--train_path` | `str` | **Required** | Path to the training data file. |
| `--val_path` | `str` | **Required** | Path to the validation data file. |
| `--output_path` | `str` | **Required** | Output directory where the train/val splits will be saved. |
| `--skip` | `int` | `1` | Downsampling density. For example, `5` means the script will read every 5th frame. |
| `--FORCE` | `flag` | `False` | Overwrite existing files in the output directory if specified. |

## Examples

### Basic

By default, the script requires a train path, val path and output directory specified as strings.

```bash
uv run scripts/training/preprocessing.py --input_path="data/processed/pipeline_demo/train.csv" --val_path="data/processed/pipeline_demo/val.csv" --output_dir="data/training/pipeline_demo/"
```

### Skip Frames

Passing the `--skip=` argument controls the downsampling density. `skip=5` reads every 5th frame.

```bash
uv run scripts/training/preprocessing.py --input_path="data/processed/pipeline_demo/train.csv" --val_path="data/processed/pipeline_demo/val.csv" --output_dir="data/training/pipeline_demo/" --skip=5
```

### Overwriting Files

To overwrite files use the `--FORCE` argument.

```bash
uv run scripts/training/preprocessing.py --input_path="data/processed/pipeline_demo/train.csv" --val_path="data/processed/pipeline_demo/val.csv" --output_dir="data/training/pipeline_demo/" --FORCE
```

---

## Function Reference

::: scripts.preprocessing.preprocessing_yolo
    options:
        show_source: false
        show_root_heading: true

---

## Update reccomendations

- Scale Out Compute: Update to chunk the dataset and run preprocessing across multiple nodes, where each node returns a .tar file of its share of the dataset. These can then be unpacked on the compute node during training. Cuts down on current preprocessing time of 1-4hrs and adds scalability if the number of videos increases.
