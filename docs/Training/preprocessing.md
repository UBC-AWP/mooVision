# # Data Preprocessing Pipeline

This page documents how to utilize `scripts/training/preprocessing.py` to compile raw frame data and bounding box annotations into structured training configurations optimized for fine-tuning **YOLO object detection models**.

---

## Summary

To transition from abstract `train.csv` and `test.csv` splits to actual model optimization, the pipeline slices raw video clips into discrete images while simultaneously extracting compressed bounding box files.

The script processes raw videos alongside zipped annotation folders (e.g., CVAT output archives), synchronizes their frames, and produces a standardized YOLO-compliant architecture:

```text
my_yolo_dataset/
├── data.yaml                <-- Defines paths to train/val subsets & class names
├── images/                  <-- Main directory for image files (.jpg)
│   ├── train/               
│   └── val/                 
└── labels/                  <-- Main directory for text annotations (.txt)
    ├── train/               
    └── val/
```

---

## Intput Requirements

| Property | Details |
|---|---|
| Format | `train.csv` |
| Content | Upstream data index of a train/test split |

Before running the preprocessing execution block, ensure your local workspace satisfies the following dependencies:

- `train.csv` / `test.csv` Paths: An upstream data index mapping individual clips to their corresponding annotation zip files.
- Environment File: A fully populated local .env configuration mapping your local volume paths for ROOT_DIR and LOCAL_DIR.
- Downloaded videos and zipped annotations (else preprocessing time will be significantly extended by file downloads).
- If your data indexes are not yet generated, execute the upstream data layer pipeline sequentially via your terminal:

```bash
uv run scripts/read_all_clips_index.py

uv run scripts/splitting.py
```

---

## Output

By default results are saved to `data/training/yolo/split_name`.

```text
data/training/yolo/split_name/
├── data.yaml     
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
...

```

---

## How it works

1) Read in `train.csv` from path, split into train and val splits.  
2) For each split, pass a list relative paths to cross-sucking videos to `extract_frames` and a list of relative paths to zipped annotations to `extract_labels`. Relative paths of cross-sucking videos are stored in the `clips_relative_path` column, and relative paths of annotations are stored in the `labelled_clips_relative_path` column.  
3) `extract_frames` reads in each video in the list, extracts frames as jpg images, and saves frames to either an `images/train/` or an `images/val/` subfolder depending on the split source.  
4) `extract_labels` reads each .zip file in the list, extracts bounding box annotations as .txt files, and saves these to either a `labels/train/` or a `labels/val/` subfolder depending on the split source.  
5) `build_yaml` creates a yaml file at `data/training/yolo/split_name/` with required classes and paths.  

### Edge Cases

**"Crucial Synchronization Requirement"**
    YOLO models link images to annotations by replacing sections of file paths: (`images` -> `labels`, `.jpg` -> `.txt`). For your dataset to sync correctly, your video processing frame rate **must exactly match** the frame indexes exported in your annotation bounding boxes. If there is a frame rate or naming mismatch, your labels will misalign with your target images, and an error will be raised.

**"Corrupted Videos, and Mislabelled Annotations"**
    OLO models link images to annotations by replacing sections of file paths: (`images` -> `labels`, `.jpg` -> `.txt`). For your dataset to sync correctly, your video processing frame rate **must exactly match** the frame indexes exported in your annotation bounding boxes. If there is a frame rate or naming mismatch, your labels will misalign with your target images, and an error will be raised. `-- To be filled out`

**"Missing Videos, or Missing Annotations"**
    OLO models link images to annotations by replacing sections of file paths: (`images` -> `labels`, `.jpg` -> `.txt`). For your dataset to sync correctly, your video processing frame rate **must exactly match** the frame indexes exported in your annotation bounding boxes. If there is a frame rate or naming mismatch, your labels will misalign with your target images, and an error will be raised. `-- To be filled out`

---

## Core Pipeline Concepts

### File Structure and Naming Mechanics

To match names across videos and annotations, and prevent namespace collision across overlapping frame numbers, both extract_frames and extract_labels use parsing functions from `scripts/matching.py` to extract the numeric ID and part ID of each video/zipped file. Adding the numericID and partID ensures there are no repeated names, and allows functionality for reading in multiple videos and zipped files.

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

**"Crucial Synchronization Requirement"**
    The labeling extraction matches text coordinates directly against image arrays via exact name mappings. Your video recording frame rate must exactly match the frame indexes exported in your annotation bounding boxes. Any timing offset will cause your labels to misalign with your frames.

### Overwrite Behaviours (FORCE)

By default, the pipeline preserves existing targets to save disk I/O time. If an output target directory is present, the script skips parsing. To discard stale data matrices and completely rebuild your dataset structures from scratch, pass the explicit overwrite flag: FORCE=True.

---

## Usage

### Basic

By default, the script requires an input path and output directory specified as strings.

```bash
uv run scripts/training/preprocessing.py --input_path="data/processed/pipeline_testing/train.csv" --output_dir="data/processed/pipeline_testing/yolo_format"
```

### Skip Frames

Passing the `--skip=` argument controls the downsampling density. `skip=5` reads every 5th frame.

```bash
uv run scripts/training/preprocessing.py --input_path="data/processed/pipeline_testing/train.csv" --output_dir="data/processed/pipeline_testing/yolo_format" --skip=5
```

### Control Training and Validation Sizes

Use the `--val_size=` to control the training and validation sizes. val_size is in [0.0, 1.0]. The training size is 1-val_size.

```bash
uv run scripts/training/preprocessing.py --input_path="data/processed/pipeline_testing/train.csv" --output_dir="data/processed/pipeline_testing/yolo_format" --val_size=0.2
```

### Changing the Random State

To control the random state for splitting train/val sets, use `--random_state=`.

```bash
uv run scripts/training/preprocessing.py --input_path="data/processed/pipeline_testing/train.csv" --output_dir="data/processed/pipeline_testing/yolo_format" --random_state=300
```

### Overwriting Files

To overwrite files use the `--FORCE` argument.

```bash
uv run scripts/training/preprocessing.py --input_path="data/processed/pipeline_testing/train.csv" --output_dir="data/processed/pipeline_testing/yolo_format" --FORCE
```

---

## Function Reference

::: scripts.training.preprocessing
    options:
        show_source: false
        show_root_heading: true