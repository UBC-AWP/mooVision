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

The script takes in a path to a `train.csv` file. Before running the preprocessing execution block, ensure your local workspace satisfies the following dependencies:

- `train.csv` / `test.csv` Paths: An upstream data index mapping individual clips to their corresponding annotation zip files.
- Environment File: A fully populated local .env configuration mapping your local volume paths for UNLABELLED_CLIPS_DIR and LABELLED_CLIPS_DIR.
- If your tracking indexes are not yet generated, execute the upstream data layer pipeline sequentially via your terminal:

```bash
uv run scripts/read_all_clips_index.py
```

```bash
uv run scripts/splitting.py
```

---

## Output

Output is a YOLO training set.

```text
Labelling format: numericID_partID_frame_frameNum.jpg (or.txt)
Examples: 0001_None_frame_000001.jpg, 0003_part01_frame_000645.jpg

my_yolo_dataset/
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

The script takes in a `train.csv` file, creates train/val splits and sends a list of relative paths of cross-sucking videos ( `clip_relative_path` col) to `extract_frames` and a list of relative paths of annotated data (`labelled_clip_relative_path` col) to `extract_labels`.

`extract_frames` reads in each video in the list, extracts frames as jpg images, and saves frames to either a `train/` or `val/` subfolder within an `images/` folder depending on the split source.

`extract_labels` reads each .zip file in the list, extracts bounding box annotations as .txt files, and saves these to either a `train/` or `val/` subfolder within an `labels/` folder depending on the split source.

!!! warning "Crucial Synchronization Requirement"
    YOLO models link images to annotations by replacing sections of file paths: (`images` -> `labels`, `.jpg` -> `.txt`)
    For your dataset to sync correctly, your video processing frame rate **must exactly match** the frame indexes exported in your annotation bounding boxes. If there is a frame rate or naming mismatch, your labels will misalign with your target images.

### Functionality

#### Naming

To handle name matching, both `extract_frames` and `extract_labels` use parsing functions from `scripts/matching.py` to extract the numeric ID and part ID of each video/zipped file. Frames are then named using the following template.

```{markdown}
Format:
numericID_partID_frame_frameNum.jpg (or.txt)

Example:
0001_None_frame_000001.jpg, 0001_None_frame_000001.txt. 
0003_part01_frame_000645.jpg, 0003_part01_frame_000645.txt

```

Since the first frame of each video is extracted as `frame_000001.jpg` be default, adding the numericID and partID ensures there are no repeated names, and allows functionality for reading in multiple videos and zipped files.

#### Splitting

The `split` argument aligned with the video source and determines which folder (train/val) the output should be saved to.

#### Skipping frames

The `skip` argument allows for skipping of frames when reading in data. For example, `skip=5` will read in every 5th frame. Reading in fewer frames will allow for faster preprocessing, and training, although it may come at the cost of increased error.

#### Overwriting Files

By default, running preprocessing will not overwrite any file, if they exist. To overwrite existing files/directories run `FORCE=True`. This will remove any existing training sets at teh output directory and recreate a new one from scratch.
 
---

## Usage

---

## Function Reference

---

::: scripts.training.preprocessing
    options:
        show_source: false
        show_root_heading: true