# MooVision

MooVision is a computer-vision pipeline for detecting **cross-sucking behaviour** in socially housed dairy calves from angled overhead pen video. The goal is to reduce the time and effort required for manual review/labeling by automatically producing candidate events, clipped video segments, and structured metadata for downstream analysis.

## Project Overview

Cross-sucking (here: sucking directed at various body parts of other calves) is a welfare concern in group-housed calves and is currently studied via manual labeling of long video recordings. This project builds a scalable workflow that:

- takes raw pen video as input
- runs a baseline detector (pretrained YOLOv8) with simple logic on top (e.g., proximity/overlap + temporal persistence)
- outputs predicted event windows and metadata (start/end time, confidence, pen, weaning stage, day)
- optionally generates clipped videos for review and evaluation

## Data Requirements (Data Collection)

- Data index file such as `all_clips_index.csv` listing cross-sucking clips, and associated data... WIP

## Repository Structure (high level)

- `src/`: library code (config, preprocessing, baseline inference, evaluation)
- `scripts/`: runnable entry points (run baseline, build clip index, etc.)
- `eda/`: EDA notebooks
- `tests/`: unit tests
- `docs/` : project documentation
- `report/`: report assets

## Environment Setup

This project uses `uv` for package management.

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

2. Move into the folder where you wish to download the repo. Clone the repo.

    ```bash
    git clone git@github.com:UBC-AWP/mooVision.git
    ```

3. Cd into the repo.

    ```bash
    cd mooVision
    ```

4. Run `uv sync` to install all dependencies.

    ```bash
    uv sync
    ```

5. Run scripts with `uv run python <script.py>`

---

## Local `.env` configuration (required)

We use a local `.env` file (stored at the **repo root**) to configure machine-specific paths (e.g., where the video clips live). This avoids hardcoding absolute paths in code.

1. Create a `.env` file at the repo root:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set your local data path, for example:

   ```bash
   ROOT_DIR=/path/to/your/root/directory
   LOCAL_DIR=/path/to/your/local/directory
   ```

3. `.env` is ignored by git (do not commit). If you need to change what variables exist, update `.env.example` instead.

## OneDrive Sync

UBC offers OneDrive accounts for researchers and research groups. If your data is hosted on OneDrive you will need to sync your account to your local computer.

1. Download the OneDrive App.

2. Sign in with the email account connected to the OneDrive folder hosting your data. For UBC researchers this is your UBC email.

3. Follow the prompts to sync your folder. Alternatively, open OneDrive in your browser, move into the folder you want to sync, and click sync in the top toolbar.

---

## Configuring your repository in `config.py`

We use `config.py` to configure paths to video and label directories. By default, the config file is setup for source videos, cross-sucking clips, and annotation labels existing in the following data structure:

```plaintext
mooVision/
└── data/
    ├── raw_cross_sucking_datalog/
    │   └── videos/                          <- Raw field footages from cameras.
    │
    ├── cross_sucking_clips/                 <- Curated video segments containing cross-sucking events.
    │   └── all_clips_index.csv              <- Index file listing curated video segments containing cross-sucking events and associated metadata.
    │
    └── cross_sucking_labelled/              <- Ground-truth frames and splits as zipped files.
     
```

To configure the project for your data structure layout, change the following variables within the `config.py` module:

```plaintext
# Videos and labels
UNLABELLED_CLIPS_DIR = ROOT_DIR / "<cross_sucking_clips_folder>"
LABELLED_CLIPS_DIR = ROOT_DIR / "<cross_sucking_labels_folder>"
SOURCE_VIDEOS_DIR = ROOT_DIR / "<source_videos_folder>"
```

To change the location of your data index file, edit the following:

```plaintext
# Raw and Processed Index Paths
INDEX_PATH = Path/to/your/index/file/<file>
```

## Pipeline Diagram
<img src="img/pipeline_diagram.png" width="370"/>

## Running the Pipeline (demo)

After configuring you `.env` and config files, run the following commands from your terminal in the MooVision root directory to move through a local demo of the project workflow. For more information see, project documentation.

1. Read in Raw index, and process for videos . Note, this will throw a lot of warnings when ran. These are telling you that the function is using the clips NOT found in fixed_clips when multiple versions of the same video are found.

   ```bash
   uv run scripts/data_reading/read_all_clips_index.py --FORCE
   ```

2. Split Data into train and test splits. This outputs the 8 main train/val/test splits tested to the `data/processed/` folder within the root directory. For more information, see the project documentation.

   ```bash
   uv run scripts/data_splitting/split_data.py --FORCE
   ```

3. Preprocess Data for fine-tuning YOLO object detection model (using demo training set). Note that the train_path, val_path, and output_path are relative to the root directory `ROOT_DIR` here. Here we run only the demo training set for efficiency purposes as running all 8 splits is a long process. A similar command can be used to run any of the other splits, for more information see project documentation.

   ```bash
   uv run scripts/preprocessing/preprocessing_yolo.py \
   --train_path="data/processed/pipeline_demo/train.csv" \
   --val_path="data/processed/pipeline_demo/val.csv" \
   --output_path="data/training/pipeline_demo/" \
   --skip=10 \
   --FORCE
   ```

    Note: it might take a while to upload files to OneDrive if you have set your root directory there.

4. Train YOLO object detection model. Note: change `--device="cpu"` to 0 for GPU, `cuda` for CUDA GPU, 'mps' for Mac GPU, or "cpu" if no GPU available. Note the dataset size is small so training should not take long. This will save the model to `data/yolo_training_runs/project` in the root directory (OneDrive if setup that way). For more information see the project documentation.

   ```bash
   uv run scripts/training/training_yolo.py --dataset="data/training/pipeline_demo/dataset/dataset.yaml" --project="pipeline_demo" --name="demo_01" --device="cpu" --epochs=1 --batch=8
   ```

5. Run baseline on test videos to capture cross-sucking events and produce associated metadata:

    - First, source your `.env` to load the path variables:

    ```bash
    source .env
    ```

   - Cross-sucking examples (~2-3 minutes):

      ```bash
      uv run scripts/models/baseline/baseline.py --csv "$ROOT_DIR/data/processed/pipeline_demo/test.csv" --frame_skip 100
      ```

   - Non-cross-sucking examples (~1-2 minutes):

      ```bash
      uv run scripts/models/baseline/baseline.py --video "sample_videos/non_cross_sucking_clip_sample/ch05_20251114073451_15s.mp4"
      ```

6. Load and Run fine-tuned YOLO model on demo video (2s buffer):

    ```bash
    uv run scripts/run_testing_2.py --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" --data_path "data/processed/pipeline_demo/test.csv" --chunk 0 --chunk_pct 1.0
    ```

    Note: if you wish to overwrite the existing result, add the argument `--overwrite` in the end

    ```bash
    uv run scripts/run_testing_2.py --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" --data_path "data/processed/pipeline_demo/test.csv" --chunk 0 --chunk_pct 1.0 --overwrite
    ```

7. Evaluate results, including frame-level bounding box IoU computed from CVAT annotations. Note that `--labelled_clips_dir` should point to the directory containing the CVAT annotation zip files for the clips being evaluated; this argument is optional and can be omitted if frame-level bbox IoU is not needed.

    Evaluate plain fine-tuned YOLO (CS detection only, no temporal linking):

    ```bash
    uv run python scripts/evaluation.py \
        --predictions "results/metadata/pipeline_demo/yolo/" \
        --ground_truth data/processed/processed_clips_index.csv \
        --output results/evaluation_report_yolo.json \
        --labelled_clips_dir "cross_sucking_labelled"
    ```

    Evaluate YOLO + Seq-NMS (with temporal linking):

```bash
uv run python scripts/evaluation.py \
    --predictions "results/metadata/pipeline_demo/seq-nms/" \
    --ground_truth data/processed/processed_clips_index.csv \
    --output results/evaluation_report_seq_nms.json \
    --labelled_clips_dir "cross_sucking_labelled"
```

8. Clip frames from results:

   ```bash
       uv run python scripts/clipping.py
   ```

---

## Baseline Cross-Sucking Detector

A baseline script for detecting cross-sucking behaviour in calves using YOLO bounding box overlap. For each input video, the script produces a JSON metadata file containing the time windows where cross-sucking may have occurred, along with the per-frame intersection box coordinates of the overlapping region.

### How It Works

1. Loads a pretrained YOLO26 model by default (the model works on COCO dataset which detects `cow` class as a proxy for calves)
2. Loads one video at a time (this setting might be changed in the future)
3. Reads every Nth frame as set by `--frame_skip` (default: 1 = every frame). This will be seen while running `baseline.py`
4. For each frame, detects all calves and identifies the pair with the highest overlap (IoU). If IoU ≥ threshold, flags the frame and records that pair's intersection box.
5. Groups consecutive flagged frames into events and filters out events shorter than a minimum duration
6. Saves a JSON metadata file with the flagged events and their intersection box coordinates

**Notes:**

1. The COCO-pretrained model was trained on adult cattle outdoors. Detection accuracy will improve significantly once fine-tuned on your own labelled calf footage
2. The model, IoU threshold, confidence threshold, and number of skipped frames can be changed into other baseline models using arguments which will be described below
3. The argument `skip_frame` allows to skip N number of frames at a time (e.g. `skip_frame = 5` means processing every 5 frames instead of just 1 frame each)

### Input (baseline)

| Property | Details |
|---|---|
| Format | `.mp4`, `.avi`, or any format supported by OpenCV |
| Content | Single video clip of calves in a pen |

### Output (baseline)

Results are saved to `results/metadata/baseline/` automatically.

**File:** `results/metadata/baseline/<video_name>_results.json`

**Example output:**

```json
{
  "identifier": "pen3_morning.mp4",
  "video_path": "/data/videos/pen3_morning.mp4",
  "model": "yolo26m.pt",
  "iou_threshold": 0.1,
  "conf_threshold": 0.5,
  "min_duration_sec": 1.0,
  "fps": 25.0,
  "total_frames": 7500,
  "total_duration_sec": 300.0,
  "frame_skip": 1,
  "cross_sucking_detected": true,
  "num_events": 2,
  "events": [
    {
      "start_sec": 12.4,
      "end_sec": 19.1,
      "duration_sec": 6.7,
      "avg_confidence": 0.71,
      "intersection_box": [
        { "frame": 310, "x1": 290, "y1": 95, "x2": 340, "y2": 280 },
        { "frame": 311, "x1": 291, "y1": 96, "x2": 341, "y2": 281 }
      ]
    }
  ]
}
```

**Output fields:**

| Field | Description |
|---|---|
| `identifier` | Video filename — used as the unique ID for this clip |
| `model` | YOLO model name used for detection |
| `iou_threshold` | IoU threshold used to flag overlapping boxes |
| `conf_threshold` | YOLO detection confidence threshold |
| `min_duration_sec` | Minimum event duration in seconds (shorter events are filtered out) |
| `fps` | Frames per second of the video |
| `total_frames` | Total number of frames in the video |
| `frame_skip` | Frame skip value (1 = every frame, 2 = every 2nd frame, etc.) |
| `cross_sucking_detected` | `true` if at least one event was flagged |
| `num_events` | Total number of flagged events |
| `start_sec` / `end_sec` | Start and end time of the event in seconds |
| `duration_sec` | Length of the event in seconds |
| `avg_confidence` | Average YOLO detection confidence across all frames in the event |
| `intersection_box` | Per-frame pixel coordinates of the overlapping region for downstream annotation |

### How to run the baseline

1. Using the default parameters:

    ```bash
    uv run scripts/baseline/baseline.py --video "<path/to/video.mp4>"
    ```

2. Using custom parameters:

    ```bash
    uv run baseline.py --video "<path/to/video.mp4>" \
                    --iou_threshold <insert_threshold> \
                    --conf_threshold <insert_threshold> \
                    --min_duration <insert_in_seconds> \
                    --model <insert_model_name> \
                    --frame_skip <insert_integer> 
    ```

**Note:** If your file path contains spaces, wrapping it in quotes will avoid shell parsing errors. The file path should look like `"\Users\mickeymouse\data\video_cross_sucking.mp4"`

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video` | str | - | Path to input video file (required) |
| `--model` | str | `yolo26x.pt` | YOLO model weights filename e.g. `yolo26n.pt`, `yolo26m.pt`, `yolo26l.pt` |
| `--iou_threshold` | float | `0.1` | Minimum IoU overlap to flag a frame. Must be between `0.0` and `1.0` |
| `--conf_threshold` | float | `0.5` | Minimum YOLO detection confidence to keep a box. Must be between `0.0` and `1.0` |
| `--min_duration` | float | `1.0` | Minimum duration in seconds a continuous overlap must last to be flagged as an event |
| `--frame_skip` | int | `1` | Process every Nth frame. Higher values are faster but may miss short events |

### Tuning tips

- **`--iou_threshold`** — lower values (e.g. `0.05`) catch more events but increase false positives. Raise it if you're getting too many flags from calves just standing close together
- **`--conf_threshold`** — raise if the model is detecting non-calf objects. Lower if calves are being missed in difficult lighting
- **`--min_duration`** — raise if brief accidental box overlaps are being flagged. A value of `1.0–2.0` seconds works well as a starting point
- **`--frame_skip`** — raise for faster processing on long videos (5–15 is a good range). Keep at `1` if you need precise event boundaries or are looking for very short events

#### Demo Examples

For demonstration purposes, we provide 2 clip samples each for cross-sucking and non-cross-sucking examples (each video ~15-17s).

- Cross-sucking examples:

    1. Example 1 (~2-3 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --video "sample_videos/cross_sucking_clip_sample/CS_0276_WEAN_d1_p2_cowT_16102025_ch02-20251016124717_19033_19045.mp4"
        ```

    2. Example 2 (~4-5 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --video "sample_videos/cross_sucking_clip_sample/CS_0002_POSTWEAN_d1_p2_cow3_02112025_ch02-20251103001956_60818_60835.mp4"
        ```

- Non-cross-sucking examples

    1. Example 1 (~1-2 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --video "sample_videos/non_cross_sucking_clip_sample/ch05_20251114073451_15s.mp4"
        ```

    2. Example 2 (~3-4 minutes):

        ```bash
        uv run scripts/baseline/baseline.py --video "sample_videos/non_cross_sucking_clip_sample/ch04_20250828075551_15s.mp4"
        ```
---

## Evaluation

The evaluation script compares baseline model predictions against ground truth annotations to measure detection performance across two levels — event level and sequence level.

### How It Works

1. Loads all baseline prediction JSON files from the results directory
2. Loads the ground truth annotations from the processed clips index CSV
3. Matches predictions to ground truth events using temporal IoU per video
4. Computes frame-level bounding box IoU across all predicted frames independently of temporal matching
5. Computes precision, recall, F1, F2 and bounding box IoU at the event level
6. Computes temporal IoU at the sequence level
7. Stratifies all results by pen and weaning stage
8. Saves a full evaluation report as a JSON file

### Input

| Property | Details |
|---|---|
| Predictions | Directory of baseline JSON files from `results/metadata/baseline/` |
| Ground truth | Processed clips index CSV from `data/raw/all_clips_index_raw.csv` |

### Output

Results are saved to the path specified by `--output`.

**Output fields:**

| Field | Description |
|---|---|
| `true_positives` | Predicted events that correctly match a ground truth event |
| `false_positives` | Predicted events with no matching ground truth event |
| `false_negatives` | Ground truth events the model missed |
| `precision` | Fraction of flagged events that were correct |
| `recall` | Fraction of real events that were found |
| `f1` | Balanced average of precision and recall |
| `f2` | Recall-weighted score — prioritizes not missing real events |
| `avg_bbox_iou` | Average spatial overlap between predicted and ground truth boxes |
| `avg_temporal_iou` | Average time window overlap between predictions and ground truth |
| `frame_level_bbox_iou` | Average spatial accuracy of bounding boxes across ALL predicted frames, independent of event matching |
| `labelled_clips_dir` | Path to CVAT annotation zip files — required for bbox IoU computation |
| `by_pen` | All metrics broken down by pen |
| `by_weaning_stage` | All metrics broken down by weaning stage |

### How to Run

1. Using default thresholds:

```bash
    uv run python scripts/evaluation.py \
        --predictions results/metadata/baseline/ \
        --ground_truth data/raw/all_clips_index_raw.csv \
        --output results/evaluation_report.json
```

2. Using custom thresholds:

```bash
    uv run python scripts/evaluation.py \
        --predictions results/metadata/baseline/ \
        --ground_truth data/raw/all_clips_index_raw.csv \
        --output results/evaluation_report.json \
        --confidence_threshold 0.6 \
        --temporal_iou_threshold 0.4
```

3. With bbox IoU using CVAT annotations:

```bash
    uv run python scripts/evaluation.py \
    --predictions results/metadata/baseline/ \
    --ground_truth data/raw/all_clips_index_raw.csv \
    --output results/evaluation_report.json \
    --labelled_clips_dir /path/to/cross_sucking_labelled
```


### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--predictions` | str | - | Directory containing baseline JSON prediction files (required) |
| `--ground_truth` | str | - | Path to processed clips index CSV (required) |
| `--output` | str | - | Path to save evaluation report JSON (optional) |
| `--confidence_threshold` | float | `0.5` | Minimum confidence score to consider a prediction |
| `--temporal_iou_threshold` | float | `0.5` | Minimum temporal IoU to count a prediction as a match |
| `--labelled_clips_dir` | str | None | Path to CVAT annotation zip files for bbox IoU. Optional |
| `--fps` | float | `30` | Frames per second of source videos |

### Tuning Tips

- **`--temporal_iou_threshold`** — lower values (e.g. `0.3`) are more lenient about timing overlap. Raise if you want stricter matching
- **`--confidence_threshold`** — raise if too many low confidence predictions are being counted. Lower if the model is being too conservative

---

## Clipping

Running the clipping script (reproduce event clips)

### Overview

Reproduces short clips from long-form videos using baseline JSON event metadata and writes them to a structured folder under REPRODUCED_CLIPS_DIR.

### Input / Output

Input: JSON metadata in LOCAL_DIR/results/metadata/baseline referencing source videos.
Output: .mp4 clips in ROOT_DIR/reproduced_clips (optionally with _boxed.mp4 variants).

### Prerequisites

Make sure your .env is set correctly (see above).
Make sure baseline metadata exists locally at: `mooVision/results/metadata/baseline`
This path is derived from `LOCAL_DIR` in `.env`: `BASELINE_METADATA_DIR = LOCAL_DIR / "results" / "metadata" / "baseline"`

### How to run

From the repo root:

```bash
uv run python scripts/clipping.py
```
  
Outputs are written to the directory configured in config.py:
REPRODUCED_CLIPS_DIR = ROOT_DIR / "reproduced_clips"

## Gemini API Configuration

This project requires a Gemini API key to run multimodal inference via the `google-genai` SDK.

### 1. Generate Your Key
1. Navigate to **[Google AI Studio](https://aistudio.google.com/)**.
2. Sign in using your Google developer account.
3. Click the **Get API Key** button in the dashboard interface.
4. Select or create a project workspace, click **Create API Key**, and copy the string (it will start with `AIzaSy`).

### 2. Configure Environment Variables
Create a `.env` file in the root directory of your project (or open your existing one) and add your copied key variable:

```env
GEMINI_API_KEY="YOUR_COPIED_AIZASY_KEY_HERE"
