# Sockeye HPC

This page documents how to run the full mooVision pipeline on the [Sockeye HPC cluster](https://arc.ubc.ca/compute-storage/ubc-arc-sockeye) at UBC ARC. The pipeline takes raw video footage, preprocess the videos, trains a YOLO model to detect cross-sucking behaviour, runs inference, and evaluates predictions against ground-truth labels.

!!! warning "Case-sensitive paths"
    The repo is `mooVision` (capital V) but the scratch data directory is `moovision` (all lowercase). Double-check which one you need before running any command.

## Pipeline Overview

The scripts are numbered in the order they should be run:

## Pipeline Overview

| Script | Purpose | Output |
|--------|---------|--------|
| `01_setup.sh` | Install dependencies and configure environment (run once) | `.venv/` environment ready |
| `02_read_and_split_data.sh` | Create train/val/test splits (per pen, no leakage) | `data/processed/processed_clips_index.csv` |
| `03_preprocessing.sh` | Convert videos into YOLO format (frames + labels, train/val split, filtering orphan frames) | `dataset/images/{train,val}/`, `dataset/labels/{train,val}/`, `dataset.yaml` (or `dataset.tar` on Sockeye) |
| `04_train_yolo.sh` | Fine-tune YOLO model on training split | `data/yolo_training_runs/split_X_model/weights/best.pt` |
| `05_baseline_testing.sh` | Run IoU-threshold baseline inference | `results/metadata/<split_strategy>/baseline/` |
| `06_model_testing.sh` | Run YOLO + Seq-NMS inference (array jobs across splits/chunks) | `results/metadata/<split_strategy>/yolo/`, `seq-nms/` |
| `07_baseline_evaluation.sh` | Evaluate baseline predictions vs ground truth | `results/evaluation/evaluation_report_<split_strategy>_baseline.json` |
| `08_model_evaluation.sh` | Evaluate YOLO + Seq-NMS predictions vs ground truth | `results/evaluation/evaluation_report_<split_strategy>_yolo.json`, `evaluation_report_<split_strategy>_seq-nms.json` |

!!! note
    All data outputs (except .venv which in in the repo mooVision/ folder) are stored on Sockeye under: `/scratch/st-nina-1/moovision/` (case-sensitive: lowercase "moovision")

---

## Prerequisites

Before running the pipeline, ensure:

- You have a Sockeye account.
- You have access to the `st-nina-1` project.
- The raw dataset has been copied to `/scratch/st-nina-1/moovision/`.
- You can SSH into Sockeye.

---

## Connecting to Sockeye (SSH)

You can connect to the Sockeye HPC cluster using SSH from your local machine.

### Basic login

```bash
ssh <your_cwl>@sockeye.arc.ubc.ca
```

Example:

```bash
ssh jsmith@sockeye.arc.ubc.ca
```

---

### First-time login notes

* You may be prompted for your CWL password.
* You may need to complete UBC ARC two-factor authentication (2FA).
* The first connection may ask you to confirm the host fingerprint — type `yes`.

---

## Connecting

```bash
SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes <your_cwl>@sockeye.arc.ubc.ca
```

Once connected, navigate to the repo and set up your environment:

```bash
cd /scratch/st-nina-1/mooVision
source .env_sockeye
uv sync
```

---

## Running the Whole Pipeline

```bash
sbatch run_training_pipeline.sh
```

---

## 01 — Setup

**Script:** `01_setup.sh`

Installs Python dependencies into the shared virtual environment using `uv`. This only needs to be run once when first setting up the project on Sockeye, or after adding new dependencies to `pyproject.toml`.

```bash
sbatch 01_setup.sh
```

!!! note
    `uv sync` must be run on a **login node**, not a compute node, because compute nodes have no internet access. The SLURM scripts handle this correctly — do not run `uv sync` inside a job that is already on a compute node.

---

## 02 — Read and Split Data

**Script:** `02_read_and_split_data.sh`

Reads the raw clip index and partitions it into train, validation, and test sets. Splits are done **per pen** to prevent data leakage — clips from the same pen never appear in both training and test sets.

```bash
sbatch 02_read_and_split_data.sh
```

Output is written to `data/processed/`, including `processed_clips_index.csv`. The generated processed_clips_index.csv serves as the canonical ground-truth file used by both inference and evaluation steps.

---

## 03 — Preprocessing

**Script:** `03_preprocessing.sh`

Converts raw video clips into the frame-level format expected by YOLO. This includes extracting frames, resizing, and generating YOLO-format label files.

```bash
sbatch 03_preprocessing.sh
```
 
This is typically the most time-consuming step. Check job progress with:

```bash
watch squeue -u $USER
```

---

## 04 — Train YOLO

**Script:** `04_train_yolo.sh`

Fine-tunes a pretrained YOLO model on the mooVision training data. One model is trained per data split. Trained weights are saved to:

```
data/yolo_training_runs/split_N_model/weights/best.pt
```

```bash
sbatch 04_train_yolo.sh
```

!!! note
    Training requires a GPU node and will queue until one is available. Expect longer wait times during peak hours.

---

## 05 — Baseline Testing

**Script:** `05_baseline_testing.sh`

Runs the **baseline** inference pipeline, which detects cross-sucking using IoU overlap thresholds and minimum duration filtering — no fine-tuned model is involved. This serves as a performance lower bound to compare against the trained YOLO model.

```bash
sbatch 05_baseline_testing.sh
```

Output is written to `results/metadata/<split_strategy>/baseline/`.

!!! note
    The baseline script does not overwrite existing results. To rerun the script and generate new results, delete the existing output files first.

## 06 — Model Testing (YOLO & Seq-NMS Inference)

**Script:** `06_model_testing.sh`

Runs the fine-tuned YOLO model across all splits as a SLURM array job (8 splits × 10 chunks = 80 tasks). Each task processes a subset (10% chunk) of the test clips for one split.

Submit all model/split combinations as a SLURM array job:

```bash
sbatch 06_model_testing.sh
```

To overwrite existing results:

```bash
sbatch --export=ALL,OVERWRITE=true 06_model_testing.sh
```

To test on just the first two array tasks before a full run:

```bash
OVERWRITE=true sbatch --array=0-1 06_model_testing.sh
```

You can also run inference manually for a single split:

```bash
uv run scripts/run_testing_2.py \
  --model_path "data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt" \
  --data_path "data/processed/pipeline_demo/test.csv" \
  --chunk 0 \
  --chunk_pct 1.0
```

!!! note
    You can try running manually using Interactive GPU session (see below) for faster result.

Output is written to `results/metadata/<split_strategy>/yolo/`.

---

## 07 — Baseline Evaluation

**Script:** `07_baseline_evaluation.sh`

Evaluates the baseline predictions from step 05 against the ground-truth labels. Computes frame-level metrics (precision, recall, F1) by matching predicted and labelled cross-sucking events using IoU.

```bash
sbatch 07_baseline_evaluation.sh
```

Or run manually:

```bash
uv run python scripts/evaluation.py \
  --predictions "results/metadata/pipeline_demo/baseline/" \
  --ground_truth data/processed/processed_clips_index.csv \
  --output results/evaluation_report_baseline.json \
  --labelled_clips_dir "cross_sucking_labelled"
```

Output reports are written to `results/evaluation/`.

---

## 08 — Model Evaluation (YOLO & Seq-NMS)

**Script:** `08_model_evaluation.sh`

Evaluates the fine-tuned YOLO predictions from step 06 against ground truth. Runs across all 8 splits and both pipeline variants (yolo, seq-nms).

Once inference is done, submit the evaluation job:

```bash
sbatch 08_model_evaluation.sh
```

Or run manually:

```bash
uv run python scripts/evaluation.py \
  --predictions "results/metadata/pipeline_demo/yolo/" \
  --ground_truth data/processed/processed_clips_index.csv \
  --output results/evaluation_report_yolo.json \
  --labelled_clips_dir "cross_sucking_labelled"

uv run python scripts/evaluation.py \
    --predictions "results/metadata/pipeline_demo/seq-nms/" \
    --ground_truth data/processed/processed_clips_index.csv \
    --output results/evaluation_report_seq_nms.json \
    --labelled_clips_dir "cross_sucking_labelled"
```

Predictions live under `results/metadata/<split_strategy>/yolo/` and `results/metadata/<split_strategy>/seq-nms/`, e.g.:

```
results/metadata/random/yolo
results/metadata/random/seq-nms
results/metadata/period_based/pen_2/yolo
results/metadata/period_based/pen_2/seq-nms
```

---

## Managing Jobs

### Monitoring jobs

```bash
squeue -u $USER          # view active jobs
watch squeue -u $USER    # auto-refresh every 2 seconds
scancel <jobid>          # cancel a specific job
scancel -u $USER         # cancel all your jobs
```

For more detailed job accounting:

```bash
sacct -j <jobid>         # job history and exit status
seff <jobid>             # efficiency report (CPU / memory usage)
```

---

### Viewing logs (very important for debugging)

Each SLURM job writes output logs by default.

```bash
cat /scratch/st-nina-1/mooVision/logs/<jobid>.out
```

```bash
cat /scratch/st-nina-1/mooVision/logs/<jobid>.err
```

If a job fails or behaves unexpectedly, always check:

* error messages in the `.out` and `.err` file
* whether the job ran out of memory or time
* whether paths or environment setup failed

---

### Expected runtime (approximate)

Pipeline steps vary in runtime depending on dataset size and GPU availability:

| Step                      | Expected runtime                   |
| ------------------------- | ---------------------------------- |
| Data splitting (`02`)     | ~1–5 minutes                            |
| Preprocessing (`03`)      | ~1–3 hours (CPU-heavy)                  |
| YOLO training (`04`)      | ~2–10+ hours per split (GPU)            |
| Baseline inference (`05`) | ~30 min – 2 hours (GPU)                 |
| Model inference (`06`)    | ~1–6 hours (depends on array size, GPU) |
| Evaluation (`07–08`)      | ~5–20 minutes                           |

If a job runs significantly longer than expected, it may be stuck or misconfigured.

---

### Debugging failed jobs checklist

If something fails, check:

* `squeue` → is it still running?
* `sacct` → did it fail or timeout?
* `.out` logs → error messages
* file paths → correct `mooVision` vs `moovision`
* resources → memory or GPU availability


### Interactive GPU session (for testing)

This interactive session is useful for rapid testing and debugging, as it allows you to run scripts manually and iterate on changes without waiting for batch jobs to start.

```bash
salloc --account=st-nina-1 --job-name=inference_testing \
  --time=01:00:00 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gpus=1
```

!!! note
    Omit `--gpu=1` unless you specifically need a GPU — GPU nodes have longer queue times.

---

## Syncing Results to OneDrive

Run these from your **local Mac**, not from Sockeye.

```bash
# Evaluation reports
rsync -avz c<your_cwl>ea@sockeye.arc.ubc.ca:/scratch/st-nina-1/moovision/results/evaluation/ \
  '/Users/<username>/Library/CloudStorage/OneDrive-SharedLibraries-UBC/Animal Welfare-mooVision - Documents/results/evaluation'

# Prediction metadata
rsync -avz <your_cwl>@sockeye.arc.ubc.ca:/scratch/st-nina-1/moovision/results/metadata/ \
  '/Users/<username>/Library/CloudStorage/OneDrive-SharedLibraries-UBC/Animal Welfare-mooVision - Documents/results/metadata'

# Training weights (splits 2–9)
for split in 2 3 4 5 6 7 8 9; do
  rsync -avz <your_cwl>@sockeye.arc.ubc.ca:/scratch/st-nina-1/moovision/data/yolo_training_runs/split_${split}_model \
    '/Users/<username>/Library/CloudStorage/OneDrive-SharedLibraries-UBC/Animal Welfare-mooVision - Documents/data/yolo_training_runs/'
done
```

---

## Useful File Operations

```bash
ls /scratch/st-nina-1/moovision/results/metadata        # check inference output
ls /scratch/st-nina-1/moovision/results/evaluation      # check evaluation output

# Delete a results folder to force a clean re-run (destructive!)
rm -rf /scratch/st-nina-1/moovision/results/metadata/random # delete the whole random/ folder
```

