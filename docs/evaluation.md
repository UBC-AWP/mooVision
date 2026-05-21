# Evaluation

This module evaluates the MooVision baseline cross-sucking detection pipeline
by comparing model predictions against ground truth annotations.

## Usage

```bash
python src/evaluation.py \
    --predictions results/metadata/baseline/ \
    --ground_truth data/raw/all_clips_index_raw.csv \
    --output results/evaluation_report.json
```

## Functions

::: src.evaluation