# Splitting Data

This page documents how to use `scripts/splitting.py` to create train and test splits for model training, and the methodology for data splitting in the MooVision project.

---

## Summary

To answer different research questions about the effectiveness of a cross-sucking detection model, this project creates four different train test splits by default. These are as follows:

1. Random shuffle split
2. Day-based split
3. Pen-based split
4. Period-based split

Since video files are large, splits are created as train and test csv's from the ground truth data file. These csv files must hold relative paths to both cross-sucking clips and annotated labels within their respective folders, allowing access to select video files for later training.

### Random Shuffle Split

**Answers:** How well does the model perform when training and test data come from very similar distributions?

**Splitting Approach:** Randomly Shuffle clips into train and test.

Both train and test clips will contain clips from different pens, days, and periods (preweaning, weaning, and postweaning). This will likely give the highest performance, but may overestimate real-world generalization.

### Day-based split

**Answers:** How well does the model work when the deployment environment looks similar to training, but on unseen days?

**Splitting Approach:** Train/test split within the same pens and periods, but grouped by day.

Example:

Train: Day 1 and Day 2 from each pen/period.  
Test: Day 3 from each pen/period.

This reduces leakage from highly similar clips recorded on the same day and provides a more realistic evaluation. This aims to test model performance in an established environment.

### Pen-based split (new environment generalization)

**Answers:** How do lighting/background/environmental/behavioural differences across pens affect performance?

**Splitting Approach:** Withhold one pen for evaluation, and rotate through each version of this holdout.

Example:

- Train Pen2 + Pen3 and Test Pen5
- Train Pen2 + Pen5 and Test Pen3
- Train Pen3 + Pen5 and Test Pen2

This approach eliminates leakage of spatial and behavioural information from training clips to testing clips. This simulates model performance on a new farm, answering how well does the model generalize to a new environment.

### Period-based split (behavioural/age generalization)

**Answers:** Does the model generalize across calf age, developmental stage, or behavioural context?

**Splitting Approach:** Withhold one period for evaluation, and rotate through each version of this holdout

Example:

- Train: preweaning + weaning; Test: postweaning
- Train: weaning + postweaning; Test: preweaning
- Train: preweaning + postweaning; Test: weaning

We expect that frequency and characteristics of cross-sucking behaviour may differ across periods, and this could affect model performance.

---

## Inputs (Data requirements)

This script requires a preprocessed data file on which to perform data spliting. This data must follow the data requirements laid out in `Data Requirements`. This file can be created by running

```{bash}
uv run scipts/read_all_clips_index.py
```

in the terminal, or by defining your own preprocessed data file and passing it as an argument when running the script. (See Specifying a Data Path)

---

## Outputs 

Data is split into train and test csv files rather than datasets with videos to save space and remove redundancy.

As a default, this script outputs `train.csv` and `test.csv` files locally in folders corresponding to different split types within `data/processed/`. The default output structure is as follows. 

```{bash}
data/
└── processed/
    ├── random/
    │   ├── train.csv
    │   └── test.csv
    ├── day_based/
    │   ├── train.csv
    │   └── test.csv
    ├── pen_based/
    │   ├── Pen_2/
    │   │   ├── train.csv
    │   │   └── test.csv
    │   ├── Pen_3/
    │   │   ├── train.csv
    │   │   └── test.csv
    │   └── Pen_5/
    │       ├── train.csv
    │       └── test.csv
    └── period_based/
        ├── PREWEANING/
        │   ├── train.csv
        │   └── test.csv
        ├── WEANING/
        │   ├── train.csv
        │   └── test.csv
        └── POSTWEANING/
            ├── train.csv
            └── test.csv
```

Train and test csv's must contain relative paths to cross-sucking clips, source videos, and annotated outputs. These are used to find appropriate videos for model training when needed.

---

## Usage

### Basic Usage

The script can be run from the terminal using the following command.

```{bash}
uv run scripts/splitting.py
```

By default, this will run all train/test splits, but will not overwrite folders or files if they already exist. 

### Overwriting Files

To overwrite files use the `--FORCE` argument.

```{bash}
uv run scripts/splitting.py --FORCE
```

### Running Specific Splits

If you do not want to run a specific split you can pass the arguments: `--no_{type}_split` where type is one of "random", "day", "period", "pen". For example,

```{bash}
uv run scripts/splitting.py --FORCE --no_random_split --no_pen_split
```

will run period-based and day-based splits, but it will not run a random shuffle split, nor a pen-based split.

### Specifying Output Directory

By default, train and test csv's are saved to folders in `data/processed/`. If you want to specifiy a new output directory you can do so using the `--output_dir=` argument.

```{bash}
uv run scripts/splitting.py --FORCE --output_dir="data/splits/"
```

### Specifying a Data Path

If you want to specify your own processed data file, you can pass this to the script with the following arguemnt.

```{bash}
uv run scripts/splitting.py --data_path="/path/to/your/data_file"
```

Note: If you are creating your own processed data file, the data must adhere to data requirements layed out in `Project Organization`.

---

## Function Reference

::: scripts.splitting
    options:
        show_source: false
        show_root_heading: true