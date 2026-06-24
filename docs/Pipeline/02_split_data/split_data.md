# Splitting Data

This page documents how to use `scripts/splitting.py` to create train/val/test splits for model training, and the methodology for data splitting experiments in the MooVision project.

---

## Summary

To answer different research questions about variable with fundamental impacts on a cross-sucking detection model, this project creates eight different train/val/test split. These are as follows:

1. Random shuffle split
2. Time-based split
3. Pen-based split (x3)
4. Period-based split (x3)

To remove redundancy, splits are create from the processed index file and held as train, val, and test csv files in the processed data folder. These csv files hold relative paths to video and labels, allowing downstream functionality for creating training sets of videos in each split.

### Random Shuffle Split

**Answers:** How well does the model perform when training and test data come from very similar distributions?

**Splitting Approach:** Randomly Shuffle clips into train, val and test (70% train, 20% val, 10% test)

Both train and test clips will contain clips from different pens, days, and periods (preweaning, weaning, and postweaning). e expect a model trained on this split to overfit due to leaked spatial information on pens, calf appearances, and behavioural tendencies. This will likely give the highest performance, but may overestimate real-world generalization. 

### Time-based split

**Answers:** How well does the model work when the deployment environment looks similar to training, but on unseen time periods?

**Splitting Approach:** Train/test split within the same pens and periods, but grouped by time period.

Example:

Train: 70% of the total time period (Day 1 + Day 2 + first 8640s of Day 3).  
Val: 20% of total time period (Day 3: 8640s - 60480s)
Test: 10% of total time period (Day 3 60480s - end)

This reduces leakage from highly similar clips recorded on the same day and provides a more realistic evaluation. This aims to test model performance in an established environment. We expect this model may overfit to the pen environments due to leaked spatial information on pens, calf appearances, and calf behaviours. Moreover, while this split was chosen to provide more training data for the model, we expect this may overfit to validation time periods. 86402 corresponds to about 02:30 AM, and 60480s to about 5:00PM. This may leave time periods will differing lighting conditions and behavioural environments that could affect model performance.

Another appraoch which gives less training data to the model is to provide a split based fully on days: train (day 1), val (day 2), test (day 3). This was not implemented but is a suggestion for future studies.

### Pen-based split (new environment generalization)

**Answers:** How do lighting/background/environmental/behavioural differences across pens affect performance?

**Splitting Approach:** Withhold one pen for evaluation, and rotate through each version of this holdout.

Example:

- Train Pen3; Val: Pen2; Test Pen5
- Train Pen2; Val: Pen5; Test Pen3
- Train Pen5; Val: Pen3; Test Pen2

This approach eliminates leakage of spatial and behavioural information from training clips to testing clips. This simulates model performance on a new farm, answering how well does the model generalize to a new environment. We note that camera angles are fairly consistent and pens exist in controlled environment, thus results may not indicate performance on new farm environments.

### Period-based split (behavioural/age generalization)

**Answers:** Does the model generalize across calf age, developmental stage, or behavioural context?

**Splitting Approach:** Withhold one period for evaluation, and rotate through each version of this holdout

Example:

- Train: postweaning; Val: weaning; Test: preweaning
- Train: weaning; Val: preweaning; Test: postweaning
- Train: preweaning; Val: postweaning; Test: weaning

We expect that frequency and characteristics of cross-sucking behaviour may differ across periods, and this could affect model performance. For instance, calf growth and changing appearances may result in the inability of learned edges (etc.) to transfer across weaning periods, resulting in poor model performance. Moreover we expect behaviour habits to change over calf growth and weaning periods leading to additional instabilities for the model.

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
    ├── processed_clips_index.csv
    ├── random/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── day_based/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── pen_based/
    │   ├── Pen_2/
    │   │   ├── train.csv
    │   │   ├── val.csv
    │   │   └── test.csv
    │   ├── Pen_3/
    │   │   ├── train.csv
    │   │   ├── val.csv
    │   │   └── test.csv
    │   └── Pen_5/
    │       ├── train.csv
    │       ├── val.csv
    │       └── test.csv
    └── period_based/
        ├── PREWEANING/
        │   ├── train.csv
        │   ├── val.csv
        │   └── test.csv
        ├── WEANING/
        │   ├── train.csv
        │   ├── val.csv
        │   └── test.csv
        └── POSTWEANING/
            ├── train.csv
            ├── val.csv
            └── test.csv
```

Train, val, and test csv's must contain relative paths to cross-sucking clips, source videos, and annotated outputs. These are used to find appropriate videos for model training when needed.

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

::: scripts.data_splitting.split_data
    options:
        show_source: false
        show_root_heading: true