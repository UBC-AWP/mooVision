# Research Questions and Data Spliting

We split our data based on several research questions. We avoid splitting on frames because we expect similar frames in the testing and training data to create lots of data leakage and over confident performance scores. Hence, we split at the clip level.

## Baseline Random Split

**Research Question:** How well does the model perform when training and test data come from very similar distributions?

**Splitting Approach:** Random Shuffle clips into train and test.

Both train and test clips will contain clips from different pens, days, and periods (preweaning, weaning, and postweaning). This will likely give the highest performance, but may overestimate real-world generalization.

## Day-based split

**Research Question:** How well does the model work when the deployment environment looks similar to training, but on unseen days?

**Splitting Approach:** Train/test split within the same pens and periods, but grouped by day.

Example:

Train: Day 1 and Day 2 from each pen/period. \\
Test: Day 3 from each pen/period.

This reduces leakage from highly similar clips recorded on the same day and provides a more realistic evaluation. This simulates a scenario in which the model is used at en established farm. We are not looking to generalize to a new environment, but rather test how well models performs once a system setup is stabalized.

## Pen-based split (leave-one-pen-out evaluation)

**Research Question:** How do lighting/background/environmental/behavioural differences across pens affect performance? 

**Splitting Approach:** Withhold one pen for evaluation, and rotate through each version of this holdout.

Example:

- Train Pen2 + Pen3 and Test Pen5
- Train Pen2 + Pen5 and Test Pen3
- Train Pen3 + Pen5 and Test Pen2

This approach eliminates leakage of spatial and behavioural information from training clips to testing clips. We aim to simulate model performance on a new farm. I.e. answering: how well does the model generalize to a new environment? We expect new camera angles, or environements included in training data would increase model performance.

## Period-based split (behavioural/age generalization)

**Research Question:** Does the model generalize across calf age, developmental stage, or behavioural context?

**Splitting Approach:** Withhold one period for evaluation, and rotate through each version of this holdout

Example:

- Train: preweaning + weaning; Test: postweaning
- Train: weaning + postweaning; Test: preweaning
- Train: preweaning + postweaning; Test: weaning

We expect that frequency and characteristics of cross-sucking behaviour may differ across periods, and this could affect model performance across weaning periods.