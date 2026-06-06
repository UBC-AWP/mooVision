import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

from config import LOCAL_DIR, ROOT_DIR, UNLABELLED_CLIPS_DIR

# model scripts live in a models/ folder next to this script
MODELS_DIR = Path(__file__).parent / "models"

# all test.csv files live somewhere under here
PROCESSED_DIR = Path(LOCAL_DIR) / "data" / "processed"

# all JSON results go here (on OneDrive)
RESULTS_DIR = Path(ROOT_DIR) / "results" / "metadata"


# command line to pick a model
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="name of the model script in models/")
parser.add_argument("--test", action="store_true", help="run only pipeline_testing/ instead of all splits")
args = parser.parse_args()

model_script = MODELS_DIR / f"{args.model}.py"
if not model_script.exists():
    print(f"ERROR: could not find {model_script}")
    sys.exit(1)


# find every test.csv under data/processed/
# rglob("test.csv") walks all subfolders and returns every test.csv it finds
if args.test:
    all_csvs = [
        csv for csv in sorted(PROCESSED_DIR.rglob("test.csv"))
        if "pipeline_testing" in csv.parts
    ]
else:
    all_csvs = [
        csv for csv in sorted(PROCESSED_DIR.rglob("test.csv"))
        if "pipeline_testing" not in csv.parts
    ]

print(f"found {len(all_csvs)} splits:")
for csv in all_csvs:
    # show the path relative to data/processed/ so it is easy to read
    print(f"  {csv.relative_to(PROCESSED_DIR)}")

