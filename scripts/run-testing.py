import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
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


for csv_path in all_csvs:

    # the split label is the folder path between data/processed/ and test.csv
    # e.g. data/processed/pen_based/pen_2/test.csv  ->  pen_based/pen_2
    split_label = csv_path.parent.relative_to(PROCESSED_DIR)

    print(f"\n--- split: {split_label} ---")

    # load this split's test.csv
    df = pd.read_csv(csv_path)
    # build full video paths and grab clip names
    video_paths = [Path(UNLABELLED_CLIPS_DIR) for p in df["clip_relative_path"]]
    clip_names  = df["clip_name"].tolist()

    # output folder mirrors the split label so results stay organised
    # e.g. results/metadata/yowo/pen_based/pen_2/
    output_dir = RESULTS_DIR / args.model / split_label
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_path, clip_name in zip(video_paths, clip_names):

        print(f"  running {clip_name} ...")
        print(f"  video path: {video_path}")

        # call the model script with the video path as the only argument
        # the model prints a JSON object to stdout, which we capture here
        result = subprocess.run(
            [sys.executable, str(model_script), "--video", str(video_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"    FAILED: {result.stderr[:200]}")
            continue

        # parse the JSON the model printed
        metadata = json.loads(result.stdout)

        # save it to results/metadata/{model}/{split}/{clip_name}.json
        out_file = output_dir / f"{clip_name}.json"
        out_file.write_text(json.dumps(metadata, indent=2))

        print(f"    saved -> {out_file}")