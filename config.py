from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(os.environ["MOOVISION_CLIPS_DIR"])
TESTROOT = Path(os.environ["TEST_DIR"])

DATA_DIR = ROOT.parent / "data"
METADATA_DIR = DATA_DIR / "metadata"

UNLABELLED_CLIPS_DIR = ROOT / "cross_sucking_clips"
LABELLED_CLIPS_DIR = ROOT / "cross_sucking_labelled"
SOURCE_VIDEOS_DIR = ROOT / "raw_cross_sucking_datalog" / "videos"

INDEX_PATH = UNLABELLED_CLIPS_DIR / "all_clips_index.csv"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
