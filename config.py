from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(os.environ["MOOVISION_CLIPS_DIR"])
TESTROOT = Path(os.environ["TEST_DIR"])
RAW_DIR = Path(os.environ["RAW_DIR"])

INDEX_PATH = ROOT / "all_clips_index.csv"
DATA_DIR = ROOT.parent / "data"
METADATA_DIR = DATA_DIR / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)