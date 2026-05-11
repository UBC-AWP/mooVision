from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(os.environ["MOOVISION_CLIPS_DIR"])
TESTROOT = Path(os.environ["TEST_DIR"])

DATA_DIR = ROOT.parent / "data"
METADATA_DIR = DATA_DIR / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)