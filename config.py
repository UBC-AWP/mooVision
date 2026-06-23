from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


def require_dir(var: str) -> Path:
    """Get a directory path from an environment variable, ensuring it exists."""
    val = os.getenv(var)
    if not val:
        raise RuntimeError(
            f"Missing {var}. Put it in .env or export it, e.g. {var}=/path/to/root"
        )
    p = Path(val).expanduser().resolve()
    if not p.is_dir():
        raise RuntimeError(f"{var} must be an existing directory. Got: {p}")
    return p


load_dotenv()

# Path on OneDrive shared library (syncs with local)
ROOT_DIR = require_dir("ROOT_DIR")

# Path on GitHub / local workspace
LOCAL_DIR = require_dir("LOCAL_DIR")
BASELINE_METADATA_DIR_LOCAL = LOCAL_DIR / "results" / "metadata" / "baseline"
METADATA_DIR_CLOUD = ROOT_DIR / "results" / "metadata" 

# Videos and labels
UNLABELLED_CLIPS_DIR = ROOT_DIR / "cross_sucking_clips"
LABELLED_CLIPS_DIR = ROOT_DIR / "cross_sucking_labelled"
SOURCE_VIDEOS_DIR = ROOT_DIR / "raw_cross_sucking_datalog" / "videos"

# Raw and Processed Index Paths
INDEX_PATH = UNLABELLED_CLIPS_DIR / "all_clips_index.csv"
PROCESSED_INDEX = ROOT_DIR / "data" / "processed" / "processed_clips_index.csv"

# Data / metadata
DATA_FOLDER_DIR = ROOT_DIR / "data"
EVALUATION_DATA_DIR = DATA_FOLDER_DIR / "results" 
BASELINE_MODEL_OUTPUT_DIR = DATA_FOLDER_DIR / "results" / "metadata" / "baseline"
METADATA_DIR = DATA_FOLDER_DIR / "clips_metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Output clips
RESULT_CLIPS_DIR = ROOT_DIR  / "results" / "result_clips"
RESULT_CLIPS_DIR.mkdir(parents=True, exist_ok=True)




