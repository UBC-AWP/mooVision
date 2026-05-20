from pathlib import Path
import os
from dotenv import load_dotenv

def require_dir(var: str) -> Path:
    """ Get a directory path from an environment variable, ensuring it exists."""
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"Missing {var}. Put it in .env or export it, e.g. {var}=/path/to/root")
    p = Path(val).expanduser().resolve()
    if not p.is_dir():
        raise RuntimeError(f"{var} must be an existing directory. Got: {p}")
    return p

load_dotenv()  

# Path on OneDrive shared library (syncs with local)
ROOT_DIR = require_dir("ROOT_DIR")

# Path for clips and index
CLIPS_DIR = ROOT_DIR / "cross_sucking_clips"
CLIPS_INDEX_DIR = CLIPS_DIR / "all_clips_index.csv"

# Path for raw videos
RAW_DIR = ROOT_DIR / "raw_cross_sucking_datalog"

# Path for data folder 
DATA_FOLDER_DIR = ROOT_DIR / "data"

# Directory to save the metadata of the clips
CLIPS_METADATA_DIR = DATA_FOLDER_DIR / "clips_metadata"
CLIPS_METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Path to the output of the reproduced clips
REPRODUCED_CLIPS_DIR = ROOT_DIR / "reproduced_clips"
REPRODUCED_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Path on GitHub
LOCAL_DIR = require_dir("LOCAL_DIR")
# Path to baseline metadata output
BASELINE_METADATA_DIR = LOCAL_DIR / "results" / "metadata" / "baseline"