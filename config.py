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

ROOT_DIR = require_dir("ROOT_DIR")

# Path on OneDrive shared library (syncs with local)
ROOT_DIR = Path(os.environ["ROOT_DIR"])
CLIPS_DIR = ROOT_DIR / "cross_sucking_clips"
RAW_DIR = ROOT_DIR / "raw_cross_sucking_datalog"
CLIPS_INDEX_PATH = CLIPS_DIR / "all_clips_index.csv"
DATA_FOLDER_DIR = ROOT_DIR / "data"
CLIPS_METADATA_DIR = DATA_FOLDER_DIR / "clips_metadata"
CLIPS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
