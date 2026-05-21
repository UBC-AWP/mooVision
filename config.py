from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

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

UNLABELLED_CLIPS_DIR = ROOT / "cross_sucking_clips"
LABELLED_CLIPS_DIR = ROOT / "cross_sucking_labelled"
SOURCE_VIDEOS_DIR = ROOT / "raw_cross_sucking_datalog" / "videos"

INDEX_PATH = UNLABELLED_CLIPS_DIR / "all_clips_index.csv"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
