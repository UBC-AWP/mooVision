from pathlib import Path

ROOT = Path("/Users/raymondwang/Library/CloudStorage/OneDrive-SharedLibraries-UBC/Animal Welfare-mooVision - Documents/cross_sucking_clips")
DATA_DIR = ROOT.parent / "data"
METADATA_DIR = DATA_DIR / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)