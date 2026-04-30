import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  
from config import TESTROOT, METADATA_DIR
from utils.video import collect_all_metadata

output_path = METADATA_DIR / "clips_metadata.csv"

print("Scanning videos...")
df = collect_all_metadata(TESTROOT)

print(f"\nTotal clips:     {len(df)}")

df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")