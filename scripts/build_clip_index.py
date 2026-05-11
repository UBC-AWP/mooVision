import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  
from config import ROOT, METADATA_DIR
from utils.get_video import collect_all_metadata

output_path = METADATA_DIR / "clips_metadata.csv"
total_start = time.time()

print("Scanning videos...")
df, failed = collect_all_metadata(ROOT)
total_elapsed = time.time() - total_start

actual_count = len(list(ROOT.rglob("*.mp4")))
print(f"\nFilesystem count:  {actual_count}")
print(f"Scanned count:     {len(df)}")
print(f"Failed:            {len(failed)}")
print(f"Total time:        {total_elapsed:.2f}s")

df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")

if failed:
    failed_log = METADATA_DIR / "failed_clips.txt"
    failed_log.write_text("\n".join(str(f) for f in failed))
    print(f"Failed list saved to {failed_log}")
else:
    print("No failed clips")