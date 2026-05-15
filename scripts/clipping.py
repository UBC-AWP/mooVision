import sys
import time
import argparse
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
from config import RAW_DIR, INDEX_PATH

index_df = pd.read_csv(INDEX_PATH)

# get all raw videos in RAW_DIR
raw_videos = {f.name: f for f in RAW_DIR.rglob("*.mp4")}

matched = index_df[index_df["source_video_basename"].isin(raw_videos.keys())]
print(f"Found {len(matched)} clips to reproduce from {matched['source_video_basename'].nunique()} raw videos\n")

