import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  
from config import ROOT

def print_folder_counts(folder, indent=0):
    mp4_count = sum(1 for f in folder.rglob("*.mp4"))
    prefix = "  " * indent
    print(f"{prefix}{folder.name:<50} {mp4_count:>6}")
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            print_folder_counts(child, indent + 1)

print(f"{'Folder':<60} {'MP4s':>6}")
print("-" * 68)
for child in sorted(ROOT.iterdir()):
    if child.is_dir():
        print_folder_counts(child)
        print()
