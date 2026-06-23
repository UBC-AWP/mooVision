"""
count number or labelled clips
"""

from pathlib import Path

path = (
    "/Users/devon/OneDrive_UBC/Animal Welfare-mooVision - Documents/cross_sucking_labelled"
)

root = Path(path)


def print_folder_counts(folder, indent=0):
    mp4_count = sum(1 for f in folder.rglob("*.zip"))
    prefix = "  " * indent
    print(f"{prefix}{folder.name:<50} {mp4_count:>6}")
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            print_folder_counts(child, indent + 1)


print(f"{'Folder':<60} {'MP4s':>6}")
print("-" * 68)
for child in sorted(root.iterdir()):
    if child.is_dir():
        print_folder_counts(child)
        print()
