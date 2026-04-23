from pathlib import Path

root = Path("/Users/skysheng/Library/CloudStorage/OneDrive-UBC/Animal Welfare-mooVision - Documents/cross_sucking_clips")

def print_folder_counts(folder, indent=0):
    mp4_count = sum(1 for f in folder.rglob("*.mp4"))
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
