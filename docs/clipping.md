# Clipping (Video Reproduction) 

This page documents how to use `scripts/clipping.py` to reproduce short video clips from longer source videos, using either:

- a **CSV index** (`split_by_index`), or
- **JSON event metadata** (`split_by_json_events`, optionally with bounding-box annotation)

---

## Configuration (`config.py` + `.env`)

This project uses a root-level `config.py` to centralize filesystem paths. It loads environment variables from a local `.env` file using `python-dotenv` (`load_dotenv()`).

### `.env` requirements

Your `.env` must define these variables:

- **`ROOT_DIR`**: Path to the OneDrive/shared-library root directory (must exist and be a directory)
- **`LOCAL_DIR`**: Path to your local project/work directory (must exist and be a directory)

Example `.env`:

```bash
ROOT_DIR=/path/to/your/root/directory
LOCAL_DIR=/path/to/your/local/directory
```

### Validation behavior (`require_dir`)

`config.py` uses:

- `require_dir(var: str) -> Path`

to validate required environment variables:

- If an env var is missing, it raises `RuntimeError` with instructions to add it to `.env` or export it.
- If the env var points to a non-existent directory, it raises `RuntimeError`.

### Paths exported by `config.py` (used by `clipping.py`)

#### Inputs
- **`RAW_DIR`** = `ROOT_DIR / "raw_cross_sucking_datalog"`  
  Root directory scanned recursively for raw `*.mp4` sources.

- **`CLIPS_INDEX_DIR`** = `ROOT_DIR / "cross_sucking_clips" / "all_clips_index.csv"`  
  CSV index consumed by `split_by_index(...)`.

- **`BASELINE_METADATA_DIR`** = `LOCAL_DIR / "results" / "metadata" / "baseline"`  
  JSON metadata input consumed by `split_by_json_events(...)` (directory of `*.json` or a single JSON file).

#### Outputs (created if missing)
- **`REPRODUCED_CLIPS_DIR`** = `ROOT_DIR / "reproduced_clips"`  
  Output root where reproduced clips are written.

### How the script wires config into function calls

When running through `run_splitting(...)`, arguments come from `config.py`:

- Index mode:
  ```python
  split_by_index(CLIPS_INDEX_DIR, REPRODUCED_CLIPS_DIR)
  ```
  (`RAW_DIR` is used internally to locate raw source videos.)

- JSON events mode:
  ```python
  split_by_json_events(BASELINE_METADATA_DIR, REPRODUCED_CLIPS_DIR)
  ```

---

## Running the script

By default, `main()` runs JSON-events splitting:

```bash
python clipping.py
```

If you want to run index mode, update `main()` to:

```python
run_splitting(split_by_index)
```

(or call `split_by_index(...)` directly in a custom script).

---

## Output folder structure (JSON events mode)

Each JSON metadata file includes a `"video_path"` like:

`.../cross_sucking_clips/Pen 2 - Group 2/POSTWEANING/Day 1/<video>.mp4`

`split_by_json_events(...)` recreates the directory structure **after** `cross_sucking_clips/` under your output root.

So clips are written to:

`REPRODUCED_CLIPS_DIR/Pen 2 - Group 2/POSTWEANING/Day 1/`

Importantly: **there is no extra per-video folder**; all clips for all source videos in the same `Day 1` folder go into that same output folder.

---

# Function reference

## reproduce_clip

```python
reproduce_clip(raw_video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> bool
```

Extract a clip from `raw_video_path` between `start_sec` and `end_sec` (seconds) and save it to `output_path`.

### Parameters

**raw_video_path** : pathlib.Path  
Path to the source `.mp4` video.

**start_sec** : float  
Start time in seconds.

**end_sec** : float  
End time in seconds.

**output_path** : pathlib.Path  
Output `.mp4` file path.

### Returns

**success** : bool  
`True` if written successfully, else `False`.

### Notes

- Uses OpenCV video I/O (`cv2.VideoCapture`, `cv2.VideoWriter`), codec `"mp4v"`.
- Frame boundaries are computed from the source FPS.
- Caller should ensure `output_path.parent` exists.

### Examples

```python
from pathlib import Path

ok = reproduce_clip(
    raw_video_path=Path("input.mp4"),
    start_sec=10.0,
    end_sec=15.5,
    output_path=Path("out/clip.mp4"),
)
print(ok)
```

---

## split_by_index

```python
split_by_index(index_path: Path, output_path: Path) -> None
```

Reproduce clips based on a CSV index file.

### Parameters

**index_path** : pathlib.Path  
Path to index CSV (typically `CLIPS_INDEX_DIR` from `config.py`).

**output_path** : pathlib.Path  
Output directory (typically `REPRODUCED_CLIPS_DIR` from `config.py`).

### Returns

None

### Notes

- Reads the CSV with `pandas.read_csv`.
- Scans `RAW_DIR` recursively for `.mp4` files and matches rows by `source_video_basename`.
- Current code processes only the first matched row (`matched[:1]`). Remove `[:1]` to process all rows.

### Expected CSV columns

At minimum:

- `source_video_basename`
- `source_video_path`
- `clip_name`
- `clip_start_in_source_sec`
- `clip_end_in_source_sec`

### Examples

```python
from config import CLIPS_INDEX_DIR, REPRODUCED_CLIPS_DIR
split_by_index(CLIPS_INDEX_DIR, REPRODUCED_CLIPS_DIR)
```

---

## split_by_json_events

```python
split_by_json_events(json_path: Path, output_dir: Path, annotate: bool = True) -> int
```

Reproduce clips based on event metadata in JSON files and optionally annotate the output with bounding boxes.

### Parameters

**json_path** : pathlib.Path  
Either:
- a single JSON file, or
- a directory containing `*.json` (**only directly in that directory, non-recursive**)

Typically comes from `BASELINE_METADATA_DIR` in `config.py`.

**output_dir** : pathlib.Path  
Output root (typically `REPRODUCED_CLIPS_DIR` from `config.py`).

**annotate** : bool, default=True  
Whether to also write `__boxed.mp4` annotated outputs.

### Returns

**total_success** : int  
Total number of successfully reproduced (unboxed) clips across all processed JSON files.

### JSON schema

Required keys:

- **`video_path`** : str  
  Source video `.mp4` path.

Optional keys:

- **`identifier`** : str  
  Naming prefix (defaults to `video_path.name`).
- **`fps`** : float  
  Used for annotation alignment (defaults to `30.0`).
- **`events`** : list[dict]  
  Event list; each event must include:
  - `start_sec`
  - `end_sec`

Optional per-event:

- **`intersection_box`** : list[dict]  
  Bounding boxes with:
  - `frame` (in **source-video** frame coordinates)
  - `x1`, `y1`, `x2`, `y2`

### Output naming

For event `i`:

- `<identifier_stem>__event{i:03d}_{start_sec:.1f}-{end_sec:.1f}.mp4`
- If `annotate=True`:
  `<identifier_stem>__event{i:03d}_{start_sec:.1f}-{end_sec:.1f}__boxed.mp4`

### Examples

```python
from config import BASELINE_METADATA_DIR, REPRODUCED_CLIPS_DIR

n = split_by_json_events(BASELINE_METADATA_DIR, REPRODUCED_CLIPS_DIR, annotate=True)
print(n)
```

---

## annotate_clip_with_boxes

```python
annotate_clip_with_boxes(
    input_clip: Path,
    output_clip: Path,
    boxes: list[dict],
    color=(0, 255, 0),
    thickness: int = 2,
) -> bool
```

Annotate a clip with bounding boxes and write an annotated clip.

### Parameters

**input_clip** : pathlib.Path  
Input clip path.

**output_clip** : pathlib.Path  
Output clip path.

**boxes** : list[dict]  
Boxes in **clip-relative** frame coordinates. Each box must include:
- `frame`
- `x1`, `y1`, `x2`, `y2`

**color** : tuple[int, int, int], default=(0, 255, 0)  
Rectangle color in BGR.

**thickness** : int, default=2  
Rectangle thickness in pixels.

### Returns

**success** : bool  
`True` if successful, else `False`.

### Notes

- Creates `output_clip.parent` if needed.
- Multiple boxes per frame are supported.

### Examples

```python
from pathlib import Path

boxes = [{"frame": 0, "x1": 10, "y1": 20, "x2": 100, "y2": 120}]
ok = annotate_clip_with_boxes(Path("clip.mp4"), Path("clip__boxed.mp4"), boxes)
print(ok)
```

---

# Troubleshooting

### Missing env vars / invalid paths
If you see errors like:

- `Missing ROOT_DIR. Put it in .env or export it...`
- `ROOT_DIR must be an existing directory...`

Check:

1. You have a `.env` file in your current working directory (or exported env vars).
2. `ROOT_DIR` and `LOCAL_DIR` point to real directories on your machine.
3. OneDrive/Shared Library is synced locally if you reference it via `ROOT_DIR`.