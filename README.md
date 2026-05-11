# MooVision

MooVision is a computer-vision pipeline for detecting **cross-sucking behaviour** in socially housed dairy calves from angled overhead pen video. The goal is to reduce the time and effort required for manual review/labeling by automatically producing candidate events, clipped video segments, and structured metadata for downstream analysis.

## Project Overview

Cross-sucking (here: sucking directed at various body parts of other calves) is a welfare concern in group-housed calves and is currently studied via manual labeling of long video recordings. This project builds a scalable workflow that:

- takes raw pen video as input
- runs a baseline detector (pretrained YOLOv8) with simple logic on top (e.g., proximity/overlap + temporal persistence)
- outputs predicted event windows and metadata (start/end time, confidence, pen, weaning stage, day)
- optionally generates clipped videos for review and evaluation

## Repository Structure (high level)

- `src/`: library code (config, preprocessing, baseline inference, evaluation)
- `scripts/`: runnable entry points (run baseline, build clip index, etc.)
- `eda/`: EDA notebooks
- `tests/`: unit tests
- `docs/` : project documentation
- `report/`: report assets

## Environment Setup

This project uses `uv` for package management.

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Clone the repo and cd into it
3. Run `uv sync` to install all dependencies
4. Run scripts with `uv run python <script.py>`

## Local `.env` configuration (required)

We use a local `.env` file (stored at the **repo root**) to configure machine-specific paths (e.g., where the video clips live). This avoids hardcoding absolute paths in code.

1. Create a `.env` file at the repo root:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and set your local data path, for example:
   ```bash
   MOOVISION_DATA_ROOT=/Users/<you>/path/to/data_root
   ```
3. `.env` is ignored by git (do not commit). If you need to change what variables exist, update `.env.example` instead.

## Outputs (baseline)

The baseline pipeline is intended to produce a structured predictions file (CSV/JSON) with event windows such as:

- `video_id`
- `start_time_s`, `end_time_s`
- `confidence`
- `pen`, `weaning_stage`, `day`