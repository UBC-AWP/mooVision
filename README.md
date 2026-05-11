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