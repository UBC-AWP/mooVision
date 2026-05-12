# Contributing to MooVision

Thank you for your interest in contributing to MooVision! This document outlines
the process for contributing code, documentation, and reporting issues.

## Table of Contents
- [Reporting Issues](#reporting-issues)
- [Getting Started](#getting-started)
- [Branch Naming](#branch-naming)
- [Making Changes](#making-changes)
- [Pull Requests](#pull-requests)
- [Documentation](#documentation)

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear, descriptive title
- A description of the problem or suggestion
- Steps to reproduce the issue (if applicable)
- Any relevant error messages or screenshots

## Getting Started

1. Fork the repository and clone it locally
2. Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Run `uv sync` to install all dependencies
4. Set up your local `.env` file:
```bash
cp .env.example .env
```
5. Edit `.env` and set your local paths:
```
MOOVISION_CLIPS_DIR=/path/to/your/clips
TEST_DIR=/path/to/your/test/data
```

## Branch Naming

Please follow this naming convention for branches:

| Type | Format | Example |
|------|--------|---------|
| New feature | `feature/short-description` | `feature/clip-indexing` |
| Bug fix | `fix/short-description` | `fix/confidence-threshold` |
| Documentation | `docs/short-description` | `docs/update-readme` |
| Testing | `test/short-description` | `test/baseline-inference` |

## Making Changes

- Keep changes focused — one feature or fix per branch
- Follow existing code style and structure
- Place library code in `src/` and runnable scripts in `scripts/`
- Add or update tests in `tests/` for any new functionality
- Run tests before submitting a PR:
```bash
uv run pytest
```
- Do not commit your `.env` file — update `.env.example` if you add new
environment variables

## Pull Requests

1. Push your branch to GitHub and open a pull request against `main`
2. Give your PR a clear title and description of what changed and why
3. Request a review from at least one team member
4. Address any review comments before merging
5. PRs should pass all tests before being merged

## Documentation

- Update the `README.md` if your changes affect setup or usage
- Add docstrings to any new functions or modules
- Update `docs/` with any relevant documentation changes
- For report changes, work within the `report/` directory