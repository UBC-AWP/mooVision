# Testing

MooVision uses `pytest` for unit testing. Tests are located in the `tests/` directory and are organized by module.

## Running Tests

To run all tests:
```bash
uv run pytest tests/ -v
```

To run tests for a specific module:
```bash
uv run pytest tests/test_read_all_clips_index.py -v
uv run pytest tests/test_evaluation.py -v
uv run pytest tests/test_splitting.py -v
uv run pytest tests/test_clipping.py -v
uv run pytest tests/models/test_baseline.py -v
uv run pytest tests/preprocessing/test_preprocessing_yolo.py -v
uv run pytest tests/training/test_training_yolo.py -v
```

To run a specific test class:
```bash
uv run pytest tests/test_read_all_clips_index.py::TestReadData -v
```

To run a specific test:
```bash
uv run pytest tests/test_read_all_clips_index.py::TestReadData::test_file_not_found_raises -v
```

## Test Structure

```
tests/
├── test_read_all_clips_index.py   # Tests for data reading and index processing
├── test_evaluation.py             # Tests for evaluation metrics
├── test_splitting.py              # Tests for data splitting
├── test_clipping.py               # Tests for clip generation
├── test_run_testing_2.py          # Tests for inference pipeline
├── matching/                      # Tests for clip name matching utilities
├── models/                        # Tests for baseline and model scripts
├── preprocessing/                 # Tests for preprocessing and frame extraction
└── training/                      # Tests for YOLO training script
```

## Writing New Tests

Tests follow the class-based style with one class per function being tested. Fixtures are defined at the top of each class under `# --- Fixtures ---` and tests are grouped by `# --- Happy path ---`, `# --- Error cases ---`, and `# --- Edge cases ---`.

```python
class TestMyFunction:

    # --- Fixtures ---

    @pytest.fixture
    def sample_input(self):
        return ...

    # --- Happy path ---

    def test_returns_expected_output(self, sample_input):
        result = my_function(sample_input)
        assert result == expected

    # --- Error cases ---

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            my_function(invalid_input)
```