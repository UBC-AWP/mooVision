# Testing

MooVision uses `pytest` for unit testing. Tests are located in the `tests/` directory and are organized by module.

## Running Tests

To run all tests:
```bash
uv run pytest tests/ -v
```

To run tests for a specific module:
```bash
uv run pytest tests/clipping/test_clipping.py -v
uv run pytest tests/evaluation/test_evaluation.py -v
uv run pytest tests/split_data/test_splitting.py -v
uv run pytest tests/read_data/test_read_all_clips_index.py -v
uv run pytest tests/run_models/test_baseline.py -v
uv run pytest tests/run_models/test_run_testing.py -v
uv run pytest tests/preprocessing/unit_tests/test_preprocessing_yolo.py -v
uv run pytest tests/preprocessing/integration_tests/test_preprocessing_yolo_integration.py -v
uv run pytest tests/training/test_training_yolo.py -v
```

To run a specific test class:
```bash
uv run pytest tests/read_data/test_read_all_clips_index.py::TestReadData -v
```

To run a specific test:
```bash
uv run pytest tests/read_data/test_read_all_clips_index.py::TestReadData::test_file_not_found_raises -v
```

## Test Structure

```
tests/
├── clipping/
│   └── test_clipping.py                          # Tests for clip generation
├── evaluation/
│   └── test_evaluation.py                         # Tests for evaluation metrics
├── preprocessing/
│   ├── integration_tests/
│   │   ├── test_extract_frames_integration.py     # Integration tests for frame extraction
│   │   └── test_preprocessing_yolo_integration.py # Integration tests for YOLO preprocessing
│   └── unit_tests/
│       ├── test_extract_frames.py                 # Unit tests for frame extraction
│       ├── test_extract_labels.py                 # Unit tests for label extraction
│       ├── test_preprocessing_yolo.py             # Unit tests for YOLO preprocessing
│       └── test_utils.py                          # Unit tests for preprocessing utilities
├── read_data/
│   ├── matching/
│   │   ├── test_extract_numeric_id.py             # Tests for numeric ID extraction
│   │   ├── test_find_match.py                     # Tests for match finding
│   │   ├── test_is_match.py                       # Tests for match validation
│   │   ├── test_parse_labelled_name.py            # Tests for parsing labelled names
│   │   └── test_parse_unlabelled_names.py         # Tests for parsing unlabelled names
│   └── test_read_all_clips_index.py               # Tests for data reading and index processing
├── run_models/
│   ├── test_baseline.py                           # Tests for baseline model
│   └── test_run_testing.py                        # Tests for inference pipeline
├── split_data/
│   └── test_splitting.py                          # Tests for data splitting
└── training/
    └── test_training_yolo.py                      # Tests for YOLO training script
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