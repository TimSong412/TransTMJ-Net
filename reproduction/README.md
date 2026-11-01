# Distribution builder

Use `run_evaluation.py` with authorized external `--manifest` and `--data-root`
arguments. See the root README for input requirements and private data handling.

`python3 reproduction/build_package.py --check` verifies the explicit allowlist
and file checksums. `--output /path/outside/this/folder/classifier_public.zip`
creates a ZIP at an explicitly supplied external destination. Private inputs,
runtime outputs, hidden files and unlisted files are excluded.

This is an inference package. Training history is not included. Package
verification does not establish the validity of the underlying study.
