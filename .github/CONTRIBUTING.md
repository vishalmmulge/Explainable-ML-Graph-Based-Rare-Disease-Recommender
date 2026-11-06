# Contributing

Thank you for contributing to the Explainable Graph-Based Rare Disease Recommendation System!

## How to Contribute

1. Fork the repository and create your branch from `master`.
2. Set up a virtual environment and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Make your changes and run the tests:

   ```powershell
   pytest tests/test_pipeline.py -v
   ```

4. Lint your code:

   ```powershell
   pip install ruff
   ruff check .
   ruff format .
   ```

5. Open a pull request against `master`. Reference the related issue in the PR description.

## Code Style

- Follow PEP 8.
- Use ruff for linting and formatting.
- Add docstrings to public functions and classes.
- Add tests for new functionality.

## Medical Disclaimer

This project is a research and educational prototype. Do not contribute code intended for clinical use without appropriate validation and review.
