# Contributing to Traceprop

Thank you for your interest in contributing.

## Getting started

```bash
git clone https://github.com/AmitoVrito/Traceprop.git
cd Traceprop
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                   # all tests
pytest tests/unit/       # unit tests only
pytest tests/integration # integration tests only
pytest --cov=traceprop   # with coverage
```

## Submitting changes

1. Open an issue first to discuss the change.
2. Fork the repo and create a branch from `main`.
3. Write tests for any new behaviour.
4. Ensure `pytest` passes with no failures.
5. Open a pull request with a clear description of what and why.

## Code style

- Follow existing patterns in the codebase.
- No external formatter is enforced; keep lines under 100 characters where reasonable.
- Type annotations are encouraged for public API functions.

## Reporting bugs

Open a GitHub issue with:
- Python version and OS
- Traceprop version (`python -c "import traceprop; print(traceprop.__version__)"`)
- Minimal reproducing example
- Full traceback

## License

By contributing you agree that your contributions will be licensed under the Apache 2.0 License.
