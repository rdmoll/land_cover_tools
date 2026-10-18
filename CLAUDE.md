# Project: land_cover_tools

Python tools for processing EO imagery to generate high-resolution land-use/land-cover data and exporting GeoTIFFs. 

## Tech stack
- Python 3.13, managed with a virtual environment in `.venv/`
- Key libraries: numpy, scipy, pandas, matplotlib
- Dependencies are listed in `requirements.txt`. Add new ones there, don't just `pip install`.
- If `requirements.txt` does not exist yet create one which includes the key libraries.

## Common commands
- Set up environment: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run tests: `pytest -q`
- Lint / format: `ruff check . && ruff format .`
- Run the main script: `python -m src.main --config configs/default.yaml`

## Project layout
- `src/` – application code
- `tests/` – pytest tests, mirroring the `src/` structure
- `configs/` – YAML config files
- `notebooks/` – exploratory work only; don't put production logic here
- `data/` – local data, **git-ignored; never commit anything from here**

## Coding conventions
- Type hints on all public functions; NumPy-style docstrings
- Prefer small, pure functions; keep I/O separate from computation
- Use `pathlib` for paths, `logging` (not `print`) for output
- Vectorize with NumPy rather than writing Python loops over arrays

## Workflow rules
- Run the tests and linter before saying a task is done
- Write or update tests for any behavior you change
- Don't modify files in `data/` or anything listed in `.gitignore`
- Ask before adding a new dependency or changing the public API

## Git
- Default branch is `main`. For anything non-trivial, work on a feature branch named `feature/<short-description>` or `fix/<short-description>`
- Commit messages: short imperative summary line (≤ 72 chars), blank line, then details if needed
- Never commit secrets, API keys, `.env` files, or large data files
- Don't force-push or rewrite history on `main`
