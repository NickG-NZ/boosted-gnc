# Agent guidelines

## Python style

- **All imports go at the top of the file.** Never import inside a function, method, or mid-module
  (e.g. no `from .gfold import check_solution` inside a function body). If an inline import seems
  needed to break a circular import, restructure the modules instead.
- Imports are sorted with isort (`[tool.isort]` in `pyproject.toml`).

Both rules are enforced by the pre-commit hook (`.pre-commit-config.yaml`: isort + ruff `E402`/`PLC0415`).
Set it up once per clone with `uv sync && uv run pre-commit install`; run it manually with
`uv run pre-commit run --all-files`.
