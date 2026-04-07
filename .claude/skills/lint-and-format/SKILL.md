---
name: lint-and-format
description: Run ruff check and ruff format on src/, tests/, and scripts/. Use before committing or when the user asks about style/lint status.
allowed-tools: Bash
---

# Lint & Format — ruff

## What to do

Run the two ruff commands in order and report the combined result:

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
```

## Reporting

- If `ruff check` passes AND `ruff format` reports "N files left
  unchanged" (no reformatting): report "All checks passed, no
  formatting changes needed."
- If `ruff format` reformatted any file: report the file count and
  which files. Formatting changes are safe to include in the current
  commit — they're idempotent.
- If `ruff check` finds violations: DO NOT auto-fix them with
  `--fix`. Surface the violations grouped by rule code and let the
  relevant specialist decide. (Auto-fix can silently rewrite logic
  the author didn't intend.)

## Ruff config

Configured in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py310"
line-length = 100
```

Do not change the config in this skill — delegate to
**infra-deployment-engineer** for any rule or config changes.
