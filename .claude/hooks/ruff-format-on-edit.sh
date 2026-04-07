#!/usr/bin/env bash
# Auto-format a Python file immediately after Claude Edit/Write.
# Safe: ruff format is idempotent and scoped to a single file.
# Non-blocking: exits 0 even if ruff is missing or the file is non-Python.

set -u

# The Claude Code hook passes tool input as JSON on stdin.
# We extract the target file_path field with jq if available, or grep fallback.
if ! command -v jq >/dev/null 2>&1; then
    # jq missing — skip silently rather than breaking edits.
    exit 0
fi

file_path="$(jq -r '.tool_input.file_path // empty')"

# Nothing to do if the field is empty.
if [[ -z "${file_path}" ]]; then
    exit 0
fi

# Only format Python files.
case "${file_path}" in
    *.py) ;;
    *) exit 0 ;;
esac

# Only format files inside this project's tracked source trees.
case "${file_path}" in
    */clinical_agent/src/*|*/clinical_agent/tests/*|*/clinical_agent/scripts/*) ;;
    *) exit 0 ;;
esac

# Run ruff format on the specific file, discarding output so we don't
# clutter Claude's tool-result stream. Never block on failure.
uv run ruff format "${file_path}" >/dev/null 2>&1 || true

exit 0
