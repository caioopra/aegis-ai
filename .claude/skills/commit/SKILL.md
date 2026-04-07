---
name: commit
description: Full pre-commit workflow for AegisNode — runs pre-commit-check, then spawns security-review and/or clinical-review based on the staged diff, then creates the commit only if every gate passes. Use instead of raw git commit for non-trivial changes.
allowed-tools: Bash, Read
argument-hint: "[commit message]"
---

# Commit Workflow — AegisNode

A structured pre-commit gate that runs tests, lint, and targeted
reviews before creating a commit. The commit message (if provided
in $ARGUMENTS) is used; otherwise, draft one from the diff.

## Workflow

### Stage 1 — Sanity check

1. Run `git status` to confirm there are staged changes.
2. Run `git diff --cached --stat` to see the scope.
3. If no staged changes, stop and tell the user to stage files first.

### Stage 2 — Automated gates (pre-commit-check)

Invoke the `pre-commit-check` skill (or run the same commands
directly):

- `uv run pytest -m "not integration" -q`
- `uv run ruff check src/ tests/ scripts/`
- `uv run ruff format src/ tests/ scripts/`

If any gate fails: STOP. Report the failure, name the specialist who
should fix it (see pre-commit-check skill), and do NOT create the
commit.

### Stage 3 — Targeted reviews

Based on which files are staged, spawn reviewers in parallel:

| Trigger files | Review to run |
|---|---|
| `src/aegis/llm.py`, `src/aegis/config.py`, `.env*`, any new outbound HTTP | `security-review` skill |
| `src/aegis/llm.py` (prompts), `src/aegis/mcp_server.py` (interactions), `src/aegis/agent/nodes.py` (safety nets), `data/guidelines/`, `data/synthea/sample_*` | `clinical-review` skill |
| Pure test / infra / config | No targeted review needed |

Run reviews in parallel when both apply. Wait for both to complete.

If either review returns a **Blocker** / **Critical** finding: STOP.
Report the findings to the user and do NOT create the commit. Let the
user decide whether to fix and re-run, or override with an explicit
instruction.

### Stage 4 — Create the commit

Only if every gate and review passed (or reviews returned only
Informational notes):

1. If `$ARGUMENTS` is non-empty, use it as the commit message.
   Otherwise, draft a concise 1-2 line message in imperative mood,
   based on the diff and the conventions in recent commits
   (`git log --oneline -10`).
2. Run:

   ```bash
   git commit -m "$(cat <<'EOF'
   <your message>

   Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
   EOF
   )"
   ```

3. Run `git log --oneline -3` and report the new commit SHA and
   message to the user.

## What NOT to do

- Do NOT run `git commit --no-verify`. If a hook fails, the workflow
  stops and the user decides.
- Do NOT use `git commit --amend`. Create a new commit and let the
  user decide whether to squash later.
- Do NOT push. This skill stops at the commit — pushing is a
  separate, explicit user action.
- Do NOT bypass a clinical or security blocker because "the tests
  pass". Blockers are blockers until the user overrides them.

## Override

The user can override a review blocker by explicitly saying
"commit anyway" or "override the review and commit". In that case,
include a `[review-override]` trailer in the commit message so the
override is traceable in git history.
