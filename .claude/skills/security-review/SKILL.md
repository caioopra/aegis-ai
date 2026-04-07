---
name: security-review
description: Review staged git changes for LGPD compliance, prompt injection risk, PII leakage, secrets, and trust boundaries before committing. Delegates to the security-privacy-reviewer sub-agent.
allowed-tools: Bash, Read
---

# Security Review — Staged Diff

Delegate to the **security-privacy-reviewer** sub-agent with a focused
review of the current staged diff.

## Steps

1. Collect the staged diff and file list:

   ```bash
   git diff --cached --stat
   git diff --cached
   ```

2. If the diff is empty, report "No staged changes to review." and stop.

3. Spawn the `security-privacy-reviewer` sub-agent via the Agent tool
   with a prompt that includes:
   - The list of changed files.
   - The exact `git diff --cached` output (trimmed if > 1500 lines).
   - An explicit instruction to follow its own review format (Blocker
     / Important / Informational / Verified clean).

   Example prompt skeleton:

   > You are reviewing the following staged changes for AegisNode
   > before commit. Focus on LGPD compliance, prompt injection risk,
   > PII leakage, and secrets. Return your review in the standard
   > format defined in your agent definition.
   >
   > Files changed:
   > <git diff --cached --stat output>
   >
   > Full diff:
   > <git diff --cached output>

4. Surface the review verdict to the user verbatim. Do NOT rewrite or
   summarize the blocker list — the user needs to see the exact
   file:line callouts.

## When to use this skill

- Before any commit that touches `src/aegis/llm.py` (prompt injection
  surface).
- Before any commit that touches `data/synthea/` or `data/guidelines/`
  (patient data / guideline poisoning).
- Before any commit that touches `src/aegis/config.py` or `.env*`.
- Before any commit that adds new outbound HTTP calls or MCP tools.

## Skip this skill only when

- The diff is purely test-only and doesn't touch production code paths.
- The diff is purely documentation (`*.md` with no secrets).
- The user explicitly says "skip security review".
