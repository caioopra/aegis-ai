---
name: phase-status
description: Show the current AegisNode implementation phase status by reading the checkbox progress tracker in notes/PLAN.md. Use at session start or when the user asks "what's next".
allowed-tools: Read, Bash
---

# Phase Status — AegisNode

Quickly surface where the project stands in the multi-phase plan.

## Steps

1. Read the progress tracker section at the top of `notes/PLAN.md`
   (typically the first ~60 lines — a list of phase checkboxes).

   ```bash
   head -80 notes/PLAN.md
   ```

2. From the checkbox list, identify:
   - **Completed phases** — lines starting with `- [x]`.
   - **In-progress phase** — the most recent phase with partial
     checkboxes, or the earliest pending phase if none in progress.
   - **Next pending phases** — the next 2-3 `- [ ]` lines.

3. Report in this shape:

   ```
   ## AegisNode Phase Status

   Last completed: Phase N — <short title>
   Current / Next: Phase M — <short title>
     - outstanding steps: <count or list>
   Upcoming (blocked by current):
     - Phase X, Phase Y

   Run /phase-status again after each commit to track progress.
   ```

4. If the user asks for details on a specific phase, read the
   corresponding `## Phase N` section from `notes/PLAN.md`.

## What NOT to do

- Do not modify `notes/PLAN.md` from this skill. Phase completion is
  marked manually after commit + review.
- Do not guess progress from git history — trust the checkboxes.
- Do not include the full plan body in the status report; keep it
  scannable.
