---
name: clinical-review
description: Review staged changes for clinical correctness, pt-BR terminology, drug doses, SNOMED/LOINC coding, and Brazilian guideline adherence. Delegates to the medical-clinical-expert sub-agent.
allowed-tools: Bash, Read
---

# Clinical Review — Staged Diff

Delegate to the **medical-clinical-expert** sub-agent for a content
review of the current staged diff.

## Steps

1. Collect the staged changes focused on clinical content:

   ```bash
   git diff --cached --stat
   git diff --cached -- \
     'src/aegis/llm.py' \
     'src/aegis/mcp_server.py' \
     'src/aegis/agent/nodes.py' \
     'data/guidelines/' \
     'data/synthea/sample_patient_*.json'
   ```

2. If the filtered diff is empty (no clinical content touched),
   report "No clinical content in the staged diff — skipping."

3. Spawn the `medical-clinical-expert` sub-agent via the Agent tool
   with a prompt that includes:
   - The file list.
   - The filtered diff.
   - An instruction to follow its own review format (Critical /
     Important / Nice to have / Not a problem).
   - A reminder that the target is Brazilian clinicians and the
     primary language is pt-BR.

4. Surface the review verdict to the user verbatim. Do not rewrite
   clinical advice.

## What counts as "clinical content"

- Prompt few-shot examples (especially `ENTITY_EXTRACTION_PROMPT`
  and `REPORT_PROMPT`).
- Drug interaction dict in `src/aegis/mcp_server.py`.
- Safety net entity types / allergy checks in `src/aegis/agent/nodes.py`.
- The `AI_DISCLAIMER` wording.
- Guideline `.txt` files under `data/guidelines/`.
- FHIR sample patient JSON files under `data/synthea/`.

## What is NOT clinical content

- LangGraph state schema / reducer annotations.
- BM25 hashing, chunk size, RRF weights.
- Provider abstraction / config loading.
- Pure test infrastructure.

For non-clinical changes, skip this skill or delegate to the
appropriate technical specialist instead.
