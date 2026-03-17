"""Ollama client wrapper with prompt templates for clinical tasks."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import ollama

from aegis.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_MEDICAL = (
    "You are a clinical assistant AI. You help physicians by expanding "
    "medical abbreviations, extracting structured data from clinical notes, "
    "and generating medical reports. Always respond in the same language as "
    "the input note. Be precise and use standard medical terminology."
)

EXPAND_NOTE_PROMPT = """\
Expand the following doctor's shorthand note into a clear, complete clinical \
note. Expand all abbreviations, normalize vital signs, and list findings explicitly.

Doctor's note:
{note}

Return a JSON object with these fields:
- "expanded_note": the full expanded clinical note
- "entities": a list of extracted medical entities, each with "text", "type" \
(one of: symptom, sign, medication, condition, vital_sign, procedure), and "original" \
(the abbreviation or shorthand used)

Respond ONLY with valid JSON, no extra text.
"""

REPORT_PROMPT = """\
Generate a structured medical report based on the following information.

## Patient Data
{patient_data}

## Clinical Note
{note}

## Relevant Guidelines
{guidelines}

{refinement_context}

Return a JSON object with these sections:
- "patient_summary": brief patient description (1-2 sentences: age, sex, key conditions)
- "findings": list of clinical findings from the note and patient data
- "assessment": clinical assessment and reasoning (reference guidelines when applicable)
- "plan": recommended plan of care (specific actions, medications with doses if applicable)
- "guideline_references": list of guideline excerpts that support the plan

If any section above has no data available (marked as "Não disponível"), note this \
limitation explicitly in the assessment. Do not invent data that is not present.

Respond ONLY with valid JSON, no extra text.
"""

ENTITY_EXTRACTION_PROMPT = """\
Extract all medical entities from this clinical note.

Note:
{note}

Return a JSON object with a single field "entities" — a list of objects, each with:
- "text": the entity as stated in the note
- "type": one of symptom, sign, medication, condition, vital_sign, procedure
- "normalized": the standard medical term (expanded, in the note's language)

If no medical entities are found, return {{"entities": []}}.

Example for "Pct 65a HAS, PA 150x95, losartana 50mg":
{{
  "entities": [
    {{"text": "HAS", "type": "condition", "normalized": "Hipertensão arterial sistêmica"}},
    {{"text": "PA 150x95", "type": "vital_sign", "normalized": "Pressão arterial 150x95 mmHg"}},
    {{"text": "losartana 50mg", "type": "medication", "normalized": "Losartana 50 mg"}}
  ]
}}

Respond ONLY with valid JSON, no extra text.
"""

SELF_RAG_DECISION_PROMPT = """\
Given the following clinical note and the entities already extracted from it, \
decide whether retrieving clinical guidelines would improve the quality of the \
final medical report.

## Clinical Note
{note}

## Extracted Entities
{entities}

Answer with a JSON object:
- "needs_retrieval": true or false
- "queries": if true, a list of 1-3 short search queries to find relevant guidelines

Respond ONLY with valid JSON, no extra text.
"""

EVALUATE_REPORT_PROMPT = """\
Evaluate the quality of this medical report by comparing it against the original \
clinical note and patient data.

## Original Clinical Note
{note}

## Patient Data
{patient_data}

## Report to Evaluate
{report}

Score each dimension from 1-5 using this rubric:
- 1 = Critical gaps or errors that could affect clinical decisions
- 2 = Important information missing or inaccurate
- 3 = Acceptable — covers main points with minor gaps
- 4 = Good — comprehensive with only trivial omissions
- 5 = Excellent — thorough, accurate, well-referenced

Dimensions to score:
- "completeness": are all findings from the note addressed in the report?
- "accuracy": is the medical reasoning sound and consistent with the data?
- "guideline_adherence": does the plan follow the clinical guidelines provided?
- "clarity": is the report clear, well-structured, and actionable?
- "overall": overall quality score considering all dimensions

Return a JSON object with these fields, each containing "score" (int 1-5) and \
"feedback" (string with specific observations).

Respond ONLY with valid JSON, no extra text.
"""


# ---------------------------------------------------------------------------
# Client functions
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


def generate(prompt: str, system_prompt: str = SYSTEM_MEDICAL) -> str:
    """Send a prompt to Ollama and return the raw text response."""
    response = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.3},
    )
    return response["message"]["content"]


def generate_json(
    prompt: str,
    system_prompt: str = SYSTEM_MEDICAL,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Send a prompt and parse the response as JSON.

    Tries Ollama's native JSON format first; falls back to extracting
    a JSON block from the text response. Retries on transient failures
    with exponential backoff.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0.2},
            )
            return json.loads(response["message"]["content"])
        except (json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "JSON parse failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                exc,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "LLM call failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                exc,
            )

        if attempt < max_retries - 1:
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.info("Retrying in %.1fs...", delay)
            time.sleep(delay)

    # Final fallback: try without format=json and extract manually
    try:
        raw = generate(prompt, system_prompt)
        return _extract_json(raw)
    except Exception:
        raise ValueError(
            f"All {max_retries} attempts failed for generate_json. "
            f"Last error: {last_error}"
        ) from last_error


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM output."""
    # Try to find JSON between ```json ... ``` or { ... }
    import re

    # Try fenced code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Try raw JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


# ---------------------------------------------------------------------------
# High-level clinical functions
# ---------------------------------------------------------------------------


def expand_note(note: str) -> dict[str, Any]:
    """Expand a doctor's shorthand note into structured clinical data."""
    prompt = EXPAND_NOTE_PROMPT.format(note=note)
    return generate_json(prompt)


def extract_entities(note: str) -> dict[str, Any]:
    """Extract medical entities from a clinical note."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(note=note)
    return generate_json(prompt)


def generate_report(
    note: str,
    patient_data: str = "Not available",
    guidelines: str = "Not available",
    refinement_context: str = "",
) -> dict[str, Any]:
    """Generate a structured medical report from a clinical note and context."""
    prompt = REPORT_PROMPT.format(
        note=note,
        patient_data=patient_data,
        guidelines=guidelines,
        refinement_context=refinement_context,
    )
    return generate_json(prompt)


def decide_retrieval(note: str, entities: list[dict]) -> dict[str, Any]:
    """Self-RAG: decide if guideline retrieval is needed."""
    prompt = SELF_RAG_DECISION_PROMPT.format(
        note=note,
        entities=json.dumps(entities, ensure_ascii=False, indent=2),
    )
    return generate_json(prompt)


def evaluate_report(
    report: dict[str, Any],
    note: str = "",
    patient_data: str = "",
) -> dict[str, Any]:
    """Evaluate the quality of a generated medical report."""
    prompt = EVALUATE_REPORT_PROMPT.format(
        report=json.dumps(report, ensure_ascii=False, indent=2),
        note=note or "Não disponível",
        patient_data=patient_data or "Não disponível",
    )
    return generate_json(prompt)
