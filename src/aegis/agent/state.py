"""Agent state schema — the typed dictionary that flows through the graph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    """State that flows through every node in the LangGraph agent.

    Fields use ``total=False`` so nodes can return partial updates
    (only the keys they modify).
    """

    # Input
    patient_note: str

    # After parse_note
    extracted_entities: list[dict[str, str]]
    patient_id: str
    patient_id_match_type: Literal["exact", "partial", "fallback", "none"]

    # After decide_retrieval
    needs_retrieval: bool
    retrieval_queries: list[str]

    # After retrieve_guidelines
    guidelines: str
    retrieval_confidence: float

    # After fetch_patient_data
    patient_data: str

    # After generate_report
    report: dict[str, Any]

    # After evaluate_report
    evaluation: dict[str, Any]

    # Retry loop
    retry_count: int

    # Error tracking (accumulated across all nodes)
    warnings: list[str]
    errors: list[str]
