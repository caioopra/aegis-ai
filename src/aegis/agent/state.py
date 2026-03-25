"""Agent state schema — the typed dictionary that flows through the graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    """State that flows through every node in the LangGraph agent.

    Fields use ``total=False`` so nodes can return partial updates
    (only the keys they modify).

    ``warnings`` and ``errors`` use an ``operator.add`` reducer so that
    parallel nodes (fan-out) can each append entries without conflicting.
    Nodes should return only **new** warnings/errors, not the accumulated list.
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
    # Tools used in fetch_patient_data
    tools_called: Annotated[list[str], operator.add]

    # After generate_report
    report: dict[str, Any]

    # After evaluate_report
    evaluation: dict[str, Any]

    # Retry loop
    retry_count: int

    # Error tracking — use ``operator.add`` reducer for parallel-safe accumulation
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
