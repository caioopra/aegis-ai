"""LangGraph state graph for the clinical agent.

Flow: parse → decide → fan-out [retrieve + fetch | fetch only] → generate → evaluate
                                                                       ↑         │
                                                                       └─────────┘
                                                                     (retry if score < 3,
                                                                      max 2 retries)

When ``needs_retrieval=True``, ``retrieve_guidelines`` and ``fetch_patient_data``
run **in parallel** (fan-out) since they are independent — RAG uses queries while
FHIR uses patient_id.  Both must complete before ``generate_report`` starts
(fan-in).  When ``needs_retrieval=False``, only ``fetch_patient_data`` runs.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from aegis.agent.nodes import (
    decide_retrieval,
    evaluate_report,
    fetch_patient_data,
    generate_report,
    parse_note,
    retrieve_guidelines,
)
from aegis.agent.state import AgentState

MAX_RETRIES = 2
MIN_ACCEPTABLE_SCORE = 3


def _route_retrieval(state: AgentState) -> list[str]:
    """Fan-out: retrieve guidelines + fetch patient data in parallel, or fetch only."""
    if state.get("needs_retrieval", False):
        return ["retrieve_guidelines", "fetch_patient_data"]
    return ["fetch_patient_data"]


def _route_after_evaluation(state: AgentState) -> str:
    """Conditional edge: retry report generation or finish."""
    retry_count = state.get("retry_count", 0)
    evaluation = state.get("evaluation", {})
    overall = evaluation.get("overall", {})
    score = (
        overall.get("score", MIN_ACCEPTABLE_SCORE)
        if isinstance(overall, dict)
        else MIN_ACCEPTABLE_SCORE
    )

    if score < MIN_ACCEPTABLE_SCORE and retry_count < MAX_RETRIES:
        return "increment_retry"
    return END


def _increment_retry(state: AgentState) -> dict:
    """Bump the retry counter before looping back to generate_report."""
    current = state.get("retry_count", 0)
    score = state.get("evaluation", {}).get("overall", {}).get("score", "?")
    return {
        "retry_count": current + 1,
        "warnings": [
            f"evaluate_report: score {score}/5 abaixo do mínimo ({MIN_ACCEPTABLE_SCORE}), "
            f"tentativa {current + 1}/{MAX_RETRIES}"
        ],
    }


def build_graph(checkpointer: Any | None = None) -> StateGraph:
    """Build and return the compiled clinical agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for session memory.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse_note", parse_note)
    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("retrieve_guidelines", retrieve_guidelines)
    graph.add_node("fetch_patient_data", fetch_patient_data)
    graph.add_node("generate_report", generate_report)
    graph.add_node("evaluate_report", evaluate_report)
    graph.add_node("increment_retry", _increment_retry)

    # Set entry point
    graph.set_entry_point("parse_note")

    # Wire edges
    graph.add_edge("parse_note", "decide_retrieval")

    # Fan-out: decide_retrieval → [retrieve_guidelines + fetch_patient_data] (parallel)
    #          or decide_retrieval → [fetch_patient_data] (skip RAG)
    graph.add_conditional_edges(
        "decide_retrieval",
        _route_retrieval,
        ["retrieve_guidelines", "fetch_patient_data"],
    )

    # Fan-in: both parallel branches converge into generate_report
    graph.add_edge("retrieve_guidelines", "generate_report")
    graph.add_edge("fetch_patient_data", "generate_report")

    graph.add_edge("generate_report", "evaluate_report")

    # Retry loop: evaluate → (retry | END)
    graph.add_conditional_edges(
        "evaluate_report",
        _route_after_evaluation,
        {
            "increment_retry": "increment_retry",
            END: END,
        },
    )
    graph.add_edge("increment_retry", "generate_report")

    kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return graph.compile(**kwargs)


# Module-level compiled graph
agent = build_graph()
