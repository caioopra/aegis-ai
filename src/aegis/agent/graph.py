"""LangGraph state graph for the clinical agent.

Flow: parse → decide → (retrieve | skip) → fetch → generate → evaluate
                                                          ↑         │
                                                          └─────────┘
                                                        (retry if score < 3,
                                                         max 2 retries)
"""

from __future__ import annotations

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


def _route_retrieval(state: AgentState) -> str:
    """Conditional edge: retrieve guidelines or skip to patient data."""
    if state.get("needs_retrieval", False):
        return "retrieve_guidelines"
    return "fetch_patient_data"


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
    warnings = list(state.get("warnings", []))
    score = state.get("evaluation", {}).get("overall", {}).get("score", "?")
    warnings.append(
        f"evaluate_report: score {score}/5 abaixo do mínimo ({MIN_ACCEPTABLE_SCORE}), "
        f"tentativa {current + 1}/{MAX_RETRIES}"
    )
    return {"retry_count": current + 1, "warnings": warnings}


def build_graph() -> StateGraph:
    """Build and return the compiled clinical agent graph."""
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
    graph.add_conditional_edges(
        "decide_retrieval",
        _route_retrieval,
        {
            "retrieve_guidelines": "retrieve_guidelines",
            "fetch_patient_data": "fetch_patient_data",
        },
    )
    graph.add_edge("retrieve_guidelines", "fetch_patient_data")
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

    return graph.compile()


# Module-level compiled graph
agent = build_graph()
