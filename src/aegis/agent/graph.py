"""LangGraph state graph for the clinical agent.

Flow: parse → decide → (retrieve | skip) → fetch → generate → evaluate
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


def _route_retrieval(state: AgentState) -> str:
    """Conditional edge: retrieve guidelines or skip to patient data."""
    if state.get("needs_retrieval", False):
        return "retrieve_guidelines"
    return "fetch_patient_data"


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

    # Set entry point
    graph.set_entry_point("parse_note")

    # Wire edges
    graph.add_edge("parse_note", "decide_retrieval")
    graph.add_conditional_edges("decide_retrieval", _route_retrieval)
    graph.add_edge("retrieve_guidelines", "fetch_patient_data")
    graph.add_edge("fetch_patient_data", "generate_report")
    graph.add_edge("generate_report", "evaluate_report")
    graph.add_edge("evaluate_report", END)

    return graph.compile()


# Module-level compiled graph
agent = build_graph()
