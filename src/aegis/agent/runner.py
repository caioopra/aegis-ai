"""Programmatic pipeline entry point for the clinical agent."""

from __future__ import annotations

import time
from typing import Any, Generator

from aegis.agent.graph import build_graph


def run_pipeline(
    note: str,
    checkpointer: Any | None = None,
    thread_id: str | None = None,
) -> dict:
    """Run the clinical agent pipeline on a note and return the final state.

    Args:
        note: The clinical note to process.
        checkpointer: Optional LangGraph checkpointer for session memory.
        thread_id: Optional thread ID for session continuity.

    Returns:
        The final AgentState as a dict with all pipeline outputs.
    """
    graph = build_graph(checkpointer=checkpointer) if checkpointer else build_graph()
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    state: dict = {}
    for step in graph.stream({"patient_note": note}, config=config):
        for _node_name, node_output in step.items():
            state.update(node_output)

    return state


def stream_pipeline(
    note: str,
    checkpointer: Any | None = None,
    thread_id: str | None = None,
) -> Generator[tuple[str, dict, float], None, None]:
    """Stream the pipeline, yielding (node_name, output, elapsed_seconds) per step.

    This is useful for UIs that want to show progress as each node completes.
    """
    graph = build_graph(checkpointer=checkpointer) if checkpointer else build_graph()
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    prev_time = time.perf_counter()
    for step in graph.stream({"patient_note": note}, config=config):
        now = time.perf_counter()
        elapsed = now - prev_time
        prev_time = now
        for node_name, node_output in step.items():
            yield node_name, node_output, elapsed
