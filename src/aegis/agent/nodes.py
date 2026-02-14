"""Processing nodes for the clinical agent graph.

Each node is a function that receives the current ``AgentState`` and returns
a dict with the keys it wants to update.
"""

from __future__ import annotations

from typing import Any

from aegis.agent.state import AgentState
from aegis.fhir import FHIRStore
from aegis.llm import (
    decide_retrieval as llm_decide_retrieval,
    evaluate_report as llm_evaluate_report,
    extract_entities,
    generate_report as llm_generate_report,
)
from aegis.mcp_server import (
    consultar_condicoes,
    consultar_medicamentos,
    consultar_paciente,
    consultar_sinais_vitais,
)
from aegis.rag.retriever import format_context, retrieve

# Module-level store for patient ID matching
_store = FHIRStore()


def _ensure_store() -> None:
    """Load FHIR data into the module-level store if empty."""
    if not _store.list_patients():
        _store.load_directory()


def _match_patient_id(entities: list[dict[str, str]]) -> str:
    """Try to match a patient ID from extracted entities.

    Looks for patient name mentions and matches against loaded FHIR data.
    Falls back to the first available patient if no match is found.
    """
    _ensure_store()
    patients = _store.list_patients()
    if not patients:
        return ""

    # Collect names/text from entities
    entity_texts = " ".join(
        e.get("text", "") + " " + e.get("normalized", "") for e in entities
    ).lower()

    # Try to match by name
    for p in patients:
        name_parts = p["name"].lower().split()
        if any(part in entity_texts for part in name_parts if len(part) > 2):
            return p["id"]

    # Fallback: first patient
    return patients[0]["id"]


# ------------------------------------------------------------------
# Node functions
# ------------------------------------------------------------------


def parse_note(state: AgentState) -> dict[str, Any]:
    """Extract medical entities and identify the patient from the note."""
    note = state["patient_note"]
    result = extract_entities(note)
    entities = result.get("entities", [])
    patient_id = _match_patient_id(entities)
    return {"extracted_entities": entities, "patient_id": patient_id}


def decide_retrieval(state: AgentState) -> dict[str, Any]:
    """Self-RAG: let the LLM decide if guideline retrieval is needed."""
    note = state["patient_note"]
    entities = state.get("extracted_entities", [])
    result = llm_decide_retrieval(note, entities)
    return {
        "needs_retrieval": result.get("needs_retrieval", False),
        "retrieval_queries": result.get("queries", []),
    }


def retrieve_guidelines(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant clinical guideline chunks via RAG."""
    queries = state.get("retrieval_queries", [])
    if not queries:
        return {"guidelines": "Nenhuma consulta de diretriz solicitada."}

    all_results: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for query in queries:
        results = retrieve(query, top_k=3)
        for r in results:
            if r["text"] not in seen_texts:
                seen_texts.add(r["text"])
                all_results.append(r)

    # Sort by score descending and take top 5 overall
    all_results.sort(key=lambda r: r["score"], reverse=True)
    return {"guidelines": format_context(all_results[:5])}


def fetch_patient_data(state: AgentState) -> dict[str, Any]:
    """Fetch patient clinical data using the MCP tool functions."""
    patient_id = state.get("patient_id", "")
    if not patient_id:
        return {"patient_data": "Paciente não identificado."}

    sections = [
        consultar_paciente(patient_id),
        consultar_condicoes(patient_id),
        consultar_medicamentos(patient_id),
        consultar_sinais_vitais(patient_id),
    ]
    return {"patient_data": "\n\n".join(sections)}


def generate_report(state: AgentState) -> dict[str, Any]:
    """Generate a structured clinical report from all available context."""
    note = state["patient_note"]
    patient_data = state.get("patient_data", "Não disponível")
    guidelines = state.get("guidelines", "Não disponível")

    report = llm_generate_report(
        note=note,
        patient_data=patient_data,
        guidelines=guidelines,
    )
    return {"report": report}


def evaluate_report(state: AgentState) -> dict[str, Any]:
    """Self-evaluate the generated report quality."""
    report = state.get("report", {})
    if not report:
        return {"evaluation": {"overall": {"score": 0, "feedback": "Nenhum relatório gerado."}}}

    evaluation = llm_evaluate_report(report)
    return {"evaluation": evaluation}
