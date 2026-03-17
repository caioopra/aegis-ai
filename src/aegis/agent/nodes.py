"""Processing nodes for the clinical agent graph.

Each node is a function that receives the current ``AgentState`` and returns
a dict with the keys it wants to update.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# Module-level store for patient ID matching
_store = FHIRStore()


def _ensure_store() -> None:
    """Load FHIR data into the module-level store if empty."""
    if not _store.list_patients():
        _store.load_directory()


def _match_patient_id(
    entities: list[dict[str, str]],
) -> tuple[str, str]:
    """Try to match a patient ID from extracted entities.

    Returns ``(patient_id, match_type)`` where match_type is one of:
    ``"exact"``, ``"partial"``, ``"fallback"``, or ``"none"``.
    """
    _ensure_store()
    patients = _store.list_patients()
    if not patients:
        return "", "none"

    # Collect names/text from entities
    entity_texts = " ".join(
        str(e.get("text", "")) + " " + str(e.get("normalized", "")) for e in entities
    ).lower()

    # Try to match by name
    for p in patients:
        name_parts = p["name"].lower().split()
        matched_parts = [part for part in name_parts if len(part) > 2 and part in entity_texts]
        if matched_parts:
            # Check if most name parts matched (exact) or just some (partial)
            match_ratio = len(matched_parts) / len([n for n in name_parts if len(n) > 2])
            match_type = "exact" if match_ratio >= 0.5 else "partial"
            return p["id"], match_type

    # Fallback: first patient
    return patients[0]["id"], "fallback"


# ------------------------------------------------------------------
# Node functions
# ------------------------------------------------------------------


def parse_note(state: AgentState) -> dict[str, Any]:
    """Extract medical entities and identify the patient from the note."""
    note = state["patient_note"]
    warnings: list[str] = list(state.get("warnings", []))

    try:
        result = extract_entities(note)
        entities = result.get("entities", [])
        if not isinstance(entities, list):
            entities = []
            warnings.append("parse_note: entities não é uma lista, usando lista vazia")
    except Exception as e:
        logger.error("parse_note failed: %s", e)
        entities = []
        warnings.append(f"parse_note: falha na extração de entidades — {e}")

    patient_id, match_type = _match_patient_id(entities)

    if match_type == "fallback":
        warnings.append(
            f"parse_note: paciente não identificado na nota, usando fallback: {patient_id}"
        )

    return {
        "extracted_entities": entities,
        "patient_id": patient_id,
        "patient_id_match_type": match_type,
        "warnings": warnings,
    }


def decide_retrieval(state: AgentState) -> dict[str, Any]:
    """Self-RAG: let the LLM decide if guideline retrieval is needed."""
    note = state["patient_note"]
    entities = state.get("extracted_entities", [])
    warnings: list[str] = list(state.get("warnings", []))

    try:
        result = llm_decide_retrieval(note, entities)
        needs = result.get("needs_retrieval", False)
        queries = result.get("queries", [])
        if not isinstance(needs, bool):
            needs = bool(needs)
        if not isinstance(queries, list):
            queries = []
    except Exception as e:
        logger.error("decide_retrieval failed: %s", e)
        # Default to retrieving — safer to have guidelines than not
        needs = True
        queries = [note[:100]]
        warnings.append(f"decide_retrieval: falha no LLM, forçando retrieval — {e}")

    return {
        "needs_retrieval": needs,
        "retrieval_queries": queries,
        "warnings": warnings,
    }


def retrieve_guidelines(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant clinical guideline chunks via RAG."""
    queries = state.get("retrieval_queries", [])
    warnings: list[str] = list(state.get("warnings", []))

    if not queries:
        return {
            "guidelines": "Nenhuma consulta de diretriz solicitada.",
            "retrieval_confidence": 0.0,
            "warnings": warnings,
        }

    try:
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
        top_results = all_results[:5]

        # Compute retrieval confidence
        if top_results:
            avg_score = sum(r["score"] for r in top_results) / len(top_results)
        else:
            avg_score = 0.0

        if avg_score < 0.5:
            warnings.append(
                f"retrieve_guidelines: confiança baixa na recuperação (avg_score={avg_score:.2f})"
            )

        return {
            "guidelines": format_context(top_results),
            "retrieval_confidence": round(avg_score, 3),
            "warnings": warnings,
        }
    except Exception as e:
        logger.error("retrieve_guidelines failed: %s", e)
        warnings.append(f"retrieve_guidelines: falha no RAG — {e}")
        return {
            "guidelines": "Diretrizes indisponíveis devido a erro na recuperação.",
            "retrieval_confidence": 0.0,
            "warnings": warnings,
        }


def fetch_patient_data(state: AgentState) -> dict[str, Any]:
    """Fetch patient clinical data using the MCP tool functions."""
    patient_id = state.get("patient_id", "")
    warnings: list[str] = list(state.get("warnings", []))

    if not patient_id:
        return {
            "patient_data": "Paciente não identificado.",
            "warnings": warnings,
        }

    sections: list[str] = []
    tools = [
        ("consultar_paciente", consultar_paciente),
        ("consultar_condicoes", consultar_condicoes),
        ("consultar_medicamentos", consultar_medicamentos),
        ("consultar_sinais_vitais", consultar_sinais_vitais),
    ]

    for tool_name, tool_fn in tools:
        try:
            sections.append(tool_fn(patient_id))
        except Exception as e:
            logger.error("fetch_patient_data.%s failed: %s", tool_name, e)
            sections.append(f"[{tool_name}: dados indisponíveis]")
            warnings.append(f"fetch_patient_data: {tool_name} falhou — {e}")

    return {
        "patient_data": "\n\n".join(sections),
        "warnings": warnings,
    }


def generate_report(state: AgentState) -> dict[str, Any]:
    """Generate a structured clinical report from all available context."""
    note = state["patient_note"]
    patient_data = state.get("patient_data", "Não disponível")
    guidelines = state.get("guidelines", "Não disponível")
    warnings: list[str] = list(state.get("warnings", []))
    retry_count = state.get("retry_count", 0)

    # Build refinement context if this is a retry
    refinement_context = ""
    if retry_count > 0:
        prev_eval = state.get("evaluation", {})
        feedback_parts = []
        for dim, data in prev_eval.items():
            if isinstance(data, dict) and "feedback" in data:
                score = data.get("score", "?")
                feedback_parts.append(f"- {dim}: {score}/5 — {data['feedback']}")
        if feedback_parts:
            refinement_context = (
                "## Previous Evaluation Feedback (improve on these points)\n"
                + "\n".join(feedback_parts)
            )

    try:
        report = llm_generate_report(
            note=note,
            patient_data=patient_data,
            guidelines=guidelines,
            refinement_context=refinement_context,
        )
    except Exception as e:
        logger.error("generate_report failed: %s", e)
        report = {
            "error": str(e),
            "patient_summary": "Erro na geração do relatório.",
            "findings": [],
            "assessment": "Não foi possível gerar a avaliação.",
            "plan": [],
        }
        warnings.append(f"generate_report: falha na geração — {e}")

    return {"report": report, "warnings": warnings}


def evaluate_report(state: AgentState) -> dict[str, Any]:
    """Self-evaluate the generated report quality."""
    report = state.get("report", {})
    warnings: list[str] = list(state.get("warnings", []))

    if not report or "error" in report:
        return {
            "evaluation": {"overall": {"score": 0, "feedback": "Nenhum relatório gerado."}},
            "warnings": warnings,
        }

    try:
        evaluation = llm_evaluate_report(
            report,
            note=state.get("patient_note", ""),
            patient_data=state.get("patient_data", ""),
        )
    except Exception as e:
        logger.error("evaluate_report failed: %s", e)
        evaluation = {
            "overall": {"score": 3, "feedback": f"Avaliação indisponível: {e}"},
        }
        warnings.append(f"evaluate_report: falha na avaliação — {e}")

    return {"evaluation": evaluation, "warnings": warnings}
