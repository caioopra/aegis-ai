"""Processing nodes for the clinical agent graph.

Each node is a function that receives the current ``AgentState`` and returns
a dict with the keys it wants to update.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Any

from aegis.agent.state import AgentState
import aegis.fhir
from aegis.llm import (
    MAX_INPUT_TOKENS,
    decide_retrieval as llm_decide_retrieval,
    estimate_tokens,
    evaluate_report as llm_evaluate_report,
    extract_entities,
    generate_report as llm_generate_report,
    truncate_to_budget,
)
from aegis.mcp_server import (
    consultar_alergias,
    consultar_condicoes,
    consultar_encontros,
    consultar_exames,
    consultar_imunizacoes,
    consultar_medicamentos,
    consultar_paciente,
    consultar_procedimentos,
    consultar_sinais_vitais,
    verificar_interacao_medicamentosa,
)
from aegis.rag.retriever import format_context, retrieve

logger = logging.getLogger(__name__)

# Mandatory disclaimer attached to every generated report. Hardcoded so the
# LLM cannot omit or soften it.
AI_DISCLAIMER = "Este relatório foi gerado por IA e NÃO substitui o julgamento clínico do médico."

# ------------------------------------------------------------------
# Dynamic tool selection mappings
# ------------------------------------------------------------------

TOOL_KEYWORDS: dict[str, list[str]] = {
    "consultar_procedimentos": [
        "procedimento",
        "cirurgia",
        "ecocardiograma",
        "eletrocardiograma",
        "cateterismo",
        "endoscopia",
        "colonoscopia",
        "biópsia",
    ],
    "consultar_exames": [
        "exame",
        "hemograma",
        "hba1c",
        "glicemia",
        "colesterol",
        "creatinina",
        "ureia",
        "tgo",
        "tgp",
        "hemoglobina",
        "laborat",
        "lab",
        "raio-x",
        "tomografia",
        "ressonância",
    ],
    "consultar_encontros": [
        "internação",
        "internado",
        "emergência",
        "pronto-socorro",
        "consulta anterior",
        "histórico",
        "encontro",
    ],
    "consultar_imunizacoes": [
        "vacina",
        "imunização",
        "vacinação",
        "covid",
        "influenza",
        "gripe",
    ],
}

# Map from tool name to the callable MCP function
_DYNAMIC_TOOL_FNS: dict[str, Any] = {
    "consultar_procedimentos": consultar_procedimentos,
    "consultar_exames": consultar_exames,
    "consultar_encontros": consultar_encontros,
    "consultar_imunizacoes": consultar_imunizacoes,
}


def _select_dynamic_tools(entities: list[dict[str, str]]) -> list[str]:
    """Examine extracted entities and return additional tool names to call.

    Checks both entity ``type`` fields and keyword matches against entity
    ``text``/``normalized`` values.
    """
    # Build a single lowercase string from all entity text for keyword matching
    entity_blob = " ".join(
        str(e.get("text", "")) + " " + str(e.get("normalized", "")) for e in entities
    ).lower()

    # Collect entity types
    entity_types = {str(e.get("type", "")).lower() for e in entities}

    selected: set[str] = set()

    # Type-based triggers
    type_tool_map: dict[str, str] = {
        "procedure": "consultar_procedimentos",
        "exam": "consultar_exames",
        "lab": "consultar_exames",
        "lab_result": "consultar_exames",
    }
    for etype, tool_name in type_tool_map.items():
        if etype in entity_types:
            selected.add(tool_name)

    # Keyword-based triggers
    for tool_name, keywords in TOOL_KEYWORDS.items():
        for kw in keywords:
            if kw in entity_blob:
                selected.add(tool_name)
                break

    return sorted(selected)


def _extract_medication_names(entities: list[dict[str, str]]) -> list[str]:
    """Return medication names from extracted entities."""
    names: list[str] = []
    for e in entities:
        etype = str(e.get("type", "")).lower()
        if etype in ("medication", "medicamento", "drug"):
            name = str(e.get("normalized", "") or e.get("text", "")).strip()
            if name:
                names.append(name)
    return names


_CPF_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")


def _match_patient_id(
    entities: list[dict[str, str]],
    note: str = "",
) -> tuple[str, str]:
    """Try to match a patient ID from extracted entities and the raw note.

    Returns ``(patient_id, match_type)`` where match_type is one of:
    ``"exact"``, ``"partial"``, ``"cpf"``, or ``"none"``.

    CPF lookup is attempted first (before name matching) when a CPF pattern
    is found in the note or entity texts.  The raw match value is passed
    directly to ``get_patient_by_cpf`` — do not log it in plaintext.
    """
    store = aegis.fhir.get_store()
    patients = store.list_patients()
    if not patients:
        return "", "none"

    # Collect names/text from entities (used both for CPF fallback and name matching)
    entity_texts = " ".join(
        str(e.get("text", "")) + " " + str(e.get("normalized", "")) for e in entities
    ).lower()

    # --- CPF branch: search note first, then entity texts ---
    cpf_match = _CPF_PATTERN.search(note) or _CPF_PATTERN.search(entity_texts)
    if cpf_match:
        logger.debug("_match_patient_id: CPF extraído da nota, tentando lookup")
        patient_dict = store.get_patient_by_cpf(cpf_match.group(0))
        if patient_dict is not None:
            return patient_dict["id"], "cpf"
        # CPF found but no hit — fall through to name matching

    # The note itself is the most reliable source for patient names,
    # since the entity extraction prompt focuses on medical entities
    # and may not extract patient names.
    search_text = f"{note.lower()} {entity_texts}"

    # Try to match by name
    for p in patients:
        name_parts = p["name"].lower().split()
        matched_parts = [
            part
            for part in name_parts
            if len(part) > 2 and re.search(r"\b" + re.escape(part) + r"\b", search_text)
        ]
        if matched_parts:
            # Check if most name parts matched (exact) or just some (partial)
            match_ratio = len(matched_parts) / len([n for n in name_parts if len(n) > 2])
            match_type = "exact" if match_ratio >= 0.5 else "partial"
            return p["id"], match_type

    # No match found — return empty to avoid using wrong patient data
    return "", "none"


# ------------------------------------------------------------------
# Node functions
# ------------------------------------------------------------------


def parse_note(state: AgentState) -> dict[str, Any]:
    """Extract medical entities and identify the patient from the note."""
    note = state["patient_note"]
    warnings: list[str] = []

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

    patient_id, match_type = _match_patient_id(entities, note=note)

    if match_type == "none" and patient_id == "":
        warnings.append("parse_note: paciente não identificado na nota")
    elif match_type == "cpf" and patient_id:
        # Consistency check: verify the resolved patient's name appears in the note.
        # A CPF that maps to a different person than the one named in the note is a
        # potential data-mix-up (e.g., wrong CPF transcribed).
        store = aegis.fhir.get_store()
        try:
            patients = store.list_patients()
            resolved_name = next((p["name"] for p in patients if p["id"] == patient_id), "").lower()
            if resolved_name:
                name_parts = [part for part in resolved_name.split() if len(part) > 2]
                name_found_in_note = any(part in note.lower() for part in name_parts)
                if not name_found_in_note:
                    warnings.append(
                        "parse_note: CPF resolveu para paciente cujo nome não consta na nota "
                        "— verifique se o CPF foi digitado corretamente"
                    )
        except Exception:
            pass  # Non-fatal — don't break the pipeline for a consistency hint

    return {
        "extracted_entities": entities,
        "patient_id": patient_id,
        "patient_id_match_type": match_type,
        "warnings": warnings,
    }


# Entity types that should always trigger guideline retrieval — having a real
# condition or prescribed medication means the report needs evidence backing.
_SAFETY_NET_ENTITY_TYPES = frozenset({"condition", "medication", "medicamento", "drug"})


def _has_clinical_entities(entities: list[dict[str, str]]) -> bool:
    """Return True if any entity is a condition or medication."""
    for e in entities:
        if str(e.get("type", "")).lower() in _SAFETY_NET_ENTITY_TYPES:
            return True
    return False


def decide_retrieval(state: AgentState) -> dict[str, Any]:
    """Self-RAG: let the LLM decide if guideline retrieval is needed.

    Safety net: when any condition or medication was extracted, force
    retrieval regardless of the LLM's decision — notes with real clinical
    content should always be grounded in guidelines.
    """
    note = state["patient_note"]
    entities = state.get("extracted_entities", [])
    warnings: list[str] = []

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
        queries = ["avaliação clínica geral"]
        warnings.append(f"decide_retrieval: falha no LLM, forçando retrieval — {e}")

    # Safety net: override LLM when clinical entities are present.
    if not needs and _has_clinical_entities(entities):
        needs = True
        warnings.append(
            "decide_retrieval: rede de segurança ativada — "
            "condições/medicamentos detectados, forçando retrieval"
        )

    # If retrieval is forced but no queries were produced, seed from entities
    # or note so retrieve_guidelines has something to search on.
    if needs and not queries:
        clinical_terms = [
            str(e.get("normalized", "") or e.get("text", "")).strip()
            for e in entities
            if str(e.get("type", "")).lower() in _SAFETY_NET_ENTITY_TYPES
        ]
        clinical_terms = [t for t in clinical_terms if t]
        queries = clinical_terms[:3] if clinical_terms else ["avaliação clínica geral"]

    return {
        "needs_retrieval": needs,
        "retrieval_queries": queries,
        "warnings": warnings,
    }


def retrieve_guidelines(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant clinical guideline chunks via RAG."""
    queries = state.get("retrieval_queries", [])
    warnings: list[str] = []

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
    """Fetch patient clinical data using base + dynamically selected MCP tools."""
    patient_id = state.get("patient_id", "")
    entities = state.get("extracted_entities", [])
    warnings: list[str] = []

    if not patient_id:
        return {
            "patient_data": "Paciente não identificado — dados clínicos não recuperados.",
            "tools_called": [],
            "warnings": warnings,
        }

    # Base tools — always called (allergy data must always be present for safety)
    base_tools: list[tuple[str, Any]] = [
        ("consultar_paciente", consultar_paciente),
        ("consultar_condicoes", consultar_condicoes),
        ("consultar_medicamentos", consultar_medicamentos),
        ("consultar_sinais_vitais", consultar_sinais_vitais),
        ("consultar_alergias", consultar_alergias),
    ]

    # Dynamic tools — selected based on entities
    dynamic_tool_names = _select_dynamic_tools(entities)
    dynamic_tools: list[tuple[str, Any]] = [
        (name, _DYNAMIC_TOOL_FNS[name]) for name in dynamic_tool_names if name in _DYNAMIC_TOOL_FNS
    ]

    all_tools = base_tools + dynamic_tools
    tools_called: list[str] = []
    sections: list[str] = []

    for tool_name, tool_fn in all_tools:
        try:
            sections.append(tool_fn(patient_id))
            tools_called.append(tool_name)
        except Exception as e:
            logger.error("fetch_patient_data.%s failed: %s", tool_name, e)
            sections.append(f"[{tool_name}: dados indisponíveis]")
            warnings.append(f"fetch_patient_data: {tool_name} falhou — {e}")
            tools_called.append(tool_name)

    # Medication interaction checks — if 2+ medications found in entities
    med_names = _extract_medication_names(entities)
    if len(med_names) >= 2:
        for med_a, med_b in combinations(med_names, 2):
            interaction_tool = "verificar_interacao_medicamentosa"
            try:
                result = verificar_interacao_medicamentosa(med_a, med_b)
                sections.append(result)
                tools_called.append(f"{interaction_tool}({med_a}, {med_b})")
            except Exception as e:
                logger.error(
                    "fetch_patient_data.%s(%s, %s) failed: %s",
                    interaction_tool,
                    med_a,
                    med_b,
                    e,
                )
                sections.append(f"[{interaction_tool}: erro ao verificar {med_a} + {med_b}]")
                warnings.append(
                    f"fetch_patient_data: {interaction_tool}({med_a}, {med_b}) falhou — {e}"
                )
                tools_called.append(f"{interaction_tool}({med_a}, {med_b})")

    if dynamic_tool_names:
        logger.info(
            "fetch_patient_data: ferramentas dinâmicas selecionadas: %s",
            dynamic_tool_names,
        )

    return {
        "patient_data": "\n\n".join(sections),
        "tools_called": tools_called,
        "warnings": warnings,
    }


def generate_report(state: AgentState) -> dict[str, Any]:
    """Generate a structured clinical report from all available context."""
    note = state["patient_note"]
    patient_data = state.get("patient_data", "Não disponível")
    guidelines = state.get("guidelines", "Não disponível")
    warnings: list[str] = []
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

    # Context budget management — truncate large sections to fit model window
    prompt_overhead = 200  # template text + JSON structure instructions
    total_tokens = (
        prompt_overhead
        + estimate_tokens(note)
        + estimate_tokens(patient_data)
        + estimate_tokens(guidelines)
        + estimate_tokens(refinement_context)
    )

    if total_tokens > MAX_INPUT_TOKENS:
        # Truncate the largest sections first; keep note untruncated (doctor's input)
        budget_remaining = MAX_INPUT_TOKENS - prompt_overhead - estimate_tokens(note)
        half_budget = budget_remaining // 2
        patient_data = truncate_to_budget(patient_data, half_budget, "dados do paciente")
        guidelines = truncate_to_budget(guidelines, half_budget, "diretrizes")
        warnings.append(
            f"generate_report: contexto truncado ({total_tokens} tokens estimados, "
            f"limite {MAX_INPUT_TOKENS})"
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

    # Mandatory AI disclaimer — hardcoded, never LLM-generated. Applies to
    # both successful and error-path reports.
    if isinstance(report, dict):
        report["disclaimer"] = AI_DISCLAIMER

    return {"report": report, "warnings": warnings}


# ------------------------------------------------------------------
# Allergy-prescription safety check
# ------------------------------------------------------------------

# Drug-class groupings for allergy cross-check.  Keys are class names;
# values are all member drug names (lowercased) that belong to that class.
ALLERGY_CLASS_GROUPS: dict[str, set[str]] = {
    "penicilina": {
        "penicilina",
        "amoxicilina",
        "ampicilina",
        "benzetacil",
        "oxacilina",
        "ampicilina+sulbactam",
        "amoxicilina+clavulanato",
    },
    "sulfa": {
        "sulfa",
        "sulfametoxazol",
        "sulfadiazina",
        "sulfassalazina",
        "bactrim",
        "sulfametoxazol+trimetoprima",
    },
    "aine": {
        "aine",
        "anti-inflamatório",
        "ibuprofeno",
        "diclofenaco",
        "naproxeno",
        "nimesulida",
        "cetoprofeno",
        "piroxicam",
        "celecoxibe",
        "aspirina",
        "aas",
        "ácido acetilsalicílico",
        "meloxicam",
        "etoricoxibe",
    },
    "cefalosporina": {
        "cefalosporina",
        "cefalexina",
        "cefazolina",
        "ceftriaxona",
        "cefuroxima",
        "cefepima",
    },
}


def _extract_allergen_names(patient_data: str) -> list[str]:
    """Return allergen class names present in the patient_data string.

    Scans the formatted text returned by ``consultar_alergias`` for known
    drug-class keywords (lowercased).  Returns unique class names only.
    """
    haystack = patient_data.lower()
    found: list[str] = []
    for class_name, members in ALLERGY_CLASS_GROUPS.items():
        for drug in members:
            if drug in haystack:
                found.append(class_name)
                break
    return found


def _extract_plan_medications(plan_items: list[str]) -> list[str]:
    """Extract drug names (lowercased) mentioned in report plan items.

    Only considers drugs present in ``ALLERGY_CLASS_GROUPS`` since the
    allergy check only needs to cross-reference those classes.
    """
    all_known_drugs: set[str] = set()
    for members in ALLERGY_CLASS_GROUPS.values():
        all_known_drugs.update(members)

    found: list[str] = []
    for item in plan_items:
        item_lower = item.lower()
        for drug in all_known_drugs:
            # Word-boundary match so "ampicilina" doesn't match "ampicilinax"
            if re.search(r"\b" + re.escape(drug) + r"\b", item_lower):
                if drug not in found:
                    found.append(drug)
    return found


def check_allergy_safety(state: AgentState) -> dict[str, Any]:
    """Cross-check the generated plan against the patient's known allergies.

    Surfaces warnings in ``report["sinais_alarme"]`` and ``state["warnings"]``
    when the plan contains a medication belonging to the same drug class as a
    known allergy.  Does NOT rewrite or block the plan — the clinician decides.
    """
    report = state.get("report", {})
    patient_data = state.get("patient_data", "")
    warnings: list[str] = []

    if not isinstance(report, dict) or "error" in report:
        return {"report": report, "warnings": warnings}

    plan_items = report.get("plan", []) or []
    if not isinstance(plan_items, list) or not plan_items:
        return {"report": report, "warnings": warnings}

    allergen_classes = _extract_allergen_names(patient_data)
    if not allergen_classes:
        return {"report": report, "warnings": warnings}

    prescribed_drugs = _extract_plan_medications(plan_items)
    if not prescribed_drugs:
        return {"report": report, "warnings": warnings}

    # Cross-check: for each prescribed drug, does it belong to an allergen class?
    conflicts: list[tuple[str, str]] = []  # (allergen_class, prescribed_drug)
    for drug in prescribed_drugs:
        for allergen_class in allergen_classes:
            members = ALLERGY_CLASS_GROUPS.get(allergen_class, set())
            if drug in members:
                conflicts.append((allergen_class, drug))
                break

    if not conflicts:
        return {"report": report, "warnings": warnings}

    # Emit warnings and prepend to sinais_alarme (most visible position)
    sinais_alarme = report.get("sinais_alarme", []) or []
    if not isinstance(sinais_alarme, list):
        sinais_alarme = []

    for allergen_class, drug in conflicts:
        warning_msg = (
            f"\u26a0 ALERTA DE ALERGIA: paciente alérgico a {allergen_class}; "
            f"medicação {drug} prescrita no plano pode causar reação."
        )
        sinais_alarme.insert(0, warning_msg)
        warnings.append(warning_msg)
        logger.warning("check_allergy_safety: %s", warning_msg)

    report["sinais_alarme"] = sinais_alarme

    return {"report": report, "warnings": warnings}


def evaluate_report(state: AgentState) -> dict[str, Any]:
    """Self-evaluate the generated report quality."""
    report = state.get("report", {})
    warnings: list[str] = []

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
