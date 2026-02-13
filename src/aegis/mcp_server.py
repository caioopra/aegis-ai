"""MCP server exposing clinical data tools backed by FHIR bundles."""

from __future__ import annotations

from datetime import date

from mcp.server.fastmcp import FastMCP

from aegis.fhir import FHIRStore, Resource

mcp = FastMCP("AegisNode Clinical Server")

# Module-level store — populated once at startup via the lifespan or
# explicitly with ``_load_store()``.
_store = FHIRStore()

# ------------------------------------------------------------------
# Known drug interactions (simplified, for educational purposes)
# ------------------------------------------------------------------

DRUG_INTERACTIONS: dict[frozenset[str], str] = {
    frozenset({"losartana", "espironolactona"}): (
        "Risco de hipercalemia. Monitorar potássio sérico regularmente."
    ),
    frozenset({"losartana", "enalapril"}): (
        "Duplo bloqueio do SRAA. Risco aumentado de hipotensão, hipercalemia "
        "e insuficiência renal. Evitar associação."
    ),
    frozenset({"metformina", "contraste iodado"}): (
        "Risco de acidose lática. Suspender metformina 48h antes e após "
        "procedimentos com contraste iodado."
    ),
    frozenset({"metformina", "álcool"}): (
        "Risco aumentado de acidose lática. Orientar paciente a evitar "
        "consumo excessivo de álcool."
    ),
    frozenset({"losartana", "ibuprofeno"}): (
        "AINEs reduzem o efeito anti-hipertensivo e aumentam risco de "
        "lesão renal. Evitar uso prolongado."
    ),
    frozenset({"losartana", "diclofenaco"}): (
        "AINEs reduzem o efeito anti-hipertensivo e aumentam risco de "
        "lesão renal. Evitar uso prolongado."
    ),
    frozenset({"hidroclorotiazida", "lítio"}): (
        "Tiazídicos reduzem a excreção renal de lítio. Risco de toxicidade "
        "por lítio. Monitorar níveis séricos."
    ),
    frozenset({"hidroclorotiazida", "digoxina"}): (
        "Hipocalemia induzida por tiazídicos potencializa toxicidade "
        "digitálica. Monitorar potássio e digoxina."
    ),
}


def _load_store() -> None:
    """Load FHIR bundles into the module-level store (idempotent)."""
    if not _store.list_patients():
        _store.load_directory()


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------


def _format_patient(patient: Resource) -> str:
    """Format a Patient resource as readable text."""
    name = FHIRStore._format_patient_name(patient)
    gender_map = {"male": "Masculino", "female": "Feminino"}
    gender = gender_map.get(patient.get("gender", ""), patient.get("gender", ""))

    birth = patient.get("birthDate", "")
    age = ""
    if birth:
        try:
            born = date.fromisoformat(birth)
            age = f" ({(date.today() - born).days // 365} anos)"
        except ValueError:
            pass

    lines = [f"Nome: {name}", f"Sexo: {gender}", f"Nascimento: {birth}{age}"]

    addresses = patient.get("address", [])
    if addresses:
        addr = addresses[0]
        parts = [
            ", ".join(addr.get("line", [])),
            addr.get("city", ""),
            addr.get("state", ""),
        ]
        lines.append(f"Endereço: {' - '.join(p for p in parts if p)}")

    return "\n".join(lines)


def _format_condition(condition: Resource) -> str:
    """Format a Condition resource as a single descriptive line."""
    text = condition.get("code", {}).get("text", "")
    if not text:
        codings = condition.get("code", {}).get("coding", [])
        text = codings[0].get("display", "Desconhecido") if codings else "Desconhecido"

    status_code = ""
    status_codings = condition.get("clinicalStatus", {}).get("coding", [])
    if status_codings:
        status_code = status_codings[0].get("code", "")

    onset = condition.get("onsetDateTime", "")
    parts = [text]
    if status_code:
        parts.append(f"(status: {status_code})")
    if onset:
        parts.append(f"desde {onset}")
    return " — ".join(parts)


def _format_medication(med: Resource) -> str:
    """Format a MedicationRequest resource as a single descriptive line."""
    med_concept = med.get("medicationCodeableConcept", {})
    text = med_concept.get("text", "")
    if not text:
        codings = med_concept.get("coding", [])
        text = codings[0].get("display", "Desconhecido") if codings else "Desconhecido"

    dosage_parts = []
    for d in med.get("dosageInstruction", []):
        if d.get("text"):
            dosage_parts.append(d["text"])

    status = med.get("status", "")
    parts = [text]
    if dosage_parts:
        parts.append(", ".join(dosage_parts))
    if status:
        parts.append(f"(status: {status})")
    return " — ".join(parts)


def _format_observation(obs: Resource) -> str:
    """Format an Observation resource as a single descriptive line."""
    text = obs.get("code", {}).get("text", "")
    if not text:
        codings = obs.get("code", {}).get("coding", [])
        text = codings[0].get("display", "Desconhecido") if codings else "Desconhecido"

    # Simple value (heart rate, weight, height)
    vq = obs.get("valueQuantity")
    if vq:
        return f"{text}: {vq['value']} {vq.get('unit', '')}"

    # Component value (blood pressure)
    components = obs.get("component", [])
    if components:
        comp_parts = []
        for comp in components:
            comp_text = comp.get("code", {}).get("text", "")
            comp_vq = comp.get("valueQuantity", {})
            if comp_vq:
                comp_parts.append(
                    f"{comp_text}: {comp_vq['value']} {comp_vq.get('unit', '')}"
                )
        return f"{text} — {'; '.join(comp_parts)}"

    return text


def _normalize_drug_name(name: str) -> str:
    """Normalize a drug name for interaction lookup."""
    return name.strip().lower()


# ------------------------------------------------------------------
# MCP Tools
# ------------------------------------------------------------------


@mcp.tool()
def listar_pacientes() -> str:
    """Lista todos os pacientes disponíveis no sistema.

    Retorna os IDs e nomes dos pacientes cadastrados. Use esta ferramenta
    primeiro para descobrir quais pacientes existem antes de consultar
    dados específicos.
    """
    _load_store()
    patients = _store.list_patients()
    if not patients:
        return "Nenhum paciente encontrado."

    lines = [f"- {p['name']} (ID: {p['id']})" for p in patients]
    return f"Pacientes disponíveis ({len(patients)}):\n" + "\n".join(lines)


@mcp.tool()
def consultar_paciente(patient_id: str) -> str:
    """Retorna os dados demográficos de um paciente pelo ID.

    Inclui nome completo, sexo, data de nascimento, idade e endereço.
    Use o ID obtido pela ferramenta listar_pacientes.
    """
    _load_store()
    patient = _store.get_patient(patient_id)
    if patient is None:
        return f"Paciente não encontrado: {patient_id}"
    return _format_patient(patient)


@mcp.tool()
def consultar_condicoes(patient_id: str) -> str:
    """Retorna as condições clínicas (diagnósticos) de um paciente.

    Lista todas as condições registradas com status clínico e data de início.
    Exemplos: hipertensão, diabetes, insuficiência cardíaca.
    """
    _load_store()
    conditions = _store.get_conditions(patient_id)
    if not conditions:
        return f"Nenhuma condição registrada para o paciente {patient_id}."

    lines = [f"- {_format_condition(c)}" for c in conditions]
    return f"Condições clínicas ({len(conditions)}):\n" + "\n".join(lines)


@mcp.tool()
def consultar_medicamentos(patient_id: str) -> str:
    """Retorna os medicamentos prescritos para um paciente.

    Lista todos os medicamentos com posologia e status da prescrição.
    Útil para verificar o tratamento atual e possíveis interações.
    """
    _load_store()
    medications = _store.get_medications(patient_id)
    if not medications:
        return f"Nenhum medicamento registrado para o paciente {patient_id}."

    lines = [f"- {_format_medication(m)}" for m in medications]
    return f"Medicamentos ({len(medications)}):\n" + "\n".join(lines)


@mcp.tool()
def consultar_sinais_vitais(patient_id: str) -> str:
    """Retorna os sinais vitais registrados para um paciente.

    Inclui pressão arterial, frequência cardíaca, peso, altura e outros
    sinais vitais disponíveis com seus valores e unidades.
    """
    _load_store()
    observations = _store.get_observations(patient_id)
    if not observations:
        return f"Nenhum sinal vital registrado para o paciente {patient_id}."

    lines = [f"- {_format_observation(o)}" for o in observations]
    return f"Sinais vitais ({len(observations)}):\n" + "\n".join(lines)


@mcp.tool()
def verificar_interacao_medicamentosa(medicamento_a: str, medicamento_b: str) -> str:
    """Verifica se há interação conhecida entre dois medicamentos.

    Informe os nomes dos medicamentos em português (ex: losartana, metformina).
    Retorna a descrição da interação e recomendações, se houver.
    """
    a = _normalize_drug_name(medicamento_a)
    b = _normalize_drug_name(medicamento_b)

    pair = frozenset({a, b})
    interaction = DRUG_INTERACTIONS.get(pair)

    if interaction:
        return (
            f"⚠ Interação encontrada entre {medicamento_a} e {medicamento_b}:\n"
            f"{interaction}"
        )
    return (
        f"Nenhuma interação conhecida entre {medicamento_a} e {medicamento_b}.\n"
        "Nota: esta verificação cobre apenas interações comuns pré-cadastradas. "
        "Consulte fontes farmacológicas completas para análise definitiva."
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
