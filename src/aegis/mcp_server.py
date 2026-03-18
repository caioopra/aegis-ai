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
        "Risco aumentado de acidose lática. Orientar paciente a evitar consumo excessivo de álcool."
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
                comp_parts.append(f"{comp_text}: {comp_vq['value']} {comp_vq.get('unit', '')}")
        return f"{text} — {'; '.join(comp_parts)}"

    return text


def _format_procedure(proc: Resource) -> str:
    """Format a Procedure resource as a single descriptive line."""
    text = proc.get("code", {}).get("text", "")
    if not text:
        codings = proc.get("code", {}).get("coding", [])
        text = codings[0].get("display", "Desconhecido") if codings else "Desconhecido"

    status = proc.get("status", "")
    performed = proc.get("performedDateTime", "")
    # performedPeriod fallback
    period = proc.get("performedPeriod", {})
    if not performed and period:
        performed = period.get("start", "")

    parts = [text]
    if status:
        parts.append(f"(status: {status})")
    if performed:
        parts.append(f"em {performed[:10]}")
    return " — ".join(parts)


def _format_diagnostic_report(report: Resource) -> str:
    """Format a DiagnosticReport resource as a descriptive block."""
    text = report.get("code", {}).get("text", "")
    if not text:
        codings = report.get("code", {}).get("coding", [])
        text = codings[0].get("display", "Desconhecido") if codings else "Desconhecido"

    effective = report.get("effectiveDateTime", "")
    conclusion = report.get("conclusion", "")

    parts = [text]
    if effective:
        parts.append(f"em {effective[:10]}")
    line = " — ".join(parts)
    if conclusion:
        line += f"\n  Conclusão: {conclusion}"
    return line


def _format_encounter(enc: Resource) -> str:
    """Format an Encounter resource as a descriptive line."""
    type_list = enc.get("type", [])
    if type_list:
        text = type_list[0].get("text", "")
        if not text:
            codings = type_list[0].get("coding", [])
            text = codings[0].get("display", "Desconhecido") if codings else "Desconhecido"
    else:
        text = "Encontro"

    enc_class = enc.get("class", {})
    class_display = enc_class.get("display", enc_class.get("code", ""))

    period = enc.get("period", {})
    start = period.get("start", "")[:10]
    end = period.get("end", "")[:10]

    reasons = enc.get("reasonCode", [])
    reason_text = reasons[0].get("text", "") if reasons else ""

    parts = [text]
    if class_display:
        parts.append(f"({class_display})")
    if start:
        date_range = start
        if end and end != start:
            date_range += f" a {end}"
        parts.append(date_range)
    line = " — ".join(parts)
    if reason_text:
        line += f"\n  Motivo: {reason_text}"
    return line


def _format_immunization(imm: Resource) -> str:
    """Format an Immunization resource as a single descriptive line."""
    vaccine = imm.get("vaccineCode", {})
    text = vaccine.get("text", "")
    if not text:
        codings = vaccine.get("coding", [])
        text = codings[0].get("display", "Desconhecido") if codings else "Desconhecido"

    occurrence = imm.get("occurrenceDateTime", "")
    status = imm.get("status", "")

    parts = [text]
    if status:
        parts.append(f"(status: {status})")
    if occurrence:
        parts.append(f"em {occurrence[:10]}")
    return " — ".join(parts)


def _format_allergy(allergy: Resource) -> str:
    """Format an AllergyIntolerance resource as a single descriptive line."""
    # Get the substance/allergen
    code = allergy.get("code", {})
    text = code.get("text", "")
    if not text:
        codings = code.get("coding", [])
        text = codings[0].get("display", "Desconhecido") if codings else "Desconhecido"

    clinical_status = ""
    status_codings = allergy.get("clinicalStatus", {}).get("coding", [])
    if status_codings:
        clinical_status = status_codings[0].get("code", "")

    category = allergy.get("category", [])
    category_text = ", ".join(category) if category else ""

    criticality = allergy.get("criticality", "")

    parts = [text]
    if clinical_status:
        parts.append(f"(status: {clinical_status})")
    if category_text:
        parts.append(f"categoria: {category_text}")
    if criticality:
        parts.append(f"criticidade: {criticality}")
    return " — ".join(parts)


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
def consultar_procedimentos(patient_id: str) -> str:
    """Retorna os procedimentos realizados em um paciente.

    Lista exames e procedimentos médicos como ecocardiograma,
    eletrocardiograma, cateterismo, cirurgias, etc.
    """
    _load_store()
    procedures = _store.get_procedures(patient_id)
    if not procedures:
        return f"Nenhum procedimento registrado para o paciente {patient_id}."

    lines = [f"- {_format_procedure(p)}" for p in procedures]
    return f"Procedimentos ({len(procedures)}):\n" + "\n".join(lines)


@mcp.tool()
def consultar_exames(patient_id: str) -> str:
    """Retorna os laudos de exames diagnósticos de um paciente.

    Inclui resultados de exames laboratoriais (hemograma, HbA1c, perfil
    lipídico, função renal) e exames de imagem com suas conclusões.
    """
    _load_store()
    reports = _store.get_diagnostic_reports(patient_id)
    if not reports:
        return f"Nenhum exame registrado para o paciente {patient_id}."

    lines = [f"- {_format_diagnostic_report(r)}" for r in reports]
    return f"Exames diagnósticos ({len(reports)}):\n" + "\n".join(lines)


@mcp.tool()
def consultar_encontros(patient_id: str) -> str:
    """Retorna o histórico de consultas e internações de um paciente.

    Lista encontros clínicos com tipo (ambulatorial, internação, emergência),
    datas e motivos. Útil para entender o histórico de atendimentos.
    """
    _load_store()
    encounters = _store.get_encounters(patient_id)
    if not encounters:
        return f"Nenhum encontro registrado para o paciente {patient_id}."

    lines = [f"- {_format_encounter(e)}" for e in encounters]
    return f"Encontros clínicos ({len(encounters)}):\n" + "\n".join(lines)


@mcp.tool()
def consultar_imunizacoes(patient_id: str) -> str:
    """Retorna as vacinas registradas para um paciente.

    Lista imunizações com nome da vacina, status e data de aplicação.
    Útil para verificar o calendário vacinal e imunizações pendentes.
    """
    _load_store()
    immunizations = _store.get_immunizations(patient_id)
    if not immunizations:
        return f"Nenhuma imunização registrada para o paciente {patient_id}."

    lines = [f"- {_format_immunization(i)}" for i in immunizations]
    return f"Imunizações ({len(immunizations)}):\n" + "\n".join(lines)


@mcp.tool()
def consultar_alergias(patient_id: str) -> str:
    """Retorna as alergias e intolerâncias registradas para um paciente.

    Lista alergias a medicamentos, alimentos e substâncias com status
    clínico e nível de criticidade. Informação crítica para segurança
    na prescrição de medicamentos.
    """
    _load_store()
    allergies = _store.get_allergy_intolerances(patient_id)
    if not allergies:
        return f"Nenhuma alergia registrada para o paciente {patient_id}."

    lines = [f"- {_format_allergy(a)}" for a in allergies]
    return f"Alergias e intolerâncias ({len(allergies)}):\n" + "\n".join(lines)


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
        return f"⚠ Interação encontrada entre {medicamento_a} e {medicamento_b}:\n{interaction}"
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
