"""LLM client wrapper with prompt templates for clinical tasks."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from aegis.providers import get_chat_provider
from aegis.providers.base import ChatProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------


class ExtractedEntity(BaseModel):
    text: str
    type: Literal[
        "symptom",
        "sign",
        "medication",
        "condition",
        "vital_sign",
        "procedure",
        "lab_result",
        "allergy",
        "family_history",
        "social_history",
    ]
    normalized: str | None = None


class EntityExtractionResult(BaseModel):
    entities: list[ExtractedEntity]


class RetrievalDecision(BaseModel):
    needs_retrieval: bool
    queries: list[str] = []
    reasoning: str = ""


class AcompanhamentoSection(BaseModel):
    proxima_visita: str = ""
    exames_a_repetir: list[str] = []
    sinais_para_escalar: list[str] = []


class ClinicalReport(BaseModel):
    patient_summary: str = ""
    findings: list[str] = []
    assessment: str = ""
    plan: list[str] = []
    guideline_references: list[str] = []
    diagnosticos_diferenciais: list[str] = []
    sinais_alarme: list[str] = []
    acompanhamento: AcompanhamentoSection = Field(default_factory=AcompanhamentoSection)
    interacoes_medicamentosas: list[str] = []
    limitacoes: list[str] = []
    disclaimer: str | None = None  # injected by nodes.py after generation


class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    feedback: str = ""


class ReportEvaluation(BaseModel):
    completeness: DimensionScore
    accuracy: DimensionScore
    guideline_adherence: DimensionScore
    clarity: DimensionScore
    safety: DimensionScore
    follow_up_quality: DimensionScore
    overall: DimensionScore


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# -- Legacy alias kept for backward compatibility in tests ----------------
SYSTEM_MEDICAL = (
    "Você é um assistente clínico de IA. Ajude médicos a expandir "
    "abreviações médicas, extrair dados estruturados de notas clínicas "
    "e gerar relatórios médicos. Seja preciso e use terminologia médica padrão. "
    "Responda sempre em português."
)

# -- Task-specific system prompts ----------------------------------------

SYSTEM_ENTITY_EXTRACTION = (
    "Você é um especialista em terminologia médica. Sua tarefa é identificar "
    "e normalizar todas as entidades clínicas em notas médicas brasileiras. "
    "Expanda abreviações, normalize termos e classifique cada entidade. "
    "Responda apenas com JSON válido."
)

SYSTEM_RAG_DECISION = (
    "Você é um assistente clínico que avalia se diretrizes clínicas devem ser "
    "consultadas para elaborar o relatório médico. Analise a nota e as entidades "
    "extraídas para tomar essa decisão. Responda apenas com JSON válido."
)

SYSTEM_REPORT_GENERATION = (
    "Você é um assistente clínico de IA especializado em gerar relatórios "
    "médicos estruturados para médicos brasileiros. Use terminologia médica "
    "padrão, referencie diretrizes quando aplicável e nunca invente dados "
    "não presentes nas fontes. Responda apenas com JSON válido."
)

SYSTEM_REPORT_EVALUATION = (
    "Você é um avaliador de qualidade de relatórios médicos. Analise o "
    "relatório comparando-o com a nota clínica original e os dados do "
    "paciente. Seja objetivo e específico no feedback. "
    "Responda apenas com JSON válido."
)

EXPAND_NOTE_PROMPT = """\
Expanda a seguinte nota médica abreviada em uma nota clínica clara e completa. \
Expanda todas as abreviações, normalize sinais vitais e liste os achados explicitamente.

Nota do médico:
{note}

Retorne um objeto JSON com estes campos:
- "expanded_note": a nota clínica completa expandida
- "entities": lista de entidades médicas extraídas, cada uma com "text", "type" \
(um de: symptom, sign, medication, condition, vital_sign, procedure) e "original" \
(a abreviação ou forma abreviada usada)

Responda APENAS com JSON válido, sem texto adicional.
"""

REPORT_PROMPT = """\
Gere um relatório médico estruturado com base nas seguintes informações.

## Dados do Paciente
{patient_data}

## Nota Clínica
{note}

## Diretrizes Relevantes
{guidelines}

{refinement_context}

Retorne um objeto JSON com EXATAMENTE estas 10 seções:

1. "patient_summary": descrição breve do paciente (1-2 frases: idade, sexo, condições principais)
2. "findings": lista de achados clínicos da nota e dos dados do paciente
3. "assessment": avaliação clínica e raciocínio (referencie diretrizes quando aplicável)
4. "plan": lista de ações do plano de cuidado (medicamentos com doses, encaminhamentos, orientações)
5. "guideline_references": lista de trechos das diretrizes que fundamentam o plano
6. "diagnosticos_diferenciais": lista de diagnósticos diferenciais, cada um com breve justificativa \
clínica (ex.: "Insuficiência cardíaca descompensada — dispneia + edema + ortopneia")
7. "sinais_alarme": lista de sinais de alerta que o clínico deve monitorar ativamente \
(ex.: "Piora súbita da dispneia — avaliar descompensação aguda")
8. "acompanhamento": objeto com três campos:
   - "proxima_visita": prazo e objetivo da próxima consulta (ex.: "Retorno em 30 dias para reavaliação da PA")
   - "exames_a_repetir": lista de exames a solicitar no retorno (ex.: "HbA1c em 3 meses")
   - "sinais_para_escalar": lista de situações que justificam contato imediato ou ida a emergência
9. "interacoes_medicamentosas": lista de interações medicamentosas identificadas pelo sistema \
(pode ser vazia se nenhuma foi detectada; inclua apenas interações com evidência clínica)
10. "limitacoes": lista de limitações deste relatório (dados ausentes, baixa confiança, \
lacunas nas diretrizes disponíveis)

Se alguma seção não tiver dados disponíveis (marcada como "Não disponível"), indique \
essa limitação em "limitacoes". Não invente dados que não estejam presentes nas fontes.

Responda APENAS com JSON válido, sem texto adicional.
"""

ENTITY_EXTRACTION_PROMPT = """\
Extraia todas as entidades médicas desta nota clínica.

Nota:
{note}

Retorne um objeto JSON com um único campo "entities" — uma lista de objetos, cada um com:
- "text": a entidade conforme escrita na nota
- "type": um de symptom, sign, medication, condition, vital_sign, procedure, \
lab_result, allergy, family_history, social_history
- "normalized": o termo médico padrão (expandido, em português)

Use os tipos da seguinte forma:
- "symptom": queixa subjetiva relatada pelo paciente (ex.: dispneia, cefaleia)
- "sign": achado objetivo no exame físico (ex.: estertores, edema)
- "medication": fármaco em uso ou prescrito (ex.: losartana 50 mg)
- "condition": doença, diagnóstico ou comorbidade (ex.: HAS, DM2)
- "vital_sign": sinal vital com valor (ex.: PA 150x95, FC 88)
- "procedure": procedimento realizado ou planejado (ex.: ECG, cateterismo)
- "lab_result": resultado de exame laboratorial (ex.: HbA1c 8.2%, creatinina 1.4)
- "allergy": alergia ou reação adversa relatada (ex.: alergia a penicilina)
- "family_history": antecedente familiar relevante (ex.: pai com IAM aos 55)
- "social_history": hábito ou contexto social (ex.: tabagismo 20 maços/ano)

Se não houver entidades médicas, retorne {{"entities": []}}.

Exemplo para "Pct 65a HAS, PA 150x95, losartana 50mg":
{{
  "entities": [
    {{"text": "HAS", "type": "condition", "normalized": "Hipertensão arterial sistêmica"}},
    {{"text": "PA 150x95", "type": "vital_sign", "normalized": "Pressão arterial 150x95 mmHg"}},
    {{"text": "losartana 50mg", "type": "medication", "normalized": "Losartana 50 mg"}}
  ]
}}

Exemplo para "M 58a, dispneia aos esforços, estertores em bases, HbA1c 8.2, alérgico a penicilina, \
pai faleceu de IAM aos 60, tabagista 30 maços/ano":
{{
  "entities": [
    {{"text": "dispneia aos esforços", "type": "symptom", "normalized": "Dispneia aos esforços"}},
    {{"text": "estertores em bases", "type": "sign", "normalized": "Estertores em bases pulmonares"}},
    {{"text": "HbA1c 8.2", "type": "lab_result", "normalized": "Hemoglobina glicada 8,2%"}},
    {{"text": "alérgico a penicilina", "type": "allergy", "normalized": "Alergia a penicilina"}},
    {{"text": "pai faleceu de IAM aos 60", "type": "family_history", \
"normalized": "Histórico familiar de infarto agudo do miocárdio (pai, 60 anos)"}},
    {{"text": "tabagista 30 maços/ano", "type": "social_history", \
"normalized": "Tabagismo 30 maços-ano"}}
  ]
}}

Exemplo para "Paciente referido para ECG e cateterismo cardíaco":
{{
  "entities": [
    {{"text": "ECG", "type": "procedure", "normalized": "Eletrocardiograma"}},
    {{"text": "cateterismo cardíaco", "type": "procedure", "normalized": "Cateterismo cardíaco"}}
  ]
}}

Responda APENAS com JSON válido, sem texto adicional.
"""

SELF_RAG_DECISION_PROMPT = """\
Dada a seguinte nota clínica e as entidades já extraídas dela, decida se \
consultar diretrizes clínicas melhoraria a qualidade do relatório médico final.

## Nota Clínica
{note}

## Entidades Extraídas
{entities}

Responda com um objeto JSON:
- "needs_retrieval": true ou false
- "queries": se true, uma lista de 1 a 3 consultas curtas para buscar diretrizes relevantes

Responda APENAS com JSON válido, sem texto adicional.
"""

EVALUATE_REPORT_PROMPT = """\
Avalie a qualidade deste relatório médico comparando-o com a nota clínica \
original e os dados do paciente.

## Nota Clínica Original
{note}

## Dados do Paciente
{patient_data}

## Relatório a Avaliar
{report}

Pontue cada dimensão de 1 a 5 usando esta rubrica geral:
- 1 = Lacunas ou erros críticos que podem afetar decisões clínicas
- 2 = Informações importantes ausentes ou imprecisas
- 3 = Aceitável — cobre os pontos principais com lacunas menores
- 4 = Bom — abrangente com apenas omissões triviais
- 5 = Excelente — completo, preciso e bem referenciado

Dimensões a pontuar (7 no total):
- "completeness": todos os achados da nota estão contemplados no relatório?
- "accuracy": o raciocínio médico é correto e consistente com os dados?
- "guideline_adherence": o plano segue as diretrizes clínicas fornecidas?
- "clarity": o relatório é claro, bem estruturado e acionável?
- "safety": o relatório identifica sinais de alerta ("sinais_alarme") relevantes e \
sinaliza interações medicamentosas quando presentes? Pontue 1 se sinais_alarme está \
vazio sem justificativa, se há interações conhecidas não reportadas, ou se há conflito \
alergia-medicação não sinalizado.
- "follow_up_quality": o campo "acompanhamento" é acionável? Verifique se há prazo \
concreto para a próxima visita, exames específicos a repetir e critérios claros para \
escalar. Pontue 1 se acompanhamento estiver vazio ou genérico demais.
- "overall": pontuação geral considerando todas as dimensões anteriores

Retorne um objeto JSON com esses 7 campos, cada um contendo "score" (int 1-5) e \
"feedback" (string com observações específicas).

Responda APENAS com JSON válido, sem texto adicional.
"""


# ---------------------------------------------------------------------------
# Token estimation & context budget
# ---------------------------------------------------------------------------

MAX_INPUT_TOKENS = 5000  # conservative for Mistral 8K (leaves ~3K for output)


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses a simple heuristic: ~4 characters per token for Portuguese/English
    mixed text. This is approximate but sufficient for budget management.
    """
    return len(text) // 4


def truncate_to_budget(
    text: str,
    max_tokens: int,
    label: str = "conteúdo",
) -> str:
    """Truncate text to fit within a token budget, adding a note if truncated."""
    current = estimate_tokens(text)
    if current <= max_tokens:
        return text
    # Truncate to approximate character count
    max_chars = max_tokens * 4
    return text[:max_chars] + f"\n\n[{label} truncado por limitação de contexto]"


# ---------------------------------------------------------------------------
# Client functions
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds

# Per-task temperatures (Phase 12 Step 2): low for deterministic structured tasks,
# slightly higher for report generation where some narrative variation is fine.
TEMP_EXTRACTION = 0.1
TEMP_RAG_DECISION = 0.1
TEMP_REPORT = 0.3
TEMP_EVALUATION = 0.1

_chat_provider: ChatProvider | None = None


def _get_chat() -> ChatProvider:
    """Return the chat provider singleton, creating it on first call."""
    global _chat_provider
    if _chat_provider is None:
        _chat_provider = get_chat_provider()
    return _chat_provider


def generate(prompt: str, system_prompt: str = SYSTEM_MEDICAL) -> str:
    """Send a prompt to the LLM and return the raw text response."""
    return _get_chat().chat(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system_prompt,
        temperature=0.3,
    )


def generate_json(
    prompt: str,
    system_prompt: str = SYSTEM_MEDICAL,
    max_retries: int = MAX_RETRIES,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Send a prompt and parse the response as JSON.

    Tries native JSON format first; falls back to extracting
    a JSON block from the text response. Retries on transient failures
    with exponential backoff.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            raw = _get_chat().chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=system_prompt,
                temperature=temperature,
                json_mode=True,
            )
            return json.loads(raw)
        except (json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "JSON parse failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                exc,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "LLM call failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                exc,
            )

        if attempt < max_retries - 1:
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.info("Retrying in %.1fs...", delay)
            time.sleep(delay)

    # Final fallback: try without json_mode and extract manually
    try:
        raw = generate(prompt, system_prompt)
        return _extract_json(raw)
    except Exception:
        raise ValueError(
            f"All {max_retries} attempts failed for generate_json. Last error: {last_error}"
        ) from last_error


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM output."""
    # Try fenced code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Find first balanced JSON object using brace counting
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


# ---------------------------------------------------------------------------
# High-level clinical functions
# ---------------------------------------------------------------------------


def expand_note(note: str) -> dict[str, Any]:
    """Expand a doctor's shorthand note into structured clinical data."""
    prompt = EXPAND_NOTE_PROMPT.format(note=note)
    return generate_json(
        prompt,
        system_prompt=SYSTEM_ENTITY_EXTRACTION,
        temperature=TEMP_EXTRACTION,
    )


def extract_entities(note: str) -> dict[str, Any]:
    """Extract medical entities from a clinical note."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(note=note)
    raw = generate_json(
        prompt,
        system_prompt=SYSTEM_ENTITY_EXTRACTION,
        temperature=TEMP_EXTRACTION,
    )
    try:
        return EntityExtractionResult.model_validate(raw).model_dump()
    except ValidationError as exc:
        logger.warning("Validação Pydantic falhou em extract_entities: %s", exc)
        return raw


def generate_report(
    note: str,
    patient_data: str = "Não disponível",
    guidelines: str = "Não disponível",
    refinement_context: str = "",
) -> dict[str, Any]:
    """Generate a structured medical report from a clinical note and context."""
    prompt = REPORT_PROMPT.format(
        note=note,
        patient_data=patient_data,
        guidelines=guidelines,
        refinement_context=refinement_context,
    )
    raw = generate_json(
        prompt,
        system_prompt=SYSTEM_REPORT_GENERATION,
        temperature=TEMP_REPORT,
    )
    try:
        return ClinicalReport.model_validate(raw).model_dump()
    except ValidationError as exc:
        logger.warning("Validação Pydantic falhou em generate_report: %s", exc)
        return raw


def decide_retrieval(note: str, entities: list[dict]) -> dict[str, Any]:
    """Self-RAG: decide if guideline retrieval is needed."""
    prompt = SELF_RAG_DECISION_PROMPT.format(
        note=note,
        entities=json.dumps(entities, ensure_ascii=False, indent=2),
    )
    raw = generate_json(
        prompt,
        system_prompt=SYSTEM_RAG_DECISION,
        temperature=TEMP_RAG_DECISION,
    )
    try:
        return RetrievalDecision.model_validate(raw).model_dump()
    except ValidationError as exc:
        logger.warning("Validação Pydantic falhou em decide_retrieval: %s", exc)
        return raw


def evaluate_report(
    report: dict[str, Any],
    note: str = "",
    patient_data: str = "",
) -> dict[str, Any]:
    """Evaluate the quality of a generated medical report."""
    prompt = EVALUATE_REPORT_PROMPT.format(
        report=json.dumps(report, ensure_ascii=False, indent=2),
        note=note or "Não disponível",
        patient_data=patient_data or "Não disponível",
    )
    raw = generate_json(
        prompt,
        system_prompt=SYSTEM_REPORT_EVALUATION,
        temperature=TEMP_EVALUATION,
    )
    try:
        return ReportEvaluation.model_validate(raw).model_dump()
    except ValidationError as exc:
        logger.warning("Validação Pydantic falhou em evaluate_report: %s", exc)
        return raw
