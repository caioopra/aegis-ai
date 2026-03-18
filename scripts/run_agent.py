"""CLI entry point to run the clinical agent on a doctor's note."""

import argparse
import json
import time

from aegis.agent.graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AegisNode — Agente clínico para notas médicas",
    )
    parser.add_argument(
        "--note",
        type=str,
        required=True,
        help="Nota clínica do médico (entre aspas)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar estado intermediário de cada etapa",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AegisNode — Agente Clínico")
    print("=" * 60)
    print(f"\nNota recebida:\n  {args.note}\n")

    graph = build_graph()
    state: dict = {}
    t0 = time.perf_counter()
    prev_time = t0

    for step in graph.stream({"patient_note": args.note}):
        now = time.perf_counter()
        elapsed = now - prev_time
        prev_time = now

        for node_name, node_output in step.items():
            state.update(node_output)

            if args.verbose:
                _print_node_verbose(node_name, node_output, elapsed)
            else:
                _print_node_brief(node_name, elapsed)

    total = time.perf_counter() - t0

    report = state.get("report", {})
    evaluation = state.get("evaluation", {})

    print("\n" + "=" * 60)
    print("RELATÓRIO CLÍNICO")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("AVALIAÇÃO DE QUALIDADE")
    print("=" * 60)
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))

    # Summary line
    warnings = state.get("warnings", [])
    retry_count = state.get("retry_count", 0)
    match_type = state.get("patient_id_match_type", "?")
    rag_conf = state.get("retrieval_confidence", 0.0)

    print("\n" + "-" * 60)
    print(
        f"Tempo total: {total:.1f}s | Retries: {retry_count} | "
        f"Paciente: {match_type} | RAG conf: {rag_conf:.2f}"
    )
    if warnings:
        print(f"Avisos ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    print("-" * 60)


def _print_node_brief(node_name: str, elapsed: float) -> None:
    print(f"  [{elapsed:5.1f}s] {node_name}")


def _print_node_verbose(node_name: str, output: dict, elapsed: float) -> None:
    print(f"\n--- {node_name} ({elapsed:.1f}s) ---")

    if node_name == "parse_note":
        entities = output.get("extracted_entities", [])
        print(
            f"  Paciente: {output.get('patient_id', 'N/A')} "
            f"({output.get('patient_id_match_type', '?')})"
        )
        print(f"  Entidades ({len(entities)}):")
        for e in entities:
            print(f"    - {e.get('text', '')} ({e.get('type', '')})")

    elif node_name == "decide_retrieval":
        print(f"  Retrieval: {output.get('needs_retrieval', '?')}")
        for q in output.get("retrieval_queries", []):
            print(f"    query: {q}")

    elif node_name == "retrieve_guidelines":
        conf = output.get("retrieval_confidence", 0.0)
        guidelines = output.get("guidelines", "")
        print(f"  Confiança: {conf:.2f}")
        print(f"  Diretrizes: {guidelines[:200]}...")

    elif node_name == "fetch_patient_data":
        data = output.get("patient_data", "")
        print(f"  Dados ({len(data)} chars): {data[:200]}...")

    elif node_name == "generate_report":
        report = output.get("report", {})
        print(f"  Seções: {list(report.keys())}")

    elif node_name == "evaluate_report":
        ev = output.get("evaluation", {})
        overall = ev.get("overall", {})
        print(f"  Score: {overall.get('score', '?')}/5 — {overall.get('feedback', '')}")

    elif node_name == "increment_retry":
        print(f"  Retry #{output.get('retry_count', '?')}")

    warnings = output.get("warnings", [])
    if warnings:
        new_warnings = [
            w
            for w in warnings
            if node_name.replace("_", " ") in w.lower() or node_name.split("_")[0] in w.lower()
        ]
        for w in new_warnings:
            print(f"  ⚠ {w}")


if __name__ == "__main__":
    main()
