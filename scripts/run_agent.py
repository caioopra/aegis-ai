"""CLI entry point to run the clinical agent on a doctor's note."""

import argparse
import json

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
    result = graph.invoke({"patient_note": args.note})

    if args.verbose:
        print("-" * 60)
        print("ENTIDADES EXTRAÍDAS:")
        for e in result.get("extracted_entities", []):
            print(f"  - {e.get('text', '')} ({e.get('type', '')})")

        print(f"\nPaciente identificado: {result.get('patient_id', 'N/A')}")
        print(f"Retrieval necessário: {result.get('needs_retrieval', 'N/A')}")

        if result.get("retrieval_queries"):
            print("Queries de busca:")
            for q in result["retrieval_queries"]:
                print(f"  - {q}")

        if result.get("guidelines"):
            print(f"\nDiretrizes encontradas:\n{result['guidelines'][:500]}...")

        if result.get("patient_data"):
            print(f"\nDados do paciente:\n{result['patient_data'][:500]}...")
        print("-" * 60)

    report = result.get("report", {})
    evaluation = result.get("evaluation", {})

    print("\n" + "=" * 60)
    print("RELATÓRIO CLÍNICO")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("AVALIAÇÃO DE QUALIDADE")
    print("=" * 60)
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
