"""CLI script to ingest clinical guidelines into Qdrant."""

import sys
from pathlib import Path

from aegis.config import settings
from aegis.rag.ingest import ingest_guidelines


def main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.guidelines_dir

    if not directory.exists():
        print(f"Erro: diretório não encontrado: {directory}")
        sys.exit(1)

    files = list(directory.glob("*.txt")) + list(directory.glob("*.pdf"))
    if not files:
        print(f"Nenhum documento encontrado em {directory}")
        sys.exit(1)

    print(f"Ingerindo documentos de: {directory}")
    print(f"Arquivos encontrados: {len(files)}")
    for f in sorted(files):
        print(f"  - {f.name}")

    count = ingest_guidelines(directory)
    print(f"\nIngestão concluída: {count} chunks armazenados no Qdrant.")


if __name__ == "__main__":
    main()
