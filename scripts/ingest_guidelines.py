"""CLI script to ingest clinical guidelines into Qdrant."""

import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

from aegis.config import settings
from aegis.rag.ingest import ingest_guidelines


def _check_qdrant() -> bool:
    """Return True if Qdrant is reachable."""
    parsed = urlparse(settings.qdrant_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6333
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.guidelines_dir

    if not directory.exists():
        print(f"Erro: diretório não encontrado: {directory}")
        sys.exit(1)

    files = list(directory.glob("*.txt")) + list(directory.glob("*.pdf"))
    if not files:
        print(f"Nenhum documento encontrado em {directory}")
        sys.exit(1)

    if not _check_qdrant():
        print(f"Erro: Qdrant não está acessível em {settings.qdrant_url}")
        print()
        print("Inicie o container com:")
        print("  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        print()
        print("Ou, se o container já existe mas está parado:")
        print("  docker start qdrant")
        sys.exit(1)

    print(f"Ingerindo documentos de: {directory}")
    print(f"Arquivos encontrados: {len(files)}")
    for f in sorted(files):
        print(f"  - {f.name}")

    count = ingest_guidelines(directory)
    print(f"\nIngestão concluída: {count} chunks armazenados no Qdrant.")


if __name__ == "__main__":
    main()
