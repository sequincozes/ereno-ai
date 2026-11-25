"""scripts/init_state.py — reset de estado do experimento (#17, M3).

Zera os artefatos gerados para começar um experimento do zero, **preservando**
os seeds versionados em ``data/`` (``baseline_dataset.csv`` e o golden
``iteration_history.json``). Roda sem Java e sem chave da Groq.

O que é apagado:
  - ``outputs/`` — datasets por iteração, respostas da LLM, configs sugeridas,
    ``metrics_history.csv``, ``iteration_history.json`` etc. (mantém ``.gitkeep``).
  - ``logs/``    — logs de execução (a menos que ``--keep-logs``).

O que **nunca** é tocado:
  - ``data/``    — seeds reprodutíveis (fonte da verdade do modo cacheado).

Uso::

    uv run python scripts/init_state.py             # limpa outputs/ e logs/
    uv run python scripts/init_state.py --dry-run    # só mostra o que faria
    uv run python scripts/init_state.py --keep-logs  # preserva logs/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from adversarial_ids.config.settings import (  # noqa: E402
    BASELINE_DATASET_PATH,
    GOLDEN_HISTORY_PATH,
    LOGS_DIR,
    OUTPUTS_DIR,
)

# Arquivos que devem sobreviver à limpeza de um diretório (versionados).
_KEEP_NAMES = {".gitkeep"}


def _clean_dir(directory: Path, dry_run: bool) -> int:
    """Remove o conteúdo de ``directory`` preservando ``_KEEP_NAMES``.

    Devolve quantos itens de topo foram (ou seriam) removidos. Recria o diretório
    vazio se ele não existir, para o loop poder gravar em seguida.
    """

    if not directory.exists():
        if dry_run:
            print(f"  [ok]   {directory} não existe — seria criado vazio.")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  [ok]   {directory} não existia — criado vazio.")
        return 0

    removed = 0
    for item in sorted(directory.iterdir()):
        if item.name in _KEEP_NAMES:
            continue

        action = "removeria" if dry_run else "removido"
        kind = "dir " if item.is_dir() else "file"
        print(f"  [{action}] {kind}: {item.relative_to(BASE_DIR)}")

        if not dry_run:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        removed += 1

    if removed == 0:
        print(f"  [ok]   {directory} já estava limpo.")

    return removed


def _check_seeds() -> None:
    """Avisa se os seeds versionados não estiverem presentes (não os apaga)."""

    for label, path in (
        ("baseline_dataset.csv", BASELINE_DATASET_PATH),
        ("iteration_history.json (golden)", GOLDEN_HISTORY_PATH),
    ):
        status = "presente" if path.exists() else "AUSENTE"
        marker = "ok" if path.exists() else "!!"
        print(f"  [{marker}]   seed {label}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reset de estado: limpa outputs/ (e logs/) preservando os seeds "
            "versionados em data/."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista o que seria removido, sem apagar nada.",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="Preserva o diretório logs/.",
    )
    args = parser.parse_args()

    mode = "DRY-RUN (nada será apagado)" if args.dry_run else "LIMPEZA"
    print("\n==============================")
    print(f"INIT STATE — {mode}")
    print("==============================")

    print(f"\nLimpando outputs/  ({OUTPUTS_DIR})")
    _clean_dir(OUTPUTS_DIR, dry_run=args.dry_run)

    if args.keep_logs:
        print(f"\nPreservando logs/  ({LOGS_DIR})  [--keep-logs]")
    else:
        print(f"\nLimpando logs/     ({LOGS_DIR})")
        _clean_dir(LOGS_DIR, dry_run=args.dry_run)

    print("\nSeeds versionados (preservados em data/):")
    _check_seeds()

    print("\nEstado " + ("inspecionado." if args.dry_run else "reinicializado."))


if __name__ == "__main__":
    main()
