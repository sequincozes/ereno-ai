"""Leitura do código-fonte dos creators do ERENO, para o injetor model-driven.

O injetor model-driven dá ao LLM o creator Java de um ataque existente e pede
um ataque novo a partir dele. Este módulo resolve **qual** arquivo é esse, lê,
enxuga (pacote e imports não ensinam nada sobre o comportamento e custam TPM)
e devolve o texto com o hash do arquivo original — o hash vai para o
``InjectorManifest``, porque o mesmo caminho em outro commit do ERENO é outro
contexto.

Só os creators configuráveis (sufixo ``C``) entram: são os que o pipeline roda
(``Attack implementation mode: CONFIGURABLE``), e os que leem o mesmo JSON que o
compilador escreve. O ``programmable`` fica de fora de propósito — ler o
interpretador da DSL para escrever na própria DSL é circular, e não é um ataque.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_CREATORS_ROOT = "src/main/java/br/ufu/facom/ereno/attacks"

# attack_key (attacks_registry) -> creator configurável, relativo a _CREATORS_ROOT.
JAVA_CREATOR_FILES: dict[str, str] = {
    "masquerade_fault": "uc03/creator/MasqueradeFakeFaultCreatorC.java",
    "random_replay": "uc01/creator/RandomReplayCreatorC.java",
    "inverse_replay": "uc02/creator/InverseReplayCreatorC.java",
    "injection": "uc05/creator/InjectionCreatorC.java",
    "high_stnum": "uc06/creator/HighStNumInjectionCreatorC.java",
    "flooding": "uc07/creator/HighRateStNumInjectionCreatorC.java",
    "grayhole": "uc08/creator/GrayHoleVictimCreatorC.java",
    "delayed_replay": "uc10/creator/DelayedReplayCreatorC.java",
    "delayed_replay_backoff": "uc10/creator/BackoffDelayedReplayCreatorC.java",
    "delayed_replay_batch_dump": "uc10/creator/BatchDumpDelayedReplayCreatorC.java",
    "delayed_replay_double_drop": "uc10/creator/DoubleDropDelayedReplayCreatorC.java",
}

_NOISE = re.compile(r"^\s*(package|import)\s+[\w.*]+\s*;\s*$")


class JavaSourceError(ValueError):
    """Erro acionável: o creator pedido não pôde ser lido."""


@dataclass(frozen=True)
class CreatorSource:
    attack_key: str
    ref: str
    sha256: str
    text: str


def list_source_attacks() -> tuple[str, ...]:
    """Ataques cujo creator o injetor model-driven sabe ler."""

    return tuple(JAVA_CREATOR_FILES)


def _compact(raw: str) -> str:
    lines = [line.rstrip() for line in raw.splitlines() if not _NOISE.match(line)]
    compacted: list[str] = []
    for line in lines:
        # Uma linha em branco separa; duas seguidas só gastam token.
        if not line and (not compacted or not compacted[-1]):
            continue
        compacted.append(line)
    return "\n".join(compacted).strip() + "\n"


def load_creator_source(attack_key: str, source_dir: str | Path) -> CreatorSource:
    """Lê o creator configurável de ``attack_key`` sob ``source_dir`` (o repo do ERENO)."""

    relative = JAVA_CREATOR_FILES.get(attack_key)
    if relative is None:
        raise JavaSourceError(
            f"Sem creator Java mapeado para {attack_key!r}. "
            f"Disponíveis: {list(JAVA_CREATOR_FILES)}."
        )

    path = Path(source_dir) / _CREATORS_ROOT / relative
    if not path.exists():
        raise JavaSourceError(
            f"Creator de {attack_key!r} não encontrado em {path}. O injetor "
            "model-driven lê o código-fonte do ERENO: aponte ERENO_SOURCE_DIR para "
            "o checkout do ERENO-2.0."
        )

    raw = path.read_bytes()
    return CreatorSource(
        attack_key=attack_key,
        ref=f"{_CREATORS_ROOT}/{relative}",
        sha256=hashlib.sha256(raw).hexdigest(),
        text=_compact(raw.decode("utf-8")),
    )
