"""Schema estrito do ataque ``injection`` (uc05).

Baseline sem a chave ``enabled`` (diferente da maioria dos outros ataques) e
com ``description``/``comment`` no topo — ambos precisam ser declarados ou
``extra="forbid"`` rejeitaria o próprio arquivo de baseline.

``cbStatus`` aqui é um **objeto** ``{"values": [...]}`` — no ``masquerade_fault``
é um inteiro solto (``Literal[0, 1]``). ``ttlMs`` aqui é um objeto ``{min, max}``
— no ``flooding`` (uc07) é um inteiro solto. Nenhum dos dois pode ser
compartilhado entre modelos.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from adversarial_ids.domain.attack_configs.ranges import FrozenConfig, IntRange


class CbStatusValues(FrozenConfig):
    values: list[Literal[0, 1]] = Field(min_length=1)


class InjectionConfig(FrozenConfig):
    """Configuração completa do ataque ``injection`` (uc05).

    ``stNum``/``sqNum``/``cbStatus``/``ttlMs``/``confRev`` só têm efeito físico
    quando ``injectionPattern == "synthetic"`` (o baseline usa ``"random"``,
    que clona mensagens legítimas e ignora essas faixas) — o modelo aceita a
    forma completa do arquivo de qualquer forma; é o catálogo de capacidades
    (``config/attack_capabilities.py``) que decide quais campos o pipeline
    intent-driven pode tocar.
    """

    attackType: Literal["injection"]
    description: str
    numInjectedMessages: int = Field(ge=1)
    randomSeed: int = Field(ge=0)
    injectionPattern: str
    comment: str

    stNum: IntRange
    sqNum: IntRange
    cbStatus: CbStatusValues
    ttlMs: IntRange
    confRev: IntRange
