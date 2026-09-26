"""Schema estrito do ataque ``programmable`` (uc11).

Diferente dos outros dez: o comportamento não é um conjunto fixo de campos, e
sim uma **lista de regras de mutação**. As regras vivem sob ``rules`` como um
**objeto** chaveado por slot (``{"r0": {...}, "r1": {...}}``), não um array —
o modelo de caminhos do repo é dict pontilhado (``rules.r0.fraction``) e não
tem suporte a índice de array; um dict de slots mantém as duas pontas (Python e
o creator Java) sobre o mesmo modelo de caminho.

Cada regra é ``{op, field, value, fraction}``:
- ``op``: ``set`` | ``add`` | ``scale`` — o que fazer com o campo;
- ``field``: um dos campos GOOSE inteiros que o creator Java sabe tocar;
- ``value``: o operando;
- ``fraction``: probabilidade em [0, 1] de a regra disparar por mensagem.

O ``field`` fica como ``str`` (e não um enum) de propósito: o enum canônico de
campos mutáveis é a allowlist da **capacidade**
(``config/capabilities/programmable.py``), que é o que o portão determinístico
aplica. Duplicá-lo aqui criaria duas fontes de verdade que envelheceriam
separadas — o schema garante o *shape*, a capacidade garante o *domínio*.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from adversarial_ids.domain.attack_configs.ranges import FrozenConfig, Probability


class MutationRule(FrozenConfig):
    """Uma regra de mutação de um slot do ataque programável."""

    op: Literal["set", "add", "scale"]
    field: str = Field(min_length=1)
    value: float
    fraction: Probability


class ProgrammableConfig(FrozenConfig):
    """Configuração completa do ataque ``programmable`` (uc11)."""

    attackType: Literal["programmable"]
    enabled: bool
    # Objeto de slots, não lista. min_length=1: um ataque sem regra nenhuma não
    # muta nada e seria indistinguível do tráfego benigno (o gate E4 o reprovaria
    # de qualquer forma, mas falhar já no schema é mais cedo e mais claro).
    rules: dict[str, MutationRule] = Field(min_length=1)
