"""Vocabulário de undersampling e a política que decide qual usar (épico E7).

O E7 entregou o undersampling como uma chave de ambiente com dois valores
(``none``/``random``), ligada por quem roda o experimento. O diagrama de
arquitetura do journal pede outra coisa na caixa do Pre-Processor:
undersamplers **intent-driven** e **policy-driven** — nomes que não descrevem
algoritmos, e sim **quem decide**. Este módulo é a metade de domínio dessa
diferença: o vocabulário das estratégias, a política declarada, e a decisão
registrada com a razão dela.

Quem decide, e o que foi deliberadamente deixado de fora
--------------------------------------------------------
``UndersamplingDecider`` tem dois valores, e nenhum deles é "o laço":

- ``config``: a estratégia veio fixada pelo chamador (env var, CLI, argumento).
  É o comportamento de antes, agora nomeado em vez de implícito.
- ``policy``: uma regra **declarada antes da execução** resolve a estratégia a
  partir do desequilíbrio observado na partição de TREINO. Determinística,
  auditável, e o artefato guarda a razão por escrito.

Não existe ``feedback``/``intent``. Um laço que ajusta o próprio
pré-processamento até a métrica melhorar deixa de medir o que o detector
enfrenta: o número que ele reporta passa a ser propriedade conjunta do ataque e
do balanceamento escolhido para favorecê-lo, e nenhuma rodada volta a ser
comparável com outra. Pela mesma razão o Red Team não escolhe o undersampler —
deixar a intenção do atacante configurar o treino do defensor é o atacante
enfraquecendo o detector por um caminho que não é o ataque.

A recusa é estrutural, não documental: ``core/undersampling_policy.py``
``decide_undersampling`` recebe contagem de classes e política, e mais nada.
Não há parâmetro por onde um ``DetectionReport`` entre.

Por que a política corrige para uma razão, e não para o equilíbrio
------------------------------------------------------------------
Balancear até a classe minoritária, num trace cuja prevalência de ataque pode
ser de 1% (o piso do E4), descarta ~99% das linhas benignas — o detector passa
a treinar num mundo que não existe e o custo é pago em falso positivo. A
política declara um **teto de desequilíbrio** (``max_imbalance_ratio``): abaixo
dele não mexe em nada, acima dele reduz a maioria até ``target_ratio`` vezes a
minoritária. ``target_ratio=1.0`` é exatamente o balanceamento total, então a
estratégia antiga continua sendo um caso particular desta, e não um ramo à
parte.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

# As estratégias registradas. Mora aqui, e não em ``core/undersamplers.py``,
# pelo mesmo motivo de ``DetectorKey`` e ``GeneratorKey``: o domínio não
# depende do core — é o core que valida seu registro contra este contrato na
# importação, nunca o contrário.
UndersamplerKey = Literal["none", "random", "ratio"]

UNDERSAMPLER_KEYS: tuple[str, ...] = get_args(UndersamplerKey)

# Quem escolheu a estratégia. Ver o docstring do módulo para o valor que não
# existe aqui de propósito.
UndersamplingDecider = Literal["config", "policy"]

UNDERSAMPLING_DECIDERS: tuple[str, ...] = get_args(UndersamplingDecider)

# A estratégia pode remover linhas da partição de treino?
_CHANGES_ROWS_BY_UNDERSAMPLER: dict[str, bool] = {
    "none": False,
    "random": True,
    "ratio": True,
}

# A estratégia precisa de um ``target_ratio`` para existir? ``random`` não
# precisa porque o teto dele é a própria classe minoritária (razão 1.0 fixa);
# ``ratio`` sem razão alvo não descreve corte nenhum.
_TARGET_RATIO_BY_UNDERSAMPLER: dict[str, bool] = {
    "none": False,
    "random": False,
    "ratio": True,
}


def undersampler_changes_rows(key: str) -> bool:
    """A estratégia ``key`` pode remover linhas de treino?"""

    return _CHANGES_ROWS_BY_UNDERSAMPLER[key]


def undersampler_requires_target_ratio(key: str) -> bool:
    """A estratégia ``key`` exige ``target_ratio`` para ficar reproduzível?"""

    return _TARGET_RATIO_BY_UNDERSAMPLER[key]


class UndersamplingPolicy(BaseModel):
    """Regra declarada **antes** da execução, que resolve a estratégia.

    Os dois números são o contrato inteiro: abaixo de ``max_imbalance_ratio``
    não se reamostra nada (o desequilíbrio está dentro do que o experimento
    aceita); acima dele a maioria é reduzida até ``target_ratio`` vezes a
    minoritária.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    # Desequilíbrio (maioria / minoritária) tolerado sem reamostrar.
    max_imbalance_ratio: float = Field(default=10.0, ge=1.0)
    # Desequilíbrio alvo quando a política decide reamostrar. 1.0 = equilíbrio.
    target_ratio: float = Field(default=3.0, ge=1.0)

    @model_validator(mode="after")
    def _target_is_at_least_as_strict_as_the_ceiling(self) -> "UndersamplingPolicy":
        if self.target_ratio > self.max_imbalance_ratio:
            raise ValueError(
                f"target_ratio ({self.target_ratio}) não pode ser maior que "
                f"max_imbalance_ratio ({self.max_imbalance_ratio}): a política "
                "estaria reamostrando para um desequilíbrio que ela própria "
                "declara intolerável, e o resultado seguiria acima do teto."
            )
        return self


class UndersamplingDecision(BaseModel):
    """O que foi decidido, por quem, e por quê — persistido no SelectionManifest.

    ``observed_imbalance`` é medido sobre a partição de **treino** antes de
    qualquer reamostragem. Ele é registrado mesmo quando a decisão veio da
    configuração: é o número que permite ler, meses depois, se aquela escolha
    fixa fazia sentido para aquele dataset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    strategy: UndersamplerKey
    decided_by: UndersamplingDecider
    target_ratio: float | None = Field(default=None, ge=1.0)
    observed_imbalance: float = Field(ge=1.0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _target_ratio_matches_the_strategy(self) -> "UndersamplingDecision":
        needs_ratio = undersampler_requires_target_ratio(self.strategy)
        if needs_ratio and self.target_ratio is None:
            raise ValueError(
                f"strategy={self.strategy!r} exige target_ratio: sem ele a "
                "decisão não descreve o corte que foi aplicado."
            )
        if not needs_ratio and self.target_ratio is not None:
            raise ValueError(
                f"strategy={self.strategy!r} não usa target_ratio "
                f"({self.target_ratio!r}) — registrá-lo descreveria um corte "
                "que não aconteceu."
            )
        return self
