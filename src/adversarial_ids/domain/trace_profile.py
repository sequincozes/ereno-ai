"""TraceProfile — o que separa a classe de ataque do tráfego normal num trace.

É o contexto do injetor data-driven: em vez de o LLM ler 100 mil linhas de
CSV, lê este resumo, calculado de forma determinística por
``core/trace_profile.py``. Persistido como ``trace_profile.json`` ao lado do
``injector_manifest.json``, para que a pergunta "o que o modelo viu antes de
propor este ataque?" tenha resposta no artefato, e não só no prompt.

A medida é a mesma do piloto E0 (``scripts/run_pilot.py``): diferença
absoluta das médias dividida pelo desvio combinado das duas classes. Colunas
constantes nos dois lados em valores diferentes têm separação total e desvio
zero — entram em ``constant_shift``, por presença, e não como infinito.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureContrast(BaseModel):
    """Uma coluna numérica, vista nas duas classes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str = Field(min_length=1)
    attack_mean: float
    normal_mean: float
    attack_std: float = Field(ge=0.0)
    normal_std: float = Field(ge=0.0)
    # ``None`` quando os dois lados são constantes no mesmo valor: não há o que
    # separar, e zero seria afirmar uma medida que não foi feita.
    separation: float | None = Field(default=None, ge=0.0)


class TraceProfile(BaseModel):
    """Resumo determinístico de um trace, para o injetor data-driven."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    trace_ref: str = Field(min_length=1)
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_column: str = Field(min_length=1)
    attack_label: str = Field(min_length=1)
    normal_label: str = Field(min_length=1)
    class_counts: dict[str, int]

    # As colunas que mais separam as classes, em ordem decrescente.
    top_features: tuple[FeatureContrast, ...]
    # Constantes nos dois lados, em valores diferentes: separação total.
    constant_shift: tuple[FeatureContrast, ...] = ()
    # As colunas que a DSL do programmable consegue mexer (ou que respondem
    # diretamente a ela), sempre presentes mesmo fora do top — é o vocabulário
    # com que o LLM vai escrever o ataque, então ele precisa vê-lo.
    lever_features: tuple[FeatureContrast, ...] = ()
