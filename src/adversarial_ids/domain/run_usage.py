"""AgentUsage — quanto uma chamada de LLM custou, em número medido (épico E11).

``LoopRecord.cost_usd`` está declarado desde o E3 e nunca foi populado por
ninguém. A linha "Operação" da tabela de aceite cobra "execução cabe no
orçamento definido", e não dá para afirmar isso sobre um campo que é sempre
``None``.

## Token é medido; dólar é derivado, e nem sempre

``input_tokens``/``output_tokens`` vêm da resposta do provedor: é contagem, não
estimativa. ``cost_usd`` é opcional de propósito — só existe quando o provedor
informa, e o repo **não** carrega tabela de preço própria. Uma tabela dessas
envelhece em silêncio (o ``MODEL_ID`` default deste projeto já precisou mudar
porque a Groq aposentou o modelo), e um custo errado é pior que custo ausente:
ele parece que serve para decidir.

Por isso o orçamento operacional se mede em tokens, que é o que sempre se sabe.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentUsage(BaseModel):
    """O consumo de uma chamada de LLM, ou a soma de várias."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_empty(self) -> bool:
        """Nenhum token e nenhum custo — o provedor não informou nada."""

        return self.total_tokens == 0 and self.cost_usd is None

    def __add__(self, other: "AgentUsage") -> "AgentUsage":
        """Soma dois consumos, preservando a distinção entre 0 e desconhecido.

        ``cost_usd`` só volta preenchido se **algum** dos lados o trouxe: somar
        um custo conhecido com um desconhecido tratando o segundo como zero
        produziria um total que parece completo e não é.
        """

        costs = [c for c in (self.cost_usd, other.cost_usd) if c is not None]

        return AgentUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=sum(costs) if costs else None,
        )

    @model_validator(mode="after")
    def _cost_without_tokens_is_suspicious(self) -> "AgentUsage":
        """Custo sem token nenhum não vem de chamada alguma que aconteceu."""

        if self.cost_usd is not None and self.cost_usd > 0 and self.total_tokens == 0:
            raise ValueError(
                f"cost_usd={self.cost_usd} sem token contabilizado: "
                "o consumo não corresponde a nenhuma chamada."
            )

        return self
