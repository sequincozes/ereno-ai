"""GeneratorManifest — de onde o trace de uma execução veio.

Quarto manifest da mesma família, e o único que olha para **montante** do
detector. ``FeatureManifest`` (E6) registra como as features foram preparadas,
``SelectionManifest`` (E7) quais sobreviveram, ``DetectorManifest`` (E8) qual
modelo aprendeu sobre elas — e este registra **qual gerador produziu o trace
sobre o qual tudo isso aconteceu**. É persistido no estágio de geração do
pipeline intent-driven como ``generator_manifest.json``, ao lado dos outros.

Por que ele não repete o ``DatasetBundle``: o bundle responde *o que está no
trace* (colunas, classes, contagem por classe, hash do conteúdo); este responde
*como o trace foi produzido*. A mesma divisão que ``DetectionReport`` (quão bem
o detector foi) e ``DetectorManifest`` (que detector era). Sem os dois lado a
lado, dois resultados com números diferentes não provam se a diferença veio da
config do ataque, do simulador ou da semente.

``produces_per_config_variation`` é o campo que existe para não deixar mentir
-----------------------------------------------------------------------------
O gerador ``cached`` devolve o **mesmo arquivo versionado** em toda execução:
ele serve para exercitar o fluxo inteiro sem Java, não para medir o efeito
físico de uma config. Consequência que já estava documentada em prosa e agora
é legível por máquina: uma campanha em modo cacheado estaciona na rodada 2
(``no_improvement``) porque o ``DetectionReport`` não se move, e qualquer
afirmação do tipo "esta intenção derrubou o recall" lida a partir de um run
com ``produces_per_config_variation=False`` é uma afirmação sobre nada.

Vocabulário de geradores
------------------------
Mora aqui, e não em ``core/generators.py``, pelo mesmo motivo do
``DetectorKey``: o domínio não depende do core — é o core que valida seu
registro contra este contrato na importação, nunca o contrário. A CLI lista as
opções de ``--generator-mode`` sem arrastar subprocesso nem Java para o import.

As chaves são ``cached`` e ``jar`` por compatibilidade com a CLI e as env vars
que já existiam; ``simulator`` é o campo que diz o que de fato produziu o
trace (``"none"`` para o replay versionado, ``"ereno"`` para o JAR). Um backend
novo — Mininet, libiec61850 — entra como uma chave a mais aqui e um
``GeneratorSpec`` a mais no registro, sem tocar em quem consome.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

GeneratorKey = Literal["cached", "jar"]

GENERATOR_KEYS: tuple[str, ...] = get_args(GeneratorKey)

# O que de fato produz o trace de cada gerador registrado. ``none`` não é
# ausência de valor: é a afirmação de que nenhum simulador rodou e o trace veio
# de um arquivo versionado.
_SIMULATOR_BY_GENERATOR: dict[str, str] = {
    "cached": "none",
    "jar": "ereno",
}

# Rodar este gerador exige a JVM e o JAR do ERENO em ``generator_runtime/``?
_JAVA_BY_GENERATOR: dict[str, bool] = {
    "cached": False,
    "jar": True,
}

# O trace muda quando a config do ataque muda? Ver o bloco acima — é o campo
# que separa "medi o efeito da intenção" de "exercitei o fluxo".
_VARIATION_BY_GENERATOR: dict[str, bool] = {
    "cached": False,
    "jar": True,
}


class GeneratorManifest(BaseModel):
    """Gerador que produziu o trace + as condições em que ele rodou."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    generator: GeneratorKey
    # Qual simulador de fato rodou. Redundante com ``generator`` hoje, porque há
    # um backend por simulador; deixa de ser quando dois backends compartilharem
    # um simulador (ex.: ERENO via JAR e ERENO embarcado).
    simulator: str = Field(min_length=1)
    requires_java: bool
    produces_per_config_variation: bool

    # -- O que foi gerado --------------------------------------------------- #
    attack_key: str = Field(min_length=1)
    # Segmento do action config do ERENO. ``None`` nos backends que não têm o
    # conceito — é o caso do ``cached``, e seria o de um backend de rede.
    segment_name: str | None = None

    # -- Sob que condições -------------------------------------------------- #
    # Semente determinística repassada ao backend (Fase 2.0). ``None`` = RNG do
    # relógio, isto é, duas execuções da mesma config não são comparáveis byte a
    # byte. Fica registrado porque é o que separa uma réplica de uma repetição.
    random_seed: int | None = None
    duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _capabilities_match_the_generator(self) -> "GeneratorManifest":
        """As capacidades são do backend, não escolha de quem grava o manifest.

        Um manifest que diga que o ``cached`` varia por config descreve um
        gerador que não existe — e é exatamente a afirmação que faria alguém
        ler um platô de campanha como resultado experimental.
        """

        expected = (
            _SIMULATOR_BY_GENERATOR[self.generator],
            _JAVA_BY_GENERATOR[self.generator],
            _VARIATION_BY_GENERATOR[self.generator],
        )
        actual = (self.simulator, self.requires_java, self.produces_per_config_variation)
        if actual != expected:
            raise ValueError(
                f"generator={self.generator!r} é "
                f"(simulator, requires_java, produces_per_config_variation)="
                f"{expected!r}, não {actual!r}."
            )
        return self
