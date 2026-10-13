"""InjectorManifest — de onde veio a intenção de uma execução.

Quinto manifest da família, e o primeiro a montante do **compilador**.
``GeneratorManifest`` registra qual simulador produziu o trace; este registra
**qual estratégia de injeção propôs o ataque**, e sobre que contexto o LLM
raciocinou para propô-lo. É persistido no estágio ``intent`` como
``injector_manifest.json``.

As três estratégias do diagrama do journal
------------------------------------------
- ``intent`` (intent-driven): o LLM lê o pedido do usuário e o catálogo inteiro
  de capacidades — é o pipeline que já existia.
- ``model`` (model-driven): o LLM lê o **código Java** de um creator do ERENO e
  propõe um ataque **novo** a partir dele.
- ``data`` (data-driven): o LLM lê o **perfil determinístico de um trace** já
  gerado (o que separa a classe de ataque do tráfego normal) e propõe um ataque
  novo a partir do comportamento observado.

As três produzem a mesma coisa — uma ``IntentSpec`` — e passam pelos mesmos
portões. O que as separa é só o contexto. É isso que faz delas estratégias de
um Injector único, e não três pipelines.

Por que ``model``/``data`` ficam presos ao ``programmable``
-----------------------------------------------------------
"Ataque novo" não pode significar código novo: o guardrail central do projeto
proíbe que saída de LLM escreva no runtime do ERENO, e um creator Java escrito
pelo modelo seria exatamente isso. A forma de um ataque novo existir sem código
novo é a DSL do ataque programável (uc11) — regras declarativas que o compilador
valida campo a campo e o schema valida em conjunto. Então ``base_attack`` é
fixado em ``programmable`` para as duas, e ``base_attack_constraint`` é o campo
que diz isso no artefato. Gerar Java de verdade é o épico E12, fora do escopo.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

InjectorKey = Literal["intent", "model", "data"]

INJECTOR_KEYS: tuple[str, ...] = get_args(InjectorKey)

# Que tipo de contexto cada estratégia entrega ao LLM, além do pedido.
ContextKind = Literal["catalog", "java_source", "trace_profile"]

_CONTEXT_BY_INJECTOR: dict[str, str] = {
    "intent": "catalog",
    "model": "java_source",
    "data": "trace_profile",
}

# O ``base_attack`` que a estratégia é obrigada a produzir. ``None`` = o LLM
# escolhe no catálogo inteiro. Ver o bloco acima para o porquê do programmable.
_CONSTRAINT_BY_INJECTOR: dict[str, str | None] = {
    "intent": None,
    "model": "programmable",
    "data": "programmable",
}


def context_kind_for(injector: str) -> str:
    return _CONTEXT_BY_INJECTOR[injector]


def base_attack_constraint_for(injector: str) -> str | None:
    return _CONSTRAINT_BY_INJECTOR[injector]


class InjectorSource(BaseModel):
    """Um insumo que o LLM leu: um arquivo Java, um trace, com o hash dele.

    O hash é o que torna a execução auditável: o mesmo caminho em outro commit
    do ERENO, ou um trace regenerado, é outro contexto — e o manifest diz qual.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InjectorManifest(BaseModel):
    """Estratégia de injeção + o contexto sobre o qual o LLM propôs o ataque."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    injector: InjectorKey
    context_kind: ContextKind
    base_attack_constraint: str | None = None

    # Para ``model``: o ataque cujo creator foi lido. Para ``data``: o rótulo da
    # classe de ataque perfilada no trace. ``None`` no ``intent``.
    subject: str | None = None
    sources: tuple[InjectorSource, ...] = ()
    # Tamanho das instruções entregues ao LLM (papel + catálogo + contexto), em
    # caracteres. É o que pesa no TPM da conta (ver docs/injectors.md).
    context_chars: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _shape_matches_the_injector(self) -> "InjectorManifest":
        """Contexto e restrição são da estratégia, não escolha de quem grava.

        Um manifest ``model`` sem restrição descreveria um injetor que escreve
        ataques fora da DSL — o que este repo não tem e não pode ter.
        """

        expected = (
            _CONTEXT_BY_INJECTOR[self.injector],
            _CONSTRAINT_BY_INJECTOR[self.injector],
        )
        actual = (self.context_kind, self.base_attack_constraint)
        if actual != expected:
            raise ValueError(
                f"injector={self.injector!r} é (context_kind, base_attack_constraint)="
                f"{expected!r}, não {actual!r}."
            )
        if self.injector != "intent" and (self.subject is None or not self.sources):
            raise ValueError(
                f"injector={self.injector!r} precisa de subject e de ao menos uma "
                "fonte: é o contexto que separa esta estratégia do intent-driven."
            )
        return self
