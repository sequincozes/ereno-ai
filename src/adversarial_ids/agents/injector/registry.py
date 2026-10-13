"""Injector — as estratégias de injeção atrás de um contrato só.

O diagrama do journal desenha o Injector como **uma** caixa com três
estratégias (intent-, model- e data-driven). Até aqui, o que existia era o
intent-driven, e "trocar de injetor" significava trocar o pipeline inteiro com
``--engine``. Este registro faz a caixa existir: as três estratégias produzem a
mesma ``IntentSpec``, que segue pelos mesmos portões e pelo mesmo pipeline. O
que muda entre elas é **o contexto que o LLM lê** antes de propor o ataque.

Mesma forma dos outros registros (detector/E8, gerador, undersampler): a chave
é um ``Literal`` no domínio, o registro é checado contra ele na importação, e
cada execução grava um manifest — aqui, ``injector_manifest.json``.

Preparar é separado de construir
--------------------------------
``prepare_injector`` faz a parte determinística — lê o creator Java, perfila o
trace, monta as instruções e o manifest — sem importar ``agno`` e sem falar com
a Groq. ``build_injector`` só embrulha o resultado num agente. Assim um creator
que não existe, ou um trace sem classe normal, falha antes de qualquer chamada
paga, e os testes cobrem tudo menos o LLM.

O que este registro **não** é
-----------------------------
Um decisor. Nada aqui escolhe a estratégia a partir de uma métrica — quem
escolhe é quem roda o experimento (``--injector``). Mesmo raciocínio do
decisor de undersampling (E7): o que otimiza a métrica não escolhe a própria
etapa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adversarial_ids.agents.injector.java_sources import (
    list_source_attacks,
    load_creator_source,
)
from adversarial_ids.config.settings import ERENO_SOURCE_DIR
from adversarial_ids.domain.injector_manifest import (
    INJECTOR_KEYS,
    InjectorManifest,
    InjectorSource,
    base_attack_constraint_for,
    context_kind_for,
)

DEFAULT_INJECTOR_KEY = "intent"


class InjectorError(ValueError):
    """Erro acionável: a estratégia não pôde ser resolvida ou preparada."""


@dataclass(frozen=True)
class InjectorSpec:
    key: str
    description: str
    # Insumo obrigatório além do prompt: ``source_attack`` (model) ou
    # ``dataset_path`` (data). Vazio no intent.
    requires: tuple[str, ...] = ()


_REGISTRY: dict[str, InjectorSpec] = {
    "intent": InjectorSpec(
        key="intent",
        description=(
            "Intent-driven: o LLM traduz o pedido numa intenção sobre um dos "
            "ataques do catálogo."
        ),
    ),
    "model": InjectorSpec(
        key="model",
        description=(
            "Model-driven: o LLM lê o creator Java de um ataque existente e "
            "propõe um ataque novo na DSL do programmable."
        ),
        requires=("source_attack",),
    ),
    "data": InjectorSpec(
        key="data",
        description=(
            "Data-driven: o LLM lê o perfil de um trace já gerado e propõe um "
            "ataque novo na DSL do programmable."
        ),
        requires=("dataset_path",),
    ),
}

if tuple(_REGISTRY) != INJECTOR_KEYS:
    raise InjectorError(
        "Registro de injetores fora de sincronia com domain.injector_manifest."
        f"InjectorKey: registro={tuple(_REGISTRY)!r}, contrato={INJECTOR_KEYS!r}."
    )


def injector_spec(key: str) -> InjectorSpec:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise InjectorError(
            f"Injetor desconhecido: {key!r}. Registrados: {list(INJECTOR_KEYS)!r}."
        ) from None


def list_injector_keys() -> tuple[str, ...]:
    return tuple(_REGISTRY)


@dataclass(frozen=True)
class PreparedInjector:
    """Tudo que um injetor precisa antes de chamar o LLM.

    ``instructions`` é ``None`` no intent: o ``IntentAgent`` monta as dele.
    """

    manifest: InjectorManifest
    instructions: str | None = None
    extra_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)


def _manifest(key: str, **fields: Any) -> InjectorManifest:
    # Contexto e restrição vêm do domínio, nunca digitados aqui de novo.
    return InjectorManifest(
        injector=key,
        context_kind=context_kind_for(key),
        base_attack_constraint=base_attack_constraint_for(key),
        **fields,
    )


def prepare_injector(
    key: str,
    *,
    source_attack: str | None = None,
    dataset_path: str | Path | None = None,
    attack_label: str | None = None,
    source_dir: str | Path = ERENO_SOURCE_DIR,
) -> PreparedInjector:
    """Lê o contexto da estratégia ``key`` e monta instruções + manifest."""

    spec = injector_spec(key)
    given = {
        "source_attack": source_attack,
        "dataset_path": dataset_path,
    }
    missing = [name for name in spec.requires if given[name] is None]
    if missing:
        raise InjectorError(f"O injetor {key!r} precisa de {missing}.")
    extra = [name for name, value in given.items() if value is not None and name not in spec.requires]
    if extra or (attack_label is not None and key != "data"):
        raise InjectorError(
            f"O injetor {key!r} não usa {extra or ['attack_label']}; passá-lo "
            "sugeriria um contexto que o LLM não vai ler."
        )

    if key == "intent":
        return PreparedInjector(manifest=_manifest(key))

    # Imports tardios: ``context`` passa pelo pacote ``agents.intent``, que
    # carrega ``agno`` na importação — e a CLI precisa listar as chaves sem isso.
    from adversarial_ids.agents.injector.context import (
        build_injector_instructions,
        render_java_context,
    )

    if key == "model":
        if source_attack not in list_source_attacks():
            raise InjectorError(
                f"Sem creator Java para {source_attack!r}. "
                f"Disponíveis: {list(list_source_attacks())}."
            )
        source = load_creator_source(source_attack, source_dir)
        instructions = build_injector_instructions(
            key, render_java_context(source.attack_key, source.ref, source.text)
        )
        return PreparedInjector(
            manifest=_manifest(
                key,
                subject=source.attack_key,
                sources=(InjectorSource(ref=source.ref, sha256=source.sha256),),
                context_chars=len(instructions),
            ),
            instructions=instructions,
        )

    from adversarial_ids.core.trace_profile import profile_trace, render_trace_profile

    profile = profile_trace(dataset_path, attack_label=attack_label)
    instructions = build_injector_instructions(key, render_trace_profile(profile))
    return PreparedInjector(
        manifest=_manifest(
            key,
            subject=profile.attack_label,
            sources=(InjectorSource(ref=profile.trace_ref, sha256=profile.trace_sha256),),
            context_chars=len(instructions),
        ),
        instructions=instructions,
        extra_artifacts={"trace_profile.json": profile.model_dump(mode="json")},
    )


def build_injector(prepared: PreparedInjector, *, model_id: str) -> Any:
    """Embrulha ``prepared`` no agente real (importa ``agno``; chama a Groq)."""

    if prepared.instructions is None:
        from adversarial_ids.agents.intent.agent import IntentAgent

        return IntentAgent(model_id=model_id)

    from adversarial_ids.agents.injector.agent import ContextualInjectorAgent

    return ContextualInjectorAgent(
        model_id=model_id,
        instructions=prepared.instructions,
        manifest=prepared.manifest,
        extra_artifacts=prepared.extra_artifacts,
    )
