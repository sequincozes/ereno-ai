"""Generator — interface única para os backends que produzem o trace.

O mesmo movimento que o épico E8 fez com o detector, agora a montante: o
``IdsEvaluator`` deixou de ter um ``RandomForestClassifier`` embutido e passou
a receber um ``Detector`` resolvido por chave; aqui o pipeline deixa de ter o
ERENO como única fonte possível de trace e passa a resolver um **gerador** por
chave, com capacidades declaradas e um manifest persistível.

Por que agora, com um backend só
---------------------------------
Não é para ter Mininet amanhã. É porque hoje "qual gerador rodou" não é uma
pergunta que o sistema saiba responder a partir dos artefatos: o estágio se
chama ``ereno`` no vocabulário do ``LoopRecord``, o modo cacheado se disfarça
de execução normal, e a diferença entre "medi o efeito da config" e "exercitei
o fluxo" vive em prosa de documentação. Um registro com capacidades declaradas
e um ``GeneratorManifest`` por execução transforma isso em dado auditável — e,
de quebra, faz o segundo backend ser aditivo em vez de cirúrgico.

O que este módulo **não** faz: nenhuma geração. Ele resolve uma chave num
``GeneratorSpec``, constrói o backend concreto a partir de um
``GeneratorRequest`` e descreve o que foi feito. Quem gera é o backend.

Geradores registrados
---------------------
- ``cached`` (default): serve ``data/baseline_dataset.csv`` — um trace
  benigno+ataque já versionado. Não usa Java e **não varia com a config**: é o
  que permite rodar o loop inteiro, os testes e o CI sem o JAR, e é por isso
  que nenhuma métrica lida dele mede o efeito de uma intenção.
- ``jar``: executa o gerador ERENO (Java) sobre o attack config compilado.
  É o único em que uma campanha é fisicamente significativa.

Honestidade sobre o estado atual
--------------------------------
Os dois backends registrados são implementados pela **mesma** classe
(``GeneratorRunner``), que já cumpre o protocolo — a diferença entre eles é
ter ou não ``cached_dataset_path``. Isto é descrição, não arquitetura
aspiracional: a costura que este módulo abre vale para o **terceiro** backend,
que implementa ``GeneratorLike`` por conta própria e entra com um
``GeneratorSpec`` novo sem que ``GeneratorRunner``, o orquestrador ou o
avaliador mudem uma linha. Mesma ressalva que o ``DetectorLike`` do E8 faz
sobre si mesmo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.domain.generator_manifest import (
    GENERATOR_KEYS,
    GeneratorManifest,
)

DEFAULT_GENERATOR_KEY = "cached"


class GeneratorError(ValueError):
    """Erro acionável: o gerador não pôde ser resolvido ou construído."""


@runtime_checkable
class GeneratorLike(Protocol):
    """A superfície que o pipeline de fato consome de um gerador.

    Uma linha só, e é o ponto: tudo que separa um backend de outro —
    subprocesso, JVM, action config, topologia de rede — é assunto interno
    dele. O que o orquestrador precisa é de um caminho para um CSV.
    """

    def generate_dataset(self, attack_config: dict[str, Any], iteration: int) -> str: ...


@dataclass(frozen=True)
class GeneratorRequest:
    """O que um backend recebe para construir-se.

    ``attack_key``/``segment_name``/``random_seed`` são genéricos: todo gerador
    precisa saber qual ataque produzir e sob que semente. ``options`` carrega o
    que é específico do backend (caminhos de runtime, comando, timeouts do
    ERENO) — um backend de rede teria outras chaves, e é por isso que elas não
    moram nos campos nomeados. Mesma divisão que ``DetectorSpec.defaults`` faz
    com os hiperparâmetros: o genérico é tipado, o específico é um mapa
    validado contra o que o backend declara precisar.
    """

    attack_key: str
    segment_name: str | None = None
    random_seed: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorSpec:
    """Descrição estática de um gerador registrado.

    As três capacidades (``simulator``, ``requires_java``,
    ``produces_per_config_variation``) são as mesmas que o
    ``GeneratorManifest`` valida — declaradas uma vez aqui e copiadas para o
    manifest por ``manifest_for``, nunca digitadas duas vezes.
    """

    key: str
    simulator: str
    requires_java: bool
    produces_per_config_variation: bool
    description: str
    build: Callable[[GeneratorRequest], GeneratorLike]
    required_options: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #
# Opções que o ``GeneratorRunner`` precisa receber, qualquer que seja o modo.
# Listadas para que uma opção esquecida falhe na construção, com o nome dela, em
# vez de virar um TypeError no meio de uma campanha.
_RUNNER_OPTIONS: tuple[str, ...] = (
    "runtime_dir",
    "output_dataset_path",
    "run_command",
    "suggested_config_path",
    "action_config_relative_path",
    "benign_action_config_relative_path",
    "benign_seed_path",
    "timeout_seconds",
    "max_retries",
    "retry_backoff_seconds",
)


def _build_runner(request: GeneratorRequest, *, cached_dataset_path: Any) -> GeneratorLike:
    options = request.options
    return GeneratorRunner(
        runtime_dir=options["runtime_dir"],
        output_dataset_path=options["output_dataset_path"],
        run_command=options["run_command"],
        suggested_config_path=options["suggested_config_path"],
        action_config_relative_path=options["action_config_relative_path"],
        benign_action_config_relative_path=options["benign_action_config_relative_path"],
        benign_seed_path=options["benign_seed_path"],
        segment_name=request.segment_name,
        cached_dataset_path=cached_dataset_path,
        timeout_seconds=options["timeout_seconds"],
        max_retries=options["max_retries"],
        retry_backoff_seconds=options["retry_backoff_seconds"],
        random_seed=request.random_seed,
    )


def _build_cached(request: GeneratorRequest) -> GeneratorLike:
    return _build_runner(
        request, cached_dataset_path=request.options["cached_dataset_path"]
    )


def _build_ereno_jar(request: GeneratorRequest) -> GeneratorLike:
    # ``cached_dataset_path=None`` é o que liga o caminho do subprocesso no
    # ``GeneratorRunner`` — a escolha do backend, hoje, é literalmente esta.
    return _build_runner(request, cached_dataset_path=None)


_REGISTRY: dict[str, GeneratorSpec] = {
    "cached": GeneratorSpec(
        key="cached",
        simulator="none",
        requires_java=False,
        produces_per_config_variation=False,
        description=(
            "Replay de um trace versionado (data/baseline_dataset.csv). Roda o "
            "fluxo inteiro sem Java; o trace é o mesmo em toda execução."
        ),
        build=_build_cached,
        required_options=_RUNNER_OPTIONS + ("cached_dataset_path",),
    ),
    "jar": GeneratorSpec(
        key="jar",
        simulator="ereno",
        requires_java=True,
        produces_per_config_variation=True,
        description=(
            "Gerador ERENO (Java) executado sobre o attack config compilado. "
            "O único em que o efeito físico de uma intenção é mensurável."
        ),
        build=_build_ereno_jar,
        required_options=_RUNNER_OPTIONS,
    ),
}

# Mesmo contrato de sincronia do E8: uma chave registrada aqui e ausente do
# ``Literal`` produziria um ``GeneratorManifest`` impossível de validar, e uma
# chave só no ``Literal`` seria uma opção de CLI que não constrói nada.
if tuple(_REGISTRY) != GENERATOR_KEYS:
    raise GeneratorError(
        "Registro de geradores fora de sincronia com domain.generator_manifest."
        f"GeneratorKey: registro={tuple(_REGISTRY)!r}, contrato={GENERATOR_KEYS!r}."
    )


def generator_spec(key: str) -> GeneratorSpec:
    """``GeneratorSpec`` registrado sob ``key``, ou ``GeneratorError``."""

    try:
        return _REGISTRY[key]
    except KeyError:
        raise GeneratorError(
            f"Gerador desconhecido: {key!r}. Registrados: {list(GENERATOR_KEYS)!r}."
        ) from None


def list_generator_keys() -> tuple[str, ...]:
    """Chaves registradas, na ordem do registro (para a ajuda da CLI)."""

    return tuple(_REGISTRY)


def build_generator(key: str, request: GeneratorRequest) -> GeneratorLike:
    """Constrói o backend ``key`` a partir de ``request``."""

    spec = generator_spec(key)
    missing = [name for name in spec.required_options if name not in request.options]
    if missing:
        raise GeneratorError(
            f"Gerador {key!r} precisa das opções {sorted(missing)!r}, "
            "que não vieram no GeneratorRequest."
        )
    return spec.build(request)


def manifest_for(
    key: str,
    *,
    attack_key: str,
    segment_name: str | None = None,
    random_seed: int | None = None,
    duration_seconds: float = 0.0,
) -> GeneratorManifest:
    """Manifest da execução, com as capacidades vindas do registro.

    As capacidades não são parâmetro de propósito: elas são do backend, e
    deixá-las ao chamador abriria espaço para um manifest afirmar que o modo
    cacheado varia por config — a afirmação que faria um platô de campanha
    passar por resultado.
    """

    spec = generator_spec(key)
    return GeneratorManifest(
        generator=spec.key,
        simulator=spec.simulator,
        requires_java=spec.requires_java,
        produces_per_config_variation=spec.produces_per_config_variation,
        attack_key=attack_key,
        segment_name=segment_name,
        random_seed=random_seed,
        duration_seconds=duration_seconds,
    )
