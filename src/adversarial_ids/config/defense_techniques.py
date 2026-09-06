"""Catálogo feature→técnica e playbooks IEC-61850 (épico E5, janela D47-54).

Duas tabelas declarativas, com propósitos diferentes, que não devem ser
confundidas:

``FEATURE_TECHNIQUES`` / ``METRIC_TECHNIQUES``
    O que a *evidência daquela execução* recomenda. Ancoram uma
    ``DefenseAction`` naquilo que o detector realmente mediu — a ponte entre
    ``DetectionReport`` e ``DefenseAction.technique``.

``PLAYBOOKS``
    O que a *família de ataque* pede em termos de IEC-61850, independente do
    que o detector reparou nesta rodada. Um playbook de replay recomenda
    validação de sequência mesmo rodando sob um modo em que nenhuma feature de
    sequência chega ao relatório — o controle de protocolo não deixa de ser
    necessário só porque o IDS está cego para ele.

## O espaço de features depende do modo, e o catálogo cobre os dois

O CSV do ERENO tem 53 colunas e o preprocessador nunca entrega todas. O que
ele descarta depende de ``PROTOCOL_FEATURES_MODE`` (ver
``docs/preprocessing.md``):

- ``drop`` (default): sobram **20 features**, e 18 são grandezas elétricas. É
  o comportamento herdado, e é por isso que só a falta forjada é detectável —
  não há uma única feature de sequência ou temporização no espaço.
- ``deltas``: sobram **29**. As 9 a mais são ``stDiff``, ``sqDiff``,
  ``SqNum``, ``tDiff``, ``timestampDiff``, ``timeFromLastChange`` e os três
  deltas de tamanho — a semântica de que replay, flooding e grayhole dependem.

Identidade sai nos dois modos: ``ethSrc``, ``gocbRef``, ``goID``, ``datSet``,
os relógios absolutos e ``StNum`` bruto. Mantê-las ensinaria o detector a
reconhecer *o publisher*, não o ataque — e a medição concorda, todas dão
informação mútua ~0 contra a classe.

Este catálogo mapeia a **união dos dois modos**, 29 features. Um teste fixa
isso contra o header do dataset versionado: nenhuma feature utilizável sem
técnica, e nenhuma chave que não possa chegar a um ``DetectionReport`` sob
modo nenhum. Mapear ``StNum`` seria catálogo morto; mapear ``sqDiff`` não é.

O lado ancorado em métrica não é redundante com isso: ele é tudo que resta
quando o detector é ``svm_rbf``, que não expõe importância nenhuma e produz
``top_features`` vazio (épico E8, ver ``docs/detectors.md``). Um plano daquela
execução se sustenta só em métrica, e ``METRIC_TECHNIQUES`` existe para que
ele ainda tenha técnica fundamentada.

O vocabulário em si é ``domain.defense_plan.DefenseTechnique``; este módulo se
declara em sincronia com ele, nunca o contrário.
"""

from __future__ import annotations

from dataclasses import dataclass

from adversarial_ids.config.attacks_registry import list_attack_keys
from adversarial_ids.domain.defense_plan import (
    DEFENSE_TECHNIQUES,
    VALIDATION_METRICS,
)

# --------------------------------------------------------------------------- #
# Famílias de feature                                                         #
# --------------------------------------------------------------------------- #
# Nomeadas porque as 18 analógicas se dividem em três leituras diferentes da
# mesma grandeza, e a técnica é a mesma para as três — repetir a tupla 18 vezes
# esconderia esse fato.
_PHASES = ("A", "B", "C")

_INSTANTANEOUS = tuple(f"{q}sb{p}" for q in ("i", "v") for p in _PHASES)
_RMS = tuple(f"{q}sb{p}RmsValue" for q in ("i", "v") for p in _PHASES)
_TRAP_AREA = tuple(f"{q}sb{p}TrapAreaSum" for q in ("i", "v") for p in _PHASES)

# As três famílias de protocolo só existem sob PROTOCOL_FEATURES_MODE="deltas";
# em "drop" nenhuma delas chega ao relatório. O catálogo as mapeia mesmo assim,
# porque o modo é do experimento e o catálogo não deve depender de qual foi
# escolhido.
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "analog_instantaneous": _INSTANTANEOUS,
    "analog_rms": _RMS,
    "analog_trap_area": _TRAP_AREA,
    "breaker_state": ("cbStatus", "cbStatusDiff"),
    "goose_sequence": ("stDiff", "sqDiff", "SqNum"),
    "goose_timing": ("tDiff", "timestampDiff", "timeFromLastChange"),
    "goose_payload_size": ("gooseLengthDiff", "apduSizeDiff", "frameLengthDiff"),
}

# Quais famílias dependem do modo "deltas" para aparecer.
PROTOCOL_FAMILIES: tuple[str, ...] = (
    "goose_sequence",
    "goose_timing",
    "goose_payload_size",
)

# --------------------------------------------------------------------------- #
# Feature → técnica                                                           #
# --------------------------------------------------------------------------- #
# Ordem = ordem de recomendação: a técnica mais específica primeiro, as de
# apoio depois. Quem monta o plano lê a primeira; quem quer alternativa lê o
# resto.
_TECHNIQUES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    # Corrente/tensão instantânea por fase. Se o detector se apoia nelas, o
    # adversário está fabricando a forma de onda — a resposta é confrontar a
    # leitura com a física (correlação entre fases, coerência com o disjuntor),
    # não ajustar limiar.
    "analog_instantaneous": (
        "physical_consistency_check",
        "feature_engineering",
        "operator_alerting",
    ),
    # RMS é a instantânea integrada: mesma leitura, janela maior. Um ataque que
    # engana a RMS sustentou a mentira por vários ciclos.
    "analog_rms": (
        "physical_consistency_check",
        "continuous_monitoring",
        "operator_alerting",
    ),
    # Soma de área trapezoidal: o discriminador dominante do masquerade_fault no
    # dataset de referência. É energia acumulada, então a inconsistência física
    # é ainda mais barata de checar do que na instantânea.
    "analog_trap_area": (
        "physical_consistency_check",
        "detector_retraining",
        "operator_alerting",
    ),
    # Estado do disjuntor. Uma mudança de estado legítima vem acompanhada de
    # incremento de stNum e de uma assinatura elétrica compatível. Sob o modo
    # "drop" é o único ponto do espaço de features onde a semântica de sequência
    # do GOOSE ainda é observável, já que os deltas saem todos.
    "breaker_state": (
        "goose_sequence_validation",
        "physical_consistency_check",
        "device_quarantine",
    ),
    # Deltas de stNum/sqNum. É a assinatura direta de replay e de injeção: um
    # quadro reproduzido repete ou retrocede a sequência, e um injetado a
    # atropela. A resposta primária é validar a sequência no subscriber;
    # autenticar (62351-6) é o que impede o quadro forjado de chegar.
    "goose_sequence": (
        "goose_sequence_validation",
        "goose_authentication",
        "publisher_binding",
        "device_quarantine",
    ),
    # Deltas temporais. Cadência fora do esperado é replay atrasado, inundação
    # ou supressão — os três se distinguem pelo sinal do desvio, e os três
    # aparecem aqui antes de aparecer em qualquer outra família.
    "goose_timing": (
        "goose_timing_analysis",
        "traffic_rate_limiting",
        "continuous_monitoring",
        "operator_alerting",
    ),
    # Deltas de tamanho de quadro/APDU. Um quadro cujo tamanho não bate com o
    # dataset configurado não deveria ser aceito: em IEC-61850 o subscriber
    # valida datSet/confRev/numDatSetEntries contra a própria configuração, que
    # é o vínculo entre publisher e o que ele tem direito de publicar.
    "goose_payload_size": (
        "publisher_binding",
        "goose_authentication",
        "device_quarantine",
    ),
}

FEATURE_TECHNIQUES: dict[str, tuple[str, ...]] = {
    feature: _TECHNIQUES_BY_FAMILY[family]
    for family, features in FEATURE_FAMILIES.items()
    for feature in features
}

# --------------------------------------------------------------------------- #
# Métrica → técnica                                                           #
# --------------------------------------------------------------------------- #
# Cobre todas as chaves de ``ValidationMetric``. É o lado que sustenta um plano
# quando ``top_features`` vem vazio (svm_rbf) ou quando o problema não é *qual*
# feature enganou o detector, e sim *quanto* ele erra.
METRIC_TECHNIQUES: dict[str, tuple[str, ...]] = {
    # Falso negativo é o risco defensivo central deste projeto: o ataque passou.
    "recall": (
        "detector_threshold_tuning",
        "class_rebalancing",
        "detector_retraining",
        "operator_alerting",
    ),
    "confusion_matrix.fn": (
        "detector_threshold_tuning",
        "class_rebalancing",
        "dataset_enrichment",
        "operator_alerting",
    ),
    # Falso positivo cansa o operador até ele parar de olhar — é um problema
    # defensivo, não só uma métrica feia.
    "precision": (
        "detector_threshold_tuning",
        "feature_engineering",
        "continuous_monitoring",
    ),
    "confusion_matrix.fp": (
        "detector_threshold_tuning",
        "feature_engineering",
        "continuous_monitoring",
    ),
    "f1": (
        "detector_retraining",
        "feature_engineering",
        "dataset_enrichment",
    ),
    "accuracy": (
        "detector_retraining",
        "class_rebalancing",
        "dataset_enrichment",
    ),
    # Acerto: confirmar que o que está sendo detectado continua sendo detectado.
    "confusion_matrix.tp": (
        "continuous_monitoring",
        "detector_retraining",
    ),
    "confusion_matrix.tn": (
        "continuous_monitoring",
        "feature_engineering",
    ),
    # Latência de inferência alta contra a cadência de retransmissão do GOOSE
    # (ordem de milissegundos) significa que a detecção chega tarde. A resposta
    # é controle compensatório enquanto ela não acompanha, não "otimizar".
    "latency_ms": (
        "network_segmentation",
        "traffic_rate_limiting",
        "operator_alerting",
    ),
}


# --------------------------------------------------------------------------- #
# Playbooks IEC-61850                                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DefensePlaybook:
    """Resposta padrão a uma família de ataque, em termos de IEC-61850.

    ``techniques`` está em ordem de aplicação, não de importância: o que se faz
    primeiro vem primeiro. ``signature_features`` lista as features *do espaço
    utilizável* que costumam denunciar o cenário — pode ser vazia quando o
    ataque não deixa rastro nas 20 colunas que sobrevivem ao preprocessador, e
    essa ausência é informação, não lacuna.
    """

    key: str
    title: str
    scenario: str
    attack_keys: tuple[str, ...]
    techniques: tuple[str, ...]
    signature_features: tuple[str, ...]
    reference: str


_PLAYBOOK_LIST: tuple[DefensePlaybook, ...] = (
    DefensePlaybook(
        key="spoofed_fault_indication",
        title="Indicação de falta forjada",
        scenario=(
            "O publisher anuncia uma falta que não ocorreu: as grandezas "
            "elétricas são fabricadas para parecer um curto e o cbStatus "
            "acompanha, induzindo a proteção a abrir o disjuntor."
        ),
        attack_keys=("masquerade_fault",),
        techniques=(
            "physical_consistency_check",
            "goose_sequence_validation",
            "operator_alerting",
            "publisher_binding",
        ),
        signature_features=(*_TRAP_AREA, *_RMS, "cbStatus"),
        reference="IEC 61850-5 (esquemas de proteção) e IEC 62351-6 §7",
    ),
    DefensePlaybook(
        key="replayed_goose_frames",
        title="Reprodução de quadros GOOSE",
        scenario=(
            "Quadros legítimos capturados antes são reinjetados — na ordem "
            "original, invertida ou atrasados. O conteúdo é autêntico; o que "
            "está errado é *quando* ele chega e como stNum/sqNum evoluem."
        ),
        attack_keys=(
            "random_replay",
            "inverse_replay",
            "delayed_replay",
            "delayed_replay_backoff",
            "delayed_replay_batch_dump",
            "delayed_replay_double_drop",
        ),
        techniques=(
            "goose_sequence_validation",
            "goose_timing_analysis",
            "goose_authentication",
            "publisher_binding",
        ),
        # Só observáveis sob PROTOCOL_FEATURES_MODE="deltas". No modo default
        # o replay não deixa rastro nenhum no espaço de features, e aí o
        # cenário depende inteiramente de controle de protocolo.
        signature_features=(
            "sqDiff",
            "stDiff",
            "SqNum",
            "tDiff",
            "timestampDiff",
            "timeFromLastChange",
        ),
        reference="IEC 62351-6 §7 (autenticação) e IEC 61850-8-1 §18.1 (stNum/sqNum)",
    ),
    DefensePlaybook(
        key="forged_state_injection",
        title="Injeção de estado forjado",
        scenario=(
            "Um publisher não autorizado emite quadros novos, eventualmente "
            "com stNum muito alto para vencer o quadro legítimo na disputa "
            "pelo subscriber."
        ),
        attack_keys=("injection", "high_stnum"),
        techniques=(
            "goose_sequence_validation",
            "publisher_binding",
            "goose_authentication",
            "device_quarantine",
        ),
        signature_features=(
            "cbStatus",
            "cbStatusDiff",
            "stDiff",
            "SqNum",
            "gooseLengthDiff",
            "apduSizeDiff",
        ),
        reference="IEC 61850-8-1 §18.1 e IEC 62351-6 §7",
    ),
    DefensePlaybook(
        key="goose_flooding",
        title="Inundação do barramento de estação",
        scenario=(
            "Volume de quadros muito acima da cadência de retransmissão "
            "esperada, saturando o subscriber e a rede."
        ),
        attack_keys=("flooding",),
        techniques=(
            "goose_timing_analysis",
            "traffic_rate_limiting",
            "network_segmentation",
            "operator_alerting",
        ),
        # Inundação comprime a cadência: tDiff/timestampDiff despencam.
        signature_features=("tDiff", "timestampDiff", "frameLengthDiff"),
        reference="IEC 61850-90-4 (engenharia de rede de subestação)",
    ),
    DefensePlaybook(
        key="frame_suppression",
        title="Supressão seletiva de quadros",
        scenario=(
            "Quadros são descartados no caminho: o subscriber deixa de receber "
            "a atualização e continua operando sobre um estado velho até o "
            "timeAllowedToLive expirar — se é que alguém observa a expiração."
        ),
        attack_keys=("grayhole",),
        techniques=(
            "goose_timing_analysis",
            "continuous_monitoring",
            "operator_alerting",
            "network_segmentation",
        ),
        # Supressão faz o oposto da inundação: o intervalo estica e a sequência
        # salta os quadros que sumiram.
        signature_features=("timeFromLastChange", "tDiff", "sqDiff"),
        reference="IEC 61850-8-1 §18.1.2 (timeAllowedToLive)",
    ),
)

PLAYBOOKS: dict[str, DefensePlaybook] = {
    playbook.key: playbook for playbook in _PLAYBOOK_LIST
}

_PLAYBOOK_BY_ATTACK: dict[str, DefensePlaybook] = {
    attack_key: playbook
    for playbook in _PLAYBOOK_LIST
    for attack_key in playbook.attack_keys
}


# --------------------------------------------------------------------------- #
# Consultas                                                                   #
# --------------------------------------------------------------------------- #
def techniques_for_feature(feature: str) -> tuple[str, ...]:
    """Técnicas recomendadas quando ``feature`` pesa no relatório.

    Devolve tupla vazia para feature desconhecida em vez de levantar: um
    ``DetectionReport`` pode citar uma coluna que este catálogo ainda não
    conhece (um dataset novo, uma feature derivada), e isso é motivo para o
    plano não ter técnica ancorada nela — não para o pipeline quebrar.
    """

    return FEATURE_TECHNIQUES.get(feature, ())


def techniques_for_metric(metric: str) -> tuple[str, ...]:
    """Técnicas recomendadas a partir de uma métrica do relatório."""

    if metric not in METRIC_TECHNIQUES:
        raise KeyError(
            f"Métrica sem técnica catalogada: {metric!r}. "
            f"Esperado um de {tuple(METRIC_TECHNIQUES)}."
        )

    return METRIC_TECHNIQUES[metric]


def playbook_for_attack(attack_key: str) -> DefensePlaybook:
    """Playbook IEC-61850 da família a que o ataque pertence."""

    if attack_key not in _PLAYBOOK_BY_ATTACK:
        raise KeyError(
            f"Ataque sem playbook: {attack_key!r}. "
            f"Catalogados: {tuple(sorted(_PLAYBOOK_BY_ATTACK))}."
        )

    return _PLAYBOOK_BY_ATTACK[attack_key]


def playbooks_for_technique(technique: str) -> tuple[DefensePlaybook, ...]:
    """Cenários em que a técnica aparece, na ordem de declaração."""

    return tuple(
        playbook for playbook in _PLAYBOOK_LIST if technique in playbook.techniques
    )


def catalogued_techniques() -> frozenset[str]:
    """Técnicas alcançáveis por alguma entrada deste catálogo."""

    return frozenset(
        technique
        for source in (
            *FEATURE_TECHNIQUES.values(),
            *METRIC_TECHNIQUES.values(),
            *(playbook.techniques for playbook in _PLAYBOOK_LIST),
        )
        for technique in source
    )


# --------------------------------------------------------------------------- #
# Sincronia com o contrato e com o registro de ataques                        #
# --------------------------------------------------------------------------- #
def _check_catalog() -> None:
    """Falha no import se o catálogo divergir do vocabulário ou do registro.

    O domínio não importa ``config``, então nada obriga os dois a andarem
    juntos — exceto isto. Mesma checagem que ``core/detectors.py`` faz contra
    ``DETECTOR_KEYS``.
    """

    unknown = catalogued_techniques() - set(DEFENSE_TECHNIQUES)
    if unknown:
        raise RuntimeError(
            f"Técnica fora de DefenseTechnique: {sorted(unknown)}."
        )

    uncovered = set(VALIDATION_METRICS) - set(METRIC_TECHNIQUES)
    if uncovered:
        raise RuntimeError(
            "Métrica de validação sem técnica catalogada: "
            f"{sorted(uncovered)}. Um plano ancorado nela ficaria sem técnica."
        )

    extra_metrics = set(METRIC_TECHNIQUES) - set(VALIDATION_METRICS)
    if extra_metrics:
        raise RuntimeError(
            f"Métrica catalogada que não existe no relatório: {sorted(extra_metrics)}."
        )

    registered = set(list_attack_keys())
    unknown_attacks = set(_PLAYBOOK_BY_ATTACK) - registered
    if unknown_attacks:
        raise RuntimeError(
            f"Playbook aponta para ataque não registrado: {sorted(unknown_attacks)}."
        )

    orphan_attacks = registered - set(_PLAYBOOK_BY_ATTACK)
    if orphan_attacks:
        raise RuntimeError(
            "Ataque registrado sem playbook: "
            f"{sorted(orphan_attacks)}. Todo ataque precisa de uma resposta."
        )

    for playbook in _PLAYBOOK_LIST:
        if len(set(playbook.techniques)) != len(playbook.techniques):
            raise RuntimeError(f"Técnica repetida no playbook {playbook.key!r}.")

        unknown_signature = set(playbook.signature_features) - set(FEATURE_TECHNIQUES)
        if unknown_signature:
            raise RuntimeError(
                f"Playbook {playbook.key!r} cita feature fora do catálogo: "
                f"{sorted(unknown_signature)}."
            )


_check_catalog()
