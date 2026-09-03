"""Registro dos tipos de ataque do gerador ERENO suportados pelo loop.

Cada ``AttackSpec`` amarra, num único lugar, tudo que o pipeline precisa saber
para rodar os agentes sobre um ataque específico:

- ``key``          — nome exposto na CLI (``--attack``).
- ``segment_name`` — nome do segmento no ``action_create_attack_dataset.json``.
                     O JAR decide o **rótulo** da classe pelo prefixo ``ucXX`` desse
                     nome (ver ``getLabelForSegmentName`` no ERENO), então ele
                     precisa casar exatamente com o action config.
- ``attack_type``  — valor do campo ``attackType`` no JSON (dispatch do JAR).
- ``label``        — rótulo que o JAR grava na coluna ``class`` para esse ataque.
                     É o que o ``IdsEvaluator`` usa para saber qual classe é o
                     ataque a ser detectado.
- ``baseline``     — arquivo de configuração inicial (em ``inputs/attacks/``).
- ``description``  — resumo curto injetado no prompt do Estrategista.

O uc04 (``masquerade_normal``) é intencionalmente omitido: o gerador não lê os
parâmetros dele (construtor sem ``AttackConfig``), então não haveria o que os
agentes otimizarem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adversarial_ids.config.settings import INPUTS_DIR

_ATTACKS_DIR = INPUTS_DIR / "attacks"


@dataclass(frozen=True)
class AttackSpec:
    key: str
    segment_name: str
    attack_type: str
    label: str
    baseline_filename: str
    description: str
    intent_capability_id: str | None = None

    @property
    def baseline_path(self) -> Path:
        return _ATTACKS_DIR / self.baseline_filename


# Ordem preservada para exibição estável na ajuda da CLI.
_SPECS: tuple[AttackSpec, ...] = (
    AttackSpec(
        key="masquerade_fault",
        segment_name="uc03_masquerade_fault",
        attack_type="masquerade_fault",
        label="masquerade_fake_fault",
        baseline_filename="uc03_masquerade_fault.json",
        description=(
            "Masquerade de falha falsa (uc03): injeta uma falha forjada em GOOSE "
            "manipulando estado do disjuntor, stNum, valores analógicos e a trap area."
        ),
        intent_capability_id="masquerade_fault.v1",
    ),
    AttackSpec(
        key="random_replay",
        segment_name="uc01_random_replay",
        attack_type="random_replay",
        label="random_replay",
        baseline_filename="uc01_random_replay.json",
        description=(
            "Replay aleatório (uc01): recaptura e reenvia mensagens legítimas com "
            "janelas, atrasos e rajadas variáveis."
        ),
    ),
    AttackSpec(
        key="inverse_replay",
        segment_name="uc02_inverse_replay",
        attack_type="inverse_replay",
        label="inverse_replay",
        baseline_filename="uc02_inverse_replay.json",
        description=(
            "Replay invertido (uc02): reenvia blocos de mensagens em ordem invertida, "
            "com atrasos e rajadas."
        ),
    ),
    AttackSpec(
        key="injection",
        segment_name="uc05_injection",
        attack_type="injection",
        label="injection",
        baseline_filename="uc05_injection.json",
        description=(
            "Injeção (uc05): insere mensagens (clonadas ou sintéticas) segundo um "
            "padrão (random/uniform/burst/synthetic)."
        ),
    ),
    AttackSpec(
        key="high_stnum",
        segment_name="uc06_high_stnum",
        attack_type="high_stnum_injection",
        label="high_StNum",
        baseline_filename="uc06_high_stnum_injection.json",
        description=(
            "Injeção com stNum elevado (uc06): força saltos anômalos de stNum "
            "simulando transições de estado inexistentes."
        ),
    ),
    AttackSpec(
        key="flooding",
        segment_name="uc07_flooding",
        attack_type="flooding",
        label="poisoned_high_rate",
        baseline_filename="uc07_flooding.json",
        description=(
            "Flooding / alta taxa (uc07): rajadas de mensagens com stNum a cada pacote "
            "e gaps mínimos, saturando o barramento."
        ),
    ),
    AttackSpec(
        key="grayhole",
        segment_name="uc08_grayhole",
        attack_type="grayhole",
        label="grayhole",
        baseline_filename="uc08_grayhole.json",
        description=(
            "Grayhole (uc08): descarte seletivo de mensagens em rajadas, com atrasos "
            "para mascarar o ataque."
        ),
    ),
    AttackSpec(
        key="delayed_replay",
        segment_name="uc10_delayed_replay",
        attack_type="delayed_replay",
        label="delayed_replay",
        baseline_filename="uc10_delayed_replay.json",
        description=(
            "Replay atrasado (uc10): retém mensagens e as reenvia mais tarde, "
            "controlando intervalo de rajada e atraso de rede."
        ),
    ),
    AttackSpec(
        key="delayed_replay_backoff",
        segment_name="uc10_delayed_replay_backoff",
        attack_type="delayed_replay_backoff",
        label="delayed_replay",
        baseline_filename="uc10_delayed_replay_backoff.json",
        description=(
            "Replay atrasado com backoff (uc10): variante que reenvia com "
            "multiplicador de taxa (rateMultiplier)."
        ),
    ),
    AttackSpec(
        key="delayed_replay_batch_dump",
        segment_name="uc10_delayed_replay_batch_dump",
        attack_type="delayed_replay_batch_dump",
        label="delayed_replay",
        baseline_filename="uc10_delayed_replay_batch_dump.json",
        description=(
            "Replay atrasado em lote (uc10): despeja as mensagens retidas em bloco, "
            "com micro-gaps entre elas."
        ),
    ),
    AttackSpec(
        key="delayed_replay_double_drop",
        segment_name="uc10_delayed_replay_double_drop",
        attack_type="delayed_replay_double_drop",
        label="delayed_replay",
        baseline_filename="uc10_delayed_replay_double_drop.json",
        description=(
            "Replay atrasado com duplo descarte (uc10): variante que descarta e "
            "reenvia mensagens forjadas."
        ),
    ),
)

ATTACK_REGISTRY: dict[str, AttackSpec] = {spec.key: spec for spec in _SPECS}

# Ataque usado por padrão — mantém o comportamento histórico (uc03).
DEFAULT_ATTACK_KEY = "masquerade_fault"


def list_attack_keys() -> list[str]:
    """Chaves de ataque na ordem de registro (para a ajuda da CLI)."""
    return [spec.key for spec in _SPECS]


def get_attack_spec(key: str) -> AttackSpec:
    try:
        return ATTACK_REGISTRY[key]
    except KeyError:
        valid = ", ".join(list_attack_keys())
        raise ValueError(
            f"Ataque desconhecido: {key!r}. Opções válidas: {valid}."
        ) from None
