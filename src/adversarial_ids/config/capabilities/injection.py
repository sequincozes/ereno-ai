"""Capacidade intent-driven de ``injection`` (uc05).

Só 2 dos 12 campos editáveis do baseline entram no catálogo — o resto é
alavanca morta ou controle de reprodutibilidade, não intensidade:

- ``randomSeed`` — semente de reprodutibilidade, não uma alavanca de ataque.
- ``stNum``/``sqNum``/``cbStatus.values``/``ttlMs``/``confRev`` (9 caminhos) —
  o próprio ``comment`` do baseline documenta que essas faixas só são lidas
  quando ``injectionPattern == "synthetic"``; o baseline usa ``"random"``, que
  clona mensagens legítimas e as ignora. Compilar uma mudança nelas produziria
  um diff sem efeito físico algum, corrompendo a política E10 (que mediria
  "sem melhora" e escalaria uma alavanca morta). Mudar o baseline para
  ``"synthetic"`` destravaria essas 9 faixas, mas também mudaria o que
  ``--engine live --attack injection`` (loop legado) otimiza — decisão de
  dados fora do escopo deste catálogo.

``injectionPattern`` mantém ``choices=("random", "uniform")`` — exclui
deliberadamente ``"burst"`` (redundante com o objeto ``burst`` de outros
ataques, sem parâmetro próprio aqui) e ``"synthetic"`` (ver acima: alternar
para ele tornaria a vivacidade das 9 faixas excluídas dependente do valor de
*outro* campo no mesmo diff — um acoplamento que ``FieldCapability`` não
consegue expressar nem ``resolve_candidate_paths`` consegue checar).
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import (
    DETECTION_AND_ACTIVITY_EFFECTS,
    DETECTION_EFFECTS,
)
from adversarial_ids.domain.intent_spec import IntentObjective

INJECTION_CAPABILITY = AttackCapability(
    capability_id="injection.v1",
    attack_key="injection",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
        FieldCapability(
            path="numInjectedMessages",
            value_type="integer",
            description="Quantidade de mensagens injetadas.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            # Piso 5: o portão E4 (DatasetBundle) exige min_attack_rows=5 por
            # padrão — um valor menor produziria um candidato que o próprio
            # pipeline rejeitaria no estágio PREPROCESS.
            minimum=5,
            maximum=5000,
        ),
        FieldCapability(
            path="injectionPattern",
            value_type="string",
            description="Padrão de injeção das mensagens clonadas.",
            effects=DETECTION_EFFECTS,
            choices=("random", "uniform"),
        ),
    ),
)
