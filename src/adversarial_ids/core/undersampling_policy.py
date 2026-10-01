"""Quem decide qual undersampler roda — configuração ou política declarada.

A caixa do Pre-Processor no diagrama do journal prevê undersamplers
"intent-driven" e "policy-driven". Até aqui nada *decidia* nada: a estratégia
era uma env var lida uma vez. Este módulo é o decisor, e tem exatamente dois
modos — ``config`` (a estratégia veio fixada) e ``policy`` (a regra declarada
resolve a partir do que ela observa no treino).

O guardrail está na assinatura
------------------------------
``decide_undersampling`` recebe **contagem de classes da partição de treino** e
**a política**. Não há parâmetro por onde um ``DetectionReport``, uma
``Metrics`` ou um ``IntentSpec`` entrem — e isso não é economia de argumentos,
é a regra. Um laço que ajusta o próprio pré-processamento até a métrica
melhorar deixa de medir o que o detector enfrenta; e deixar a intenção do
atacante escolher o balanceamento do treino do defensor é enfraquecer o
detector por um caminho que não é o ataque. As duas coisas se fecham sozinhas
quando a função não tem como ver nem a métrica nem a intenção.

É a mesma figura do E10, com o sinal trocado: lá a política determinística
substitui uma segunda chamada de LLM; aqui ela substitui uma otimização que
ninguém pediu. Em ambos os casos o que a torna aceitável é a decisão ficar
escrita, com a razão, num artefato que se lê depois sem reexecutar nada
(``UndersamplingDecision``, persistida no ``SelectionManifest``).
"""

from __future__ import annotations

from typing import Mapping

from adversarial_ids.core.undersamplers import UndersamplerError
from adversarial_ids.domain.undersampling import (
    UNDERSAMPLER_KEYS,
    UndersamplingDecision,
    UndersamplingPolicy,
)

# O modo que delega a escolha à política. Não é uma estratégia (não existe um
# undersampler chamado "policy"): é o decisor, e por isso fica fora de
# ``UndersamplerKey``. O ``SelectionManifest`` registra a estratégia resolvida
# em ``undersampling`` e este fato em ``undersampling_decided_by``.
POLICY_MODE = "policy"

UNDERSAMPLING_MODES: tuple[str, ...] = UNDERSAMPLER_KEYS + (POLICY_MODE,)


def observed_imbalance(class_counts: Mapping[str, int]) -> float:
    """Razão entre a classe mais frequente e a menos frequente do treino.

    ``1.0`` quando há uma classe só — não há desequilíbrio a medir, e é o
    valor que faz qualquer teto declarado aceitar o conjunto como está.
    """

    positive = [count for count in class_counts.values() if count > 0]
    if not positive:
        raise UndersamplerError(
            "Não há contagem de classe positiva na partição de treino — não há "
            "desequilíbrio a medir nem o que reamostrar."
        )
    return max(positive) / min(positive)


def decide_undersampling(
    *,
    class_counts: Mapping[str, int],
    mode: str,
    policy: UndersamplingPolicy,
) -> UndersamplingDecision:
    """Resolve ``mode`` numa estratégia concreta, com a razão por escrito.

    ``mode`` é uma chave de ``UndersamplerKey`` (decidido por configuração) ou
    ``"policy"`` (decidido pela regra declarada). Qualquer outra coisa é erro
    com a lista de modos válidos — e não um default silencioso, que é como um
    experimento passa a rodar sem o balanceamento que alguém pensou ter ligado.
    """

    imbalance = observed_imbalance(class_counts)

    if mode in UNDERSAMPLER_KEYS:
        return _decided_by_config(mode, imbalance=imbalance, policy=policy)
    if mode == POLICY_MODE:
        return _decided_by_policy(imbalance=imbalance, policy=policy)

    raise UndersamplerError(
        f"Modo de undersampling desconhecido: {mode!r}. "
        f"Use um de {list(UNDERSAMPLING_MODES)!r}."
    )


def _decided_by_config(
    strategy: str, *, imbalance: float, policy: UndersamplingPolicy
) -> UndersamplingDecision:
    target_ratio = policy.target_ratio if strategy == "ratio" else None
    reason = (
        f"Estratégia {strategy!r} fixada na configuração; o desequilíbrio "
        f"observado no treino ({imbalance:.2f}:1) não participou da escolha."
    )
    return UndersamplingDecision(
        strategy=strategy,
        decided_by="config",
        target_ratio=target_ratio,
        observed_imbalance=imbalance,
        reason=reason,
    )


def _decided_by_policy(
    *, imbalance: float, policy: UndersamplingPolicy
) -> UndersamplingDecision:
    if imbalance <= policy.max_imbalance_ratio:
        return UndersamplingDecision(
            strategy="none",
            decided_by="policy",
            target_ratio=None,
            observed_imbalance=imbalance,
            reason=(
                f"Desequilíbrio do treino ({imbalance:.2f}:1) dentro do teto "
                f"declarado ({policy.max_imbalance_ratio:.2f}:1) — nenhuma "
                "linha descartada."
            ),
        )

    # ``ratio`` e não ``random``: equilibrar até a minoritária num trace de
    # prevalência baixa descarta quase todo o tráfego benigno, e o detector
    # passa a treinar numa distribuição que nenhuma subestação produz. A
    # política corrige até o alvo declarado e para aí.
    return UndersamplingDecision(
        strategy="ratio",
        decided_by="policy",
        target_ratio=policy.target_ratio,
        observed_imbalance=imbalance,
        reason=(
            f"Desequilíbrio do treino ({imbalance:.2f}:1) acima do teto "
            f"declarado ({policy.max_imbalance_ratio:.2f}:1) — maioria reduzida "
            f"a {policy.target_ratio:.2f}x a classe minoritária."
        ),
    )
