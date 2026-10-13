"""E0 — validade do gerador campo a campo.

Pergunta: cada parâmetro que a arquitetura expõe ao atacante muda de fato o
tráfego gerado? Um campo inerte é uma alavanca que o atacante pode "mexer" sem
efeito nenhum, e uma busca que o escolhe gasta orçamento em nada. Os 4 bugs do
gerador achados em 13/10/2026 eram exatamente isso, e só apareceram por acaso.

Desenho pareado: a variante (só aquele campo alterado, pelo compilador de
sempre) e a config de referência são geradas **na mesma seed** — mesmo tráfego
normal, mesma RNG. Então:

- traces byte a byte iguais → campo **inerte**, sem ambiguidade estatística;
- senão, a deriva por feature das linhas de ataque (variante vs. referência na
  mesma seed) é comparada ao ruído da própria feature entre seeds da
  referência. O campo é **efetivo** quando alguma feature deriva acima do piso
  em todas as seeds medidas.

Só o tráfego é medido aqui. O que a mudança faz com um detector é outra
pergunta, respondida em outro experimento.
"""

from __future__ import annotations

import gzip
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from adversarial_ids.config.attack_capabilities import get_attack_capability
from adversarial_ids.config.attacks_registry import get_attack_spec
from adversarial_ids.core.intent_compiler import compile_attack_candidate
from adversarial_ids.core.preprocessor import always_drop_for
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
    IntentSpec,
)

MIN_DRIFT = 0.05
DRIFT_MARGIN = 2.0
_LABEL_CANDIDATES = ("class", "label")

# Os dois sentidos em que o compilador move um campo: rumo a um ataque menos
# intenso, ou mais intenso. Um campo já no limite pode não se mover num deles.
DIRECTIONS: dict[str, DesiredEffect] = {
    "intensity_down": DesiredEffect.LOWER_RECALL,
    "intensity_up": DesiredEffect.INCREASE_ATTACK_ACTIVITY,
}


@dataclass(frozen=True)
class FieldVariant:
    attack_key: str
    path: str
    direction: str
    config: dict[str, Any] | None
    diff: tuple[dict[str, Any], ...] = ()
    error: str | None = None


def field_variants(attack_key: str, *, intensity: str = "high") -> list[FieldVariant]:
    """Uma variante por (campo, sentido) que o catálogo admite."""

    capability = get_attack_capability(attack_key)
    variants: list[FieldVariant] = []
    for field in capability.fields:
        for direction, effect in DIRECTIONS.items():
            if effect not in field.effects:
                continue
            intent = IntentSpec(
                source_prompt=f"E0: {attack_key}.{field.path} ({direction})",
                objective=IntentObjective.ASSESS_IDS_ROBUSTNESS,
                base_attack=attack_key,
                desired_effect=effect,
                intensity=IntentIntensity(intensity),
                restrictions=IntentRestrictions(
                    allowed_fields=(field.path,), max_fields_changed=1
                ),
                seed=42,
            )
            try:
                candidate = compile_attack_candidate(intent)
            except Exception as error:  # noqa: BLE001 — vira linha do relatório
                variants.append(FieldVariant(attack_key, field.path, direction, None, error=str(error)))
                continue
            variants.append(
                FieldVariant(
                    attack_key,
                    field.path,
                    direction,
                    candidate.config,
                    tuple(change.model_dump(mode="json") for change in candidate.diff),
                )
            )
    return variants


def content_hash(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    digest = hashlib.sha256()
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_column(frame: pd.DataFrame) -> str:
    lowered = {c.lower(): c for c in frame.columns}
    for name in _LABEL_CANDIDATES:
        if name in lowered:
            return lowered[name]
    return frame.columns[-1]


def attack_features(path: Path) -> tuple[pd.DataFrame, float]:
    """Linhas de ataque (numéricas, sem identidade/relógio) e a prevalência."""

    frame = pd.read_csv(path, low_memory=False)
    label = _label_column(frame)
    is_attack = frame[label].astype(str).str.lower() != "normal"
    drop = [c for c in always_drop_for("deltas") if c in frame.columns] + [label]
    numeric = frame.drop(columns=drop).select_dtypes(include="number")
    return numeric[is_attack.to_numpy()], float(is_attack.mean())


def separation(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    """|Δ médias| / desvio combinado, coluna a coluna (mesma medida do piloto E0).

    Coluna constante nos dois lados em valores diferentes vale ``inf``
    (separação total); constante e igual vale 0.
    """

    out: dict[str, float] = {}
    for column in left.columns.intersection(right.columns):
        a, b = left[column].dropna(), right[column].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        delta = abs(float(a.mean()) - float(b.mean()))
        pooled = math.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2)
        if not math.isfinite(delta) or not math.isfinite(pooled):
            continue
        out[column] = (math.inf if delta > 0 else 0.0) if pooled == 0 else delta / pooled
    return out


def moved_features(drift: dict[str, float], noise: dict[str, float]) -> list[str]:
    return sorted(
        feature
        for feature, value in drift.items()
        if value > max(MIN_DRIFT, DRIFT_MARGIN * noise.get(feature, 0.0))
    )


def classify(identical: Sequence[bool], moved_per_seed: Sequence[Sequence[str]]) -> str:
    """``inert`` / ``effective`` / ``inconclusive``.

    ``inert``: traces idênticos em todas as seeds. ``effective``: alguma
    feature se moveu acima do piso em **todas** as seeds. Fora isso — bytes
    diferentes sem deriva consistente — fica ``inconclusive``: o campo muda
    algo (talvez só o consumo da RNG), mas nada mensurável acima do ruído.
    """

    if all(identical):
        return "inert"
    if moved_per_seed and all(moved_per_seed):
        return "effective"
    return "inconclusive"


def noise_floor(reference_traces: Sequence[Path]) -> dict[str, float]:
    """Ruído entre seeds da referência: o máximo, por feature, sobre os pares."""

    frames = [attack_features(path)[0] for path in reference_traces]
    floor: dict[str, float] = {}
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            for feature, value in separation(frames[i], frames[j]).items():
                if math.isfinite(value):
                    floor[feature] = max(floor.get(feature, 0.0), value)
    return floor


def reference_config(attack_key: str) -> dict[str, Any]:
    from adversarial_ids.shared.json_io import load_json

    return load_json(get_attack_spec(attack_key).baseline_path)
