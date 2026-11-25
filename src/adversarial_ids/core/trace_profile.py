"""Perfil determinístico de um trace: o contexto do injetor data-driven.

``profile_trace`` lê um CSV do ERENO e devolve um ``TraceProfile``: quais
colunas mais separam a classe de ataque do tráfego normal, e como estão as
colunas que a DSL do ataque programável consegue mexer. ``render_trace_profile``
transforma isso no bloco de texto que vai às instruções do LLM.

Nenhum LLM aqui. O modelo só lê o resultado — o mesmo princípio do resto do
pipeline: o que é medida é Python, o que é proposta é LLM.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd

from adversarial_ids.domain.trace_profile import FeatureContrast, TraceProfile

# Mesmos candidatos e mesma ordem que ``dataset_bundle_builder`` e
# ``IdsEvaluator`` usam para achar o rótulo; fallback na última coluna.
_LABEL_COLUMN_CANDIDATES = (
    "class",
    "label",
    "classe",
    "target",
    "attack",
    "is_attack",
    "category",
    "type",
)

NORMAL_LABEL = "normal"

# Colunas que a DSL do programmable escreve (cbStatus, StNum, SqNum, TTL,
# confRev, t) ou que derivam diretamente delas no trace (os deltas). É o
# vocabulário do ataque que o LLM vai propor.
LEVER_FEATURES: tuple[str, ...] = (
    "cbStatus",
    "StNum",
    "SqNum",
    "gooseTimeAllowedtoLive",
    "confRev",
    "stDiff",
    "sqDiff",
    "tDiff",
    "timestampDiff",
    "timeFromLastChange",
)

DEFAULT_TOP_K = 10


class TraceProfileError(ValueError):
    """Erro acionável: o trace não pôde ser perfilado."""


def _find_label_column(columns: list[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    for name in _LABEL_COLUMN_CANDIDATES:
        if name in lowered:
            return lowered[name]
    return columns[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contrast(column: str, attack: pd.Series, normal: pd.Series) -> FeatureContrast | None:
    a = attack.dropna()
    b = normal.dropna()
    if len(a) < 2 or len(b) < 2:
        return None

    a_mean, b_mean = float(a.mean()), float(b.mean())
    a_std, b_std = float(a.std(ddof=1)), float(b.std(ddof=1))
    if not all(math.isfinite(value) for value in (a_mean, b_mean, a_std, b_std)):
        return None

    pooled = math.sqrt((a_std**2 + b_std**2) / 2)
    delta = abs(a_mean - b_mean)
    separation = delta / pooled if pooled > 0.0 else None

    return FeatureContrast(
        feature=column,
        attack_mean=round(a_mean, 6),
        normal_mean=round(b_mean, 6),
        attack_std=round(a_std, 6),
        normal_std=round(b_std, 6),
        separation=round(separation, 6) if separation is not None else None,
    )


def profile_trace(
    trace_path: str | Path,
    *,
    attack_label: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> TraceProfile:
    """Perfila ``trace_path``: classe de ataque contra a classe ``normal``.

    ``attack_label`` escolhe a classe quando o trace tem mais de uma além de
    ``normal``; com uma só, ela é inferida. Levanta ``TraceProfileError`` quando
    não há classe normal, não há classe de ataque, ou a escolha é ambígua.
    """

    if top_k < 1:
        raise TraceProfileError(f"top_k precisa ser >= 1 (veio {top_k}).")

    path = Path(trace_path)
    if not path.exists():
        raise TraceProfileError(f"Trace não encontrado: {path}")

    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise TraceProfileError(f"Trace vazio: {path}")

    label_column = _find_label_column(list(frame.columns))
    labels = frame[label_column].astype(str)
    counts = {str(name): int(count) for name, count in labels.value_counts().items()}

    if NORMAL_LABEL not in counts:
        raise TraceProfileError(
            f"O trace não tem a classe {NORMAL_LABEL!r} em {label_column!r}; "
            f"classes: {sorted(counts)}."
        )
    attack_classes = sorted(name for name in counts if name != NORMAL_LABEL)
    if attack_label is None:
        if len(attack_classes) != 1:
            raise TraceProfileError(
                "Informe a classe de ataque a perfilar: o trace tem "
                f"{len(attack_classes)} classes além de normal ({attack_classes})."
            )
        attack_label = attack_classes[0]
    elif attack_label not in counts or attack_label == NORMAL_LABEL:
        raise TraceProfileError(
            f"Classe de ataque {attack_label!r} não está no trace; "
            f"disponíveis: {attack_classes}."
        )

    numeric = frame.drop(columns=[label_column]).select_dtypes(include="number")
    attack = numeric[labels == attack_label]
    normal = numeric[labels == NORMAL_LABEL]

    contrasts = [
        contrast
        for column in numeric.columns
        if (contrast := _contrast(column, attack[column], normal[column])) is not None
    ]
    measured = sorted(
        (c for c in contrasts if c.separation is not None),
        key=lambda c: (-c.separation, c.feature),
    )
    constant_shift = tuple(
        c for c in contrasts if c.separation is None and c.attack_mean != c.normal_mean
    )
    by_name = {c.feature: c for c in contrasts}
    levers = tuple(by_name[name] for name in LEVER_FEATURES if name in by_name)

    return TraceProfile(
        trace_ref=str(path),
        trace_sha256=_sha256(path),
        label_column=label_column,
        attack_label=attack_label,
        normal_label=NORMAL_LABEL,
        class_counts=counts,
        top_features=tuple(measured[:top_k]),
        constant_shift=constant_shift,
        lever_features=levers,
    )


def _row(contrast: FeatureContrast) -> str:
    separation = "—" if contrast.separation is None else f"{contrast.separation:.3g}"
    return (
        f"| `{contrast.feature}` | {separation} | {contrast.attack_mean:.6g} ± "
        f"{contrast.attack_std:.3g} | {contrast.normal_mean:.6g} ± "
        f"{contrast.normal_std:.3g} |"
    )


def render_trace_profile(profile: TraceProfile) -> str:
    """O perfil como bloco Markdown, na forma que o LLM lê."""

    header = "| coluna | separação | ataque (média ± desvio) | normal (média ± desvio) |\n|---|---|---|---|"
    counts = ", ".join(f"`{name}`={count}" for name, count in sorted(profile.class_counts.items()))
    lines = [
        "## Perfil do trace observado",
        "",
        f"Classe perfilada: `{profile.attack_label}` contra `{profile.normal_label}` "
        f"(linhas por classe: {counts}). Separação = |Δ médias| / desvio combinado; "
        "acima de 0.2 já é um efeito, acima de 0.8 é forte.",
        "",
        "### Colunas que mais separam o ataque do normal",
        "",
        header,
        *(_row(c) for c in profile.top_features),
    ]
    if profile.constant_shift:
        lines += [
            "",
            "### Constantes nos dois lados, em valores diferentes (separação total)",
            "",
            header,
            *(_row(c) for c in profile.constant_shift),
        ]
    if profile.lever_features:
        lines += [
            "",
            "### Colunas que as regras do `programmable` mexem (direta ou indiretamente)",
            "",
            header,
            *(_row(c) for c in profile.lever_features),
        ]
    return "\n".join(lines) + "\n"
