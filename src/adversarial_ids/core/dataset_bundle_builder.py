"""build_dataset_bundle — gates de integração do trace do ERENO (épico E4).

Cobre o que o gate D6-14 exige antes de qualquer avaliação: "trace novo tem
hash, classes e volume válidos" e "nenhuma avaliação ocorre sem DatasetBundle
aprovado". Não é o preprocessador modular completo (fit/transform, NaN,
leakage) — isso é o épico E6 (``core/preprocessor.py::FeaturePreprocessor``,
ver ``docs/preprocessing.md``), que consome o trace já aprovado por este
gate; aqui só validamos o que o ``DatasetBundle`` (contrato congelado) já
promete: hash de conteúdo, classes presentes e volume mínimo por classe —
este último tanto em valor absoluto quanto em prevalência da classe de ataque.

Um trace que não passa neste gate nunca vira um ``DatasetBundle`` — o
``IntentLoopOrchestrator`` (E3) marca o estágio ``preprocess`` como falho e
para o pipeline antes do detector.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from adversarial_ids.domain.dataset_bundle import DatasetBundle

# Mesmos candidatos e mesma ordem de fallback (última coluna) que
# ``IdsEvaluator._find_label_column`` usa sobre o DataFrame — aqui operamos
# direto no cabeçalho do CSV, sem carregar o dataset inteiro em memória.
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


class DatasetGateError(ValueError):
    """Trace do ERENO rejeitado antes de alimentar o detector (gate do E4)."""


def _find_label_index(columns: tuple[str, ...]) -> int:
    if not columns:
        raise DatasetGateError("Trace sem colunas no cabeçalho.")
    lowered = [column.lower() for column in columns]
    for name in _LABEL_COLUMN_CANDIDATES:
        if name in lowered:
            return lowered.index(name)
    return len(columns) - 1


def build_dataset_bundle(
    trace_path: str | Path,
    *,
    lineage_run_id: str,
    expected_attack_label: str,
    min_attack_rows: int = 5,
    min_normal_rows: int = 5,
    min_attack_prevalence: float = 0.01,
) -> DatasetBundle:
    """Valida ``trace_path`` e devolve um ``DatasetBundle`` aprovado.

    Levanta ``DatasetGateError`` (mensagem acionável) quando o trace não
    existe, está vazio, não tem a classe de ataque esperada, não tem a
    classe ``normal``, ou qualquer uma das duas fica abaixo do piso de
    volume. O hash é sobre os bytes exatos do arquivo — dois traces com o
    mesmo conteúdo produzem o mesmo ``content_hash``, o que permite detectar
    reuso/deduplicação de traces entre execuções.

    ``min_attack_rows``/``min_normal_rows`` são pisos **absolutos** e só
    pegam trace vazio ou quebrado. ``min_attack_prevalence`` é o piso
    **proporcional** (fração das linhas rotuladas que são da classe de
    ataque) e cobre o caso que os absolutos deixam passar: um trace íntegro
    em que o ataque é raro demais para ser medido. As duas checagens são
    distintas de propósito — um trace pode ter 27 linhas de ataque (muito
    acima do piso absoluto de 5) e ainda assim 0,05% de prevalência, caso em
    que qualquer recall/precision que sair dali é ruído amostral, não
    evidência sobre o detector. ``0.0`` desliga só o piso proporcional.
    """

    if not 0.0 <= min_attack_prevalence <= 1.0:
        raise ValueError(
            "min_attack_prevalence precisa estar em [0.0, 1.0] "
            f"(veio {min_attack_prevalence})."
        )

    path = Path(trace_path)
    if not path.exists():
        raise DatasetGateError(f"Trace não encontrado: {path}")

    content = path.read_bytes()
    if not content:
        raise DatasetGateError(f"Trace vazio: {path}")
    content_hash = hashlib.sha256(content).hexdigest()

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            raise DatasetGateError(f"Trace sem cabeçalho: {path}") from None

        columns = tuple(column.strip() for column in header)
        label_index = _find_label_index(columns)

        class_counts: dict[str, int] = {}
        for row in reader:
            if len(row) <= label_index:
                continue
            label = row[label_index].strip()
            if not label:
                continue
            class_counts[label] = class_counts.get(label, 0) + 1

    if not class_counts:
        raise DatasetGateError(f"Trace sem registros rotulados: {path}")

    if expected_attack_label not in class_counts:
        raise DatasetGateError(
            f"Classe de ataque esperada {expected_attack_label!r} ausente em "
            f"{path}. Classes encontradas: {sorted(class_counts)}."
        )
    attack_rows = class_counts[expected_attack_label]
    if attack_rows < min_attack_rows:
        raise DatasetGateError(
            f"Volume da classe de ataque {expected_attack_label!r} abaixo do "
            f"piso em {path}: {attack_rows} < {min_attack_rows}."
        )

    labeled_rows = sum(class_counts.values())
    prevalence = attack_rows / labeled_rows
    if prevalence < min_attack_prevalence:
        raise DatasetGateError(
            f"Prevalência da classe de ataque {expected_attack_label!r} abaixo "
            f"do piso em {path}: {attack_rows}/{labeled_rows} = "
            f"{prevalence:.4%} < {min_attack_prevalence:.4%}. O trace é íntegro, "
            "mas raro demais para medir detecção — as métricas sairiam ruído "
            "amostral. Quem está reprovado é o dataset, não a intenção: gere "
            "de novo com o ataque mais intenso ou com menos tráfego benigno."
        )

    normal_rows = sum(
        count for label, count in class_counts.items() if label.lower() == "normal"
    )
    if normal_rows == 0:
        raise DatasetGateError(f"Classe 'normal' ausente em {path}.")
    if normal_rows < min_normal_rows:
        raise DatasetGateError(
            f"Volume da classe 'normal' abaixo do piso em {path}: "
            f"{normal_rows} < {min_normal_rows}."
        )

    return DatasetBundle(
        trace_path=str(path),
        columns=columns,
        classes=tuple(sorted(class_counts)),
        class_counts=class_counts,
        content_hash=content_hash,
        lineage_run_id=lineage_run_id,
    )
