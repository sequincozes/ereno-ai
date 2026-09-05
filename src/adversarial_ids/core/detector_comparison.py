"""compare_detectors — roda N detectores sob um protocolo e ranqueia (épico E8).

Fecha o gate de saída da janela D36-46: o registro de ``core/detectors.py``
tornou RF/DT/SVM intercambiáveis, mas cada execução treinava um só. Este
módulo é o que produz o *relatório comparativo* — treina cada detector sobre o
mesmo baseline, avalia sobre a mesma variante, e devolve um
``DetectorComparison`` com o ranking por F1/recall/latência.

Nenhuma lógica de detecção, preparo ou métrica nova: é um laço sobre
``IdsEvaluator`` + ``build_detection_report``, exatamente as peças que uma
execução única já usa. O valor está em três garantias que só existem quando os
detectores rodam juntos:

- **Mesmo protocolo, de verdade.** Todos recebem o mesmo par de datasets, a
  mesma seed e os mesmos parâmetros de preparo. O que cada um de fato treinou
  é conferido no fim (espaço de features idêntico) e registrado em
  ``protocol_consistent`` — a garantia do E8 tem um furo conhecido quando a
  seleção de features do E7 está ligada, e aqui ele fica detectável.
- **Falha isolada não derruba os demais.** Cada detector roda dentro do seu
  próprio try/except; um SVM que estoure tempo ou memória vira uma linha
  ``failed`` com a causa, e os outros continuam. Exigência literal da tabela de
  aceite ("falha isolada não derruba os demais").
- **Fallback RF preservado.** ``random_forest`` é apenas mais uma linha, com o
  mesmo caminho de código de sempre — comparar não muda o detector default.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from adversarial_ids.core.detection_reporter import build_detection_report
from adversarial_ids.core.detectors import DETECTOR_KEYS, detector_spec
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.domain.detector_comparison import (
    DetectorComparison,
    DetectorRun,
    RankingMetric,
    rank_runs,
)


def compare_detectors(
    baseline_dataset_path: str,
    variant_dataset_path: str,
    *,
    detectors: Sequence[str] = DETECTOR_KEYS,
    ranking_metric: RankingMetric = "f1",
    target_attack_label: str | None = None,
    drop_cb_status: bool = False,
    scaler: str | None = None,
    evaluator_kwargs: dict[str, Any] | None = None,
) -> DetectorComparison:
    """Treina cada detector no baseline, avalia na variante, devolve o comparativo.

    ``scaler=None`` (default) deixa cada detector resolver a própria escala,
    que é o modo honesto de comparar "cada detector no seu melhor preparo".
    Passar ``scaler="standard"`` força todos à mesma escala — necessário para
    uma ablação controlada, e a única forma de garantir espaço de features
    idêntico quando a seleção do E7 está ligada (ver ``docs/detectors.md``).

    ``evaluator_kwargs`` repassa o resto da configuração do ``IdsEvaluator``
    (``feature_selection``, ``undersampling``, ``test_size``, ``random_state``)
    **igual para todos** — é o que mantém o protocolo comum; por isso é um dict
    único e não um por detector.
    """

    if not detectors:
        raise ValueError("compare_detectors precisa de pelo menos um detector.")

    seen: set[str] = set()
    ordered: list[str] = []
    for key in detectors:
        detector_spec(key)  # levanta DetectorError para chave desconhecida
        if key not in seen:
            seen.add(key)
            ordered.append(key)

    shared = dict(evaluator_kwargs or {})
    runs: list[DetectorRun] = []
    split: str | None = None

    for key in ordered:
        try:
            evaluator = IdsEvaluator(
                detector=key,
                drop_cb_status=drop_cb_status,
                target_attack_label=target_attack_label,
                scaler=scaler,
                **shared,
            )
            evaluator.train_baseline(baseline_dataset_path)

            run_split = (
                f"train_test_{int((1 - evaluator.test_size) * 100)}_"
                f"{int(evaluator.test_size * 100)}_seed{evaluator.random_state}"
            )
            # Todos compartilham test_size/random_state (vêm do mesmo `shared`),
            # então divergir aqui seria um bug de construção, não um resultado.
            split = split or run_split

            report = build_detection_report(
                evaluator, variant_dataset_path, split=run_split
            )
            manifest = evaluator.detector_manifest
            assert manifest is not None  # train_baseline concluiu

            runs.append(
                DetectorRun(
                    detector=key,  # type: ignore[arg-type]  # validado por detector_spec
                    status="succeeded",
                    report=report,
                    resolved_scaler=manifest.resolved_scaler,
                    importance_kind=manifest.importance_kind,
                    trained_rows=manifest.trained_rows,
                    trained_features=manifest.trained_features,
                )
            )
        except Exception as error:  # falha isolada não derruba os demais
            print(f"[COMPARE] {key}: FALHOU ({type(error).__name__}: {error})")
            runs.append(
                DetectorRun(
                    detector=key,  # type: ignore[arg-type]
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            )

    if split is None:
        # Nenhum detector chegou a treinar: sem partição comum não há
        # comparativo a emitir, e inventar uma string de split seria descrever
        # um protocolo que não aconteceu.
        raise RuntimeError(
            "Nenhum detector concluiu o treino; não há protocolo comum a relatar. "
            f"Causas: {[run.error for run in runs]!r}"
        )

    consistent, notes = _check_protocol(runs)

    return DetectorComparison(
        baseline_dataset=str(baseline_dataset_path),
        variant_dataset=str(variant_dataset_path),
        split=split,
        ranking_metric=ranking_metric,
        runs=tuple(runs),
        ranking=rank_runs(tuple(runs), metric=ranking_metric),
        protocol_consistent=consistent,
        protocol_notes=notes,
    )


def _check_protocol(runs: list[DetectorRun]) -> tuple[bool, tuple[str, ...]]:
    """Confere que os detectores bem-sucedidos treinaram sobre o mesmo espaço.

    Divergir não é erro de execução — é um resultado sobre o experimento, e
    precisa aparecer no relatório em vez de derrubá-lo. O caso conhecido é a
    seleção de features do E7 ligada com escalas diferentes por detector.
    """

    succeeded = [run for run in runs if run.status == "succeeded"]
    spaces = {run.trained_features for run in succeeded}

    notes: list[str] = []
    if len(spaces) > 1:
        by_space = {
            features: sorted(r.detector for r in succeeded if r.trained_features == features)
            for features in spaces
        }
        notes.append(
            "Detectores treinaram sobre espaços de features diferentes: "
            + "; ".join(
                f"{detectors} -> {len(features)} features {list(features)!r}"
                for features, detectors in sorted(by_space.items(), key=lambda i: i[1])
            )
            + ". Causa provável: seleção de features (E7) ajustada sobre X já "
            "escalado, com escalas diferentes por detector. Passe scaler= para "
            "forçar a mesma escala nos dois lados (ver docs/detectors.md)."
        )

    failed = [run for run in runs if run.status == "failed"]
    if failed:
        notes.append(
            "Detectores fora do ranking por falha: "
            + ", ".join(f"{run.detector} ({run.error})" for run in failed)
        )

    # Só a divergência de espaço de features quebra a comparabilidade; uma
    # falha isolada apenas reduz o conjunto comparado, e por isso vira nota
    # sem derrubar a flag.
    return len(spaces) <= 1, tuple(notes)


def format_comparison_table(comparison: DetectorComparison) -> str:
    """Renderiza o comparativo como tabela de texto para stdout/relatório."""

    header = f"{'#':>2}  {'detector':<14} {'F1':>7} {'recall':>7} {'prec':>7} {'latência':>10}  {'escala':<9} {'importância':<12}"
    lines = [header, "-" * len(header)]

    position = {key: index + 1 for index, key in enumerate(comparison.ranking)}
    for run in sorted(comparison.runs, key=lambda r: (position.get(r.detector, 99), r.detector)):
        if run.status == "failed":
            lines.append(f"{'-':>2}  {run.detector:<14} {'FALHOU':>7}  {run.error}")
            continue
        report = run.report
        assert report is not None
        lines.append(
            f"{position[run.detector]:>2}  {run.detector:<14} "
            f"{report.f1:>7.4f} {report.recall:>7.4f} {report.precision:>7.4f} "
            f"{report.latency_ms:>9.1f}ms  {run.resolved_scaler or '-':<9} "
            f"{run.importance_kind or '-':<12}"
        )

    lines.append("")
    lines.append(f"split: {comparison.split}   |   ranking por: {comparison.ranking_metric}")
    lines.append(f"vencedor: {comparison.winner or '(nenhum detector concluiu)'}")
    if not comparison.protocol_consistent:
        lines.append("AVISO: protocolo NÃO uniforme entre os detectores.")
    for note in comparison.protocol_notes:
        lines.append(f"  - {note}")

    return "\n".join(lines)
