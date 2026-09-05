"""Roda os detectores registrados sob um protocolo e emite o relatório comparativo.

Execute com ``uv run python scripts/compare_detectors.py``. É o artefato que o
gate de saída da janela D36-46 pede ("mesmo split/protocolo; relatório
comparativo; fallback RF preservado") — o registro do épico E8 tornou RF/DT/SVM
intercambiáveis, este script é o que efetivamente os compara.

Grava ``outputs/detector_comparison.json`` (um ``DetectorComparison`` válido) e
imprime a tabela ranqueada por F1/recall/latência.

Exemplos:

    # Todos os detectores no baseline cacheado (sem Java, sem Groq)
    uv run python scripts/compare_detectors.py

    # Só as árvores, ranqueando por latência
    uv run python scripts/compare_detectors.py --detector random_forest decision_tree \\
        --rank-by latency_ms

    # Ablação controlada: mesma escala para todos, com seleção de features ligada
    uv run python scripts/compare_detectors.py --scaler standard \\
        --feature-selection mutual_info --feature-selection-top-k 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adversarial_ids.config.settings import BASELINE_DATASET_PATH, OUTPUTS_DIR  # noqa: E402
from adversarial_ids.core.detector_comparison import (  # noqa: E402
    compare_detectors,
    format_comparison_table,
)
from adversarial_ids.domain.detector_manifest import DETECTOR_KEYS  # noqa: E402
from adversarial_ids.shared.json_io import save_json  # noqa: E402

DEFAULT_OUTPUT_PATH = OUTPUTS_DIR / "detector_comparison.json"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline",
        default=str(BASELINE_DATASET_PATH),
        help="CSV de treino (default: o baseline versionado em data/).",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "CSV de avaliação. Default: o mesmo do --baseline, que exercita o "
            "protocolo mas NÃO mede variação física de ataque (mesma limitação "
            "do modo cacheado); use um trace do ERENO para um comparativo real."
        ),
    )
    parser.add_argument(
        "--detector",
        nargs="+",
        choices=list(DETECTOR_KEYS),
        default=list(DETECTOR_KEYS),
        metavar="MODELO",
        help="Detectores a comparar. Default: todos (" + ", ".join(DETECTOR_KEYS) + ").",
    )
    parser.add_argument(
        "--rank-by",
        choices=("f1", "recall", "latency_ms"),
        default="f1",
        help="Métrica que ordena o ranking; as outras duas desempatam.",
    )
    parser.add_argument(
        "--attack-label",
        default=None,
        help="Rótulo da classe de ataque no CSV (ancora a detecção da classe).",
    )
    parser.add_argument(
        "--scaler",
        choices=("none", "standard"),
        default=None,
        help=(
            "Força a mesma escala para todos. Default: cada detector resolve a "
            "sua (standard para os SVMs). Forçar é necessário para uma ablação "
            "controlada com --feature-selection ligado."
        ),
    )
    parser.add_argument("--feature-selection", choices=("none", "mutual_info"), default="none")
    parser.add_argument("--feature-selection-top-k", type=int, default=None)
    parser.add_argument("--undersampling", choices=("none", "random"), default="none")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Onde gravar o JSON (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    comparison = compare_detectors(
        args.baseline,
        args.variant or args.baseline,
        detectors=args.detector,
        ranking_metric=args.rank_by,
        target_attack_label=args.attack_label,
        scaler=args.scaler,
        evaluator_kwargs={
            "feature_selection": args.feature_selection,
            "feature_selection_top_k": args.feature_selection_top_k,
            "undersampling": args.undersampling,
            "test_size": args.test_size,
            "random_state": args.seed,
        },
    )

    output_path = Path(args.output)
    save_json(output_path, comparison.model_dump(mode="json"))

    print()
    print(format_comparison_table(comparison))
    print()
    print(f"Relatório salvo em: {output_path}")

    # Exit code diz se o comparativo é utilizável como evidência do gate: um
    # protocolo não uniforme, ou um detector que nem rodou, é resultado válido
    # de investigar — mas não é "comparação justa" concluída.
    if not comparison.protocol_consistent:
        return 1
    if any(run.status == "failed" for run in comparison.runs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
