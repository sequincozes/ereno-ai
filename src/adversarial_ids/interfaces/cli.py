"""Interface de linha de comando do adversarial-ids."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from adversarial_ids.config.attacks_registry import (
    DEFAULT_ATTACK_KEY,
    list_attack_keys,
)
from adversarial_ids.config.settings import (
    GENERATOR_MODE,
    INTENT_LOOP_DEFAULT_ROUNDS,
    ITERATION_HISTORY_PATH,
    LOOP_RECORDS_PATH,
    MODEL_ID,
    TOTAL_ITERATIONS,
)
from adversarial_ids.domain.loop_record import LoopRecord, LoopStageStatus
from adversarial_ids.interfaces.experiment_runner import (
    ExperimentRunner,
    create_default_runner,
)

RunIntentLoop = Callable[..., tuple[LoopRecord, ...]]


def _dashboard_app_path() -> Path:
    return Path(__file__).parent / "dashboard" / "app.py"


def _print_dashboard_hint(*, engine: str, stdout: TextIO) -> None:
    """Mostra onde estão os resultados e como abrir o dashboard."""

    app = _dashboard_app_path()
    if engine == "live":
        print(f"\nResultados salvos em: {ITERATION_HISTORY_PATH}", file=stdout)
    print(
        "\nPara ver o panorama (desempenho + explicabilidade) no dashboard:\n"
        f"  uv run --extra dashboard streamlit run {app}\n"
        "  Depois abra: http://localhost:8501\n"
        "  (ou rode com --open-dashboard para abrir automaticamente.)",
        file=stdout,
    )


def _launch_dashboard(*, stdout: TextIO, stderr: TextIO) -> None:
    """Abre o dashboard Streamlit (bloqueia até Ctrl+C). Streamlit abre o navegador."""

    if importlib.util.find_spec("streamlit") is None:
        print(
            "streamlit não está instalado. Rode: uv sync --extra dashboard",
            file=stderr,
        )
        return
    app = _dashboard_app_path()
    print(
        "\nAbrindo o dashboard em http://localhost:8501 (Ctrl+C para encerrar)...",
        file=stdout,
    )
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)])


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa o experimento adversarial do Laboratório Inteligente."
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Identificador do modelo Groq.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=TOTAL_ITERATIONS,
        help="Quantidade de iterações adversariais.",
    )
    parser.add_argument(
        "--generator-mode",
        choices=("cached", "jar"),
        default=GENERATOR_MODE,
        help="Modo do gerador sintético (usado apenas com --engine live).",
    )
    parser.add_argument(
        "--engine",
        choices=("demo", "live", "intent"),
        default="demo",
        help=(
            "demo (default): replay do histórico golden, sem Groq/Java. "
            "live: loop real Strategist->Analyst (exige GROQ_API_KEY). "
            "intent: pipeline intent-driven ponta a ponta (E3/E4; exige "
            "GROQ_API_KEY e --prompt)."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Apenas com --engine intent: intenção em linguagem natural "
            "(ex.: 'reduza o recall variando a temporização da falha'). O "
            "ataque-base vem da própria intenção, não de --attack."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=INTENT_LOOP_DEFAULT_ROUNDS,
        help=(
            "Apenas com --engine intent: rodadas encadeadas pela política de "
            "feedback (E10). Da segunda em diante a intenção vem da política, "
            "não de uma nova chamada de LLM. Distinto de --iterations, que "
            "pertence ao loop legado (--engine live/demo)."
        ),
    )
    parser.add_argument(
        "--orchestration",
        choices=("team", "direct"),
        default="team",
        help=(
            "Apenas com --engine live: 'team' (default) dirige os agentes por um "
            "agno.team.Team; 'direct' usa o encadeamento determinístico."
        ),
    )
    parser.add_argument(
        "--persona",
        choices=("conservative", "aggressive"),
        default="conservative",
        help=(
            "Apenas com --engine live: perfil do Estrategista. 'conservative' "
            "altera 1 campo por iteração; 'aggressive' altera até 3."
        ),
    )
    parser.add_argument(
        "--attack",
        choices=list_attack_keys(),
        default=DEFAULT_ATTACK_KEY,
        metavar="TIPO",
        help=(
            "Apenas com --engine live: tipo de ataque do ERENO que os agentes vão "
            "otimizar. Opções: " + ", ".join(list_attack_keys()) + "."
        ),
    )
    parser.add_argument(
        "--open-dashboard",
        action="store_true",
        help="Ao terminar, abre o dashboard Streamlit com o panorama dos resultados.",
    )
    return parser


def _stage_marker(status: LoopStageStatus) -> str:
    return {
        LoopStageStatus.SUCCEEDED: "OK",
        LoopStageStatus.FAILED: "FALHOU",
        LoopStageStatus.SKIPPED: "PULADO",
        LoopStageStatus.RUNNING: "RODANDO",
        LoopStageStatus.PENDING: "PENDENTE",
    }.get(status, status.value)


def _print_loop_record_summary(
    record: LoopRecord, *, stdout: TextIO, round_label: str | None = None
) -> None:
    header = f"\nLoop intent-driven concluído: run_id={record.run_id}"
    if round_label:
        header += f" ({round_label})"
    print(header, file=stdout)
    for stage in record.stages:
        line = f"  [{_stage_marker(stage.status)}] {stage.name}"
        if stage.artifact_ref:
            line += f" -> {stage.artifact_ref}"
        if stage.error:
            line += f" ({stage.error})"
        print(line, file=stdout)


def _print_campaign_summary(records: tuple[LoopRecord, ...], *, stdout: TextIO) -> None:
    total = len(records)
    for index, record in enumerate(records, start=1):
        round_label = f"Rodada {index}/{total}" if total > 1 else None
        _print_loop_record_summary(record, stdout=stdout, round_label=round_label)
    print(f"Registro salvo em: {LOOP_RECORDS_PATH}", file=stdout)


def _run_intent_engine(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    stderr: TextIO,
    run_intent_loop: RunIntentLoop | None,
) -> int:
    if not args.prompt:
        print(
            "Erro: --engine intent exige --prompt (a intenção em linguagem natural).",
            file=stderr,
        )
        return 2

    if args.rounds < 1:
        print(f"Erro: --rounds precisa ser >= 1 (veio {args.rounds}).", file=stderr)
        return 2

    if run_intent_loop is None:
        from adversarial_ids.agents.orchestrator.intent_live import (
            run_intent_loop as run_intent_loop,
        )

    try:
        records = run_intent_loop(
            prompt=args.prompt,
            model_id=args.model_id,
            generator_mode=args.generator_mode,
            rounds=args.rounds,
        )
    except KeyboardInterrupt:
        print("Execução interrompida pelo usuário.", file=stderr)
        return 130
    except Exception as error:  # fronteira da interface
        print(f"Erro ao executar o loop intent-driven: {error}", file=stderr)
        return 1

    _print_campaign_summary(records, stdout=stdout)
    has_failed_stage = any(
        stage.status == LoopStageStatus.FAILED
        for record in records
        for stage in record.stages
    )
    return 1 if has_failed_stage else 0


def run_cli(
    runner: ExperimentRunner | None = None,
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    run_intent_loop: RunIntentLoop | None = None,
) -> int:
    args = create_parser().parse_args(argv)

    if args.engine == "intent":
        return _run_intent_engine(
            args, stdout=stdout, stderr=stderr, run_intent_loop=run_intent_loop
        )

    generator_mode = args.generator_mode
    if runner is None:
        runner = create_default_runner(
            args.engine,
            orchestration=args.orchestration,
            persona=args.persona,
        )
        if args.engine == "demo":
            # O replay golden não usa Java; força o modo cacheado mesmo quando o
            # default de settings resolve para 'jar' (JAR presente na máquina).
            generator_mode = "cached"

    try:
        records = runner.run(
            iterations=args.iterations,
            model_id=args.model_id,
            generator_mode=generator_mode,
            attack=args.attack,
        )
    except KeyboardInterrupt:
        print("Execução interrompida pelo usuário.", file=stderr)
        return 130
    except Exception as error:  # fronteira da interface
        print(f"Erro ao executar o experimento: {error}", file=stderr)
        return 1

    if not records:
        print("Execução concluída sem registros.", file=stdout)
        return 0

    final_record = records[-1]
    final_f1 = final_record.metrics.f1_score_attack
    f1_text = "indisponível" if final_f1 is None else f"{final_f1:.4f}"
    print(
        "Execução concluída: "
        f"{len(records)} registro(s), "
        f"última iteração={final_record.iteration}, F1 final={f1_text}.",
        file=stdout,
    )

    _print_dashboard_hint(engine=args.engine, stdout=stdout)
    if args.open_dashboard:
        _launch_dashboard(stdout=stdout, stderr=stderr)

    return 0


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
