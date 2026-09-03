"""orchestrator/intent_loop.py — Orchestrator v2 do pipeline intent-driven (E3).

Encadeia INTENT → GENERATOR → ERENO → PREPROCESS → DETECTOR sobre peças já
existentes — nenhuma lógica de geração, compilação ou detecção nova aqui,
só o controlador que amarra os estágios e registra cada um como um
``LoopStage`` do ``LoopRecord`` (contrato congelado, ação 72h #2):

- INTENT      — ``IntentLike.interpret(prompt)`` (o ``IntentAgent`` real ou
  um stub de teste), já validado pelos dois portões de ``compile_intent``.
- GENERATOR   — ``core.intent_compiler.compile_attack_candidate`` (E2).
- ERENO       — ``GeneratorRunner.generate_dataset`` sobre o
  ``AttackCandidate.config`` compilado.
- PREPROCESS  — ``core.dataset_bundle_builder.build_dataset_bundle`` (E4):
  hash, classes e volume do trace validados antes do detector rodar.
- DETECTOR    — ``IdsEvaluator`` (Random Forest já existente) treinado no
  baseline do ataque e avaliado sobre o ``DatasetBundle`` aprovado,
  empacotado como ``DetectionReport`` (``core.detection_reporter``).

DEFENDER (E5, ``DefensePlan``) e FEEDBACK (E10, política de retroalimentação)
não têm lógica própria nesta entrega — entram no ``LoopRecord`` como
estágios ``skipped`` com o motivo, nunca fabricados como se tivessem rodado.

Diferente do ``AdversarialWorkflow`` (loop legado Strategist↔Analyst, N
iterações de uma ``AttackConfig`` ajustada por tool calling livre), este
orquestrador resolve **uma** intenção em linguagem natural por chamada —
não itera. Os dois loops compartilham o núcleo determinístico
(``GeneratorRunner``/``IdsEvaluator``) mas não o controlador nem o contrato
de histórico (``LoopRecord`` aqui, ``IterationRecord`` lá).

Resiliente por estágio: uma falha em qualquer estágio marca aquele
``LoopStage`` como ``failed`` (com a mensagem de erro) e interrompe o
pipeline — mas ``run()`` sempre devolve o ``LoopRecord`` já persistido em
vez de propagar a exceção, para que o estágio e a causa fiquem rastreáveis
mesmo quando a execução não completa (critério de aceite da camada
"Operação": "falhas têm estágio/causa").

Nota de modo cacheado: em ``generator_mode="cached"`` o dataset do baseline
e o do candidato compilado são o **mesmo arquivo** (limitação já documentada
do ``GeneratorRunner``/loop legado — ver README "Limitações conhecidas") —
o DetectionReport resultante não reflete a variação física do ataque. Use
``generator_mode="jar"`` para medir o efeito real da intenção compilada.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from adversarial_ids.config.attacks_registry import AttackSpec, get_attack_spec
from adversarial_ids.config.settings import (
    BASELINE_DATASET_PATH,
    GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
    GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH,
    GENERATOR_OUTPUT_DATASET_PATH,
    GENERATOR_RUN_COMMAND,
    GENERATOR_RUNTIME_DIR,
    INTENT_LOOP_MIN_ATTACK_ROWS,
    INTENT_LOOP_MIN_NORMAL_ROWS,
    INTENT_LOOP_OUTPUT_DIR,
    LOOP_RECORDS_PATH,
)
from adversarial_ids.core.dataset_bundle_builder import build_dataset_bundle
from adversarial_ids.core.detection_reporter import build_detection_report
from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.core.intent_compiler import compile_attack_candidate
from adversarial_ids.domain.dataset_bundle import DatasetBundle
from adversarial_ids.domain.detection_report import DetectionReport
from adversarial_ids.domain.intent_spec import IntentSpec
from adversarial_ids.domain.loop_record import LoopRecord, LoopStage, LoopStageStatus
from adversarial_ids.shared.json_io import load_json, save_json
from adversarial_ids.shared.loop_record_store import append_loop_record
from adversarial_ids.shared.run_id import new_run_id

_VALID_GENERATOR_MODES = ("cached", "jar")


@runtime_checkable
class IntentLike(Protocol):
    """Qualquer compilador prompt→intenção (``IntentAgent`` real ou stub de teste)."""

    def interpret(self, prompt: str) -> IntentSpec: ...


class IntentLoopOrchestrator:
    """Controlador do pipeline intent-driven (E3), uma intenção por execução."""

    def __init__(
        self,
        *,
        intent_agent: IntentLike,
        generator_mode: str = "cached",
        output_dir: Path | str = INTENT_LOOP_OUTPUT_DIR,
        save_path: Path | str | None = LOOP_RECORDS_PATH,
        min_attack_rows: int = INTENT_LOOP_MIN_ATTACK_ROWS,
        min_normal_rows: int = INTENT_LOOP_MIN_NORMAL_ROWS,
        cached_dataset_path: Path | str | None = None,
    ) -> None:
        if generator_mode not in _VALID_GENERATOR_MODES:
            raise ValueError(
                f"generator_mode inválido: {generator_mode!r}. "
                f"Use um de {_VALID_GENERATOR_MODES}."
            )
        self.intent_agent = intent_agent
        self.generator_mode = generator_mode
        self.output_dir = Path(output_dir)
        self.save_path = Path(save_path) if save_path is not None else None
        self.min_attack_rows = min_attack_rows
        self.min_normal_rows = min_normal_rows
        # Override de teste: aponta o modo cacheado para uma seed pequena em
        # vez de settings.BASELINE_DATASET_PATH (o baseline real, ~80k
        # linhas). Produção usa o default (None -> BASELINE_DATASET_PATH).
        self.cached_dataset_path = (
            Path(cached_dataset_path) if cached_dataset_path is not None else None
        )

    # ------------------------------------------------------------------ #
    # Loop principal                                                      #
    # ------------------------------------------------------------------ #
    def run(self, prompt: str, *, seed: int = 42) -> LoopRecord:
        """Roda o pipeline ponta a ponta para ``prompt`` e devolve o ``LoopRecord``.

        Não recebe um ``attack`` separado: o ataque-base vem de
        ``IntentSpec.base_attack``, extraído do próprio ``prompt`` pelo
        ``IntentLike`` (LLM ou stub) — um seletor de ataque redundante na
        CLI/API poderia divergir do que a intenção compilada realmente usa.

        Nunca levanta a exceção de um estágio para fora: qualquer falha é
        capturada, anexada ao ``LoopStage`` correspondente, e o registro
        (parcial) é persistido e devolvido do mesmo jeito. Inspecione
        ``record.stages`` para saber onde/por que uma execução parou.
        """

        run_id = new_run_id()
        run_dir = self.output_dir / run_id
        stages: list[LoopStage] = []
        started = time.perf_counter()
        record_seed = seed
        resolved: dict[str, Any] = {}

        try:
            intent = self._stage(
                stages, run_dir, "intent",
                lambda: self.intent_agent.interpret(prompt),
                persist_as="intent.json",
            )
            record_seed = intent.seed

            def _compile_candidate() -> Any:
                # Resolve o AttackSpec aqui (não antes): é o que amarra o
                # GENERATOR/ERENO/DETECTOR ao ataque-base que a intenção
                # compilada de fato usa, não a um parâmetro externo separado.
                resolved["spec"] = get_attack_spec(intent.base_attack)
                return compile_attack_candidate(intent)

            candidate = self._stage(
                stages, run_dir, "generator",
                _compile_candidate,
                persist_as="attack_candidate.json",
            )
            spec = resolved["spec"]
            generator = self._build_generator(run_dir, spec)

            trace_path = self._stage(
                stages, run_dir, "ereno",
                lambda: generator.generate_dataset(candidate.config, iteration=1),
            )

            dataset_bundle = self._stage(
                stages, run_dir, "preprocess",
                lambda: build_dataset_bundle(
                    trace_path,
                    lineage_run_id=run_id,
                    expected_attack_label=spec.label,
                    min_attack_rows=self.min_attack_rows,
                    min_normal_rows=self.min_normal_rows,
                ),
                persist_as="dataset_bundle.json",
            )

            self._stage(
                stages, run_dir, "detector",
                lambda: self._run_detector(generator, spec, dataset_bundle),
                persist_as="detection_report.json",
            )
        except Exception:
            # A causa já foi anexada ao LoopStage correspondente por
            # `_stage` — aqui só interrompemos o pipeline; o LoopRecord com
            # o que foi concluído (+ o estágio que falhou) é persistido
            # abaixo do mesmo jeito.
            pass
        else:
            stages.append(
                LoopStage(
                    name="defender",
                    status=LoopStageStatus.SKIPPED,
                    error="DefensePlan (E5) ainda não implementado nesta entrega.",
                )
            )
            stages.append(
                LoopStage(
                    name="feedback",
                    status=LoopStageStatus.SKIPPED,
                    error="Política de feedback (E10) ainda não implementada nesta entrega.",
                )
            )

        record = LoopRecord(
            run_id=run_id,
            source_prompt=prompt,
            seed=record_seed,
            stages=tuple(stages),
            total_duration_seconds=time.perf_counter() - started,
        )

        if self.save_path is not None:
            append_loop_record(self.save_path, record)

        return record

    # ------------------------------------------------------------------ #
    # Estágio DETECTOR — treina no baseline, avalia o candidato compilado #
    # ------------------------------------------------------------------ #
    def _run_detector(
        self,
        generator: GeneratorRunner,
        spec: AttackSpec,
        dataset_bundle: DatasetBundle,
    ) -> DetectionReport:
        baseline_config = load_json(spec.baseline_path)
        baseline_dataset_path = generator.generate_dataset(baseline_config, iteration=0)

        evaluator = IdsEvaluator(drop_cb_status=False, target_attack_label=spec.label)
        evaluator.train_baseline(baseline_dataset_path)

        split = (
            f"train_test_{int((1 - evaluator.test_size) * 100)}_"
            f"{int(evaluator.test_size * 100)}_seed{evaluator.random_state}"
        )
        return build_detection_report(
            evaluator,
            dataset_bundle.trace_path,
            split=split,
        )

    # ------------------------------------------------------------------ #
    # Núcleo (GeneratorRunner por execução, isolado por run_id)          #
    # ------------------------------------------------------------------ #
    def _build_generator(self, run_dir: Path, spec: AttackSpec) -> GeneratorRunner:
        cached_dataset = None
        if self.generator_mode == "cached":
            cached_dataset = self.cached_dataset_path or BASELINE_DATASET_PATH
        return GeneratorRunner(
            runtime_dir=GENERATOR_RUNTIME_DIR,
            output_dataset_path=GENERATOR_OUTPUT_DATASET_PATH,
            run_command=GENERATOR_RUN_COMMAND,
            suggested_config_path=str(run_dir / "attack_config.json"),
            action_config_relative_path=GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
            benign_action_config_relative_path=(
                GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH
            ),
            benign_seed_path=BASELINE_DATASET_PATH,
            segment_name=spec.segment_name,
            cached_dataset_path=cached_dataset,
        )

    # ------------------------------------------------------------------ #
    # Runner de estágio — mede duração, persiste artefato, registra falha #
    # ------------------------------------------------------------------ #
    def _stage(
        self,
        stages: list[LoopStage],
        run_dir: Path,
        name: str,
        fn: Any,
        *,
        persist_as: str | None = None,
    ) -> Any:
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            stages.append(
                LoopStage(
                    name=name,
                    status=LoopStageStatus.FAILED,
                    duration_seconds=time.perf_counter() - start,
                    error=str(exc),
                )
            )
            raise

        duration = time.perf_counter() - start
        artifact_ref: str | None
        if persist_as is not None:
            artifact_path = run_dir / persist_as
            save_json(artifact_path, result.model_dump(mode="json"))
            artifact_ref = str(artifact_path)
        elif isinstance(result, str):
            artifact_ref = result
        else:
            artifact_ref = None

        stages.append(
            LoopStage(
                name=name,
                status=LoopStageStatus.SUCCEEDED,
                duration_seconds=duration,
                artifact_ref=artifact_ref,
            )
        )
        return result
