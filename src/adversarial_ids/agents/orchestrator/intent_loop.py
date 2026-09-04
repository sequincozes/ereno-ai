"""orchestrator/intent_loop.py — Orchestrator v2 do pipeline intent-driven (E3).

Encadeia INTENT → GENERATOR → ERENO → PREPROCESS → DETECTOR → DEFENDER sobre
peças já existentes — nenhuma lógica de geração, compilação, detecção ou
defesa nova aqui, só o controlador que amarra os estágios e registra cada um
como um ``LoopStage`` do ``LoopRecord`` (contrato congelado, ação 72h #2):

- INTENT      — ``IntentLike.interpret(prompt)`` (o ``IntentAgent`` real ou
  um stub de teste), já validado pelos dois portões de ``compile_intent``.
- GENERATOR   — ``core.intent_compiler.compile_attack_candidate`` (E2).
- ERENO       — ``GeneratorRunner.generate_dataset`` sobre o
  ``AttackCandidate.config`` compilado.
- PREPROCESS  — ``core.dataset_bundle_builder.build_dataset_bundle`` (E4):
  hash, classes e volume do trace validados antes do detector rodar.
- DETECTOR    — ``IdsEvaluator`` (Random Forest já existente) treinado no
  baseline do ataque e avaliado sobre o ``DatasetBundle`` aprovado,
  empacotado como ``DetectionReport`` (``core.detection_reporter``). O
  ``FeatureManifest`` (E6) do preprocessador ajustado no baseline é
  persistido como ``feature_manifest.json`` neste mesmo estágio — não é um
  ``LoopStage`` próprio, é artefato do DETECTOR.
- DEFENDER    — ``DefenderLike.defend(report, report_ref=...)`` (E5, o
  ``DefenderAgent`` real ou um stub de teste), já validado por
  ``agents.defender.tools.validate_plan_against_report`` contra o
  ``DetectionReport`` da mesma execução — toda evidência do ``DefensePlan``
  precisa bater com um valor real do relatório.

- FEEDBACK    — ``core.feedback_policy.decide_feedback`` (E10): decide,
  deterministicamente a partir do ``DetectionReport`` e do ``DefensePlan``
  desta rodada, se a campanha continua e qual é a próxima ``IntentSpec``.
  Nenhuma LLM roda neste estágio — mesmo guardrail central do pipeline.

``run(prompt)`` resolve **uma** rodada (mantém a assinatura histórica: uma
intenção em linguagem natural por chamada). ``run_campaign(prompt, rounds=N)``
encadeia até N rodadas: a partir da segunda, a ``IntentSpec`` não vem mais do
``IntentLike`` — vem de ``FeedbackDecision.next_intent`` da rodada anterior, e
o estágio ``intent`` é registrado como ``skipped`` (nenhuma chamada de LLM
naquela rodada, nunca fabricado como se tivesse rodado). Cada rodada ainda é
um ``LoopRecord`` completo, persistido individualmente; a linhagem da
campanha vive em ``LoopRecord.round``/``parent_run_id``.

Diferente do ``AdversarialWorkflow`` (loop legado Strategist↔Analyst, N
iterações de uma config de ataque ajustada por tool calling livre), este
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
``generator_mode="jar"`` para medir o efeito real da intenção compilada. Por
consequência, uma campanha em modo cacheado para na rodada 2 com
``no_improvement``: a métrica-objetivo não se move porque o dataset é o mesmo.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from adversarial_ids.config.attacks_registry import AttackSpec, get_attack_spec
from adversarial_ids.config.settings import (
    BASELINE_DATASET_PATH,
    FEEDBACK_MIN_DELTA,
    GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
    GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH,
    GENERATOR_MAX_RETRIES,
    GENERATOR_OUTPUT_DATASET_PATH,
    GENERATOR_RETRY_BACKOFF_SECONDS,
    GENERATOR_RUN_COMMAND,
    GENERATOR_RUNTIME_DIR,
    GENERATOR_TIMEOUT_SECONDS,
    INTENT_LOOP_DEFAULT_ROUNDS,
    INTENT_LOOP_MIN_ATTACK_ROWS,
    INTENT_LOOP_MIN_NORMAL_ROWS,
    INTENT_LOOP_OUTPUT_DIR,
    LOOP_RECORDS_PATH,
)
from adversarial_ids.core.dataset_bundle_builder import build_dataset_bundle
from adversarial_ids.core.detection_reporter import build_detection_report
from adversarial_ids.core.feedback_policy import decide_feedback
from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.core.intent_compiler import compile_attack_candidate
from adversarial_ids.domain.dataset_bundle import DatasetBundle
from adversarial_ids.domain.defense_plan import DefensePlan
from adversarial_ids.domain.detection_report import DetectionReport
from adversarial_ids.domain.feedback_decision import FeedbackDecision, RoundOutcome
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


@runtime_checkable
class DefenderLike(Protocol):
    """Qualquer gerador de DefensePlan (``DefenderAgent`` real ou stub de teste).

    Mais largo que ``IntentLike`` por um kwarg: ``report_ref`` é o caminho
    determinístico do ``detection_report.json`` desta execução, repassado
    para que ``Evidence.detection_report_ref`` também seja validável contra
    a execução real, não apenas contra os valores das métricas.
    """

    def defend(
        self, report: DetectionReport, *, report_ref: str | None = None
    ) -> DefensePlan: ...


class IntentLoopOrchestrator:
    """Controlador do pipeline intent-driven (E3), uma intenção por execução."""

    def __init__(
        self,
        *,
        intent_agent: IntentLike,
        defender_agent: DefenderLike,
        generator_mode: str = "cached",
        output_dir: Path | str = INTENT_LOOP_OUTPUT_DIR,
        save_path: Path | str | None = LOOP_RECORDS_PATH,
        min_attack_rows: int = INTENT_LOOP_MIN_ATTACK_ROWS,
        min_normal_rows: int = INTENT_LOOP_MIN_NORMAL_ROWS,
        cached_dataset_path: Path | str | None = None,
        feedback_min_delta: float = FEEDBACK_MIN_DELTA,
    ) -> None:
        if generator_mode not in _VALID_GENERATOR_MODES:
            raise ValueError(
                f"generator_mode inválido: {generator_mode!r}. "
                f"Use um de {_VALID_GENERATOR_MODES}."
            )
        self.intent_agent = intent_agent
        self.defender_agent = defender_agent
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
        # Ganho mínimo para a campanha considerar que a rodada progrediu (E10).
        # Exposto no construtor pelo mesmo motivo de ``min_attack_rows``: em
        # modo cacheado a métrica nunca se move, então um teste que precise
        # rodar mais de duas rodadas passa 0.0 aqui.
        self.feedback_min_delta = feedback_min_delta

    # ------------------------------------------------------------------ #
    # Loop principal                                                      #
    # ------------------------------------------------------------------ #
    def run(self, prompt: str, *, seed: int = 42) -> LoopRecord:
        """Roda **uma** rodada ponta a ponta e devolve o ``LoopRecord``.

        Não recebe um ``attack`` separado: o ataque-base vem de
        ``IntentSpec.base_attack``, extraído do próprio ``prompt`` pelo
        ``IntentLike`` (LLM ou stub) — um seletor de ataque redundante na
        CLI/API poderia divergir do que a intenção compilada realmente usa.

        Nunca levanta a exceção de um estágio para fora: qualquer falha é
        capturada, anexada ao ``LoopStage`` correspondente, e o registro
        (parcial) é persistido e devolvido do mesmo jeito. Inspecione
        ``record.stages`` para saber onde/por que uma execução parou.

        Para encadear rodadas pela política de feedback, use
        ``run_campaign``.
        """

        record, _decision = self._run_round(prompt, seed=seed)
        return record

    def run_campaign(
        self,
        prompt: str,
        *,
        rounds: int = INTENT_LOOP_DEFAULT_ROUNDS,
        seed: int = 42,
    ) -> tuple[LoopRecord, ...]:
        """Encadeia até ``rounds`` rodadas; a rodada N+1 nasce da política (E10).

        A rodada 1 interpreta ``prompt`` com o ``IntentLike``. Cada rodada
        seguinte roda a ``IntentSpec`` que o estágio FEEDBACK da anterior
        produziu — sem nova chamada de LLM do lado Red. A campanha para no
        primeiro destes casos: a decisão diz para parar, algum estágio da
        rodada falhou (o FEEDBACK nem chega a rodar), ou ``rounds`` foi
        atingido.

        Nunca levanta, pelo mesmo motivo de ``run()``: devolve os registros
        já concluídos (todos individualmente persistidos), e a causa da
        parada fica no ``LoopStage`` que falhou ou no ``feedback.json`` da
        última rodada.
        """

        if rounds < 1:
            raise ValueError(f"rounds precisa ser >= 1 (veio {rounds}).")

        records: list[LoopRecord] = []
        history: tuple[RoundOutcome, ...] = ()
        next_intent: IntentSpec | None = None
        parent_run_id: str | None = None

        for round_index in range(1, rounds + 1):
            record, decision = self._run_round(
                prompt,
                seed=seed,
                intent_override=next_intent,
                round_index=round_index,
                parent_run_id=parent_run_id,
                max_rounds=rounds,
                history=history,
            )
            records.append(record)

            if decision is None or not decision.should_continue:
                break

            history += (
                RoundOutcome(
                    round=round_index,
                    run_id=record.run_id,
                    objective_value=decision.objective_value,
                ),
            )
            next_intent = decision.next_intent
            parent_run_id = record.run_id

        return tuple(records)

    # ------------------------------------------------------------------ #
    # Uma rodada — os sete estágios, o registro e a decisão de feedback   #
    # ------------------------------------------------------------------ #
    def _run_round(
        self,
        prompt: str,
        *,
        seed: int = 42,
        intent_override: IntentSpec | None = None,
        round_index: int = 1,
        parent_run_id: str | None = None,
        max_rounds: int = 1,
        history: tuple[RoundOutcome, ...] = (),
    ) -> tuple[LoopRecord, FeedbackDecision | None]:
        """Roda os sete estágios de uma rodada e devolve registro + decisão.

        A decisão volta junto (em vez de ser relida do ``feedback.json``) para
        que ``run_campaign`` não dependa de I/O para saber se continua. É
        ``None`` quando algum estágio falhou antes do FEEDBACK.
        """

        run_id = new_run_id()
        run_dir = self.output_dir / run_id
        stages: list[LoopStage] = []
        started = time.perf_counter()
        record_seed = seed
        resolved: dict[str, Any] = {}
        decision: FeedbackDecision | None = None

        try:
            if intent_override is None:
                intent = self._stage(
                    stages, run_dir, "intent",
                    lambda: self.intent_agent.interpret(prompt),
                    persist_as="intent.json",
                )
            else:
                intent = self._inherited_intent_stage(
                    stages, run_dir, intent_override, round_index
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

            detection_report = self._stage(
                stages, run_dir, "detector",
                lambda: self._run_detector(generator, spec, dataset_bundle, run_dir),
                persist_as="detection_report.json",
            )

            report_ref = str(run_dir / "detection_report.json")
            defense_plan = self._stage(
                stages, run_dir, "defender",
                lambda: self.defender_agent.defend(
                    detection_report, report_ref=report_ref
                ),
                persist_as="defense_plan.json",
            )

            decision = self._stage(
                stages, run_dir, "feedback",
                lambda: decide_feedback(
                    intent=intent,
                    report=detection_report,
                    plan=defense_plan,
                    run_id=run_id,
                    round_index=round_index,
                    max_rounds=max_rounds,
                    parent_run_id=parent_run_id,
                    history=history,
                    min_delta=self.feedback_min_delta,
                ),
                persist_as="feedback.json",
            )
        except Exception:
            # A causa já foi anexada ao LoopStage correspondente por
            # `_stage` — aqui só interrompemos o pipeline; o LoopRecord com
            # o que foi concluído (+ o estágio que falhou) é persistido
            # abaixo do mesmo jeito.
            pass

        record = LoopRecord(
            run_id=run_id,
            source_prompt=prompt,
            seed=record_seed,
            stages=tuple(stages),
            total_duration_seconds=time.perf_counter() - started,
            round=round_index,
            parent_run_id=parent_run_id,
        )

        if self.save_path is not None:
            append_loop_record(self.save_path, record)

        return record, decision

    # ------------------------------------------------------------------ #
    # Estágio INTENT herdado (rodadas 2+) — nenhuma chamada de LLM        #
    # ------------------------------------------------------------------ #
    def _inherited_intent_stage(
        self,
        stages: list[LoopStage],
        run_dir: Path,
        intent: IntentSpec,
        round_index: int,
    ) -> IntentSpec:
        """Persiste a intenção que a política (E10) produziu, sem chamar o LLM.

        Registrado como ``skipped``, nunca ``succeeded``: o status precisa
        continuar dizendo a verdade sobre o que rodou nesta rodada — o
        ``IntentLike`` real não foi chamado, então não há duração de LLM a
        medir nem uma proposta a validar de novo (já passou pelos dois
        portões quando a rodada anterior a produziu).
        """

        artifact_path = run_dir / "intent.json"
        save_json(artifact_path, intent.model_dump(mode="json"))
        stages.append(
            LoopStage(
                name="intent",
                status=LoopStageStatus.SKIPPED,
                artifact_ref=str(artifact_path),
                error=(
                    f"Intenção herdada da rodada {round_index - 1} pela "
                    "política de feedback (E10): nenhuma chamada de LLM "
                    "nesta rodada."
                ),
            )
        )
        return intent

    # ------------------------------------------------------------------ #
    # Estágio DETECTOR — treina no baseline, avalia o candidato compilado #
    # ------------------------------------------------------------------ #
    def _run_detector(
        self,
        generator: GeneratorRunner,
        spec: AttackSpec,
        dataset_bundle: DatasetBundle,
        run_dir: Path,
    ) -> DetectionReport:
        baseline_config = load_json(spec.baseline_path)
        baseline_dataset_path = generator.generate_dataset(baseline_config, iteration=0)

        evaluator = IdsEvaluator(drop_cb_status=False, target_attack_label=spec.label)
        evaluator.train_baseline(baseline_dataset_path)

        # Manifest do preprocessador (E6), ajustado no baseline — não no
        # ``dataset_bundle.trace_path`` (a variante é só transformada, nunca
        # ajustada). Persistido antes de ``build_detection_report`` de
        # propósito: se a avaliação falhar o gate binário logo abaixo, o
        # manifest já está em disco para diagnosticar o que foi ajustado.
        manifest = evaluator.feature_manifest
        if manifest is not None:
            save_json(run_dir / "feature_manifest.json", manifest.model_dump(mode="json"))

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
            timeout_seconds=GENERATOR_TIMEOUT_SECONDS,
            max_retries=GENERATOR_MAX_RETRIES,
            retry_backoff_seconds=GENERATOR_RETRY_BACKOFF_SECONDS,
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
