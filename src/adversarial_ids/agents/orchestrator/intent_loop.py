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
  hash, classes e volume do trace validados antes do detector rodar — volume
  tanto absoluto quanto em prevalência da classe de ataque, porque um trace
  íntegro mas com ataque raro demais produz métrica de acaso.
- DETECTOR    — ``IdsEvaluator`` treinado no baseline do ataque e avaliado
  sobre o ``DatasetBundle`` aprovado, empacotado como ``DetectionReport``
  (``core.detection_reporter``). Qual modelo treina vem do parâmetro
  ``detector`` (E8: ``random_forest`` default, ``decision_tree``,
  ``svm_linear``, ``svm_rbf`` — ver ``core/detectors.py``). Os três manifests
  do preparo/treino são persistidos neste mesmo estágio:
  ``feature_manifest.json`` (E6, preprocessador ajustado no baseline),
  ``selection_manifest.json`` (E7, seleção de features + undersampling) e
  ``detector_manifest.json`` (E8, detector treinado + escala aplicada).
  Nenhum dos três é um ``LoopStage`` próprio — são artefatos do DETECTOR.
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

``resume_campaign(run_id)`` continua uma campanha interrompida (estágio que
falhou, orçamento de tokens, processo morto) a partir do ledger, sem reexecutar
nem regravar rodada nenhuma; o ponto de retomada é resolvido por
``campaign_resume.plan_resume``. Ver ``docs/feedback_policy.md``.

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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from adversarial_ids.agents.orchestrator.campaign_resume import (
    CampaignResumeError,
    plan_resume,
)
from adversarial_ids.config.attacks_registry import AttackSpec, get_attack_spec
from adversarial_ids.config.settings import (
    BASELINE_DATASET_PATH,
    DETECTOR_MODE,
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
    INTENT_LOOP_TOKEN_BUDGET,
    INTENT_LOOP_MIN_ATTACK_PREVALENCE,
    INTENT_LOOP_MIN_ATTACK_ROWS,
    INTENT_LOOP_MIN_NORMAL_ROWS,
    INTENT_LOOP_OUTPUT_DIR,
    LOOP_RECORDS_PATH,
)
from adversarial_ids.core.dataset_bundle_builder import build_dataset_bundle
from adversarial_ids.core.detection_reporter import build_detection_report
from adversarial_ids.core.defense_rules import evaluate_plan_rules
from adversarial_ids.core.detectors import detector_spec
from adversarial_ids.core.feedback_policy import decide_feedback
from adversarial_ids.core.generators import (
    GeneratorLike,
    GeneratorRequest,
    build_generator,
    manifest_for,
)
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.core.intent_compiler import compile_attack_candidate
from adversarial_ids.domain.dataset_bundle import DatasetBundle
from adversarial_ids.domain.defense_plan import DefensePlan
from adversarial_ids.domain.detection_report import DetectionReport
from adversarial_ids.domain.feedback_decision import FeedbackDecision, RoundOutcome
from adversarial_ids.domain.intent_spec import IntentSpec
from adversarial_ids.domain.loop_event import LoopEvent
from adversarial_ids.domain.run_usage import AgentUsage
from adversarial_ids.domain.generator_manifest import GENERATOR_KEYS
from adversarial_ids.domain.loop_record import LoopRecord, LoopStage, LoopStageStatus
from adversarial_ids.shared.json_io import load_json, save_json
from adversarial_ids.shared.loop_event_store import (
    JsonlEventSink,
    LoopEventSink,
    fanout,
)
from adversarial_ids.shared.loop_record_store import (
    append_loop_record,
    load_loop_records,
)
from adversarial_ids.shared.redaction import redact
from adversarial_ids.shared.run_id import new_run_id

# Vocabulário de geradores: vem do registro, para a CLI e o orquestrador
# nunca aceitarem uma chave que não constrói nada.
_VALID_GENERATOR_MODES = GENERATOR_KEYS


@runtime_checkable
class IntentLike(Protocol):
    """Qualquer compilador prompt→intenção (``IntentAgent`` real ou stub de teste)."""

    def interpret(self, prompt: str) -> IntentSpec: ...


@runtime_checkable
class DefenderLike(Protocol):
    """Qualquer gerador de DefensePlan (``DefenderAgent`` real ou stub de teste).

    Mais largo que ``IntentLike`` por dois kwargs: ``report_ref`` é o caminho
    determinístico do ``detection_report.json`` desta execução, repassado
    para que ``Evidence.detection_report_ref`` também seja validável contra
    a execução real, não apenas contra os valores das métricas; ``attack_key``
    é o ataque-base que a intenção compilada usou, e liga os achados de playbook
    da avaliação por regras (E5) — que aconselham, nunca recusam.
    """

    def defend(
        self,
        report: DetectionReport,
        *,
        report_ref: str | None = None,
        attack_key: str | None = None,
    ) -> DefensePlan: ...


def _no_sink(event: LoopEvent) -> None:
    """Sink neutro: uma rodada sem observador nenhum não é caso especial."""


@dataclass
class _RoundContext:
    """O estado vivo de uma rodada: onde gravar, o que já rodou, quem observa.

    Os três andavam soltos como variáveis locais de ``_run_round`` e eram
    repassados a cada ``_stage`` na mão. Juntá-los é o que torna impossível
    registrar um ``LoopStage`` sem emitir o ``LoopEvent`` correspondente — a
    timeline ao vivo e o registro final saem do mesmo lugar, e não podem
    discordar sobre o que aconteceu.
    """

    run_id: str
    run_dir: Path
    round_index: int
    emit: LoopEventSink = _no_sink
    stages: list[LoopStage] = field(default_factory=list)
    _sequence: int = 0

    def event(self, kind: str, **fields: Any) -> None:
        """Emite um evento já numerado; o sink nunca derruba a execução."""

        event = LoopEvent.model_validate(
            {
                "run_id": self.run_id,
                "round": self.round_index,
                "sequence": self._sequence,
                "kind": kind,
                **fields,
                # Redigido aqui, e não em quem chama: este é o único caminho
                # pelo qual uma mensagem vira evento, então nenhum chamador
                # futuro consegue esquecer (ver shared/redaction.py).
                "message": redact(fields.get("message")),
            }
        )
        self._sequence += 1
        self.emit(event)

    def record_stage(self, stage: LoopStage) -> None:
        """Anexa o estágio ao registro e anuncia o mesmo fato como evento."""

        self.stages.append(stage)
        self.event(
            "stage_finished",
            stage=stage.name,
            status=stage.status,
            duration_seconds=stage.duration_seconds,
            artifact_ref=stage.artifact_ref,
            message=stage.error,
        )


@dataclass(frozen=True)
class ReplicateResult:
    """Resumo de um lote de réplicas (Fase 2.R): os registros e a estatística.

    Runtime, não um contrato persistido — a estatística é derivável dos
    ``LoopRecord``/``feedback.json`` do lote a qualquer momento, então não vira
    um artefato próprio. ``objective_values`` traz só as réplicas que chegaram ao
    FEEDBACK com uma métrica-objetivo; ``n_total`` conta todas, para que a
    diferença revele quantas falharam ou não tinham métrica de evasão.
    """

    batch_id: str
    records: tuple[LoopRecord, ...]
    objective_metric: str | None
    objective_values: tuple[float, ...]
    n_total: int
    llm_calls: int

    @property
    def mean(self) -> float | None:
        from statistics import fmean

        return fmean(self.objective_values) if self.objective_values else None

    @property
    def stdev(self) -> float | None:
        from statistics import stdev

        # Desvio amostral pede >= 2 pontos; com um só, o desvio é 0 por definição
        # da amostra, mas reportá-lo como número esconderia que há uma medida só.
        return stdev(self.objective_values) if len(self.objective_values) >= 2 else None


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
        min_attack_prevalence: float = INTENT_LOOP_MIN_ATTACK_PREVALENCE,
        cached_dataset_path: Path | str | None = None,
        feedback_min_delta: float = FEEDBACK_MIN_DELTA,
        detector: str = DETECTOR_MODE,
        event_sink: LoopEventSink | None = None,
        token_budget: int = INTENT_LOOP_TOKEN_BUDGET,
    ) -> None:
        if generator_mode not in _VALID_GENERATOR_MODES:
            raise ValueError(
                f"generator_mode inválido: {generator_mode!r}. "
                f"Use um de {_VALID_GENERATOR_MODES}."
            )
        # Valida a chave já na construção (levanta ``DetectorError``), não no
        # estágio DETECTOR: uma chave inválida é erro de configuração do
        # chamador, não uma falha de estágio a registrar no ``LoopRecord``.
        detector_spec(detector)
        self.detector = detector
        self.intent_agent = intent_agent
        self.defender_agent = defender_agent
        self.generator_mode = generator_mode
        self.output_dir = Path(output_dir)
        self.save_path = Path(save_path) if save_path is not None else None
        self.min_attack_rows = min_attack_rows
        self.min_normal_rows = min_normal_rows
        self.min_attack_prevalence = min_attack_prevalence
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
        # Teto de tokens da campanha (E11); 0 desliga. Ver `run_campaign`.
        if token_budget < 0:
            raise ValueError(
                f"token_budget precisa ser >= 0 (veio {token_budget}); "
                "use 0 para desligar o limite."
            )
        self.token_budget = token_budget
        # Observador opcional da timeline (E11). O sink de arquivo é sempre
        # ligado, por execução; este aqui é quem *mais* quer ver — a UI ao vivo
        # registra um sink de memória, um teste registra uma lista.
        self.event_sink = event_sink

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

        record, _decision, _intent = self._run_round(prompt, seed=seed)
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
        rodada falhou (o FEEDBACK nem chega a rodar), ``rounds`` foi atingido,
        ou o orçamento de tokens acabou.

        O orçamento (E11) é checado **entre** rodadas e fica fora da política de
        feedback de propósito: a política decide se vale a pena continuar
        *cientificamente*, e o teto decide se dá para continuar
        *operacionalmente*. Misturar os dois faria um estouro de custo aparecer
        como um veredito sobre o experimento. Uma campanha cujos agentes não
        informam consumo nunca é interrompida por aqui — ver ``_round_usage``.

        Nunca levanta, pelo mesmo motivo de ``run()``: devolve os registros
        já concluídos (todos individualmente persistidos), e a causa da
        parada fica no ``LoopStage`` que falhou ou no ``feedback.json`` da
        última rodada.
        """

        if rounds < 1:
            raise ValueError(f"rounds precisa ser >= 1 (veio {rounds}).")

        return self._chain(prompt, seed=seed, first_round=1, max_rounds=rounds)

    def run_replicates(
        self, prompt: str, *, seeds: Sequence[int]
    ) -> ReplicateResult:
        """Roda a **mesma** intenção sob várias seeds de geração (Fase 2.R).

        O ponto: entre réplicas só a geração muda; a intenção é a mesma. Então o
        LLM é chamado **uma única vez** — na primeira réplica — e a ``IntentSpec``
        resolvida é reusada nas demais, variando só a seed que vai ao gerador. N
        réplicas custam 1 chamada de intent, não N (decisivo com o TPM da conta).

        Cada réplica é um ``LoopRecord`` próprio, com sua ``seed`` e um
        ``replicate_batch_id`` comum. Devolve um ``ReplicateResult`` com a
        métrica-objetivo por réplica e a estatística do lote (média/desvio).

        Só é fisicamente significativo em ``generator_mode="jar"``: em cached todo
        trace é o mesmo arquivo e as réplicas colapsam — o chamador (CLI) avisa.
        Nunca levanta por falha de estágio, como ``run``/``run_campaign``: uma
        réplica que falha entra no lote com seu ``LoopStage`` de causa e fica de
        fora da estatística.
        """

        if not seeds:
            raise ValueError("run_replicates precisa de ao menos uma seed.")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"as seeds do lote precisam ser distintas: {list(seeds)}.")

        batch_id = new_run_id()
        records: list[LoopRecord] = []
        values: list[float] = []
        metric: str | None = None
        base_intent: IntentSpec | None = None
        llm_calls = 0

        for index, seed in enumerate(seeds):
            first = base_intent is None
            note = None if first else (
                f"Réplica {index + 1}/{len(seeds)} do lote {batch_id} sob seed "
                f"{seed}: intenção reusada da primeira réplica, sem chamada de LLM."
            )
            record, decision, intent = self._run_round(
                prompt,
                intent_override=base_intent,
                intent_note=note,
                seed_override=seed,
                replicate_batch_id=batch_id,
            )
            records.append(record)
            # Só a primeira réplica chama o LLM; guarda a intenção para as demais.
            # Se o próprio estágio INTENT falhou (intent None), a próxima tenta de
            # novo — melhor que o lote inteiro herdar um None.
            if first and intent is not None:
                base_intent = intent
                llm_calls += 1
            if decision is not None and decision.objective_value is not None:
                metric = decision.objective_metric
                values.append(decision.objective_value)

        return ReplicateResult(
            batch_id=batch_id,
            records=tuple(records),
            objective_metric=metric,
            objective_values=tuple(values),
            n_total=len(records),
            llm_calls=llm_calls,
        )

    def resume_campaign(
        self, run_id: str, *, rounds: int | None = None
    ) -> tuple[LoopRecord, ...]:
        """Retoma a campanha de ``run_id`` de onde ela parou, sem duplicar rodadas.

        O ponto de retomada vem de ``campaign_resume.plan_resume`` sobre o ledger
        (``save_path``): refaz a última rodada se ela falhou, roda a próxima se a
        política mandou continuar e ninguém rodou, e não faz nada se a política
        encerrou — devolve ``()`` nesse caso. Só as rodadas novas voltam; as já
        gravadas nunca são reexecutadas nem regravadas.

        Levanta ``CampaignResumeError`` quando não há o que retomar por erro do
        chamador (sem ledger, ``run_id`` desconhecido, artefato apagado, detector
        diferente do da campanha). Uma vez retomada, a campanha segue as mesmas
        regras de ``run_campaign`` — inclusive nunca levantar por falha de
        estágio. O orçamento de tokens conta só as rodadas desta chamada:
        retomar uma campanha cortada pelo teto é a decisão explícita de gastar
        mais, e contar o gasto anterior a cortaria de novo na primeira rodada.
        """

        if self.save_path is None:
            raise CampaignResumeError(
                "Retomar exige o ledger: este orquestrador foi montado com "
                "save_path=None."
            )
        point = plan_resume(load_loop_records(self.save_path), run_id, rounds=rounds)
        if point.action == "done":
            return ()
        if point.detector is not None and point.detector != self.detector:
            raise CampaignResumeError(
                f"A campanha de {run_id!r} treinou {point.detector!r}; retomar com "
                f"{self.detector!r} misturaria dois experimentos no mesmo ledger."
            )

        # Só a nova tentativa precisa de nota própria: uma rodada que `continue`
        # roda é uma rodada comum, e a nota de sempre ("herdada da rodada
        # anterior pela política") já é a verdade sobre ela.
        note = (
            f"Intenção reaproveitada de {point.intent_source} para refazer a "
            f"rodada {point.round_index} (retomada): nenhuma chamada de LLM "
            "nesta rodada."
            if point.action == "retry" and point.intent is not None
            else None
        )

        return self._chain(
            point.tip.source_prompt,
            seed=point.tip.seed,
            first_round=point.round_index,
            max_rounds=point.max_rounds,
            next_intent=point.intent,
            intent_note=note,
            parent_run_id=point.parent_run_id,
            history=point.history,
            retry_of=point.retry_of,
        )

    def _chain(
        self,
        prompt: str,
        *,
        seed: int,
        first_round: int,
        max_rounds: int,
        next_intent: IntentSpec | None = None,
        intent_note: str | None = None,
        parent_run_id: str | None = None,
        history: tuple[RoundOutcome, ...] = (),
        retry_of: str | None = None,
    ) -> tuple[LoopRecord, ...]:
        """O laço de rodadas, do ponto em que a campanha está até o teto.

        ``intent_note`` e ``retry_of`` valem só para a primeira rodada do laço —
        a que a retomada refaz; as seguintes são rodadas comuns da campanha.
        """

        records: list[LoopRecord] = []

        for round_index in range(first_round, max_rounds + 1):
            record, decision, _intent = self._run_round(
                prompt,
                seed=seed,
                intent_override=next_intent,
                intent_note=intent_note,
                round_index=round_index,
                parent_run_id=parent_run_id,
                max_rounds=max_rounds,
                history=history,
                retry_of=retry_of,
            )
            records.append(record)
            intent_note = None
            retry_of = None

            if decision is None or not decision.should_continue:
                break

            if self._budget_exhausted(records):
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
        intent_note: str | None = None,
        round_index: int = 1,
        parent_run_id: str | None = None,
        max_rounds: int = 1,
        history: tuple[RoundOutcome, ...] = (),
        retry_of: str | None = None,
        seed_override: int | None = None,
        replicate_batch_id: str | None = None,
    ) -> tuple[LoopRecord, FeedbackDecision | None, IntentSpec | None]:
        """Roda os sete estágios de uma rodada e devolve registro + decisão + intenção.

        A decisão volta junto (em vez de ser relida do ``feedback.json``) para
        que ``run_campaign`` não dependa de I/O para saber se continua. É
        ``None`` quando algum estágio falhou antes do FEEDBACK. A intenção
        resolvida volta também para que ``run_replicates`` reaproveite a mesma
        ``IntentSpec`` entre réplicas sem uma segunda chamada de LLM; é ``None``
        quando o próprio estágio INTENT falhou.

        ``seed_override`` (Fase 2.R) troca a seed usada na geração e gravada no
        registro, sem alterar a ``IntentSpec`` — "mesma intenção, seed de geração
        diferente". Precede ``intent.seed``.
        """

        run_id = new_run_id()
        run_dir = self.output_dir / run_id
        # Disco sempre, observador ao vivo quando houver: uma execução pela CLI
        # deixa a timeline em `events.jsonl` sem ninguém pedir, e a UI só
        # acrescenta um segundo destino. Nenhum dos dois lados sabe do outro.
        ctx = _RoundContext(
            run_id=run_id,
            run_dir=run_dir,
            round_index=round_index,
            emit=fanout(JsonlEventSink(run_dir / "events.jsonl"), self.event_sink),
        )
        stages = ctx.stages
        ctx.event("run_started", message=prompt)
        started = time.perf_counter()
        record_seed = seed
        resolved: dict[str, Any] = {}
        decision: FeedbackDecision | None = None
        intent: IntentSpec | None = None

        try:
            if intent_override is None:
                intent = self._stage(
                    ctx, "intent",
                    lambda: self.intent_agent.interpret(prompt),
                    persist_as="intent.json",
                )
            else:
                intent = self._inherited_intent_stage(
                    ctx,
                    intent_override,
                    note=intent_note
                    or (
                        f"Intenção herdada da rodada {round_index - 1} pela "
                        "política de feedback (E10): nenhuma chamada de LLM "
                        "nesta rodada."
                    ),
                )
            # Fase 2.R: a seed de geração pode ser trocada por réplica sem mexer
            # na IntentSpec. Precede intent.seed e é o que vai para o registro.
            effective_seed = seed_override if seed_override is not None else intent.seed
            record_seed = effective_seed

            def _compile_candidate() -> Any:
                # Resolve o AttackSpec aqui (não antes): é o que amarra o
                # GENERATOR/ERENO/DETECTOR ao ataque-base que a intenção
                # compilada de fato usa, não a um parâmetro externo separado.
                resolved["spec"] = get_attack_spec(intent.base_attack)
                return compile_attack_candidate(intent)

            candidate = self._stage(
                ctx, "generator",
                _compile_candidate,
                persist_as="attack_candidate.json",
            )
            spec = resolved["spec"]
            # A seed chega ao JAR aqui (Fase 2.0): baseline e variante da mesma
            # rodada compartilham a seed, isolando o efeito da config da variação
            # de RNG. Em réplicas (2.R) é a seed varrida do lote. Só o modo jar a usa.
            generator = self._build_generator(run_dir, spec, random_seed=effective_seed)

            trace_path = self._stage(
                ctx, "ereno",
                lambda: generator.generate_dataset(candidate.config, iteration=1),
            )
            # Quarto manifest da rodada, ao lado de feature/selection/detector —
            # e o único a montante do detector: de onde este trace veio. Sem
            # ele, "qual gerador produziu estes números" só existe na linha de
            # comando que alguém digitou, e some junto com o terminal. O nome do
            # estágio continua "ereno" por compatibilidade com os LoopRecords já
            # gravados; quem diz o simulador de verdade é o manifest.
            save_json(
                run_dir / "generator_manifest.json",
                manifest_for(
                    self.generator_mode,
                    attack_key=spec.key,
                    segment_name=spec.segment_name,
                    random_seed=effective_seed,
                    duration_seconds=ctx.stages[-1].duration_seconds,
                ).model_dump(mode="json"),
            )

            dataset_bundle = self._stage(
                ctx, "preprocess",
                lambda: build_dataset_bundle(
                    trace_path,
                    lineage_run_id=run_id,
                    expected_attack_label=spec.label,
                    min_attack_rows=self.min_attack_rows,
                    min_normal_rows=self.min_normal_rows,
                    min_attack_prevalence=self.min_attack_prevalence,
                ),
                persist_as="dataset_bundle.json",
            )

            detection_report = self._stage(
                ctx, "detector",
                lambda: self._run_detector(generator, spec, dataset_bundle, run_dir),
                persist_as="detection_report.json",
            )

            report_ref = str(run_dir / "detection_report.json")
            defense_plan = self._stage(
                ctx, "defender",
                lambda: self._run_defender(
                    detection_report,
                    report_ref=report_ref,
                    attack_key=intent.base_attack,
                    run_dir=run_dir,
                ),
                persist_as="defense_plan.json",
            )

            decision = self._stage(
                ctx, "feedback",
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

        # O consumo do LLM de intent conta só quando o LLM foi de fato chamado
        # nesta rodada (intent_override is None). Sem isso, uma rodada que reusa a
        # intenção — rodada 2+ de campanha ou réplica 2+ de um lote — somaria de
        # novo os tokens da última chamada (``last_usage`` fica setado), inflando
        # o custo. É o que torna verdadeira a conta "1 chamada de LLM para N
        # réplicas" (2.R).
        usage = self._round_usage(include_intent=intent_override is None)
        record = LoopRecord(
            run_id=run_id,
            source_prompt=prompt,
            seed=record_seed,
            stages=tuple(stages),
            cost_usd=usage.cost_usd if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            total_duration_seconds=time.perf_counter() - started,
            round=round_index,
            parent_run_id=parent_run_id,
            max_rounds=max_rounds,
            retry_of=retry_of,
            replicate_batch_id=replicate_batch_id,
        )

        if self.save_path is not None:
            append_loop_record(self.save_path, record)

        # Fecha a timeline. O estágio que falhou já se anunciou com a causa; o
        # evento terminal diz que não vem mais nada — sem ele, quem observa não
        # distingue uma execução que parou de uma que ainda está pensando.
        failed = next(
            (
                stage
                for stage in record.stages
                if stage.status is LoopStageStatus.FAILED
            ),
            None,
        )
        ctx.event(
            "run_finished",
            status=(
                LoopStageStatus.FAILED if failed else LoopStageStatus.SUCCEEDED
            ),
            duration_seconds=record.total_duration_seconds,
            artifact_ref=str(self.save_path) if self.save_path else None,
            message=(
                f"Interrompido no estágio {failed.name}: {failed.error}"
                if failed
                else None
            ),
        )

        return record, decision, intent

    # ------------------------------------------------------------------ #
    # Estágio INTENT herdado (rodadas 2+) — nenhuma chamada de LLM        #
    # ------------------------------------------------------------------ #
    def _inherited_intent_stage(
        self,
        ctx: _RoundContext,
        intent: IntentSpec,
        *,
        note: str,
    ) -> IntentSpec:
        """Persiste uma intenção que já existia, sem chamar o LLM.

        Vem da política (E10), da rodada 2 em diante, ou da rodada que falhou e
        está sendo refeita pela retomada; ``note`` diz qual das duas.

        Registrado como ``skipped``, nunca ``succeeded``: o status precisa
        continuar dizendo a verdade sobre o que rodou nesta rodada — o
        ``IntentLike`` real não foi chamado, então não há duração de LLM a
        medir nem uma proposta a validar de novo (já passou pelos dois
        portões quando foi produzida).
        """

        artifact_path = ctx.run_dir / "intent.json"
        save_json(artifact_path, intent.model_dump(mode="json"))
        ctx.record_stage(
            LoopStage(
                name="intent",
                status=LoopStageStatus.SKIPPED,
                artifact_ref=str(artifact_path),
                error=note,
            )
        )
        return intent

    # ------------------------------------------------------------------ #
    # Estágio DETECTOR — treina no baseline, avalia o candidato compilado #
    # ------------------------------------------------------------------ #
    def _run_detector(
        self,
        generator: GeneratorLike,
        spec: AttackSpec,
        dataset_bundle: DatasetBundle,
        run_dir: Path,
    ) -> DetectionReport:
        baseline_config = load_json(spec.baseline_path)
        baseline_dataset_path = generator.generate_dataset(baseline_config, iteration=0)

        evaluator = IdsEvaluator(
            drop_cb_status=False,
            target_attack_label=spec.label,
            detector=self.detector,
        )
        evaluator.train_baseline(baseline_dataset_path)

        # Manifest do preprocessador (E6), ajustado no baseline — não no
        # ``dataset_bundle.trace_path`` (a variante é só transformada, nunca
        # ajustada). Persistido antes de ``build_detection_report`` de
        # propósito: se a avaliação falhar o gate binário logo abaixo, o
        # manifest já está em disco para diagnosticar o que foi ajustado.
        manifest = evaluator.feature_manifest
        if manifest is not None:
            save_json(run_dir / "feature_manifest.json", manifest.model_dump(mode="json"))

        # Manifest de seleção de features + undersampling (E7) — mesmo motivo
        # de persistir antes do gate: se a avaliação falhar logo abaixo, já
        # está em disco o que a seleção/undersampling decidiu sobre o treino.
        selection_manifest = evaluator.selection_manifest
        if selection_manifest is not None:
            save_json(
                run_dir / "selection_manifest.json", selection_manifest.model_dump(mode="json")
            )

        # Manifest do detector (E8) — terceiro artefato do mesmo estágio, pelo
        # mesmo motivo de precedência: se o gate binário abaixo derrubar a
        # avaliação, já está em disco qual detector treinou e sob que escala.
        detector_manifest = evaluator.detector_manifest
        if detector_manifest is not None:
            save_json(
                run_dir / "detector_manifest.json", detector_manifest.model_dump(mode="json")
            )

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
    # Contabilidade de consumo (E11)                                      #
    # ------------------------------------------------------------------ #
    def _budget_exhausted(self, records: list[LoopRecord]) -> bool:
        """A campanha já gastou o teto de tokens configurado?

        Rodadas que não informaram consumo contam como zero *nesta soma*, e não
        como desconhecido: elas não somam ao gasto conhecido, mas também não
        impedem que o gasto conhecido estoure o teto.
        """

        if self.token_budget <= 0:
            return False

        spent = sum(record.total_tokens or 0 for record in records)

        return spent >= self.token_budget

    @staticmethod
    def _usage_of(agent: Any) -> AgentUsage | None:
        """O consumo da última chamada daquele agente, se ele souber informar.

        Lido por ``getattr`` e não por um método do protocolo: ``IntentLike`` e
        ``DefenderLike`` existem para que stubs determinísticos rodem o
        orquestrador inteiro, e exigir contabilidade deles obrigaria todo stub a
        fingir um número que não tem.
        """

        usage = getattr(agent, "last_usage", None)

        return usage if isinstance(usage, AgentUsage) else None

    def _round_usage(self, *, include_intent: bool = True) -> AgentUsage | None:
        """Soma o que os agentes gastaram nesta rodada.

        ``None`` quando nenhum informou: zero token é uma afirmação sobre as
        chamadas, ``None`` é a confissão de que não se sabe — e um orçamento
        comparado contra um zero inventado aprovaria qualquer coisa.

        ``include_intent=False`` numa rodada que reusou a intenção (não chamou o
        LLM de intent): ``intent_agent.last_usage`` ainda guarda o consumo da
        última chamada real, e somá-lo aqui contaria o mesmo gasto de novo.
        """

        candidates = [self._usage_of(self.defender_agent)]
        if include_intent:
            candidates.insert(0, self._usage_of(self.intent_agent))
        reported = [usage for usage in candidates if usage is not None]
        if not reported:
            return None

        total = reported[0]
        for usage in reported[1:]:
            total = total + usage

        return total

    # ------------------------------------------------------------------ #
    # Estágio DEFENDER — plano validado + o veredito das regras (E5)      #
    # ------------------------------------------------------------------ #
    def _run_defender(
        self,
        detection_report: DetectionReport,
        *,
        report_ref: str,
        attack_key: str,
        run_dir: Path,
    ) -> DefensePlan:
        """Produz o plano e persiste ao lado dele a avaliação por regras.

        ``defend`` já recusa o plano cuja recomendação não responde à evidência
        citada — a avaliação aqui é a *mesma*, recalculada sobre um plano que já
        passou, para deixar em disco o eixo que o portão não usa: o playbook
        IEC-61850 daquela família, que aconselha sem bloquear. Reavaliar custa
        nada (é função pura sobre dois modelos) e evita que ``defend`` devolva
        duas coisas só para carregar o relatório de volta.

        ``defense_rules.json`` é o artefato que sustenta o gate da janela
        D47-54: ``grounded_actions``/``total_actions`` é, literalmente, a fração
        de recomendações ligadas a evidência.
        """

        plan = self.defender_agent.defend(
            detection_report, report_ref=report_ref, attack_key=attack_key
        )

        rules = evaluate_plan_rules(
            plan,
            detection_report,
            attack_key=attack_key,
            detection_report_ref=report_ref,
        )
        save_json(run_dir / "defense_rules.json", rules.model_dump(mode="json"))

        return plan

    # ------------------------------------------------------------------ #
    # Núcleo (um gerador por execução, isolado por run_id)                #
    # ------------------------------------------------------------------ #
    def _build_generator(
        self, run_dir: Path, spec: AttackSpec, *, random_seed: int | None = None
    ) -> GeneratorLike:
        """Resolve o backend de geração pela chave e o constrói.

        ``options`` reúne o que é específico do backend ERENO (runtime, comando,
        timeouts); o que é genérico — qual ataque, qual segmento, qual semente —
        são campos do ``GeneratorRequest``. Um backend que não seja o ERENO lê
        outras chaves de ``options`` e entra sem passar por aqui.
        """

        request = GeneratorRequest(
            attack_key=spec.key,
            segment_name=spec.segment_name,
            random_seed=random_seed,
            options={
                "runtime_dir": GENERATOR_RUNTIME_DIR,
                "output_dataset_path": GENERATOR_OUTPUT_DATASET_PATH,
                "run_command": GENERATOR_RUN_COMMAND,
                "suggested_config_path": str(run_dir / "attack_config.json"),
                "action_config_relative_path": GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
                "benign_action_config_relative_path": (
                    GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH
                ),
                "benign_seed_path": BASELINE_DATASET_PATH,
                "cached_dataset_path": self.cached_dataset_path or BASELINE_DATASET_PATH,
                "timeout_seconds": GENERATOR_TIMEOUT_SECONDS,
                "max_retries": GENERATOR_MAX_RETRIES,
                "retry_backoff_seconds": GENERATOR_RETRY_BACKOFF_SECONDS,
            },
        )
        return build_generator(self.generator_mode, request)

    # ------------------------------------------------------------------ #
    # Runner de estágio — mede duração, persiste artefato, registra falha #
    # ------------------------------------------------------------------ #
    def _stage(
        self,
        ctx: _RoundContext,
        name: str,
        fn: Any,
        *,
        persist_as: str | None = None,
    ) -> Any:
        ctx.event("stage_started", stage=name)
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            # A exceção de um cliente HTTP costuma trazer a requisição que
            # falhou, cabeçalho de autorização incluído — e este texto vai para
            # o disco e para a tela.
            ctx.record_stage(
                LoopStage(
                    name=name,
                    status=LoopStageStatus.FAILED,
                    duration_seconds=time.perf_counter() - start,
                    error=redact(str(exc)),
                )
            )
            raise

        duration = time.perf_counter() - start
        artifact_ref: str | None
        if persist_as is not None:
            artifact_path = ctx.run_dir / persist_as
            save_json(artifact_path, result.model_dump(mode="json"))
            artifact_ref = str(artifact_path)
        elif isinstance(result, str):
            artifact_ref = result
        else:
            artifact_ref = None

        ctx.record_stage(
            LoopStage(
                name=name,
                status=LoopStageStatus.SUCCEEDED,
                duration_seconds=duration,
                artifact_ref=artifact_ref,
            )
        )
        return result
