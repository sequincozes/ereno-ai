"""Piloto E0 — valida as famílias de ataque antes de gastar orçamento na campanha.

Execute com ``uv run python scripts/run_pilot.py``. É o experimento E0 do artigo
("Piloto e validade das variantes"): gerar baseline e variantes de cada família,
confirmar que mexer nos parâmetros **muda o tráfego** e que o ataque **continua
presente**, e medir tempo/tokens para fixar o orçamento dos experimentos
seguintes. Uma família que não passa aqui tem a causa registrada no relatório —
reduzir escopo sem esse registro é perder a razão pela qual o escopo encolheu.

Três medidas, nessa ordem, porque cada uma cobre o que a anterior deixa passar:

1. **Portão de volume (E4).** ``build_dataset_bundle`` sobre cada trace: classes
   presentes, contagens e prevalência da classe de ataque. Pega trace vazio,
   quebrado ou raro demais para medir. Não diz nada sobre o ataque estar ativo.
2. **Assinatura.** Separação (diferença padronizada de médias) entre as linhas de
   ataque e as linhas normais do *mesmo* trace, sobre as features de protocolo.
   Um trace pode ter 50% de rótulo de ataque e linhas indistinguíveis do tráfego
   normal — o rótulo vem do gerador, não da física. Isto é evidência estatística
   de que a classe é distinguível, **não** prova de semântica preservada: para
   isso o relatório também grava amostras de mensagens (``--sample-rows``) para
   inspeção manual, que é o que o roteiro do E0 pede.
3. **Deriva contra o piso de ruído, feature a feature.** O gerador ERENO **não é
   determinístico**: a mesma config rodada duas vezes produz bytes diferentes.
   Medido em 17/09/2026 no ``random_replay`` — dois hashes distintos para o mesmo
   JSON. Logo "o hash mudou" não é evidência de nada. O piloto gera a baseline
   ``--replicates`` vezes e mede, **em cada coluna**, a maior separação entre
   réplicas: o ruído daquela coluna. Uma variante é distinta da baseline quando
   pelo menos ``--min-moved-features`` colunas derivam acima de ``--min-drift``
   *e* de ``--drift-margin`` vezes o ruído da própria coluna. Comparar contra um
   piso único (o máximo global) foi a primeira versão e fazia o veredito da
   família oscilar entre execuções idênticas — ver o comentário em ``_score``.

As variantes saem do compilador determinístico (``core/intent_compiler.py``)
aplicado a intensidades crescentes — a mesma mecânica que a política de feedback
usa da rodada 2 em diante. Nenhuma LLM participa da geração: o piloto mede o
gerador e os dados, não o agente. O custo em token entra por ``--llm-probe``,
que faz **uma** chamada real do ``IntentAgent`` por família só para registrar o
consumo do estágio ``intent``; ele espaça as chamadas em ``--llm-probe-spacing``
segundos porque o prompt com o catálogo de capacidades pesa ~8,2k tokens e o
tier ``on_demand`` da Groq tem TPM 8000 — duas chamadas seguidas viram
``rate_limit_exceeded``.

O relatório (``outputs/pilot/pilot_report.json``) é artefato de experimento, não
contrato de pipeline: nada no ``src/`` o lê, então ele fica em JSON simples com
``schema_version``, e não como modelo em ``domain/``.

Exemplos:

    # Piloto completo das três famílias do E0 (precisa do JAR; ~1 min)
    uv run python scripts/run_pilot.py

    # Uma família só, com mais réplicas para firmar o piso de ruído
    uv run python scripts/run_pilot.py --attack grayhole --replicates 3

    # Incluindo a medição de token do estágio intent (precisa de GROQ_API_KEY)
    uv run python scripts/run_pilot.py --llm-probe
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from adversarial_ids.config.attacks_registry import (  # noqa: E402
    AttackSpec,
    get_attack_spec,
    list_attack_keys,
)
from adversarial_ids.config.settings import (  # noqa: E402
    BASELINE_DATASET_PATH,
    GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
    GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH,
    GENERATOR_MAX_RETRIES,
    GENERATOR_MODE,
    GENERATOR_OUTPUT_DATASET_PATH,
    GENERATOR_RETRY_BACKOFF_SECONDS,
    GENERATOR_RUN_COMMAND,
    GENERATOR_RUNTIME_DIR,
    GENERATOR_TIMEOUT_SECONDS,
    INTENT_LOOP_MIN_ATTACK_PREVALENCE,
    INTENT_LOOP_MIN_ATTACK_ROWS,
    INTENT_LOOP_MIN_NORMAL_ROWS,
    MODEL_ID,
    OUTPUTS_DIR,
    PROTOCOL_FEATURES_MODE,
)
from adversarial_ids.core.dataset_bundle_builder import (  # noqa: E402
    DatasetGateError,
    build_dataset_bundle,
)
from adversarial_ids.core.generator_runner import GeneratorRunner  # noqa: E402
from adversarial_ids.core.intent_compiler import compile_attack_candidate  # noqa: E402
from adversarial_ids.core.preprocessor import always_drop_for  # noqa: E402
from adversarial_ids.domain.intent_spec import (  # noqa: E402
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
    IntentSpec,
)
from adversarial_ids.shared.json_io import load_json, save_json  # noqa: E402

# As três famílias do E0: uma de forja de conteúdo, uma de replay e uma de
# descarte. Cobrem os três mecanismos de ataque que o artigo discute sem exigir
# o cruzamento de todas as onze registradas.
DEFAULT_ATTACKS: tuple[str, ...] = ("masquerade_fault", "random_replay", "grayhole")
DEFAULT_INTENSITIES: tuple[str, ...] = ("low", "medium", "high")
PILOT_DIR = OUTPUTS_DIR / "pilot"
DEFAULT_REPORT_PATH = PILOT_DIR / "pilot_report.json"
NORMAL_LABEL = "normal"


# --------------------------------------------------------------------------- #
# Separação entre dois conjuntos de linhas                                    #
# --------------------------------------------------------------------------- #
def _numeric_features(frame: pd.DataFrame, protocol_features: str) -> pd.DataFrame:
    """Colunas numéricas de feature, sem identidade nem rótulo.

    Usa a lista de descarte do preprocessador para não medir separação sobre
    relógio absoluto e endereço MAC — colunas que separam as classes pelo bloco
    de geração, não pelo ataque.
    """

    dropped = set(always_drop_for(protocol_features)) | {"class", "label"}
    keep = [column for column in frame.columns if column not in dropped]

    return frame[keep].select_dtypes(include="number")


def _separation(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    """Diferença padronizada de médias, coluna a coluna, entre dois recortes.

    Devolve ``{"by_feature": {...}, "max": float, "top": [...], "constant":
    [...]}``. ``constant`` lista as colunas em que os dois lados são constantes
    em valores diferentes: a separação ali é total, e dividir por um desvio zero
    daria infinito — que não cabe em JSON nem em comparação com limiar. Elas
    contam como evidência, mas por presença, não por magnitude.
    """

    shared = [column for column in left.columns if column in right.columns]
    by_feature: dict[str, float] = {}
    constant: list[str] = []

    for column in shared:
        a = left[column].dropna()
        b = right[column].dropna()
        if a.empty or b.empty:
            continue

        delta = abs(float(a.mean()) - float(b.mean()))
        if len(a) < 2 or len(b) < 2:
            continue

        pooled = math.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2)
        if not math.isfinite(pooled) or not math.isfinite(delta):
            continue
        if pooled == 0.0:
            if delta > 0.0:
                constant.append(column)
            continue

        by_feature[column] = delta / pooled

    ranked = sorted(by_feature.items(), key=lambda item: item[1], reverse=True)

    return {
        "by_feature": {name: round(value, 6) for name, value in ranked},
        "max": round(ranked[0][1], 6) if ranked else 0.0,
        "top": [{"feature": name, "separation": round(value, 6)} for name, value in ranked[:5]],
        "constant_shift": constant,
    }


def _has_signal(separation: dict[str, Any], threshold: float) -> bool:
    return bool(separation["constant_shift"]) or separation["max"] >= threshold


def _count_above(separation: dict[str, Any], threshold: float) -> int:
    """Quantas features passam do limiar — a largura do efeito, não só o pico."""

    above = sum(1 for value in separation["by_feature"].values() if value >= threshold)

    return above + len(separation["constant_shift"])


def _moved_features(
    drift: dict[str, Any],
    noise_by_feature: dict[str, float],
    noise_constant: set[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Colunas que a variante moveu de verdade, uma comparação por coluna.

    Dois critérios, e a coluna precisa dos dois. ``--min-drift`` é o piso
    absoluto: sem ele, uma coluna estabilíssima entre réplicas (ruído ~0)
    passaria por qualquer deriva, inclusive uma que não significa nada.
    ``--drift-margin`` é o piso relativo: a deriva tem de superar o ruído
    *daquela* coluna por essa folga.
    """

    moved: list[dict[str, Any]] = []

    for feature in drift["constant_shift"]:
        if feature in noise_constant:
            continue
        moved.append({"feature": feature, "drift": None, "noise": None, "kind": "constant_shift"})

    for feature, value in drift["by_feature"].items():
        if feature in noise_constant:
            continue
        noise = noise_by_feature.get(feature, 0.0)
        if value >= args.min_drift and value > noise * args.drift_margin:
            moved.append(
                {
                    "feature": feature,
                    "drift": round(value, 6),
                    "noise": round(noise, 6),
                    "kind": "shift",
                }
            )

    return sorted(moved, key=lambda item: item["drift"] or float("inf"), reverse=True)


# --------------------------------------------------------------------------- #
# Geração                                                                     #
# --------------------------------------------------------------------------- #
def _build_generator(spec: AttackSpec, run_dir: Path, mode: str) -> GeneratorRunner:
    """Mesma construção que o loop por intenção usa, por família e por diretório."""

    return GeneratorRunner(
        runtime_dir=GENERATOR_RUNTIME_DIR,
        output_dataset_path=GENERATOR_OUTPUT_DATASET_PATH,
        run_command=GENERATOR_RUN_COMMAND,
        suggested_config_path=str(run_dir / "attack_config.json"),
        action_config_relative_path=GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
        benign_action_config_relative_path=GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH,
        benign_seed_path=BASELINE_DATASET_PATH,
        segment_name=spec.segment_name,
        cached_dataset_path=BASELINE_DATASET_PATH if mode == "cached" else None,
        timeout_seconds=GENERATOR_TIMEOUT_SECONDS,
        max_retries=GENERATOR_MAX_RETRIES,
        retry_backoff_seconds=GENERATOR_RETRY_BACKOFF_SECONDS,
    )


def _variant_intent(attack_key: str, effect: str, intensity: str, args: argparse.Namespace) -> IntentSpec:
    return IntentSpec(
        source_prompt=(
            f"Piloto E0: variante {intensity} de {attack_key} para o efeito {effect}."
        ),
        objective=IntentObjective.ASSESS_IDS_ROBUSTNESS,
        base_attack=attack_key,
        desired_effect=DesiredEffect(effect),
        restrictions=IntentRestrictions(max_fields_changed=args.max_fields),
        intensity=IntentIntensity(intensity),
        seed=args.seed,
    )


def _generate(
    runner: GeneratorRunner,
    config: dict[str, Any],
    *,
    index: int,
    run_dir: Path,
    tag: str,
) -> tuple[Path | None, float, str | None]:
    """Gera um trace e o move para um nome estável dentro do diretório da família.

    Uma geração que falha **não** derruba o piloto: devolve ``(None, segundos,
    erro)`` e o chamador registra a causa. O gerador recusa configurações que
    todos os portões do repo aprovam — um intervalo ``{min, max}`` degenerado
    (min == max) passa pelo clamp do compilador e pelo schema do ataque, e o
    ERENO o rejeita com "The lower limit (X) must be less than the upper limit
    (X)". Descobrir isso é justamente o serviço do piloto; abortar no meio
    perderia as famílias ainda não medidas.
    """

    save_json(run_dir / f"{tag}_config.json", config)

    start = time.perf_counter()
    try:
        produced = Path(runner.generate_dataset(config, index))
    except Exception as exc:  # noqa: BLE001 — a causa vira linha do relatório
        return None, time.perf_counter() - start, str(exc)
    seconds = time.perf_counter() - start

    destination = run_dir / f"{tag}.csv"
    if produced != destination:
        shutil.move(str(produced), destination)

    return destination, seconds, None


def _inspect(
    trace_path: Path,
    spec: AttackSpec,
    args: argparse.Namespace,
    run_dir: Path,
    tag: str,
) -> dict[str, Any]:
    """Portão de volume + assinatura + amostra de mensagens de um trace."""

    result: dict[str, Any] = {"tag": tag, "trace": str(trace_path)}

    try:
        bundle = build_dataset_bundle(
            trace_path,
            lineage_run_id=f"pilot_{spec.key}_{tag}",
            expected_attack_label=spec.label,
            min_attack_rows=INTENT_LOOP_MIN_ATTACK_ROWS,
            min_normal_rows=INTENT_LOOP_MIN_NORMAL_ROWS,
            min_attack_prevalence=args.min_attack_prevalence,
        )
    except DatasetGateError as exc:
        result["gate"] = {"status": "failed", "error": str(exc)}
        return result

    labeled = sum(bundle.class_counts.values())
    attack_rows = bundle.class_counts.get(spec.label, 0)
    result["gate"] = {
        "status": "passed",
        "content_hash": bundle.content_hash,
        "class_counts": dict(bundle.class_counts),
        "attack_rows": attack_rows,
        "prevalence": round(attack_rows / labeled, 6) if labeled else 0.0,
        "columns": len(bundle.columns),
    }

    frame = pd.read_csv(trace_path)
    frame.columns = [column.strip() for column in frame.columns]
    label_column = "class" if "class" in frame.columns else "label"
    labels = frame[label_column].astype(str).str.strip()

    attack_frame = frame[labels == spec.label]
    normal_frame = frame[labels == NORMAL_LABEL]

    features_attack = _numeric_features(attack_frame, args.protocol_features)
    features_normal = _numeric_features(normal_frame, args.protocol_features)

    signature = _separation(features_attack, features_normal)
    signature["has_signal"] = _has_signal(signature, args.min_signature)
    # Quantas features carregam a assinatura, não só a maior: uma separação
    # enorme numa coluna só e uma separação moderada em oito são evidências
    # diferentes sobre o mesmo ataque.
    signature["features_with_signal"] = _count_above(signature, args.min_signature)
    result["signature"] = signature

    if args.sample_rows > 0:
        sample_path = run_dir / f"{tag}_sample.csv"
        attack_frame.head(args.sample_rows).to_csv(sample_path, index=False)
        result["sample"] = str(sample_path)

    return result


def _attack_rows(trace_path: Path, spec: AttackSpec, protocol_features: str) -> pd.DataFrame:
    frame = pd.read_csv(trace_path)
    frame.columns = [column.strip() for column in frame.columns]
    label_column = "class" if "class" in frame.columns else "label"
    labels = frame[label_column].astype(str).str.strip()

    return _numeric_features(frame[labels == spec.label], protocol_features)


# --------------------------------------------------------------------------- #
# Piloto de uma família                                                       #
# --------------------------------------------------------------------------- #
def run_attack(attack_key: str, args: argparse.Namespace) -> dict[str, Any]:
    spec = get_attack_spec(attack_key)
    # O rótulo separa execuções da mesma família: comparar "3 campos" com "11
    # campos" exige que a segunda não apague os traces da primeira, e o gerador
    # não é determinístico — o que foi sobrescrito não volta igual.
    run_dir = PILOT_DIR / (f"{attack_key}__{args.label}" if args.label else attack_key)
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = _build_generator(spec, run_dir, args.generator_mode)

    report: dict[str, Any] = {
        "attack": attack_key,
        "label": spec.label,
        "segment_name": spec.segment_name,
        "baseline_config": str(spec.baseline_path),
        "replicates": [],
        "variants": [],
        "reasons": [],
    }

    baseline_config = load_json(spec.baseline_path)
    index = 0
    seconds_total = 0.0

    print(f"\n[{attack_key}] baseline x{args.replicates}")
    for replicate in range(1, args.replicates + 1):
        tag = f"baseline_{replicate}"
        trace, seconds, error = _generate(
            runner, baseline_config, index=index, run_dir=run_dir, tag=tag
        )
        index += 1
        seconds_total += seconds
        if trace is None:
            report["replicates"].append(
                {"tag": tag, "seconds": round(seconds, 3), "generation": {"status": "failed", "error": error}}
            )
            print(f"  {tag}: geração falhou — {error.splitlines()[0] if error else ''}")
            continue
        entry = _inspect(trace, spec, args, run_dir, tag)
        entry["generation"] = {"status": "passed"}
        entry["seconds"] = round(seconds, 3)
        report["replicates"].append(entry)
        print(f"  {tag}: {seconds:.1f}s — {entry['gate']['status']}")

    print(f"[{attack_key}] variantes: {', '.join(args.intensity)}")
    for intensity in args.intensity:
        tag = f"variant_{intensity}"
        entry: dict[str, Any] = {"tag": tag, "intensity": intensity}
        try:
            candidate = compile_attack_candidate(
                _variant_intent(attack_key, args.effect, intensity, args)
            )
        except ValueError as exc:
            # Efeito não suportado pela capability, ou nenhum campo se moveu.
            # É resultado do piloto, não acidente: a família fica sem aquela
            # variante e a causa vai para o relatório.
            entry["compile"] = {"status": "failed", "error": str(exc)}
            report["variants"].append(entry)
            print(f"  {tag}: compilação recusada — {exc}")
            continue

        entry["compile"] = {
            "status": "passed",
            "capability_id": candidate.capability_id,
            "diff": [change.model_dump(mode="json") for change in candidate.diff],
        }
        trace, seconds, error = _generate(
            runner, candidate.config, index=index, run_dir=run_dir, tag=tag
        )
        index += 1
        seconds_total += seconds
        if trace is None:
            entry["generation"] = {"status": "failed", "error": error}
            entry["seconds"] = round(seconds, 3)
            report["variants"].append(entry)
            print(f"  {tag}: geração falhou — {error.splitlines()[0] if error else ''}")
            continue
        entry.update(_inspect(trace, spec, args, run_dir, tag))
        entry["tag"] = tag
        entry["generation"] = {"status": "passed"}
        entry["seconds"] = round(seconds, 3)
        report["variants"].append(entry)
        print(f"  {tag}: {seconds:.1f}s — {entry['gate']['status']}")

    report["seconds_total"] = round(seconds_total, 3)
    _score(report, spec, args, run_dir)

    return report


def _score(report: dict[str, Any], spec: AttackSpec, args: argparse.Namespace, run_dir: Path) -> None:
    """Piso de ruído, deriva de cada variante e veredito da família."""

    passed_replicates = [
        item for item in report["replicates"] if item.get("gate", {}).get("status") == "passed"
    ]
    reasons: list[str] = report["reasons"]

    for item in report["replicates"]:
        if item.get("generation", {}).get("status") == "failed":
            reasons.append(f"{item['tag']}: o gerador recusou a baseline — {item['generation']['error']}")
        elif item.get("gate", {}).get("status") == "failed":
            reasons.append(f"{item['tag']}: reprovada no portão de volume — {item['gate']['error']}")

    # Piso de ruído **por feature**: a maior separação que cada coluna exibe
    # entre réplicas da MESMA configuração. Tudo abaixo disso, naquela coluna, é
    # o gerador e não o parâmetro.
    #
    # Um piso único (o máximo global) foi a primeira versão disto e não
    # funcionou: o máximo sobre ~40 colunas é o máximo de estimativas ruidosas,
    # varia de execução para execução e faz o veredito da família oscilar com o
    # sorteio do gerador. Medido em 18/09/2026 — o grayhole saiu 3/3 numa
    # execução e 1/3 na seguinte, sem nada ter mudado. Por coluna, cada
    # comparação é entre as mesmas duas distribuições e o veredito para de
    # depender de qual coluna teve azar.
    frames = {
        item["tag"]: _attack_rows(Path(item["trace"]), spec, args.protocol_features)
        for item in passed_replicates
    }
    noise_by_feature: dict[str, float] = {}
    noise_constant: set[str] = set()
    noise_detail: list[dict[str, Any]] = []

    for left, right in combinations(sorted(frames), 2):
        pair = _separation(frames[left], frames[right])
        noise_detail.append({"pair": [left, right], "max": pair["max"], "top": pair["top"]})
        for feature, value in pair["by_feature"].items():
            noise_by_feature[feature] = max(noise_by_feature.get(feature, 0.0), value)
        # Coluna constante em valores diferentes entre duas réplicas: o próprio
        # gerador a desloca por inteiro, então ela não serve de evidência.
        noise_constant.update(pair["constant_shift"])

    if len(passed_replicates) < 2:
        reasons.append(
            "Menos de duas réplicas válidas: sem piso de ruído, a deriva de uma "
            "variante não é distinguível da variação do gerador."
        )

    noise_max = max(noise_by_feature.values(), default=0.0)
    report["noise_floor"] = {
        "max_separation": round(noise_max, 6),
        "by_feature": {
            feature: round(value, 6)
            for feature, value in sorted(
                noise_by_feature.items(), key=lambda item: item[1], reverse=True
            )
        },
        "unusable_features": sorted(noise_constant),
        "pairs": noise_detail,
        "note": (
            "Separação entre réplicas da mesma configuração, por feature. O "
            "gerador ERENO não é determinístico: a mesma config produz bytes "
            "diferentes a cada execução. Cada coluna precisa superar o próprio "
            "piso, não o maior piso da tabela."
        ),
    }

    baseline_rows = pd.concat(frames.values()) if frames else pd.DataFrame()
    distinct = 0

    for variant in report["variants"]:
        if variant.get("compile", {}).get("status") != "passed":
            reasons.append(
                f"{variant['tag']}: o compilador recusou a intenção — "
                f"{variant['compile']['error']}"
            )
            continue
        if variant.get("generation", {}).get("status") == "failed":
            reasons.append(
                f"{variant['tag']}: o gerador recusou a configuração compilada — "
                f"{variant['generation']['error'].splitlines()[0]}"
            )
            continue
        if variant.get("gate", {}).get("status") != "passed":
            reasons.append(
                f"{variant['tag']}: reprovada no portão de volume — "
                f"{variant.get('gate', {}).get('error', 'sem trace')}"
            )
            continue
        if baseline_rows.empty:
            continue

        drift = _separation(
            _attack_rows(Path(variant["trace"]), spec, args.protocol_features),
            baseline_rows,
        )
        moved = _moved_features(drift, noise_by_feature, noise_constant, args)
        variant["drift"] = {
            "max": drift["max"],
            "top": drift["top"],
            "constant_shift": drift["constant_shift"],
            # Quais colunas o parâmetro moveu além do ruído daquela coluna — e
            # quantas. Um campo categórico que vira satura `max` sozinho: ler só
            # o pico faz uma coluna invertida parecer o tráfego inteiro tendo se
            # deslocado. É esta lista, e não `max`, que sustenta o veredito.
            "moved_features": moved,
            "moved_count": len(moved),
            "min_drift": args.min_drift,
            "drift_margin": args.drift_margin,
            "distinct_from_baseline": len(moved) >= args.min_moved_features,
        }
        if variant["drift"]["distinct_from_baseline"]:
            distinct += 1
        else:
            reasons.append(
                f"{variant['tag']}: nenhuma feature (0 de {args.min_moved_features} "
                f"exigidas) se moveu acima de {args.min_drift} e de "
                f"{args.drift_margin}x o ruído da própria coluna — o parâmetro não "
                "mudou o tráfego de forma distinguível da variação do gerador."
            )

    signatures = [
        item["signature"]["has_signal"]
        for item in report["replicates"] + report["variants"]
        if "signature" in item
    ]
    if signatures and not all(signatures):
        reasons.append(
            "Algum trace não apresenta assinatura: as linhas de ataque não se "
            f"separam das normais acima de {args.min_signature} em nenhuma feature."
        )

    compiled = [v for v in report["variants"] if v.get("compile", {}).get("status") == "passed"]
    if not compiled:
        reasons.append("Nenhuma variante compilou — não há o que comparar com a baseline.")

    report["distinct_variants"] = distinct
    report["verdict"] = "passed" if not reasons and distinct > 0 else "failed"
    if distinct == 0 and not reasons:
        report["reasons"].append("Nenhuma variante ficou distinguível da baseline.")
    report["run_dir"] = str(run_dir)


# --------------------------------------------------------------------------- #
# Sonda de token do estágio intent                                            #
# --------------------------------------------------------------------------- #
def _llm_probe(attacks: list[str], args: argparse.Namespace) -> dict[str, Any]:
    """Uma chamada real do IntentAgent por família, só para medir consumo.

    Importa o agente aqui dentro para que o caminho sem ``--llm-probe`` não puxe
    ``agno``/Groq — mesma disciplina do ``live.py`` no orquestrador.
    """

    from adversarial_ids.agents.intent.agent import IntentAgent

    agent = IntentAgent(model_id=args.model_id)
    probes: list[dict[str, Any]] = []
    total = 0

    for position, attack_key in enumerate(attacks):
        if position and args.llm_probe_spacing > 0:
            # TPM 8000 no tier on_demand e ~8,2k tokens por prompt: duas
            # chamadas na mesma janela viram rate_limit_exceeded, que aparece
            # de fora como "a LLM não chamou submit_intent_spec".
            print(f"  aguardando {args.llm_probe_spacing:.0f}s (TPM da conta Groq)...")
            time.sleep(args.llm_probe_spacing)

        prompt = f"Avalie a robustez do IDS reduzindo o recall do ataque {attack_key}."
        entry: dict[str, Any] = {"attack": attack_key, "prompt": prompt}
        try:
            spec = agent.interpret(prompt)
            entry["status"] = "passed"
            entry["base_attack"] = spec.base_attack
            entry["desired_effect"] = spec.desired_effect.value
        except Exception as exc:  # noqa: BLE001 — a medição não derruba o piloto
            entry["status"] = "failed"
            entry["error"] = str(exc)

        usage = agent.last_usage
        if usage is None or usage.is_empty:
            entry["usage"] = None
        else:
            entry["usage"] = usage.model_dump(mode="json")
            total += usage.total_tokens

        probes.append(entry)
        print(f"  intent[{attack_key}]: {entry['status']} — {entry['usage']}")

    return {
        "model_id": args.model_id,
        "probes": probes,
        "total_tokens": total,
        "note": (
            "Uma chamada do estágio intent por família. A campanha real também "
            "paga o Defender por rodada; o consumo completo de uma execução sai "
            "em LoopRecord.total_tokens (outputs/loop_records.json)."
        ),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--attack",
        nargs="+",
        choices=list_attack_keys(),
        default=list(DEFAULT_ATTACKS),
        metavar="ATAQUE",
        help="Famílias a pilotar. Default: " + ", ".join(DEFAULT_ATTACKS) + ".",
    )
    parser.add_argument(
        "--effect",
        choices=[effect.value for effect in DesiredEffect],
        default=DesiredEffect.LOWER_RECALL.value,
        help="Efeito desejado que orienta a direção das variantes.",
    )
    parser.add_argument(
        "--intensity",
        nargs="+",
        choices=[item.value for item in IntentIntensity],
        default=list(DEFAULT_INTENSITIES),
        metavar="NIVEL",
        help="Intensidades compiladas, uma variante por nível.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help=(
            "Gerações da MESMA baseline. Duas é o mínimo para existir piso de "
            "ruído; o default é três porque o piso é o máximo sobre os pares, e "
            "com um par só ele subestima a variação do gerador."
        ),
    )
    parser.add_argument(
        "--drift-margin",
        type=float,
        default=2.0,
        help=(
            "Quantas vezes o ruído da PRÓPRIA coluna a deriva precisa superar "
            "para aquela feature contar como movida."
        ),
    )
    parser.add_argument(
        "--min-drift",
        type=float,
        default=0.05,
        help=(
            "Piso absoluto de deriva por feature. Sem ele, uma coluna quase sem "
            "ruído entre réplicas passaria por qualquer deslocamento."
        ),
    )
    parser.add_argument(
        "--min-moved-features",
        type=int,
        default=1,
        help="Quantas features precisam se mover para a variante contar como distinta.",
    )
    parser.add_argument(
        "--min-signature",
        type=float,
        default=0.2,
        help=(
            "Separação mínima ataque-vs-normal em alguma feature para o trace "
            "contar como tendo assinatura (0.2 é a convenção de efeito pequeno)."
        ),
    )
    parser.add_argument(
        "--max-fields",
        type=int,
        default=3,
        help="Teto de campos que cada variante pode mexer (IntentRestrictions).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed da IntentSpec.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=20,
        help="Linhas de ataque salvas por trace para inspeção manual (0 desliga).",
    )
    parser.add_argument(
        "--min-attack-prevalence",
        type=float,
        default=INTENT_LOOP_MIN_ATTACK_PREVALENCE,
        help="Piso proporcional do portão E4 (0.0 desliga só o proporcional).",
    )
    parser.add_argument(
        "--protocol-features",
        choices=("drop", "deltas"),
        default="deltas",
        help=(
            "Quais colunas entram na medição. Default 'deltas': sem os deltas de "
            "sequência e temporização, replay e grayhole não têm assinatura "
            "alguma para medir."
        ),
    )
    parser.add_argument(
        "--generator-mode",
        choices=("jar", "cached"),
        default=GENERATOR_MODE,
        help=(
            "Default: o do ambiente. 'cached' serve o mesmo dataset sempre e "
            "portanto NÃO pilota nada — existe só para depurar o script."
        ),
    )
    parser.add_argument(
        "--llm-probe",
        action="store_true",
        help="Mede o consumo do estágio intent com chamadas reais (precisa de GROQ_API_KEY).",
    )
    parser.add_argument(
        "--llm-probe-spacing",
        type=float,
        default=90.0,
        help="Segundos entre chamadas da sonda, para não estourar o TPM da conta.",
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="Modelo usado pela sonda.")
    parser.add_argument(
        "--discard-traces",
        action="store_true",
        help=(
            "Apaga os CSVs ao final, mantendo hash, amostra e relatório. Atenção: "
            "o gerador não é determinístico, então um trace apagado não volta "
            "igual."
        ),
    )
    parser.add_argument(
        "--label",
        default="",
        help=(
            "Sufixo do diretório e do relatório desta execução. Sem ele, rodar a "
            "mesma família duas vezes sobrescreve os traces da primeira."
        ),
    )
    parser.add_argument(
        "--llm-probe-only",
        action="store_true",
        help="Só mede o consumo do estágio intent, sem gerar trace nenhum.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=f"Caminho do relatório (default: {DEFAULT_REPORT_PATH}, com --label no nome).",
    )

    return parser


def _bytes_written(attacks: list[dict[str, Any]]) -> int:
    """Soma o que esta execução deixou em disco: traces, amostras e configs."""

    total = 0
    for attack in attacks:
        run_dir = Path(attack["run_dir"])
        for entry in attack["replicates"] + attack["variants"]:
            candidates = [
                entry.get("trace"),
                entry.get("sample"),
                str(run_dir / f"{entry['tag']}_config.json"),
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                path = Path(candidate)
                if path.is_file():
                    total += path.stat().st_size

    return total


def format_report(report: dict[str, Any]) -> str:
    lines = ["Piloto E0 — famílias", ""]
    header = f"{'família':<20} {'veredito':<10} {'variantes':<10} {'ruído':<9} {'tempo':<8}"
    lines.append(header)
    lines.append("-" * len(header))

    for attack in report["attacks"]:
        variants = f"{attack['distinct_variants']}/{len(attack['variants'])}"
        lines.append(
            f"{attack['attack']:<20} {attack['verdict']:<10} {variants:<10} "
            f"{attack['noise_floor']['max_separation']:<9.3f} "
            f"{attack['seconds_total']:<8.1f}"
        )

    for attack in report["attacks"]:
        if attack["reasons"]:
            lines.append("")
            lines.append(f"{attack['attack']}:")
            for reason in attack["reasons"]:
                lines.append(f"  - {reason}")

    budget = report["budget"]
    lines.append("")
    lines.append(
        f"Gerações: {budget['generations']} | total {budget['seconds_total']:.1f}s | "
        f"média {budget['seconds_per_generation']:.1f}s por trace | "
        f"disco {budget['megabytes_on_disk']:.0f} MB"
    )
    if budget["llm"] is None:
        lines.append("Token: não medido (rode com --llm-probe para incluir o estágio intent).")
    else:
        lines.append(f"Token (sonda intent): {budget['llm']['total_tokens']}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    if args.replicates < 1:
        print("--replicates precisa ser >= 1.", file=sys.stderr)
        return 2
    if args.generator_mode == "cached":
        print(
            "AVISO: em modo cacheado todo trace é o mesmo arquivo. O piloto vai "
            "rodar, mas o piso de ruído e a deriva serão zero por construção — "
            "isso não é resultado sobre nenhuma família.\n",
            file=sys.stderr,
        )

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    attacks = [] if args.llm_probe_only else [
        run_attack(attack_key, args) for attack_key in args.attack
    ]

    llm = None
    if args.llm_probe or args.llm_probe_only:
        print("\nSonda de token do estágio intent")
        llm = _llm_probe(args.attack, args)

    generations = sum(
        len(attack["replicates"]) + len([v for v in attack["variants"] if "seconds" in v])
        for attack in attacks
    )
    seconds_generating = sum(attack["seconds_total"] for attack in attacks)

    if args.discard_traces:
        for attack in attacks:
            for entry in attack["replicates"] + attack["variants"]:
                trace = entry.get("trace")
                if trace:
                    Path(trace).unlink(missing_ok=True)
                    entry["trace_discarded"] = True

    # Custo em disco, medido depois do descarte para não anunciar espaço que
    # não está mais ocupado: cada trace do ERENO passa de 40 MB e o piloto
    # default guarda quinze. É esse número que diz se a campanha inteira cabe
    # na máquina. Conta só o que *esta* execução escreveu — varrer o diretório
    # inteiro somaria os traces das famílias que ficaram de fora do --attack.
    bytes_on_disk = _bytes_written(attacks)

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "generator_mode": args.generator_mode,
            "effect": args.effect,
            "intensities": list(args.intensity),
            "replicates": args.replicates,
            "drift_margin": args.drift_margin,
            "min_signature": args.min_signature,
            "max_fields_changed": args.max_fields,
            "seed": args.seed,
            "min_drift": args.min_drift,
            "min_moved_features": args.min_moved_features,
            "min_attack_prevalence": args.min_attack_prevalence,
            "protocol_features": args.protocol_features,
            "protocol_features_default": PROTOCOL_FEATURES_MODE,
        },
        "attacks": attacks,
        "budget": {
            "generations": generations,
            "seconds_total": round(seconds_generating, 3),
            "seconds_per_generation": round(seconds_generating / generations, 3)
            if generations
            else 0.0,
            "wall_clock_seconds": round(time.perf_counter() - started, 3),
            "megabytes_on_disk": round(bytes_on_disk / 1_048_576, 1),
            "llm": llm,
        },
    }

    output_path = Path(args.output) if args.output else (
        PILOT_DIR / (f"pilot_report_{args.label}.json" if args.label else "pilot_report.json")
    )
    save_json(output_path, report)

    print()
    print(format_report(report))
    print()
    print(f"Relatório salvo em: {output_path}")

    # Exit code diz se o E0 liberou as famílias pedidas para o E1. Uma família
    # reprovada é resultado legítimo — e é exatamente o que precisa ser lido
    # antes de reduzir escopo.
    return 0 if all(attack["verdict"] == "passed" for attack in attacks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
