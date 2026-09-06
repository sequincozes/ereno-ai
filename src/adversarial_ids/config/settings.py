import os
from pathlib import Path

from dotenv import load_dotenv


# Carrega a configuração local antes de avaliar qualquer variável de ambiente.
load_dotenv()

# ============================================================
# Base paths
# ============================================================
#
# Layout src/: este arquivo vive em
#   <repo>/src/adversarial_ids/config/settings.py
# Logo, a raiz do repositório é parents[3] e o diretório do pacote é
# parents[1]. Dados, saídas e o runtime do gerador ficam FORA de src/
# (na raiz); apenas os prompts versionados vivem dentro do pacote.

PACKAGE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parents[3]

INPUTS_DIR = BASE_DIR / "inputs"
PROMPTS_DIR = PACKAGE_DIR / "prompts"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Seeds versionados (fixtures reprodutíveis) — vivem FORA de outputs/.
DATA_DIR = BASE_DIR / "data"
BASELINE_DATASET_PATH = DATA_DIR / "baseline_dataset.csv"
GOLDEN_HISTORY_PATH = DATA_DIR / "iteration_history.json"

# ============================================================
# Input files
# ============================================================

PROMPT_PATH = PROMPTS_DIR / "strategist.md"
ATTACK_JSON_PATH = INPUTS_DIR / "uc03_masquerade_fault.json"

# ============================================================
# Output files and directories
# ============================================================

ITERATION_HISTORY_PATH = OUTPUTS_DIR / "iteration_history.json"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

# ============================================================
# LLM models used in the experiments
# ============================================================

MODEL_IDS = [
    "groq/compound-mini",
    "openai/gpt-oss-20b",
    "groq/compound",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
]

# Default model used when running only:
#   adversarial-ids
# (ou: python -m adversarial_ids.interfaces.cli)
MODEL_ID = os.getenv("MODEL_ID", "llama-3.1-8b-instant").strip()

# ============================================================
# Agent configuration
# ============================================================

TEMPERATURE = 0.2
TOTAL_ITERATIONS = 30

# Context control to reduce token usage
HISTORY_WINDOW = 1
TOP_FEATURE_IMPORTANCES = 5
MAX_EDITABLE_FIELDS = 12

# Rate-limit control
SLEEP_BETWEEN_MODELS_SECONDS = 60

# ============================================================
# Synthetic generator runtime configuration
# ============================================================

GENERATOR_RUNTIME_DIR = BASE_DIR / "generator_runtime"

GENERATOR_JAR_PATH = GENERATOR_RUNTIME_DIR / "ereno-generator.jar"

GENERATOR_ATTACK_CONFIG_RELATIVE_PATH = "config/attacks/uc03_masquerade_fault.json"

GENERATOR_ACTION_CONFIG_RELATIVE_PATH = "config/actions/action_create_attack_dataset.json"

GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH = (
    "config/actions/action_create_benign_dataset.json"
)

# This filename must match the output file defined inside:
# generator_runtime/config/actions/action_create_attack_dataset.json
GENERATOR_OUTPUT_DATASET_PATH = (
    GENERATOR_RUNTIME_DIR
    / "target"
    / "training"
    / "training_dataset_claude_v6.csv"
)

GENERATOR_RUN_COMMAND = [
    "java",
    "-jar",
    str(GENERATOR_JAR_PATH),
    GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
]

# ------------------------------------------------------------
# Modo do gerador: "cached" (lê data/baseline_dataset.csv, sem Java)
# ou "jar" (executa o ERENO real). O padrão é rodar SEM Java quando o
# JAR não está presente — assim M1/M2/M4 e o CI rodam o loop localmente.
# Sobrescreva com a env var GENERATOR_MODE=jar|cached.
# ------------------------------------------------------------

def _default_generator_mode() -> str:
    return "jar" if GENERATOR_JAR_PATH.exists() else "cached"


GENERATOR_MODE = os.getenv("GENERATOR_MODE", _default_generator_mode()).strip().lower()

# ------------------------------------------------------------
# Robustez do subprocesso do JAR (janela D15-24 do roadmap: "loop
# confiável e multi-ataque" — quality gates já cobertos por E4, isto cobre
# retries/timeout). Só se aplica ao modo "jar"; o modo "cached" nunca chama
# subprocess. Sobrescreva via env var quando o JAR local for sabidamente
# mais lento/instável do que o default.
# ------------------------------------------------------------

GENERATOR_TIMEOUT_SECONDS = float(os.getenv("GENERATOR_TIMEOUT_SECONDS", "300"))
GENERATOR_MAX_RETRIES = int(os.getenv("GENERATOR_MAX_RETRIES", "1"))
GENERATOR_RETRY_BACKOFF_SECONDS = float(
    os.getenv("GENERATOR_RETRY_BACKOFF_SECONDS", "2")
)

# ============================================================
# Pipeline intent-driven (E3/E4) — Orchestrator v2
# ============================================================

# LoopRecord (INTENT→...→FEEDBACK), persistido append-only por execução.
LOOP_RECORDS_PATH = OUTPUTS_DIR / "loop_records.json"

# Artefatos de cada execução (intent.json, attack_candidate.json, datasets,
# dataset_bundle.json, detection_report.json), isolados por run_id.
INTENT_LOOP_OUTPUT_DIR = OUTPUTS_DIR / "intent_loop"

# Piso de volume do gate de integração do ERENO (E4): abaixo disso o trace é
# rejeitado antes de alimentar o detector (ver core/dataset_bundle_builder.py).
INTENT_LOOP_MIN_ATTACK_ROWS = 5
INTENT_LOOP_MIN_NORMAL_ROWS = 5

# Campanha multi-rodada do estágio FEEDBACK (E10). O default é 1 rodada — o
# mesmo comportamento de antes do E10 — porque cada rodada extra paga um ciclo
# completo de ERENO + treino/avaliação do detector.
INTENT_LOOP_DEFAULT_ROUNDS = 1

# Ganho mínimo na métrica-objetivo para a campanha considerar que a rodada
# progrediu; abaixo disso a política para com 'no_improvement' em vez de
# queimar rodadas num platô (ver core/feedback_policy.py).
FEEDBACK_MIN_DELTA = 0.01

# ------------------------------------------------------------
# Preprocessador de features (E6): "modular" ajusta o
# FeaturePreprocessor (core/preprocessor.py) só na partição de treino —
# sem leakage. "legacy" mantém o comportamento anterior ao E6
# (core/ids_evaluator.py ajustava colunas constantes e vocabulário
# categórico sobre o dataset inteiro, antes do train_test_split), só para
# comparar métricas antes/depois da correção. Sobrescreva com a env var
# PREPROCESSOR_MODE=legacy; "modular" é o default e o caminho recomendado.
# ------------------------------------------------------------

PREPROCESSOR_MODE = os.getenv("PREPROCESSOR_MODE", "modular").strip().lower()

# ------------------------------------------------------------
# Undersampling + seleção de features (épico E7): consomem
# FeatureManifest.feature_columns (E6) como ponto de partida, não fazem
# parte do FeaturePreprocessor — ver core/feature_selector.py e
# core/undersampler.py. Default "none" nos dois: comportamento idêntico a
# antes do E7 existir, até alguém optar explicitamente por uma estratégia.
# Sobrescreva com as env vars abaixo.
# ------------------------------------------------------------

FEATURE_SELECTION_MODE = os.getenv("FEATURE_SELECTION_MODE", "none").strip().lower()

_feature_selection_top_k_raw = os.getenv("FEATURE_SELECTION_TOP_K", "").strip()
FEATURE_SELECTION_TOP_K = int(_feature_selection_top_k_raw) if _feature_selection_top_k_raw else None

_feature_selection_min_score_raw = os.getenv("FEATURE_SELECTION_MIN_SCORE", "").strip()
FEATURE_SELECTION_MIN_SCORE = (
    float(_feature_selection_min_score_raw) if _feature_selection_min_score_raw else None
)

UNDERSAMPLING_MODE = os.getenv("UNDERSAMPLING_MODE", "none").strip().lower()

# ------------------------------------------------------------
# Detector plugável (épico E8): qual modelo o IdsEvaluator treina —
# "random_forest" (default, comportamento idêntico ao de antes do E8),
# "decision_tree", "svm_linear" ou "svm_rbf". Ver core/detectors.py e
# docs/detectors.md. Sobrescreva com a env var DETECTOR_MODE.
#
# Cuidado ao regenerar fixtures: como PREPROCESSOR_MODE/FEATURE_SELECTION_MODE,
# esta é uma env var que muda o modelo treinado — por isso
# scripts/generate_golden_history.py fixa o detector explicitamente em vez de
# herdar este default, para que um DETECTOR_MODE no shell de alguém não
# reescreva o histórico golden com outro modelo.
# ------------------------------------------------------------

DETECTOR_MODE = os.getenv("DETECTOR_MODE", "random_forest").strip().lower()

# Escala aplicada pelo FeaturePreprocessor (E6). Vazio (default) = delegar ao
# detector: "standard" para os SVMs (sensíveis a escala), "none" para as
# árvores — ver core/detectors.py::recommended_scaler. Defina
# DETECTOR_SCALER=none|standard para forçar os dois lados a usar a mesma
# escala numa comparação controlada entre detectores.

_detector_scaler_raw = os.getenv("DETECTOR_SCALER", "").strip().lower()
DETECTOR_SCALER = _detector_scaler_raw or None

# Quais colunas de protocolo GOOSE chegam ao detector (ver docs/preprocessing.md).
# "drop" (default) descarta identidade e deltas juntos — o comportamento herdado
# do commit inicial do framework. "deltas" descarta só a identidade, deixando
# stDiff/sqDiff/tDiff e companhia disponíveis: são elas que carregam a semântica
# de sequência e temporização, e sem elas replay/flooding/grayhole não têm como
# ser detectados. Default conservador de propósito: virar a chave move toda a
# linha de base de métricas, então a decisão é do experimento, com ablação.
PROTOCOL_FEATURES_MODE = os.getenv("PROTOCOL_FEATURES_MODE", "drop").strip().lower()
