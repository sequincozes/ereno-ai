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
