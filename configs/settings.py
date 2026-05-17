from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

PROMPT_PATH = BASE_DIR / "prompts" / "claude_prompt.txt"

ATTACK_JSON_PATH = BASE_DIR / "inputs" / "uc03_masquerade_fault.json"
PERFORMANCE_RESULTS_PATH = BASE_DIR / "inputs" / "performance_results.json"

OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

LLM_RESPONSE_PATH = OUTPUTS_DIR / "llm_response.txt"
ITERATION_HISTORY_PATH = OUTPUTS_DIR / "iteration_history.json"
SUGGESTED_CONFIG_PATH = OUTPUTS_DIR / "suggested_attack_config.json"

METRICS_CSV_PATH = OUTPUTS_DIR / "metrics_history.csv"
LLM_RESPONSES_DIR = OUTPUTS_DIR / "llm_responses"
ATTACK_CONFIGS_DIR = OUTPUTS_DIR / "attack_configs"

EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

# Modelos disponíveis para testar um por vez.
MODEL_IDS = [
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "groq/compound",
    "groq/compound-mini",
]

# Modelo padrão, caso você rode apenas: python main.py
MODEL_ID = "llama-3.1-8b-instant"

TEMPERATURE = 0.2
TOTAL_ITERATIONS = 30

# Otimização de tokens
HISTORY_WINDOW = 1
TOP_FEATURE_IMPORTANCES = 5
MAX_EDITABLE_FIELDS = 12

# Pausas para reduzir risco de rate limit
SLEEP_BETWEEN_ITERATIONS_SECONDS = 8
SLEEP_BETWEEN_MODELS_SECONDS = 60

ERENO_PROJECT_PATH = str(BASE_DIR / "ERENO-2.0")

ERENO_ATTACK_CONFIG_RELATIVE_PATH = "config/attacks/uc03_masquerade_fault.json"

ERENO_OUTPUT_DATASET_PATH = str(
    BASE_DIR / "ERENO-2.0" / "target" / "training" / "training_dataset_claude_v6.csv"
)

ERENO_RUN_COMMAND = [
    "mvn",
    "exec:java",
    "-Dexec.mainClass=br.ufu.facom.ereno.ActionRunner",
    "-Dexec.args=config/actions/action_create_attack_dataset.json",
]