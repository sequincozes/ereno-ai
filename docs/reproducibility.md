# Reprodutibilidade — rodar sem Java (modo cacheado)


O gerador sintético do ERENO é um `.jar` Java pesado que fica **fora do repo**
(gitignored). Para destravar o trabalho paralelo de M1/M2/M4 e o CI, o núcleo
oferece um **modo cacheado**: em vez de rodar o Java, ele serve um dataset
benigno+ataque já versionado. O objetivo é exercitar o fluxo (agentes, memória,
dashboard), não reproduzir a variação física por configuração.

---

## 1. Pré-requisitos

- [`uv`](https://docs.astral.sh/uv/) instalado.
- **Java não é necessário** no modo cacheado.
- **Chave da Groq** só é necessária para chamar o agente Estrategista real
  (tool calling). Os testes, o golden e o reset rodam sem chave.

```bash
uv sync --extra dev
```

---

## 2. Os dois modos do gerador

Selecionados em `config/settings.py` (`GENERATOR_MODE`), sobrescrevíveis pela
variável de ambiente `GENERATOR_MODE`:

| Modo     | O que faz                                                        | Precisa de Java? |
| -------- | ---------------------------------------------------------------- | ---------------- |
| `cached` | Copia `data/baseline_dataset.csv` como o dataset de cada iteração | Não              |
| `jar`    | Executa o gerador ERENO real sobre o `attack_config`             | Sim (+ o `.jar`) |

**Padrão:** o modo cai automaticamente em `cached` quando o JAR
(`generator_runtime/ereno-generator.jar`) **não** está presente, e em `jar`
quando está. Para forçar:

```bash
GENERATOR_MODE=cached uv run adversarial-ids --iterations 2
```

No modo cacheado o dataset é o mesmo em toda iteração — os configs, diffs, logs
e o histórico continuam sendo produzidos normalmente nos dois modos.

### Como obter o gerador ERENO (JAR)

O JAR é um uber-jar (~25 MB) mantido **fora do versionamento de código**
(gitignored, por peso e por ser artefato de build). Ele é necessário **apenas**
para o modo `jar`; os modos `cached`/`demo`, os testes e o golden rodam sem ele.

Ele é distribuído como **asset da release** da entrega. Baixe-o direto para
`generator_runtime/` (requer acesso ao repositório privado — use o `gh` CLI
autenticado):

```bash
gh release download v1.0.0 --repo doffyGC/ERENO-AI-LAB \
  --pattern "ereno-generator.jar" --dir generator_runtime
```

Verifique a integridade do arquivo baixado (SHA-256):

```text
152f3d6108a4c5df33eb7ecd0c0858533f47afa5f97f0cf99858bb904badf830  ereno-generator.jar
```

```bash
# Linux/macOS
sha256sum generator_runtime/ereno-generator.jar
# PowerShell
Get-FileHash generator_runtime/ereno-generator.jar -Algorithm SHA256
```

Alternativamente, reconstrua o JAR a partir do projeto ERENO original
(`mvn clean package` gera `target/ERENO-1.0-SNAPSHOT-uber.jar`) e copie o
resultado para `generator_runtime/ereno-generator.jar`. Depois de posicionado, o
`GENERATOR_MODE` cai automaticamente em `jar`.

---

## 3. Seeds versionados (fonte da verdade)

Vivem em `data/` (fora de `outputs/`) e **não são apagados** pelo reset:

| Arquivo                       | O que é                                                              |
| ----------------------------- | ------------------------------------------------------------------- |
| `data/baseline_dataset.csv`   | Dataset benigno+ataque servido no modo cacheado (coluna `class`).   |
| `data/iteration_history.json` | Histórico **golden**: uma lista de `IterationRecord` estável.        |

O golden é consumido pelo Dashboard (#19) e pelos testes sem precisar de API,
Java ou da execução dos agentes reais.

---

## 4. Determinismo

- `IdsEvaluator` usa `random_state=42` (split treino/teste + Random Forest),
  então as métricas são reprodutíveis.
- Os stubs `FakeStrategist` / `FakeAnalyst` (em `tests/fakes/`) são
  **determinísticos**: dada a mesma iteração, devolvem sempre a mesma saída.
- O gerador do golden (`scripts/generate_golden_history.py`) usa um **timestamp
  fixo**, então `data/iteration_history.json` fica byte-estável entre
  regenerações.

Resultado: rodar o golden duas vezes produz exatamente o mesmo arquivo.

---

## 5. Comandos

### Verificar a infra (sem chave, sem Java)

```bash
uv run pytest
```

Exercita o modo cacheado, os schemas de `domain/`, a persistência do histórico
(`ExperimentMemory`) e os stubs.

### Rodar o loop cacheado e regenerar o golden

```bash
uv run python scripts/generate_golden_history.py
```

Encadeia `FakeStrategist → gerador cacheado → IdsEvaluator → FakeAnalyst` e
grava cada passo como `IterationRecord` em `data/iteration_history.json`.

### Resetar o estado do experimento

```bash
uv run python scripts/init_state.py            # limpa outputs/ e logs/
uv run python scripts/init_state.py --dry-run  # só mostra o que faria
uv run python scripts/init_state.py --keep-logs
```

Limpa os artefatos gerados **preservando os seeds em `data/`**. Use antes de uma
execução limpa. É idempotente e seguro: nunca toca em `data/`.

### Rodar a CLI com o agente real (precisa de chave da Groq)

```bash
cp .env.example .env        # preencha GROQ_API_KEY
uv run adversarial-ids --model-id llama-3.1-8b-instant --iterations 2
```

Sem o JAR, a geração cai em modo cacheado automaticamente.

---

## 6. Como o histórico é persistido

`core/experiment_memory.py` (`ExperimentMemory`) acumula `IterationRecord` e
grava em disco no mesmo formato do golden:

```json
{ "history": [ { "iteration": 0, "attack_config": { ... }, "metrics": { ... }, "..." : "..." } ] }
```

- `add_record(record)` — caminho do Orquestrador (#16), com o registro já tipado.
- `add_iteration(...)` — adaptador a partir de dicts frouxos, usado pela CLI;
  valida contra os schemas de `domain/` antes de guardar.
- `ExperimentMemory.load(path)` — reconstrói a memória de um histórico salvo
  (memória vazia se o arquivo não existir), permitindo retomar de onde parou.

Por serem `IterationRecord` válidos, os históricos gerados pela CLI e pelo golden
têm o **mesmo contrato** — o Dashboard (#19) e a memória do Estrategista (#8)
leem qualquer um dos dois sem tratamento especial.
