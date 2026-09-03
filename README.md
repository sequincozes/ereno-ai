# ERENO AI Lab — Laboratório Adversarial de IDS para Smart Grids

Loop adversarial **Red Team × Blue Team** que gera ataques sintéticos
parametrizáveis (padrão IEC-61850 / GOOSE, via ERENO) para **validar e refinar um
sistema de detecção de intrusão (IDS)**. Um agente Estrategista (Red Team) propõe
mudanças na configuração do ataque a cada iteração; o núcleo regenera o dataset,
treina/avalia o IDS e mede o impacto; um agente Analista (Blue Team) explica quais
features enganaram o detector e recomenda mitigação. Os dois agentes são
implementados com o framework **Agno** e orquestrados por um `agno.team.Team`.

## Assunto geral escolhido

**Segurança de Informação.**

O projeto corresponde diretamente ao exemplo do enunciado: *"Sistema que gera
ataques parametrizáveis para validação e refinamento de sistemas de detecção de
intrusão"*. Não usamos dados sensíveis reais — todo o tráfego é **sintético**,
gerado pelo simulador ERENO sobre cenários IEC-61850.

## Descrição do problema

Sistemas de detecção de intrusão para subestações de energia (Smart Grids)
costumam ser avaliados contra um conjunto **estático** de ataques. Isso esconde
uma fragilidade: pequenas variações nos parâmetros do ataque (temporização,
campos manipulados, intensidade) podem derrubar a taxa de detecção sem que a
equipe de defesa perceba **por quê**. Falta um processo que (1) explore
sistematicamente variações de ataque, (2) meça o efeito de cada variação sobre o
IDS e (3) explique, em linguagem acionável, quais características do tráfego
levaram à evasão.

Este sistema fecha esse ciclo: transforma a avaliação do IDS em um **loop
adversarial iterativo e explicável**, produzindo tanto evidência quantitativa (F1,
matriz de confusão, diffs de configuração) quanto diagnóstico qualitativo
(features enganosas + mitigações sugeridas).

## Usuário-alvo

**Pesquisadores e engenheiros de segurança** de infraestruturas críticas
(Smart Grids / subestações IEC-61850) e **times de Blue Team / SOC** responsáveis
por validar e endurecer modelos de detecção de intrusão. O sistema serve a quem
precisa testar a robustez de um IDS contra ataques variados e entender as causas
de evasão, sem escrever manualmente cada variante de ataque.

## Integrantes do grupo

| Nome completo | Matrícula | Username do GitHub |
|---|---|---|
| Guilherme Chagas | `2310102067` | [@doffyGC](https://github.com/doffyGC) |
| Camilla Borchhardt | `2310200485` | [@camillabdt](https://github.com/camillabdt) |
| João Pedro | `2010101298` | [@joao-pedro-rdo](https://github.com/joao-pedro-rdo) |
| Inaurrara | `2510200480` | [@Inaurrara](https://github.com/Inaurrara) |

Divisão de responsabilidades em [docs/team.md](docs/team.md).

## Fluxo principal do sistema

O fluxo completo (`--engine live`) executa, por iteração:

1. **Baseline** — o núcleo gera/serve o dataset de ataque inicial
   (`inputs/uc03_masquerade_fault.json`), treina o IDS (Random Forest,
   scikit-learn) e mede as métricas de referência.
2. **Estrategista (Red Team, Agno)** — recebe a configuração atual, as métricas e
   o histórico e propõe, via *tool calling* estruturado, uma nova
   `AttackConfig` (persona `conservative` altera 1 campo; `aggressive`, até 3).
3. **Núcleo** — aplica o patch na configuração, regenera o dataset (modo `cached`
   sem Java, ou `jar` com o gerador ERENO real), re-treina e re-avalia o IDS.
4. **Analista (Blue Team, Agno)** — interpreta as novas métricas, identifica as
   features que enganaram o detector e recomenda mitigações, com a saída validada
   contra as próprias métricas por regras determinísticas.
5. **Memória** — cada passo vira um `IterationRecord` tipado, persistido em
   `outputs/iteration_history.json`.
6. **Apresentação** — CLI imprime o resumo; o dashboard Streamlit mostra desempenho
   (F1 por iteração, matriz de confusão, diffs) e explicabilidade.

A orquestração real encadeia **Estrategista → núcleo → Analista** por meio de um
`agno.team.Team` em *route mode* (`--orchestration team`, default), com fallback
determinístico (`--orchestration direct`).

Há também um **modo demonstração** (`--engine demo`, default) que faz replay de um
histórico *golden* versionado, sem Groq nem Java — útil para avaliação rápida e
reprodutível.

## Tecnologias utilizadas

- **Python ≥ 3.10**
- **[Agno](https://docs.agno.com/)** — agentes (`Agent`), times (`Team`) e tools
- **[Groq](https://groq.com/)** — provedor do modelo de linguagem (default
  `llama-3.1-8b-instant`)
- **Pydantic v2** — contratos de domínio e validação (`AttackConfig`, `Metrics`,
  `IterationRecord`, `StrategistOutput`, `AnalystOutput`)
- **pandas** + **scikit-learn** — treino/avaliação do IDS (Random Forest)
- **Streamlit** — dashboard de desempenho e explicabilidade
- **ERENO** — gerador sintético de tráfego IEC-61850 / GOOSE (JAR opcional)
- **SHAP** (opcional) — explicações de importância de features
- **uv** — gerenciamento de dependências e execução
- **pytest** — testes automatizados

## Pré-requisitos

- Python 3.10 ou superior;
- [`uv`](https://docs.astral.sh/uv/);
- Java e o JAR do ERENO **somente** para o modo `jar`;
- chave Groq **somente** para funcionalidades que chamem os agentes reais.

A demonstração cacheada, a CLI no modo demo, o dashboard e os testes **não**
precisam de Java nem de chave Groq.

## Instalação

```bash
git clone https://github.com/doffyGC/ERENO-AI-LAB.git
cd ERENO-AI-LAB
uv sync --extra dev --extra dashboard
```

Para habilitar as explicações opcionais por SHAP:

```bash
uv sync --extra shap
```

O projeto usa backend `setuptools` e é compatível com instalação editável, mas
`uv` é o fluxo adotado pelo repositório.

## Configuração das variáveis de ambiente

Linux/macOS:

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

Variáveis (documentadas em `.env.example`):

```env
GROQ_API_KEY=
MODEL_ID=llama-3.1-8b-instant
# GENERATOR_MODE=cached
```

- `GROQ_API_KEY`: **sensível**; obrigatória apenas para chamadas reais à Groq
  (`--engine live`);
- `MODEL_ID`: opcional; possui o default acima;
- `GENERATOR_MODE`: opcional; aceita `cached` ou `jar` (default: `jar` se o JAR
  existir, senão `cached`).

Nunca versione `.env` — ele já está no `.gitignore`. O `.env.example` deve
permanecer sem segredos.

## Instruções de execução

### CLI — modo demonstração (default, sem Groq/Java)

Faz replay do histórico *golden* por meio do `CachedHistoryRunner`:

```bash
uv run adversarial-ids --iterations 2
```

Saída esperada:

```text
Execução concluída: 3 registro(s), última iteração=2, F1 final=1.0000.
```

### CLI — loop adversarial real (`--engine live`)

Executa Estrategista → núcleo → Analista (exige `GROQ_API_KEY`; o modo `jar`
também exige o JAR do ERENO):

```bash
uv run adversarial-ids --engine live --generator-mode cached --iterations 3
```

Para colocar os agentes trabalhando sobre **outro tipo de ataque** do ERENO,
use `--attack` no modo `jar` (o gerador precisa regenerar o dataset por config):

```bash
uv run adversarial-ids --engine live --generator-mode jar --attack random_replay --iterations 3
```

Principais opções (`uv run adversarial-ids --help`):

- `--iterations`: número de variantes após o baseline;
- `--engine`: `demo` (replay golden) ou `live` (loop real);
- `--attack`: apenas no `live` — tipo de ataque a otimizar. Opções:
  `masquerade_fault` (default), `random_replay`, `inverse_replay`, `injection`,
  `high_stnum`, `flooding`, `grayhole`, `delayed_replay`, `delayed_replay_backoff`,
  `delayed_replay_batch_dump`, `delayed_replay_double_drop`. Os parâmetros
  editáveis são derivados automaticamente da configuração de cada ataque
  (`inputs/attacks/`); com `--generator-mode cached` o `--attack` é ignorado
  (o dataset cacheado é o do masquerade);
- `--orchestration`: apenas no `live` — `team` (default, `agno.team.Team`) ou
  `direct` (encadeamento determinístico);
- `--persona`: `conservative` (default) ou `aggressive`;
- `--generator-mode`: `cached` (sem Java) ou `jar`;
- `--model-id`: modelo Groq no modo live;
- `--open-dashboard`: abre o dashboard ao terminar.

Ao final, a CLI informa onde os resultados foram salvos
(`outputs/iteration_history.json`) e como abrir o dashboard.

### CLI — pipeline intent-driven (`--engine intent`)

Executa o pipeline INTENT→GENERATOR→ERENO→PREPROCESS→DETECTOR→DEFENDER→
FEEDBACK a partir de uma intenção em linguagem natural (exige
`GROQ_API_KEY`):

```bash
uv run adversarial-ids --engine intent --prompt "reduza o recall variando a temporização da falha"
```

Para encadear rodadas pela política de feedback (E10) — a partir da segunda
rodada, a intenção vem da política, não de uma nova chamada de LLM:

```bash
uv run adversarial-ids --engine intent --prompt "reduza o recall variando a temporização da falha" \
  --generator-mode jar --rounds 3
```

`--rounds` (default 1) é específico do `--engine intent` — distinto de
`--iterations`, que pertence ao loop legado (`--engine live`/`demo`). Cada
rodada gera um `LoopRecord` próprio em `outputs/loop_records.json`,
encadeado pelo `parent_run_id` da rodada anterior.

Não há `--attack` no motor intent: o ataque-base vem da própria intenção
(`IntentSpec.base_attack`, extraída do prompt pelo `IntentAgent` a partir do
catálogo real de capacidades — ver `docs/attack_capabilities.md`). Os 11
ataques registrados têm capacidade intent-driven habilitada, mas **os 10 que
não são `masquerade_fault` só são fisicamente mensuráveis em
`--generator-mode jar`**: em `cached` o dataset servido é sempre o de
`masquerade_fault` (`data/baseline_dataset.csv`), então o portão de
integração (PREPROCESS) rejeita qualquer outro ataque por rótulo de classe
ausente. As quatro variantes `delayed_replay*` também compartilham a mesma
classe rotulada no dataset (`delayed_replay`), então esse portão não as
distingue entre si.

### ERENO AI — frontend completo (recomendado)

Interface Streamlit que substitui o uso via terminal: **toda** a execução
(configuração de parâmetros, disparo do loop com logs ao vivo e o dashboard de
resultados) acontece pela UI, sem CLI. Quatro páginas — Visão Geral,
Configuração, Execução e Resultados:

```bash
uv run --extra dashboard streamlit run src/adversarial_ids/interfaces/dashboard/studio_app.py
```

Depois abra <http://localhost:8501>. Na página **Configuração** você escolhe
motor (`demo`/`live`), ataque, iterações, persona, orquestração, modo do gerador
e modelo; em **Execução** o loop roda em background com os logs
(`[ORCH]`/`[VALIDATOR]`/…) transmitidos num painel de terminal; em **Resultados**
fica o panorama (KPIs de F1, evolução das métricas, matriz de confusão,
importância de features, diffs de configuração e o diagnóstico do Analista).

### Dashboard Streamlit (versão simples/legada)

```bash
uv run --extra dashboard streamlit run src/adversarial_ids/interfaces/dashboard/app.py
```

O dashboard carrega automaticamente a última execução e oferece três fontes:
**Carregar última execução** (`outputs/`), **Demo cacheada (golden)** (`data/`) e
**Executar ao vivo (Groq)**. Exibe desempenho (resumo, F1 por iteração, matriz de
confusão, comparação de configurações) e explicabilidade (diagnóstico, features
enganosas e mitigações).

### Regenerar / resetar o estado

```bash
uv run python scripts/generate_golden_history.py   # regenera o histórico golden
uv run python scripts/init_state.py                # limpa outputs/ e logs/ (preserva data/)
uv run python scripts/init_state.py --dry-run      # mostra o que faria
```

### Testes

```bash
uv run --extra dashboard --extra dev pytest        # suíte completa
uv run python -m compileall src                    # verificação de sintaxe
```

## Estrutura de diretórios

```text
ERENO-AI-LAB/
├── data/                    # seed CSV e histórico golden (fonte da verdade)
├── docs/                    # equipe, arquitetura, reprodutibilidade, roteiro do vídeo
├── generator_runtime/       # configs e local esperado do JAR do ERENO
├── inputs/                  # configurações de ataque (baseline e adversarial)
├── scripts/                 # geração golden, reset e utilitários experimentais
├── src/adversarial_ids/
│   ├── agents/
│   │   ├── strategist/      # agente Red Team (Agno) + tools, personas, parser
│   │   ├── analyst/         # agente Blue Team (Agno) + tools de validação
│   │   └── orchestrator/    # workflow.py (agente-agnóstico) + live.py (agentes reais + Team)
│   ├── config/              # settings.py (caminhos e variáveis de ambiente)
│   ├── core/                # gerador, avaliador do IDS e memória do experimento
│   ├── domain/              # modelos Pydantic compartilhados
│   ├── interfaces/          # cli.py, experiment_runner.py, dashboard/app.py
│   ├── prompts/             # instruções versionadas dos agentes (strategist.md, analyst.md)
│   └── shared/              # JSON I/O, patch, validação e logging
├── tests/                   # contratos, cache, CLI, dashboard, agentes e orquestração
├── .env.example             # variáveis de ambiente documentadas
├── pyproject.toml           # dependências e metadados do pacote
└── README.md
```

Detalhes de arquitetura em [docs/architecture.md](docs/architecture.md);
reprodutibilidade em [docs/reproducibility.md](docs/reproducibility.md).

## Checklist Técnico Obrigatório

### Tema e Escopo

- [x] O projeto escolhe um dos assuntos gerais definidos no enunciado. — *Segurança de Informação (seção "Assunto geral escolhido").*
- [x] O problema resolvido está descrito claramente. — *Seção "Descrição do problema".*
- [x] O usuário-alvo está definido. — *Seção "Usuário-alvo".*
- [x] O fluxo principal de uso está descrito. — *Seção "Fluxo principal do sistema".*
- [x] O escopo implementado é compatível com o problema proposto. — *Loop adversarial funcional com agentes reais (`--engine live`).*

### Agno e Agentic AI

- [x] O sistema usa Python. — *Todo o `src/adversarial_ids/`; `pyproject.toml`.*
- [x] O sistema usa o framework Agno. — *Dependência `agno`; `agents/strategist/agent.py`, `agents/analyst/agent.py`, `agents/orchestrator/workflow.py`.*
- [x] O sistema implementa pelo menos um agente funcional. — *`StrategistAgent` e `AnalystAgent`.*
- [x] O uso do agente é necessário para o fluxo principal do sistema. — *`run_live_workflow` encadeia os agentes reais (`agents/orchestrator/live.py`).*
- [x] O prompt, instruções ou configuração do agente estão versionados no repositório. — *`src/adversarial_ids/prompts/strategist.md` e `analyst.md`.*
- [x] O sistema usa pelo menos uma ferramenta, integração, workflow, time de agentes ou chamada estruturada. — *`agno.team.Team` em route mode (`_build_team`), tools `submit_strategist_output` e `validate_output_against_metrics`.*

### Memória, Persistência ou Base de Conhecimento

- [x] O sistema mantém algum estado, histórico, memória, base de conhecimento ou artefato persistido. — *`ExperimentMemory` grava `IterationRecord` em `outputs/iteration_history.json`.*
- [x] O projeto documenta onde esse estado é armazenado. — *[docs/reproducibility.md](docs/reproducibility.md) (seções 3 e 6); `data/` (golden) e `outputs/` (execuções).*
- [x] O projeto permite reconstruir, limpar ou inicializar esse estado quando necessário. — *`scripts/init_state.py`, `scripts/generate_golden_history.py` e `ExperimentMemory.load(path)`.*

**RAG:** não se aplica — o projeto não usa RAG (o conhecimento vem das métricas do
IDS e do histórico tipado, não de um corpus recuperado).

- [ ] O corpus ou fonte de dados está documentado. — *N/A (sem RAG).*
- [ ] A estratégia de ingestão está documentada. — *N/A (sem RAG).*
- [ ] A estratégia de chunking ou indexação está documentada. — *N/A (sem RAG).*
- [ ] A resposta indica fontes, trechos ou evidências usadas. — *N/A (sem RAG); a saída do Analista referencia as features/métricas concretas da iteração.*

### Validação e Qualidade da Saída

- [x] O sistema tem alguma estratégia para validar, revisar ou justificar a saída gerada. — *`validate_output_against_metrics` (regras determinísticas) + schemas Pydantic + verificação de iteração em `TeamAnalyst.analyze`.*
- [x] A validação é demonstrável por teste, regra, agente avaliador, checklist, métrica, comparação com fonte ou revisão estruturada. — *`tests/test_analyst_tools.py`, `tests/test_analyst_agent.py`, `tests/test_domain_schemas.py`, `tests/test_live_wiring.py`.*
- [x] O sistema informa limitações, incertezas ou casos em que não consegue responder adequadamente. — *F1 exibido como "indisponível" quando ausente (`cli.py`); `ValueError` em iteração incompatível; seção "Limitações conhecidas".*

### Interface ou Execução

- [x] O sistema oferece uma forma clara de uso (CLI, Streamlit, ...). — *CLI (`adversarial-ids`) e dashboard Streamlit.*
- [x] O fluxo principal pode ser executado seguindo instruções do README.md. — *Seção "Instruções de execução".*
- [x] O projeto inclui dados de exemplo, prompts de exemplo ou comandos de exemplo. — *`data/`, `inputs/`, `prompts/` e comandos de exemplo no README.*

### Reprodutibilidade

- [x] O repositório contém README.md. — *Este arquivo.*
- [x] O repositório contém pyproject.toml. — *`pyproject.toml`.*
- [x] O repositório contém .gitignore. — *`.gitignore`.*
- [x] O repositório contém .env.example. — *`.env.example`.*
- [x] O README.md explica como instalar dependências. — *Seção "Instalação".*
- [x] O README.md explica como configurar variáveis de ambiente. — *Seção "Configuração das variáveis de ambiente".*
- [x] O README.md explica como executar o sistema. — *Seção "Instruções de execução".*
- [x] O README.md lista os integrantes com nome, matrícula e username do GitHub. — *Seção "Integrantes do grupo".*

### Engenharia de Software

- [x] O código está organizado em módulos ou diretórios com responsabilidades claras. — *`agents/`, `core/`, `domain/`, `interfaces/`, `shared/`.*
- [x] A lógica do agente está separada da interface. — *`agents/` (lógica) × `interfaces/` (CLI/dashboard) via `ExperimentRunner`.*
- [x] Configurações e credenciais não estão hardcoded. — *`config/settings.py` + `.env` (nada de segredos no código).*
- [x] O código usa nomes descritivos. — *Ex.: `StrategistAdapter`, `validate_output_against_metrics`, `IdsEvaluator`.*
- [x] O código usa type hints nas assinaturas principais. — *`from __future__ import annotations`; assinaturas tipadas em todo o pacote.*
- [x] Funções e classes duráveis têm docstrings. — *Ex.: `live.py`, `workflow.py`, `experiment_memory.py`.*
- [x] O projeto possui testes, scripts de verificação ou exemplos executáveis. — *`tests/` (pytest) e `scripts/`.*
- [x] O projeto trata erros esperados no fluxo principal. — *`run_cli` captura `KeyboardInterrupt`/`Exception`; validações levantam `ValueError`.*

### Segurança e Dados

- [x] O repositório não contém chaves de API, senhas, tokens ou segredos. — *`GROQ_API_KEY` só em `.env` (gitignored); `.env.example` vazio.*
- [x] O repositório não contém dados pessoais sensíveis. — *Apenas tráfego sintético IEC-61850 (ERENO).*
- [x] O projeto usa .env.example para documentar variáveis de ambiente. — *`.env.example`.*
- [x] Quando usa dados externos, o projeto descreve origem e restrições de uso. — *Origem ERENO e modo cacheado documentados em [docs/reproducibility.md](docs/reproducibility.md).*

## Limitações conhecidas

- **Sem chave Groq / sem Java**, apenas o modo `demo` (replay golden) e os testes
  rodam; o loop real (`--engine live`) exige `GROQ_API_KEY` e conectividade.
- No **modo cacheado**, o mesmo `data/baseline_dataset.csv` é servido em todas as
  iterações — ele exercita o fluxo (agentes, memória, dashboard), mas **não**
  reproduz a variação física do ataque por configuração; para isso, use o modo
  `jar` com o gerador ERENO real.
- `MODEL_ID` não altera os resultados no modo `demo` (histórico golden fixo).
- O gerador ERENO (JAR) e o dataset benigno de entrada ficam **fora do repo**
  (gitignored, por peso); veja [docs/reproducibility.md](docs/reproducibility.md)
  para obtê-los/reconstruí-los.
- A qualidade do diagnóstico do Analista e das propostas do Estrategista depende do
  modelo Groq escolhido e pode variar entre execuções (não determinístico no live).
- O IDS de referência é um Random Forest simples; não representa um detector de
  produção, e sim um alvo controlado para o estudo adversarial.
- Uma campanha `--engine intent --rounds N` em **modo cacheado** para na rodada 2
  (`no_improvement`): a métrica-objetivo não se move porque o dataset é o mesmo em
  toda rodada. Use `--generator-mode jar` para uma campanha fisicamente mensurável.
  Veja [docs/feedback_policy.md](docs/feedback_policy.md).