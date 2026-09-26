# Fase 2 — comportamentos de ataque fora do catálogo

Plano vivo. Escrito em 26/09/2026, depois da tag `intent-driven-v0.2.0` (Fase 1,
valores ditados). Decisões desta rodada de planejamento marcadas com **[decidido
26/09]**.

## Objetivo

Hoje o usuário escolhe **qual** dos 11 ataques do ERENO, com **qual** efeito e
intensidade (Fase 0/1), e desde a Fase 1 pode **ditar valores exatos** dos
campos daquele ataque. O que ele ainda não pode é descrever um **comportamento
que não existe** entre os 11 — "descarte só as mensagens de trip e reenvie o
estado anterior 200 ms depois" não mapeia para nenhum ataque catalogado.

A Fase 2 entrega um **ataque programável**: um ataque genérico no ERENO dirigido
por **regras declarativas em JSON**. Um comportamento novo passa a ser **dado
validado, não código Java novo**.

## Guardrail (o mesmo da Fase 1, estendido)

O LLM emite as **regras** (dados tipados); Python determinístico e um schema no
Java decidem se elas são válidas e as executam. **Nenhum Java é escrito pelo
LLM.** É a extensão natural da regra central: na Fase 1 o LLM propunha um valor e
o catálogo decidia se ele existia; aqui o LLM propõe uma regra e o schema decide
se ela é executável. Um comportamento impossível **falha**, não é ajustado em
silêncio.

## Stack

Inalterada e assim permanece (definição do TCC): agentes em **Agno 2.7.2** sobre
**Groq**; núcleo determinístico em Python; gerador em Java (ERENO). A Fase 2 não
introduz framework de agente novo.

## A realidade dos dois repositórios

O trabalho Java é no repo irmão `../ERENO-2.0`, **branch `feat/journal`** (criada
da `dev-ian`, que é a branch de onde o JAR atual foi compilado — ver a nota de
memória `ereno-jar-origem`). O `ereno-ai` consome o JAR compilado em
`generator_runtime/ereno-generator.jar`.

**Disciplina de build (inegociável):** ao recompilar o JAR, comparar a saída
contra o JAR atual **antes** de trocá-lo em `generator_runtime/`. Os 11 ataques
existentes têm de sair idênticos byte a byte para as mesmas configs — os
resultados do piloto E0 e a byte-stability do golden dependem disso. A tag
`intent-driven-v0.2.0` é o ponto de retorno seguro no lado Python; no lado Java,
anotar o commit da `dev-ian` que gerou o JAR corrente.

## O que a investigação de 26/09 encontrou

Levantamento no fonte do ERENO (`feat/journal`) e no lado Python:

1. **A infraestrutura de seed já existe.** `ConfigLoader.RNG` é um
   `ThreadScopedRandom`; a config `randomSeed` chama `setSeed()`; e
   `IED.randomBetween` já sorteia por essa RNG. **Mas 6 creators furam a RNG com
   `Math.random()`** (22 ocorrências): uc01 random_replay, uc02 inverse_replay,
   uc03 masquerade_fault, uc05 injection, uc06 high_stnum, uc08 grayhole. É a
   causa do não determinismo que o E0 mediu (dois `content_hash` distintos para a
   mesma config). O conserto é pontual, não uma reconstrução.
2. **Acrescentar um ataque é mecânico, mas toca vários pontos.**
   - Java: um `case` em `CreateAttackDatasetAction.createAttackDevice` (é um
     `switch` fixo por `attackType`); uma classe IED + uma Creator (a interface é
     `MessageCreator.generate(IED, int)`); um rótulo em `Labels.LABELS` e um ramo
     em `getLabelForSegmentName` (mapeia segmento→rótulo por prefixo `ucXX` — o
     programável precisa de um prefixo livre, ex. `uc11`); um segmento em
     `config/actions/action_create_attack_dataset.json`.
   - Python: `AttackSpec` em `config/attacks_registry.py` (key, segment_name,
     label, description, intent_capability_id); baseline em
     `inputs/attacks/`; modelo congelado em `domain/attack_configs/`; módulo de
     capacidade em `config/capabilities/`.
3. **Superfície de mutação = o que `Goose` expõe.** Setters disponíveis:
   `setCbStatus`, `setStNum`, `setSqNum`, `setT`, `setGooseTimeAllowedtoLive`,
   `setSvPrefixCsv`, `setLabel`, `setConfRev`, `setPublisherTxTs`,
   `setSubscriberRxTs`, timestamp. A DSL de mutação mira esse conjunto.
4. **`AttackConfig` já lê JSON arbitrário** via `getRaw()` (`JsonObject` do Gson)
   — um Creator genérico pode ler uma lista de regras declarativas sem
   infraestrutura de parsing nova.
5. **Restrição de TPM.** O prompt do IntentAgent está em ~6,24k/8000 tokens. O
   ataque programável **não pode** enumerar campos no catálogo como os outros 11
   (foi o que estourou o TPM na Fase 1) — ele entra com uma descrição compacta da
   gramática de regras.

## Sub-fases

Cada uma é entregável e verificável isoladamente.

### 2.0 — Determinismo (fundação) **[ENTREGUE 26/09]**

Rotear os `Math.random()` dos creators pela `ConfigLoader.RNG`, e propagar
`IntentSpec.seed` → `GeneratorRunner` → `randomSeed` no action config →
`ConfigLoader.setSeed`. Antes, o `seed` da intenção existia no contrato mas não
chegava ao gerador.

- **Lado Java (`../ERENO-2.0`, `feat/journal`):** substituídos os 22
  `Math.random()` por `ConfigLoader.RNG.nextDouble()` em 5 creators (uc01, uc02,
  uc03, uc06, uc08). O uc05 já era determinístico (usa `randomSeed` próprio +
  `randomBetween`, que passa pela RNG). `CreateAttackDatasetAction` passou a ler
  um `randomSeed` do topo do action config e chamar `ConfigLoader.setSeed` antes
  de gerar.
- **Lado Python:** `GeneratorRunner` ganhou `random_seed` (default `None`,
  compatível) e grava/remove `randomSeed` no action config ao selecionar o
  segmento; `intent_loop._build_generator` passa `intent.seed` (baseline e
  variante da rodada compartilham a seed, comparação pareada).
- **DoD — verificado:** a mesma config (masquerade_fault) rodada duas vezes com
  `seed=123` deu **hash idêntico** (`dc67e1cb…`, 80019 linhas); com `seed=999`,
  hash diferente. Smoke test em 4 dos 5 ataques alterados (random_replay,
  masquerade_fault, high_stnum, grayhole) produz a classe de ataque normalmente.
  A distribuição é preservada por construção (`Math.random()` e
  `RNG.nextDouble()` são ambos uniformes em [0,1)); a diferença é só a fonte da
  RNG. 1476 testes Python passam.
- **Achado lateral:** `inverse_replay` (uc02) falha na geração (`returncode=3`) —
  **mas falha idêntico no JAR antigo (v2.0)**, então é bug pré-existente, fora do
  conjunto validado no E0, não regressão desta fase. Registrar como item à parte.
- **Pendências de fechamento do 2.0:** commitar o JAR novo em `generator_runtime`
  (gitignored — trocado localmente, backup em scratch); atualizar
  `docs/pilot_e0.md` (a seção "O gerador não é determinístico" agora tem
  ressalva: vale só sem `randomSeed`).

### 2.R — Modo de réplicas (varredura de seeds) **[ENTREGUE 26/09]**

**Uso:** `uv run adversarial-ids --engine intent --generator-mode jar --prompt
"..." --replicates 5` (varre `42..46`) ou `--seeds 42,101,777`. O LLM é chamado
uma vez; o resumo traz média ± desvio da métrica-objetivo.

**Verificado (JAR + Groq, 26/09):** 3 réplicas (seeds 42/43/44) → **1 chamada de
LLM**, três `content_hash` **distintos** (a seed chega ao gerador), as três
chegaram ao FEEDBACK. O `masquerade_fault` deu recall 1.0 nas três (métrica não
sensível para essa config; a variância aparece em configs evasivas). Correção
embutida: o consumo do LLM de intent agora conta só na rodada/réplica que de fato
chamou o LLM — antes, uma rodada que reusava a intenção somava de novo os tokens
(bug latente também no round 2+ de campanha). 1497 testes passam.

O texto abaixo é o desenho original.

---


Contrapartida da 2.0. A 2.0 tornou cada seed reprodutível; a 2.R usa isso para
rodar **N seeds do mesmo experimento** e reportar a métrica-objetivo como
**média ± desvio**, que é o que o TCC precisa para uma afirmação estatística (um
único trace é uma amostra, não uma medida). Independe da 2.1/2.2/2.3 — vale para
os 11 ataques atuais e para o programável.

- **Insight que a arquitetura já oferece (economia de TPM):** entre réplicas, só
  a **geração** muda; a **intenção** é a mesma (mesmo prompt → mesma
  `IntentSpec`). Então a réplica chama o LLM **uma vez** e reaproveita a
  `IntentSpec` compilada, variando só o `seed` passado ao `GeneratorRunner`. N
  réplicas custam **1** chamada de intent, não N — decisivo com o TPM de 8.000
  (ver `docs/target_values.md`/E0). É a mesma separação intenção↔geração que já
  sustenta a retomada e a política de feedback (o lado Red é determinístico da
  rodada 2 em diante).
- **Onde vive:** um controlador fino que, dada uma `IntentSpec` já compilada,
  itera sobre uma lista de seeds e chama o núcleo determinístico
  (`GeneratorRunner` → `IdsEvaluator` → `DetectionReport`) por seed. Cada réplica
  é um `LoopRecord` próprio (o `seed` já é campo dele), ligadas por um id de
  lote. A agregação (média ± desvio da métrica-objetivo, e o intervalo) lê o
  ledger — nada de novo contrato de "resultado agregado" antes de precisar.
- **Interface:** `--replicates N` (varre `seed, seed+1, …`) ou `--seeds a,b,c`
  (lista explícita) no `--engine intent`; alternativamente um
  `scripts/run_replicates.py` para manter a CLI enxuta. Decidir na 2.R.
- **Relação com a campanha (E10):** uma réplica é **uma rodada** repetida sob
  seeds diferentes — não uma campanha multi-rodada. As duas são ortogonais:
  pode-se replicar cada rodada de uma campanha, mas o primeiro corte replica só
  a rodada 1 (a intenção do prompt), que é o caso do artigo.
- **DoD:** `--replicates 5` sobre um ataque JAR-mensurável produz 5
  `LoopRecord` com seeds distintos, **uma** chamada de LLM no lote, e um resumo
  com média ± desvio da métrica-objetivo. Reexecutar o lote com a mesma lista de
  seeds reproduz os mesmos números (a 2.0 garante isso).
- **Cuidado:** só faz sentido em `--generator-mode jar`. Em cached, todo trace é
  o mesmo arquivo e as réplicas colapsam — o modo deve avisar, como o piloto já
  faz.

### 2.1 — Fatia vertical: ataque programável, DSL só de mutação **[ENTREGUE 26/09]**

Ver `docs/programmable_attack.md`. Entregue ponta a ponta: `attack_key
"programmable"` (uc11) no Java (creator/IED lendo `rules` como dict de slots +
label + switch) e no Python (AttackSpec, baseline, schema, capacidade, playbook).
O compilador materializa as regras **sem código novo**: reusa o `target_values`
da Fase 1 — a capacidade expõe `rules.rN.op/field/value` como campos author-only
(efeito vazio, nunca varridos pela heurística) e `rules.rN.fraction` como a
alavanca de intensidade. **Verificado com LLM real:** "multiplique o sqNum por 3
em ~40%" → o LLM escolheu `programmable` e autorou a regra; o JAR gerou a classe
`programmable`. 1537 testes passam. Escopo: só mutação (seleção/temporização é
2.2). Nuance documentada: o baseline traz 2 slots com regra default.

O texto abaixo é o desenho original.

---


Um `attack_key = "programmable"` novo, com **um** tipo de regra: mutação de um
campo GOOSE (`set` / `add` / `scale`) numa fração das mensagens. Ligado ponta a
ponta.

- **Java:** `uc11` (prefixo livre) — `ProgrammableIEDC` + `ProgrammableCreatorC`
  que lê `rules: [{op, field, value, fraction}]` de `AttackConfig.getRaw()`,
  aplica via os setters de `Goose`, e valida cada regra contra um allowlist de
  campos mutáveis (o schema Java); rótulo `programmable` em `Labels`; case no
  switch; segmento `uc11_programmable` no action config.
- **Python:** `AttackSpec("programmable", "uc11_programmable", "programmable",
  ...)`; baseline mínima em `inputs/`; `domain/attack_configs/programmable.py`
  (modelo congelado da lista de regras); `config/capabilities/programmable.py`
  (a gramática de regras como capacidade — quais campos, quais ops, faixas);
  o compilador `intent_compiler.py` aprende a materializar regras (não só
  ajustar campos escalares).
- **DoD:** um prompt que descreve um comportamento que **não** é um dos 11 (ex.
  "duplique 10% das mensagens de trip com o stNum incrementado") compila para
  regras, gera um trace, passa os gates do E4 e produz um `DetectionReport`. O
  loop fecha.

### 2.2 — Alargar a DSL (fora do escopo do primeiro corte)

Regras de **seleção** (probabilidade, condição sobre `stNum`/`cbStatus`) e de
**temporização** (atraso, descarte, duplicação, reordenação). Cada tipo entra
como dado, validado por schema nas duas pontas. Só depois de 2.1 provar o loop.

### 2.3 — Prompt/TPM, validação e release

- Entrada compacta do programável no catálogo do IntentAgent (a gramática, não a
  enumeração de campos) — orçar tokens desde o 2.1.
- Golden prompts do ataque programável (`tests/golden_prompts/`).
- Validação com LLM real (`openai/gpt-oss-120b`), espaçada pelo TPM.
- `docs/programmable_attack.md`; tag `intent-driven-v0.3.0`.

## Riscos

- **Escopo da DSL.** Geral demais é impossível de validar e de caber no prompt;
  estreito demais não entrega "comportamentos novos". Mitigação: fatia vertical
  2.1 com um só tipo de regra. **[decidido 26/09]**
- **TPM.** Cada tipo de regra paga tokens no prompt do IntentAgent (~6,24k/8000
  hoje). Orçar por incremento; a compressão de efeitos da Fase 1 comprou folga,
  mas ela é finita.
- **Regressão dos 11 ataques ao recompilar.** Mitigação: comparação byte a byte
  do JAR antes de trocar, os 11 idênticos.
- **Rótulo/label novo desalinhado entre Java e Python** (índice em `Labels` vs.
  `AttackSpec.label`). Mitigação: um teste que casa os dois.

## Estado

- Fase 0/1 entregues; tags `intent-driven-v0.1.0` (loop + retomada) e
  `intent-driven-v0.2.0` (valores ditados). 1476 testes passam sem Groq/Java.
- Branch Java `feat/journal` criada da `dev-ian`; configs locais da dev-ian em
  `git stash` (ver memória `ereno-jar-origem`).
- **2.0 (determinismo) ENTREGUE** — seed reprodutível verificada; JAR recompilado
  e trocado em `generator_runtime`.
- **2.R (modo de réplicas) ENTREGUE** — `--replicates`/`--seeds`, 1 chamada de LLM
  por lote, verificado com JAR + Groq.
- **2.1 (ataque programável, DSL de mutação) ENTREGUE** — `programmable` (uc11)
  ponta a ponta, autoração de regras via `target_values`, verificado com JAR +
  Groq. Ver `docs/programmable_attack.md`.
- Próximo passo: **2.2 (alargar a DSL: seleção + temporização)**.
