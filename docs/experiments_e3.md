# E3 — Defesa aplicada e corrida armamentista (RQ3): protocolo

Fixado em 20/10/2026, **antes** de qualquer execução e enquanto o B3/B4 do E1
ainda roda; as cinco decisões abertas do rascunho foram fechadas pela proposta
(ver "Decisões"). Vale a mesma regra do E1: mudar um item depois de ver um
número de teste invalida a comparação, e mudanças pós-piloto ficam registradas
na seção "Piloto".

## Pergunta

O E1 mede só o atacante, contra um detector congelado. O E3 fecha o loop: a
cada rodada o atacante procura evasão contra o detector atual, e a defesa é
**aplicada** — o detector é retreinado com o que o atacante achou.

- **RQ3a (a defesa fecha a brecha?):** a variante que evadiu o detector da
  rodada *r* deixa de evadir o detector da rodada *r+1*?
- **RQ3b (a defesa acumula?):** o detector da última rodada continua pegando as
  variantes das rodadas anteriores (sem esquecimento)?
- **RQ3c (quem ganha a corrida?):** a evasão que o atacante consegue contra o
  detector atual cai ao longo das rodadas?
- **RQ3d (quanto custa endurecer?):** o detector endurecido perde desempenho no
  ataque de referência ou no tráfego normal?

## Desenho

Uma **linhagem** = (família, braço do atacante, repetição). Cada linhagem é
independente: o D₁ de uma repetição depende do A₀ dela.

```
D₀ = detector congelado do E1 (mesmo treino, mesmos limiares)
para r = 0 … R−1:
    campanha do atacante contra D_r (congelado durante a campanha)  → A_r
    D_{r+1} = retreino com referência + variantes de A_0 … A_r
              + recalibração dos limiares no normal de calibração
avaliação final: matriz cruzada D_i × A_j nas seeds de teste
```

- **R = 4** rodadas de ataque (A₀…A₃) e 5 detectores (D₀…D₄).
- **A rodada 0 é o E1.** D₀ é o `FrozenIds` do E1 e a campanha da rodada 0 é a
  campanha do E1 do mesmo braço e repetição (mesmas seeds, mesmo detector,
  mesmo orçamento). Ela é reaproveitada do ledger, não rodada de novo: o E3
  só paga as rodadas 1…R−1.
- Durante uma campanha o detector não muda — dentro da rodada, a evasão é medida
  contra um alvo fixo, exatamente como no E1.

## A defesa aplicada

**Treinamento adversarial**, determinístico — é a única técnica do
catálogo do Defensor (`config/defense_techniques.py`, `detector_retraining`) que
se executa sem intervenção humana; regras de rede, segmentação ou autenticação
GOOSE não têm como ser aplicadas a um RF de forma automática e honesta.

- **Quais variantes entram:** os **k = 3** candidatos válidos de maior objetivo de
  busca da rodada *r* (o selecionado e os dois seguintes; menos, se a campanha
  teve menos válidos; as 20 campanhas do B1 no E1 têm ≥ 3 cada). Só candidatos que passaram pelo oráculo
  (λ = 0,5): uma variante que "evade porque desapareceu" não ensina nada à defesa.
- **Em que seeds:** cada variante é regenerada nas **seeds de treino 101–103** —
  nunca nas de busca ou teste. Nenhuma mensagem que o atacante viu ou que mede o
  resultado entra no treino.
- **Só as linhas de ataque.** O benigno é função da seed (ERENO `c36bee2`): o
  normal do trace da variante na seed 101 é idêntico ao da referência na 101.
  Adicioná-lo duplicaria o normal a cada variante; entram só as linhas rotuladas
  como ataque. Isso também segura a memória do treino, que cresceria ~3× por
  rodada com os traces inteiros (a máquina de 15,6 GB já matou processo no E1).
- **Acumulado:** D_{r+1} vê a referência e as variantes de **todas** as rodadas
  até *r*. Sem isso, "esquecimento" seria um efeito do desenho, não do detector.
- **Pipeline fixo:** RF, `protocol_features="deltas"`, preprocessamento E6, sem
  seleção nem undersampling — o mesmo do E1. Só os **dados** de treino mudam.
  Isso é o guardrail do projeto aplicado ao E3: nada que olha a métrica escolhe
  preprocessamento, limiar ou features (`core/undersampling_policy.py`).
- **Recalibração:** os limiares de D_{r+1} são recalibrados no normal das seeds
  de calibração 201–202, pela mesma regra do E1. É o ponto de operação do
  defensor, não um ajuste à variante: a calibração nunca vê ataque.

### O papel do Defensor (agente)

Para o selecionado de cada rodada, o `DefenderAgent` (`gpt-oss-20b`, como no
B4) produz um `DefensePlan` sobre o `DetectionReport` dele contra D_r nas seeds
de busca (`candidate_report` do E1), passando pelos portões do E5. O plano **não
decide** o retreino (que é fixo acima); ele é registrado e medido:

- o plano recomendou retreino? Conta `detector_retraining` ou
  `dataset_enrichment`, as duas técnicas do catálogo que descrevem a defesa
  aplicada (concordância com a defesa);
- as features que ele cita como evidência estão entre as 5 que mais mudam de
  importância (Gini, em módulo) entre D_r e D_{r+1}? (fração das citadas;
  liga com o E4, SHAP vs. features citadas).

Descartado: o plano escolher *quais* candidatos entram no retreino. Ficaria
mais "multi-agente", mas põe um LLM dentro do treino do detector — o que o
guardrail central do projeto evita (o LLM só produz intenção ou plano). Os
planos são um passo à parte (`run_e3.py plans`), depois das campanhas: não
interferem na corrida e gastam a cota do 20b, não a do atacante.

## Braços do atacante

| Braço | Atacante por rodada | Famílias | Repetições | Custo |
|---|---|---|---|---|
| E3-B1 | busca aleatória (B1 do E1) | as 4 | 5 | sem tokens |
| E3-LLM | o vencedor do E1 entre B3 e B4 → **B3, recortado em 22/10** | 2: `masquerade_fault`, `high_stnum` → **`high_stnum`** | 5 → **2** | Groq |

- **E3-B1** é o braço principal: barato, completo, e responde RQ3a–d sozinho.
- **Famílias do E3-LLM**, fixadas antes do B3/B4 e só pelo B1 do E1: a de maior
  evasão suave (`masquerade_fault`, 0,515) e a de maior evasão@10
  (`high_stnum`, 0,189). Duas famílias cabem em ~4–5 dias de cota; as quatro
  pediriam ~8–9. O E3-LLM é lido família a família ao lado do E3-B1.
- **E3-LLM** responde se um atacante adaptativo **reage** à defesa. Regra fixada
  agora, antes do resultado do E1: entra B4 se B4−B3 "separa" a favor do B4 no
  agregado do E1; senão B3 (mais barato). O mínimo de 5 repetições do E1 vale aqui.
- **Memória entre rodadas (E3-LLM):** no início da rodada *r*, os selecionados
  das rodadas anteriores são re-medidos contra D_r nas seeds de busca e entram no
  histórico do atacante ("isto funcionava e foi corrigido"). É o que deixa o
  atacante aprender com a defesa. O E3-B1 não tem memória por construção.
- Orçamento por rodada: 10 propostas, a mesma caixa e a mesma cadeia de
  validade do E1.

## Controles

- **Sem defesa:** o atacante continua contra D₀ pelo mesmo número de rodadas
  (é rodar as campanhas extra contra o detector do E1). Separa "o atacante
  melhorou porque a defesa mudou" de "o atacante melhorou porque teve mais
  consultas". Só no E3-B1, e lá sai quase de graça: a busca aleatória não lê
  histórico, então as propostas da linhagem sem defesa são **as mesmas** da
  defendida (mesma RNG por família, repetição, rodada e índice) e os traces de
  busca já estão em cache — muda só o detector que as pontua. No E3-LLM o
  controle dobraria a cota; a comparação lá é com a rodada 0 e com o E3-B1 da
  mesma família.
- **Retreino placebo:** D'_{r+1} recebe tantos traces quanto D_{r+1} recebeu de
  variantes, mas da config de referência em seeds de treino extras (111, 112,
  …, até 146 com R = 4), e as linhas de ataque são amostradas (semente fixa)
  até o **mesmo número** que D_{r+1} recebeu. Separa "o detector melhorou
  porque viu as variantes" de "o detector melhorou porque viu mais dado". Só no
  E3-B1, e só avaliado (entra na matriz), nunca atacado: responde o fechamento
  (RQ3a) e o custo (RQ3d), não a corrida.

## Medidas

Tudo nas **seeds de teste 401–405**, nenhuma vista por atacante ou retreino.
Métrica primária: **evasão suave** (a mesma do E1). Evasão@10 e AP ao lado.

- **Matriz cruzada** `E[i][j]` = evasão suave de A_j contra D_i, para
  i = 0…R e j = 0…R−1. É a figura principal do E3.
- **RQ3a, fechamento:** `E[r][r] − E[r+1][r]` por rodada (a brecha que a
  defesa fechou).
- **RQ3b, esquecimento:** `E[R][j] − E[j+1][j]` para cada j < R−1
  (positivo = o detector final voltou a deixar passar o que já tinha corrigido).
- **RQ3c, corrida:** a sequência `E[r][r]`, r = 0…R−1 — o que o atacante
  consegue contra o detector do momento. Comparação pré-definida: rodada R−1
  contra rodada 0, e contra o controle sem defesa na mesma rodada.
- **RQ3d, custo:** no ponto primário (10 alarmes/10k), recall no ataque de
  referência e alarmes falsos por 10k no normal de teste, para D₀…D_R. Aqui
  entram também **precisão, recall e F1** do ataque de referência: a
  prevalência da referência é fixa, então o F1 compara detectores sem a
  distorção que tem na comparação entre variantes (ver E1, "Medidas").
  **Secundário, registrado depois do piloto** (ver "Depois do piloto"): alarmes
  falsos por 10k no normal **do trace da variante**, `F[r+1][r] − F[r][r]`
  (`trace_alarm_cost`), e o mesmo contra o placebo.
- **Defensor:** fração de planos aceitos pelo portão, fração que recomenda
  `detector_retraining`, sobreposição entre features citadas e a mudança de
  importância (top-5).
- **Custo:** tokens, chamadas, tempo de geração e de retreino por rodada.

**Comparações** (pareadas por família × repetição, bootstrap de 1000
reamostragens estratificado por família, IC 95% percentil, "separa" só se
exclui zero — o mesmo do E1): fechamento médio; `E[R−1][R−1] − E[0][0]`;
`E[R−1][R−1]` com defesa − sem defesa; esquecimento médio; recall de
referência D_R − D₀. Cada família reportada ao lado do agregado.

## Resultados possíveis e o que cada um diz

- Fechamento > 0 e corrida caindo: a defesa endurece e o atacante acha cada vez
  menos — a co-evolução favorece a defesa.
- Fechamento > 0 mas corrida estável: a defesa corrige o que vê, o atacante acha
  outra brecha a cada rodada — "whack-a-mole", e o catálogo de cada família
  diz quantas brechas há.
- Fechamento ≈ 0: retreino não basta para essa família (as features não separam
  a variante) — evidência para outra defesa, não para mais rodadas.
- Esquecimento > 0: retreinar com o novo empurra para fora o antigo.

Os quatro são publicáveis; nenhum é "o experimento falhou".

## Custo estimado

- Proposta: ~35–80 s (3 gerações + oráculo, medido no E1). Retreino: 3 traces de
  referência + as linhas de ataque acumuladas, por linhagem e rodada.
- **E3-B1** (3 rodadas novas × 4 famílias × 5 reps × 10 propostas = 600
  propostas): ~12–27 h de geração, sem tokens (o piloto de 22/10 dá ~25–30 h
  com as matrizes; ver "Depois do piloto"). O controle sem defesa reusa os
  traces de busca (mesmas propostas): custa o oráculo e a pontuação, não a
  geração. O retreino gera 9 traces por rodada e linhagem (3 variantes × 3
  seeds de treino); o placebo, 36 traces de referência por família (seeds
  111–146, compartilhados entre as repetições). Roda à noite, como o B1/B2 do
  E1, com `JAVA_TOOL_OPTIONS=-Xmx2g`.
- **Planos do Defensor:** um por rodada e linhagem defendida — 80 no E3-B1
  (4 × 5 × 4) e 40 no E3-LLM, ~5k tokens cada no `gpt-oss-20b`.
- **E3-LLM** com B3 (~28k tokens por campanha no `gpt-oss-120b`): 2 famílias ×
  5 reps × 3 rodadas = 30 campanhas ≈ 840k tokens ≈ 4–5 dias de cota;
  4 famílias ≈ 8–9 dias. Com B4 (~38k no 120b + ~27k no `gpt-oss-20b` do
  Defensor, por campanha) a conta no 120b sobe ~35%.
- Só começa **depois** do B3/B4 do E1 (mesmo runtime do ERENO e mesma cota).

## Implementação

- `experiments/arms_race.py`:
  - `Lineage` e `DetectorLineage`: os ledgers da linhagem, `D_r`, o placebo `D'_r` e a memória do atacante.
  - `top_variants`.
  - A matriz: `score_detector` e `matrix_values`.
  - As medidas: `race_measures`, `placebo_closure`, `reference_cost` e `trace_alarm_cost`.
  - `saturated_budgets`: lê como NaN o orçamento sem ponto de operação (`metrics.is_saturated`).
  - As do Defensor: `importance_shift` e `plan_measures`.
  - Nenhum modelo é persistido, porque retreinar `D_r` é determinístico.
- `FrozenIds.extra_attack` / `IdsEvaluator.fit_traces(attack_only_paths=…)`:
  linhas só de ataque no treino, com teto opcional (placebo). Vazio = o E1,
  inalterado.
- `ProposalContext.race_round`/`prior`: a rodada 0 sorteia exatamente o que o
  E1 sorteou; as seguintes sorteiam de novo. O atacante LLM vê as rodadas
  anteriores numa seção própria da mensagem, que não existe no E1.
- `scripts/run_e3.py` tem os comandos `campaign`, `matrix`, `plans` (os três retomáveis, com código 3 em cota esgotada) e `report`. Saídas em `outputs/experiments/e3/`:
  - `ledger/<família>/<linhagem>/rep<r>/round<k>.jsonl`
  - `ids/…/D<k>.json`
  - `results/…/round<k>.json`
  - `matrix/<família>/<linhagem>/rep<r>.json`
  - `plans/…/round<k>.json`
  - `results.json`
- Testes em `tests/test_experiment_arms_race.py` (sem Java nem Groq).

## Piloto

1 família (`high_stnum`, a que mais evadiu no B1 do E1), braço E3-B1, 1
repetição, R = 2. Mede tempo de retreino e memória, confere que D₁ fecha a
brecha de A₀ (se nem isso acontecer, o problema é o desenho e não a família) e
que o placebo é distinguível. Não entra nos resultados.

## Depois do piloto (22/10/2026)

O piloto rodou em 22/10 (`outputs/experiments/e3/pilot/`). O desenho
funcionou: D₁ fecha A₀ (evasão suave de 0,186 para 0,0004 no teste) e o
placebo do mesmo tamanho não fecha (0,212), então o ganho vem das variantes,
não de mais dado. Contra D₁, o atacante B1 só chegou a 0,027. Custo na
referência no ponto de 10/10k: recall de 0,993 para 0,984 e alarmes de 7,7
para 14,1 por 10k (D₀ → D₂). Tempo: ~5,5 min por rodada de campanha e ~19 min
para a matriz (5 detectores, até 1,2M linhas), sem estourar a memória. Isso dá
~25–30 h para o E3-B1 completo.

O piloto mostrou duas coisas que o protocolo não previa. As duas foram
registradas **antes** das campanhas do E3-B1 e do E3-LLM:

1. **Ponto de operação saturado vira NaN, não evasão 1,0.** O retreino só
   soma linhas de ataque. Com isso, o desequilíbrio do treino cresce a cada
   rodada (1:1, 4:1, 7:1 no piloto, ~13:1 em D₄), e o limiar de calibração
   sobe junto: em 1/10k ele foi de 0,85 a 0,93 e, no D₂, chegou a 1,00, com
   **zero alarmes possíveis**. A "evasão 1,0" desse ponto vem do limiar, não do
   ataque.
   - **Regra:** quando a taxa de alarmes na calibração é 0
     (`metrics.is_saturated`), as taxas daquele orçamento valem NaN e o bloco
     leva `saturated: true`. As contagens ficam.
   - **Onde vale:**
     - na matriz e no `report` (`saturated_budgets`; uma matriz gravada antes
       da correção é corrigida na leitura);
     - no custo na referência;
     - no histórico que o atacante LLM lê (aparece "n/d", e não "1.000", que
       faria o atacante achar que evadiu tudo);
     - no passo `plans`: se o ponto primário saturou, o Defensor não recebe um
       relatório com recall 0. A rodada fica fora da medida do Defensor, sem
       contar como plano recusado.
   - **O que não muda:** a evasão suave, métrica primária, não usa limiar.
     Nenhuma campanha, seleção ou oráculo muda.
   - **Ressalva:** se o ponto de 10/10k saturar em D₃ ou D₄, a evasão@10 e o
     RQ3d daquela linhagem e rodada ficam sem valor (NaN). Isso é reportado
     como limite do retreino que só soma ataque, não escondido.
   - **Descartado:** balancear o treino (undersampling ou peso de classe).
     Mudaria o pipeline do detector entre rodadas, o que o guardrail do E3
     proíbe (só os **dados** mudam).

2. **Custo de alarme no trace da variante (RQ3d, secundário).** Os deltas de
   uma mensagem normal (`stDiff`, `sqDiff`, `timestampDiff`…) são calculados
   contra a mensagem anterior do fluxo, que pode ser de ataque. Assim, um
   detector pode "fechar" uma variante alarmando também o normal em volta
   dela. No piloto, no ponto de 10/10k e no normal dos traces de A₀:
   - D₀ alarmava **7%** (704/10k);
   - D₁ alarma **20%** (1986/10k);
   - o placebo D'₁ alarma 5%.

   Parte do fechamento, portanto, é pago com alarme no contexto do ataque, e o
   custo medido no normal da referência (RQ3d) não enxerga isso. Passa a ser
   reportado, sem mudar nada na corrida:
   - `trace_alarm_cost` = `F[r+1][r] − F[r][r]`, com F = alarmes falsos/10k no
     normal do trace de A_r;
   - a diferença contra o placebo, com o mesmo bootstrap pareado dos demais.

   Fica **secundário**: o primário do RQ3d continua o normal da referência, que
   é o normal "de verdade" do operador. Este mede o que o retreino custa perto
   do ataque.

**Desvio no E3-LLM (22/10/2026).** A primeira campanha B3 × high_stnum carregou
`metrics.py` um minuto antes da regra 1 entrar. Em três rodadas o atacante viu
"evasão@1 = 1,000" em todas as propostas: rep1 r3, rep2 r2 e rep2 r3, todas
com o ponto de 1/10k saturado. Essas rodadas foram arquivadas com o sufixo
`.bak-20261022-saturado`, junto com o cartão D3 da rep2, e refeitas com o
código novo. Nos ledgers refeitos o ponto de 1/10k é NaN, e o atacante vê "n/d".
As rodadas 1 (e a 2 da rep1) não tinham ponto saturado e ficaram como estavam.
Evasão suave no teste, antes e depois:
- rep1 r3: 0,017 → 0,006;
- rep2 r2: 0,038 → 0,029.

## Decisões (20/10/2026)

O usuário delegou as cinco à proposta do rascunho:

1. Defesa aplicada = treinamento adversarial com k = 3 variantes por rodada.
2. O plano do Defensor só é registrado e medido; não escolhe variantes.
3. R = 4 rodadas.
4. E3-LLM em 2 famílias (`masquerade_fault`, `high_stnum`), pela regra acima.
5. O placebo entra, no E3-B1, só avaliado.

## Recorte do E3-LLM (22/10/2026)

O prazo do artigo não comporta os 4–5 dias de cota do E3-LLM completo, e o B4
do E1 foi recortado no mesmo dia (`docs/experiments_e1.md`, "Custo e cota").
Decidido antes de qualquer resultado do B4 e de qualquer rodada do E3:

- **Braço B3, fixado agora**, em vez de esperar a regra do B4−B3. É uma
  quebra da regra de 20/10, e por isso fica registrada: a escolha é por custo
  (~28k tokens por campanha contra ~65k do B4), não por resultado, que ainda
  não existe.
- **1 família, `high_stnum`** (a de maior evasão@10 no B1 do E1), **2
  repetições** (reps 1 e 2) e R = 4. A rodada 0 é o B3 do E1 dessas reps. São
  6 campanhas novas, ~170k tokens no 120b (~1 dia de cota), mais 8 planos do
  Defensor no 20b.
- **Leitura: estudo de caso, não estatística.** Com 2 linhagens, o agregado
  pareado do `report` não tem poder. O E3-LLM é lido pelas matrizes D_i × A_j
  de cada linhagem, ao lado do E3-B1 de `high_stnum` nas mesmas reps, e
  responde descritivamente se o atacante LLM reage à defesa (a memória entre
  rodadas). RQ3a–d com intervalo de confiança vêm só do E3-B1.
- Sem controle sem defesa, como já era no E3-LLM.

Comandos (depois do B4 do E1, nunca junto com outra campanha):

```bash
uv run python scripts/run_e3.py campaign --arm B3 --family high_stnum --reps 1 2
uv run python scripts/run_e3.py matrix   --arm B3 --family high_stnum --reps 1 2
uv run python scripts/run_e3.py plans    --arm B3 --family high_stnum --reps 1 2
```
