# E1 — Busca e feedback (RQ1–RQ2): protocolo

Fixado em 14/10/2026, **antes** das execuções. Mudar qualquer item abaixo depois
de ver um resultado de teste invalida a comparação; mudanças feitas após o
piloto (que não entra nos resultados) ficam registradas na seção "Piloto".

## Pergunta

- **RQ1:** com o mesmo orçamento de propostas, um injetor adaptativo (LLM) acha
  evasão válida maior que a configuração de referência, a busca aleatória e a
  política determinística?
- **RQ2:** dar ao injetor o plano do Defensor, além das métricas, melhora a busca?

## Condições

| Id | Estratégia | Quem propõe o próximo candidato | LLM |
|----|-----------|--------------------------------|-----|
| B0 | Referência | ninguém — a baseline da família, sem busca | não |
| B1 | Busca aleatória | sorteio uniforme na caixa de busca (abaixo) | não |
| B2 | Controle determinístico em malha aberta | a escada de `core/feedback_policy.py` (intensidade low→medium→high, depois +1 campo), sem a regra de platô | não |
| B3 | Atacante com métricas | LLM vê catálogo da família, caixa, histórico de candidatos e métricas de busca | sim |
| B4 | Atacante com feedback completo | B3 + o último `DefensePlan` validado do Defensor | sim (2 chamadas/rodada) |

B0 é o ponto de partida, não um competidor com o mesmo número de consultas.
Sem a regra de platô o B2 **não olha métrica nenhuma**: é um cronograma fixo
de alavancas, e o artigo o descreve assim — controle determinístico em malha
aberta, não "política com feedback". B2 roda **sem** a regra de platô da política (`no_improvement`): ela economiza
rodadas no loop de produção, mas aqui cortaria o orçamento do B2 e a comparação
deixaria de ser a orçamento igual. A escada esgotada encerra a campanha mais
cedo (o que é registrado); a comparação usa o melhor resultado até cada
orçamento consumido.

B3 e B4 usam o **mesmo** modelo, temperatura, prompt-base, ferramenta
(`submit_intent_spec`) e portões; a única diferença é o bloco do plano de defesa.
Temperatura: **0,7** para os dois (é ela que dá variação entre repetições).
O LLM continua produzindo só a intenção — compilação, caixa e validade são
determinísticas, como no pipeline.

## Fixos

- **Famílias:** `masquerade_fault`, `random_replay`, `inverse_replay`,
  `high_stnum` (catálogos `.v2` pós-E0). `injection` saiu em 15/10, depois das
  referências e antes de qualquer campanha: teto de evasão (ver
  "Referências"). Se o piloto mostrar que a cota da Groq não comporta as
  quatro, corta-se para três famílias —
  nunca para menos de 5 repetições, que é o mínimo para o bootstrap por
  campanha dizer alguma coisa. `grayhole` fica fora
  do E1 (caso-limite, paper próprio).
- **Detector:** Random Forest, `protocol_features="deltas"`, preprocessamento E6,
  sem seleção nem undersampling. Treinado uma vez por família nas seeds de
  treino com a config de referência e **congelado** (`experiments/protocol.py::FrozenIds`).
- **Seeds:** treino 101–103, calibração 201–202, busca 301–303, teste 401–405
  (`SeedPlan`). Três seeds de busca, não duas: o objetivo de busca é uma média
  por seed, e com duas o ruído entre seeds compete com a diferença entre
  candidatos. Nenhuma estratégia vê um número de teste.
- **Orçamentos de alarme:** 1, 10 e 100 alarmes falsos por 10 mil mensagens
  normais, calibrados no normal das seeds de calibração. **Ponto primário: 10.**
- **Orçamento de busca:** 10 propostas por campanha. Uma proposta inválida
  (não compila, sai da caixa, falha na geração, piso de volume, ataque
  indetectável pelo oráculo) **conta** no orçamento.
- **Repetições:** 5 por (família, estratégia). Repetição `r` usa `seed=r` no
  sorteio do B1 e na `IntentSpec` do B2 (que embaralha a ordem dos campos); no
  B3/B4 a variação vem da amostragem do LLM.

## Caixa de busca

O mesmo espaço para todas as estratégias:

- os campos editáveis do catálogo da família (todos, não só os de evasão);
- os limites do catálogo; um lado sem limite no catálogo é limitado a
  um **limite físico** (aprovado em 15/10; escalas medidas no benigno das
  seeds de treino 101–103, `scripts/run_e1.py::PHYSICAL_BOUNDS`). Só afeta
  `masquerade_fault`:
  - `fault.durationMs` ≤ **1000 ms**. No gerador o campo não é a duração da
    falha: é um atraso somado ao timestamp da cópia. O heartbeat do
    publicador chega a 1,031 s; um atraso maior faria a cópia cruzar a
    próxima mensagem legítima (reordenação, outro ataque);
  - `analog.deltaAbs` ≤ **10**: a maior |corrente| do benigno (10,31), das
    faltas reais simuladas. O gerador aplica o mesmo desvio a corrente e a
    tensão (|v| ≤ 0,14), acoplamento que a caixa não separa — limitação;
  - `trapArea.multiplier` ≤ **2,5**, o máximo da referência. Faltas reais
    movem as somas de trapézio 0,71–1,41× a mediana; a referência já
    excede isso, e a caixa não estica além dela.

  Sem limite físico, um lado aberto seria 4 × a referência (do `max` do par);
- no máximo **4** campos alterados por candidato;
- pares `{min,max}` com `min < max` estrito (o ERENO recusa `min == max`).

Uma proposta fora da caixa é inválida (`out_of_box`).

## Validade de um candidato

Em ordem; a primeira falha dá o motivo:

1. compila (`compile_attack_candidate`) e cabe na caixa;
2. gera nas seeds de busca;
3. piso de volume: ≥ 50 linhas de ataque e prevalência ≥ 1% em cada seed
   (o mesmo piso do E4);
4. **o ataque continua detectável por um oráculo**: um RF com o mesmo
   preprocessamento, treinado no trace do próprio candidato na seed 301 e
   avaliado na 302, precisa de `AP ≥ p + λ·(1 − p)`, com `p` a prevalência
   do ataque na 302. **λ = 0,5**, fixado em 15/10 a partir das referências
   (as quatro famílias do E1 têm λ observado ≥ 0,9998; ver "Referências").
   Se alguma referência não separar do normal nem com o
   oráculo, o problema é do critério, não da família — e o critério é revisto
   antes de qualquer execução.

O item 4 é a defesa contra "evasão por desaparecimento": uma config que apaga o
ataque (ex.: `fault.prob = 0`) evade qualquer detector sem expor ponto cego
nenhum. Evasão do detector congelado só conta se um detector que **viu** a
variante ainda a separa do normal. A referência de cada família é checada pelo
mesmo critério; se a referência falhar, a família sai do E1 com o motivo.

## Medidas

- **Objetivo de busca** (o que B2–B4 veem e o que seleciona): **evasão
  suave** = 1 − score médio do detector congelado nas mensagens de ataque,
  média das três seeds de busca. Desempate: menor AP médio, depois a proposta
  mais antiga. *Mudado em 15/10, depois do piloto e antes de qualquer execução
  principal* (ver "Piloto"); até então era a evasão no orçamento de 10.
- **Seleção e teste:** ao fim da campanha, o melhor candidato válido é
  regenerado nas 5 seeds de teste e medido contra o detector congelado. Campanha
  sem candidato válido registra teste ausente (e entra na taxa de validade).
- **Por campanha (unidade de análise):** evasão suave de teste (métrica
  primária) e evasão de teste nos três orçamentos (média das 5 seeds), AP,
  alarmes falsos por 10k, validade (fração das propostas feitas — uma escada
  esgotada não propôs nada inválido), tokens, chamadas de LLM, tempo.
- **Curva:** melhor evasão suave de busca até a consulta *k*, por estratégia e
  família.
- **Comparações pré-definidas** (pareadas por família × repetição, bootstrap
  de 1000 reamostragens de campanhas, IC 95% percentil; "separa" só se o IC
  exclui zero): B3−B1, B3−B2, **B4−B3** (RQ2), B1−B0 e B2−B0 como contexto.
  O bootstrap do agregado é **estratificado por família** (reamostra
  repetições dentro de cada família): as famílias têm espaços de busca muito
  diferentes (`injection` tem 2 campos, `masquerade_fault` 11), e uma
  reamostragem livre deixaria o agregado depender de quantas vezes cada
  família foi sorteada. Cada família é reportada ao lado do agregado.

## O que o LLM vê

- **B3:** catálogo da família (mesma renderização do `IntentAgent`, só a
  família), a caixa de busca, e o histórico das propostas anteriores da
  campanha: campos/valores alterados, validade (e motivo) e as métricas de
  busca (evasão nos três orçamentos, AP).
- **B4:** tudo do B3 + o último `DefensePlan` que passou pelo portão do E5,
  produzido pelo `DefenderAgent` sobre um `DetectionReport` do último candidato
  válido. Plano recusado pelo portão = rodada sem plano, registrada.

### A evidência do Defensor muda com o candidato

O `DetectionReport` do B4 traz as contagens no ponto primário, somadas nas
seeds de busca, e como `top_features` o **SHAP médio absoluto da classe de
ataque calculado só sobre as mensagens de ataque que o detector deixou passar**
(falsos negativos no ponto primário) — "o que fez o detector chamar estas
mensagens de normais". Quando o candidato não tem falso negativo, o SHAP é
calculado sobre todas as mensagens de ataque dele. O campo `split` do
relatório diz qual das duas populações foi usada.

Por que não as importâncias Gini do RF congelado: o detector não muda durante
a campanha, então elas seriam **as mesmas em toda rodada**. O Defensor
receberia a mesma evidência a cada candidato, o plano tenderia a se repetir, e
um "B4 = B3" mediria o desenho do experimento, não o valor do feedback
defensivo. O SHAP sobre os falsos negativos muda com o candidato porque as
mensagens que evadem mudam com ele.

## Custo e cota

Groq gratuito: 8000 TPM, 1000 RPD, e TPD por modelo (não exposto nos headers).
O executor espaça as chamadas, contabiliza tokens por chamada e, em erro de
cota diária, salva o ledger e retoma depois — a campanha é retomável e nunca
repete uma proposta já registrada.

## Saídas

`outputs/experiments/e1/`:
- `ledger/<família>/<estratégia>/rep<r>.jsonl` — uma linha por proposta;
- `ids/<família>.json` — treino e limiares do detector congelado;
- `results.json` — teste do selecionado por campanha + B0.

## Piloto

O piloto (1 família, 1 repetição por estratégia) serve para medir tempo e
tokens e fixar o modelo do LLM e a escala. Não entra nos resultados.

Decide, nesta ordem, com os tokens medidos por chamada:

1. as cinco famílias cabem na cota diária em prazo aceitável? Se sim, ficam;
2. se não, o Defensor passa para outro modelo (`openai/gpt-oss-20b`), que tem
   cota diária própria — o atacante continua o mesmo no B3 e no B4, que é o
   que a comparação B4−B3 exige;
3. se ainda não couber, três famílias, mantendo 5 repetições.

**Decisão (15/10):** passo 2. Projeção do piloto para 4 famílias × 5
repetições: ~1,85M tokens, ~9 dias num modelo só no teto diário estimado
(200k/modelo, não confirmado). O Defensor vai para `openai/gpt-oss-20b`
(default de `--defender-model-id`); o atacante segue em `openai/gpt-oss-120b`
nas duas condições. As quatro famílias ficam (~6 dias).

### O que o piloto mostrou (15/10/2026, `masquerade_fault`, rep 1)

| | propostas | válidas | chamadas | tokens | evasão@10 |
|---|---|---|---|---|---|
| B1 | 10 | 10 | 0 | 0 | 0,000 |
| B2 | 6 (escada esgotada) | 6 | 0 | 0 | 0,000 |
| B3 | 10 | 9 | 10 | 27k | 0,000 |
| B4 | 10 | 8 | 17 | 65k | 0,000 |

- **Evasão por orçamento presa em zero.** As 34 propostas válidas tiveram
  evasão 0 nos três orçamentos e oráculo AP 1,0. A causa é o detector, não a
  busca: o RF dá score exatamente 0 a praticamente todo o tráfego normal, então
  os limiares de 10 e 100 alarmes calibram em 0,0 (o de 1 alarme, em 0,02) e
  qualquer voto de árvore em "ataque" dispara. Com esse objetivo, toda
  comparação B×B daria zero por construção.
- **O score médio do ataque se move.** Recalculado sobre as mesmas propostas
  (traces em cache): 1,00 na referência, até 0,44 nas variantes com
  `incrementStNumOnFault=False` ou `sqnumMode=normal`, ~0,83 numa linha que o
  B3 achou (desvio analógico mínimo, multiplicador ~1). Daí a troca do
  objetivo para a evasão suave. A evasão por orçamento continua reportada: é
  ela que diz se alguma mensagem de fato passa.
- **Custo medido:** ~35–80 s por proposta (3 gerações + oráculo); B3 ≈ 2,7k
  tokens por chamada, B4 ≈ 3,8k (17 chamadas: 10 do atacante, 7 do Defensor;
  o reuso do plano poupou 2). Portões funcionando: o LLM ditou 5 campos uma
  vez (`no_intent`) e inverteu um par outra (`compile_failed`); Defensor com
  8/8 planos aceitos.

## Implementação

- `experiments/search.py` — caixa (`SearchBox`), cadeia de validade
  (`CandidateEvaluator`, `Oracle`), B1 (`RandomSearch`), B2 (`OpenLoopLadder`,
  sobre `core/feedback_policy.py::escalate_intent`), ledger e campanha
  retomável, seleção e teste. Sem LLM.
- `agents/attacker/` — B3/B4 (`LlmAttacker`), prompt em `prompts/attacker.md`.
  Toda estratégia propõe uma `IntentSpec` (o B1 como `target_values`), então
  compilador, caixa e validade são o mesmo código para as quatro.
- `scripts/run_e1.py` — `references` (IDS congelado, B0, AP do oráculo nas
  referências → λ), `campaign` (com `--pilot` para o piloto) e `report`.
- Erros do provedor são lidos antes de contar a proposta: limite por minuto
  espera e repete (não conta); cota diária ou prompt grande demais param a
  campanha sem registrar a proposta em curso. Uma resposta sem chamada da
  ferramenta, ou recusada pelos portões, **conta** como proposta inválida
  (`no_intent`).
- O B4 reusa o plano quando o último candidato válido não mudou desde a
  consulta anterior ao Defensor: mesma evidência, nenhuma chamada a mais.
- O B2 tem no máximo 6 degraus (4 em `injection`) até a largura da caixa; a
  curva repete o melhor valor depois do esgotamento.

## Referências (15/10/2026)

`scripts/run_e1.py references`, detector congelado de verdade (seeds de
treino 101–103, limiares calibrados em 201–202):

| Família | evasão@10 busca | evasão@10 teste | AP teste | AP oráculo | p | λ observado |
|---|---|---|---|---|---|---|
| masquerade_fault | 0,000 | 0,000 | 1,000 | 1,000 | 0,375 | 1,000 |
| random_replay | 0,000 | 0,000 | 1,000 | 1,000 | 0,500 | 1,000 |
| inverse_replay | 0,000 | 0,000 | 1,000 | 1,000 | 0,500 | 1,000 |
| high_stnum | 0,008 | 0,007 | 1,000 | 1,000 | 0,500 | 1,000 |
| injection | 0,981 | 0,980 | 0,346 | 0,363 | 0,048 | 0,331 |

- Evasão suave de teste (o objetivo adotado depois do piloto) na referência:
  masquerade 0,000, random_replay 0,002, inverse_replay 0,001, high_stnum
  0,007 (injection 0,656).
- As quatro famílias do E1 partem de detecção perfeita: B0 está no piso, e
  toda evasão que uma estratégia achar é mensurável.
- `injection` evade 98% **sem busca nenhuma**: nenhuma estratégia pode
  melhorar sobre o B0, e a comparação entre elas mediria ruído. Ela também era
  o único caso que puxaria λ para baixo — nem um RF treinado nela passa de
  AP 0,36, coerente com um ataque que clona mensagens legítimas. Sai do E1
  como achado próprio (ponto cego do detector na referência), não como falha.

## Pendências antes das execuções principais

- piloto de tokens e a decisão de escala acima.
