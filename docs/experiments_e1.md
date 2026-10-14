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
  `injection`, `high_stnum` (catálogos `.v2`/`.v1` pós-E0). Se o piloto mostrar
  que a cota da Groq não comporta as cinco, corta-se para três famílias —
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
  **4 × o valor de referência** do campo (ou do `max` do par, se for par).
  Só afeta `masquerade_fault` (`fault.durationMs`, `analog.deltaAbs`,
  `trapArea.multiplier`). **Pendente:** trocar o fator por um limite físico
  justificado (duração de falha plausível, desvio analógico plausível) antes
  das execuções principais;
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
   do ataque na 302. **λ ainda não está fixado**: antes das execuções o
   oráculo roda nas 5 referências, e λ fica abaixo do pior valor observado
   nelas, com folga. Se alguma referência não separar do normal nem com o
   oráculo, o problema é do critério, não da família — e o critério é revisto
   antes de qualquer execução.

O item 4 é a defesa contra "evasão por desaparecimento": uma config que apaga o
ataque (ex.: `fault.prob = 0`) evade qualquer detector sem expor ponto cego
nenhum. Evasão do detector congelado só conta se um detector que **viu** a
variante ainda a separa do normal. A referência de cada família é checada pelo
mesmo critério; se a referência falhar, a família sai do E1 com o motivo.

## Medidas

- **Objetivo de busca** (o que B2–B4 veem e o que seleciona): evasão no ponto
  primário, média das duas seeds de busca. Desempate: menor AP médio, depois a
  proposta mais antiga.
- **Seleção e teste:** ao fim da campanha, o melhor candidato válido é
  regenerado nas 5 seeds de teste e medido contra o detector congelado. Campanha
  sem candidato válido registra teste ausente (e entra na taxa de validade).
- **Por campanha (unidade de análise):** evasão de teste nos três orçamentos
  (média das 5 seeds), AP, alarmes falsos por 10k, validade (fração das 10
  propostas), tokens, chamadas de LLM, tempo.
- **Curva:** melhor evasão de busca até a consulta *k*, por estratégia e família.
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

## Pendências antes das execuções principais

- λ do oráculo, a partir das 5 referências;
- limite físico para os lados abertos da caixa (masquerade);
- piloto de tokens e a decisão de escala acima.
