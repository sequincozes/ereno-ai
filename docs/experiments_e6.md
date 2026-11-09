# E6 — custo (09/11/2026)

O E6 não roda nada. Ele lê o custo do E1 e do E3 a partir do que já foi
gravado (`scripts/report_e6.py` → `outputs/experiments/e6/results.json`).

## Fontes

- **Chamadas e tokens:** vêm dos `results/*.json`. O `campaign_result` soma o
  uso que cada chamada ao LLM grava no ledger. No B4, a soma inclui os dois
  modelos, atacante e Defensor, sem separação.
- **Tempo de uma proposta:** é o `elapsed_s` do ledger, que cobre a geração
  dos traces das seeds de busca, a medição e a latência do provedor.
- **Retreino e matriz cruzada:** nenhum dos dois grava a própria duração. O
  tempo é estimado pela diferença entre os horários de gravação de arquivos
  consecutivos:
  - retreino: `D{r}.json` menos o `round{r-1}.json`, só para r = 2 e 3;
  - campanha e teste: `round{r}.json` menos o `D{r}.json`;
  - matriz: diferença entre os `rep*.json` consecutivos.

  Usa-se a mediana, descartando intervalos de 120 min ou mais: execuções
  retomadas deixam buracos de horas, que são pausa, não trabalho.

## Resultados

| Unidade | n | Chamadas | Tokens | Tempo | Validade |
|---|---|---|---|---|---|
| E1 B1 (campanha) | 20 | 0 | 0 | 6,5 min (38,7 s/proposta) | 1,00 |
| E1 B2 (para em 6 propostas) | 20 | 0 | 0 | 2,8 min | 1,00 |
| E1 B3 (120b) | 20 | 10 | 26,9k (24,2k–29,3k) | 10,1 min (60,5 s/proposta) | 0,86 |
| E1 B4 (120b + 20b) | 10 | 20,3 | 70,5k | 13,1 min (78,8 s/proposta) | 0,91 |
| E3 B1, rodada | 60 | 0 | 0 | retreino 2,9 + campanha e teste 6,4 min | 1,00 |
| E3 B1 sem defesa, rodada | 60 | 0 | 0 | 0,3 + 2,0 min (traces em cache) | 1,00 |
| E3 B3, rodada | 15 | 10 | 25,8k | 3,5 (n=7) + 7,0 min | 0,93 |
| Matriz B1 (com placebo) | 20 | 0 | 0 | 30,5 min | — |
| Matriz B3 | 5 | 0 | 0 | 13,2 min (mediana de só 3 intervalos) | — |
| Plano do Defensor, E3 (20b) | 28 | 1 | 5,0k | — | 27/28 aceitos |

**Defensor no B4:** foram 83 consultas novas, das quais 80 aceitas; 7 planos
reaproveitados e 10 propostas sem candidato válido.

**Tokens no total:** 1,77M de tokens de LLM.
- **120b:** cerca de 1,21M, que somam:
  - o E1 B3 (538k);
  - o E3 B3 (387k);
  - a parte do atacante no B4 (705k − 83 × 5,0k).
- **20b:** cerca de 0,56M. A parte do B4 é estimada pelos ~5k por plano.

Do total do B3, 81% é input.

**Compute:** cerca de 35 h numa estação, somando:
- campanhas do E1: 8,7 h;
- linhagens do E3-B1: 9,4 h com defesa e 2,3 h sem;
- matrizes: 10,4 h;
- E3-LLM: 3,7 h.

O valor bate com as ~9,5 h observadas na `campaign --arm B1`.

## Leitura

1. **Quem domina o tempo é a geração, não o LLM.** O LLM acrescenta cerca de
   22 s por proposta, e o Defensor mais 18 s.
2. **O custo em tokens é previsível**, porque o tamanho do contexto é estável.
   Um plano custa tanto quanto duas propostas, e por isso o B4 gasta 2,6× os
   tokens do B3 sem ganho mensurável (B4−B3 −0,003).
3. **O gargalo é a cota diária da Groq.** O 120b precisou de cerca de 6 dias de
   cota, e o 20b de cerca de 3, contra 35 h de compute.

## Ressalvas

- Os tempos de retreino e de matriz são estimativas por horário de gravação.
- O 1º retreino (D1) não tem marca de início.
- O cache de traces barateia configurações repetidas, e é por isso que o braço
  sem defesa sai quase de graça.
- Não há custo monetário: tudo rodou no tier gratuito, e preço de provedor
  muda.
