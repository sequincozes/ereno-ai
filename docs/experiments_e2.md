# E2 (reduzido) — transferência das variantes do E1 para outros detectores

Executado em 09/11/2026 (`scripts/run_e2.py`), em resposta à revisão do artigo
("a não-robustez é propriedade do RF ou do problema?").

## Protocolo

- Mesmo protocolo do E1: seeds de treino 101–103, calibração 201–202 e teste 401–405;
  pipeline `protocol_evaluator`, com `PROTOCOL_FEATURES_MODE=deltas` e os limiares
  por orçamento calibrados no normal da calibração. Muda só o detector do `FrozenIds`.
- As configs avaliadas são as **selecionadas contra o RF** por cada campanha do E1
  (B1–B4, todas as repetições). A seleção não é refeita: a pergunta é de
  transferência.
- Só usa traces do cache (`CachedOnlyStore`): nada é gerado, e o JAR não é chamado.
- O RF entra como controle de integridade: o script aborta se a evasão suave do RF
  não reproduzir o `results/` do E1. Passou nas 4 famílias.
- Detectores: `random_forest`, `xgboost`, `decision_tree` e `svm_linear`. No SVM, o
  score é a função de decisão, então a evasão suave sai como NaN e valem o AP e a
  evasão por orçamento.

## Resultados (teste; média das campanhas)

Evasão suave (B0 = referência):

| Família | Detector | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| high_stnum | RF | 0,007 | 0,166 | 0,102 | 0,231 | 0,223 |
| | XGBoost | 0,004 | 0,227 | 0,061 | 0,278 | 0,260 |
| | DT | 0,004 | 0,190 | 0,046 | 0,290 | 0,277 |
| inverse_replay | RF | 0,001 | 0,180 | 0,027 | 0,191 | — |
| | XGBoost | 0,000 | 0,333 | 0,049 | 0,359 | — |
| | DT | 0,000 | 0,349 | 0,049 | 0,373 | — |
| masquerade_fault | RF | 0,000 | 0,515 | 0,514 | 0,537 | 0,539 |
| | XGBoost | 0,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| | DT | 0,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| random_replay | RF | 0,002 | 0,092 | 0,062 | 0,189 | — |
| | XGBoost | 0,000 | 0,001 | 0,000 | 0,001 | — |
| | DT | 0,000 | 0,126 | 0,102 | 0,357 | — |

AP, variante − referência (todas as campanhas, estratificado por família):
RF −0,025 [−0,032, −0,018]; XGBoost −0,193 [−0,227, −0,158];
DT −0,314 [−0,341, −0,284]; SVM linear −0,297 [−0,320, −0,274].

Evasão suave, variante − referência: RF +0,252; XGBoost +0,397; DT +0,439.
Todos os intervalos excluem zero.

B3 − B1 (pareado por repetição, estratificado):

| Métrica | RF | XGBoost | DT | SVM linear |
|---|---|---|---|---|
| evasão suave | +0,049 [+0,011, +0,088] | +0,019 [−0,025, +0,056] | +0,089 [+0,016, +0,166] | — |
| evasão@10 | +0,030 [+0,018, +0,042] | +0,023 [−0,037, +0,074] | +0,025 [−0,145, +0,176] | +0,014 [−0,013, +0,041] |
| AP | −0,011 [−0,022, +0,000] | −0,078 [−0,133, −0,032] | −0,119 [−0,167, −0,068] | −0,074 [−0,113, −0,030] |

## Leitura

1. **A fragilidade é do espaço de ataque, não do RF.** As variantes escolhidas
   contra o RF transferem para os três outros detectores, quase sempre com efeito
   **maior** do que no próprio RF. O RF, por fazer média sobre muitas árvores, é o
   mais robusto dos quatro. Na queda de AP ele é de longe o que menos perde.
2. **masquerade_fault é o caso extremo.** O RF mantém AP = 1,0 com evasão suave de
   ~0,5, ou seja, as variantes ficam mais perto da fronteira, mas continuam
   separáveis. Já XGBoost, DT e SVM caem para AP 0,10–0,55, e XGBoost/DT dão score
   ~0 às mensagens de ataque (evasão suave 1,0).
3. **random_replay contra o XGBoost é a exceção.** Nenhuma variante transfere
   (evasão suave ≤ 0,001, AP 1,0).
4. **A vantagem do LLM (B3 > B1) se mantém no sentido, mas nem sempre separa.**
   No AP, ela separa em XGBoost, DT e SVM, e com efeito maior do que no RF, onde
   foi medida. Na evasão suave, separa no RF e no DT, mas não no XGBoost.
5. **B2 transfere pouco.** A escada sem feedback produz variantes que só os
   detectores mais simples deixam passar.

## Ressalvas

- Os orçamentos ficam NaN para XGBoost/DT em masquerade (e para DT em high_stnum e
  random_replay): o normal da calibração empata em score ~0, o limiar fica acima de
  todos os normais, e `is_saturated` marca o ponto como "sem ponto de operação".
  Esse é o empate *por baixo* (separação perfeita na calibração), diferente do
  empate *por cima* que motivou o NaN no E3. A leitura conservadora foi mantida.
  Evasão suave e AP carregam o resultado.
- No DT, a "evasão suave" é praticamente a evasão do arg-max: as folhas são quase
  puras e o score é ~0/1.
- O SVM linear já começa fraco em high_stnum: evasão@10 de 0,32 na própria
  referência.
- A seleção foi feita contra o RF. Um atacante que mirasse cada detector poderia
  evadir mais. Os números acima são um piso da transferência, não um teto.
- E2 completo (regras de uma feature) segue fora.
