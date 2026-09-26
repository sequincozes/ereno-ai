# Piloto E0 — validade das variantes por família de ataque

O E0 é o primeiro experimento do artigo e não produz resultado sobre o IDS:
ele decide **quais famílias de ataque podem ser usadas** nos experimentos
seguintes, e a que custo. O roteiro pede três coisas — gerar baseline e
variantes de `masquerade_fault`, `random_replay` e `grayhole`; confirmar que
mexer nos parâmetros altera o tráfego e preserva o ataque; estimar tempo e
tokens para fixar o orçamento — e pede explicitamente que uma família
reprovada tenha a causa registrada *antes* de o escopo encolher.

Quem executa é `scripts/run_pilot.py`:

```bash
uv run python scripts/run_pilot.py                      # as três famílias (~90s, precisa do JAR)
uv run python scripts/run_pilot.py --attack grayhole --replicates 5
uv run python scripts/run_pilot.py --llm-probe          # inclui o custo em token do estágio intent
uv run python scripts/run_pilot.py --llm-probe-only     # só o custo em token, sem gerar trace
uv run python scripts/run_pilot.py --attack grayhole --max-fields 11 --label campos11  # execução separada
```

Saída: `outputs/pilot/pilot_report.json` (JSON simples com `schema_version`; é
artefato de experimento, nada em `src/` o lê) e, por família,
`outputs/pilot/<ataque>/` com cada trace, a config que o gerou e uma amostra de
linhas de ataque para inspeção manual. Código de saída 0 se todas as famílias
pedidas passaram, 1 se alguma reprovou.

## As variantes não vêm de LLM

Cada variante sai de uma `IntentSpec` compilada por `core/intent_compiler.py`,
uma por intensidade (`low`/`medium`/`high`), sob o mesmo `desired_effect`. É a
mecânica que a política de feedback (E10) usa da rodada 2 em diante — ou seja,
o piloto mede exatamente o gerador de variação que a campanha vai usar, sem
gastar chamada de modelo e sem introduzir variabilidade de LLM na medição.

O custo em token entra por outro caminho: `--llm-probe` faz **uma** chamada real
do `IntentAgent` por família só para registrar o consumo (`--llm-probe-only`
mede sem gerar trace nenhum). As chamadas são espaçadas em 90s porque o prompt
com o catálogo de capacidades injetado pesa 6,66k tokens e o tier `on_demand` da
conta tem TPM 8000 — duas chamadas na mesma janela viram `rate_limit_exceeded`,
que aparece de fora como "a LLM não chamou submit_intent_spec".

## Três medidas, e o que cada uma deixa passar

**1. Portão de volume (E4).** `build_dataset_bundle` sobre cada trace: classes
presentes, contagens, prevalência da classe de ataque. Pega trace vazio,
quebrado, sem a classe esperada ou raro demais para medir. Não diz nada sobre o
ataque estar ativo.

**2. Assinatura.** Diferença padronizada de médias entre as linhas de ataque e
as linhas normais **do mesmo trace**, por feature. Um trace pode ter 50% de
linhas rotuladas como ataque e ainda assim ser indistinguível do tráfego normal:
o rótulo vem do gerador, não da física. O relatório traz o pico, quantas
features passam do limiar e o ranking — porque uma separação enorme numa coluna
só e uma separação moderada em oito são evidências diferentes sobre o mesmo
ataque.

Isto é evidência estatística de que a classe é **distinguível**, não prova de
semântica preservada. É por isso que o script também salva
`<tag>_sample.csv`: a inspeção das mensagens é parte do E0 e nenhuma estatística
a substitui.

**3. Deriva contra o ruído do gerador.** É a medida que justifica o piloto
existir, e a que mais deu trabalho.

## O gerador não é determinístico (sem `randomSeed`)

> **Atualização (Fase 2.0, 26/09/2026):** isto valia porque os creators
> sorteavam por `Math.random()`, fora da RNG semeada. A Fase 2.0 roteou esses
> sorteios pela `ConfigLoader.RNG` e fez o `GeneratorRunner` gravar um
> `randomSeed` no action config — com a seed fixa, a mesma config passa a
> produzir **os mesmos bytes** (verificado: `seed=123` deu hash idêntico em duas
> execuções). O texto abaixo descreve o comportamento **sem** `randomSeed`, que
> continua sendo o default fora do pipeline intent-driven. Ver
> `docs/phase2_plan.md`.

A mesma configuração, rodada duas vezes **sem `randomSeed`**, produz **bytes
diferentes** (medido em 17/09/2026 no `random_replay`: dois `content_hash`
distintos para o mesmo JSON). Consequência direta: "o hash da variante é
diferente do hash da baseline" não é evidência de nada — seria diferente de
qualquer jeito.

Por isso o piloto gera a baseline `--replicates` vezes (default 3) e mede a
separação **entre réplicas da mesma config**. Isso é o ruído do gerador, e é o
piso que qualquer deriva precisa superar.

A primeira versão usava um piso único: o máximo dessa separação sobre todas as
features. Não funcionou. O máximo sobre ~40 colunas é o máximo de estimativas
ruidosas; ele mesmo oscila entre execuções, e com ele o veredito da família
oscilava junto — em 18/09/2026, `grayhole` saiu 3/3 numa execução e 1/3 na
seguinte, sem nada ter mudado entre as duas.

A versão atual compara **coluna a coluna**: o ruído de `timestampDiff` é o piso
de `timestampDiff`. Uma feature conta como movida quando sua deriva supera
`--min-drift` (piso absoluto, default 0.05 — sem ele, uma coluna quase sem ruído
passaria por qualquer deslocamento) **e** `--drift-margin` vezes o ruído da
própria coluna (default 2x). A variante é distinta da baseline quando pelo menos
`--min-moved-features` colunas (default 1) se movem. Com isso o veredito por
família passou a se repetir entre execuções idênticas.

Duas exclusões deliberadas: colunas de identidade e relógio absoluto
(`always_drop_for`, o mesmo descarte do preprocessador) ficam de fora da
medição, senão a "separação" mediria o bloco de geração e não o ataque; e uma
coluna que o próprio gerador desloca por inteiro entre réplicas entra em
`unusable_features` — ela não pode servir de evidência para nada.

O default de `--protocol-features` aqui é `deltas`, e não o `drop` do resto do
repo: sem `stDiff`/`sqDiff`/`tDiff`/`timestampDiff` não sobra nada em que replay
e grayhole possam ter assinatura.

## Resultado da execução de 18/09/2026

Três famílias, efeito `lower_recall`, três intensidades, três réplicas de
baseline, 18 gerações: 88s gerando (~4,9s por trace), 891 MB em disco.

| família | veredito | variantes distintas | o que carrega a deriva |
|---|---|---|---|
| `masquerade_fault` | passou | 3/3 | `cbStatus`, `cbStatusDiff`, `SqNum`, `sqDiff`, `stDiff` (6–7 features) |
| `random_replay` | passou | 3/3 | elétricas em `low`/`medium`; `timestampDiff` e `timeFromLastChange` em `high` |
| `grayhole` | reprovou | 2/3 | só `timestampDiff`, no limite do ruído |

A primeira execução, antes da correção descrita abaixo, tinha reprovado também
o `random_replay`.

### Corrigido: o compilador fechava intervalos

A primeira rodada do piloto matou a variante `high` do `random_replay` no
gerador: o compilador produzia `windowS = {min: 2.0, max: 2.0}` e o ERENO
recusava o processo inteiro com *"The lower limit (2000) must be less than the
upper limit (2000)"*. O clamp de pares preservava `min <= max`, o schema do
ataque valida `min <= max`, o portão de capacidade não olha pares — um
intervalo degenerado passava por tudo que o repo tem e morria no Java.

Não era um caso isolado: varrendo cada campo de cada ataque sozinho, o clamp
antigo produzia par degenerado em **22 pares distintos, em 10 dos 11 ataques
registrados** (`windowS`, `burst`, `gapMs`, `dropRate`, `burstInterval`,
`networkDelayMs`, `jump`, `blockLen`, `analog.deltaAbs`, ...). Atingia a
campanha do E1 do mesmo jeito que atingiu o piloto.

`intent_compiler._clamp_paired` passou a garantir `min <` **estrito**: quando o
valor calculado cruzaria o irmão, **o irmão vira o limite efetivo** e o mesmo
passo da intensidade é reaplicado contra ele. Não há épsilon inventado — a
distância sai da política de escalada que já estava em uso — e a escada segue
monotônica (`windowS.max` agora dá 6.42 / 4.05 / 2.9 em low/medium/high, contra
um piso de 2.0). Em pares de inteiros, quando o arredondamento cola no irmão o
valor recua uma unidade; se nem isso couber, o campo não se move e some do
`diff`, que passa a dizer a verdade sobre o que mudou.

A invariante é varrida em `tests/test_intent_compiler_ranges.py`: todo ataque,
todo efeito, toda intensidade, **e cada campo compilado sozinho** — que é o
caso perigoso, um limite empurrado contra um irmão congelado.

### O que continua de pé

**`grayhole` é a família frágil, nos dois eixos.** Sua assinatura contra o
tráfego normal é 0.80 e **uma única feature** passa do limiar de 0.2 (contra 14
no `masquerade_fault` e 7 no `random_replay`). A variação também é magra: só
`timestampDiff` se move, e a variante `low` fica dentro do ruído do gerador —
é ela que reprova a família. O efeito `lower_recall` empurra
`dropRate`/`burstDropProb` para baixo, ou seja, torna o ataque mais discreto; o
espaço de busca que sobra é estreito.

**Em `random_replay`, só a intensidade alta mexe na temporização.** As variantes
`low` e `medium` movem 11 features, todas **elétricas** (`vsbARmsValue`,
`isbARmsValue`, trap areas): mudar a janela de replay muda *quais* mensagens
legítimas são recapturadas, e com elas os valores analógicos carregados. Só em
`high` — a variante que a correção acima destravou — a deriva alcança
`timestampDiff` e `timeFromLastChange`, que é onde um replay deveria aparecer.
Atribuir a deriva de `low`/`medium` à temporização do ataque seria ler errado o
que a medição diz.

## O grayhole está espremido pelos dois lados

A reprovação do `grayhole` com `lower_recall` e 3 campos não bastava para cortar
a família — podia ser o piloto que rodou estreito. Duas execuções de diagnóstico
(`--label`, que separa os diretórios para não sobrescrever os traces do E0)
fecharam a questão, e a resposta é mais interessante que "passa" ou "não passa":

| execução | campos | prevalência low→high | deriva low/medium/high |
|---|---|---|---|
| `lower_recall`, 3 campos | 3 | — | 0.045 / 0.073 / 0.163 |
| `lower_recall`, 11 campos | 9–11 | 0.27 → 0.41 | 0.039 / 0.084 / 0.115 |
| `increase_attack_activity` | 6 | 0.13 → **abaixo do piso** | 0.108 / 0.063 / — |

As duas direções mexem na prevalência em sentidos **opostos**, e as duas têm
parede:

- Empurrar para **evasão** significa descartar menos. Sobram mais mensagens
  rotuladas como ataque (0.23 na baseline → 0.41), mas o tráfego fica cada vez
  mais parecido com a baseline — a variante `low` cai dentro do ruído do
  gerador e é ela que reprova a família.
- Empurrar para **atividade** significa descartar mais. Em intensidade alta
  sobram 115 linhas de ataque em 50.112 (0.23%), e o portão de prevalência do E4
  rejeita o dataset: um recall calculado sobre 115 amostras é ruído amostral.
  A classe de ataque do grayhole desaparece justamente quando o ataque fica
  mais agressivo, porque as linhas dele *são* as mensagens que ele não descartou.

E em toda configuração testada — 3, 6, 9 ou 11 campos, nas duas direções — a
deriva se concentra em **uma única feature**, `timestampDiff`. A família é
usável, mas numa faixa estreita e num eixo só: com 11 campos, `medium` e `high`
são distintas da baseline e monotônicas. O E1 pode usá-la desde que não conte
com a intensidade `low` nem com a direção de atividade.

## Orçamento medido

**Geração:** ~4,9s por trace e ~45 MB por trace no JAR (18 gerações = 88s e
891 MB). É o custo dominante em disco e o menor em dinheiro.

**Estágio `intent`:** 6.657 tokens de entrada e 228–259 de saída por chamada,
medidos em 18/09/2026 com `openai/gpt-oss-120b` sobre as três famílias
(`--llm-probe-only`). Praticamente tudo é o prompt: o catálogo de capacidades
injetado. Ou seja **~6,9k tokens por chamada contra um TPM de 8.000** — uma
chamada por minuto é o teto da conta, e as três só passaram porque a sonda as
espaça em 90s. É restrição de conta, não de código, e vale para qualquer
campanha: uma rodada por minuto, no melhor caso.

A parcela do `Defender` ainda não foi medida — ela depende do `DetectionReport`
da rodada e só aparece num ciclo completo, em `LoopRecord.total_tokens`.

## Como ler o relatório

Por família: `verdict`, `reasons` (vazio quando passou), `noise_floor`
(`by_feature`, `unusable_features`, os pares de réplicas), `replicates` e
`variants`. Cada variante traz o `diff` compilado (quais campos mudaram, de que
valor para qual), o portão de volume, a assinatura e o bloco `drift` com
`moved_features` — a lista que sustenta o veredito. `budget` fecha com gerações,
tempo por trace, disco e, se `--llm-probe` rodou, o consumo do estágio intent.

Rodar em `--generator-mode cached` executa o script mas não pilota nada: todo
trace é o mesmo arquivo, o ruído é zero por construção e nenhuma variante se
distingue. O script avisa e segue.

## Efeito colateral de qualquer execução com o JAR

O `GeneratorRunner` reescreve `generator_runtime/config/actions/` e
`generator_runtime/config/attacks/` a cada geração, e esses arquivos são
versionados. Depois de rodar o piloto, `git status` mostra a última variante
compilada dentro deles — `git checkout generator_runtime/` antes de commitar.
Não é específico do piloto: vale para qualquer execução em modo `jar`.
