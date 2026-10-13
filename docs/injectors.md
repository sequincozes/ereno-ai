# Injector: intent-, model- e data-driven

O diagrama do journal desenha o Injector como uma caixa com três estratégias.
Até 13/10/2026 só o intent-driven existia, e trocar de injetor significava
trocar o pipeline inteiro com `--engine`. Agora as três vivem atrás de um
contrato só, dentro do pipeline intent-driven:

```bash
# intent-driven (default, o comportamento de sempre)
uv run adversarial-ids --engine intent --prompt "reduza o recall variando a temporização da falha"

# model-driven: o LLM lê o creator Java de um ataque e propõe um ataque novo
uv run adversarial-ids --engine intent --generator-mode jar \
  --injector model --injector-source inverse_replay \
  --prompt "proponha uma variante mais discreta do replay invertido"

# data-driven: o LLM lê o perfil de um trace e propõe um ataque novo
uv run adversarial-ids --engine intent --generator-mode jar \
  --injector data --injector-dataset outputs/intent_loop/<run>/dataset_iteration_1.csv \
  --prompt "reproduza a assinatura temporal do ataque observado"
```

## O contrato

As três estratégias implementam o mesmo `interpret(prompt) -> IntentSpec` que o
orquestrador já consumia (`IntentLike`). A `IntentSpec` passa pelos mesmos
portões, é compilada pelo mesmo compilador e roda pelo mesmo pipeline —
gerador, pré-processador, detector, defensor, feedback. **O que separa as
estratégias é só o contexto que o LLM lê antes de propor o ataque:**

| `--injector` | contexto | o LLM pode devolver |
|---|---|---|
| `intent` | o catálogo de capacidades inteiro (12 ataques) | qualquer ataque do catálogo |
| `model` | o creator Java de `--injector-source` + o catálogo do `programmable` | só `programmable`, com regra autorada |
| `data` | o perfil de `--injector-dataset` + o catálogo do `programmable` | só `programmable`, com regra autorada |

Os contratos de saída do pipeline não mudam: o pipeline intent-driven segue
gravando `LoopRecord`, e o loop legado (`--engine live`) segue gravando
`IterationRecord`. O Injector unifica a **entrada**, não a saída.

## Por que model e data ficam presos ao programmable

"Ataque novo" não pode significar código novo. O guardrail central do projeto é
que nenhuma saída de LLM escreve no runtime do ERENO — e um creator Java escrito
pelo modelo seria exatamente isso. O ataque programável (uc11) é a forma de um
ataque novo existir sem código novo: regras declarativas (mutar, descartar,
repetir, atrasar, reordenar, selecionar em quem agir) que o compilador valida
campo a campo e o schema valida em conjunto (ver `docs/programmable_attack.md`).

Gerar Java de verdade é o épico E12, fora do escopo dos 60 dias.

Um terceiro portão, depois dos dois que toda `IntentSpec` já atravessa
(`agents/injector/gate.py`), confere que o que voltou é o que a estratégia
promete:

- `base_attack` é `programmable`;
- há ao menos um caminho `rules.` em `target_values`. Sem regra autorada, o
  programmable roda as regras do baseline — e o "ataque novo" que o manifest
  anuncia seria o ataque padrão.

## O contexto de cada estratégia

**model** (`agents/injector/java_sources.py`): o creator configurável (sufixo
`C`) do ataque, lido de `ERENO_SOURCE_DIR` (default: o repo irmão
`../ERENO-2.0`; o JAR não carrega fonte). Pacote e imports são removidos — não
ensinam nada sobre o comportamento e custam TPM. O `programmable` não é
fonte: ler o interpretador da DSL para escrever na própria DSL é circular.

**data** (`core/trace_profile.py`): um perfil determinístico do CSV, não o CSV.
Para cada coluna numérica, média e desvio na classe de ataque e na `normal`, e a
separação entre elas — a mesma medida do piloto E0, |Δ médias| / desvio
combinado. O LLM vê as colunas que mais separam, as constantes em valores
diferentes (separação total, sem virar infinito) e, sempre, as colunas que as
regras do programmable mexem. Com mais de uma classe de ataque no trace,
`--injector-label` escolhe qual perfilar.

Só o catálogo do `programmable` entra nas instruções, não os 12 ataques: a
estratégia só pode devolver `programmable`, e o catálogo inteiro ocuparia ~7k
dos 8k tokens por minuto da conta, sem sobrar espaço para o código que o modelo
precisa ler. Medido em 13/10: as instruções do `model` têm 9–14,5 mil
caracteres conforme o creator, as do `data` ~8 mil, contra 20 mil do `intent`.

## Artefatos

A rodada que chama o LLM grava, **antes** da chamada:

- `injector_manifest.json` (`InjectorManifest`): a estratégia, o tipo de
  contexto, a restrição de `base_attack`, o `subject` (o ataque cujo creator foi
  lido, ou a classe perfilada), as fontes com o **sha256** de cada uma, e o
  tamanho das instruções. O hash importa: o mesmo caminho em outro commit do
  ERENO, ou um trace regenerado, é outro contexto.
- `trace_profile.json` (`TraceProfile`), só no `data`: o que o modelo viu.

Antes da chamada porque, se ela falhar, o contexto é o primeiro lugar onde se
olha. As rodadas seguintes de uma campanha herdam a intenção pela política de
feedback (E10), não chamam o LLM e não regravam o manifest.

O validador do `InjectorManifest` recusa um manifest que não bata com a
estratégia: um `model` sem a restrição ao `programmable`, ou um `model`/`data`
sem fonte, descreveria um injetor que este repo não tem.

## Retomada

`--resume` só chama o LLM quando refaz uma rodada cuja intenção nunca chegou a
existir. Nesse caso, a estratégia precisa ser a mesma da campanha — o
orquestrador lê o `injector_manifest.json` da rodada refeita e recusa outra.
Passe as mesmas flags `--injector*` da execução original. Nos outros casos a
intenção vem do ledger e o injetor nem roda.

## Validado com LLM real (13/10/2026, `--generator-mode jar`)

- **model**, sobre `inverse_replay`, pedido "variante mais discreta: só parte
  das mensagens, atraso pequeno": `add timestamp +0.02` em 10% das mensagens.
  Os sete estágios passaram. É uma **aproximação** do replay, e o modelo não
  usou `duplicate` — o que o creator faz e a DSL não exprime (escolher
  mensagens de um passado aleatório) ficou de fora, como o prompt manda.
- **data**, sobre um trace do `inverse_replay` (assinatura: `timestampDiff`
  1,8 ms no ataque contra 0,7 s no normal), pedido "reproduza a assinatura
  temporal": precisou de dois ajustes de prompt, e os dois portões fizeram o
  trabalho deles antes disso.
  1. Autorou 10 caminhos e declarou `max_fields_changed=2` → recusado na
     intenção. O bloco comum ganhou a regra de contagem junto do
     `target_values`.
  2. Leu certo o perfil, mas traduziu "mensagens coladas" como `set timestamp
     0` → recusado pelo schema do programmable. O prompt do data ganhou a
     tabela assinatura → regra (delta perto de zero é `duplicate`, cujas cópias
     saem adjacentes com o mesmo relógio).
  3. Com os dois: `duplicate` em 20% + `add timestamp +0.04` em 50%. Intenção,
     compilador, gerador e detector passaram.

Na terceira execução, quem falhou foi o **Defender**, não o Injector: o LLM
escreveu o `detection_report_ref` (um caminho Windows) com as barras escapadas
duas vezes, e o validador recusou o plano. É do E5, intermitente — a rodada do
model passou pelo mesmo estágio.

## O que o Injector não é

Um decisor. Nada escolhe a estratégia a partir de uma métrica — quem escolhe é
quem roda o experimento. Mesmo raciocínio do decisor de undersampling
(`docs/feature_selection.md`): o que otimiza a métrica não escolhe a própria
etapa.

## Adicionar uma estratégia

1. A chave no `InjectorKey` (`domain/injector_manifest.py`), com o
   `ContextKind` e a restrição de `base_attack` dela.
2. Um `InjectorSpec` no `_REGISTRY` de `agents/injector/registry.py` (a
   importação recusa registro e contrato fora de sincronia), e o ramo de
   `prepare_injector` que lê o contexto.
3. O papel em `prompts/injector_<chave>.md`; o bloco comum
   (`prompts/injector_common.md`) já vale para qualquer estratégia presa ao
   `programmable`.
