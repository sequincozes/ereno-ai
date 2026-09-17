# Observabilidade e UI do loop intent-driven (E11)

Fecha a janela D55-60 do lado da operação. O critério de pronto do épico é
literal — *"usuário vê estágio, duração, artefatos, erro e resultado"* — e a
linha "Operação" da tabela de aceite acrescenta *"timeout, retry, logs
estruturados, segredo → falhas têm estágio/causa; prompts não expõem chaves;
execução cabe no orçamento definido"*.

## O que existia e o que faltava

`LoopRecord`/`LoopStage` (E3) já carregavam `name`, `status`,
`duration_seconds`, `artifact_ref` e `error` por estágio. Faltavam três coisas:

1. **Ninguém lia.** As duas UIs conheciam só `IterationRecord` (loop legado
   Strategist↔Analyst); o motor intent-driven não era alcançável pela interface.
   A CLI imprimia estágio, artefato e erro — mas não a duração, que media e
   jogava fora.
2. **Nada era observável durante a execução.** O `LoopRecord` só existe depois
   que a campanha termina. Enquanto ela roda, quem olha não vê nada — e a UI
   tapava esse buraco por conta própria, deduzindo a fase corrente por
   substring de `stdout` capturado
   (`studio/bridge.py::progress()` procurava `"=== ITERAÇÃO"` e `"concluída"`).
   Renomear uma mensagem de log quebrava a barra de progresso sem nenhum teste
   ficar vermelho.
3. **`cost_usd` era declarado e nunca populado.** Não dá para afirmar que a
   execução cabe no orçamento sobre um campo que é sempre `None`.

## LoopEvent: o que está acontecendo agora

`domain/loop_event.py` é o contrato da transição. Quatro tipos, e a divisão é
estrutural: `run_started`/`run_finished` falam da execução (`stage=None`),
`stage_started`/`stage_finished` falam de um dos sete estágios. `status` e
`duration_seconds` só existem no par `*_finished` — um evento de início que
afirmasse duração estaria mentindo, e o validador recusa em vez de deixar a UI
decidir se acredita.

A ordem é `sequence`, não `created_at`: dois estágios rápidos cabem no mesmo
timestamp, e o relógio pode andar para trás numa sincronização de NTP.

O vocabulário de estágio (`LoopStageName`) e de status (`LoopStageStatus`) vem
de `loop_record`. Evento e registro falam dos mesmos sete estágios, ou a
timeline ao vivo e o histórico contariam histórias diferentes da mesma execução
— e um teste do orquestrador fixa exatamente essa igualdade, campo a campo.

### Onde os eventos nascem

`IntentLoopOrchestrator._stage` é o ponto único: ele já media início, fim,
duração, artefato e causa da falha. O que mudou é que as três variáveis soltas
da rodada (`stages`, `run_dir`, e agora o emissor) viraram um `_RoundContext`,
cujo `record_stage` anexa o `LoopStage` **e** emite o `LoopEvent` na mesma
chamada. É o que torna impossível registrar um sem anunciar o outro.

### Onde os eventos vão

`shared/loop_event_store.py`. Um sink é `(LoopEvent) -> None` e nada mais — sem
classe abstrata, porque o de arquivo é classe (tem caminho), o da UI é um
`append` e um teste passa `lambda`. `fanout` compõe vários e engole a exceção de
cada um: observabilidade quebrada é um problema menor que um pipeline
interrompido por causa dela.

O sink de arquivo é **sempre** ligado, por execução: rodar pela CLI deixa
`events.jsonl` no diretório da run sem ninguém pedir. Quem passa um `event_sink`
para `run_intent_loop`/`IntentLoopOrchestrator` é quem precisa dos eventos
*antes* do fim — a UI ao vivo.

**JSONL, e não o `save_json` do resto do repo.** Um array JSON só é legível
depois do colchete final, ou seja, depois que a execução terminou — exatamente
quando o arquivo deixa de servir para observar. Linha a linha, a UI lê a
timeline de uma execução em curso, e uma execução que morreu no meio deixa em
disco tudo que chegou a acontecer. A leitura pula linha truncada em vez de
falhar: é justamente dessa execução que mais se quer ver a timeline.

## A tela

Página **Loop intent-driven** do studio
(`interfaces/dashboard/studio/view_loop.py`). Dispara a campanha a partir de uma
intenção em português — prompt, rodadas, modo do gerador e detector — e mostra
as cinco coisas que a janela cobra, cada uma da sua fonte:

| o que | de onde |
|---|---|
| estágio corrente, erro | `LoopEvent` ao vivo |
| duração, artefatos | `LoopStage` do `LoopRecord` persistido |
| resultado | os JSON que a execução deixou no diretório dela |

O resultado precisa da terceira fonte porque o `LoopRecord` guarda o *caminho*
do artefato, não o conteúdo: `loop_bridge.outcome_for` abre
`detection_report.json`, `defense_plan.json` e `defense_rules.json`. Todo campo
é opcional — uma execução que parou no ERENO não tem relatório de detecção, e
isso é informação, não erro.

Estágio que nunca rodou aparece como **pendente** em vez de sumir da tabela:
numa execução que parou no terceiro, o que falta é metade da informação.

`loop_bridge.py` é irmã de `bridge.py` e diferente dela no essencial: aquela
captura `stdout` e adivinha; esta registra um sink e lê evento tipado.
`progress()` aqui é estágios fechados sobre esperados (sete por rodada), sem
heurística e sem o teto artificial de 98% que a outra precisa.

## Segredo

`shared/redaction.py`. O caminho pelo qual uma chave escaparia não é o prompt
que escrevemos — é o **erro** que gravamos de volta: `_stage` registra
`str(exc)`, e a exceção de um cliente HTTP costuma trazer a requisição que
falhou, cabeçalho de autorização incluído. Aquele texto vai para
`loop_records.json`, para `events.jsonl` e para a tela.

Duas defesas, e a primeira é a que vale:

1. **O valor real.** Toda env var cujo nome contém `API_KEY`, `SECRET`, `TOKEN`,
   `PASSWORD`, `CREDENTIAL`… tem seu valor procurado literalmente no texto. Não
   depende de a chave ter formato reconhecível — só de ela estar no ambiente,
   que é exatamente o caso quando ela vazou de lá. Valores com menos de 12
   caracteres são ignorados: um `DEBUG_TOKEN=1` apagaria todo dígito 1 das
   mensagens.
2. **O formato.** `gsk_…`, `sk-…`, `Bearer …` — pega a chave que não é a deste
   processo.

A mensagem em volta sobrevive: `401 Unauthorized para ***` continua
diagnosticável.

**Por que não no contrato.** Seria tentador fazer `LoopStage`/`LoopEvent`
redigirem sozinhos. Não: o contrato passaria a ler `os.environ`, e um modelo
congelado que depende do ambiente deixa de ser reproduzível. A redação mora no
ponto de emissão, que é onde o texto nasce.

## Orçamento

`domain/run_usage.py::AgentUsage` + `agents/usage.py`. Os agentes registram em
`last_usage` o que a resposta do provedor declarou, lido por `getattr` em vez de
importar o tipo do `agno` — funciona igual com agente real, stub, ou uma versão
da lib em que o objeto de métricas mudou de nome.

Contabiliza-se **antes** dos portões: a chamada foi paga mesmo quando a saída é
recusada logo depois. Um orçamento que só conta chamada bem-sucedida não mede o
que se gastou.

**Token, não dólar.** `input_tokens`/`output_tokens` são contagem informada pelo
provedor; `cost_usd` só existe quando ele informa, e o repo não carrega tabela
de preço própria — ela envelhece em silêncio (o `MODEL_ID` default deste projeto
já precisou mudar porque a Groq aposentou o modelo), e um custo errado é pior
que custo ausente porque parece que serve para decidir. Somar um custo conhecido
com um desconhecido não finge que o segundo é zero.

`INTENT_LOOP_TOKEN_BUDGET` (env var, default `0` = desligado) é o teto da
campanha. Ele é checado **entre** rodadas e fica fora da política de feedback do
E10 de propósito: a política decide se vale a pena continuar *cientificamente*,
o teto decide se dá para continuar *operacionalmente*. Misturar os dois faria um
estouro de custo aparecer como veredito sobre o experimento.

Duas consequências deliberadas:

- **a rodada em curso sempre termina** — um `LoopRecord` pela metade não é mais
  barato, só menos útil;
- **campanha cujos agentes não informam consumo nunca é interrompida** por aqui.
  Zero token é uma afirmação sobre a chamada; `None` é a confissão de que não se
  sabe, e comparar um orçamento contra um zero inventado puniria quem só não tem
  a informação.

O consumo aparece no rodapé do resumo da CLI e no painel da execução no studio —
e só quando foi informado: um `0 tokens` impresso porque ninguém contou seria
pior que a ausência.

## Timeout e retry

Já existiam antes do E11, no `GeneratorRunner`:
`GENERATOR_TIMEOUT_SECONDS`, `GENERATOR_MAX_RETRIES` e
`GENERATOR_RETRY_BACKOFF_SECONDS` (ver `config/settings.py`). O E11 não os
mudou; o que ele acrescenta é a falha daquele estágio aparecer com estágio e
causa na timeline e na tela, em vez de só no `stdout`.

## Testes

- contrato do `LoopEvent`: evento de estágio sem estágio, evento de execução com
  estágio, `*_finished` sem status, início declarando resultado, e a igualdade
  de vocabulário com o `LoopRecord` (`tests/test_loop_events.py`);
- sinks: uma linha JSON válida por evento, `snapshot` que não aliasa a lista
  viva, `fanout` que sobrevive a um sink que explode, leitura que pula linha
  truncada e ordena por `(round, sequence)` e não por timestamp;
- emissão: a timeline completa de uma execução, a igualdade campo a campo entre
  evento e `LoopStage`, o `events.jsonl` gravado sem ninguém pedir, o estágio
  que falha nomeando a causa, a rodada da campanha em cada evento, e um sink
  quebrado que não derruba a execução (`tests/test_intent_loop_orchestrator.py`);
- ponte da UI: diretório derivado do artefato e não de `settings`, os três
  artefatos de resultado, execução que morreu cedo, artefato corrompido lido
  como ausente, `progress()` por estágio fechado, estágio corrente e causa da
  falha (`tests/test_studio_loop_bridge.py`);
- a página em si, via `AppTest`: abre sem histórico, e mostra status, duração e
  nome do artefato de uma execução salva, com os estágios que não rodaram como
  pendentes;
- redação: o valor real sem formato reconhecível, os padrões conhecidos, o valor
  curto que não é segredo, e o caminho de ponta a ponta — uma chave na exceção
  do agente não chega ao `LoopStage` nem à timeline (`tests/test_redaction.py`);
- consumo e orçamento: soma que preserva custo desconhecido, custo sem token
  recusado, leitura tolerante da resposta do provedor, agregação no
  `LoopRecord`, o teto que corta a próxima rodada e o que nunca corta
  (`tests/test_run_usage.py`);
- duração e consumo no resumo da CLI (`tests/test_cli_intent_engine.py`).
