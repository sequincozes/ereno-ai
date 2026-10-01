# Geradores — de onde o trace vem

O pipeline treina um IDS sobre um trace. Até aqui, "qual gerador produziu esse
trace" não era uma pergunta que o sistema soubesse responder a partir dos
artefatos: o ERENO era a única fonte possível, o estágio se chama `ereno` no
vocabulário do `LoopRecord`, e o modo cacheado — que devolve sempre o mesmo
arquivo — se parecia com uma execução normal. O registro de geradores
(`core/generators.py`) faz o mesmo movimento que o épico E8 fez com o detector,
só que a montante.

## Os dois backends registrados

| chave | simulador | precisa de Java | varia com a config |
|---|---|---|---|
| `cached` (default) | — | não | **não** |
| `jar` | ERENO | sim | sim |

`cached` serve `data/baseline_dataset.csv`, um trace benigno+ataque versionado.
É o que permite rodar o loop inteiro, os testes e o CI sem o JAR.

`jar` executa o gerador ERENO (Java) sobre o attack config compilado.

```bash
uv run adversarial-ids --engine intent --generator-mode jar --prompt "..."
```

As chaves são `cached`/`jar` por compatibilidade com a CLI e as env vars que já
existiam. O campo que diz o que de fato produziu o trace é `simulator`, no
manifest — hoje há um backend por simulador, mas não precisa continuar assim
(ERENO via JAR e ERENO embarcado seriam dois backends, um simulador).

## `produces_per_config_variation`, o campo que existe para não deixar mentir

É a diferença entre **medir o efeito de uma intenção** e **exercitar o fluxo**.

Em `cached` o trace é o mesmo arquivo em toda execução, então o
`DetectionReport` não se move entre rodadas: uma campanha estaciona na rodada 2
com `no_improvement`, e qualquer leitura do tipo "esta intenção derrubou o
recall" feita sobre um run cacheado é uma afirmação sobre nada. Isso já estava
documentado em prosa; agora está num campo que uma máquina lê, gravado em
`generator_manifest.json` ao lado do resultado.

A capacidade vem do **registro**, nunca de quem grava o manifest, e o contrato
de domínio recusa um manifest que discorde do backend que ele nomeia. Um
manifest afirmando que `cached` varia com a config descreve um gerador que não
existe — e é exatamente a afirmação que faria um platô de campanha passar por
resultado experimental.

## O quarto manifest

`generator_manifest.json` fecha a cadeia de proveniência de uma rodada, ao lado
dos três que já existiam:

| manifest | responde |
|---|---|
| `generator_manifest.json` | de onde o trace veio |
| `feature_manifest.json` (E6) | como as features foram preparadas |
| `selection_manifest.json` (E7) | quais sobreviveram, sobre quantas linhas |
| `detector_manifest.json` (E8) | qual modelo aprendeu, com quais hiperparâmetros |

Ele não repete o `DatasetBundle`: o bundle responde *o que está no trace*
(colunas, classes, contagem, hash do conteúdo), o manifest responde *como ele
foi produzido*. Mesma divisão que `DetectionReport` (quão bem o detector foi) e
`DetectorManifest` (que detector era). Com os dois lado a lado, uma diferença
entre dois resultados é atribuível à config, ao simulador ou à semente; sem
eles, não é atribuível a nada.

`random_seed` registra a semente repassada ao backend (Fase 2.0). `None`
significa RNG do relógio — duas execuções da mesma config não são comparáveis
byte a byte, o que é a diferença entre uma réplica e uma repetição.

## Adicionar um backend

O diagrama de arquitetura do journal põe ERENO, Mininet e libiec61850 lado a
lado. Para entrar, um backend precisa de:

1. uma classe que cumpra `GeneratorLike` — `generate_dataset(attack_config,
   iteration) -> caminho do CSV`, uma linha só: tudo que separa um backend de
   outro (subprocesso, JVM, topologia de rede) é assunto interno dele;
2. uma chave a mais em `GeneratorKey` (`domain/generator_manifest.py`) e nas
   três tabelas de capacidade ao lado dela;
3. um `GeneratorSpec` a mais no registro, declarando capacidades, as opções que
   ele exige e como se constrói.

Nada mais muda: nem `GeneratorRunner`, nem o orquestrador, nem o avaliador. A
checagem de sincronia roda na importação — uma chave registrada e ausente do
`Literal` produziria um manifest impossível de validar, e uma chave só no
`Literal` seria uma opção de CLI que não constrói nada; as duas falham antes de
qualquer campanha começar.

## Duas honestidades sobre o estado atual

**Os dois backends registrados são a mesma classe.** `cached` e `jar` são ambos
`GeneratorRunner`, e a escolha entre eles é literalmente passar ou não
`cached_dataset_path`. Isto é descrição, não arquitetura aspiracional: a costura
que o registro abre vale para o **terceiro** backend, que se implementa sozinho.
Mesma ressalva que o `DetectorLike` do E8 faz sobre si mesmo.

**O estágio ainda se chama `ereno`.** `LoopStageName` é um `Literal` do domínio,
gravado nos `LoopRecord` que já existem em disco, lido pelo dashboard e fixado
em vários testes; renomeá-lo é mudança de schema, não de nome de variável.
Enquanto não for feita, quem diz o simulador de verdade é o manifest, não o nome
do estágio.
