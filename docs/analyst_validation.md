# Validação da saída do agente Analista

## Objetivo

O agente Analista atua como componente Blue Team do loop adversarial.
Ele recebe as métricas produzidas pelo IDS Random Forest, interpreta
as evidências disponíveis e recomenda mitigações defensivas.

A resposta da LLM não é aceita diretamente. Toda saída passa por
validação tipada e por regras determinísticas antes de ser entregue
ao Orquestrador ou persistida no histórico.

## Contrato de saída

A resposta deve respeitar o modelo `AnalystOutput`, composto por:

- número da iteração;
- lista de features relevantes;
- diagnóstico defensivo;
- lista de mitigações;
- nível de severidade.

Cada feature contém:

- nome da feature;
- valor de importância;
- explicação em linguagem natural.

As mitigações permitidas são:

- `threshold`;
- `feature`;
- `retrain`.

A severidade deve ser:

- `high`;
- `medium`;
- `low`.

## Validação por schema

O modelo Pydantic `AnalystOutput` rejeita:

- campos extras;
- iteração negativa;
- importância negativa;
- severidade fora dos valores permitidos;
- tipo de mitigação inválido;
- diagnóstico ou recomendação vazios;
- resposta incompatível com o contrato.

## Validação contra as métricas

A função `validate_output_against_metrics` verifica se a resposta
está sustentada pelos dados reais da iteração.

São rejeitadas:

1. features que não aparecem nas evidências;
2. features duplicadas;
3. valores de importância alterados;
4. severidade incompatível com o F1;
5. iteração diferente da solicitada;
6. ausência de explicação quando existem features disponíveis.

## Uso de SHAP e feature importances

Quando valores SHAP estão disponíveis, eles são usados como evidência
principal.

Quando SHAP não está disponível, o sistema utiliza
`top_feature_importances`, produzido pelo Random Forest, como fallback.

O agente apenas interpreta essas evidências. Ele não calcula nem
inventa valores de importância.

## Regra de severidade

A severidade é calculada deterministicamente a partir do F1 da classe
`masquerade`:

- F1 menor que 0.50: `high`;
- F1 maior ou igual a 0.50 e menor que 0.80: `medium`;
- F1 maior ou igual a 0.80: `low`;
- F1 ausente: `low`.

A LLM recebe a severidade esperada no campo `required_severity`,
mas a resposta ainda é validada novamente pelo código.

## Segurança e escopo

O agente:

- analisa somente dados sintéticos;
- não acessa sistemas reais;
- não produz código ofensivo;
- não altera configurações de ataque;
- não inventa métricas;
- não afirma causalidade sem evidência;
- produz somente recomendações defensivas.

## Testes

Os testes automatizados cobrem:

- contrato `AnalystOutput`;
- tipos de mitigação;
- limites de severidade;
- prioridade de SHAP;
- fallback para feature importances;
- detecção de feature inventada;
- detecção de importância alterada;
- detecção de severidade incorreta;
- detecção de iteração incorreta;
- conversão de dict, JSON e modelo Pydantic;
- execução sem chamadas externas usando agente falso.

O teste manual com Groq está disponível em:

```text
scripts/test_analyst_real.py

Ele não faz parte da suíte automática porque realiza chamada externa
e consome tokens da API.
