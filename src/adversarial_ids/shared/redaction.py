"""redaction — nada que vá para disco ou para a tela carrega uma chave (E11).

A linha "Operação" da tabela de aceite cobra "prompts não expõem chaves". O
caminho pelo qual uma chave escaparia neste projeto não é o prompt que
escrevemos: é o **erro** que voltamos a gravar. ``_stage`` registra
``str(exc)`` no ``LoopStage`` e no ``LoopEvent``, e a exceção de um cliente HTTP
costuma trazer a requisição que falhou — cabeçalho de autorização incluído.
Aquele texto vai para `loop_records.json`, para `events.jsonl` e para a tela,
onde qualquer um que receba o diretório da execução o lê.

## Duas defesas, e a primeira é a que vale

1. **O valor real.** Toda variável de ambiente cujo nome parece de segredo tem
   seu valor procurado literalmente no texto. É a defesa forte: não depende de a
   chave ter um formato reconhecível, só de ela estar no ambiente — que é
   exatamente o caso quando ela vazou de lá.

2. **O formato.** Padrões conhecidos (``gsk_…`` da Groq, ``sk-…``, ``Bearer …``)
   pegam o caso em que a chave que vazou não é a que está no ambiente deste
   processo — um segredo colado num prompt, a chave de outro serviço.

Nenhuma das duas é infalível sozinha e as duas juntas não provam ausência de
vazamento. O que elas dão é o que dá para dar honestamente: o segredo que este
processo conhece não sai daqui em texto claro.

## Por que não validar isso no contrato

Seria tentador fazer ``LoopStage``/``LoopEvent`` redigirem sozinhos. Não: o
contrato passaria a ler ``os.environ``, e um modelo congelado que depende do
ambiente deixa de ser reproduzível — a mesma entrada produziria arquivos
diferentes em máquinas diferentes, e o fixture golden do repo depende do
contrário. A redação mora no ponto de emissão, que é onde o texto nasce.
"""

from __future__ import annotations

import os
import re

PLACEHOLDER = "***"

# Nome de env var que denuncia segredo. Casado por substring, em maiúsculas: é
# larga de propósito, porque o custo de redigir demais (uma URL borrada num log)
# é muito menor que o de redigir de menos.
_SECRET_NAME_MARKERS: tuple[str, ...] = (
    "API_KEY",
    "APIKEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
)

# Valor curto demais não é chave, e procurá-lo literalmente borraria texto
# legítimo — um `DEBUG=1` no ambiente apagaria todo dígito 1 das mensagens.
_MIN_SECRET_LENGTH = 12

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Groq: gsk_ seguido do corpo da chave.
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    # OpenAI e derivados: sk-, sk-proj-, etc.
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    # Qualquer Authorization: Bearer <token>.
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{12,}"),
)


def secret_values(environ: dict[str, str] | None = None) -> tuple[str, ...]:
    """Os valores de ambiente que este processo deve considerar segredo.

    Recebe o ambiente como parâmetro para ser testável sem mexer no processo —
    o default é o de verdade.
    """

    source = os.environ if environ is None else environ

    return tuple(
        value
        for name, value in source.items()
        if value
        and len(value) >= _MIN_SECRET_LENGTH
        and any(marker in name.upper() for marker in _SECRET_NAME_MARKERS)
    )


def redact(text: str | None, *, environ: dict[str, str] | None = None) -> str | None:
    """Devolve ``text`` sem segredo reconhecível; ``None`` continua ``None``.

    Preserva a mensagem em volta: o objetivo é um erro que continue
    diagnosticável — "401 Unauthorized para ***" ensina o que "***" sozinho não
    ensinaria.
    """

    if not text:
        return text

    cleaned = text
    # Os valores reais primeiro: se a chave aparecer, ela some antes de qualquer
    # padrão ter a chance de recortá-la pela metade e deixar um pedaço para trás.
    for value in secret_values(environ):
        cleaned = cleaned.replace(value, PLACEHOLDER)

    for pattern in _PATTERNS:
        cleaned = pattern.sub(PLACEHOLDER, cleaned)

    return cleaned
