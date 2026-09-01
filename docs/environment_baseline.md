# Baseline de reprodutibilidade

Este registro atende à primeira entrega do plano de 60 dias: antes de ampliar a
arquitetura, a toolchain deve executar a suíte inteira de forma verificável.

## Baseline inicial - 1 de setembro de 2026

| Item | Resultado |
|---|---|
| Python | 3.12.13 |
| uv | 0.11.2 |
| pytest | 9.1.1 |
| Suíte | 103 passaram, 1 pulado |
| Duração | 20,43 s |

O teste pulado cobre a integração opcional com SHAP, que não faz parte dos
extras `dev` e `dashboard`; ele não representa falha da suíte padrão. O cache
do `uv` foi fixado em `.uv-cache/` pelo `uv.toml`, removendo a dependência do
cache global do usuário.

## Como verificar novamente

```powershell
uv sync --extra dev --extra dashboard
uv run python scripts/verify_environment.py
```

O segundo comando executa toda a suíte e grava o relatório gerado em
`outputs/environment_baseline.json`. O relatório inclui versões, contagens dos
testes, duração, código de saída e o SHA-256 do `uv.lock`; como pertence a
`outputs/`, ele não é versionado.

Um baseline só é considerado válido quando `pytest_exit_code` é zero.
