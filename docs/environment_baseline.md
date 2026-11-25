# Reproducibility baseline

This record fulfils the first deliverable of the 60-day plan: before widening
the architecture, the toolchain must run the whole suite in a verifiable way.

## Initial baseline - September 1, 2026

| Item | Result |
|---|---|
| Python | 3.12.13 |
| uv | 0.11.2 |
| pytest | 9.1.1 |
| Suite | 103 passed, 1 skipped |
| Duration | 20.43 s |

The skipped test covers the optional SHAP integration, which is not part of the
`dev` and `dashboard` extras; it does not represent a failure of the default
suite. The `uv` cache was pinned to `.uv-cache/` by `uv.toml`, removing the
dependency on the user's global cache.

## How to verify again

```powershell
uv sync --extra dev --extra dashboard
uv run python scripts/verify_environment.py
```

The second command runs the whole suite and writes the generated report to
`outputs/environment_baseline.json`. The report includes versions, test counts,
duration, exit code and the SHA-256 of `uv.lock`; since it belongs to
`outputs/`, it is not versioned.

A baseline is only considered valid when `pytest_exit_code` is zero.
